"""Stage bookkeeping for staged, resumable ``pgs-recon`` runs.

A reconstruction is thirteen stages, one per binary invocation. ``--from``/``--to``
select a contiguous window of them so a single output directory can be filled in
by several cluster jobs, each sized for the stages it runs. State lives in the
run's ``metadata.json`` under a ``stages`` key.

The motivation is ``RefineMesh``: it is a memory hog with no reliable a-priori
bound, so a whole-pipeline job has to be sized for its worst case -- which either
wastes a large-memory allocation on hours of cheap SfM, or gets OOM-killed after
that work is already done. Splitting lets refine get a big-memory node and
nothing else pay for it.

What runs is decided by an **artifact graph**, not by position in ``STAGES``.
Artifacts are nodes, stages are edges: ``STAGE_IO`` declares which semantic
**roles** each stage consumes and produces, and one forward pass over the
pipeline shape maintains a role -> **binding** (which stage currently owns the
role, and where its artifact is). A stage is **dirty** when its record is
missing or not complete, when its own arguments changed, when a stage that
produces something it consumes is dirty, or when a role it consumes is now bound
to a different path than the one it recorded. The range is policy on top of that
set: dirty inside it runs, dirty before ``--from`` is an error, dirty after
``--to`` is a warning.

Three invariants hold this together, and all three are load-bearing:

* **Staged runs must produce the artifact names a single-shot run would.** Output
  names come from ``layout`` and depend only on the paths a stage consumes, so a
  rehydrated input produces exactly the output a freshly built one would.
  Anything that breaks that makes a resumed run's outputs diverge from a fresh
  one's.
* **Nothing runs that the range did not ask for.** A job sized for texturing must
  never quietly start refining, which is the exact blowup this exists to prevent.
  Missing prerequisites are an error, never backfilled.
* **The manifest is the record; nothing here touches the filesystem.** Dirtiness
  is computed from two dicts (the parsed args and the stage records), which is
  what makes it testable and what keeps a half-created output directory from
  changing the plan. Deleting an intermediate by hand therefore does not trigger
  a rebuild -- that is ``--rerun``.

This depends on every MVS intermediate being portable (``MVSI`` scene + ``.ply``
geometry, ``--archive-type -1`` on all four builders, dense cloud handed to
``ReconstructMesh`` via ``-p``). Boost-binary ``.mvs`` files OOM when read by a
different Boost version, so without that, passing artifacts between
differently-built containers would fail on load.
"""
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime as dt, timezone as tz
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from pgs_recon import layout

try:  # not available on every platform
    import resource
except ImportError:  # pragma: no cover
    resource = None

# Stage order matches main()'s existing call order, one entry per binary.
STAGES: Tuple[str, ...] = (
    'import', 'features', 'matches', 'filter', 'sfm', 'robust', 'autoscale',
    'colorize', 'convert', 'densify', 'reconstruct', 'refine', 'texture',
)


class StageError(Exception):
    """A planning failure with a user-facing message.

    Raised instead of ``sys.exit`` so the planner can be exercised by tests;
    ``main()`` catches it and exits. ``utility.ToolFailed`` is the same move for
    execution failures, and carries the child's exit status with it.
    """


class IO(NamedTuple):
    """A stage's local edges in the artifact graph."""
    needs: Tuple[str, ...]  # consumed roles that must be bound
    may: Tuple[str, ...]    # consumed roles that may legitimately be unbound
    makes: Tuple[str, ...]  # roles this stage produces or rebinds


# Which artifact roles each stage consumes and produces. Roles are *rebound* as
# the pipeline runs (``sfm`` is produced by import, then rebound by sfm, robust
# and autoscale), so the topology is not a static DAG over role names: it is
# derived per run by walking the shape in order and tracking what currently owns
# each role. This table declares only local consumes/produces.
#
# A role a stage passes through untouched must NOT be listed in ``makes``.
# ``mvs_reconstruct`` and ``mvs_refine`` both return their input scene unchanged,
# so ``scene`` stays bound to whatever built it (convert or densify). Declaring a
# pass-through as a production creates a phantom edge -- harmless today because
# both stages already depend on the real producer via another role, but exactly
# the sort of thing that generates spurious cascades once the graph is
# load-bearing.
#
# ``sfm`` consumes whichever matches its engine wants: ``matches_filtered`` for
# every engine but ``direct``, which triangulates known poses against the
# unfiltered ``matches``. Both are declared, the latter optional.
STAGE_IO: Dict[str, IO] = {
    'import':      IO((),                                       (),              ('sfm', 'view_pairs')),
    'features':    IO(('sfm',),                                 (),              ('features',)),
    'matches':     IO(('sfm', 'features'),                      ('view_pairs',), ('matches',)),
    'filter':      IO(('sfm', 'matches'),                       (),              ('matches_filtered',)),
    'sfm':         IO(('sfm', 'features', 'matches_filtered'),  ('matches',),    ('sfm',)),
    'robust':      IO(('sfm', 'features', 'matches'),           (),              ('sfm',)),
    'autoscale':   IO(('sfm',),                                 (),              ('sfm',)),
    'colorize':    IO(('sfm',),                                 (),              ('colorized',)),
    'convert':     IO(('sfm',),                                 (),              ('scene',)),
    'densify':     IO(('scene',),                               (),              ('scene', 'cloud')),
    'reconstruct': IO(('scene',),                               ('cloud',),      ('mesh',)),
    'refine':      IO(('scene', 'mesh'),                        (),              ('mesh',)),
    'texture':     IO(('scene', 'mesh'),                        (),              ('mesh',)),
}

# Which stage owns which argument. Keep adjacent to the parser definitions in
# reconstruct.py so the two are edited together; validate_arg_map() enforces
# that every parser dest lands in exactly one bucket.
STAGE_ARGS: Dict[str, Tuple[str, ...]] = {
    'import': ('input', 'focal_length', 'new_importer', 'import_pgs_scan',
               'import_calib', 'matching_pairs_radius'),
    'features': ('describer_method', 'describer_preset', 'describer_upright'),
    'matches': ('matching_method', 'matching_ratio', 'matching_pairs_file'),
    'filter': ('matching_geometric_model',),
    'sfm': ('mvg_recon_method', 'mvg_priors', 'mvg_refine_intrinsics',
            'mvg_initializer', 'sfm_ba'),
    'robust': ('mvg_robust', 'robust_ba'),
    'autoscale': ('mvg_autoscale', 'autoscale_method', 'autoscale_marker_pix',
                  'autoscale_include_from', 'autoscale_exclude_from'),
    'colorize': (),
    'convert': (),
    'densify': ('mvs_densify', 'densify_resolution_level', 'mask_value'),
    'reconstruct': ('free_space_support', 'mvs_smooth'),
    'refine': ('mvs_refine', 'decimation_factor', 'refine_resolution_level',
               'refine_min_resolution', 'refine_scales', 'refine_scale_step'),
    'texture': ('name', 'file_type', 'texture_resolution_level',
                'texture_max_size'),
}

# Per-node or whole-run facts. Overriding these never warns: a 128-core SfM node
# and a big-memory refine node legitimately differ in --threads and --path.
GLOBAL_ARGS: Tuple[str, ...] = (
    'config', 'output', 'log_level', 'path', 'cam_db', 'threads',
)

# Flow control for this invocation. Never persisted, never inherited on resume.
# ``mvs`` is here because --no-mvs is a deprecated alias for --to colorize: it
# truncates the range rather than configuring anything.
CONTROL_ARGS: Tuple[str, ...] = ('from_stage', 'to_stage', 'rerun', 'dry_run',
                                 'mvs')

# Args excluded from the manifest's effective-arg record, so a resume never
# inherits them. Beyond flow control, these describe *this invocation* rather
# than the reconstruction: ``output`` is required on every run, so recording it
# only parks a stale absolute path from another runtime's mounts in the
# manifest; an inherited ``threads`` silently pins a later job to a previous
# node's core count; an inherited ``log_level`` silently keeps a debugging run's
# DEBUG.
#
# ``path`` is here for the same reason, and it is the one that bites: inherited,
# a prefix recorded by job 1 is handed to ``toolchain.configure()`` by every
# later job, which makes ``$PGS_RECON_PREFIX`` unreachable on exactly the nodes
# it exists for -- a staged run's big-memory refine node, whose install prefix
# legitimately differs from the node that ran SfM. That is the same shadowing
# that cost ``--path`` its parser default (ADR 0005); persisting it here would
# have reintroduced it one level up. Nothing is lost by dropping it: which
# prefix a run used is still in ``parsed``, in ``runs[].argv``, and in the
# absolute argv[0] of every recorded command.
#
# ``cam_db`` stays persisted. Unset it records ``None`` and re-derives from
# whatever prefix is in force, and set it names a file the user chose --
# provenance of the reconstruction rather than a property of the node.
NO_PERSIST = frozenset(CONTROL_ARGS) | {'config', 'output', 'threads',
                                        'log_level', 'path'}

_MISSING = object()


def utc_now() -> str:
    return dt.now(tz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def stage_index(name: str) -> int:
    return STAGES.index(name)


def arg_owner(dest: str) -> Optional[str]:
    """Return the stage owning ``dest``, ``'GLOBAL'``, or None if unclassified."""
    if dest in GLOBAL_ARGS or dest in CONTROL_ARGS:
        return 'GLOBAL'
    for stage, dests in STAGE_ARGS.items():
        if dest in dests:
            return stage
    return None


def validate_arg_map(parser) -> None:
    """Fail loudly at startup if the ownership map and the parser disagree.

    Without this, wiring a new ``--refine-foo`` into the ``mvs_refine`` call but
    forgetting the map leaves the flag unowned, so overriding it is silently
    warn-and-ignored even when refine is the stage being run -- a no-op that
    costs a cluster allocation to discover.
    """
    buckets: Dict[str, List[str]] = {'GLOBAL': list(GLOBAL_ARGS) + list(CONTROL_ARGS)}
    buckets.update({s: list(d) for s, d in STAGE_ARGS.items()})
    seen: Dict[str, List[str]] = {}
    for bucket, dests in buckets.items():
        for dest in dests:
            seen.setdefault(dest, []).append(bucket)

    declared = {a.dest for a in parser._actions if a.dest not in ('help',)}
    problems = []
    missing = sorted(declared - set(seen))
    if missing:
        problems.append(f'arguments not classified in STAGE_ARGS/GLOBAL_ARGS: '
                        f'{missing}')
    dupes = sorted(d for d, b in seen.items() if len(b) > 1)
    if dupes:
        problems.append(f'arguments classified in more than one bucket: '
                        f'{[(d, seen[d]) for d in dupes]}')
    stale = sorted(set(seen) - declared)
    if stale:
        problems.append(f'classified arguments the parser no longer defines: '
                        f'{stale}')
    unknown = sorted(set(STAGE_ARGS) ^ set(STAGE_IO))
    if unknown:
        problems.append(f'STAGE_ARGS and STAGE_IO disagree about the stage list: '
                        f'{unknown}')
    if problems:
        raise StageError('BUG: ' + '; '.join(problems))


def pipeline_shape(args) -> Tuple[str, ...]:
    """The stages this reconstruction contains at all, in order.

    Enable flags declare the shape; ``--from``/``--to`` select a window of it.
    """
    shape = ['import', 'features', 'matches', 'filter', 'sfm']
    if args.mvg_robust:
        shape.append('robust')
    if args.mvg_autoscale is not None:
        shape.append('autoscale')
    shape.append('colorize')
    shape.append('convert')
    if args.mvs_densify:
        shape.append('densify')
    shape.append('reconstruct')
    if args.mvs_refine:
        shape.append('refine')
    shape.append('texture')
    return tuple(sorted(shape, key=stage_index))


def describe_shape(shape: Sequence[str]) -> str:
    absent = [s for s in STAGES if s not in shape]
    text = f'{shape[0]}..{shape[-1]}'
    if absent:
        text += ', no ' + ', '.join(absent)
    return text


# --------------------------------------------------------------------------
# manifest i/o
# --------------------------------------------------------------------------

def load_manifest(path: Path) -> Dict:
    """Load an existing ``metadata.json``, or return an empty manifest."""
    if not Path(path).is_file():
        return {}
    try:
        meta = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise StageError(f'Cannot read existing manifest {path}: {e}. Move it '
                         f'aside to start over in this directory.')
    if not isinstance(meta, dict):
        raise StageError(f'Existing manifest {path} is not a JSON object.')
    return meta


def write_manifest(path: Path, meta: Dict, strict: bool = False) -> None:
    """Write the manifest atomically.

    An OOM-killed refine is the expected failure mode here, so a SIGKILL
    mid-write must not be able to leave a truncated manifest behind.

    ``strict`` makes a write failure fatal, for the stage transitions: the
    manifest is the record, so work that cannot be recorded should not be done.
    The tolerant default is for best-effort flushes (``atexit``, ``abort()``),
    where raising would fire during shutdown or mask a live exception.
    """
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    try:
        with tmp.open('w') as f:
            f.write(json.dumps(meta, indent=4, sort_keys=False, default=str))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        if strict:
            raise StageError(f'Cannot write the manifest {path}: {e}. It is the '
                             f'only record of what has finished, so the run '
                             f'stops here rather than continue unrecorded.')
        logging.getLogger(__name__).error(f'Failed to write manifest: {e}')


def rel_to(p, root: Path) -> str:
    """Path relative to the output dir, for storage in the manifest.

    Absolute paths may belong to another runtime (a different Docker or
    Apptainer mount), so recorded artifacts are stored relative and re-rooted on
    read.

    Anything outside the output dir (the image directory, a ``--cam-db``) has no
    relative form and is recorded absolute. Returning it as given would let a
    relative ``-i images/`` be recorded as ``images``, which ``abs_from()``
    would then re-root to ``<output>/images``.
    """
    p = Path(p).resolve()
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def abs_from(rel: str, root: Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


# --------------------------------------------------------------------------
# argument merging
# --------------------------------------------------------------------------

def explicit_dests(parser_factory, argv: Sequence[str]) -> set:
    """The dests the user actually supplied, on the CLI or via ``--config``.

    Re-parses ``argv`` against a copy of the parser whose defaults are a
    sentinel, so an argument set to its own default value still counts as
    explicit. That matters for drift: the manifest, not the parser default, is
    the baseline on resume.
    """
    sentinel = object()
    p = parser_factory()
    for action in p._actions:
        # The config-file action must keep a real default; configargparse reads
        # it before parsing.
        if action.dest in ('help', 'config'):
            continue
        action.default = sentinel
        action.required = False
    try:
        ns, _ = p.parse_known_args(list(argv))
    except SystemExit:
        raise
    return {d for d, v in vars(ns).items()
            if v is not sentinel and d not in ('help', 'config')}


def apply_stored(args, stored: Dict, explicit: set) -> None:
    """Fold the manifest's effective args into ``args`` as defaults, in place.

    Anything given on the CLI or config file wins here; the range is not yet
    known. ``revert_out_of_range()`` walks back the overrides the range does not
    authorize.
    """
    for dest, value in stored.items():
        if dest in NO_PERSIST or dest in explicit or not hasattr(args, dest):
            continue
        setattr(args, dest, value)


def revert_out_of_range(args, stored: Dict, explicit: set, from_stage: str,
                        to_stage: str, logger) -> None:
    """Undo overrides aimed at stages outside the range, warning about each.

    The out-of-range value must still govern because it determines downstream
    filenames: flipping ``--mvs-densify`` on a ``--from refine`` job would
    otherwise make refine look for a ``scene_dense_mesh.ply`` that was never
    built. Reverting forgives the user error rather than rejecting the run, and
    it happens *before* the graph is built, so the graph never sees the
    override -- the principle being not to mutate parts of the graph this run is
    not prepared to recompute. The manifest keeps recording the effective value,
    not the ignored override, so the log stays truthful.

    Note the asymmetry on a **first** run: ``stored`` is empty, so the
    ``dest not in stored`` guard lets out-of-range overrides through silently and
    persists them. That is load-bearing, not an oversight -- it is what lets
    ``submit_recon_pipeline.sh`` job 1 (``--to convert``) carry ``--refine-*``
    arguments forward to job 3. Tightening it would break the submit script.
    """
    lo, hi = stage_index(from_stage), stage_index(to_stage)
    for dest in sorted(explicit):
        if dest in NO_PERSIST or dest not in stored or not hasattr(args, dest):
            continue
        owner = arg_owner(dest)
        if owner == 'GLOBAL' or owner is None:
            continue
        if lo <= stage_index(owner) <= hi:
            continue
        given, value = getattr(args, dest), stored[dest]
        if given != value:
            flag = '--' + dest.replace('_', '-')
            logger.warning(
                f'{flag}={given!r} configures stage {owner!r}, which is outside '
                f'the range {from_stage}..{to_stage}; ignoring it and using the '
                f'recorded value {value!r}.')
        setattr(args, dest, value)


def drifted_stages(args, stages: Dict, explicit: set,
                   shape: Sequence[str]) -> Dict[str, List[str]]:
    """Stages whose own args differ from what they were last run with.

    Explicit args only, and only ones the record itself carries: a fingerprint
    over every argument would make any future version that adds a defaulted
    flag dirty every stage of every existing output directory.
    """
    drift: Dict[str, List[str]] = {}
    for stage in shape:
        recorded = (stages.get(stage) or {}).get('args')
        if not recorded:
            continue
        for dest in STAGE_ARGS.get(stage, ()):
            if dest not in explicit or dest not in recorded:
                continue
            if recorded[dest] != getattr(args, dest, _MISSING):
                drift.setdefault(stage, []).append(dest)
    return drift


# --------------------------------------------------------------------------
# range
# --------------------------------------------------------------------------

def resolve_range(args, shape: Sequence[str]) -> Tuple[str, str]:
    """Validate and default the ``--from``/``--to`` bounds (both inclusive)."""
    from_stage = args.from_stage or shape[0]
    to_stage = args.to_stage or shape[-1]
    for flag, stage in (('--from', from_stage), ('--to', to_stage)):
        if stage not in shape:
            raise StageError(f'{flag} {stage} is not part of this pipeline '
                             f'({describe_shape(shape)}). Either enable the '
                             f'stage or name one that runs.')
    if stage_index(from_stage) > stage_index(to_stage):
        raise StageError(f'--from {from_stage} comes after --to {to_stage}.')
    return from_stage, to_stage


# --------------------------------------------------------------------------
# execution tracking
# --------------------------------------------------------------------------

class Binding(NamedTuple):
    """What currently owns an artifact role.

    ``path`` is None while the producing stage is dirty: the artifact will exist
    once that stage runs, but its location is not knowable until it does.
    """
    stage: str
    path: Optional[Path]


class StageTracker:
    """Decides what runs, rehydrates its inputs, and records the outcome.

    Planning is one forward pass over the shape (``_walk``), maintaining a role
    -> :class:`Binding` map and marking each stage dirty or clean; ``_resolve_plan``
    then applies the range to that set. Nothing in either step touches the
    filesystem -- the manifest is the record.

    ``chain`` is the live binding map during execution, seeded with the state
    just before ``from_stage`` and updated as stages run or are skipped.
    ``require(role)``/``path(role)`` hand a stage its inputs as recorded paths,
    which is the whole mechanism by which a resumed job's outputs land where a
    single-shot run's would: it consumes the same paths, and ``layout`` derives
    the same names from them.
    """

    def __init__(self, metadata: Dict, manifest_path: Path, root: Path,
                 args, shape: Sequence[str], from_stage: str, to_stage: str,
                 rerun: bool = False, drift: Dict[str, List[str]] = None,
                 logger: logging.Logger = None):
        self.meta = metadata
        self.manifest_path = Path(manifest_path)
        self.args = args
        self.root = Path(root)
        self.shape = tuple(shape)
        self.from_stage = from_stage
        self.to_stage = to_stage
        self.rerun = rerun
        self.drift = drift or {}
        self.log = logger or logging.getLogger(__name__)
        self.stages = self.meta.setdefault('stages', {})
        self._active: Optional[str] = None
        self._t0 = 0.0
        self._cmds: set = set()
        self._ran: List[str] = []
        # Why each dirty stage is dirty, keyed by stage. Clean stages are absent.
        self.dirty: Dict[str, str] = {}
        # Binding state as each stage sees it, so the plan can report a stage's
        # inputs even though later stages rebind the same roles.
        self._chain_at: Dict[str, Dict[str, Binding]] = {}
        self._plan: Dict[str, str] = {}
        self._walk()
        self._resolve_plan()
        self.chain: Dict[str, Binding] = dict(self._chain_at[self.from_stage])

    # -- planning ---------------------------------------------------------
    def _in_range(self, stage: str) -> bool:
        return (stage_index(self.from_stage) <= stage_index(stage)
                <= stage_index(self.to_stage))

    def _absorb(self, stage: str, chain: Dict[str, Binding]) -> None:
        """Bind a completed stage's recorded outputs.

        Only roles the record actually carries are bound: a non-PGS import
        records no ``view_pairs``, and binding one anyway would hand ``matches``
        a pairs file that was never written.
        """
        for role, rel in ((self.stages.get(stage) or {}).get('outputs')
                          or {}).items():
            chain[role] = Binding(stage, abs_from(rel, self.root))

    def _walk(self) -> None:
        """Derive the dirty set and the per-stage bindings in one pass."""
        chain: Dict[str, Binding] = {}
        for stage in self.shape:
            self._chain_at[stage] = dict(chain)
            reason = self._dirty_reason(stage, chain)
            if reason:
                self.dirty[stage] = reason
                # The paths are unknowable until it runs, but the ownership is
                # not: that is what makes the cascade work.
                for role in STAGE_IO[stage].makes:
                    chain[role] = Binding(stage, None)
            else:
                self._absorb(stage, chain)

    def _dirty_reason(self, stage: str, chain: Dict[str, Binding]) -> str:
        """Why ``stage`` must run, or ``''`` if its recorded result still holds."""
        record = self.stages.get(stage) or {}
        if not record:
            return 'never run'
        status = record.get('status')
        if status != 'complete':
            return f'recorded as {status or "unknown"!r}, not complete'
        if self.rerun and self._in_range(stage):
            return '--rerun'
        if stage in self.drift:
            flags = ', '.join('--' + d.replace('_', '-')
                              for d in self.drift[stage])
            return f'args changed: {flags}'

        io = STAGE_IO[stage]
        unbound = [r for r in io.needs if r not in chain]
        if unbound:
            return f'inputs unavailable: {", ".join(unbound)}'
        consumed = [r for r in io.needs + io.may if r in chain]
        # A rebuilt input may land at the same deterministic path -- sfm rewrites
        # recon_dir/sfm_data.bin -- so only the producer's dirtiness reveals that
        # this stage's recorded result is stale.
        rebuilt = sorted({chain[r].stage for r in consumed
                          if chain[r].stage in self.dirty}, key=stage_index)
        if rebuilt:
            return f'inputs rebuilt by {", ".join(rebuilt)}'
        # And an input may move without any producer being dirty: drop densify
        # from the shape and reconstruct's ``scene`` goes from mvs/scene_dense.mvs
        # back to convert's mvs/scene.mvs. Roles the record does not carry are
        # not compared -- an optional role that was bound but not consumed (a
        # view_pairs file the user overrode with --matching-pairs-file none) has
        # no recorded value to differ from.
        recorded = record.get('inputs') or {}
        moved = sorted(r for r in consumed
                       if r in recorded and chain[r].path is not None
                       and rel_to(chain[r].path, self.root) != recorded[r])
        if moved:
            return f'inputs changed: {", ".join(moved)}'
        return ''

    def _resolve_plan(self) -> None:
        """Apply the range to the dirty set, once, so the logged plan executes.

        ``blocked`` is a dirty stage before ``--from``: a prerequisite this run
        is not authorized to build. ``main()`` refuses to start on any of them.
        """
        for stage in STAGES:
            if stage not in self.shape:
                status = 'off'
            elif stage_index(stage) < stage_index(self.from_stage):
                status = 'blocked' if stage in self.dirty else 'done'
            elif stage_index(stage) > stage_index(self.to_stage):
                status = 'after'
            else:
                status = 'run' if stage in self.dirty else 'skip'
            self._plan[stage] = status

    def status_of(self, stage: str) -> str:
        return self._plan[stage]

    def plan(self) -> List[Tuple[str, str]]:
        return [(s, self._plan[s]) for s in STAGES]

    def first_to_run(self) -> Optional[str]:
        for stage, status in self.plan():
            if status == 'run':
                return stage
        return None

    def prereq_errors(self) -> List[str]:
        """One message per in-shape stage before ``--from`` that is not usable.

        Prerequisites are never auto-backfilled: a job sized for texturing must
        not silently start refining. This is the same dirtiness every other
        stage is judged by, so there is no second rule to keep in sync.
        """
        return [f'{s}: {self.dirty[s]}' for s in self.shape
                if self._plan[s] == 'blocked']

    def stale_after_range(self) -> List[str]:
        """In-shape stages past ``--to`` that this run *invalidates*.

        A warning, not an error: dirtiness is computed rather than stored, so a
        later run covering these stages recomputes it and re-runs them. The
        state the range check would otherwise guard against -- stale but
        recorded ``complete``, silently passing a prerequisite check -- is
        unreachable.

        Only stages that were recorded ``complete`` count. One that has never
        run is not stale, it is merely unfinished, which is what the range
        already says -- otherwise every ``--to convert`` job on a fresh
        directory would warn about the whole MVS tail.
        """
        return [s for s in self.shape
                if self._plan[s] == 'after' and s in self.dirty
                and (self.stages.get(s) or {}).get('status') == 'complete']

    # -- chain ------------------------------------------------------------
    def require(self, role: str) -> Path:
        """``chain[role]``'s path, or a :class:`StageError` naming what is missing.

        The accessor for an input a stage cannot run without.
        ``prereq_errors()`` should already have refused the run, so reaching
        here means the plan and ``STAGE_IO`` disagree -- but a stage must never
        be handed ``None`` and left to hand it to a binary.
        """
        binding = self.chain.get(role)
        if binding is None or binding.path is None:
            raise StageError(f'no {role!r} artifact is recorded for this run; '
                             f'cannot resume at '
                             f'{self._active or self.from_stage}.')
        return binding.path

    def path(self, role: str) -> Optional[Path]:
        """``chain[role]``'s path, or None if unbound or not yet built."""
        binding = self.chain.get(role)
        return None if binding is None else binding.path

    def has(self, role: str) -> bool:
        return self.path(role) is not None

    def log_plan(self) -> None:
        """Log the resolved plan. Emitted on every run, and all --dry-run does."""
        self.log.info(f'shape:  {describe_shape(self.shape)}')
        self.log.info(f'range:  {self.from_stage}..{self.to_stage}')
        first = self.first_to_run()
        stale = self.stale_after_range()
        for stage, status in self.plan():
            if status == 'off':
                continue
            if status == 'done':
                self.log.info(f'  {stage:<12} complete  (prerequisite)')
            elif status == 'blocked':
                self.log.info(f'  {stage:<12} MISSING   '
                              f'(prerequisite: {self.dirty[stage]})')
            elif status == 'skip':
                self.log.info(f'  {stage:<12} complete  (skip)')
            elif status == 'after':
                note = ', now stale' if stage in stale else ''
                self.log.info(f'  {stage:<12} --        (past --to{note})')
            else:
                self.log.info(f'  {stage:<12} PENDING   -> run  '
                              f'[{self.dirty[stage]}]')
                # Inputs are only knowable for the first stage to run; later
                # stages consume artifacts that do not exist yet.
                if stage == first:
                    for role, binding in sorted(self._chain_at[stage].items()):
                        if binding.path is not None:
                            self.log.info(f'       {role} = '
                                          f'{rel_to(binding.path, self.root)}')
        if stale:
            self.log.warning(
                f'this run invalidates {", ".join(stale)}, which sit past --to '
                f'{self.to_stage} and so are left stale. Nothing here is lost -- '
                f'a later run covering {stale[-1]} rebuilds them -- but until '
                f'then the final textured mesh is the old one, sitting at its '
                f'usual filename.')

    # -- execution --------------------------------------------------------
    def begin(self, stage: str) -> bool:
        """True if ``stage`` should run now; records ``running`` if so."""
        status = self.status_of(stage)
        if status == 'off':
            self.log.debug(f'{stage}: not part of this pipeline, skipping')
            return False
        if status in ('done', 'blocked', 'after'):
            return False
        if status == 'skip':
            self.log.info(f'{stage}: already complete, skipping')
            self._absorb(stage, self.chain)
            return False

        record = self.stages.get(stage) or {}
        if record.get('status') == 'running':
            self.log.warning(
                f"'{stage}' is recorded as running (pid {record.get('pid')} @ "
                f"{record.get('host')}, started {record.get('started')}). If "
                f"that job is still alive, both will write to "
                f"{rel_to(layout.mvs_dir(self.root), self.root)}/ and the "
                f"results will be garbage. Continuing.")
        elif record.get('status') == 'complete':
            self.log.info(f'{stage}: re-running ({self.dirty[stage]})')

        self._active = stage
        self._t0 = time.monotonic()
        self._cmds = set(self.meta.get('commands', {}))
        self.stages[stage] = {
            'status': 'running',
            'host': socket.gethostname(),
            'pid': os.getpid(),
            'started': utc_now(),
            # Effective args, recorded so a later run can detect drift.
            'args': {d: getattr(self.args, d, None)
                     for d in STAGE_ARGS.get(stage, ())},
        }
        write_manifest(self.manifest_path, self.meta, strict=True)
        return True

    def end(self, stage: str, inputs: Dict[str, Path] = None,
            outputs: Dict[str, Path] = None) -> None:
        """Mark ``stage`` complete and bind its outputs.

        ``inputs`` is what the planner compares against next time, so record
        every role the stage actually consumed, under the same role names
        ``STAGE_IO`` uses. ``outputs`` must list only roles the stage really
        produced -- not one it passed through untouched.
        """
        record = self.stages.setdefault(stage, {})
        record['status'] = 'complete'
        record['finished'] = utc_now()
        record['elapsed_s'] = round(time.monotonic() - self._t0, 1)
        record['inputs'] = {r: rel_to(p, self.root)
                            for r, p in (inputs or {}).items() if p is not None}
        record['outputs'] = {r: rel_to(p, self.root)
                             for r, p in (outputs or {}).items()
                             if p is not None}
        new = [c for ts, c in self.meta.get('commands', {}).items()
               if ts not in self._cmds]
        if new:
            record['commands'] = new
        self._record_rusage(record)
        for role, path in (outputs or {}).items():
            if path is not None:
                self.chain[role] = Binding(stage, Path(path))
        self._ran.append(stage)
        self._active = None
        write_manifest(self.manifest_path, self.meta, strict=True)
        self.log.info(f'{stage}: complete in {record["elapsed_s"]}s')

    def _record_rusage(self, record: Dict) -> None:
        if resource is None:
            return
        try:
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        except (OSError, ValueError):  # pragma: no cover
            return
        # ru_maxrss is a monotonic high-water mark over all reaped children, so
        # it is this stage's true peak only when it is the sole stage this
        # process ran -- the common case in a staged run.
        record['max_rss'] = usage.ru_maxrss
        record['max_rss_units'] = 'bytes' if sys.platform == 'darwin' else 'kb'
        record['max_rss_exact'] = not self._ran

    def abort(self) -> None:
        """Mark an in-flight stage failed. Anything not complete is re-runnable.

        The write is best-effort: this runs while an exception propagates, and a
        lost ``failed`` record costs legibility only -- the stage stays
        ``running``, which is equally not-complete and equally re-runnable.
        """
        if self._active is None:
            return
        record = self.stages.setdefault(self._active, {})
        record['status'] = 'failed'
        record['finished'] = utc_now()
        record['elapsed_s'] = round(time.monotonic() - self._t0, 1)
        new = [c for ts, c in self.meta.get('commands', {}).items()
               if ts not in self._cmds]
        if new:
            record['commands'] = new
        self._record_rusage(record)
        self._active = None
        write_manifest(self.manifest_path, self.meta)
