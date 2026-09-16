"""Locate a pgs-recon run's artifacts from the manifest it wrote.

``pgs-retexture`` starts from a finished ``pgs-recon`` output directory and must
answer a non-obvious question: *which* of the many SfM files in there is the
solved scene the mesh was built from. The
answer is the scene fed to ``openMVG2openMVS`` -- after any robust-triangulation
and autoscale step, before colorize -- so it is no single stage's output, which
is why the convert stage records its *input*. Getting this wrong yields a
calibration or texture in the wrong frame, silently.

Two manifest formats are read: the stage records ``pgs-recon`` writes today (see
``pgs_recon/stages.py``), and, for output dirs predating them, the recorded
command log. Either way every path is rebuilt *relative to* ``recon_dir``: the
absolute paths in the manifest may belong to another runtime (a Docker mount, a
since-moved scratch disk), so they are never trusted as written.

Each artifact has its own resolver returning a `Resolved`, so callers demand only
what they use: ``pgs-retexture`` needs the mesh and its frame and calls
``.require()`` on both, while a caller wanting only a pose need never ask for the
mesh, and can tolerate an unresolvable SfM when it has been handed one -- which
is what lets a run that stopped before MVS still be worked against.

``pgs-calibrate`` was the other caller and is gone (ADR 0011). Its replacement,
``pgs-localize``, is C++ and takes explicit ``--input-scene``/``--matches-dir``
paths rather than reading this manifest: a second consumer of a format
``stages.py`` owns is what ADR 0006 exists to prevent.
"""
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple, Optional

from pgs_recon.stages import find_manifest

logger = logging.getLogger(__name__)


class Resolved(NamedTuple):
    """An artifact that was located, or the reason it was not.

    Exactly one field is set: ``path`` on success, ``reason`` (a complete,
    user-facing explanation) on failure. Callers that cannot proceed without the
    artifact call `require`; callers for which it is optional read ``path`` and
    branch, using ``reason`` to explain themselves in a log line.
    """
    path: Optional[Path]
    reason: Optional[str]

    def require(self) -> Path:
        """Return the path, or exit with the reason it could not be resolved."""
        if self.path is None:
            sys.exit(self.reason)
        return self.path


def load_manifest(recon_dir: Path):
    """Read the run's manifest. Returns ``(meta_path, meta)``.

    Required unconditionally: the manifest is the run's tracking record, and its
    absence means ``recon_dir`` is not a pgs-recon output at all -- there is
    nothing to resolve against and no frame to trust.

    Reads whichever name is there (``find_manifest``), so these tools keep
    working on directories built before 2.0 renamed it. Unlike ``pgs-recon``,
    nothing here writes it back, so nothing moves.
    """
    meta_path = find_manifest(recon_dir)
    if not meta_path.is_file():
        sys.exit(f'No {meta_path.name} in {recon_dir}; '
                 f'is this a pgs-recon output directory?')
    return meta_path, json.loads(meta_path.read_text())


def _stage_status(stages: dict, name: str) -> str:
    """A stage's recorded status, or ``'never run'`` if it has no record."""
    return (stages.get(name) or {}).get('status') or 'never run'


def _sfm_from_stage_records(stages: dict, meta_path: Path,
                            recon_dir: Path) -> Resolved:
    """The convert stage's recorded input SfM.

    Only a ``complete`` stage carries inputs/outputs -- ``pgs-recon`` replaces
    the whole record when a stage starts -- so a missing one means either "never
    ran" or "crashed", and the recorded status is what tells them apart.
    """
    rel = ((stages.get('convert') or {}).get('inputs') or {}).get('sfm')
    if rel:
        return Resolved(recon_dir / rel, None)
    status = _stage_status(stages, 'convert')
    if status == 'never run':
        return Resolved(None, f'{meta_path} records no convert stage; the run '
                              f'stopped before MVS (--to colorize?), so it has '
                              f'no solved SfM in the MVS frame. Finish the '
                              f'reconstruction through convert (pgs-recon -o '
                              f'{recon_dir}), or name a solved SfM in the frame '
                              f'you want with --sfm-data.')
    return Resolved(None, f'{meta_path} records convert as {status!r}, not '
                          f'complete, so the MVS scene it produces is not '
                          f'available. Finish the reconstruction first: '
                          f'pgs-recon -o {recon_dir}')


def _mesh_from_stage_records(stages: dict, meta_path: Path,
                             recon_dir: Path) -> Resolved:
    """The texture stage's recorded output mesh (see `_sfm_from_stage_records`
    for why an absent record is read together with the stage's status)."""
    rel = ((stages.get('texture') or {}).get('outputs') or {}).get('mesh')
    if rel:
        return Resolved(recon_dir / rel, None)
    status = _stage_status(stages, 'texture')
    if status == 'never run':
        return Resolved(None, f'{meta_path} records no texture stage; there is '
                              f'no textured mesh in {recon_dir}.')
    return Resolved(None, f'{meta_path} records texture as {status!r}, not '
                          f'complete, so its textured mesh is missing or '
                          f'half-written. Finish the reconstruction first: '
                          f'pgs-recon -o {recon_dir}')


def _sfm_from_commands(meta: dict, meta_path: Path,
                       recon_dir: Path) -> Resolved:
    """Fallback for manifests predating stage records: recover the SfM basename
    from the recorded ``openMVG2openMVS`` command, then re-root it under
    ``mvg/recon_dir``."""
    sfm_name = None
    for cmd in meta.get('commands', {}).values():
        if 'openMVG2openMVS' in cmd:
            toks = cmd.split()
            if '-i' in toks:
                sfm_name = Path(toks[toks.index('-i') + 1]).name
    if sfm_name is None:
        return Resolved(None, f'{meta_path} records no openMVG2openMVS step; '
                              f'the run had no MVS stage (--no-mvs / --to '
                              f'colorize?), so it has no solved SfM in the MVS '
                              f'frame. Name one with --sfm-data.')
    return Resolved(recon_dir / 'mvg' / 'recon_dir' / sfm_name, None)


def _mesh_from_commands(meta: dict, meta_path: Path,
                        recon_dir: Path) -> Resolved:
    """Fallback for manifests predating stage records: the textured mesh is
    ``mvs/<name>.<file_type>`` from the run's parsed args."""
    parsed = meta.get('parsed', {})
    name, file_type = parsed.get('name'), parsed.get('file_type')
    if not name or not file_type:
        return Resolved(None, f'{meta_path} is missing name/file_type; '
                              f'cannot locate the textured mesh.')
    return Resolved(recon_dir / 'mvs' / f'{name}.{file_type}', None)


def _resolve(recon_dir: Path, label: str, from_stages, from_commands) -> Resolved:
    """Run the right per-format resolver, then confirm the file is really there.

    A path the manifest records but disk does not have is a failure like any
    other -- reported as a reason rather than returned -- so a caller that gets a
    ``path`` back always gets one it can open.
    """
    meta_path, meta = load_manifest(recon_dir)
    stages = meta.get('stages') or {}
    if stages:
        source = 'stage records'
        resolved = from_stages(stages, meta_path, recon_dir)
    else:
        source = 'legacy command log'
        resolved = from_commands(meta, meta_path, recon_dir)
    if resolved.path is None:
        return resolved
    if not resolved.path.is_file():
        return Resolved(None, f'{label} recorded in {meta_path} is missing from '
                              f'disk: {resolved.path}')
    logger.info(f'Resolved {label.lower()} from {recon_dir} via {source}: '
                f'{resolved.path.name}')
    return resolved


def resolve_solved_sfm(recon_dir: Path) -> Resolved:
    """The solved SfM the reconstruction's mesh was built from (its frame)."""
    return _resolve(recon_dir, 'Solved SfM',
                    _sfm_from_stage_records, _sfm_from_commands)


def resolve_textured_mesh(recon_dir: Path) -> Resolved:
    """The textured mesh the reconstruction produced."""
    return _resolve(recon_dir, 'Textured mesh',
                    _mesh_from_stage_records, _mesh_from_commands)
