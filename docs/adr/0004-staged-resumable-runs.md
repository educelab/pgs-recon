# Split a reconstruction into resumable stages recorded in `metadata.json`

## Context

`RefineMesh` is a memory hog with no reliable a-priori bound, so a
whole-pipeline job has to be sized for its worst case. That either wastes a
large-memory allocation on hours of cheap SfM, or gets OOM-killed *after* that
work is already done. We need one reconstruction to span several cluster jobs,
each sized for the stages it runs, each picking up where the last stopped — so
refine gets a big-memory node and nothing else pays for it.

This is only safe because every MVS intermediate is portable
([ADR 0003](./0003-portable-mvs-intermediates.md)); handing Boost-binary
artifacts between differently-built containers would OOM on load.

## Decision

Thirteen stages, one per binary invocation, in `main()`'s existing order:
`import features matches filter sfm robust autoscale colorize convert densify
reconstruct refine texture`. `pgs-recon --from <stage> --to <stage>` selects an
**inclusive, contiguous** window; both default to the ends, so existing
invocations are unchanged. State lives in the run's existing
`<output>/metadata.json` under a new `stages` key. See `pgs_recon/stages.py`.

**What runs is decided by an artifact graph, not by position in the stage list.**
Artifacts are nodes, stages are edges. `STAGE_IO` declares, per stage, the
semantic **roles** it consumes (`needs`/`may`) and produces (`makes`); one
forward pass over the pipeline shape maintains a role → **binding** — which
stage currently owns the role and where its artifact is. A stage is **dirty**
when:

1. it has no record, or its record is not `complete`;
2. its own explicit arguments differ from the ones its record was run with;
3. a stage that produces something it consumes is dirty; or
4. a role it consumes is now bound to a different path than the one it recorded.

`--rerun` adds every in-range stage. The range is then policy on top of that
set: dirty **inside** it runs, dirty **before** `--from` is a hard error naming
each stage, dirty **after** `--to` is a warning naming each stage.

Four things were deliberately *not* done:

- **No second state file.** The manifest is `metadata.json`, which already
  records every command; a parallel state file would be a second thing to keep
  consistent with it.
- **The range does not replace the enable flags.** `--mvg-robust`,
  `--mvs-densify`, `--mvs-refine` etc. still declare which stages *exist* in a
  reconstruction (its **shape**); the range selects a window of that shape. A
  contiguous range simply cannot express "skip refine but run texture". A
  disabled stage inside the range is skipped silently; `--from <disabled stage>`
  is a hard error, because it means the job script and the shape disagree.
- **No lockfile.** A pre-existing `running` record warns loudly with host and
  pid and then **proceeds**. Slurm dependencies serialize a real chain, so a
  genuine overlap is user error — whereas an OOM-killed job can never release a
  lock, which would make every legitimate retry need an override flag that then
  lives permanently in the job template and voids the protection anyway.
- **No existence checks.** See the trap below: the manifest is the record.

The payoff is that **re-running the original command verbatim is a full resume**:
after an OOM in refine, the same `pgs-recon -i … -o … -n …` skips
import…reconstruct and starts at refine. No range flags needed.

## Consequences / non-obvious traps

- **Manifest writes are per-stage and atomic.** A record goes in as `running`
  *before* the binary launches and flips to `complete` after `run_command`
  returns, so a killed stage is never marked complete; every write is temp-file
  + `os.replace`. Being OOM-killed mid-write is the *expected* failure here, so
  neither is optional. Writes also merge rather than clobber — the old
  whole-file rewrite would have erased run 1's `commands` on run 2.
- **Recorded paths are relative to `--output`** and re-rooted on read: absolute
  paths may belong to another runtime's Docker/Apptainer mounts.
- **Roles are rebound, so the topology is derived per run, not declared.** `sfm`
  is produced by `import`, then rebound by `sfm`, `robust` and `autoscale`; the
  role a stage consumes therefore depends on which stages the shape contains.
  `STAGE_IO` declares only *local* consumes/produces, and the walk over the shape
  turns that into edges. Do not try to write the graph down as a static DAG over
  role names — it does not exist independently of the shape.
- **A role a stage passes through untouched must not be declared as produced.**
  `mvs_reconstruct` and `mvs_refine` both return their input scene unchanged, so
  `scene` stays bound to whatever built it (convert, or densify). Declaring the
  pass-through as a production would create a phantom edge: harmless today, since
  both already depend on the real producer through `mesh`, but exactly the sort
  of thing that generates spurious cascades once the graph is load-bearing. The
  stage record must not list it in `outputs` either, for the same reason.
- **Both the producer clause and the rebinding clause are needed; neither
  subsumes the other.** `sfm` rewrites `mvg/recon_dir/sfm_data.bin` at the same
  deterministic path every time, so only the *producer's* dirtiness reveals that
  `robust` must re-run (clause 3). Conversely, dropping `--mvs-densify` moves
  `reconstruct`'s `scene` from `mvs/scene_dense.mvs` back to convert's
  `mvs/scene.mvs` with nothing incomplete anywhere, and only the *rebinding*
  reveals it (clause 4). Stages outside the shape bind nothing, which is what
  makes clause 4 fire on a shrunken shape. A stale `complete` record for an
  out-of-shape stage stays in the manifest and is simply never consulted.
- **Rehydration is by semantic role** (`scene`, `mesh`, `cloud`), not by artifact
  name. The names are *chained* (`scene_dense_mesh_refine.ply`), so a name
  encodes which stages ran and cannot be reconstructed from args without
  duplicating the naming logic; a role's recorded path is read from the manifest
  and handed to the stage as-is. This is safe only because an output name is
  derived from the artifacts a stage consumes and from nothing else —
  **byte-identical artifact names between a single-shot and a staged run is the
  acceptance test** (automated in `test_pipeline.py`), and anything that breaks it
  makes resumed outputs diverge. Every wrapper therefore consumes its inputs
  **explicitly**, including the three OpenMVG ones that used to reach into the
  `paths` dict for the current `sfm`: a stage that records a role it did not
  actually consume through the bindings makes that record decorative, and clause
  4 cannot see a rebinding it never looked at. *(Originally written when wrappers
  took `paths` + `*_key` strings; [ADR 0005](./0005-wrappers-mirror-the-binary.md)
  replaced those with explicit `Path` arguments, which serves this reason
  strictly better.)*
- **No existence checks anywhere; the manifest is the record.** Dirtiness is
  computed from two dicts, which is what makes the planner testable and keeps a
  half-created output directory from changing the plan. The motivating failure —
  an OOM-killed `RefineMesh` — is already covered by status: a killed stage is
  recorded `failed`, never `complete`, so it is dirty on status alone.
  Consequence: deleting an intermediate by hand no longer triggers a rebuild;
  that is what `--rerun` is for. Do not "helpfully" add `path.exists()` back —
  `main()` creates `mvg/matches_dir/` before any check could run, so an
  existence check on `features`' recorded output (that directory) would make
  `features` permanently un-dirtyable.
- **Effective args are persisted, so `--input` and `--name` are not required on
  resume.** Without this a bare resume job would hit `--input required=True`, and
  an omitted `--name` mutates into `<timestamp>_<input-stem>` — inventing a new
  name and writing the texture to a different filename. `NO_PERSIST` excludes
  what describes *this invocation* rather than the reconstruction: flow control,
  plus `output` (required anyway, so recording it only parks another runtime's
  absolute path in the manifest), `threads` (would pin a later job to a previous
  node's core count), `log_level` (would inherit a debugging run's `DEBUG`) and
  `path` — an inherited install prefix would be handed to
  `toolchain.configure()` by every later job, making `$PGS_RECON_PREFIX`
  unreachable on precisely the nodes it exists for: the big-memory refine node,
  whose prefix legitimately differs from the node that ran SfM. Which prefix a
  run used is still recoverable from `parsed`, from `runs[].argv`, and from the
  absolute `argv[0]` of every recorded command. `cam_db` does stay recorded:
  unset it re-derives from whatever prefix is in force, and set it names a file
  the user chose — provenance of the reconstruction, not a property of the node.
- **Arg drift is resolved by an arg→stage ownership map**, comparing explicit
  arguments only — never a fingerprint over all of them, which would make any
  future version that adds a defaulted flag dirty every stage of every existing
  output directory. An override targeting a stage *inside* the range re-runs it
  (the primary use case: retry refine with a new `--refine-resolution-level`);
  *outside* the range it is warned and reverted to the manifest value, because
  out-of-range values determine downstream filenames — flipping `--mvs-densify`
  on a `--from refine` job would send refine looking for a `scene_dense_mesh.ply`
  that was never built. Reverting happens *before* the graph is built, so the
  graph never sees the override: the principle is not to mutate parts of the
  graph this run is not prepared to recompute. Global args (`--threads`,
  `--path`: per-node facts) apply silently. The manifest always records the
  **effective** value, never the ignored override.
- **On a *first* run, out-of-range overrides are kept, silently.** `stored` is
  empty, so `revert_out_of_range`'s `dest not in stored` guard lets them through
  and they are persisted. That asymmetry is load-bearing: it is what lets
  `submit_recon_pipeline.sh` job 1 (`--to convert`) carry `--refine-*` arguments
  forward to job 3. Tightening it would break the submit script.
- **The map is validated at startup**, not only in a test: every parser `dest`
  must land in exactly one bucket or the run exits, which also holds for the
  branches CI cannot import the parser on.
  Otherwise a new `--refine-foo` wired into the `mvs_refine` call but forgotten
  in the map is *unowned*, so overriding it gets warn-and-ignored **even when
  refine is the stage being run** — a silent no-op that costs a cluster
  allocation to discover. Keep `STAGE_ARGS` (and `STAGE_IO`) adjacent to the
  parser definitions.
- **A range that leaves later stages dirty warns; it is not an error.** With
  densify newly enabled and `--to reconstruct`, refine and texture are reported
  stale and the job proceeds. This is safe *because dirtiness is computed rather
  than stored*: the state a hard error would guard against — stale but recorded
  `complete`, silently passing a later job's prerequisite check — is unreachable,
  since the next run recomputes it from the same bindings. The warning matters
  anyway, because `mvs_texture` always writes `mvs/<name>.<ext>`: a stale final
  mesh sits at exactly the expected filename rather than under an obviously
  different name.
- **A completed in-range stage re-runs when a stage feeding it will run**, since
  its input is about to be rebuilt. Without this,
  `--from refine --refine-resolution-level 2` (with `--to` unset) would re-refine
  and then *skip* texture, leaving a fresh `scene_refine.ply` beside a stale
  `obj.obj`. The cascade follows the graph rather than stage order, so
  invalidating `colorize` — a leaf whose `colorized` ply nothing consumes —
  re-runs colorize and nothing else. It never crosses `--to`, so
  `--from refine --to refine` still runs exactly one stage.
- **Prerequisites are validated but never backfilled.** Any in-shape stage
  before `--from` that is dirty is an error naming it, and the run exits
  non-zero; this is the same dirtiness every other stage is judged by, not a
  second rule. A job sized for texturing must never quietly start refining — the
  exact blowup this exists to prevent.
- **`--mvs`/`--no-mvs` is a deprecated hidden alias for `--to colorize`**, since
  it only ever truncated the shape. It lives in `CONTROL_ARGS`, which are never
  persisted — as a global arg, a recorded `--no-mvs` would have made an output
  directory stay SfM-only until `--mvs` was passed back explicitly.
  `pipeline_shape()` therefore always contains the MVS stages, and `--no-mvs`
  with a conflicting `--to` is an error.
- **`retexture.py`'s `resolve_recon_inputs()` reads the stage records**, which is
  why records store **inputs** as well as outputs: the SfM that fed
  `openMVG2openMVS` is not any single stage's output (it is whichever of
  `sfm`/`robust`/`autoscale` ran last). The old behavior — grepping every
  recorded command string for `openMVG2openMVS`, parsing the token after `-i`,
  re-rooting at `mvg/recon_dir/`, and re-deriving the mesh name from
  `name` + `file_type` — is kept as a fallback so pre-existing output
  directories stay re-texturable, and the path taken is logged.
- **`--dry-run` resolves everything and executes nothing, and writes nothing**:
  it prints loaded args, shape, skip-vs-run per stage with a reason, rehydrated
  inputs and the prereq check, then returns *before* the output directories are
  created, before the config file, and before `effective_args`/`shape`/`runs`
  are touched. A dry run that persisted its arguments would be worse than
  useless: `--dry-run --no-mvs-refine` would silently drop refine from every
  later run in that directory. Given no range it doubles as a status query, and
  every real run logs the same plan block.
- **This has the repo's only unit tests** (`tests/test_stages.py` for the
  planner — stdlib `unittest`, no fixtures and no filesystem; `test_tracker.py`
  for the records `StageTracker` writes, which needs a temp dir but still no
  binaries; plus dry-run cases in `test_reconstruct.py` that skip when
  `configargparse`/`sfm_utils` are absent). `test_tracker` drives whole
  pipelines through the real `begin()`/`end()` and plans against the manifest
  they leave, which is what keeps `test_stages`' hand-written model of those
  records honest. This is possible because the pure functions raise `StageError`
  instead of calling `sys.exit`; `main()` catches it. `utility.run_command` now
  makes the same move with `ToolFailed`, which also preserves the child's exit
  status — so an OOM-killed stage exits 137 instead of a uniform 1.
- **Per-stage resource records** come from `resource.getrusage(RUSAGE_CHILDREN)`,
  which makes node sizing empirical. `ru_maxrss` is a monotonic high-water mark
  over *all* reaped children — exact when a job runs one stage (the common case
  here), "max so far" otherwise — and its units are KB on Linux, bytes on macOS.
- **`--output` must be on a filesystem every job can see.** pgs-recon has no
  copy logic; node-local scratch staging belongs in the Slurm script. The
  contract stays "one directory, many runs".
