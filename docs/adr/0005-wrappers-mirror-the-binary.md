# Wrappers mirror the binary; the pipeline owns policy

**Status: proposed — not yet implemented.** Describes MR2 of the issue #17
series; see [the plan](../wrapper-refactor-plan.md). Until it lands, the wrappers
still take a `paths` dict plus `*_key` strings as described under Context.

## Context

Every stage function in `openmvg.py`/`openmvs.py` took a `paths: Dict[str, Path]`
plus `*_key` strings: it read its input from the dict, derived an output filename,
inserted that under a new key, and returned the key. Data flow between stages was
therefore implicit — you traced key strings — and reuse from outside `pgs-recon`
meant fabricating a dict (`retexture.py` pre-seeded `sfm_ir`/`mesh` purely to have
names to pass, and `calibrate.py` read a `sfm_expanded` key it never set).

[ADR 0004](./0004-staged-resumable-runs.md) deliberately *extended* that
convention — *"Every wrapper therefore takes its inputs as `*_key` arguments,
including the three OpenMVG ones that used to reach for `paths['sfm']`
directly"* — but its stated reason was that a wrapper must consume its input
**explicitly** rather than reaching for a global, so a stage cannot record a role
it did not actually consume through the bindings. A `Path` parameter serves that
reason strictly better than a key does. This is a refinement of 0004, not a
reversal.

Three unrelated jobs were also conflated in the one dict: toolchain lookup
(`BIN`, `MVS_BIN`, `CAM_DB`), stage data flow, and the `metadata['paths']` debug
dump. Converting only the data flow would have left callers still hand-building
the toolchain half, which is the reuse problem.

## Decision

**A wrapper translates one Python call into one binary invocation, and does
nothing else.** Free functions, one per binary, no classes to instantiate.

- **Parameters mirror the binary's own flag names.** `mask_value` →
  `ignore_mask_label`, `marker_pix` → `min_marker_pix`. `None` still means *omit
  the flag*, so the binary's default wins — this is load-bearing and unchanged.
- **Toolchain config is process-wide, not threaded.** `configure(prefix=...,
  recorder=...)` once in `main()`; `using(...)` for scoped overrides. Resolution
  is `explicit kwarg → configure() → $PGS_RECON_PREFIX → /usr/local → error
  naming the tier it came from`, performed **at call time**.
- **`toolchain.run()` is the single chokepoint** that records to
  `metadata['commands']` and then executes. No wrapper takes a `metadata`
  argument any more, and none can forget to record.
- **Return a path only when the *binary* chose the name.** That is `mvg_sfm`
  (`recon_dir/sfm_data.bin`) and `mvg_localize` (`sfm_data_expanded.json`).
  Everywhere else the caller named the output and already holds it, so wrappers
  return `None`. The pass-through `(scene_key, mesh_key)` returns of
  `mvs_reconstruct`/`mvs_refine` are gone — every caller discarded the scene, and
  ADR 0004 already forbids declaring a pass-through as a production.
- **Naming moves out**, to `layout.py` — see
  [ADR 0006](./0006-stage-named-artifacts.md).
- **The wrapper is a library surface; the pipeline owns policy.** Every flag is
  reachable by a downstream caller. Invariants that are *ours* are enforced and
  tested at the pipeline layer, not by removing parameters.

`StageTracker.key()` is deleted. It existed only to bridge the role→`Path` chain
into the key convention by inventing synthetic `resume_<role>` keys;
`tracker.path(role)` was always the real accessor.

## Consequences / non-obvious traps

- **`archive_type` is a parameter, but its default is `-1`, not `None`.** It is
  therefore *always emitted*, which is what
  [ADR 0003](./0003-portable-mvs-intermediates.md) actually requires — 0003's
  concern is the flag being **implicit** (so an upstream default change silently
  reintroduces Boost archives), not its being overridable. A library user passing
  `archive_type=0` is making a conscious choice. Do not "tidy" this into the
  `None`-means-omit convention.
- **`point_cloud` is nullable with a default**, because `None` is a *legitimate*
  configuration in our own pipeline (the no-densify shape), not a mistake. The
  ADR 0003 invariant — `-p` must be passed whenever densify ran, or
  `ReconstructMesh` silently builds from the sparse cloud — is a **pipeline
  test** asserting `-p` is in the argv whenever the `cloud` role is bound. Both
  0003 invariants are now testable for the first time; previously they held only
  because no caller passed anything else.
- **`resolve_exe` validates against the local filesystem, so it is incompatible
  with containerized remote execution.** Where a binary lives is a property of
  the execution environment, not of the submitting process. If Slurm/Apptainer
  execution is ever wanted, the change is: put the tool *name* in argv and move
  resolution into the runner (local runner resolves + validates, container runner
  prefixes `apptainer exec`). That is deliberately not built now — see below.
- **No `Runner` abstraction.** It was designed and dropped: its only current
  consumer would be tests, and `mock.patch` on the chokepoint serves those with
  no production indirection. A speculative Slurm runner would also have needed
  the resolution change above, so building the seam without it would have bought
  nothing. Expect this to be re-proposed; the forward path is one line per
  wrapper precisely *because* wrappers are pure translation.
- **Resolution must stay late-bound.** A module-level default bound at
  definition time (`def f(bin_dir=_bin_dir)`) freezes at import, so a later
  `configure()` silently does nothing. Discovery must also stay off the import
  path: CI runs the test suite *before* it builds `dependencies/`, so
  `import openmvs` has to work on a machine with no OpenMVS.
- **The test seam is `mock.patch` on `toolchain.run_command`**, so `run()` must
  reference it as a module global rather than capturing it. A fake must learn each
  output path by parsing argv — **never** by calling `layout`, which would make
  the fixture reimplement the code under test.
- **`run_command` raises `ToolFailed` rather than `sys.exit`-ing** (a separate
  change, landing ahead of this). `sys.exit` destroyed the child's exit code, so an
  OOM-killed `RefineMesh` reported exit 1 instead of a signal death, and the bare
  `except:` swallowed `KeyboardInterrupt`. `StageError` had already made this move
  for the planner, for the same testability reason.
- **`metadata['commands']`'s format is unchanged** — timestamp → joined command
  string. `recon_dir._sfm_from_commands` greps those strings for
  `openMVG2openMVS` and parses the token after `-i` as the legacy-manifest
  fallback, so the format is a compatibility surface, not an internal detail.
- **`metadata['paths']` shrinks to the output layout.** Nothing reads it
  programmatically: `recon_dir.py` resolves artifacts from the stage records and,
  failing that, the command log.
- **The Python importers are not binary wrappers.** `init_sfm_generic2` and
  `init_sfm_pgs` stay free functions outside the toolchain, taking explicit
  `Path`s. `init_sfm_pgs` returns `Optional[Path]` for the view-pairs file: only
  the importer knows whether the scan was a grid scan, which is the same
  "return what the caller cannot predict" rule.
