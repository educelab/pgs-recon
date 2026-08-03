# Wrappers mirror the binary; the pipeline owns policy

**Status: accepted, implemented.** Landed as MR2 of the issue #17 series; see
[the plan](../wrapper-refactor-plan.md). It moved naming into `layout.py` while
reproducing the old chained names exactly;
[ADR 0006](./0006-stage-named-artifacts.md) then changed those names, and
`layout.py`'s return values were all it had to touch.

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
`tracker.path(role)` was always the real accessor. Its one load-bearing side
effect — refusing to hand a stage a role nothing has produced — survives as
`tracker.require(role)`, which returns the `Path` or raises `StageError`.
`path()` remains for roles that may legitimately be unbound (`cloud`,
`view_pairs`), and the tracker no longer takes a `paths` dict at all, only the
output root.

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
- **The working directory is derived, not passed.** `toolchain.work_dir(*artifacts)`
  returns the single directory its arguments share and raises if they disagree.
  Every OpenMVS stage addresses its scene and geometry by basename against `-w`,
  and `openMVG2openMVS` writes relative to `cwd`, so a wrapper that took the
  directory as a parameter could be handed one its artifacts do not live in —
  which fails inside the binary, on the file it *did* find, rather than at the
  call. It is the third job in `toolchain` for that reason: it is about running,
  not about naming. It raises `ArtifactsNotColocated`, a `ToolFailed` — the same
  move `ToolNotFound` makes, so the mis-wiring lands in the run's log beside the
  stage that caused it instead of as a traceback.
- **`-w` is the frame the scene's image paths resolve against, which is the
  reason it is derived rather than passed.** Established by reading the pinned
  revision (`ca991d5`) rather than assumed: `openMVG2openMVS` writes each image
  name relative to the scene file's own directory
  (`main_openMVG2openMVS.cpp:54-55,112`), and OpenMVS resolves it against
  `WORKING_FOLDER_FULL` on load and re-saves it relative to the same on save
  (`Scene.cpp:144,267`). Two different anchors that coincide only while `-w`
  holds the scene. A `-w` pointing elsewhere finds no images, or writes a scene
  whose image paths are now relative to the wrong root. The relativity is also
  what makes a finished `mvs/` directory movable, which is
  [ADR 0003](./0003-portable-mvs-intermediates.md)'s portability claim — and why
  nothing checks the images a scene references: the frame is guaranteed by both
  sides, and moving the directory keeps it.
- **The artifacts' co-location is now our invariant, not the binary's.** The same
  read settles what `-w` does *not* constrain: `-i`, `-m`, `-o` and `-p` all go
  through `MAKE_PATH_SAFE` (`Common.h:101`), which keeps an absolute path verbatim
  and otherwise joins onto `-w` and collapses `folder/../` textually
  (`Util.h:407-438`). So a mesh outside the working directory is reachable, both
  `../`-relative and absolute — the opposite of `-M`, where absolute is the one
  unusable spelling. `work_dir` is therefore stricter than `RefineMesh` and
  `TextureMesh` require, exactly as `basename_in` once was. It stays strict for
  now because relaxing it is a behaviour change and `-w` itself does not relax;
  densify additionally pairs its dense cloud with its scene *by name*
  (`layout.densify_cloud`), which needs those two co-located whatever `-w`
  accepts. Tracked as issue #19 rather than done here.
- **Auxiliary path arguments go absolute.** A mask folder or a view-neighbors
  list is not one of the basename-addressed artifacts, so `work_dir` is not
  consulted for it — it must not be, since a folder's parent is not the folder.
  `MAKE_PATH_SAFE` means absolute is the spelling that cannot be silently
  re-rooted against a working directory the caller did not choose.
- **The flag surface is tabulated, so "complete" is testable.**
  `tests/test_openmvs.SURFACES` transcribes each binary's own
  `options_description` groups and asserts every entry is reachable, spelled
  correctly, and omitted when unset; `NOT_MIRRORED` names the exclusions
  (`--help`, `--config-file`, `--process-priority`, `--verbosity`,
  `--cuda-device`, and each app's undocumented "Hidden options" group). This ADR's
  promise was unenforced for two MRs and had already lapsed — `mvs_refine`
  reached six of nineteen options, missing the two that disable the CGAL remesh
  that dominates refine's wall clock at this pin. A signature cannot distinguish
  "we decided not to" from "we forgot"; the table can.
- **The OpenMVG half is tabulated the same way, and had lapsed further.**
  `tests/test_openmvg.SURFACES` does for the nine OpenMVG binaries and our
  `pgs-global-scaler` what the MVS table does for four, transcribed from the
  `cmd.add` calls at that pin (`c92ed1b`). Across the ten, 101 options are
  registered; 60 of the 99 we mirror were reachable before this, and the two we do
  not are named. The worst gap was `mvg_sfm` — eight of `openMVG_main_SfM`'s
  eighteen — and the ten it missed included **every** knob the GLOBAL and STELLAR
  engines read (`-R`/`-T`, `-G`/`-g`), both engines `pgs-recon
  --mvg-recon-method` offers and one of them its default. Nothing failed, because
  OpenMVG accepts a flag its engine ignores in silence; the tuning simply could
  not be expressed. `mvg_autoscale` reached nine of its eighteen mirrored options,
  missing `--scale-method` and the mesh pair among them, and that binary is *ours*
  — which is the argument for tabulating a surface even when we own both sides of
  it. Only `mvg_colorize_sfm`, `mvg_localize` and `mvg_to_mvs` were already
  complete, and the first has two flags in total.
- **A binary's *registrations* are the surface, not the usage text it prints.**
  The two disagree at this pin: `main_SfM.cpp` documents `--triangulation_method`
  and `--resection_method` with no short form while registering `-t` and `-r`.
  Transcribe `cmd.add`, because that is what parses.
- **The same audit applies to a flag's *values*, and `tests/test_reconstruct`
  now tabulates those.** `pgs-recon --matching-method ANNL2` was offered for as
  long as the parser has existed, and stopped parsing when upstream replaced ANN
  with HNSW — so choosing it got you `Invalid Nearest Neighbor method` and a dead
  matches stage, *after* features had run. The three HNSW matchers that replaced
  it were never added, which left no approximate matcher reachable at all. Fixed
  here, with `ACCEPTED` transcribing what each binary parses so a `choices` list
  cannot drift again. The assertion is a **subset**, not equality: offering fewer
  values than the binary accepts is a curation (`--describer-method` omits
  `SIFT_ANATOMY`, deliberately), while offering one it rejects is only ever a bug.
- **OpenMVG's flags cannot be derived from the parameter name, so the table
  carries both.** `openmvs.py` maps `max_face_area` → `--max-face-area`
  mechanically; OpenMVG cannot, in two independent ways. The same letter means
  different things per binary (`-f` is `--force`, `--focal`, `--match_file` and
  `--refine_intrinsic_config` in four different apps), so no global mapping
  exists; and the long spellings mix `snake_case`, `camelCase` and bare words. The
  wrappers therefore pass short flags — which is also what the recorded argv has
  always held — and `SURFACES` is `{keyword: flag}` rather than a list of names.
  A per-binary table invites one specific transcription error, two arguments
  mapped to one letter, which `CmdLine` would resolve silently by taking the last;
  a test asserts each binary's spellings are unique.
- **How a flag is spelled follows how it was registered, and the three kinds are
  not interchangeable.** `make_option` over a string or number is the ordinary
  `None`-omits case. `make_option` over a **bool** takes a value, so it is
  tri-state here: `None` omits, `False` emits `0`, `True` emits `1`. That is
  load-bearing exactly once — `group_camera_model`'s upstream default is *true*,
  so `False` is the only way to turn intrinsic sharing off and omission is not
  the same thing. `make_switch` carries no value at all (the binary reads
  `cmd.used('P')`), so `False` and absent are one argv and those stay plain
  `bool`. `compute_features`'s `upright` keeps the plain-`bool` shape it predates
  this table with, on the same footing as `mvs_reconstruct`'s
  `free_space_support`: `0` is already OpenMVG's default there.
- **`--force` is mirrored rather than excluded, and left unset.** It is tempting
  to file it with `--verbosity` as being about the run, but it changes whether the
  binary recomputes — the computation, not the process. Leaving it omitted is what
  makes a killed features stage resumable at all: OpenMVG skips every image whose
  regions it already finds, which is ADR 0004's premise doing its work through a
  default rather than through anything we wrote. `NOT_MIRRORED` is therefore two
  entries, both `pgs-global-scaler`'s (`--help`, `--progress`); OpenMVG's
  `CmdLine` registers no equivalent of either.
- **`threads` reaches `-n` only in an OpenMP build.** Upstream registers it inside
  `#ifdef OPENMVG_USE_OPENMP` in `ComputeFeatures`, `SfM_Localization` and
  `openMVG2openMVS`, and a build without OpenMP *rejects* the flag rather than
  ignoring it. Ours has it; a library user's prefix may not, which is the one
  place this surface is a property of the build and not of the pin.
- **Where the binary's own join is more forgiving, translate instead of
  restricting.** `openMVG_main_SfM`'s `-M` looks like the same problem and is
  not: `main_SfM.cpp` joins it onto `-m` with `create_filespec`, and that join
  resolves `../`, so `sub/matches.bin` and `../other/matches.bin` both work — the
  matches file need not be in the regions directory at all. Only an *absolute*
  `-M` is broken, because stlplus concatenates (`/regions//abs/path`). So
  `toolchain.relative_to_dir(artifact, directory)` returns the spelling the
  binary will resolve rather than rejecting anything: the wrapper stays a
  complete library surface over the flag, which is this ADR's whole premise, and
  the one unusable spelling becomes unreachable by construction. An earlier
  revision enforced co-location here; that was a restriction the binary does not
  impose, and the distinction is why the two helpers differ.
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
- **`--path` lost its `/usr/local/` default**, in `pgs-recon`, `pgs-retexture`
  and `pgs-calibrate` alike. A parser default is passed to `configure()` like any
  other value, so it would shadow `$PGS_RECON_PREFIX` and make that tier
  unreachable from every entry point we ship. The default now lives in exactly
  one place, `toolchain.DEFAULT_PREFIX`. For the same reason `path` is in
  `NO_PERSIST` (ADR 0004): a prefix inherited from the manifest would shadow the
  environment on every job after the first, which is where a staged run most
  needs it. Consequence: the config file each app
  writes omits **every** unset argument, because a literal `path = None` read
  back through `-c` would be parsed as the string `'None'` and send the run
  looking for its binaries under `./None`. (That trap already existed for
  `focal-length` and friends; `--path` is what made it worth fixing.)
- **The Python importers are not binary wrappers.** `init_sfm_generic2` and
  `init_sfm_pgs` stay free functions outside the toolchain, taking explicit
  `Path`s. `init_sfm_pgs` returns `Optional[Path]` for the view-pairs file: only
  the importer knows whether the scan was a grid scan, which is the same
  "return what the caller cannot predict" rule.
