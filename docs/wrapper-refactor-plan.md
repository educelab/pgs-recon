# Implementation plan: wrapper interface refactor (issue #17)

Working document for the four-MR series that removes the `paths`-dict + `*_key`
convention from the OpenMVG/OpenMVS wrappers. Delete this file when MR3 merges —
the durable decisions live in [ADR 0005](./adr/0005-wrappers-mirror-the-binary.md)
and [ADR 0006](./adr/0006-stage-named-artifacts.md).

## Settled decisions

Resolved in design review; recorded here so the plan reads standalone.

| # | Decision |
|---|---|
| 1 | Names are rationalized, not preserved. Safe because **`layout` writes names; the manifest locates artifacts** — a completed stage's filename is never recomputed. |
| 2 | Scheme is `<stage>_<role>.<ext>`. Stage-prefixed, not bare role names, because roles are rebound and bare names would collide. |
| 3 | Wrappers are a **library surface** — every flag reachable. Invariants that are ours are enforced and tested at the **pipeline layer**, not by removing parameters. `archive_type` defaults to `-1` and is always emitted; `point_cloud` is nullable with a default. |
| 4 | The rename extends to the MVG chain, not just `mvs/`. |
| 5 | Freeze anything already shape-independent: binary-chosen names, `matches*`, `landmarks*`, and the deliverable `mvs/<name>.<ext>`. |
| 6 | Verification is unit tests **plus** end-to-end fake-binary tests. One real reconstruction as a manual gate. |
| 7 | `init_sfm_pgs` is converted too; returns `Optional[Path]` for view-pairs. |
| 8 | Split into MRs on the **risk axis** — interface change (no-op) separately from the rename. |
| 9 | `run_command` raises `ToolFailed` instead of `sys.exit`. |

No `Runner` abstraction. See ADR 0005 for why, and for the forward path if
Slurm/Apptainer execution is ever wanted.

---

## MR0 · `run_command` raises

Lands first: `toolchain.run()`'s contract depends on it.

- [x] `utility.py` — add `ToolFailed(command, returncode)`; raise instead of
      `sys.exit`; drop the bare `except:` so `KeyboardInterrupt` propagates
  - `ToolFailed.exit_code` does the POSIX translation once (`128 + signum` for a
    signal, 127 for a binary that never started), so each `main()` stays 3 lines
- [x] `apps/reconstruct.py`, `apps/retexture.py`, `apps/calibrate.py`,
      `apps/convert.py` — one `except ToolFailed` per `main()`, exiting
      `128 + signum` for signal deaths (so an OOM reads as 137 in a Slurm log)
  - The latter three grew a `main()`/`_main()` split, matching `reconstruct.py`.
    These four are every caller that can reach `run_command`
- [x] `utils/images.py` — non-`main()` caller; confirm it should propagate
      *(it should: both callers catch in `main()`)*
- [x] Test: a failing fake command raises rather than exiting, and
      `StageTracker` still records `failed` *(`tests/test_utility.py`)*
- [x] Prose that documented the old convention: ADR 0004, `CLAUDE.md`,
      `StageError`'s docstring, `_main`'s `abort()` comment

**Why:** `sys.exit(str)` always exits 1, destroying the child's exit code — an
OOM-killed `RefineMesh` (the failure ADR 0004 exists for) was indistinguishable
from any other error. `StageError` already made this move for the planner.

---

## MR1 · Foundation *(additive, nothing imports it yet)*

- [x] `pgs_recon/layout.py` — naming as pure functions, **reproducing today's
      chained names exactly**
  - [x] Preserve `refine_mesh()` deriving from the **scene**, not the mesh
        (ADR 0003: `scene_dense_refine.ply`)
  - [x] Preserve `densify_cloud()` as the scene stem with `.ply`
  - Functions are named for the MR3 endpoint (`<stage>_<role>`), so that rename
    is a change of bodies. Argument rule: **the output root, unless the name
    chains off an input artifact, in which case that artifact** — so MR3 also
    drops the input argument from `robust_sfm`/`autoscale_sfm`/`colorize_sfm` and
    the four MVS chain functions, and every signature becomes `(output, ...)`
  - Also covers the layout `main()` spells out today: `manifest()`, `config()`,
    `directories()` for the mkdir pass
- [x] `pgs_recon/toolchain.py`
  - [x] `resolve_exe()` — `explicit kwarg → configure() → $PGS_RECON_PREFIX →
        /usr/local → error naming the tier`. **At call time**, never import time
    - `ToolNotFound` subclasses `ToolFailed`, so every `main()`'s existing
      handler covers it and it exits 127. Also distinguishes a file that is
      present but not executable
    - A caller-supplied prefix is absolutised (and `~` expanded) as it is
      accepted, preserving today's `Path(args.path).resolve()`. Load-bearing,
      not tidiness: `subprocess` resolves a relative argv[0] against `cwd`, so
      a relative prefix would pass `resolve_exe`'s check and *then* fail to
      exec under `mvg_to_mvs`/`mvs_texture`, which both run with `cwd=mvs/`.
      `$PGS_RECON_PREFIX` is the tier that makes this reachable — no argument
      parser stands between it and us
    - The public accessor is `effective_prefix()`, not `prefix()`, which would
      be shadowed by the `prefix` keyword argument in four of this module's
      own functions
  - [x] `configure()` / `using()` (context manager, restores on exit)
  - [x] `run()` — record to `metadata['commands']`, then execute. Must reference
        `run_command` as a module global so `mock.patch` works
  - [x] `Recorder` — also records non-binary Python steps
        (`init_sfm_generic2`, `init_sfm_pgs`), which today hand-write fake
        command strings
    - `cam_db()` moves here too: it is prefix-relative, and ADR 0005 counts it
      as the toolchain half of the old `paths` dict. Deliberately not checked for
      existence
- [x] `tests/test_layout.py` — exact names for every shape permutation
- [x] `tests/test_toolchain.py` — resolution tiers, error message names the tier,
      no filesystem access at import
  - `make_fake_prefix()` lives here, for MR2's end-to-end tests to import

---

## MR2 · Migration *(large, atomic, provably no-op)*

Atomic by necessity: the moment a signature changes, every caller changes with it.

- [x] `openmvg.py` — 10 functions to pure CLI translation
- [x] `openmvs.py` — 4 functions; co-location enforced instead of failing
      silently
  - The check landed as `toolchain.work_dir(*artifacts)` rather than a private
    `openmvs._work_dir()`: `mvg_to_mvs` needs the same derivation for its `cwd`,
    and duplicating it in both wrapper modules was the worse of the two shapes
- [x] `pgs_data.py` — `init_sfm_pgs` to explicit `Path`s, returning
      `Optional[Path]` for view-pairs
- [x] Flag-name mirroring: `mask_value` → `ignore_mask_label`,
      `marker_pix` → `min_marker_pix`, `decimation_factor` → `decimate`,
      `free_space` → `free_space_support`, `resolution_lvl` →
      `resolution_level`, `max_size` → `max_texture_size`, `file_format` →
      `export_type`
- [x] Drop the pass-through `(scene_key, mesh_key)` returns from
      `mvs_reconstruct`/`mvs_refine`
- [x] `apps/reconstruct.py` — `run_pipeline` migrated; one `configure()` in
      `main()`; every `metadata=metadata` argument gone
- [x] `stages.py` — delete `StageTracker.key()`
  - Its guard survives as `require()`, which returns the `Path` and raises
    `StageError` when a mandatory role is unbound. The tracker now takes the
    output root, not a `paths` dict
- [x] `apps/retexture.py` — drop the `sfm_ir`/`mesh` aliases and the
      `paths[out_key]` write-back
- [x] `apps/calibrate.py` — drop the `sfm_db` rebind
- [x] Shrink `paths` to the output layout for the manifest dump
- [x] `docs/adr/0005-wrappers-mirror-the-binary.md` *(status flipped; the
      `work_dir` and `--path` consequences added)*
- [x] `CONTEXT.md` — `Role` entry no longer defined against `paths` keys
      *(written in MR1)*
- [x] ADR 0004 — the `*_key` paragraph, now that the reason it gave is served by
      explicit `Path`s
- [x] **`README.md` — document `$PGS_RECON_PREFIX`.** MR1 added it as a
      resolution tier but deliberately left it undocumented outside the ADR:
      nothing imports `toolchain` yet, so advertising it would promise a knob
      that does not turn. It becomes user-facing the moment `main()` calls
      `configure()`, and the README's binary-discovery section (the `--path`
      default of `/usr/local/`) is where it belongs. `CLAUDE.md`'s "Binary
      discovery at runtime" needs the same sentence
  - Doing this exposed that `--path`'s own `/usr/local/` default made the
    environment tier unreachable from every entry point we ship. The default
    moved to `toolchain.DEFAULT_PREFIX` alone, which in turn required the config
    writers to omit unset arguments (a round-tripped `path = None` would send the
    next run looking under `./None`). Recorded in ADR 0005

### Tests

- [x] Argv assertions via `mock.patch('pgs_recon.toolchain.run_command', ...)`
      *(`tests/test_pipeline.py`)*
  - [x] `--archive-type -1` present on all four MVS stages
  - [x] `-p` present whenever the `cloud` role is bound — including a cloud
        rehydrated from the manifest by a later job, which is the case that
        actually breaks meshes

  *Neither ADR 0003 invariant had ever been tested; they held only because no
  caller passed anything else.*
- [x] End-to-end fake-binary tests
  - [x] `make_fake_prefix()` — touch + `chmod 0755` the binary names, so
        `resolve_exe` is exercised for real
  - [x] `fake_binary()` — parses the output path off **argv**, never from
        `layout`. A fixture that consulted `layout` would reimplement the code
        under test. It also *checks its inputs*, so a mis-wired stage fails on
        the missing input rather than producing an empty artifact
  - [x] Single-shot run vs the same shape staged across three windows
        (`--to convert`, `--from densify --to reconstruct`, `--from refine`) →
        assert identical artifact trees. This automates ADR 0004's stated
        acceptance test for the first time
  - [x] `pgs-retexture` against a recon directory it did not build — the reuse
        case ADR 0005 exists for, previously unreachable without fabricating a
        `paths` dict
- [x] `tests/test_tracker.py` — rewritten; `fake_run` now mirrors
      `run_pipeline`'s `layout` calls instead of the key threading
- [x] `tests/test_stages.py` — `build_records()` names artifacts through
      `layout` (so MR3 does not have to touch it; `test_layout` is what pins the
      literals), and the `key()` test became a `require()` test

**Acceptance gate:** byte-identical artifact tree before and after. Filenames do
not change in this MR by construction, and `test_pipeline` now asserts the whole
tree for the default and densified shapes.

---

## MR3 · Rename *(small, isolated)*

Entire behavioural risk confined to one file's return values.

- [ ] `layout.py` — return values change to `<stage>_<role>.<ext>`:

  | Stage | Was | Now |
  |---|---|---|
  | robust | `sfm_data_structured.bin` | `robust_sfm.bin` |
  | autoscale | `sfm_data_structured_scaled.bin` | `autoscale_sfm.bin` |
  | colorize | `sfm_data_..._colorized.ply` | `colorize_sfm.ply` |
  | convert | `scene.mvs` | `convert_scene.mvs` |
  | densify | `scene_dense.mvs` / `.ply` | `densify_scene.mvs` / `densify_cloud.ply` |
  | reconstruct | `scene_dense_mesh.ply` | `reconstruct_mesh.ply` |
  | refine | `scene_dense_refine.ply` | `refine_mesh.ply` |

- [ ] Test expectations updated. Larger than it looks: the *fixtures* follow the
      rename on their own (`fake_run`/`build_records` call `layout`), but ~20
      **assertions** spell the chained names out literally —
      `tests/test_tracker.py` (`:201, 220, 265, 274, 381, 416, 425`),
      `tests/test_stages.py` (`:397, 398, 423`) and `tests/test_pipeline.py`
      (the expected-tree list, the `-p`/`-i` flag checks, the staged-vs-single
      spot-check). All mechanical. Frozen names (`mvg/sfm_data.json`,
      `recon_dir/sfm_data.bin`, `matches_filtered.bin`) do not move
- [ ] **Legacy-resume test**: a manifest carrying the *old* chained names with
      every stage complete goes clean on a verbatim re-run. Pure dict logic, no
      filesystem — belongs in `test_stages.py`
- [ ] `docs/adr/0006-stage-named-artifacts.md` *(written)*
- [ ] `CONTEXT.md` — **Artifact name** term *(written; ahead of the code until
      this MR lands)*
- [ ] ADR 0003 — prose naming `scene_dense.mvs` / `scene_dense_refine.ply`
- [ ] ADR 0004 — prose naming `mvs/scene.mvs`, `scene_dense_mesh.ply`,
      `scene_refine.ply`, and the `mvs_scene_dense_mesh_refine` key example
- [ ] `apptainer/submit_recon_pipeline.sh:112` — the comment explaining that
      densify renames the mesh chain. The constraint relaxes; the advice (keep
      shape flags on job 1) stands
- [ ] **Manual gate before merge:** one real reconstruction on a known dataset.
      The fake-binary tests cannot observe whether `-p` actually prevented a
      sparse-cloud mesh — only real OpenMVS can

---

## Open follow-ups

Raised in review of MR2 and **not yet closed**. The series is not finished until
these are, so they are tracked here rather than in an MR thread that disappears.

- [ ] **CI never runs the unit suite where the binaries exist.** `test:unit`
      (`python:3.9-slim`) and `test:python` (`ubuntu:22.04`) both run against a
      `/usr/local` with no OpenMVG in it, so any test that reaches the built-in
      default prefix passes for the wrong reason. Three did: they asserted
      `ToolNotFound` for real tool names and failed inside
      `ghcr.io/educelab/pgs-recon:edge` while CI stayed green. MR2 fixed those
      with `tests/test_toolchain.ABSENT_TOOL`, but nothing stops the next one —
      add a job running `python -m unittest discover -s tests` inside the built
      image. Cheap, and it is the only environment that resembles production.
- [ ] **`mvg_to_mvs` co-locates the scene and the undistorted images, but
      nothing checks the images the scene references.** `toolchain.work_dir`
      covers the two paths the wrapper is handed; the MVS stages then read image
      paths out of the scene itself. A scene pointing at images that moved fails
      inside OpenMVS. Out of scope for the rename, worth deciding on before the
      series closes.
- [ ] **Is `work_dir`'s co-location requirement real for OpenMVS, or only
      assumed?** This is the same question `-M` turned out to answer the other
      way. `openMVG_main_SfM` looked like it needed the matches file beside the
      regions and does not: it joins `-M` onto `-m`, and the join resolves
      `../`, so any location is reachable (`b25cb6e`). Whether OpenMVS's `-w` is
      equally forgiving is **unestablished** — it is a different codebase, using
      Boost program_options and its own path handling, and nothing here has been
      checked against it. If it is forgiving, `work_dir` is over-strict for
      exactly the reason `basename_in` was, and `mvs_refine`/`mvs_texture` should
      translate rather than refuse.

      Two things do *not* relax either way, and are the reason this is a question
      rather than a presumed bug: densify pairs its dense cloud with its scene
      **by name** (`layout.densify_cloud` is the scene's stem with `.ply`), which
      needs them in one directory whatever `-w` accepts; and `mvg_to_mvs` writes
      relative to `cwd` rather than to a `-w`, so its own derivation is
      unaffected. Settle it by compiling against the installed OpenMVS the way
      `create_filespec` was checked — reading `-w` handling out of
      `DensifyPointCloud`/`RefineMesh` — rather than by inspection.

---

## Known non-issues

Verified during design review, recorded so they are not re-litigated:

- **Nothing locates an artifact by reconstructing a chained name.** Every
  suffix-construction site is inside a wrapper body; inputs come from the
  manifest (`recon_dir.py`) or `tracker.path()`, both reading recorded paths
  verbatim.
- **`metadata['paths']` has no programmatic consumer.** `recon_dir.py` resolves
  from stage records, falling back to the command log.
- **`metadata['commands']` format is a compatibility surface** —
  `recon_dir._sfm_from_commands` greps it for `openMVG2openMVS`. Keep
  timestamp → joined-string.
- **Densify does still write a scene `.mvs`** (holding the *sparse* cloud; the
  dense cloud goes to `.ply`). It is the only MVS stage that writes a scene —
  reconstruct/refine/texture emit only the mesh under `-1`.
