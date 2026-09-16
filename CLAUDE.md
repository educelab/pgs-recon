# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python pipeline that reconstructs 3D models from photogrammetry image sets by
orchestrating the [OpenMVG](https://github.com/openMVG/openMVG) (Structure-from-Motion)
and [OpenMVS](https://github.com/cdcseacave/openMVS) (Multi-View Stereo) C++
toolchains. The Python code is almost entirely a **subprocess orchestration layer**:
it builds argument lists and shells out to compiled binaries. Almost no
reconstruction math happens in Python — the heavy lifting is in the external
executables.

## Build & install

The compiled binaries are NOT part of this repo. They are built from `dependencies/`
via a CMake superbuild that compiles OpenMVG, OpenMVS, VCG, CGAL, OpenCV, Eigen,
Ceres, libjpeg, plus the in-tree `pgs-recon-utilities` (C++ tools in
`dependencies/utilities/`, which produce `pgs-global-scaler`, `pgs-sfm-orient`,
`pgs-generate-markers`, `pgs-decimate` and `pgs-localize`). `pgs-decimate` is the
one that needs VCG, whose headers the superbuild already installs for OpenMVS;
`BuildPGSUtils.cmake` passes `-DVCG_ROOT` the way `BuildOpenMVS.cmake` does.
`pgs-localize` is the one C++20 target (bvh v2 needs `std::span`), which is why
the one file that has to meet openMVG's bundled cereal -- it does not compile at
C++20 -- is its own small C++17 library (ADR 0011). bvh and libcore arrive by
`FetchContent` at configure time, so the utilities build needs network.

```shell
# Build the C++ dependencies (slow; installs to dependencies/installed/ by default)
cmake -S dependencies -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build/

# Install the Python pipeline (Python 3.9+)
python3 -m pip install .
```

Iterating on the C++ utilities does **not** need a superbuild. Everything they
link against is already in the published image, so build only
`dependencies/utilities` there -- minutes rather than hours:

```shell
docker run --rm -v "$PWD":/src -w /src ghcr.io/educelab/pgs-recon:edge \
  bash -lc 'cmake -S dependencies/utilities -B /tmp/bu -DCMAKE_BUILD_TYPE=Release \
            && cmake --build /tmp/bu -j'
```

Mount a host directory as the build dir to keep the CMake cache and object files
between runs. The image carries OpenMVG, OpenCV, Eigen, VCG, EduceLabCore and
Boost 1.74 under `/usr/local`; libcore and bvh still arrive by `FetchContent`, so
the configure step needs network.

Disable building bundled libs with `-DBUILD_<EIGEN|JPEG|JPEG_TURBO|OPENCV|CGAL>=OFF`
to use system versions. Note: OpenMVS and OpenCV must link against the same libjpeg.
The libpng and libtiff development headers must **either both be present or both
be absent**. OpenMVG bundles zlib 1.2.3 whenever it misses either one; if the
other is then a system library it keeps calling the system zlib, and the
statically linked 1.2.3 wins at symbol resolution — aborting with `libpng error:
bad parameters to zlib` the first time a run reads a mask. `BuildOpenMVG.cmake`
fails configuration on that mismatch. The images install both.

The only tests are `tests/`: `python3 -m unittest discover -s tests`. They cover
the staged-run planner (`test_stages.py`, pure logic, no filesystem), the stage
records `StageTracker` writes (`test_tracker.py`, a temp dir but no binaries),
artifact naming (`test_layout.py`), binary resolution and the run/record
chokepoint (`test_toolchain.py`), the MVS and MVG wrappers' flag surfaces
(`test_openmvs.py` and `test_openmvg.py`, whose `SURFACES` tables are what make
ADR 0005's "every flag reachable" enforceable — the MVG one maps each keyword
argument to its argv flag, because OpenMVG's spellings are per-binary and cannot
be derived; the MVS one additionally checks itself against the installed
binaries' generated `--help`, per binary, so a tool the published image does not
carry yet skips rather than taking the others down), `run_command`'s exit
statuses
(`test_utility.py`), and `pgs-recon` end to end against a prefix of fake binaries
plus its `--dry-run` (`test_pipeline.py`, `test_reconstruct.py`, which skip
themselves when `configargparse`/`sfm_utils`/`exiftool` are missing), and the
shapes `utils.charuco` promises its consumers whatever OpenCV returned
(`test_charuco.py`, which draws a synthetic sample square at a known
pixels-per-cm, and needs `cv2`). The planner-and-wrappers core is
stdlib-only, so it runs anywhere in seconds — no reconstruction math is
exercised, only what the pipeline asks the binaries to do. The C++ tools carry their own: `pgs-decimate --self-test` checks a measured
deviation against the analytic answer on a generated sphere, and
`pgs-localize --self-test` renders a generated scene from a known pose and
checks the conventions that are silent when wrong -- the position map
reprojecting onto its own pixel centres, depth being camera-space Z rather than
slant range, the UV origin, and nearest-hit occlusion. Neither is reached by the
Python suite; both need the built binary. CI
(`.gitlab-ci.yml`) runs that suite three ways — bare Python, with the Python deps
installed, and inside `ghcr.io/educelab/pgs-recon:edge` (`test:in-image`, the only
one where the binaries exist, so tests reaching the default prefix cannot pass for
the wrong reason) — then verifies that
dependencies build and the package pip-installs on Ubuntu 22.04 / 24.04. The
canonical GitHub Actions workflow (`.github/workflows/build_docker.yml`)
builds/publishes Docker images.

## Running

Most development/testing happens via Docker (see README), but the entry points run
directly once installed. The main pipeline:

```shell
pgs-recon -i <image_dir> -o <output_dir> --name <object_name>
```

All console scripts are declared in `setup.cfg` under `[options.entry_points]`.
`pgs-recon` (`apps/reconstruct.py:main`) is the full pipeline; the other `pgs-*`
console scripts are standalone utilities (mesh centering, format conversion, mask
generation, scan inspection, quality checks, etc.) mapping to modules in
`pgs_recon/apps/` and `pgs_recon/utils/`. Five more `pgs-*` commands are bare C++
binaries with no Python wrapper at all — `pgs-global-scaler`, `pgs-sfm-orient`,
`pgs-generate-markers`, `pgs-decimate` and `pgs-localize` — built from
`dependencies/utilities/src/`.

`pgs-localize` puts a camera that was never in the reconstruction into the solved
scene and emits the one-view calibration `pgs-retexture --calibration` consumes.
It carries two correspondence backends — sparse (`--input-scene` +
`--matches-dir`) and render-and-match (`--mesh` + a prior pose), which renders the
mesh in the query's own modality and lifts matched keypoints through the render's
position map — and chains them when given both. It replaced `pgs-calibrate`,
which was deleted; see ADR 0011 for the measurement behind that.

### Binary discovery at runtime

`pgs_recon/toolchain.py` resolves every binary under an install prefix, **at call
time**, from the first of: the hidden `--path <prefix>` arg → `$PGS_RECON_PREFIX`
→ `/usr/local/` (`toolchain.DEFAULT_PREFIX`), i.e. it assumes the
container/install layout, not the CMake default of `dependencies/installed/`.
When running outside Docker, point one of the first two at the dir containing
`bin/`; a failure names both the path it looked for and which of those tiers
chose the prefix. OpenMVG binaries and our `pgs-*` utilities live in `bin/`,
OpenMVS's in `bin/OpenMVS/`. The OpenMVG camera sensor database is expected at
`<prefix>/lib/openMVG/sensor_width_camera_database.txt` (override with the hidden
`--cam-db`).

## Architecture

### Pipeline flow (`pgs_recon/apps/reconstruct.py`)

`run_pipeline()` runs the stages in fixed order; each stage names its output, calls
a wrapper that builds one binary invocation, and reports the result to the
tracker:

1. **Import / SfM init** → one of: `init_sfm_pgs` (PGS scan dirs, `-p`),
   `init_sfm_generic2` (new EXIF-based importer), or `init_sfm_generic` (default,
   uses OpenMVG's image listing tool).
2. **Features** (`compute_features`) → **Matches** (`compute_matches`) →
   **Geometric filter** (`geometric_filter`) — all OpenMVG (`pgs_recon/openmvg.py`).
3. **SfM reconstruction** (`mvg_sfm`, or `mvg_compute_known` for the `direct` method).
4. Optional **robust triangulation**, **autoscale** (`mvg_autoscale` → `pgs-global-scaler`),
   then **colorize** (`mvg_colorize_sfm`).
5. **MVG→MVS conversion** (`mvg_to_mvs`).
6. MVS stages in `pgs_recon/openmvs.py`: optional `mvs_densify` → `mvs_reconstruct`
   → optional `coarsen` → optional `mvs_refine` → optional `mvs_decimate` →
   `mvs_texture`. Final textured mesh lands at `<output>/mvs/<name>.obj` (or
   `.ply`).

`--no-mvs` stops after the SfM/colorize stage.

The `coarsen` stage drives `mvs_decimate` too, to a *face count* rather than a
deviation budget, replacing `RefineMesh`'s own single-threaded CGAL
pre-refinement pass — whose cost at a fixed target varies 135x between meshes
(ADR 0009). It is on by default whenever `refine` is, and being in the shape is
what makes `refine` run `--decimate 1 --ensure-edge-size 2`: the binary guards
the edge-size pass on its own decimation, so one flag without the other is a
silent 3.9x regression in refine input. That coupling is derived in
`stages.refine_flags` rather than folded into `args`, so it cannot be inherited
across a shape change. `--ensure-edge-size 2` ("force") is also the only value
upstream lets reach the later image scales, where the pass remeshes what
subdivision just grew;
`dependencies/patches/openMVS-v2.4-ScopeEnsureEdgeSizeToInput.diff` scopes it to
the first scale, where the other two values already sat (ADR 0010) — so the
wrapper's `2` means something narrower here than in stock OpenMVS.
`stages.COARSEN_RATIO` is the target fraction (0.375, derived as
`6 * 1.0 px^2 / 16`) and `utils.ply.face_count` reads the input's count out of
its PLY header — the only place the pipeline reads a mesh file.

`mvs_decimate` is the one MVS-side wrapper that is not OpenMVS: it drives our
`pgs-decimate`, which coarsens a mesh as far as a *measured* deviation budget
allows and writes a JSON report of what that cost (ADR 0008). `--mvs-decimate`
puts it in the shape, on by default, and `stages.decimate_target()` names what it
coarsens to: `--decimate-max-faces` if given, else `DECIMATE_RATIO` (0.3) of the
input's face count — the same ladder `coarsen_target()` uses, plus
`--decimate-ratio 0`, which drops the face target so `--decimate-max-error`
searches alone. `--decimate-prefer` is deliberately *not* in that ladder: it
exists to arbitrate the two budgets, so a precedence rule between them would
delete it. `coarsen` carries the identical interface — same targets, same
tie-break, same search, same measurement and geometry flags, generated from
`_add_decimation_options()` so the two cannot drift — and differs only in three
defaults: the ratio, and the two measurement flags it turns down to the floor
because nothing consumes the deviation it produces. Its search is available and
off: no `--coarsen-max-error` is what keeps it to one round, which is what ADR
0009 is actually about. `decimate` is not gated on `mvs_refine` (ADR 0008
Decision 7) -- without refine it coarsens `reconstruct`'s mesh, which is the
largest thing the pipeline ever hands `texture` -- but the *default ratio* has no
derivation there, so `warn_decimate_without_refine()` says so unless the caller
stated a ratio or a count. The ratio is refine's last act read backwards — one uniform
subdivision, measured at 3.80–3.93x across twelve cluster runs, a property of
`--refine-scales` rather than of the object, exactly as `COARSEN_RATIO` is a
property of the rig.

The budgets *used to be* the enable flag: the stage was in the
shape while either was truthy, so `0` removed it on a resume where an omitted
flag would have inherited the recorded budget. A defaulted target can never be
falsy, so `--decimate-ratio` could not have a default while that held. The
budgets now only bound the stage, and `0` on either means "not this target".

### How a stage gets its paths (ADR 0005, ADR 0006)

Stages pass `Path`s, and nothing else. There is no `paths` dict threaded through
them and no `metadata` argument:

- **Inputs** come from the tracker's role bindings — `tracker.require('scene')`
  for something a stage cannot run without, `tracker.path('cloud')` for something
  that may legitimately be absent. Both are recorded paths read back from the
  manifest, never recomputed names.
- **Outputs** are named by `pgs_recon/layout.py`, pure functions over the output
  root: an intermediate is `<stage>_<role>.<ext>` and nothing in its name records
  which other stages ran (ADR 0006). A wrapper never invents a filename; `layout`
  is the only place a name is written, which is what makes renaming safe.
- **Where the binaries are and what records them** is process-wide:
  `toolchain.configure(prefix=..., recorder=...)` once in `main()`.
  `toolchain.run()` is the single chokepoint that appends to
  `metadata['commands'][timestamp]` (a compatibility surface: `recon_dir.py` greps
  it) and then executes, so no wrapper can forget to record what it ran. An
  `atexit` hook writes the manifest to `<output>/pgs-recon.json`; the effective
  config goes to `<output>/*_recon_config.txt`. `stages.find_manifest()` is what
  reads it, falling back to a pre-2.0 `metadata.json` (ADR 0007) — the one
  filesystem check in `stages.py`. `tracker.end(facts=...)` records numbers
  about what a stage did (`coarsen`'s face counts) beside its paths; nothing in
  the planner reads them, so a fact cannot make a stage dirty.

When adding a stage: add a `layout` function for its output, add the wrapper as a
pure argv builder ending in `run()`, and wire it in `run_pipeline` between
`tracker.begin()`/`tracker.end()`, reporting the roles it consumed and produced.
Invariants that are ours rather than the binary's (e.g. always handing
`ReconstructMesh` the dense cloud) belong in `run_pipeline`, not in the wrapper —
the wrappers stay a complete library surface over each binary's flags.

### Module layout

- `pgs_recon/openmvg.py`, `pgs_recon/openmvs.py` — thin wrappers, one function per
  binary, all routing through `toolchain.run`. Each is its binary's *complete*
  registered flag surface, transcribed from the pinned source and held to that by
  the `SURFACES` tables above; when upstream moves, the wrapper and the table are
  edited together. In `openmvg.py` a flag's spelling depends on how OpenMVG
  registered it: `make_switch` flags are plain `bool` (no value exists to emit),
  `make_option` over a bool is tri-state (`None` omits, `False` sends `0`), and
  everything else is `None`-means-omit.
- `pgs_recon/toolchain.py` — binary resolution, the run/record chokepoint, and
  what the binaries that resolve names against a directory need: `work_dir()`
  returns the directory a set of artifacts shares, refusing with
  `ArtifactsNotColocated` (a `ToolFailed`) when they disagree — OpenMVS really
  does require co-location. `relative_to_dir()` instead *translates*, for
  `openMVG_main_SfM`'s `-M`, whose join onto `-m` resolves `../`, so the matches
  file may live anywhere; only an absolute path is unusable there. `run()` is also
  the one place argv becomes strings, so `_argv_token` narrows every `int`
  subclass there rather than in each wrapper's flag builder — that is what makes a
  `bool` reach argv as `0`/`1` and an `IntEnum` as its value (`str()` of one is its
  member name before Python 3.11), including for flags a wrapper assembles by hand.
- `pgs_recon/layout.py` — what every artifact of a run is called.
- `pgs_recon/pgs_data.py` — import logic for EduceLab "PGS Scan" directories,
  including grid-scan neighbor lookup that generates an OpenMVG **view pairs file**
  (limits matching to spatial neighbors via `--matching-pairs-radius`).
- `pgs_recon/utility.py` — `run_command` (subprocess wrapper that raises
  `ToolFailed`, carrying the child's exit status; each app's `main()` catches it
  and exits `128 + signum` for a signal death) and timestamp helper.
- `pgs_recon/utils/` — shared helpers: `apps.py` (logging setup), `geometry.py`,
  `quality.py`, `charuco.py`, `wavefront.py`, `educelab.py` (ChArUco/board detection),
  `recon_dir.py` (locate a finished run's SfM/mesh from its manifest),
  `ply.py` (one function: the face count in a PLY header, which is how `coarsen`
  sizes its target),
  `sfm_json.py` (OpenMVG SfM_Data JSON surgery: cereal polymorphic registration,
  extrinsic frame transforms),
  `visibility.py` (the pinhole projection `pgs-retexture`'s projective-UV path
  maps through, and the z-buffer that says which faces that one camera actually
  saw — ADR 0012),
  `images.py` (`read_srgb`, the **single** reader
  behind `pgs-convert` and `pgs-retexture`). OpenMVG reads
  sRGB, not CIELab, and imageio hands back a Lab TIFF's samples undecoded, so
  the photometric and WhitePoint tags are read per *file* — a capture set can
  mix colorspaces. This used to be ImageMagick for all but `pgs-convert`, which
  ignores WhitePoint and decodes every Lab file as D65 while the EduceLab
  captures are untagged D50; `read_srgb` reproduces it bit-for-bit on 16-bit
  greyscale and 8-bit RGB and differs only there. Nothing shells out for pixels
  any more, and `imagemagick` is gone from the Docker/Apptainer images.
- `pgs_recon/apps/` — standalone CLI utilities (one `main()` each). Apps must NOT
  import each other; anything two apps need belongs in `utils/` (the three modules
  above were extracted from `retexture.py` for exactly this reason).

### Key external Python dependency

`PySfMUtils` (imported as `sfm_utils` / `sfm`) provides the `Scene`, `View`,
`Intrinsic*`, `Pose` model used by the importers and the OpenMVG camera-db loader.
`educelab-imgproc` and `PyExifTool` (wrapping the `exiftool` binary) are also core.

## Conventions

- Wrappers are pure command builders: assemble a `command` list, append optional
  flags conditionally (`None` means *omit the flag*, so the binary's own default
  wins), then `toolchain.run`. Match this style for new binary wrappers rather
  than calling `subprocess` or `run_command` directly.
- Args use `configargparse`; hidden/internal flags use `configargparse.SUPPRESS`.
- The package version lives in `setup.cfg` (`version = ...`).

## Agent skills

### Issue tracker

Issues live as GitLab issues on `educelab/pgs-recon`, driven with the `glab` CLI.
See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, unrenamed: `needs-triage`, `needs-info`,
`ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See
`docs/agents/domain.md`.
