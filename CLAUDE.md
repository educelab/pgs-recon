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
`dependencies/utilities/`, which produce `pgs-global-scaler` and `pgs-generate-markers`).

```shell
# Build the C++ dependencies (slow; installs to dependencies/installed/ by default)
cmake -S dependencies -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build/

# Install the Python pipeline (Python 3.9+)
python3 -m pip install .
```

Disable building bundled libs with `-DBUILD_<EIGEN|JPEG|JPEG_TURBO|OPENCV|CGAL>=OFF`
to use system versions. Note: OpenMVS and OpenCV must link against the same libjpeg.

The only tests are `tests/`: `python3 -m unittest discover -s tests`. They cover
the staged-run planner (`test_stages.py`, pure logic, no filesystem), the stage
records `StageTracker` writes (`test_tracker.py`, a temp dir but no binaries),
artifact naming (`test_layout.py`), binary resolution and the run/record
chokepoint (`test_toolchain.py`), `run_command`'s exit statuses
(`test_utility.py`), and `pgs-recon` end to end against a prefix of fake binaries
plus its `--dry-run` (`test_pipeline.py`, `test_reconstruct.py`, which skip
themselves when `configargparse`/`sfm_utils`/`exiftool` are missing). All
stdlib-only, so they run anywhere in seconds — no reconstruction math is
exercised, only what the pipeline asks the binaries to do. CI
(`.gitlab-ci.yml`) runs that suite, then verifies that
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
commands are standalone utilities (mesh centering, format conversion, mask
generation, scan inspection, quality checks, etc.) mapping to modules in
`pgs_recon/apps/` and `pgs_recon/utils/`.

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
   → optional `mvs_refine` → `mvs_texture`. Final textured mesh lands at
   `<output>/mvs/<name>.obj` (or `.ply`).

`--no-mvs` stops after the SfM/colorize stage.

### How a stage gets its paths (ADR 0005, ADR 0006)

Stages pass `Path`s, and nothing else. There is no `paths` dict threaded through
them and no `metadata` argument:

- **Inputs** come from the tracker's role bindings — `tracker.require('scene')`
  for something a stage cannot run without, `tracker.path('cloud')` for something
  that may legitimately be absent. Both are recorded paths read back from the
  manifest, never recomputed names.
- **Outputs** are named by `pgs_recon/layout.py`, pure functions over the output
  root (plus, until ADR 0006 lands, the input artifact a chained name derives
  from). A wrapper never invents a filename; `layout` is the only place a name is
  written, which is what makes renaming safe.
- **Where the binaries are and what records them** is process-wide:
  `toolchain.configure(prefix=..., recorder=...)` once in `main()`.
  `toolchain.run()` is the single chokepoint that appends to
  `metadata['commands'][timestamp]` (a compatibility surface: `recon_dir.py` greps
  it) and then executes, so no wrapper can forget to record what it ran. An
  `atexit` hook writes the manifest to `<output>/metadata.json`; the effective
  config goes to `<output>/*_recon_config.txt`.

When adding a stage: add a `layout` function for its output, add the wrapper as a
pure argv builder ending in `run()`, and wire it in `run_pipeline` between
`tracker.begin()`/`tracker.end()`, reporting the roles it consumed and produced.
Invariants that are ours rather than the binary's (e.g. always handing
`ReconstructMesh` the dense cloud) belong in `run_pipeline`, not in the wrapper —
the wrappers stay a complete library surface over each binary's flags.

### Module layout

- `pgs_recon/openmvg.py`, `pgs_recon/openmvs.py` — thin wrappers, one function per
  binary, all routing through `toolchain.run`.
- `pgs_recon/toolchain.py` — binary resolution, the run/record chokepoint, and
  what the binaries that resolve names against a directory need: `work_dir()`
  returns the directory a set of artifacts shares, refusing with
  `ArtifactsNotColocated` (a `ToolFailed`) when they disagree — OpenMVS really
  does require co-location. `relative_to_dir()` instead *translates*, for
  `openMVG_main_SfM`'s `-M`, whose join onto `-m` resolves `../`, so the matches
  file may live anywhere; only an absolute path is unusable there.
- `pgs_recon/layout.py` — what every artifact of a run is called.
- `pgs_recon/pgs_data.py` — import logic for EduceLab "PGS Scan" directories,
  including grid-scan neighbor lookup that generates an OpenMVG **view pairs file**
  (limits matching to spatial neighbors via `--matching-pairs-radius`).
- `pgs_recon/utility.py` — `run_command` (subprocess wrapper that raises
  `ToolFailed`, carrying the child's exit status; each app's `main()` catches it
  and exits `128 + signum` for a signal death) and timestamp helper.
- `pgs_recon/utils/` — shared helpers: `apps.py` (logging setup), `geometry.py`,
  `quality.py`, `charuco.py`, `wavefront.py`, `educelab.py` (ChArUco/board detection),
  `recon_dir.py` (locate a finished run's SfM/mesh from its `metadata.json`),
  `sfm_json.py` (OpenMVG SfM_Data JSON surgery: cereal polymorphic registration,
  extrinsic frame transforms), `images.py` (8-bit sRGB normalization).
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
