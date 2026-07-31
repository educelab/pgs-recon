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
and `pgs-recon`'s `--dry-run` (`test_reconstruct.py`, which skips itself when
`configargparse`/`sfm_utils` are missing). All stdlib-only, so they run anywhere
in seconds. Nothing else — the reconstruction stages themselves are only exercised by
running the pipeline. CI (`.gitlab-ci.yml`) runs that suite, then verifies that
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

`pgs-recon` looks for binaries under a `--path` prefix that **defaults to
`/usr/local/`** (a hidden arg), i.e. it assumes the container/install layout, not
the CMake default of `dependencies/installed/`. When running outside Docker, pass
`--path <prefix>` pointing at the dir containing `bin/`. The OpenMVG camera sensor
database is expected at `<path>/lib/openMVG/sensor_width_camera_database.txt`
(override with the hidden `--cam-db`).

## Architecture

### Pipeline flow (`pgs_recon/apps/reconstruct.py`)

`main()` runs the stages in fixed order; each stage is a function that constructs a
binary invocation and calls `run_command`:

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

### Two dicts threaded through every stage

- **`paths`**: a `Dict[str, Path]` built up incrementally in `main()`. Stage
  functions take a `*_key` string argument naming the input path and **return a new
  key string** for their output, which they also insert into `paths`. This is how
  output of one stage feeds the next (e.g. `sfm_key`, `mvs_key`, `mesh_key`). When
  adding a stage, follow this convention: derive the output path, store it under a
  new key, return the key.
- **`metadata`**: a dict whose `metadata['commands'][timestamp]` records the exact
  command line of every binary invocation. An `atexit` hook writes it to
  `<output>/metadata.json`, and a full run config is written to
  `<output>/*_recon_config.txt`. Pass `metadata` into any new stage so the run stays
  reproducible.

### Module layout

- `pgs_recon/openmvg.py`, `pgs_recon/openmvs.py` — thin wrappers, one function per
  binary, all routing through `utility.run_command`.
- `pgs_recon/pgs_data.py` — import logic for EduceLab "PGS Scan" directories,
  including grid-scan neighbor lookup that generates an OpenMVG **view pairs file**
  (limits matching to spatial neighbors via `--matching-pairs-radius`).
- `pgs_recon/utility.py` — `run_command` (subprocess wrapper that `sys.exit`s on
  failure) and timestamp helper.
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

- Stage functions are pure command builders: assemble a `command` list, append
  optional flags conditionally, record to `metadata`, then `run_command`. Match this
  style for new binary wrappers rather than calling `subprocess` directly.
- Args use `configargparse`; hidden/internal flags use `configargparse.SUPPRESS`.
- The package version lives in `setup.cfg` (`version = ...`).
