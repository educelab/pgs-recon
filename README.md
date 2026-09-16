# PGS Recon
A Python-based pipeline for reconstructing photogrammetry datasets using 
[OpenMVG](https://github.com/openMVG/openMVG) and 
[OpenMVS](https://github.com/cdcseacave/openMVS).

## Usage
The simplest way to get started is to pull the Docker image of this project and 
run `pgs-recon` on your directory of images:

```shell
# Download the image
docker pull ghcr.io/educelab/pgs-recon:latest

# Run reconstruction on a directory of images in the current working directory
# Flags:
#  -v .:/working       - Mounts the current working directory to '/working' 
#                        inside the container
#  -i /working/images  - Path to the images inside the container
#  -o /working/recon   - Output directory inside the container
#  --name my-object    - Descriptive name for the scanned object/scene. This is 
#                        used to name the output file. If not provided, defaults
#                        to a name derived from the current timestamp and the 
#                        name of the input directory
docker run -v .:/working ghcr.io/educelab/pgs-recon:latest \
  pgs-recon -i /working/images/ -o /working/recon/ --name my-object
```

Upon successful completion of the pipeline, your reconstructed model can be 
found in `recon/mvs/my-object.obj`.

### What lands in the output directory
Every intermediate is named `<stage>_<role>`, after the stage that produced it,
so a half-finished directory can be read for what has happened so far. Optional
stages are marked; the rest are always present:

```
recon/
  pgs-recon.json                  # the manifest: what ran, and with what arguments
  my-object_recon_config.txt      # the effective arguments, loadable with -c
  mvg/
    sfm_data.json                 # the imported scene
    matches_dir/                  # per-image features, matches[_filtered].bin
    recon_dir/
      sfm_data.bin                # the solve
      robust_sfm.bin              # --mvg-robust
      autoscale_sfm.bin           # --mvg-autoscale
      landmarks[_scaled].ply      # --mvg-autoscale: the markers it scaled from
      colorize_sfm.ply            # sparse cloud coloured from the images
  mvs/
    convert_scene.mvs             # the interface scene every MVS stage reads
    undistorted_images/
    densify.mvs  densify.ply      # --mvs-densify: scene + the dense cloud
    reconstruct_mesh.ply
    coarsen_mesh.ply              # --mvs-coarsen (on by default): refine's input
    refine_mesh.ply               # --mvs-refine (on by default)
    my-object.obj                 # the deliverable, + .mtl and texture image
```

**Locate an artifact through the manifest, not by rebuilding its name.** Every
stage records the paths it consumed and produced, relative to the output
directory, and those records are what a resumed job reads — which is what lets
these names change without invalidating a directory that already exists:

```shell
jq -r '.stages.texture.outputs.mesh' recon/pgs-recon.json   # the textured mesh
jq -r '.stages.convert.inputs.sfm'   recon/pgs-recon.json   # the solved SfM it came from
```

Upgrading from 1.7, where the manifest was `metadata.json` and intermediates were
named by chaining (`scene_dense_refine.ply`)? Those directories are still read,
but a 1.7 manifest carries no per-stage record, so a run against one rebuilds it
from the start. See [docs/migrating-to-2.0.md](docs/migrating-to-2.0.md).

### Staged and resumable runs
The pipeline records what it has finished in `<output>/pgs-recon.json`, so
**re-running the same command in the same output directory resumes it** rather
than starting over. After a crash or an out-of-memory kill during mesh
refinement, this picks up at `refine`:

```shell
pgs-recon -i images/ -o recon/ --name my-object
```

`--from`/`--to` (both inclusive) restrict a run to a contiguous window of the
fifteen pipeline stages:

```
import  features  matches  filter  sfm  robust  autoscale  colorize
convert  densify  reconstruct  coarsen  refine  decimate  texture
```

This lets one reconstruction be split across several cluster jobs, each sized
for the stages it runs — useful because `RefineMesh` needs far more memory than
the rest of the pipeline, and sizing a whole-pipeline job for its worst case
wastes a large allocation on hours of cheap SfM:

```shell
J1=$(sbatch --mem=32G  --parsable job1.sh)   # pgs-recon -i $IMGS -o $OUT -n obj --to reconstruct
J2=$(sbatch --mem=256G --parsable --dependency=afterok:$J1 job2.sh)  # pgs-recon -o $OUT --from coarsen --to refine
        sbatch --mem=64G           --dependency=afterok:$J2 job3.sh  # pgs-recon -o $OUT --from texture
```

The later jobs need neither `-i` nor `--name`: every argument of the first run is
recorded in the manifest and reloaded, so only what changes has to be repeated.
`apptainer/submit_recon_pipeline.sh` is a worked example of this: it submits the
OpenMVG stages to a CPU node, densification to a GPU node, and
mesh/refine/texture to a high-memory node, chained with `afterok`. Notes:

* `--dry-run` resolves and prints the whole plan — loaded arguments, pipeline
  shape, which stages will run or be skipped, rehydrated input paths, and the
  prerequisite check — then exits without launching a binary. With no range it
  doubles as a status query for an output directory.
* Stages already recorded complete are skipped. A stage re-runs if its own
  arguments changed, if a stage producing one of its inputs re-runs, if one of
  its inputs now comes from somewhere else, or if `--rerun` is given. So
  retrying just the expensive step is
  `pgs-recon -o recon/ --from refine --refine-resolution-level 2`, which
  re-refines and re-textures but touches nothing before it.
* Changing the shape is allowed at any point. Adding `--mvs-densify` to a
  finished reconstruction re-runs densify and the mesh stages, and dropping it
  again re-runs them against the sparse cloud. Filenames stay put either way:
  an artifact is named for the stage that wrote it, not for the stages upstream
  of it.
* Stages before `--from` are never run implicitly: if one is incomplete or its
  inputs have moved, the run fails immediately, naming each, instead of quietly
  doing work the job was not sized for.
* If the range stops before stages the run invalidates, those stages are named
  in a warning and rebuilt by the next run that covers them. The final textured
  mesh keeps its usual `mvs/<name>.obj` filename in the meantime, so check the
  warning rather than the filename.
* What is on disk is never consulted — `<output>/pgs-recon.json` is the record.
  If you delete an intermediate by hand, use `--rerun` to rebuild it.
* An argument aimed at a stage outside the range is ignored with a warning,
  because it would change what the stages in range consume, and this run is not
  sized to rebuild them. Per-invocation settings are exempt and can differ freely between
  jobs: `--path` and `--cam-db` apply silently, and `--threads`, `--log-level`,
  `--config` and `--output` are not recorded at all, so they never leak into a
  later job.
* `--output` must be on a filesystem every job can see. `pgs-recon` does no
  copying of its own; stage node-local scratch in and out around it.
* `--no-mvs` is deprecated: use `--to colorize` for an SfM-only run. The old flag
  still works (it sets `--to colorize` and warns) but will be removed.

### Preparing the mesh for refinement
`refine` is the pipeline's slowest and hungriest stage, and it used to run out of
two different resources. Out of **memory** is the familiar one, and resuming the
same command picks up where the kill happened.

Out of **wall clock** looked different: no progress in the log, one core pinned
at 100%, and memory flat. That was mesh *preparation* rather than the
optimization. Before refining anything, `RefineMesh` decimates its input by CGAL
Garland-Heckbert edge collapse, single-threaded and silent — and at an identical
decimation target its cost varies **135x** between meshes, having killed
seventeen refine jobs on the wall clock. Adding cores or memory does not help,
and input size does not predict it: the largest mesh in the measured corpus
finished in a fifth of the time the worst one took, at 79% of its size.

**The `coarsen` stage does that reduction instead**, driving our `pgs-decimate`
to the same face target with a 1.7x spread, and tells `RefineMesh` to skip both
of its own preparation passes. It is on by default whenever `refine` is, and it
is where the wall clock went: on one measured fragment, 37m07s of preparation
became 7m40s. See [ADR 0009](docs/adr/0009-coarsen-before-refine.md) for the
measurements.

The target is `--coarsen-ratio` of the input mesh's face count, defaulting to
`0.375` — which is the target `RefineMesh` computed for itself on every one of
119 measured runs, being `6 x` the median projected face area (1.0 px², densify
reconstructing at one face per pixel) over `--refine-max-face-area`'s default of
16. It is a property of the rig and of `--densify-resolution-level`, not of the
fragment, which is why it never varied. Change it if either of those changed:

```shell
# Coarsen harder, at the cost of a coarser refine input
pgs-recon -o recon/ --from coarsen --coarsen-ratio 0.2

# Or state the count outright, to match a previous run's refine input exactly
pgs-recon -o recon/ --from coarsen --coarsen-max-faces 9761507

# Or hand the reduction back to RefineMesh
pgs-recon -o recon/ --from reconstruct --no-mvs-coarsen
```

Refinement never sees the coarsened mesh as such: `RefineMesh`'s `EnsureEdgeSize`
pass immediately re-reduces it by a further 3.6–4.1x, and *that* is what
iteration zero starts from. So the stage states a face budget rather than a
deviation budget — a geometric guarantee about a surface refinement is about to
move guarantees nothing about what comes out. The guarantee belongs on the
deliverable, which is the `decimate` stage below.

The stage's record carries the face counts, because `RefineMesh`'s
`Decimated faces N (100%, …)` line was the only place that 0.375 ratio was ever
observable and disabling the pass removes it:

```shell
jq '.stages.coarsen | {input_faces, target_faces, achieved_faces, achieved_ratio}' \
   recon/pgs-recon.json
```

`--refine-decimate` and `--refine-ensure-edge-size` are `RefineMesh`'s own
preparation flags and stay available as explicit overrides. Left unset with
`coarsen` in the pipeline they are `1` and `2` — *both*, and not by coincidence:
the binary guards the edge-size pass on its own decimation
(`SceneRefine.cpp:556`), so `--decimate 1` alone skips that pass too and refine
subdivides the mesh as given, measured at 3.9x the intended refine input with
nothing in its log to say so. Both flags follow the stage rather than the
caller for exactly that reason, and overriding one warns rather than silently
changing what the other means.

`2` means *force on the input mesh* in these images. Upstream it forces the pass
at every image scale of the refinement loop, the later ones isotropically
remeshing the mesh subdivision has just grown — which only `--decimate 1` ever
reaches, and which no stock OpenMVS run performs. A patched `RefineMesh` scopes
it to the first scale, where `0` and `1` already confined it (ADR 0010). The fix
lives in the C++ toolchain, so it arrives with a rebuilt image rather than with a
`pgs-recon` upgrade.

`--refine-decimate` is a face *fraction* and has nothing to do with the
`decimate` stage below, which states a geometric bound. It was called
`--decimation-factor` before 2.0. `--refine-max-face-area` is *not* a remedy for
a slow refine: it is the denominator auto-decimation divides by, so raising it
decimates harder, and it bounds subdivision rather than this.

If refine is not worth its cost on a given dataset, `--no-mvs-refine` drops it
from the pipeline shape — and `coarsen` with it, that stage only ever preparing a
mesh for refinement — and textures the reconstructed mesh directly.

### Decimating the deliverable
A finished mesh is millions of faces, which is more than the geometry justifies
and more than MeshLab opens comfortably. The `decimate` stage reduces it as far
as a **deviation budget** allows — the largest distance any point of either
surface may end up from the other — and *measures* what it achieved rather than
predicting it, so the guarantee is a number you can check rather than a flag you
trust. Sharp edges and ridges survive, which is the requirement for inscribed
surfaces:

```shell
# No point may move more than 0.2 mm. The units are the solved scene's, so this
# is millimetres only because --mvg-autoscale put it in them.
pgs-recon -i images/ -o recon/ -n obj --mvg-autoscale 12.7 \
          --decimate-max-error 0.2

# A scan that was never scaled has no physical units, so budget faces instead
pgs-recon -o recon/ --decimate-max-faces 2000000

# Both: as coarse as 2M faces, but never worse than 0.5 units. They pull in
# opposite directions, and --decimate-prefer decides; the default protects the
# deviation budget.
pgs-recon -o recon/ --decimate-max-error 0.5 --decimate-max-faces 2000000
```

Notes:

* **Stating a budget is what enables the stage** — there is no `--mvs-decimate`.
  A budget of `0` is how you turn it back *off* on a resumed run: every argument
  is inherited from the manifest, so merely dropping the flag keeps the recorded
  budget. A `0` turns the *stage* off, so it also drops the other budget if that
  one was inherited — pass both if you meant to change targets rather than stop
  decimating (`--decimate-max-error 0 --decimate-max-faces 2000000` keeps the
  face budget). Turning it off re-runs `texture` against the un-decimated mesh
  and nothing earlier. A negative budget is refused: `0` is the off switch, and
  a negative distance or face count has no other reading.
* The stage sits between `refine` and `texture`, so the deliverable is textured
  at its final resolution rather than textured twice.
* **The search costs real time**, because every round decimates *and* measures,
  and the measurement is ten samples per face of the candidate — so a probe that
  barely coarsens anything is the expensive one. Two flags price that, and
  neither can cost you the deviation bound:
  `--decimate-min-gain` (default `0.1`) stops the search once it can still
  remove less than that share of the current result's faces, and
  `--decimate-quadric-seed` starts it from a threshold you supply instead of one
  derived from the budget. Neither flag turns the stage on.
* **Where a seed comes from: the same object's last run.** The derived seed is
  very nearly a constant, and it overshoots every real run by between 370x and
  95,000x. At one fixed budget the right threshold varies about 255x between
  objects but only about 4-9x between reruns of one object, so the object's own
  history is the only predictor on offer today. Take `search.quadric_error` out
  of that object's previous `decimate_report.json` and pass it as
  `--decimate-quadric-seed`; it seeds the search rather than replacing it, so
  the remaining few-fold is what the search is still there to close. On one real
  mesh, feeding back its own converged threshold took the search from 7 rounds
  and 55 s to 3 rounds and 24 s, for a marginally coarser result. Do not carry a
  seed between different objects.
* `mvs/decimate_report.json` records faces in and out, the measured max, mean
  and RMS deviation in both directions, what was cleaned, whether the mesh is
  non-manifold, every round the search tried, and both which bound stopped the
  search and why it stopped probing. The manifest records its
  path under the `deviation` role.
* A world-unit budget on a run with no `autoscale` stage is a number with no
  physical meaning, and the run warns about it.
* `pgs-decimate` is the same tool standalone, and works on any mesh, including
  the deliverable of a run that finished months ago. A textured mesh loses its
  UVs, with a warning — decimate first, then re-texture. `pgs-decimate --help`
  lists the geometry, search and measurement options the stage leaves at their
  defaults, and `--self-test` checks the measurement against a generated mesh
  whose exact surface is known.

### Docker images
We provide multi-architecture (x86, arm64) Docker images in the 
[GitHub Container Registry](https://github.com/educelab/pgs-recon/pkgs/container/pgs-recon).
Simply pull our container and Docker will select the appropriate image for your
host platform:
```shell
# Pull the latest release
docker pull ghcr.io/educelab/pgs-recon:latest

# Pull the latest edge version
docker pull ghcr.io/educelab/pgs-recon:edge

# Pull a specific version
docker pull ghcr.io/educelab/pgs-recon:2.0.0
```

CUDA-enabled images are available by appending `-cudaX.X` to any of the standard
tags. We currently only provide images for CUDA 12.4 and 12.8:
```shell
# Pull the latest CUDA 12.4 release
docker pull ghcr.io/educelab/pgs-recon:latest-cuda12.4

# Pull the latest CUDA 12.8 release
docker pull ghcr.io/educelab/pgs-recon:latest-cuda12.8
```

All project tools can be launched directly using `docker run`:
```shell
$ docker run ghcr.io/educelab/pgs-recon pgs-recon --help
usage: pgs-recon [-h] [--config CONFIG] [--input INPUT] --output OUTPUT
                 [--name NAME] [--file-type {ply,obj}] [--focal-length n]
                 [--new-importer | --no-new-importer]
                 [--import-pgs-scan | --no-import-pgs-scan | -p]
                 [--import-calib IMPORT_CALIB] [--import-capture n]
...
```

## Utilities
In addition to the main `pgs-recon` pipeline, this project ships several
standalone tools. All of them can be launched through the Docker image in the
same way as `pgs-recon` (e.g. `docker run ... pgs-sfm-orient --help`).

### `pgs-sfm-orient`
Centers, orients, and (optionally) scales a reconstructed mesh using the
EduceLab sample square / ArUco markers detected directly in the **SfM scene
images**. It is the SfM-based counterpart to `pgs-center`: where `pgs-center`
detects the sample square in the mesh's UV texture (which requires a coherent,
reordered texture map), `pgs-sfm-orient` detects and triangulates the markers
from the original images, so it works regardless of how the mesh was textured.

The translation comes from the mesh's oriented-bounding-box center (so the
object lands at the origin), the orientation and scale come from the markers,
and the result is written as a transformed mesh and/or a 4×4 similarity
transform. The input mesh must already be in the SfM coordinate frame.

```shell
docker run -v .:/working ghcr.io/educelab/pgs-recon \
  pgs-sfm-orient \
    -i /working/recon/mvg/recon_dir/sfm_data.bin \
    --input-mesh /working/recon/mvs/my-object.obj \
    -o /working/recon/mvs/my-object-centered.obj \
    --save-transform /working/recon/orient.npy \
    -s 0.47
```

Key options:
* `-i, --input-scene` — the SfM scene file (markers are detected in its images).
* `--input-mesh` — mesh (`.obj`/`.ply`) in the SfM frame; enables OBB-center
  translation and the bounding-box orientation fallback.
* `-o, --output-mesh` — write the transformed mesh (requires `--input-mesh`).
* `--save-transform` — write the 4×4 transform as a NumPy `.npy`, compatible
  with `pgs-center --load-transform` and `pgs-localize`/`pgs-retexture
  --sfm-transform`.
* `-s, --marker-size` — marker size in the desired world units (required unless
  `--no-scale` or `--orient-method bbox`).
* `--orient-method {auto,aruco,bbox}` — orientation source (default `auto`: use
  markers if detected, otherwise fall back to the mesh bounding box). `aruco`
  fails if no markers are found; `bbox` ignores markers and requires a mesh.
* `--no-scale` — skip scale estimation (output rotation + translation only).

At least one of `--output-mesh` or `--save-transform` is required.

### `pgs-localize`
Puts a camera that was never part of the reconstruction into the solved scene
and emits a reusable pose + intrinsic, so `pgs-retexture --calibration` can
texture the mesh from it. It replaces the former `pgs-calibrate`; see
[ADR 0011](docs/adr/0011-localize-by-rendering-the-mesh.md).

There are two ways to find the 3D↔2D correspondences a resection needs, and the
tool carries both:

| backend | how | needs |
| --- | --- | --- |
| **render-and-match** | render a textured mesh **in the query's own modality** from a prior pose, match render against query, and lift the render-side keypoints through the render's own position map | `--mesh` plus a prior pose |
| **sparse** | match the query's descriptors against the scene structure's, the way `openMVG_main_SfM_Localization` does | `--input-scene` plus `--matches-dir` (the ~2.2 GB of `.feat`/`.desc` regions) |

Give all four and they **chain**, which is the recommended mode: the sparse
backend resects, its pose becomes the prior, and render-and-match refines from
there. Measured over twelve datasets, render-and-match lands at 1.81–2.53 px
where the sparse path lands at 3.96–23.13 px — but the sparse path is a fine
*prior*, and render-and-match converges from one 861 px off.

```shell
docker run --rm -v $(pwd):/working ghcr.io/educelab/pgs-recon:edge \
  pgs-localize \
    -i /working/spectral/object+MB940IR_015_FN.tif \
    -m /working/recon/object_IR940.obj \
    -c /working/overhead-camera.txt \
    --output-calibration /working/object_calibration.json \
    --output-camera /working/object_solved.txt \
    --report /working/object_localize.json \
    --qa-render /working/object_qa.png
```

Key options:
* `-i, --image` — the query image to localize.
* `-m, --mesh` — a textured mesh **in the same modality as the query**. The tool
  knows nothing about modality and assumes the caller paired them.
* `-c, --camera` — a camera file (below) supplying K, always, and the prior pose
  for the mesh backend.
* `-s, --input-scene` / `--matches-dir` — the sparse backend's scene and regions.
* `--sfm-transform` — a 4×4 `.npy` (from `pgs-sfm-orient --save-transform` or
  `pgs-center`), applied to the scene **at load** so the resection happens in the
  mesh's frame from the start. Pass it when localizing against a mesh that was
  centered after reconstruction; omit it when the scene and the mesh share a
  frame. Unchecked either way — whether a centered mesh exists is not something
  the tool can see.
* `--output-calibration` / `--output-camera` — at least one is required. The
  first is the one-view openMVG scene `pgs-retexture --calibration` consumes; the
  second is the flat camera file below, which is a valid input to `--camera`.
* `--report` — a JSON QA sidecar: match counts, inlier count and spread,
  held-out reprojection statistics, the solved standoff, and a pass/review/fail
  grade per gate.
* `--qa-render` — re-render at the **solved** pose and write it plus a difference
  image against the query. Worth doing: a frame error is a proper rigid motion,
  so it moves no residual at all, and looking is the only thing that catches it.
* `--expected-standoff` / `--standoff-tolerance` — gate the solved
  camera-to-scene distance. Optional, and the only gate that does not depend on
  the correspondence set.
* `--mask` — restrict query-side feature detection to a mask's non-zero region
  (generate one with `pgs-generate-mask`).
* `--self-test` — render a generated scene from a known pose and check the
  conventions that are silent when wrong.

`pgs-localize --help` lists the rest; every tunable is a flag, and the ones with
a derivation carry it in the help string.

### Camera calibration file format
`pgs-localize` reads and writes camera parameters in a single plain-text
**camera calibration file**. It is a flat list of `key value` entries, one per
line; blank lines and lines beginning with `#` are ignored, and unrecognized
keys are skipped (so the same file can carry both an intrinsic and a pose, and
each consumer reads only what it needs).

| Key | Meaning |
| --- | --- |
| `fx`, `fy` | Focal length in **pixels** (x and y). `fy` defaults to `fx` if omitted. OpenMVG carries a single focal, so the two must match. |
| `cx`, `cy` | Principal point in **pixels**. |
| `width`, `height` | Image resolution (pixels) the intrinsic is calibrated at. |
| `k1`, `k2`, `k3` | Radial distortion coefficients (OpenCV/OpenMVG order). Optional; absent means no distortion. |
| `pose` | 16 whitespace-separated floats: a **row-major 4×4 world-to-camera** matrix in OpenCV convention (`x_cam = R·X + t`). |

`pose` holds the world-to-camera **translation**, not the camera centre. The two
differ by a sign and a rotation, `C = −Rᵀt`, so a file whose `t` reads
`(2.67, −2.75, 154.16)` describes a camera at `C = (−2.67, −2.75, 154.16)`.
Misreading one for the other is a proper rigid motion that no residual can
reveal, which is why `pgs-localize` logs the centre it derived on load.

Example (an overhead camera with mild barrel distortion):

```
# my overhead RGB camera
fx 18250.0
fy 18250.0
cx 3000.0
cy 2000.0
width 6000
height 4000
k1 -0.082
k2 0.011
k3 0.0
pose 0.9998 0.0011 -0.0203 12.4 -0.0009 0.9999 0.0102 -8.1 0.0203 -0.0102 0.9997 423.7 0 0 0 1
```

Two flags use this format:

* **`pgs-localize --camera <file>`** reads it as a *precalibrated* query camera.
  It requires the intrinsic keys (`fx`, `cx`, `cy`, `width`, `height`); `fy` and
  the `k*` distortion are optional, and the `pose`, when present, becomes the
  prior the mesh backend renders from. The intrinsic is scaled to the query
  image's resolution automatically, and the distortion is honored — openMVG
  undistorts the query before resectioning. A camera supplied this way is never
  re-fitted; without one, the focal is recovered by DLT.
* **`pgs-localize --output-camera <file>`** writes the solved camera in this
  format. The tool's output is a valid input to its own next run, which is what
  makes a constant-prior scheme cheap to maintain. It is also what
  [registration-toolkit](https://github.com/educelab/registration-toolkit) reads
  as `rt_reorder_texture --camera-file`.

## Install from source
### Install dependencies
The Python scripts use executables provided by the OpenMVG and OpenMVS projects. 
The included CMake project will compile both of these projects and their 
dependencies. Before configuring the CMake project, please preinstall the 
following dependencies:
* CMake 3.17+
* Boost 1.70+
* GMP and MPFR
* ExifTool
* (Optional) NASM (Required by jpeg-turbo)
* (Optional) Ceres Solver
* (Optional) CUDA Toolkit

After the dependencies have been installed, configure and build the CMake 
project to compile the required executables:
```shell
cmake -S dependencies -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build/
```

**Notes**:
* OpenMVS and OpenCV must be linked against the same version of libjpeg.

### Install the Python pipeline scripts
Use a recent version of `pip` to install the Python scripts:
```shell
# Requires Python 3.9+
python3 -m pip install .
```

After installation, the reconstruction script can be run from the shell:
```shell
pgs-recon --help
```

#### Telling the tools where the binaries are
The Python pipeline shells out to the compiled OpenMVG/OpenMVS/`pgs-*` binaries,
which it looks for under an install prefix containing `bin/` (OpenMVG and our own
tools) and `bin/OpenMVS/`. Inside our Docker/Apptainer images that prefix is
`/usr/local/`, which is the default, so nothing needs setting. Elsewhere — most
often a CMake build left in its default `dependencies/installed/` — point
`$PGS_RECON_PREFIX` at it:

```shell
export PGS_RECON_PREFIX="$PWD/dependencies/installed"
pgs-recon -i images/ -o recon/ --name my-object
```

Every Python entry point (`pgs-recon`, `pgs-retexture`, ...) reads it, and each
also takes a `--path <prefix>` argument that wins over the environment. The
`pgs-*` C++ tools — `pgs-localize`, `pgs-decimate`, `pgs-sfm-orient`,
`pgs-global-scaler`, `pgs-generate-markers` — shell out to nothing, so they need
no prefix; run them from `<prefix>/bin/` directly. A
missing binary is reported with the path that was searched and which of the three
tiers chose the prefix — the argument, the environment, or the built-in default —
so a typo in any of them is unambiguous. The OpenMVG camera sensor database is
expected at `<prefix>/lib/openMVG/sensor_width_camera_database.txt`.

Unlike most arguments, `--path` is deliberately *not* inherited from a previous
run's manifest when a staged run resumes (see `--from`/`--to` above), so each job
of a split reconstruction picks up the prefix of the node it lands on.

### Advanced Installation
#### Installation Location
By default, executables created by this CMake project will be installed to 
`dependencies/installed/`. The installation location can be changed by setting 
the CMake installation prefix flag:
```shell
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ ..
```

#### Disable compilation of extra libraries
In addition to VCG, OpenMVG, and OpenMVS, the CMake project also compiles a 
number of required software libraries. We provide corresponding CMake flags to 
control the compilation of these libraries. To use a system-provided version of 
these libraries, set the library's flag to `OFF`:

```cmake
BUILD_EIGEN: If ON, builds Eigen 3.2
BUILD_JPEG: If ON, builds libjpeg
BUILD_JPEG_TURBO: If ON, builds libjpeg-turbo (depends BUILD_JPEG=ON)
BUILD_OPENCV: If ON, builds OpenCV
BUILD_CGAL: If ON, builds CGAL
```

## Building a Docker image
Docker images can be built by running the following from the root of the 
project directory:
```shell
docker build -t pgs-recon:dev .
```

By default, this image only supports a CPU-based reconstruction pipeline. 
A CUDA-enabled image can be built by passing the `BASE_IMAGE` and `USE_CUDA`
build args:
```shell
docker build -t pgs-recon:dev \
  --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  --build-arg USE_CUDA=ON \
  -t pgs-recon:dev-cuda \
  .
```

`BASE_IMAGE` should be an `nvidia/cuda:*-devel-ubuntu*` Docker image 
[[link]](https://hub.docker.com/r/nvidia/cuda/tags?name=devel-ubuntu). 
While this can theoretically be set to any Ubuntu and CUDA version, this has 
only been tested on:
 - CUDA 12.4, Ubuntu 22.04 (with and without CUDNN)
 - CUDA 12.8, Ubuntu 22.04 (with and without CUDNN)

`USE_CUDA` should be either `ON` or `OFF [default]`. If `USE_CUDA=OFF`, CUDA 
will not be used even if you provide a CUDA-enabled base image.

## Building an Apptainer image
Apptainer images can be built by running the following from the root of the 
project directory:
```shell
apptainer build pgs-recon.sif apptainer/pgs-recon.def
```

By default, this image only supports a CPU-based reconstruction pipeline. 
A CUDA-enabled image can be built by passing the provided build args file for
your required CUDA version:
```shell
apptainer build pgs-recon.sif \
  --build-arg-file apptainer/buildargs-cuda12.4.env \
  apptainer/pgs-recon.def
```