# What the `byu-registration` branch adds

*A walkthrough for someone new to `pgs-recon`. It covers the 14 commits on this
branch since it split from `main`.*

## The short version

This branch adds an optional tool, **`pgs-localize-set`**. It takes a folder of
hand-held infrared photos of an object (in practice, a papyrus fragment mounted
on a plate) and works out exactly where the camera was for each one, relative to
a 3D model the main pipeline already built. Once that is known, each photo can
be "painted" onto the 3D model as its texture.

Size: about 6,700 lines added across 32 files. Roughly 1,900 of those are new
C++, 2,200 are new Python, 1,400 are tests, and 650 are documentation.

---

## 1. Background you need first

### What `pgs-recon` already did

`pgs-recon` turns a set of photos taken by a fixed camera rig into a 3D model
(this is called *photogrammetry*). The Python code mostly calls existing C++
programs ([OpenMVG](https://github.com/openMVG/openMVG) and
[OpenMVS](https://github.com/cdcseacave/openMVS)) in the right order. The output
is a **mesh**: a 3D surface made of triangles, with an image (the **texture**)
wrapped over it.

### The problem this branch solves

The researchers also had **extra photos** of the same plate, taken later by
hand, with a different camera, at about 980 nm (infrared). The rig never saw
these photos, so nothing knew where they belonged on the model. The goal was to
texture the mesh with them.

Doing that means **localizing** each photo, which means solving for its
**camera calibration**:

| term | meaning |
| --- | --- |
| **pose** | where the camera was in 3D and which way it pointed |
| **intrinsics (K)** | the camera's internal properties: focal length, image center, and **lens distortion** (the bending that makes straight lines curve near a photo's edges) |
| **reprojection error** | how many pixels a 3D point lands from where it should in the photo. Under ~1 px means the photo and mesh line up to within a pixel |

The repo already had `pgs-localize`, which does this for **one** photo when you
give it a rough starting pose. Hand-held photos are harder:

- **No starting pose.** Nobody recorded where the camera was held.
- **Unknown camera.** The lens and its distortion weren't calibrated. All the
  photos share *one* camera, though, so its K can be solved once for the whole
  set.
- **Flat object, shot nearly straight down.** In this geometry a wrong focal
  length looks almost identical to a camera that is closer or farther away, so
  the focal length is hard to pin down (see §4).
- **Different wavelength.** The model's normal texture is in color, and ink on
  papyrus looks very different in color than in infrared. Matching infrared
  photos to a color texture fails.
- **Uneven overlap.** Some photos overlap a lot and some not at all, so the
  method can't depend on photos matching each other.

### How the design was recorded

The design is in an *Architecture Decision Record* (ADR), a document that says
what was decided and why:
[`docs/adr/0013-localize-a-handheld-view-set.md`](adr/0013-localize-a-handheld-view-set.md).
It was written **before** any code (commit `993a3f1`) and updated with real
measurements as the work went on. Reading its "Context" table is the fastest way
to see why each piece exists.

---

## 2. How the new tool works

You run one command, `pgs-localize-set`. It works through these phases in order:

1. **Choose a reference.** This is the 3D model every photo is matched against.
   It is the rig's **IR940** mesh: the same geometry, textured from the rig's own
   940 nm infrared capture, which looks much like the 980 nm hand-held photos.
2. **Get an overhead.** This is a picture of the whole plate from above where
   every pixel's 3D position is known. It is either the rig's real overhead photo
   or a **virtual overhead**, a render of the reference mesh from above (§3.6).
3. **Localize each photo on its own.** Each photo is matched against the
   overhead to get a rough pose. The mesh is then rendered from that rough pose
   and the photo is matched against the render to refine the pose
   (*render-and-match*). Each photo's result depends only on the mesh, so one bad
   photo can't spoil the others.
4. **Find tie points.** Features visible in two or more photos are matched and
   placed in 3D. Photos that share no features with the rest become separate
   **groups**.
5. **Joint adjustment.** One shared K and every photo's pose are solved
   together, while the mesh anchors stay fixed.
6. **Quality check (QA).** The tool measures how well the photos line up with
   each other and with the overhead.
7. **Texture.** It writes one textured mesh per photo and, optionally, one mesh
   textured from all of them (§3.7).

Every phase can **resume**: it records fingerprints of its inputs, and if you
rerun with the same inputs it prints "current, skipping" rather than redoing
work.

### Where the code lives

| piece | language | role |
| --- | --- | --- |
| `pgs-localize-joint` ([`localize_joint.cpp`](../dependencies/utilities/src/localize_joint.cpp)) | C++ | the math: rendering, matching, the joint solve, QA. Built only with `-DPGS_BUILD_VIEWSET=ON` (off by default) |
| [`pgs_recon/viewset/`](../pgs_recon/viewset/) | Python | the orchestration: decides what runs, in what order, with which files. Its entry point is `pgs-localize-set` |
| `pgs-localize-field` ([`viewset/field.py`](../pgs_recon/viewset/field.py)) | Python | a diagnostic that shows how far off one photo is, tile by tile |

The Python package is a **one-way add-on**. It imports from the core pipeline,
but the core never imports it, so it can be left out of an install without
anything else noticing.

---

## 3. The commits, in order

### 3.1 The design: `993a3f1`, Propose ADR 0013
This commit wrote down the plan before any code existed. It also fixed a
pass/fail test for a first experiment in advance, so the result couldn't be
interpreted after the fact to fit.

### 3.2 The skeleton: `7709008`, Add pgs-localize-joint and pgs-localize-set
- **Shared C++ library.** The code behind the existing `pgs-localize` moved into
  a library (`localize_core`) that both the old and the new tool link against.
  The old tool's output was checked to be byte-for-byte identical afterwards, so
  nothing broke.
- **Describe once, match many times.** One function was split in two so the
  reference image's features are computed once and reused for every photo.
- **Texture-validity mask.** Parts of the mesh that no rig camera saw have no
  real color. The renderer now marks those pixels so no features are taken from
  them.
- **New C++ commands.** `position-map` renders the 3D position of every overhead
  pixel. `view` localizes one hand-held photo without a starting pose.
- **Python package.** The structure for file naming, input checks, grouping,
  resume, and the pipeline. Two parts were stubs, filled in by later commits.
- **Tests** against a synthetic scene (a generated textured plane) where the
  right answer is known. A hand-held view localized to within 0.011 units.

### 3.3 The joint solve: `61df30d`, Implement pgs-localize-joint adjust
- Solves **one** K (focal length, center, three distortion terms) plus every
  photo's pose, using the [Ceres](http://ceres-solver.org/) optimization library.
  It is written from scratch instead of reusing OpenMVG's solver because the mesh
  anchors must stay fixed while the tie points move.
- **The key safeguard: a focal-length check.** Every run tries a range of focal
  lengths and looks at the cost of each one. If the curve is **flat**, the photos
  don't determine the focal length, so the solver doesn't trust it. Without this
  check, a synthetic test drifted to a focal length of 11,652 when the true value
  was 1,600, and every camera ended up 377 units off, *while the error numbers
  improved*. No reprojection check would have caught it.
- Photos that get *worse* under the joint solve keep their individual result.
- On synthetic data with a starting K 5% off, the solve recovered the focal
  length to 0.03% and every camera position to about 0.013 units.

### 3.4 Finishing the pipeline: `9fbb675`, Group SfM, resume for every phase, end-to-end test
- Each group of overlapping photos now gets its own structure-from-motion solve,
  which produces the tie points. A group that fails to solve just contributes no
  tie points and the run continues.
- `--camera` lets you give a starting K, which is required for photos with no
  EXIF focal-length metadata (common for multispectral TIFFs).
- The joint adjustment and texturing phases now resume too. Before this, a
  second run silently redid both.
- An **end-to-end test** runs the whole CLI twice on synthetic data and checks
  that the second run redoes nothing.

### 3.5 Measuring quality: `655b470`, Add QA
QA warps one photo into another photo's view, through the mesh, and measures the
remaining shift in small tiles. It runs three comparisons, labeled by how much
they can be trusted:

| comparison | independent? | graded? |
| --- | --- | --- |
| hand-held photo vs hand-held photo | yes | yes |
| overhead vs each photo | partly | yes |
| each photo vs the reference render | no: the photos were fitted to this render, so the check is circular | reported only |

One finding: grading on the average shift *size* over-reports error, because
random noise alone gives a nonzero size. Grading on the *coherent* shift (the
length of the median shift vector) fixed that. Calibrations known to be within
~0.5 px went from grading 0.7–1.9 px to 0.09–0.49 px.

### 3.6 Fixes from real data: `3e43cf4`, Fix tie points and K on near-nadir plates
This is the first run on real data: fragment PHerc0128 Cr04, with photos tilted
only 1–3° from straight down ("near-nadir").
- **Tie points were zero.** OpenMVG's standard approach couldn't get started on
  a flat plate seen from above. The fix was to place tie points using each
  photo's mesh-derived pose. Result: **8,737 tie points instead of 0.**
- **The flatness threshold was too strict.** The focal curve now counts as flat
  below a 25% rise (it was 1%). Cr04's 3–5% rise had passed the old threshold and
  moved the image center by 161 px.
- **Distortion is still solved when the focal length can't be.** Distortion is
  something no pose change can absorb, so it is always worth solving.
- Result: photo pairs line up to 0.06 and 0.97 px, and the corner error dropped
  from +4.8…+9.5 px to −2.3…+2.3 px.

### 3.7 Removing manual steps: `3cee662`, Bring the helper scripts into the repo
Running Cr04 had needed one-off scripts. They became flags:
- `--virtual-overhead`: render the overhead from the mesh when the plate has no
  real registered overhead photo.
- `--vignette-mask`: build a mask of the lens's dark corners (vignette) from the
  first photo.
- `pgs-localize-field`: a diagnostic replacing two scripts.
- macOS's hidden `._*` files are no longer mistaken for photos.

### 3.8 One combined texture: `7cf2e67`, Add --combined-texture
- Merges every photo's calibration into one scene and textures the mesh from all
  of them at once, with OpenMVS's `TextureMesh`. The output is
  `out/combined/combined.obj`.
- The vignette mask is applied per photo, which removed dark arcs at the edges.
- If this phase fails, the run doesn't fail, because the per-photo meshes already
  exist.
- A shared helper (`ensure_ply_mesh`) moved from `pgs-retexture` into
  [`utils/mesh_staging.py`](../pgs_recon/utils/mesh_staging.py) so both tools can
  use it. This follows the repo rule that apps don't import each other.

### 3.9 Documentation and bug fixes: `f3eabcd`, `6cb4d62`, `1d7ca78`, `8036651`
- **README guide** (`f3eabcd`): a beginner-friendly usage section covering the
  terms, the phases, building, inputs, an example run, the outputs, and reading
  QA.
- **Crash on external drives** (`6cb4d62`): on an exFAT drive mounted in Docker
  Desktop, just checking a `._*` file raised a permission error. The fix skips
  hidden files by name *before* checking them.
- **Stale overhead render** (`1d7ca78`): switching to a different reference mesh
  reused the old render, so every photo was matched against the wrong image.
  Each render now records what it was made from and is redone when that changes.
- **Troubleshooting table** (`8036651`): common error messages, what causes
  them, and how to fix each one.

---

## 4. The last two fixes: `3f5f2ca` and `8af9bae`

1. **The reference mesh is now textured by default** (`8af9bae`). If you pass
   `--ir-reference-mesh`, that mesh is also the one that gets textured, unless
   you override it with `--mesh`. Before, the default was the reconstruction's
   own mesh, which only matched when the reference had been retextured from it.
   Changes are in [`viewset/cli.py`](../pgs_recon/viewset/cli.py); the README was
   updated to match.
2. **Reusing an output folder for a different photo set no longer keeps stale
   data** (`3f5f2ca`). The import step used to skip whenever its output file existed. It now
   checks fingerprints of the photos and `--camera`, and on a mismatch it clears
   the whole `sfm/` folder and re-imports. The matching step and each group's
   solve now also redo themselves when their inputs change. Changes are in
   [`viewset/pipeline.py`](../pgs_recon/viewset/pipeline.py).
3. **New tests** for both changes in
   [`tests/test_viewset.py`](../tests/test_viewset.py) (`TestImportSet`,
   `TestPlateMesh`).

`python3 -m unittest tests.test_viewset` passes all 51 tests inside
`ghcr.io/educelab/pgs-recon:edge`. On a bare Python the six new tests skip,
because they need `numpy` and `configargparse`.

---

## 5. Limits and open questions

- The ADR's status is still **"proposed"**. The approach has been measured on one
  real plate (Cr04), not on many.
- QA's pass/fail thresholds are **provisional**.
- On flat, nearly straight-down sets, the focal length usually **can't** be
  solved from the photos, so it stays at the starting value. A checkerboard
  calibration of the camera (passed with `--fix-intrinsic`) is the reliable fix.
- There is a known ~0.57 px offset between the test renderer and OpenMVS, which
  place pixel centers differently. It's documented and left for a separate fix.
- `pgs-localize-joint` is **not in the published Docker image**. It has to be
  built with `-DPGS_BUILD_VIEWSET=ON` (see the README).

## 6. Where to read more

- **To use it:** README, section "`pgs-localize-set`".
- **For why it's built this way:** ADR 0013.
- **For the single-photo method it builds on:** ADR 0011.
- **For the commit messages, which carry the measurements:**
  `git log main..byu-registration`.
