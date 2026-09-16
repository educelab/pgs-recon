# Localize a camera by rendering the mesh it will texture

**Status: accepted, validated on 12 datasets.** Replaces `pgs-calibrate`
([ADR 0002](./0002-calibrate-new-camera-for-retexture.md)) with `pgs-localize`,
a C++ tool carrying two correspondence backends. The numbers below are measured
over the twelve paired photogrammetry/spectral captures spanning
2022-12-09 … 2025-10-30 that the source investigation used, and against the
poses that investigation solved.

## Context

ADR 0002 put a new camera into a solved scene by matching its features against
the reconstruction's *sparse structure* — `openMVG_main_SfM_Localization`, which
is what `pgs-calibrate` drove. That is a fine idea and it works when it works.
Run over the whole corpus, scoring both cameras on the same independent
correspondences, it **failed on six of twelve**:

```
working (6)  3.96, 4.15, 4.46, 4.47, 5.33, 5.51 px
FAILED  (6)  10.47, 11.36, 14.00, 14.07, 20.06, 23.13 px    <- nothing in between
```

Every one of the six produced a pose, posted a clean RMSE on its own tracks, and
passed `validate_localized_intrinsic`. The worst was 3.47 mm at p99 with a 17 px
systematic bias. They span both capture campaigns, so nothing in the metadata
predicts them.

The mechanism is worth stating because it is what makes the failure quiet. At
near-fronto-parallel geometry a wrong focal is **unobservable in the residual**:

| | f | solved Z | ratio |
|---|---|---|---|
| production | 21000 | 148.32 cm | — |
| measured | 21700.2 | 153.21 cm | 21700/21000 = 1.0333 vs 153.21/148.32 = **1.0330** |

The focal error is absorbed into standoff to four significant figures. Anyone
tuning a focal by watching RMSE will tune it wrong. (`pipeline_registration.py`
passed `--focal-length 21000` against a true 21700.2 ± 0.24 px, stable to 0.004%
across three years — a 3.23% error on every dataset. Correcting it buys 0.5%;
that is not the finding. The finding is that the residual cannot see it.)

Two further defects were found in the same pass:

- **`--mask` and `--generate-mask` were silent no-ops.** They drop an OpenMVG
  `<stem>_mask.png` sibling next to the query, which is the right convention for
  `ComputeFeatures` on a normal scene — but localization runs
  `openMVG_main_SfM_Localization -q <query_dir>`, which *scans that directory as
  images*. Measured: the query's feature count was unchanged at 60917, and the
  mask itself was featurised as a spurious 56-feature view.
- **The sparse path needs `matches_dir` regions** — 840 `.feat` + 840 `.desc`,
  ~2.2 GB per dataset — and fails with `Invalid regions` without them. So it
  cannot run against a published dataset whose `intermediate.tar.gz` has not been
  extracted.

## Decision

**One tool, two correspondence backends, and everything else shared.**
`pgs-calibrate` and this have the same goal — put a camera into a solved scene
and emit a reusable calibration — and differ in exactly one thing: where the
3D↔2D correspondences come from. Intrinsic handling, pose refinement, frame
conventions, validation and the emitted `.json` are common.

| backend | how | needs |
|---|---|---|
| sparse | match the query's descriptors against the structure's | `--input-scene` + `--matches-dir` |
| render-and-match | render a same-modality textured mesh from a prior pose, match render↔query, lift through the render's own position map | `--mesh` + a prior |

**Chained — all four — is the recommended mode.** The sparse path is a poor
*answer* but a fine *prior*: its worst measured failure is 23.13 px, while
render-and-match converges from a generic prior **861 px** off. So sparse
supplies the prior, render-and-match supplies the accuracy, and "no prior
exists" stops being a reason to keep two tools.

Measured over twelve datasets: render-and-match **12/12 at 1.81–2.53 px**,
sparse **3.96–23.13 px** and wrong on half.

### 1. It is a C++ tool, not a Python app

A bare binary in `bin/` beside `pgs-decimate` and `pgs-sfm-orient`, with no
Python wrapper and no console script. `pgs-calibrate` was Python because it
*orchestrated* a binary; this one has to be in the process the localization runs
in, because three of the four things worth fixing are only fixable there:

1. **the query is described in memory**, so there is no `-q` directory to scan
   and the mask bug cannot exist by construction;
2. **`--sfm-transform` is applied to the scene at load**, not to the pose on the
   way out;
3. **K comes from `--camera` and is never fitted**, when one is given.

### 2. The scene is transformed at load, not the pose on the way out

`pgs-calibrate` resected in the SfM frame and then re-expressed the *result* in
the centered frame. `pgs-localize` applies `--sfm-transform` to the scene with
`ApplySimilarity` before anything is matched, and four things fall out rather
than being arranged:

1. the resected pose is natively in the mesh's frame;
2. **the structure is too** — so the sparse backend's 3D points and the mesh
   backend's render-lifted points live in one coordinate system and can go into a
   *single* solve rather than a resect-then-refine sequence;
3. the standoff gate becomes writable at all. The raw resection centre on a real
   dataset is `(-28.64, -69.93, -114.46)`; transformed it is `(≈0, ≈0, 153)`.
   There is no way to write the check against the SfM frame;
4. there is one frame in the tool, and no "which frame is this in" bookkeeping.

That last one is worth real money. A frame error of this kind is a proper rigid
motion, and the pose absorbs it *exactly*: a 180° error about Z in the source
work posted a **bit-identical** held-out reprojection rms (1.741 px) either way.
No residual, inlier count or cost can reveal it. Two such errors were made in
that work; both were caught only by rendering the solved pose and looking.

`--sfm-transform` is **plain optional** — no assertion, no warning, no geometric
cross-check. Whether a centered mesh exists is an external condition the tool
cannot see, and localizing against an uncentered scene is the more common
workflow. `--qa-render` and the standoff gate are therefore the only defences
against getting it wrong.

### 3. The renderer is ray-cast, and it is `rt_reorder_texture`'s

`registration-toolkit`'s `--projection camera` loop, reproduced
(`ReorderUnorganizedTexture.cpp`, camera-mode branch). **Every measured number
in this ADR was produced by rendering through it**, so matching it keeps the
validation describing this renderer and not a different one.

Ray-cast rather than the scanline rasterizer the original plan specified. Per
output pixel, build the pinhole ray through the pixel centre, intersect a BVH,
take the nearest hit. That deletes all three traps a rasterizer has here: there
is no `1/z` to remember to interpolate linearly, no perspective-correct `uv/z`,
and no back-projection — the world position is exact by construction,
`pos = C + d·dist`. Backface culling becomes moot for the same reason, a hit
being the nearest surface regardless of facing. Buffers start at NaN so a miss
is distinguishable from a valid sample at the origin, and a keypoint is lifted
only when all four of its position-map neighbours are finite.

Two conventions are inherited deliberately and are worth naming, because both
are invisible when wrong:

- **`bvh.prim_ids` permutes primitives**, so a hit index must be mapped back
  before it means a face; and the BVH's barycentrics are relative to the
  **second** vertex, which is why the sampler's argument order is rotated.
- **UV→pixel is pixel-centre**, `x = u·cols − 0.5` — the convention a
  texture-mapper means by a UV and the one OpenMVS authored the atlas against.
  This is the one place the renderer knowingly departs from RT, which uses the
  corner-aligned `x = u·(cols−1)`; the two were measured against each other and
  came out indistinguishable (below), so the correct one is the one kept. A
  render is therefore not bit-comparable with `rt_reorder_texture`'s.

One deliberate **divergence** from RT: the ray directions are not undistorted
(`ReorderUnorganizedTexture.cpp:1122`), so this renders the ideal pinhole camera
rather than the distorted camera's image space. Distortion is handled on the
other side, by leaving the query keypoints in the query's own distorted
coordinates and letting openMVG undistort them before P3P — which is both faster
and lossier-free than warping a render and its position map. The two renderers
are therefore bit-comparable only on an undistorted camera, which is every camera
this has been measured on: `k1` measures as zero on the rig.

### 4. Local normalization is not optional

Subtract a Gaussian local mean, divide by the local SD (σ ≈ 15 px), clip to ±3,
on **both** images. This is what makes "same modality" a weak enough requirement
to be useful: the two images will have had different processing even in the same
band — in the measured case the mesh texture carried a 15-knot tone curve and the
query a linear stretch — and local normalization erases that entirely. Without
it the match rate collapses.

### 5. The similarity RANSAC is an outlier filter, nothing more

A 2D similarity cannot express the true render-to-query relation, which is
perspective-plus-relief. What it is for is **rejecting the sample square**: the
printed label is a loose insert in its holder and the tray is not fixed relative
to the holder, so the square sits in a different physical place in the
photogrammetry capture than in the spectral one — and it is the most
feature-dense object in the frame, carrying 12% of the query's features from 5%
of it.

`--mask` exists and is honest about what it buys. Tested on all twelve by baking
the square out: bias improves on 11 of 12, and three datasets go from 4–5 px to
~2.2 px. But **zero of the six sparse failures are rescued**, the corpus mean
does not move (10.08 → 10.00 px), and one dataset got substantially worse
(14.00 → 27.75 px) because removing the highest-contrast 12% left that solve
under-constrained. A working mask is worth having; it is not a remedy.

### 6. One pass. Do not iterate

Re-rendering from the refined pose and re-solving was measured over twelve
datasets and changes nothing: `PHerc0013Cr04` went 1.820 → 1.821 → 1.838 px over
a second and third pass. `--second-pass-min-inliers` exists as a guard on a thin
first pass and defaults to **0 = off**.

### 7. Held-out metrics, and one gate that does not depend on them

QA is a 400 px spatial checkerboard: solve on one colour, score on the other,
then emit the pose solved on everything. One extra resection. The published
thresholds were measured this way and **do not transfer to a plain inlier RMS**,
which is a different statistic over a set the solve itself chose. The report key
is `heldout` and not `independent` on purpose: the matches were still found using
a render from the pose being scored, so it bounds the fit, not the registration.

| metric | observed over 12 | review | fail |
|---|---|---|---|
| held-out `rms_px` | 1.81–2.53 (mean 2.10, sd 0.25) | > 3.5 | > 5.0 |
| `p99_px` | 3.99–5.65 (mean 4.77) | > 8 | > 12 |
| `\|bias\|_px` | 0.06–0.40 (mean 0.17) | > 1.2 | > 3.0 |
| `n_inliers` | 162–1946 (median 601) | < 250 | < 64 |
| `\|Z − expected\|` | sd 0.395 cm, range 1.42 cm | > tol | > 2·tol |

Four of those are baked in and one is supplied, and the difference is the point:
rms, p99, bias and inlier count are properties of the *method*, while a standoff
of 154.2 cm is a property of *one rig*. It is supplied the way the rig's other
properties are — as `expected_standoff` / `standoff_tolerance` keys in the
camera file, beside K and the prior pose, with `--expected-standoff` /
`--standoff-tolerance` overriding per run. A check this load-bearing does not
belong in every call site, and the camera file's own `pose` already carries that
distance as `t_z`, so a caller repeating it on a command line would be keeping a
second copy free to drift. The gate runs only when set, the solved standoff is
always reported, and a run that ends up ungated says so.

**The standoff row is the only gate independent of the correspondence set**,
which is exactly what makes it the one that catches a frame error. It is also
the tightest invariant available on this rig: the solved camera-to-scene distance
is rigid to 0.3 mm across the corpus (sd 0.395 cm over a 1.42 cm range) while
lateral position floats by 1.7 cm through the planar translation/tilt degeneracy.
A caller-supplied expected distance plus tolerance would have caught all six
sparse failures.

## Validation

Run from the single generic prior (`R = diag(1, −1, −1)`, centre
`(−2.67, −2.75, 154.16)`, `f = 21700.2`, pp `(4000.9, 3591.2)`) against the
IR940 meshes and the 940 FN band, `--describer-preset HIGH`: **12/12 converged**.

Against `specreg-tools/cams/*_p2.cam`, the poses the source investigation solved:

- standoff differs by **−0.024 … +0.029 cm** (max 0.29 mm), sd 0.14 mm;
- the lateral difference matches `−Δpp·Z/f` to ~0.006 cm on **all 24
  components**.

That second line is the result. The entire remaining difference between this
tool and the reference is the principal point each was given — the reference used
a per-dataset leave-one-out value, this run used the corpus one — not the method.

Held-out metrics against the published ranges over the same twelve:

| metric | published | here |
|---|---|---|
| `rms_px` | 1.81–2.53, mean 2.10 | 1.53–2.47, mean 1.92 |
| `p99_px` | 3.99–5.65, mean 4.77 | 3.88–9.41, mean 5.63 |
| `\|bias\|_px` | 0.06–0.40, mean 0.17 | 0.08–0.37, mean 0.23 |
| `n_inliers` | 162–1946, median 601 | 684–10934, mean 4262 |
| standoff | sd 0.395, range 1.42 | sd 0.403, range 1.43 |

### The describer preset, settled by measurement

The plan defaulted `--describer-preset` to ULTRA, as a guess at mapping the
prototype's "`cv::SIFT` at 40 000 features" onto openMVG's knobs. The corpus was
run both ways and the default is now **HIGH**:

| | HIGH | ULTRA |
|---|---|---|
| held-out `rms_px`, mean | 1.920 | 1.873 |
| `p99_px`, mean | 5.634 | 5.048 |
| `\|bias\|_px`, mean | 0.225 | 0.177 |
| wall clock, mean | 66 s | 335 s (**5.1×**) |
| gates passed | 11/12 | 11/12 |
| standoff vs reference, mean / max | 0.132 / 0.288 mm | 0.141 / 0.313 mm |

ULTRA sets `first_octave = -1`, which upscales an 8176 × 6132 query 2× before
building the scale space, and it does produce a better *fit*: every held-out
statistic improves, `p99` by 10%. What it does not produce is a better *pose*.
The two agree on the solved standoff to **36 µm on average and 170 µm at worst**
— inside the noise floor of the matcher itself, which a same-camera control puts
at 0.345 px ≈ 24 µm — and HIGH is if anything marginally closer to the
independently solved reference poses. The same dataset trips the same `p99`
review gate under both.

So the extra features improve the residual over the correspondences they add
without moving the answer, at five times the cost. HIGH is the default; ULTRA is
for a case where match counts are genuinely thin, which nothing in this corpus
is — HIGH already finds 7× the prototype's inliers.

`n_inliers` is 7× higher because openMVG's SIFT at `HIGH` finds far more than the
`cv::SIFT` the prototype capped at 40 000 features, and **`p99` runs higher for
the same reason**: a larger correspondence set has a longer tail at the same rms.
One dataset (`PHerc0017Cr3_Osloense`, p99 9.41) trips the published review
threshold of 8 on that alone. The thresholds are kept as published — a review
band firing on the widest tail in the corpus is doing its job — but a reader
comparing a `pgs-localize` report against the workplan's table should know the
correspondence sets are not the same size.

### A worked example of the quiet failure, from this tool

Run uncalibrated — `--input-scene` with no `--camera`, so DLT estimates the
focal — `PHerc0013Cr04` produced this:

| | recovered | true |
|---|---|---|
| focal | 35954 px | 21700 px (66% high) |
| standoff | 239.5 cm | 153.2 cm (56% high) |
| held-out `rms_px` | **1.768** | — |

Gates: `rms pass, p99 pass, bias pass, inliers review, standoff unset`. A camera
56% wrong in depth, posting a 1.77 px residual, passing three of five gates —
and the one row that would have caught it was **unset**, because no
`--expected-standoff` was supplied. This is §2's degeneracy reproduced by the
new tool rather than quoted from the old one, and it is the whole argument for
the standoff gate in a single run.

`ComplainAboutIntrinsic` did not fire here, and could not: the recovered
principal point (2450, 4927) lands inside the image. That check is a port of
`pgs-calibrate`'s `validate_localized_intrinsic` and catches the *loud*
degeneracy — a non-positive focal, a principal point off the sensor, which the
same configuration does produce on other runs, DLT being nondeterministic. It is
not, and cannot be, a check on whether the focal is *right*. Nothing in the
residual can be. Only the standoff gate can, and only when the caller arms it.

### The UV convention, settled by measurement

The plan left `UVToPixel` on `rt_reorder_texture`'s corner-aligned convention
for comparability, with the instruction to measure the technically correct
pixel-centre one and switch if it moved nothing. The two differ by a *ramp*, not
a constant — `x_corner − x_centre = 0.5 − u` — so it is a half-texel warp across
the atlas; and because the 3D lift comes from the position map rather than from
the texture, the wrong one is a systematic texture-against-geometry
misregistration rather than a wash. Worth measuring.

Swept both ways over all twelve:

| | corner | centre | paired difference (centre − corner) |
|---|---|---|---|
| held-out `rms_px` | 1.9173 | 1.9028 | −0.0146 ± 0.0103 (SE), centre better 7/12 |
| `p99_px` | 5.4725 | 5.5291 | +0.0565 ± 0.209 (SE), centre better 6/12 |
| `\|bias\|_px` | 0.2261 | 0.2065 | −0.0196 ± 0.0171 (SE), centre better 7/12 |
| standoff vs reference | 0.1249 mm | 0.1321 mm | 30 µm apart on average |

Nothing clears 1.5 standard errors and every win rate is a coin flip. It moves
nothing, so **pixel-centre is what the renderer does**, and corner-alignment is
gone rather than kept behind a flag: between two indistinguishable options the
correct one wins outright, and a knob whose two settings cannot be told apart is
a question posed to every future caller that the measurement has already
answered. The cost is that a render is no longer bit-comparable with
`rt_reorder_texture`'s — worth naming, since matching that renderer is what §3
argues keeps this validation honest, and the table above is the evidence that
the half-texel is not what the honesty rested on. The per-dataset validation
table was measured corner-aligned; the difference is inside its noise either way.

### The sparse backend against the tool it reimplements

The sparse path is a reimplementation, so the question is whether it *is* one.
It was checked against `pgs-calibrate` itself — still installed in the published
`edge` image — given the same scene, the same regions, the same query, the same
`--sfm-transform` and the same intrinsic.

Pose **parameters** are the wrong thing to compare here. On `PHerc0013Cr04` the
two solved centres sit 6.2 mm apart with a 0.23° rotation between them, which
looks alarming until the direction is checked: a 0.23° tilt at Z = 1532 mm needs
a 6.1 mm lateral shift to compensate, and the measured offset is 6.16 mm. That
is the planar translation/tilt degeneracy this rig has, not a discrepancy — and
the old tool slides along it by the same amount between reruns of *itself*.

So the comparison is over what the poses **do**: reproject the mesh's own
vertices through each camera and measure the pixel disagreement, with the tools'
own run-to-run spread as the baseline. Both are RANSAC-based and neither is
seeded here.

| | old vs old | new vs new | **old vs new** |
|---|---|---|---|
| `PHerc0013Cr04`, 3 runs each | 0.688 px | 0.545 px | **0.702 px** |
| `PHerc0018Cr04`, 3 runs each | 2.794 px | 2.043 px | **2.275 px** |

Mean reprojection difference over the mesh surface. On the second dataset —
chosen because it showed the widest single-run gap of the five tried — the new
path agrees with `pgs-calibrate` *better than `pgs-calibrate` agrees with
itself*. It is also consistently the more repeatable of the two, which follows
from describing the query in memory rather than round-tripping it through a
prepared JPEG.

Single-run old-vs-new over five datasets (the ones carrying both an RGB PSC
query and a solved scene): mean reprojection 0.39, 0.52, 0.70, 1.11 and 1.81 px,
with the solved standoff agreeing to between 0.15 and 0.43 mm. Every one of
those is inside the corresponding within-tool spread.

### The backends against each other

The backends were checked against each other on `PHerc0013Cr04`, all three modes
emitting a standoff against the reference's 153.209:

| mode | standoff | held-out rms | inliers |
|---|---|---|---|
| sparse only (RGB PSC query) | 153.220 | 3.44 px | 208 |
| render-and-match (940 query) | 153.226 | 1.61 px | 4441 |
| chained, fused (940 query) | **153.207** | 1.97 px | 4616 |

Fusing moves the emitted pose closer to the reference — 0.02 mm against 0.17 mm
— while the held-out rms reads *worse*, because the test half now contains
sparse correspondences whose own residual is 3.4 px. The two numbers are not
measuring the same population, and the pose is the deliverable.
`--fuse-sparse 0` keeps the sparse resection as the prior only, for a held-out
statistic directly comparable with the table above.

## Consequences

- **`pgs-calibrate` is deleted**, with its console script and its tests. It had
  exactly one caller (`acquisition-workflow/pipeline/pipeline_registration.py`),
  so this is a replacement, not a coexistence. `save_camera_file` and
  `parse_intrinsic_file` moved into the C++ tool with it.
  `pgs_recon/utils/sfm_json.py` survives — `pgs-retexture` uses it — and so does
  `recon_dir.resolve_solved_sfm`, for the same reason.
- **Nothing downstream changes.** The emitted calibration is the same one-view
  openMVG scene `pgs-retexture --calibration` and `repoint_calibration` consume,
  and the emitted camera file is the same flat format
  `rt_reorder_texture --camera-file` reads. Both are written through openMVG's
  own cereal serializer, which sidesteps `fix_polymorphic_registration` entirely.
- **There is no `--recon-dir`.** openMVG-style explicit paths: `--input-scene`
  plus `--matches-dir`. Reimplementing a manifest reader in C++ would create a
  second consumer of a format `stages.py` owns
  ([ADR 0006](./0006-stage-named-artifacts.md)), and
  `pipeline_registration.py` already passes `--sfm-data` explicitly for exactly
  that reason.
- **`pgs-localize` is the one C++20 target** in `dependencies/utilities/`: bvh v2
  needs `std::span`. openMVG's bundled cereal does *not* compile at C++20 (its
  `StaticObject::LockGuard` declares a defaulted copy constructor, which P1008
  stopped leaving an aggregate behind, so cereal's own `return LockGuard{};`
  finds no constructor), so the one file that has to meet cereal —
  `localize_describer.cpp`, which reads a matches directory's
  `image_describer.json` — is its own small C++17 library. No cereal type crosses
  that boundary.
- **The describer for the sparse backend is never a flag.** It is read from the
  scene's own `image_describer.json`, because the query's descriptors have to be
  commensurable with the database's and the database cannot be re-described.
  That is also why the sparse path describes the *plain* 8-bit query while the
  mesh path describes the locally normalized one: the database regions were
  computed from ordinary photographs.
- **The query modality changes** for the caller, from the RGB composite to the
  native 940 band, which is what makes it same-band against the IR940 mesh.
  `DEFAULT_FOCAL_LENGTH = 21000` is deleted rather than corrected.
- **The `mvg` extraction from `intermediate.tar.gz` is no longer needed** for
  registration. The mesh backend reads neither the SfM scene nor the regions, so
  it runs against a published dataset with nothing extracted. Keep the extraction
  only if the pipeline elects to run the sparse backend too, for the automatic
  prior.

## Not settled

- **Image decode is `cv::imread` alone, with no libtiff fallback.** The plan
  called for carrying `rt::ReadImage`'s fallback, because OpenCV chokes on some
  scientific TIFFs. It is not here: `cv::imread` read all twelve 16-bit
  MegaVision query TIFFs and all twelve atlases without complaint, so the
  fallback would be ~240 lines and a libtiff link that nothing in evidence
  exercises. If a capture ever fails to decode, that is where to add it —
  `ReadMeshBundle` warns by name and the query path errors by name, so the
  symptom will point straight at it.
- **The LK window (21 × 21, 3 levels) is OpenCV's default, not a measurement.**
  The refinement it does is worth a measured 6% (2.077 → 1.948 px mean); the
  window size behind that number is not recorded anywhere.
- **PnP RANSAC threshold, iterations and confidence** are unspecified in every
  source document; `--residual-error 0` leaves openMVG's own.
- **There is no independent geometric probe.** Every measured number here is on
  the papyrus itself. The sample square is not a valid reference because it
  moves. Standoff is the only correspondence-independent check that exists, and
  it is one number.
- **Accuracy is bounded by the reconstruction, not the camera model.** Sweeping
  `f` from 18000 to 26000 moves held-out rms by 0.1 px with no minimum in range;
  using the image centre instead of the measured principal point costs 0.011 px.
  A control — spectral 940 against spectral 1050, same camera, same capture,
  aligned by construction — puts the matcher itself at **0.345 px (24 µm)**. So
  of the ~2 px floor, roughly 1.9 px originates in the mesh and its texture.
  Better intrinsics buy consistency; better reconstructions buy accuracy.
