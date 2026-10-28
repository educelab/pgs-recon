# Localize a hand-held view set against the rig's IR940 mesh

**Status: proposed, unvalidated.** Nothing below has been measured on hand-held
data. The design is settled to the point where a feasibility spike (Phase 1) can
decide it, and that spike's pass/fail criteria are fixed here *before* it runs so
the result cannot be read to fit. Every number quoted is either from
[ADR 0011](./0011-localize-by-rendering-the-mesh.md) or derived from one of its
numbers, and says so.

Builds on [ADR 0011](./0011-localize-by-rendering-the-mesh.md) and
[ADR 0002](./0002-calibrate-new-camera-for-retexture.md); changes neither. The
emitted calibrations are the same one-view scenes `pgs-retexture --calibration`
already consumes, so [ADR 0012](./0012-depth-test-the-projective-texture.md)'s
projective texture applies to each view unchanged.

## Context

`pgs-localize` puts **one** camera into a solved scene: one query image, a prior
pose, a K that is either supplied or fitted by DLT from that one image, and a
standoff gate that is a constant of one rig. That fits the overhead camera. We
now need the same thing for a **hand-held view set**: many IR images of one
plate, from one camera held by hand, to be textured onto the mesh one view at a
time.

What is known about that data, and what each fact rules out:

| fact | consequence |
|---|---|
| Hand-held IR, ~980 nm; the mesh's main texture is RGB | Render-and-match against the mesh as normally textured is cross-band, which ADR 0011 never measured. Its local normalization absorbs *processing* differences, not band differences. |
| No pose metadata; poses irregular | No generic prior (ADR 0011 used one). The sparse backend is cross-band too, so not a dependable prior source either. |
| One camera, K unknown, focus fixed | One K exists and is shared by every view, so it can be solved once. But ADR 0011's degeneracy applies: near-fronto-parallel, a wrong focal is absorbed into standoff. |
| Angle and distance held roughly constant, by hand | Varied distance cannot be relied on to make focal observable; *tilt* variation, which a hand never removes entirely, may. |
| Noticeable lens distortion toward the edges | Pinhole is not an option; `pinhole_radial_k3` from the start. |
| Overlap between views variable, sometimes too thin to match | The set may fall into several disconnected groups and some singletons. Nothing may *require* view-to-view matches. |
| No standoff is fixed | The standoff gate, the one correspondence-independent check in ADR 0011, has no counterpart. |
| The rig scan carries an IR940 capture | A near-band texture from the same session and solve as the mesh, with the rig's coverage. See Decision 2. |
| A spectral set of the same plate, with the same bands as ADR 0011's, **is** registered to the scan | A whole-plate view at a known pose, in 940 nm: the source of the prior. See Decision 2. |
| Spectral and 3D captures: one session. Hand-held: a different one. No record of handling in between. | **Assumed:** no fragment moved between sessions. The plate is then in the arrangement the mesh describes, up to a rigid motion the poses absorb. See Decision 5 for what still checks it. |
| Lighting handled differently in the two sessions, details unknown | Shading on relief may differ even in the same band. |
| No glass in any session | No reflection or refraction terms. |
| Hand-held footprint on the plate unknown | Measured in Phase 1 rather than assumed. |

## Decision

### 1. Camera calibration, not image registration

Each hand-held view gets a pose and an intrinsic in the mesh's frame, not a 2D
warp onto a reference image. A homography is exact only for a plane: relief of
height *h* under a tilt difference *θ* between two views leaves roughly
*h*·tan *θ* of parallax, which at close-up resolution is many pixels, and a
warp cannot see occlusion or reach a surface the reference view did not image.
Calibration is also the shape every existing consumer takes.

Image-to-image matching appears inside the method -- the similarity RANSAC
(ADR 0011 §5) and the view-to-view tie points -- but only to find
correspondences. Its output is never the result.

### 2. Anchor to the rig's IR940 mesh; take the prior from the overhead view

A hand-held view needs two things from outside its own set: a texture it can be
matched against in its own band, and a starting pose. They come from different
sources.

1. **The IR reference mesh is the rig's IR940 capture**, applied by **capture
   retexture** (`pgs-retexture --capture`, ADR 0001) -- the kind of mesh ADR 0011
   was validated against. Three things make it the right anchor:
   - *Band.* 940 nm against the hand-held's ~980 nm is a 40 nm gap. ADR 0011
     already measured a wider one: its control matched spectral 940 against
     spectral 1050 -- 110 nm apart -- at **0.345 px**. That control held camera,
     capture and lighting fixed, so it isolates band and says nothing about
     lighting (see Phase 1), but it makes the band gap the smaller risk by a
     distance.
   - *Coverage.* Every rig camera contributes, so the texture covers what the rig
     saw rather than what one viewpoint saw.
   - *No intermediate registration.* The texture comes from the same solve as the
     geometry, so render-and-match anchors each hand-held view to the mesh
     directly.
2. **The prior comes from the overhead spectral view.** Its 940 nm image covers
   the whole plate at a pose ADR 0011 already solved. A hand-held view, resampled
   to the overhead's pixel scale, is matched against that image -- 2D↔2D, near
   band, the similarity RANSAC absorbing scale and in-plane rotation -- and the
   matches are lifted through the mesh's **position map rendered once from the
   overhead pose**. That is a resection with no prior, and it seeds
   render-and-match against the IR reference mesh from the hand-held's own pose:
   ADR 0011's chained mode, with the overhead view in the place of the sparse
   backend. The overhead pose only has to land the hand-held view inside
   render-and-match's basin -- ADR 0011 measured convergence from **861 px** off
   -- so its error does not reach the emitted pose.

What this leaves:

- **The accuracy floor is the reconstruction's.** ADR 0011 attributes ~1.9 px of
  its ~2 px floor to the mesh and its texture. That floor carries over in kind;
  its size for a shorter-focal, closer camera is not measured. It is not
  inherited from the overhead registration, and nothing correlates it across
  views through one shared solve.
- **Coverage gaps remain, smaller.** Faces no rig camera saw take an empty colour
  in the atlas; faces with no UVs render with a position but no colour
  (`localize_render.cpp`). Either way the render has a boundary no photograph
  shows. The render therefore needs a **texture-validity mask**, covering both,
  handed to feature detection so no keypoint is taken from it. The atlas is built
  with an empty colour chosen so that it can be told from dark surface.
- **The resolution gap is the atlas's.** Render-and-match runs at roughly the
  IR940 atlas's texel scale, so detail finer than that reaches the solve only
  through view-to-view tie points. The overhead's coarser scale (~70 µm/px,
  derived from ADR 0011's control: 0.345 px = 24 µm) limits only the prior.
- **Frames.** Capture retexture textures the mesh's own geometry, so the IR
  reference mesh is in the mesh's frame by construction. The overhead
  calibration is not: it must be in the same frame, centered or not, and ADR 0011
  §2 showed a frame error is invisible to every residual.

**Fallbacks, chosen knowingly rather than built first.** A plate with no rig
IR940 capture can texture the mesh from the overhead 940 band instead
(`pgs-retexture --calibration`). The hand-held views then inherit the overhead
registration: a floor of ~0.15 mm (1.81–2.53 px held-out at ~70 µm/px), the same
bias in every view, and one viewpoint's coverage. A plate with no registered
overhead view needs a manual seed per group. The tool names which path it took.

*Amended: a virtual overhead instead of a manual seed.* The prior needs only an
image of the plate whose pixels have known world positions, and a render of the
IR reference mesh from straight above is one. `--virtual-overhead` writes that
camera from the mesh's bounding box and renders it in the same pass as its
position map (`position-map --output-image`). CR04, which has no registered
overhead, localized all three of its views through it. It costs one check: the
overhead QA comparison would only compare the reference with itself, so it is
not run, and the hand-held pairs are the only graded evidence. The manual seed
remains unbuilt; a plate with neither an overhead nor a reference mesh is still
refused.

### 3. Structure from the hand-held views, triangulated from the mesh poses

The strongest signal in the data is hand-held IR against hand-held IR: same
camera, same band, same session. So the set is matched on its own, with the
existing `openmvg.py` wrappers (`init_sfm_generic`, `compute_features`,
`compute_matches`, `geometric_filter`), starting K from EXIF and the sensor
database as the import already does.

The match graph is split into connected components, and each component's
tracks are triangulated from its views' **mesh-only poses**
(`mvg_compute_known`, direct mode) -- not reconstructed from scratch. Views in
no component are **singletons**, localized individually through Decision 2
alone.

*Amended after the first real plate.* This originally ran incremental SfM
(`mvg_sfm`) per component, and the joint adjustment re-triangulated its tracks
into the mesh's frame. On PHerc0128 Cr04 it never started: a plate is nearly
flat and hand-held views of it look nearly straight down, which is the
degenerate case for the relative pose incremental SfM initializes from, and
every component failed with no tie points. Every view already has a pose from
Decision 2 before this phase runs, so nothing needs solving -- the tracks need
only intersecting. The adjustment still re-triangulates them and keeps only
which keypoints are one point. On Cr04 that gave 8737 tracks from three views,
none rejected.

### 4. K is solved in the mesh-anchored adjustment, once, across all groups

One shared intrinsic -- focal, principal point, k1–k3 -- adjusted together with
every pose, with the mesh points **fixed** and the tie points free. All groups go
into one adjustment: they share no tie points but they share the camera, which
pools every group's tilt variety into the one solve that needs it. The tie
points start where the mesh-only poses and the starting K triangulate them
(Decision 3), and K is freed in the adjustment that follows.

The two kinds of evidence constrain different parameters, and the design leans on
that: distortion comes mainly from the tie points, which are dense at the frame
edges where distortion lives; focal and principal point come mainly from the
fixed mesh points seen at varied tilts -- in effect Zhang's calibration, with the
object as the board.

Whether focal is observable is **measured on every run**, never assumed. The
method is ADR 0011's: hold f at a ladder of values, re-solve, report the cost
curve. A clear minimum is reported with its width. A flat curve -- ADR 0011's
overhead result, no minimum from 18000 to 26000 px -- means this data does not
determine K; the report says so, and K must come from a board calibration or a
previous run of the same camera. The report carries each group's tilt spread so a
flat curve can be traced to its cause.

**A flat curve means no adjusted K is emitted.** It is not a weak measurement
of the focal but none, and the adjustment does not stay put: it drifts along the
focal/standoff degeneracy while its residuals *improve*. Measured on untilted
synthetic views (`tests/test_localize_synthetic.py`, `TestAdjustUntilted`):
f 1600 -> 11652, k1 0 -> -2.44, every camera 377 units from the truth, anchors
reprojecting at 0.17 px against 0.31 before. The divergence gate cannot catch
this, being a reprojection test. So when the curve is flat and K is not fixed,
every view keeps its mesh-only calibration at the starting K, the report's
`intrinsic_emitted` says `start`, and the run warns. On tilted views the same
fixture recovers f to 0.03%, the principal point to 0.3 px and k1 to 0.002, with
the curve's ends 30-38x its minimum.

The stakes are asymmetric. A focal error absorbed into standoff projects the mesh
almost identically from near the solved pose, so it costs texture alignment only
in proportion to relief. A distortion error displaces the frame edges regardless.

*Amended after the first real plate (PHerc0128 Cr04, three near-nadir views,
tilts of 1-3 degrees).* Two things were wrong:

- **The flatness line was too low.** At 1% the curve passed with its ends at
  1.03x and 1.05x its minimum, and the "observed" focal came with the principal
  point moved 161 px and an overlapping pair worse than before the adjustment
  (0.82 -> 2.02 px coherent shift). The line is now 25%, provisional, between
  that plate and the synthetic tilted views' 30-38x.
- **A flat curve threw away the distortion with the focal.** Focal and principal
  point ride the tilt; radial distortion does not, since no pose can absorb it.
  So a flat curve now holds f and the principal point at the starting K and
  re-solves for distortion alone, and `intrinsic_emitted` says `distortion`.
  On Cr04 that measured ~4 px of displacement at the frame corner. Taking it
  out cut the radial error between each view and its render from +4.8 to
  +9.5 px at the corner (mesh-only) to -2.3 to +2.3 px, and both pairs passed
  (0.06 and 0.97 px). The untilted synthetic fixture recovers k1 at the held
  focal's scale, with no camera moving more than a unit. `start` is now only
  the fallback for a distortion-only solve that fails or is not a physical
  camera.

k2 and k3 are freed only when there are tie points. On a narrow lens the
r^4 and r^6 terms are nearly collinear with r^2 over the frame -- Cr04's corner
is at normalized radius 0.24 -- so their coefficients come out large
(k2 = 2.6, k3 = -39 with ties) while describing an ordinary monotone curve.
Anchors alone see them only through the mesh's error, so without ties only k1
is adjusted.

A ChArUco board shot with the same camera, lens and focus is **optional**: a
check on the self-calibrated K, or its replacement where the curve is flat.
Focus was fixed for the capture, so there is one K and no per-view focal term.

### 5. No movement is assumed, and checked

The hand-held images come from a different session than the mesh, with no record
of handling in between. This design **assumes no fragment moved**: the plate is
then the arrangement the mesh describes, up to a rigid motion of the whole plate
that the poses absorb, and a group's tie points, triangulated from mesh-only
poses, describe the same surface the anchors do. Decision 4's joint adjustment
rests on that.

The assumption is cheap to check and silent when wrong -- tie points and mesh
anchors pulling against each other converge somewhere plausible -- so it is
checked rather than trusted:

- the joint adjustment uses a robust loss;
- every view is **also** solved against the mesh alone, and the report gives each
  view's divergence between the two poses. Under the assumption it is small;
  a view where it is not is flagged, and its mesh-only pose is the one emitted.
  Agreement with the mesh outranks agreement with the set;
- the sample square is masked in the overhead matching regardless. It is not a
  fragment, and ADR 0011 §5 measured it moving between captures of one plate.

Aligning independently moved pieces separately is **out of scope**. If the
divergence check fires on more than an occasional view, the assumption is wrong
for that plate and this ADR is reopened, not patched.

### 6. QA: what is independent and what is circular

Agreement with the IR940 texture is **circular** -- that texture is what the
views were registered to -- so it shows convergence, not accuracy. It is
reported, labelled as such, and gates nothing.

Agreement with the overhead 940 image is **partly independent**. The overhead
pose is a separate solve, and here it supplies only the prior; the emitted pose
comes from the rig mesh. So two solves that disagree reveal an error in one of
them -- but both rest on the same mesh, so an error in the mesh they would share.

The checks that gate:

| check | what it catches |
|---|---|
| Per-view held-out score (ADR 0011's checkerboard), against the *joint* K | A poor fit. Bounds the fit, not the registration. |
| Overlap consistency: two overlapping views projected onto the mesh and compared, IR against IR | Misregistration as doubling. Independent of the overhead entirely. |
| Per-view divergence, mesh-only vs joint (Decision 5) | A violated no-movement assumption |
| Focal cost curve (Decision 4) | An undetermined K |
| Hand-held and overhead 940 projected onto the mesh and compared | Disagreement between two separate solves. Not an error in the mesh they share. |
| `--qa-render` overlay per view | Frame errors, which ADR 0011 showed are invisible to every residual |

The reconstruction's floor (Decision 2) is stated in every report as the accuracy
bound, and is why more views do not lower it.

**The comparisons are graded on the coherent shift, not the typical one.**
`pgs-localize-joint qa` warps one image into another view through the mesh and
phase-correlates them tile by tile. Misregistration moves the tiles together;
per-tile noise points every way. So the graded value is the length of the
tiles' *median shift vector*, and the median shift *magnitude* is reported only
as spread. Grading the magnitude was tried first and is wrong: a magnitude cannot
be negative, so noise alone gives it a floor. Measured on the synthetic plate,
calibrations known to be within ~0.5 px graded at 0.7-1.9 px that way -- the
circular check, against the very texture they were solved on, at 0.7 -- and at
0.09-0.49 px as a coherent shift. A view moved 0.3 units (~8 px) still fails
every pair it is in. Comparisons on fewer than 10 tiles are reported and not
graded. The review and fail lines (1 and 3 px) are provisional.

### 7. Output

- One one-view calibration per view, the shape `pgs-retexture --calibration`
  reads today, with a `pinhole_radial_k3` intrinsic. `utils/visibility.py`
  already applies k1–k3 when it projects vertices.
- A set-level openMVG scene and one camera file carrying the shared K.
- A manifest recording which overhead calibration, which mesh and which band the
  set was localized through. Re-registering the overhead view or re-reconstructing
  the mesh makes every hand-held calibration made from them stale; the manifest
  is how that is detected.
- Texturing is a per-view, per-modality `pgs-retexture --calibration` run. The
  projective UVs depend only on camera and mesh, so one OBJ per view serves every
  modality by swapping `map_Kd` (ADR 0002).

### 8. An optional component, in this repository

Nothing in `pgs-recon`'s pipeline changes: this runs after a finished
reconstruction and only reads it. The dependency is one-way -- the component
imports from the core, the core never imports it.

| piece | form |
|---|---|
| C++ joint tool (provisional name `pgs-localize-joint`) | A new target in `dependencies/utilities/` behind a CMake option (provisional `PGS_BUILD_VIEWSET`), **default OFF**, so the superbuild and the published image are unchanged unless it is turned on |
| Python orchestrator (provisional `pgs-localize-set`) | Its own subpackage (provisional `pgs_recon/viewset/`), one console script, an optional pip extra if it needs dependencies. Uses `utils/recon_dir.py`, `sfm_json.py`, `images.py`, `visibility.py`, `openmvg.py` and `toolchain`; calls `pgs-retexture` as a subprocess, since apps may not import each other |
| Resumability | Its own: a view whose calibration and report exist and match the recorded inputs is skipped. **Not** `StageTracker`, which is bound to `pgs-recon`'s `STAGES` tuple and `pipeline_shape`; generalizing it would change the core this component must not touch |

Two changes to existing code are unavoidable, both additive:

1. **`localize_core`.** The `localize_*` sources compile straight into the
   `pgs-localize` executable today. They move into a static library both targets
   link. A build change, not a behaviour change.
2. **The texture-validity mask** becomes a field of the renderer's output.
   `pgs-localize` does not consume it, so its behaviour -- and every number in
   ADR 0011, measured without it -- is unchanged. `--self-test` covers the
   renderer either way.

The separate render camera, the reusable overhead position map, correspondence
export and the joint adjustment all live in the new tool. Any of them may later be
offered back to `pgs-localize` as flags; nothing here depends on that.

## Phase 1: the spike that decides this

Run on a handful of hand-held views from one or two plates, before any of
Decision 8 is built. Spike code stays outside the package. The IR reference mesh
in (a) below is the rig IR940 capture, applied by capture retexture
(Decision 2).

| measure | how | fails if |
|---|---|---|
| **Footprint** | Seed a few views by hand, resect, report each footprint in overhead pixels, IR940 atlas texels and mm | Footprints of a few hundred overhead pixels -- too little to find against the whole plate. Not fatal: match each *group's* combined footprint instead (Decision 3 already builds groups) |
| **Correspondence** | Correct matches per view against (a) the IR reference render, locally normalized, (b) a geometry-only shaded-relief render, (c) both. With and without the validity mask | Most views below ADR 0011's inlier fail gate (64) in every configuration |
| **Lighting** | (a) vs (b) above | (a) poor where (b) is good: shading is the obstacle -- carry (b) forward. ADR 0011's 940-vs-1050 control makes band the unlikely cause |
| **Prior without seed** | Each view, resampled to overhead pixel scale, against the overhead image | It does not find the seeded location for most views |
| **No-movement assumption** | Residual maps within each view; the `--qa-render` overlay | Residuals that split into clusters, or visible non-rigid change. Not a design fork: the assumption is wrong for that plate (Decision 5) |
| **Focal** | Focal cost curve over the seeded views with free K | Flat. Not fatal: board calibration (Decision 4) |

**Go** if correspondence and prior-without-seed pass. Lighting and focal failures
change which path is taken; correspondence or prior failing reopens Decision 2.
The results are recorded in this ADR, which then becomes accepted or is amended.

## Ordered tasks

Each phase depends on the one before it unless noted.

0. **Glossary.** Add *hand-held camera* (an **overhead camera**'s counterpart:
   many poses, not one), *view set*, and *IR reference mesh* to `CONTEXT.md`.
   Independent of everything else.
1. **Spike** (above). Ends with a written go/no-go in this ADR.
2. **Core seams.** Extract `localize_core`; add the texture-validity mask to the
   render output. Done when `pgs-localize --self-test` passes and `pgs-localize`'s
   output on an existing dataset is bit-identical to before.
3. **Per-view localization in `pgs-localize-joint`.** The overhead backend
   (matching against the overhead image, lifted through a precomputed overhead
   position map), render-and-match against the IR reference mesh with the mask,
   per-view held-out scores, correspondence export. Done when a singleton view
   localizes with no seed.
4. **Joint adjustment.** Shared K and distortion across all groups, mesh points
   fixed, tie points free, robust loss; the focal cost curve; per-view divergence.
   Done when the spike's plates produce a K whose curve has a minimum, or a report
   that says it does not.
5. **Orchestrator `pgs-localize-set`.** Inputs found through `recon_dir`; IR
   reference mesh built through `pgs-retexture --capture`, or reused when the
   run already has one; hand-held SfM split into groups
   through the `openmvg.py` wrappers; the C++ tool driven per set; per-view
   resume; batch retexture; the manifest of Decision 7.
6. **QA.** Overlap consistency between hand-held views; the overhead
   comparison, labelled partly independent; the reconstruction's floor in every
   report; the circular check reported and labelled.
7. **Tests, build and docs.** stdlib tests that skip without the binary, in the
   pattern `tests/` already uses; the CMake option; whether the published image
   turns it on; a CLAUDE.md section. This ADR's status updated from what the
   corpus measured.

## Not settled

- **Footprint**, in overhead pixels and in IR940 atlas texels. Phase 1 measures
  it.
- **The floor for this camera.** ADR 0011's ~2 px was measured on the overhead
  camera; how the mesh-and-texture floor translates to a closer, shorter-focal
  camera is unknown.
- **Assumptions taken as given:** no fragment moved between sessions; focus was
  fixed. The first is checked per view (Decision 5); the second is not checkable
  from the data short of a per-view focal term, which the design leaves out.
- **Gate thresholds for divergence and overlap consistency.** No data exists to
  set them. ADR 0011's held-out gates were measured on the overhead camera and
  may not transfer to a shorter-focal, tilted camera; they are a starting point,
  to be re-measured.
- **Provisional names** -- `pgs-localize-joint`, `pgs-localize-set`,
  `pgs_recon/viewset/`, `PGS_BUILD_VIEWSET` -- until the code lands.
