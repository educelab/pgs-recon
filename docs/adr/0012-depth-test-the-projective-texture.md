# Depth-test the projective texture, and answer per face in fractions

**Status: accepted, measured on one dataset.** Amends
[ADR 0002](./0002-calibrate-new-camera-for-retexture.md), whose consequence
"Projective UV has no occlusion test" this reverses. The numbers below are from
`PHerc0002Cr1_Osloense`: a 2,693,527-face mesh through the 8176x6132 spectral
camera `pgs-localize` solved for it.

## Context

`pgs-retexture --calibration` without `--use-openmvs` textures by projecting
every vertex through the one calibrated view and writing the pixel it lands on
as a UV (ADR 0002). Two tests decided whether a face was textured: all three
vertices in front of the camera and inside the image, and -- unless
`--no-backface-cull` -- the face pointing at the camera.

Neither of those is a visibility test. A face hidden behind a nearer part of the
same mesh passes both: it is in frame, and it faces the camera. It therefore
took the *occluder's* pixels, and nothing downstream marked it. ADR 0002 called
this negligible "for the single-sided surface meshes this targets" and pointed
at `--use-openmvs` for the cases where it is not.

It is not negligible. On the fragment above, **137,028 faces (5.1%) are hidden
entirely and another 22,464 straddle an occlusion boundary** -- 159,492 faces,
5.9% of the mesh, that were being textured with pixels belonging to something
in front of them. The geometry is not exotic: the fragments sit under glass on a
tray, and every raised plate rim hides a band of whatever is behind it. The band
is *zero* at the principal point and widens toward the frame edge, because its
width is the rim's height times the tangent of the off-axis angle. That is the
signature that identifies it -- see "How it was validated" below.

`--use-openmvs` is not an answer to this. It resolves occlusion per texel, but
it also regenerates UVs into a resampled atlas, which is the whole thing the
projective path exists to avoid: its UVs point at the original full-resolution
image and are identical across modalities, so every modality of a capture reuses
one OBJ and swaps only `map_Kd`.

## Decision

### 1. Rasterize the mesh into a z-buffer through the same camera

`pgs_recon/utils/visibility.py`. Sample each projected triangle at the pixel
centres it covers, keep the nearest depth per pixel, then ask each face how many
of its own samples survive. Camera-space Z, not range. Perspective-correct depth
(interpolate 1/z), because the barycentrics are screen-space.

Python and NumPy, in `utils/`, because the projective path is pure Python today
and pulling a built binary into it would make a `pip install .` insufficient to
run it. Cost at the scale above: **~5 s and ~2 GB**, against ~7 s for the rest
of the run. A 1.28M-face synthetic case at 4000x3000 runs in 1.8 s.

### 2. The answer is a fraction, not a flag

A face straddling an occlusion boundary is *partly* imaged, and a mesh cannot
texture half a triangle. `visible_fraction` returns the share of each face's
samples that nothing nearer covers, and `--occlusion-coverage` (default 1.0)
is where the caller decides what is enough. The boundary costs one face either
way; the choice is between a face-wide fringe of foreground smeared onto the
hidden surface and a face-wide fringe with no texture at all. Defaulting to 1.0
means "textured" keeps its plain meaning: the camera saw this face, whole.

This is deliberately *not* an enable flag. `--no-occlusion-cull` turns the test
off; `--occlusion-coverage 0` is refused rather than quietly meaning the same
thing. That is the lesson of the decimation budgets (ADR 0008, and CLAUDE.md's
note on it): a threshold that doubles as the off switch cannot have a default
and cannot be set to zero for its own sake.

### 3. Every face writes to the depth buffer, including a sub-pixel one

Faces are sampled at the pixel centres they cover **plus** one guaranteed sample
at the face's own projected centroid (projected, not averaged from the projected
vertices -- under perspective those are different points).

Without that fallback the test fails silently in exactly the case this tool is
for. The mesh is solved at the rig's ~1 px ground sample distance; the localized
camera stands further off (154 cm here), so its faces are *smaller* than its
pixels. Sample only covered pixel centres and such a mesh puts almost nothing in
the buffer, every face reads as visible, and the run looks like a pass.

### 4. The tolerance is slope-scaled, and that half is not a knob

A sample counts as occluded only when something is nearer by more than a
tolerance with two terms:

- **depth-relative**, `DEFAULT_DEPTH_BIAS = 1e-3` of the sample's own depth --
  0.5 mm at a 0.5 m standoff. Float error and mesh noise. `--occlusion-bias`.
- **slope-scaled**, `_SLOPE_PIXELS = 1.5` times the face's depth gradient *per
  pixel*, the gradient measured as the face's depth span over its projected
  extent (floored at `_MIN_EXTENT_PX = 0.1`, below which a face is a sliver and
  its measured gradient is noise over an arbitrarily small number).

The second term is not a fudge factor covering for the first. Once faces are
sub-pixel (decision 3), several *distinct* faces land in one pixel at genuinely
different depths, and the nearest wins: on a tilted fine mesh every face but one
per pixel reads as occluded by its own neighbour, and the surface speckles. This
was observed before it was fixed -- a 45-degree plane at 90x90 cells lost faces
to itself until the tolerance followed the gradient. It is per *pixel* rather
than per face on purpose: a large face may span a lot of depth across many
pixels and must keep a tight tolerance, or a real occluder in front of it is
missed. This is the standard shadow-map slope bias.

Three constants, none of them derived from the rig the way `COARSEN_RATIO` is
(ADR 0009). They are bounds on float error and on how far apart two samples in
one pixel can be, so they are properties of the sampling, not of the optics.

## How it was validated

Two checks, because "5.1% hidden" is equally consistent with a working depth
test and a broken one.

1. **The shape of it.** Rendering each dropped face's own footprint (not the
   nearest face per pixel -- a hidden face is by definition behind another one,
   so a nearest-face render draws its occluder and never it) puts the dropped
   bands on the inside of every plate rim, **asymmetric, widening toward the
   frame edge and vanishing near the principal point**. That is parallax, and no
   self-occlusion artifact has that structure.
2. **The connectivity of it.** Clustering the 137,028 hidden faces over the
   mesh's own edge adjacency: **1,657 connected regions, largest 8,874 faces,
   0.53% singletons, 93.9% in regions of 50+ faces.** Speckle is isolated faces.

The synthetic suite (`tests/test_visibility.py`) holds the cases that separate a
real depth test from a plausible one: no self-occlusion on flat, tilted, and
tilted-sub-pixel surfaces; occlusion by disconnected geometry; the straddling
fraction; sub-pixel meshes in both roles (occluder and occluded).

## Consequences / non-obvious traps

- **Default-on is a behaviour change.** A projective retexture run before this
  textured ~5% more faces than one run after it, and the difference is faces
  that were wrong. 2.0 is alpha; `--no-occlusion-cull` reproduces the old
  output exactly.
- **Unseen faces keep their geometry and lose their UVs**, which is what
  out-of-view faces already did -- the single-view analogue of OpenMVS'
  `--empty-color`. Nothing is dropped from the mesh.
- **The whole mesh occludes**, back-facing and out-of-view faces included: a
  solid surface is occluded by its own far side too.
- **A face seen near edge-on is tolerant of occlusion**, because its per-pixel
  gradient is enormous. That is the honest degenerate case: such a face is at a
  grazing angle where the texture is garbage regardless.
- **The app keeps the in-view and back-face tests; the module owns only the
  depth test.** They look like one predicate but are not: `visible_fraction`
  answers over whatever samples a face has *inside* the frame, while the app's
  rule is that all three vertices must be in frame, which is a statement about
  whether its UVs mean anything.

## Ruled out

- **`--use-openmvs`.** Already the fallback ADR 0002 named, and it is a
  different deliverable: a resampled atlas with regenerated UVs, which is what
  the projective path exists to avoid. Both now resolve occlusion; they differ
  in what they hand back.
- **Reusing `pgs-localize`'s renderer** (`localize_render.cpp`, ADR 0011),
  which already ray-casts a mesh through a pinhole camera with nearest-hit
  occlusion and is self-tested for exactly that. It is the better long-term
  answer -- one renderer, two consumers -- and it was not taken here because it
  would put a built binary in the path of a code path that currently needs only
  NumPy, and because what that renderer returns (a position map and a depth
  map) is not what this needs (a per-face fraction), so the consumer would have
  to rasterize face footprints against the depth map anyway. The conventions are
  deliberately the same on both sides: camera-space Z, pixel centres, nearest
  hit. **If they are ever changed, they must be changed in both.**
- **Subdividing faces along the occlusion boundary**, which would make the
  fraction unnecessary by making every face wholly visible or wholly hidden. It
  changes the mesh, and the mesh is the deliverable a retexture must not touch.
- **Supersampling the depth buffer** to separate sub-pixel faces instead of
  biasing. It costs 4x memory per level and never fully separates them; the
  slope term is exact about what it is for.
