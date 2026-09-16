# Localize a new camera against an existing scene to texture from a novel pose

**The calibrate half is superseded** by
[ADR 0011](./0011-localize-by-rendering-the-mesh.md): `pgs-calibrate` is deleted
and `pgs-localize` replaces it. The *problem* this ADR states, the split into
calibrate-then-retexture, and everything about `pgs-retexture --calibration`
still stand unchanged — the calibration `.json` has the same shape and the same
consumers. What changed is where the correspondences come from, because matching
the query against the sparse structure was measured wrong on six of twelve
datasets, silently. The superseded passages are marked in place below.

## Context

ADR 0001's `pgs-retexture` covers a modality captured **at the same capture
positions as an existing rig camera**: it reuses that camera's solved poses and
matches modality images to views purely by the
`{prefix}_{camera}_{position}_{capture}` filename convention.

We also need to texture from a camera that was **never part of the
reconstruction** — e.g. an overhead registration camera that images the whole
fragment top-down. There is no solved pose to reuse, and its filenames do not
follow the rig convention. That camera typically captures several co-registered
modalities (RGB, IR, ...) from one physical position, and we want to texture the
mesh with each modality individually.

## Decision

Split the work into **calibrate** then **retexture**, because the pose +
intrinsic is a property of the physical camera position and is shared by every
modality it captured — so solve it once, reuse it many times.

- **`pgs-calibrate`** localizes one image (the feature-richest modality, usually
  RGB) into the existing solved scene with `openMVG_main_SfM_Localization`, then
  extracts just that localized view (pose + freshly estimated intrinsic) into a
  one-view `*_calibration.json`. Correspondence is geometric (feature matching),
  so no filename convention is required.

  **Superseded by [ADR 0011](./0011-localize-by-rendering-the-mesh.md).** The
  tool is now `pgs-localize` and the recommended correspondence source is a
  *render* of the mesh at a prior pose rather than the scene's sparse structure:
  render-and-match measured 1.81–2.53 px over twelve datasets where this path
  measured 3.96–23.13 px and was wrong on six of them, silently, passing its own
  validation every time. The sparse path survives as a backend and as a source of
  the prior. The extracted one-view `*_calibration.json` is unchanged, which is
  why the next bullet needs no amendment.
- **`pgs-retexture --calibration <json>`** textures the mesh from that one view.
  By default it does **projective UV mapping**: it projects the mesh through the
  calibrated camera so the OBJ's UVs index the *original* full-resolution image
  directly, with `map_Kd` pointed at it. `--use-openmvs` instead runs the
  `openMVG2openMVS` → `TextureMesh` path (resampled atlas, true occlusion).

## Why projective UV mapping is the default

OpenMVS `TextureMesh` regenerates UVs and bakes the source pixels into new atlas
images — it downsamples/recompresses the original and splits it across pages
(observed: one overhead RGB became two 8192² atlas textures, both resampled
copies of the same image). It also cannot point UVs back at the source.

Projecting the mesh through the known camera avoids all of that: no resampling
(full original fidelity) and, crucially, the UVs depend only on the camera +
mesh — both shared across a position's modalities — so they are computed once
and every modality just swaps `map_Kd`. That is the natural fit for "retexture
the same mesh with RGB, then IR, then ...". Validated on real data: 84.5% of the
mesh falls in the overhead view; the surface mesh is single-sided (no hidden
underside), so back-face culling only trims ~3% of grazing edge faces.

Validated on real data (PHerc1428Cr04 overhead RGB vs. the rig scene): 305
inliers, resection RMSE 0.83 px, 1/1 poses localized.

**Read that last line in the light of [ADR 0011](./0011-localize-by-rendering-the-mesh.md).**
A resection RMSE on the solve's own tracks is the statistic that cannot see this
path's failure mode: at near-fronto-parallel geometry a wrong focal is absorbed
into standoff to four significant figures and the residual stays clean. All six
of the corpus failures posted one. Held-out reprojection against a spatial split,
plus a standoff check, are what discriminate.

## Consequences / non-obvious traps

- **Frame.** Localize against the *same* solved SfM that fed openMVG2openMVS
  (`recon_dir/sfm_data.bin`, which carries structure); the recovered pose then
  lands in the mesh's frame. `resolve_recon_inputs` already finds that scene.
  The SfM must still contain structure — localization matches the query's 2D
  features against the reconstruction's 3D landmarks.
- **Original scene is read-only.** `openMVG_main_SfM_Localization` writes the
  query's new regions to a private `-u` match dir and the expanded scene to a
  private `-o` dir; the reconstruction's `matches_dir`/`recon_dir` are untouched.
- **Camera model.** Default to **pinhole** (`-c 1`), not OpenMVG's radial-3
  default. (Still true, and `pgs-localize --camera-model` carries the same
  default for the same reason.) A long-focal overhead view has negligible distortion; under DLT
  resection radial-3 overfits it (observed `disto_k3` ~180 and a principal point
  pushed far off-center), and that bogus distortion warps image corners during
  undistortion. Pinhole gave an equal/better fit (RMSE 0.785) with the principal
  point at image center.
- **Resection method.** Default to **DLT** (`-R 0`), which "does not use
  intrinsic data" and so recovers focal length for an uncalibrated camera. The
  P3P methods assume a known intrinsic; only select them when a focal is known.

  **Amended by [ADR 0011](./0011-localize-by-rendering-the-mesh.md).**
  `pgs-localize --resection-method auto` picks DLT when K is unknown and P3P when
  it is known, rather than defaulting to one and refusing the wrong combination.
  `pgs-calibrate` had to refuse it because the stock binary falls back to DLT
  silently; in process there is nothing to fall back from.
- **Colorspace.** The overhead RGB is a **CIELab** TIFF; OpenCV would read the
  L/a/b channels as B/G/R, corrupting both SIFT and texture color. Normalize via
  ImageMagick `convert -colorspace sRGB -depth 8` (reads the embedded
  colorspace and bit depth). This differs from `convert_modality_images`' fixed
  bit-shift: that keeps a *set* of frames mutually consistent for one atlas,
  whereas each overhead modality is textured independently, so per-image tone
  mapping is fine.

  **Superseded.** ImageMagick ignores a Lab TIFF's `WhitePoint` tag and decodes
  every one as D65; the MegaVision captures carry no tag and are D50 (confirmed
  against the ColorChecker in frame — D50 fit better on all 7 patches where the
  two hypotheses diverge). Once `pgs-convert` grew an in-process decode that
  honors the tag, the same file textured differently depending on which tool
  read it. `utils.images.read_srgb` is now the single reader for all three apps,
  and ImageMagick is gone from the images. It reproduces the old output
  bit-for-bit on 16-bit greyscale and 8-bit RGB (both verified), differing only
  on Lab, where it is the one that is right. The claim above that per-image tone
  mapping was happening was also wrong: `-depth 8` is a straight linear
  requantization, `round(v / 257)`.
- **Resolution lock.** The calibrated intrinsic is tied to the calibration
  image's pixel dimensions. `pgs-retexture --calibration` aborts if the modality
  image's dimensions differ; all modalities must share the camera's resolution.
- **Projective UV culls back-faces and out-of-view triangles.** The UV
  convention is the standard OBJ one (v flipped: image top-left -> UV top).
  This bullet used to end "and has no occlusion test ... use `--use-openmvs`
  when true occlusion handling is needed", which is no longer true: it now
  depth-tests too, and the surface that overhangs itself no longer takes the
  foreground's pixels. See ADR 0012 for what that test is and what it costs.
- **OpenMVS needs >=2 images.** TextureMesh rejects a single-image scene with
  `error: invalid project` (verified against v2.3.0), even though one view is
  enough to texture. `repoint_calibration` therefore writes the lone overhead
  view twice (same pose + intrinsic, the image copied to a distinct basename so
  openMVG2openMVS's by-basename undistorted output does not collide). The
  texture is unaffected — both views are identical.
- **Single-view coverage.** One overhead view sees only the top surface; faces
  it cannot see take `--empty-color`. Seam-leveling defaults (global on, local
  off) are inherited from `pgs-retexture`.
- **Resection is under-constrained for a near-planar overhead view.** Focal,
  camera distance, and principal point trade off, so estimates vary run-to-run
  (RANSAC) — observed focal 18000-20000 px and the principal point drifting in
  y. Reprojection stays sub-pixel so texturing aligns, but a known focal length
  (`--focal-length`, plus a P3P `--resection-method`) stabilizes it.
