# Re-texture a mesh with an alternate modality by rebuilding a single-camera MVS scene

## Context

We need to re-texture an existing reconstructed mesh using an alternate imaging
modality (e.g. IR940) captured at the same capture positions as one of the rig
cameras. OpenMVS (verified against v2.3.0) has **no native "texture with a
different image set" option**, so `pgs-retexture` reuses the original SfM
solution, filtered to the modality's camera and re-pointed at the modality
images, then runs `openMVG2openMVS` + `TextureMesh`.

## Decision

Rebuild a minimal MVS scene rather than hand-edit the binary `.mvs` or wait for
upstream support. The flow: convert modality images to 8-bit → export the solved
`SfM_Data` to JSON (views/intrinsics/extrinsics only) → keep only the target
camera's views, re-pointed at the matching modality image (matched by capture
position) → `openMVG2openMVS` (undistorts with the original intrinsics) →
`TextureMesh` against the existing mesh.

## Consequences / non-obvious traps

- **Frame.** Feed the *solved* SfM (`recon_dir/sfm_data.bin`) and the
  *un-centered* mesh (`.obj` before `*_center.tfm.npy` is applied). The
  rig-prior import (`mvg/sfm_data.json`) is a different frame and will misalign.
- **Orphan pruning.** `openMVG2openMVS` rejects scenes carrying intrinsics or
  poses not referenced by any kept view; prune them.
- **Cereal polymorphic registration.** openMVG's JSON registers each intrinsic
  type once (high bit on `polymorphic_id` + `polymorphic_name`); later ones
  reference it by bare id. Dropping the registering intrinsic orphans the type
  ("Could not find type id N") — promote the first kept intrinsic to carry it.
- **Mesh format.** OpenMVS's OBJ reader is strict and mis-resolves a relative
  `mtllib` under `-w`; convert the input mesh to a geometry-only PLY (TextureMesh
  regenerates UVs anyway).
- **Radiometry.** Seam leveling defaults off, and 16-bit→8-bit uses a fixed
  bit-shift (uniform across frames), so texture intensities stay a faithful copy
  of the source modality. A single camera only covers part of the surface;
  unseen faces take `--empty-color`.
