# Re-texture a mesh with an alternate modality by rebuilding an MVS scene

## Context

We need to re-texture an existing reconstructed mesh using an alternate imaging
modality (e.g. IR940) captured at the same capture positions as the rig cameras.
OpenMVS (verified against v2.3.0) has **no native "texture with a different image
set" option**, so `pgs-retexture` reuses the original SfM solution, filtered to
the cameras the modality images cover and re-pointed at those images, then runs
`openMVG2openMVS` + `TextureMesh`.

## Decision

Rebuild a minimal MVS scene rather than hand-edit the binary `.mvs` or wait for
upstream support. The flow: convert modality images to 8-bit → export the solved
`SfM_Data` to JSON (views/intrinsics/extrinsics only) → keep the views whose
`(camera, position)` a modality image covers, re-pointed at that image →
`openMVG2openMVS` (undistorts with the original intrinsics) → `TextureMesh`
against the existing mesh.

The texturing images are one *capture* of a PGS scan directory, so every camera
that fired in it contributes; `--camera-index` narrows that set when only some
are wanted. The single-camera case is not special — it is the same flow over a
capture that fired one camera, or the `--calibration` mode below, where a camera
that was never in the solve is placed in the solved frame by `pgs-calibrate` and
contributes one view at one pose.

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
  Views need no such repair: they serialize without a `polymorphic_name`, so
  dropping any view cannot orphan a type.
- **Mesh format.** OpenMVS's OBJ reader is strict and mis-resolves a relative
  `mtllib` under `-w`; convert the input mesh to a geometry-only PLY (TextureMesh
  regenerates UVs anyway).
- **Radiometry.** Seam leveling defaults off, 16-bit→8-bit uses a fixed bit-shift
  (uniform across frames, and across the cameras of a capture, since exposure and
  gain belong to the capture), and an image already 8-bit in a format the
  toolchain reads is copied rather than re-encoded — so texture intensities stay
  a faithful copy of the source modality. Faces no contributing camera saw take
  `--empty-color`.
- **Inferred values are recorded, not replayed.** The capture and camera set a
  run resolved to go in the run's manifest. Only the capture is written back into
  the sidecar config: the camera set is derived from it, so replaying a config
  with `--capture` overridden would otherwise silently texture from the previous
  capture's cameras.
