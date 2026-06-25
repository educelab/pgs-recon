# pgs-recon

Photogrammetry reconstruction pipeline orchestrating OpenMVG (SfM) and OpenMVS
(MVS). This glossary fixes the project-specific terms whose ambiguity tends to
cause mistakes; it is not a description of the code.

## Language

**Capture position**:
A physical rig pose, identified by the position index in the
`{prefix}_{camera}_{position}_{capture}` filename. The same capture position is
shared across all cameras and all modalities of a scan, so it is the key that
aligns an image from one set to a solved view in another.
_Avoid_: shot, frame, station

**Camera**:
One of the fixed physical cameras in the rig (the `{camera}` index). Each camera
has its own solved intrinsic. Distinct from a **view**.
_Avoid_: sensor, lens

**View**:
A single solved image in an SfM scene — one camera at one capture position, with
a pose and an intrinsic reference. 840 views = 5 cameras × 168 capture positions.
_Avoid_: photo, frame

**Modality**:
An alternate imaging condition captured at the same capture positions as an
existing camera (e.g. IR940 illumination vs monochrome white light). A modality
reuses that camera's solved poses and intrinsics; only the pixels differ.
_Avoid_: channel, mode, lighting

**Solved frame** (a.k.a. MVS frame):
The arbitrary coordinate frame produced by the SfM solve
(`mvg/recon_dir/sfm_data.bin`). The dense cloud, the mesh, and the MVS scene all
live here. This is the frame anything fed to OpenMVS must be in.
_Avoid_: world frame, scene frame

**Rig-prior import**:
The initial imported scene (`mvg/sfm_data.json`) carrying the rig's prior poses
in physical units (camera centers in the hundreds). NOT the solved frame — do
not feed it to OpenMVS expecting the mesh to align.
_Avoid_: input scene, initial sfm

**Centering transform**:
The 4×4 transform in `*_center.tfm.npy` that maps the un-centered solved-frame
mesh to the centered/scaled mesh variants. The bare reconstruction `.obj` is
un-centered (solved frame); the transform is stored, not pre-applied to it.
_Avoid_: alignment, normalization

**Localization**:
Resectioning a *new* image into an existing solved scene by matching its 2D
features against the reconstruction's 3D structure (OpenMVG
`SfM_Localization`), recovering that image's pose (and, for an uncalibrated
camera, intrinsic) in the solved frame. Distinct from the **rig-prior import**:
the scene is already solved and is not re-solved. This is what `pgs-calibrate`
does. _Avoid_: registration, alignment, SfM

**Calibration** (camera calibration artifact):
The reusable single-view `*_calibration.json` emitted by `pgs-calibrate`: one
localized view carrying a pose + intrinsic in the solved frame. Because a
physical camera position is shared across its modalities, the calibration is
solved once and reused to texture with each modality. NOT the ChArUco/board
calibration of `educelab`, and NOT the autoscale step. _Avoid_: calib, intrinsics

**Overhead camera**:
A camera that imaged the object but was NOT part of the rig reconstruction
(e.g. a top-down registration camera). It has no solved pose, so it must be
**localized** before its images can texture the mesh. Contrast a **modality**,
which reuses an existing rig camera's solved poses. _Avoid_: external camera, witness camera

## Retexture output layout

`pgs-retexture` does not own an output directory. It writes **into an existing
`pgs-recon` output** as a continuation of that run: its artifacts land in the
recon's own `mvg/` and `mvs/` as siblings of the recon's files, and it never
overwrites anything pre-existing. Correctness comes from naming, not a guard —
the same convention `pgs-recon` itself relies on.

**Working dir** (`--working-dir`/`-w`): where artifacts are written. Defaults to
`--recon-dir`. (Earlier versions carved a separate `retexture/<name>/` subtree;
that is gone.)

**Stem**: the namespacing token prefixed onto every retexture artifact, so they
coexist with the recon's files and with other retexture runs. It is **derived,
not supplied** — there is no `--name` flag. The stem is the `--output-mesh`
filename stem if that flag is given, otherwise the modality input's name (the
image directory name in the default mode, the image stem in `--calibration`
mode).

**`--output-mesh`/`-o`**: the exact path + filename of the final textured mesh.
Its extension sets the output format (overrides `--file-type`, with a warning).
The mesh, its `.mtl`, and the texture image travel together to that path, all
renamed to the target stem with `map_Kd` patched, so the deliverable is
self-contained anywhere. Omitted, the final mesh defaults to `mvs/<stem>.obj`
(or the chosen `--file-type`) — a sibling of the recon's `mvs/<recon>.obj`.

Resulting layout (stem `IR940`, recon name `scroll`):

    recon/
      mvg/  IR940_sfm_full.json          # exported SfM (no longer a generic name)
            IR940_sfm.json               # filtered/re-pointed to the modality
      mvs/  IR940_modality/              # 8-bit modality images
            IR940_undistorted_images/    # NOT the recon's shared undistorted_images/
            IR940_scene.mvs
            IR940_input.ply              # staged copy of the mesh being textured
            IR940.obj                    # final, beside the recon's scroll.obj
      IR940_retexture_metadata.json      # sidecar; recon's metadata.json untouched
      <datetime>_IR940_retexture_config.txt

The only hard collision the convention removes is the **undistorted-images
dir**, which both tools otherwise name `undistorted_images/`. The
**projective-UV** path (`--calibration` without `--use-openmvs`) is OBJ-only, so
a non-`.obj` `--output-mesh` there warns and forces `.obj`.
