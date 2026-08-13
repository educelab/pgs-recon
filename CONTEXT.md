# pgs-recon

Photogrammetry reconstruction pipeline orchestrating OpenMVG (SfM) and OpenMVS
(MVS). This glossary fixes the project-specific terms whose ambiguity tends to
cause mistakes; it is not a description of the code.

## Language

**Capture**:
One slot in a scan's per-position image stack — the `{capture}` index in the
`{prefix}_{camera}_{position}_{capture}` filename, described by the matching
`scan.capture_settings` entry (its lights, exposure, gain, participating cameras,
and a free-text `name`). Every capture spans the same capture positions, but each
declares its own subset of cameras, so two captures need not contain the same
images — one may be all five rig cameras under white light, the next a single
camera under IR940. A capture is usually a **modality**, but may also be a
*derived* product of other captures (an RGB composed from bands), which is why
the term names the scan's slot and not what fills it. A capture is identified by
its index; the `name` is a label, not a key.
_Avoid_: capture group, cap, band, pass, exposure

**Capture position**:
A physical rig pose, identified by the position index in the
`{prefix}_{camera}_{position}_{capture}` filename. The same capture position is
shared across all cameras and all captures of a scan, so it is the key that
aligns an image from one set to a solved view in another. Distinct from a
**capture**, with which it shares a word: a capture spans every capture position.
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
reuses that camera's solved poses and intrinsics; only the pixels differ. This is
the reconstruction-level term: it says how an image set is *consumed*, whereas a
**capture** is the scan-level slot the images arrive in. A capture that is an
imaging condition is a modality; one that is a derivative of other captures is
not.
_Avoid_: channel, mode, lighting

**Capture retexture**:
Re-texturing an existing mesh with a different **capture** of the same scan —
the same rig cameras at the same capture positions, under different
illumination. Correspondence between an image and a solved view is by
`(camera, position)`, so every camera present in both the solved capture and the
texturing capture can contribute. The solve and the texture are two captures of
one scan, and which capture each came from is the only thing that distinguishes
them — so the texturing capture is named by its index, like any other capture,
not by having been separated into a directory of its own. Distinct from
**localized-camera retexture**, with which it shares a tool.
_Avoid_: modality swap, remap, reprojection

**Localized-camera retexture**:
Re-texturing an existing mesh with images from a camera that was never part of
the solve — an **overhead camera** brought into the solved frame by
**localization**. There are no capture positions to correspond, because the
camera's pose comes from its **calibration** rather than from the rig, so a
single image can texture the whole mesh. Inherently one camera; that is what
makes it different in kind from **capture retexture**, which is inherently as
many cameras as the capture has.
_Avoid_: calibration retexture, external retexture

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

**Calibration** (a family of senses):
Three steps here all calibrate something, and naming which one is meant is the
whole job of the word:

- the **SfM solve** calibrates the rig's cameras against the scene, recovering
  their poses (and intrinsics) from the images themselves — the `sfm` and
  `robust` stages;
- **autoscale** calibrates the scene's *scale*, fixing the solved frame's
  arbitrary unit against a marker of known physical size — the `autoscale`
  stage, via `pgs-global-scaler`;
- **`pgs-calibrate`** calibrates a *new* camera into an already-solved scene, by
  **localizing** it.

Unqualified, and especially as an artifact, "calibration" means the third:
`*_calibration.json`, the reusable single-view calibration emitted by
`pgs-calibrate`, one localized view carrying a pose + intrinsic in the solved
frame. Because a physical camera position is shared across its modalities, that
calibration is solved once and reused to texture with each modality.

NOT the ChArUco/board calibration of `educelab`.
_Avoid_: calib, intrinsics

**Overhead camera**:
A camera that imaged the object but was NOT part of the rig reconstruction
(e.g. a top-down registration camera). It has no solved pose, so it must be
**localized** before its images can texture the mesh. Contrast a **modality**,
which reuses an existing rig camera's solved poses. _Avoid_: external camera, witness camera

**Stage**:
One of the thirteen steps of a reconstruction, each exactly one binary
invocation, named for what it does rather than for the binary (`densify`,
`refine`, `texture`). A stage is the unit a run can start and stop at, and the
unit whose completion is recorded.
_Avoid_: step, phase, pass, task

**Pipeline shape**:
Which stages a given reconstruction consists of at all — declared by the enable
flags (`--mvs-densify`, `--mvs-refine`, ...) of the run in progress. A later run
may legitimately change it, which re-runs whatever the change invalidates
(warning about any of it left outside the range). Distinct from the **range**:
the contiguous window of that shape a single run executes (`--from`/`--to`). The
shape says what the reconstruction *is*; the range says what this job *does*.
_Avoid_: pipeline, stage list, workflow

**Role**:
A semantic artifact slot a stage consumes or produces — `sfm`, `features`,
`matches`, `matches_filtered`, `view_pairs`, `colorized`, `scene`, `cloud`,
`mesh`. Roles are how a resumed job finds its inputs, and they are *rebound* as a
run proceeds, so a role names the slot and never the file in it. Declared per
stage in `stages.STAGE_IO`. Distinct from the **artifact name**, which also
records which stage produced it.
_Avoid_: artifact type, slot, key

**Artifact name**:
An intermediate's filename, `<stage>_<role>.<ext>` — the stage that produced it
and the role it fills, so it does not encode which *other* stages ran
(`refine_mesh.ply` whether or not densify is in the shape). The convention
replaces name *chaining*, so it applies exactly where chaining occurred: a name
already independent of the shape keeps it, including every name a binary chooses
for itself (`mvg/sfm_data.json`, `recon_dir/sfm_data.bin`) and the final
deliverable `mvs/<name>.<ext>`, which is a user-facing contract. An artifact name
is only ever *written* — a resumed job finds an existing artifact through the
**manifest**, never by rebuilding its name
([ADR 0006](./docs/adr/0006-stage-named-artifacts.md), which replaced the chained
names an output directory used to carry: `scene_dense_refine.ply`). One stage
cannot spell out two roles: `densify` hands `DensifyPointCloud` a single `-o`
from which it names both files it writes, so the pair is `densify.mvs` /
`densify.ply` — stem for the stage, suffix for the role.
_Avoid_: path, key, stem, filename

**Binding**:
Which stage currently owns a role, and where that artifact is. Roles are
*rebound* as a run proceeds — `sfm` is produced by `import`, then rebound by the
`sfm`, `robust` and `autoscale` stages — so a binding is only meaningful at a
point in the pipeline, and the stage graph is derived from the shape rather than
declared. A binding's path is unknown while its producing stage is **dirty**.
_Avoid_: chain entry, path, mapping

**Dirty**:
A stage whose recorded result no longer holds, and so must run: no record or a
record that is not `complete`, its own arguments changed, a stage producing
something it consumes is dirty, or a role it consumes is now bound to a
different path than it recorded. Dirtiness is **computed** each run, never
stored — which is why a range that leaves later stages dirty is a warning rather
than an error. Never inferred from what is on disk.
_Avoid_: stale, invalid, out-of-date, needs-rebuild

**Manifest**:
The run's `pgs-recon.json`, the single record of what a reconstruction has
finished: each stage's status, the artifacts it produced, and the effective
arguments it ran with. It is what makes a run resumable and what a later job
consults instead of being told again.
_Avoid_: state file, checkpoint, log, metadata

**Effective arguments**:
The arguments a stage actually ran with, after the manifest's recorded values
have been merged with this invocation's and out-of-range overrides discarded.
Distinct from what was typed on the command line — the manifest records the
effective values, never the ignored ones.
_Avoid_: args, options, config, defaults

**Interface scene** (`MVSI`):
An OpenMVS `.mvs` written in the Boost-independent interface format — cameras,
poses, and image paths, with its own versioned header. It is portable across
Boost versions, compilers, and architectures, unlike the Boost project format
(`MVS\0`) OpenMVS otherwise writes. Every scene handed between stages is an
interface scene, and all geometry travels beside it as `.ply`.
_Avoid_: mvs file, scene file, project, archive

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
filename stem if that flag is given, otherwise the modality input's name: the
scan directory's name plus the capture it textures from (`scan_c3`) in capture
retexture, the image stem in `--calibration` mode. The capture belongs in the
stem because one scan directory is the input for all of its captures, so the
directory name alone would name every capture's artifacts identically.

**`--output-mesh`/`-o`**: the exact path + filename of the final textured mesh.
Its extension sets the output format (overrides `--file-type`, with a warning).
The mesh, its `.mtl`, and the texture image travel together to that path, all
renamed to the target stem with `map_Kd` patched, so the deliverable is
self-contained anywhere. Omitted, the final mesh defaults to `mvs/<stem>.obj`
(or the chosen `--file-type`) — a sibling of the recon's `mvs/<recon>.obj`.

Resulting layout (stem `scan_c3` — capture 3 of the scan dir `scan/`, recon name
`scroll`):

    recon/
      mvg/  scan_c3_sfm_full.json        # exported SfM (no longer a generic name)
            scan_c3_sfm.json             # filtered/re-pointed to the modality
      mvs/  scan_c3_modality/            # 8-bit modality images
            scan_c3_undistorted_images/  # NOT the recon's shared undistorted_images/
            scan_c3_scene.mvs
            scan_c3_input.ply            # staged copy of the mesh being textured
            scan_c3.obj                  # final, beside the recon's scroll.obj
      scan_c3_retexture.json             # sidecar; recon's manifest untouched
      <datetime>_scan_c3_retexture_config.txt

`--camera-index` deliberately stays out of the stem: it selects a subset of one
capture's cameras rather than a different deliverable, so encoding it would make
the ordinary whole-capture name depend on how the flag was spelled. Two runs of
one capture with different camera subsets therefore share a stem and overwrite —
legal, and reported path by path before it happens; `--output-mesh` is how they
are kept side by side.

The sidecar config and the manifest are not the same record. The config is
**replayed** — every line is re-parsed as if typed — so it carries only what was
asked for, plus the capture (pinned, so a scan that later grows another still
replays the same one). The manifest is **read by nobody**, so it states what the
run resolved: `capture` and `cameras` beside the `parsed` arguments. Writing the
derived camera set into the config instead would silently narrow a replay that
overrides `--capture`.

The only hard collision the convention removes is the **undistorted-images
dir**, which both tools otherwise name `undistorted_images/`. The
**projective-UV** path (`--calibration` without `--use-openmvs`) is OBJ-only, so
a non-`.obj` `--output-mesh` there warns and forces `.obj`.
