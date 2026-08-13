# Multi-camera capture retexture

**Status**: not implemented. This is a bootstrap document for the change, not a
record of a decision made.

## The gap

`pgs-retexture` serves two modes that are different in kind (see `CONTEXT.md`,
**Capture retexture** and **Localized-camera retexture**):

| | Capture retexture (default) | Localized-camera retexture (`--calibration`) |
|---|---|---|
| Where the images come from | another **capture** of the same scan | a camera that was never in the solve |
| How a pose is found | `(camera, position)` match against solved views | the camera's **calibration**, via `pgs-calibrate` |
| Cameras involved | as many as the capture has | exactly one, inherently |

**Both modes are currently restricted to one camera.** For localized-camera
retexture that is inherent. For capture retexture it is a limitation: nothing
about matching on `(camera, position)` requires a single camera, and with the
whole rig present you re-point *every* view rather than filtering to one.

The restriction dates from when an alternate capture *was* a single camera — a
`scan.capture_settings` entry like `{"name": "Center+IR940", "cameras": [3]}`.
Scans now carry captures in which the whole rig fires under a different
illumination, which is the same shift that motivated `--import-capture` on the
importer. The two features are independent: `--import-capture` chooses the
capture the mesh is *solved* from, this change chooses the capture it is
*textured* from, and any pair is legitimate (solve capture 2, texture capture 3).

## What the code does today

Default-mode flow, the legacy-mode `else` branch of `_main()` in
`pgs_recon/apps/retexture.py` (symbols rather than line numbers throughout — this
is a spec for future work, and the lines will move):

1. `index_modality_images(image_dir, camera_index)`
   Groups files by `(camera, position)` into `by_cam: Dict[cam, Dict[pos, Path]]`,
   then **throws away every camera but one**, returning `(camera_index, pos_map)`
   where `pos_map: Dict[pos, Path]`.
   - Infers the camera when the directory holds exactly one (the
     `if camera_index is None` branch).
   - `sys.exit`s when the directory holds several and `--camera-index` is absent
     ("contains multiple camera indices"). This is the mode's single-camera
     constraint surfacing as an error, not a collision policy.
   - Warns and keeps the last file when two images claim the same
     `(camera, position)` ("Multiple modality images for camera") — a genuine
     duplicate-slot condition, which happens when the directory holds more than
     one capture.
2. `convert_modality_images(pos_map, out_dir, bit_shift)`
   8-bit conversion, returns `Dict[pos, str]` (output basename). Output names are
   `f'{src.stem}.jpg'`, so they already carry the camera index and **will not
   collide across cameras**. Only the key type is wrong.
3. `sfm_to_json(...)` then `filter_sfm_for_camera(...)`
   Keeps views whose parsed camera equals `camera_index`, re-points each at
   `pos_to_name[pos]`, drops orphan intrinsics/poses, and `sys.exit`s with
   "No views matched camera" if nothing survives. Views with no modality image
   are dropped with a warning and counted in `missing`.
4. `mvg_to_mvs` → `mvs_texture`, unchanged by this work.

## What has to change

Keying is the whole job: every `pos` key becomes `(cam, pos)`.

- `index_modality_images` → return the full `by_cam`, or a flat
  `Dict[(cam, pos), Path]`, plus the set of cameras found. Keep the duplicate
  `(camera, position)` warning — it still means "you pointed at more than one
  capture", which is still a user error. Drop the multi-camera `sys.exit`.
- `convert_modality_images` → key by `(cam, pos)`. Body is otherwise untouched.
- `filter_sfm_for_camera` → `filter_sfm_for_cameras`, taking a camera *set* (or
  `None` for "every camera present in both"). The orphan-intrinsic and
  orphan-pose pruning already generalizes — `fix_polymorphic_registration` takes
  a list. The `missing` warning becomes per `(camera, position)`.
- `--camera-index/-k` → accepts multiple values, defaulting to the
  intersection of the solve's cameras and the texturing capture's. Keeping the
  single-value spelling working matters; it is in released docs.

## Open questions for whoever does it

1. **Default when the sets differ.** Solve capture has cameras `{0..4}`,
   texturing capture has `{1,3}`. Texture from `{1,3}` and drop the rest, or
   refuse and make the user say so? Dropping loses coverage silently; refusing is
   noisy for the common case. Leaning toward: texture the intersection, log the
   dropped cameras at warning level.
2. **Is more cameras actually better here?** More views means more texture
   candidates and more seams to blend. `TextureMesh` picks per-face; nobody has
   measured whether a five-camera IR texture beats a one-camera one on these
   scans. Worth one experiment before assuming the multi-camera path is the
   preferred one rather than merely the possible one.
3. **Exposure uniformity across cameras.** `convert_modality_images` applies a
   fixed bit-shift uniformly to preserve relative radiometry *within* a camera
   (see its docstring). Across cameras of one capture the exposure and gain are
   shared (`capture_settings[i]['exp']`, `['gain']`), so a single shift should
   still be right — but this is an assumption, not something the code checks.
4. **The stem.** `stem` derives from the modality *directory* name (assigned in
   `_main()`, in the `else` of the `output_mesh` suffix check), and the artifact
   layout in `CONTEXT.md` assumes one stem per run. Multi-camera
   does not change that, but confirm nothing downstream assumes one camera per
   stem.
5. **Localized-camera mode must stay single-camera.** Whatever the signature
   change, `--calibration` keeps its one-image contract. The two modes share a
   `main()`; they should not come to share this parameter.

## Related

- `CONTEXT.md` — **Capture**, **Capture retexture**, **Localized-camera
  retexture**, **Modality**.
- `pgs_recon/apps/retexture.py` module docstring — states the limit as a limit
  and points here.
- `--import-capture` on `pgs-recon` — the sibling feature on the importer;
  `pgs_recon/pgs_data.py`.
- ADR 0001 (`docs/adr/0001-retexture-with-alternate-modality.md`) — why
  `pgs-retexture` exists at all.
- ADR 0002 (`docs/adr/0002-calibrate-new-camera-for-retexture.md`) — why the
  `--calibration` mode exists.
