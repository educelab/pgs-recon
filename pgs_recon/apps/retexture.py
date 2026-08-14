"""Re-texture an existing reconstructed mesh using an alternate imaging
modality (e.g. IR940) captured at the same camera positions as the cameras of
the original reconstruction.

This tool serves two modes that are different in kind, and only one of them is
inherently single-camera:

  - **Capture retexture** (default mode): the texturing images are another
    *capture* of the same scan -- the same rig, the same capture positions,
    different illumination. Correspondence is by ``(camera, position)``, so
    every camera present in both captures contributes.
  - **Localized-camera retexture** (``--calibration``): the images come from a
    camera that was never in the solve, placed in the solved frame by
    ``pgs-calibrate``. One camera, one pose, no positional correspondence --
    inherently single-camera, and unaffected by everything below.

OpenMVS has no native "texture with a different image set" option (verified
against OpenMVS v2.3.0), so this tool rebuilds a minimal MVS scene from the
original SfM solution restricted to the texturing capture's cameras, with those
views re-pointed at the modality images. The pipeline is:

  1. (optional) Convert 16-bit modality images to 8-bit with a fixed linear
     map (bit-shift), uniform across all frames to preserve relative radiometry
     and keep the merged texture seamless.
  2. Convert the solved OpenMVG SfM_Data to JSON (views/intrinsics/extrinsics
     only) and filter it to the requested cameras, re-pointing each view at the
     modality image sharing its ``(camera, position)``.
  3. openMVG2openMVS on the filtered scene -> undistorts the modality images
     with the original camera intrinsics and writes a new MVS scene.
  4. TextureMesh the *existing* mesh against that scene.

The mesh and the regenerated scene must share a coordinate frame. This runs as
an optional stage *after* a normal ``pgs-recon`` run: point ``--recon-dir`` at
that run's output directory and both inputs are taken from it via its
manifest — the solved SfM fed to openMVG2openMVS (after any
robust/autoscale step, NOT the rig-prior import) and the textured mesh as
output by TextureMesh (before any centering transform). Either can be supplied
directly instead (``--sfm-data``, ``--mesh``), and in ``--calibration`` mode the
calibration .json *is* the scene, so the reconstruction's SfM is never read;
each artifact is only demanded of ``--recon-dir`` when the run actually reads it.

CAPTURE RETEXTURE TAKES A PGS SCAN DIRECTORY, not a directory of pre-separated
modality images: ``-i`` must hold a ``metadata.json``, and ``--capture`` names
which of the captures in it to texture from (inferred when the scan holds only
one). The metadata is what says how to read the directory -- ``scan.file_prefix``
and ``scan.format`` select the images, exactly as ``pgs_data.import_pgs_scan``
does for the solve, so two images of one ``(camera, position)`` in different
formats or from different captures cannot be confused for each other.

The correspondence between a modality image and a camera pose in the SfM solution
is established *entirely by filename* — there is no EXIF, ordering, or geometric
fallback. Names must match ``{prefix}{camera}_{position}[_{capture}]``, and
matching is keyed on ``(camera, position)``. Concretely:

  - Modality images are indexed by ``(camera, position)`` for the selected
    capture; every camera in it is textured from, unless ``--camera-index``
    narrows the set.
  - Each SfM view's stored filename is parsed with the same scan prefix (any
    extension, since a solve may have been imported from converted copies), and
    views are re-pointed at the modality image sharing their
    ``(camera, position)``.

The two image sets need not cover the same cameras or positions. A solved view
with no modality image is simply not textured from — an expected consequence of a
capture that fired fewer cameras. A modality image with no solved view is warned
about loudly: it was never calibrated, so nothing can place it.

This means the original reconstruction must itself have been run on PGS-named
images (e.g. imported via ``pgs-import`` / ``init_sfm_pgs``). If the SfM views
carry arbitrary filenames (e.g. a generic EXIF-based import), none will parse and
the run aborts saying so.
"""
import atexit
import json
import logging
import shutil
import sys
from datetime import datetime as dt, timezone as tz
from pathlib import Path
from typing import Dict, Optional, Sequence

import configargparse
import imageio.v3 as iio
import numpy as np

from pgs_recon import toolchain
from pgs_recon.openmvg import mvg_to_mvs
from pgs_recon.openmvs import mvs_texture
from pgs_recon.stages import write_manifest
from pgs_recon.toolchain import Recorder, resolve_exe, run
from pgs_recon.utility import ToolFailed
from pgs_recon.utils.apps import setup_logging
from pgs_recon.utils.images import (drop_alpha, prepare_8bit_image, read_srgb,
                                    to_uint8, to_uint8_shifted)
from pgs_recon.utils.recon_dir import (
    load_manifest,
    resolve_solved_sfm,
    resolve_textured_mesh,
)
from pgs_recon.utils.scan_names import parse_scan_name, parse_view_name
from pgs_recon.utils.sfm_json import (
    bare_polymorphic_id,
    camera_from_calibration,
    fix_polymorphic_registration,
    transform_sfm_extrinsics,
)

logger = logging.getLogger(__name__)


def index_modality_images(scan_dir: Path, capture: Optional[int],
                          cameras: Optional[Sequence[int]]):
    """Index one capture of a PGS scan directory by ``(camera, position)``.

    ``scan_dir`` must be a PGS scan directory: its ``metadata.json`` is what says
    which files are images (``scan.file_prefix``, ``scan.format``) and what the
    captures in it are, so this reads the directory the same way the importer
    reads it for the solve. Selection is by *filename*, matching
    ``pgs_data.select_capture``: a capture the metadata does not declare is a
    warning (a derived capture need not be declared), and an empty selection is
    the only hard failure.

    ``capture`` is inferred when the scan holds exactly one; several without a
    choice is an error, because texturing from an arbitrary one of them is
    silently wrong. ``cameras`` *restricts* the capture's cameras rather than
    asserting them: a requested camera the capture does not hold is warned about
    loudly and skipped, and only an empty result is fatal.

    Returns ``(capture, prefix, {(camera, position): path})``.
    """
    meta_path = scan_dir / 'metadata.json'
    if not meta_path.is_file():
        sys.exit(f'No metadata.json in {scan_dir}. Capture retexture reads a '
                 f'PGS scan directory (the same one the reconstruction was '
                 f'imported from), and selects the texturing capture from it '
                 f'with --capture.')
    # The metadata is what says how to read the directory, so a malformed one is
    # reported against the file rather than as a KeyError from inside the read.
    try:
        scan_meta = json.loads(meta_path.read_text())
        prefix = scan_meta['scan']['file_prefix']
        ext = scan_meta['scan']['format'].lower()
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
        sys.exit(f'Could not read scan.file_prefix / scan.format from '
                 f'{meta_path}: {e}')

    by_capture: Dict[int, Dict[tuple, Path]] = {}
    matched = 0
    for p in sorted(scan_dir.glob(f'{prefix}*.{ext}')):
        matched += 1
        cam, pos, cap = parse_scan_name(p.name, prefix, ext)
        if cap is None or cam is None or pos is None:
            logger.warning(f'Skipping unrecognized filename: {p.name}')
            continue
        images = by_capture.setdefault(cap, {})
        existing = images.get((cam, pos))
        if existing is not None:
            logger.warning(f'Multiple capture {cap} images for camera {cam} '
                           f'position {pos}: keeping {p.name}, '
                           f'dropping {existing.name}')
        images[(cam, pos)] = p

    # An empty selection has two causes, and they point at different fixes: no
    # file matched the metadata's prefix/format, or files did and none of their
    # names follow the convention.
    if not by_capture:
        if matched:
            sys.exit(f'None of the {matched} images matching {prefix}*.{ext} in '
                     f'{scan_dir} follow the naming convention '
                     f'{prefix}{{camera}}_{{position}}[_{{capture}}].{ext}')
        sys.exit(f'No images matching {prefix}*.{ext} in {scan_dir}')

    if capture is None:
        if len(by_capture) > 1:
            sys.exit(f'{scan_dir} holds captures {sorted(by_capture)}; specify '
                     f'which one to texture from with --capture')
        capture = next(iter(by_capture))
    elif capture not in by_capture:
        sys.exit(f'No images for capture {capture}. Captures present in the '
                 f'filenames: {sorted(by_capture)}')

    # The metadata names a capture; the filenames are what select it. A capture
    # it does not declare is legitimate (a derived one need not be), so warn.
    settings = scan_meta['scan'].get('capture_settings')
    declared = settings is not None and 0 <= capture < len(settings)
    if settings is not None and not declared:
        logger.warning(f'Texturing from capture {capture}, but the scan declares '
                       f'{len(settings)} capture(s): '
                       f'{[c.get("name", i) for i, c in enumerate(settings)]}')

    images = by_capture[capture]
    available = sorted({cam for cam, _pos in images})
    if cameras is not None:
        requested = sorted(set(cameras))
        absent = [c for c in requested if c not in available]
        if absent:
            logger.warning(f'--camera-index requested camera(s) {absent}, which '
                           f'capture {capture} does not contain (it has '
                           f'{available}); they are skipped')
        images = {k: v for k, v in images.items() if k[0] in set(requested)}
        if not images:
            sys.exit(f'None of the requested cameras {requested} have images in '
                     f'capture {capture} (it has {available})')
        available = sorted({cam for cam, _pos in images})

    label = ''
    if declared and settings[capture].get('name'):
        label = f' ({settings[capture]["name"]})'
    logger.info(f'Texturing from capture {capture}{label}: {len(images)} images '
                f'from camera(s) {available}')
    return capture, prefix, images


def convert_modality_images(img_map: Dict[tuple, Path], out_dir: Path,
                            bit_shift: int) -> Dict[tuple, str]:
    """Ensure every modality image is an 8-bit file usable by openMVG/OpenMVS.

    16-bit images are mapped to 8-bit with a fixed bit-shift (>> bit_shift),
    applied identically to every frame. Float images are scaled from an assumed
    [0, 1] range by a fixed factor. All conversions are uniform across frames to
    preserve relative radiometry — including *across* cameras, since the exposure
    and gain of a capture are properties of the capture, not of each camera in
    it.

    An image that is already 8-bit in a format the toolchain reads is copied
    through byte-for-byte rather than re-encoded: ``-i`` is a scan directory, so
    a scan captured as JPEG would otherwise pay a second lossy generation to say
    nothing new. Returns ``(camera, position)`` -> output filename (basename).

    Reads through :func:`read_srgb` rather than OpenCV, which would take a
    CIELab TIFF channel-for-channel as BGR. A decoded image has no spare high
    bits to drop, so the shift does not apply to it -- it is already sRGB.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_names: Dict[tuple, str] = {}
    copied = 0
    for key, src in sorted(img_map.items()):
        try:
            image = read_srgb(src)
        except (OSError, ValueError) as e:
            sys.exit(f'Could not read modality image: {src}: {e}')
        # Read first regardless: a .png says nothing about its bit depth. Only
        # these formats, so a passed-through file cannot be a (TIFF-only) Lab
        # one reaching the toolchain undecoded.
        if image.dtype == np.uint8 and src.suffix.lower() in ('.jpg', '.jpeg',
                                                              '.png'):
            shutil.copy2(src, out_dir / src.name)
            out_names[key] = src.name
            copied += 1
            continue
        # A 16-bit RGBA PNG misses the copy above, so the JPEG writer -- which
        # refuses a fourth channel -- is where it would land. Channel count is
        # otherwise left alone: a greyscale modality image has no reason to
        # triple in size.
        pixels = drop_alpha(image.pixels)
        img = (to_uint8_shifted(pixels, bit_shift)
               if image.dtype == np.uint16 and not image.decoded
               else to_uint8(pixels))
        # The source stem carries camera, position and capture, so output names
        # cannot collide across the cameras of one capture.
        out_name = f'{src.stem}.jpg'
        iio.imwrite(out_dir / out_name, img, quality=100)
        out_names[key] = out_name
    logger.info(f'Prepared {len(out_names)} 8-bit modality images in {out_dir} '
                f'({copied} copied unchanged, {len(out_names) - copied} '
                f'converted)')
    return out_names


def repoint_calibration(calibration_json: Path, image: Path,
                        out_json: Path) -> None:
    """Re-point a single-view ``pgs-calibrate`` calibration at ``image``.

    The calibration carries one localized view (pose + intrinsic) in the solved
    frame. Texturing a different modality from the same physical pose is just a
    pixel swap: set the lone view's filename to ``image`` and root the scene at
    its directory. The modality image MUST match the calibrated intrinsic's pixel
    dimensions (same camera, same resolution), or undistortion/projection in
    openMVG2openMVS would be wrong; this is validated and aborts on mismatch.

    OpenMVS TextureMesh rejects a single-image scene ("invalid project",
    verified against v2.3.0), so an inert 1x1 dummy view is appended to satisfy
    its >=2 image requirement. A 1x1 image has effectively zero resolution, so
    OpenMVS's view-quality ranking never selects it for any face (verified: it
    textures 0 faces) — it only pads the image count, and costs ~nothing on disk
    (vs. copying the full-size modality image). The dummy gets its own 1x1
    intrinsic so openMVG2openMVS undistorts it trivially.

    The dummy is written as a tiny JPEG (not PNG): openMVG2openMVS undistorts it
    to an output file of the same format, and its libpng writer fails on a 1x1
    image ("bad parameters to zlib"), whereas the libjpeg path handles it fine.

    ``image`` must be a path inside a writable work directory: a sibling file
    ``__retex_dummy__.jpg`` is created there. Do not pass a path into a
    read-only source tree.
    """
    data = json.loads(calibration_json.read_text())
    views = data.get('views', [])
    intrinsics = data.get('intrinsics', [])
    if len(views) != 1 or len(data.get('extrinsics', [])) != 1 \
            or len(intrinsics) != 1:
        sys.exit(f'Calibration {calibration_json} must contain exactly one '
                 f'view, pose and intrinsic; is this a pgs-calibrate output?')
    vd = views[0]['value']['ptr_wrapper']['data']

    # Header only: the dimensions are all this needs, and a modality capture is
    # large enough that decoding one to read two of its fields is not free.
    try:
        h, w = iio.improps(image).shape[:2]
    except (OSError, ValueError, IndexError) as e:
        sys.exit(f'Could not read modality image: {image}: {e}')
    if (w, h) != (vd['width'], vd['height']):
        sys.exit(f'Modality image {image.name} is {w}x{h} but the calibration '
                 f'was solved for {vd["width"]}x{vd["height"]}. All modalities '
                 f'must share the calibrated camera\'s resolution.')

    vd['filename'] = image.name
    vd['local_path'] = ''

    # Append an inert 1x1 dummy view (with its own 1x1 intrinsic + a copied
    # pose) so OpenMVS sees a >=2 image scene. New keys/cereal ptr ids are
    # placed above everything already present to avoid collisions.
    dummy_img = image.with_name('__retex_dummy__.jpg')
    iio.imwrite(dummy_img, np.zeros((1, 1, 3), np.uint8))
    new_view_key = max(v['key'] for v in views) + 1
    new_intr_key = max(i['key'] for i in intrinsics) + 1
    next_ptr = max(e['value']['ptr_wrapper']['id']
                   for e in views + intrinsics) + 1

    dummy_view = json.loads(json.dumps(views[0]))  # deep copy
    dummy_view['key'] = new_view_key
    dvd = dummy_view['value']['ptr_wrapper']['data']
    dvd['id_view'] = new_view_key
    dvd['id_pose'] = new_view_key
    dvd['id_intrinsic'] = new_intr_key
    dvd['filename'] = dummy_img.name
    dvd['local_path'] = ''
    dvd['width'] = dvd['height'] = 1
    dummy_view['value']['ptr_wrapper']['id'] = next_ptr
    data['views'].append(dummy_view)

    dummy_intr = json.loads(json.dumps(intrinsics[0]))
    dummy_intr['key'] = new_intr_key
    dummy_intr['value']['ptr_wrapper']['id'] = next_ptr + 1
    did = dummy_intr['value']['ptr_wrapper']['data']
    did['width'] = did['height'] = 1
    did['focal_length'] = 1.0
    did['principal_point'] = [0.5, 0.5]
    # Reference the already-registered polymorphic type by bare id (drop the
    # registration bit + name; intrinsics[0] is the registering instance).
    dummy_intr['value']['polymorphic_id'] = \
        bare_polymorphic_id(intrinsics[0]['value'])
    dummy_intr['value'].pop('polymorphic_name', None)
    data['intrinsics'].append(dummy_intr)

    dummy_pose = json.loads(json.dumps(data['extrinsics'][0]))
    dummy_pose['key'] = new_view_key
    data['extrinsics'].append(dummy_pose)

    data['root_path'] = str(image.parent.resolve())
    data['structure'] = []
    data['control_points'] = []
    out_json.write_text(json.dumps(data, indent=2))
    logger.info(f'Re-pointed calibration at {image.name} ({w}x{h}); '
                f'wrote scene (+1x1 dummy view) -> {out_json}')


def load_obj_mesh(mesh_path: Path):
    """Load an OBJ as (vertices Nx3 float64, triangles Mx3 int), ignoring any
    existing texture coords/normals/materials (we regenerate UVs). Polygons are
    fan-triangulated; face vertex references use the first (position) index."""
    verts = []
    faces = []
    with mesh_path.open() as fh:
        for line in fh:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('f '):
                idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def project_texture_mesh(calibration_json: Path, texture_image: Path,
                         mesh_path: Path, out_obj: Path,
                         backface_cull: bool = True,
                         recorder: Recorder = None) -> None:
    """Texture a mesh by projecting it through the calibrated view, so the OBJ's
    UVs index the *original* modality image directly (no OpenMVS atlas, no
    resampling). Because the UVs depend only on the camera + mesh — identical
    across modalities — every modality reuses these UVs and only swaps ``map_Kd``.

    Each vertex is projected with the calibrated pose/intrinsic to a pixel, then
    to a UV (image origin is top-left, OBJ's is bottom-left, so v is flipped).
    A triangle is textured only if all three vertices are in front of the camera
    and inside the image, and (if ``backface_cull``) the face points toward the
    camera — which drops a closed mesh's hidden underside and grazing edges.
    Triangles outside the view are omitted (that surface was not imaged), the
    single-view analogue of OpenMVS' empty-color.

    NOTE: this does not do depth-based occlusion, so a surface that overhangs
    itself would project the foreground onto the hidden region. For the open
    surface meshes this targets the effect is negligible; use ``--use-openmvs``
    when true occlusion handling is required.
    """
    R, C, f, cx, cy, W, H, disto = camera_from_calibration(calibration_json)
    V, F = load_obj_mesh(mesh_path)
    if len(V) == 0 or len(F) == 0:
        sys.exit(f'Mesh {mesh_path} has no geometry to texture')

    Xc = (R @ (V - C).T).T
    Z = Xc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        x = Xc[:, 0] / Z
        y = Xc[:, 1] / Z
    if disto is not None:
        r2 = x * x + y * y
        rad = 1.0 + disto[0] * r2 + disto[1] * r2 * r2 + disto[2] * r2 ** 3
        x, y = x * rad, y * rad
    u = f * x + cx
    v = f * y + cy

    valid = (Z > 1e-9) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    keep = valid[F].all(axis=1)
    if backface_cull:
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        normal = np.cross(v1 - v0, v2 - v0)
        centroid = (v0 + v1 + v2) / 3.0
        keep &= np.einsum('ij,ij->i', normal, C - centroid) > 0
    Fk = F[keep]
    if len(Fk) == 0:
        logger.warning('No triangles fall within the calibrated view; '
                       'the output mesh will have no texture '
                       '(is the calibration for this mesh?).')

    # OBJ texture coords: flip v; clamp tiny FP overshoot.
    uv = np.column_stack([np.clip(u / W, 0.0, 1.0),
                          np.clip(1.0 - v / H, 0.0, 1.0)])

    out_obj.parent.mkdir(parents=True, exist_ok=True)
    # Copy the original image beside the mesh, renamed to match the mesh stem so
    # the texture is consistently named and easy to pair with its OBJ. The
    # original's extension/format is preserved (no conversion).
    tex_dst = out_obj.parent / f'{out_obj.stem}{texture_image.suffix}'
    if texture_image.resolve() != tex_dst.resolve():
        shutil.copy(texture_image, tex_dst)
    mtl = out_obj.with_suffix('.mtl')
    with mtl.open('w') as fh:
        fh.write('newmtl material_0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\n')
        fh.write(f'map_Kd {tex_dst.name}\n')
    with out_obj.open('w') as fh:
        fh.write(f'mtllib {mtl.name}\n')
        np.savetxt(fh, V, fmt='v %.6f %.6f %.6f')
        np.savetxt(fh, uv, fmt='vt %.6f %.6f')
        fh.write('usemtl material_0\n')
        # In-view faces: textured (v/vt/vt references).
        if len(Fk) > 0:
            f1k = Fk + 1                  # OBJ indices are 1-based
            face_cols = np.column_stack([f1k[:, 0], f1k[:, 0], f1k[:, 1], f1k[:, 1],
                                         f1k[:, 2], f1k[:, 2]])
            np.savetxt(fh, face_cols, fmt='f %d/%d %d/%d %d/%d')
        # Out-of-view faces: preserved without UV (v references only).
        Fu = F[~keep]
        if len(Fu) > 0:
            np.savetxt(fh, Fu + 1, fmt='f %d %d %d')

    if recorder is not None:
        recorder.note(f'project_texture_mesh mesh={mesh_path.name} '
                      f'texture={tex_dst.name} -> {out_obj.name}')
    logger.info(f'Projected texture: {len(Fk)}/{len(F)} triangles textured '
                f'({100.0 * len(Fk) / len(F):.1f}% of mesh in view); '
                f'map_Kd={tex_dst.name} -> {out_obj}')


def sfm_to_json(sfm_path: Path, out_json: Path) -> Path:
    """Export an OpenMVG SfM_Data (.bin/.json) to JSON with only views,
    intrinsics, and extrinsics (drops structure/control points). A ``.json``
    input is re-exported anyway to strip structure and normalize."""
    command = [
        resolve_exe('openMVG_main_ConvertSfM_DataFormat'),
        '-i', sfm_path.resolve(),
        '-o', out_json.resolve(),
        '-V', '-I', '-E',
    ]
    run(command)
    return out_json


def filter_sfm_for_cameras(sfm_json: Path, prefix: str,
                           modality_dir: Path,
                           key_to_name: Dict[tuple, str],
                           out_json: Path) -> int:
    """Rewrite the SfM scene to the views that have a modality image, each
    re-pointed at it. Returns kept view count.

    The cameras textured from are whatever ``key_to_name`` and the solve share;
    ``--camera-index`` has already narrowed the former, so no camera filter is
    applied here. The two absences are not symmetric: a solved view with no
    modality image is dropped quietly (the texturing capture simply fired fewer
    cameras, or fewer positions), while a modality image with no solved view is
    warned about loudly, because it was never calibrated and nothing can place it.
    """
    data = json.loads(sfm_json.read_text())
    kept = []
    parsed_views = 0
    missing: Dict[int, int] = {}
    matched_keys = set()
    kept_intrinsics = set()
    kept_poses = set()
    for v in data['views']:
        vd = v['value']['ptr_wrapper']['data']
        parsed = parse_view_name(Path(vd['filename']).name, prefix)
        if parsed is None:
            continue
        parsed_views += 1
        cam, pos = parsed
        if (cam, pos) not in key_to_name:
            missing[cam] = missing.get(cam, 0) + 1
            continue
        vd['filename'] = key_to_name[(cam, pos)]
        vd['local_path'] = ''
        matched_keys.add((cam, pos))
        kept_intrinsics.add(vd['id_intrinsic'])
        kept_poses.add(vd['id_pose'])
        kept.append(v)

    if not kept:
        # Nothing parsing at all is a different failure from nothing matching:
        # the first says the recon was not built from PGS-named images of this
        # scan, the second that its cameras and positions are elsewhere.
        if parsed_views == 0:
            sys.exit(f'No SfM view in {sfm_json.name} matched the scan prefix '
                     f'{prefix!r} ({len(data["views"])} views did not parse). '
                     f'Was this reconstruction built from this scan, with '
                     f'PGS-named images?')
        sys.exit('No solved view shares a (camera, position) with the modality '
                 'images; nothing to texture')

    # Images the solve cannot place: they were not part of the reconstruction, so
    # there is no pose to texture them from. Loud, because it means the capture
    # covers cameras or positions the solve does not.
    uncalibrated = sorted(set(key_to_name) - matched_keys)
    if uncalibrated:
        cams = sorted({cam for cam, _pos in uncalibrated})
        logger.warning(f'{len(uncalibrated)} modality image(s) have no solved '
                       f'view and are unused (camera(s) {cams}, e.g. '
                       f'{key_to_name[uncalibrated[0]]}). Those '
                       f'(camera, position) pairs were not reconstructed.')

    # Drop orphan intrinsics/poses; openMVG2openMVS rejects scenes that carry
    # intrinsics or poses not referenced by any view.
    data['views'] = kept
    intrinsics = [i for i in data.get('intrinsics', [])
                  if i['key'] in kept_intrinsics]
    data['intrinsics'] = fix_polymorphic_registration(
        data.get('intrinsics', []), intrinsics)
    data['extrinsics'] = [e for e in data.get('extrinsics', [])
                          if e['key'] in kept_poses]
    data['root_path'] = str(modality_dir.resolve())
    data['structure'] = []
    data['control_points'] = []
    out_json.write_text(json.dumps(data, indent=2))
    cameras = sorted({cam for cam, _pos in matched_keys})
    logger.info(f'Filtered SfM to {len(kept)} views for camera(s) {cameras}')
    if missing:
        # Not necessarily an absence in the capture: --camera-index has already
        # narrowed the image set, and from here the two are indistinguishable.
        logger.info('Solved views with no modality image, per camera: '
                    + ', '.join(f'{cam}: {n}'
                                for cam, n in sorted(missing.items())))
    return len(kept)


def relocate_textured_mesh(produced, produced_mesh: Path, target: Path) -> Path:
    """Move a freshly textured mesh and its sidecar files into ``target``'s dir.

    ``produced`` is the set of files TextureMesh just wrote (mesh + .mtl +
    texture image(s)); they all share ``target``'s stem because the stem was
    derived from --output-mesh, so basenames are preserved and the OBJ/MTL's
    relative references stay valid — only the directory changes. The caller
    must restrict ``produced`` to stem-matching files so unrelated artifacts
    TextureMesh drops in the work dir (e.g. its ``TextureMesh-*.log``) are not
    swept along. Returns the moved mesh path.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    moved_mesh = None
    for src in produced:
        dst = target.parent / src.name
        if src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))
        if src.name == produced_mesh.name:
            moved_mesh = dst
    return moved_mesh or (target.parent / produced_mesh.name)


def ensure_ply_mesh(mesh_path: Path, work_dir: Path,
                    out_stem: str = None) -> Path:
    """Stage a mesh into ``work_dir`` in a form OpenMVS can reliably load.

    OpenMVS' OBJ reader is strict (and mis-resolves a relative ``mtllib`` when
    a working folder is set), so OBJ inputs are converted to a geometry-only
    binary PLY (vertices + triangles). Texture coordinates/materials are
    irrelevant here because TextureMesh regenerates its own UVs. PLY inputs are
    copied through unchanged. Either way the result lives in ``work_dir`` so
    TextureMesh can reference it by basename. ``out_stem``, if given, names the
    staged file ``<out_stem>_input.ply`` so it cannot clash with the recon's
    own files in a shared working dir.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out_name = f'{out_stem}_input.ply' if out_stem else None
    if mesh_path.suffix.lower() == '.ply':
        out = work_dir / (out_name or mesh_path.name)
        if mesh_path.resolve() != out.resolve():
            shutil.copy(mesh_path, out)
        return out

    verts = []
    faces = []
    with mesh_path.open() as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('f '):
                idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
                for i in range(1, len(idx) - 1):  # triangulate fan
                    faces.append((idx[0], idx[i], idx[i + 1]))

    v = np.asarray(verts, dtype='<f4')
    fcs = np.asarray(faces, dtype='<i4')
    out = work_dir / (out_name or (mesh_path.stem + '.ply'))
    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        f'element vertex {len(v)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        f'element face {len(fcs)}\n'
        'property list uchar int vertex_indices\n'
        'end_header\n'
    )
    face_rec = np.empty(len(fcs), dtype=[('n', 'u1'), ('i', '<i4', 3)])
    face_rec['n'] = 3
    face_rec['i'] = fcs
    with out.open('wb') as fh:
        fh.write(header.encode('ascii'))
        fh.write(v.tobytes())
        fh.write(face_rec.tobytes())
    logger.info(f'Converted {mesh_path.name} -> {out.name} '
                f'({len(v)} verts, {len(fcs)} tris)')
    return out


def main():
    """Entry point. A failed binary exits with *its* status, not 1."""
    try:
        _main()
    except ToolFailed as e:
        logging.getLogger('pgs-retexture').error(f'{e}')
        sys.exit(e.exit_code)


def _main():
    parser = configargparse.ArgumentParser(
        prog='pgs-retexture',
        description='Re-texture an existing mesh with an alternate imaging '
                    'modality captured at one camera\'s positions.')
    parser.add_argument('--config', '-c', is_config_file=True,
                        help='Config file path')
    parser.add_argument('--modality-images', '-i', required=True,
                        help='Modality image input. Without --calibration: a PGS '
                             'SCAN DIRECTORY (with its metadata.json), whose '
                             '--capture supplies the texturing images; they are '
                             'matched to SfM views by the PGS-scan filename '
                             'convention {prefix}{camera}_{position}_{capture}. '
                             'With --calibration: a SINGLE image file captured '
                             'from the calibrated pose (any filename).')
    parser.add_argument('--calibration', default=None,
                        help='A pgs-calibrate calibration .json (one localized '
                             'view). Textures the mesh from that new pose with '
                             'the single image given by -i, instead of reusing a '
                             'rig camera\'s positions. --capture, --camera-index '
                             'and --sfm-data are ignored in this mode.')
    parser.add_argument('--use-openmvs', action='store_true',
                        help='With --calibration, texture via OpenMVS TextureMesh '
                             '(regenerates UVs into a resampled atlas; does true '
                             'occlusion) instead of the default projective UV '
                             'mapping (UVs point at the original full-res image, '
                             'reused across modalities).')
    parser.add_argument('--no-backface-cull', action='store_true',
                        help='With projective UV mapping, keep faces pointing '
                             'away from the camera (default culls them, dropping '
                             'a closed mesh\'s hidden underside).')
    parser.add_argument('--convert-texture', action='store_true',
                        help='With projective UV mapping and --calibration: '
                             'convert the modality image to 8-bit sRGB before '
                             'copying it as the texture. '
                             'Needed for CIELab TIFFs and other non-sRGB '
                             'inputs that would render with wrong colors in '
                             'standard viewers. Default: copy the original '
                             'file as-is (full fidelity for standard sRGB).')
    parser.add_argument('--recon-dir', '-r', required=True,
                        help='A pgs-recon output directory. The solved SfM and '
                             'the textured mesh are located from its '
                             'manifest; each is only required if this run '
                             'actually reads it, so overriding both (--sfm-data '
                             'or --calibration, plus --mesh) needs nothing from '
                             'the reconstruction but its manifest. The recon must '
                             'have used PGS-named images (its SfM view filenames '
                             'must match the convention above) unless '
                             '--calibration is given.')
    parser.add_argument('--sfm-data', '-s', default=None,
                        help='Override the solved OpenMVG SfM_Data (.bin/.json) '
                             'to texture from. Defaults to the SfM that produced '
                             'the mesh in --recon-dir. NOT the rig-prior import '
                             'scene. Ignored in --calibration mode.')
    parser.add_argument('--sfm-transform', default=None,
                        help='4x4 .npy transform matrix saved by pgs-center '
                             '--save-transform. In non-calibration mode, '
                             're-expresses SfM camera poses in the centered mesh '
                             'coordinate frame before building the MVS scene. '
                             'Must be paired with --mesh pointing at the centered '
                             'mesh. Ignored in --calibration mode (run '
                             'pgs-calibrate with --sfm-transform instead to embed '
                             'the transform in the calibration).')
    parser.add_argument('--mesh', default=None,
                        help='Override the mesh to texture. Use to supply a '
                             'centered or ground-plane-removed mesh in place of '
                             'the original reconstruction output; the recon\'s own '
                             'mesh is then not required to exist.')
    parser.add_argument('--working-dir', '-w', default=None,
                        help='Directory for retexture artifacts (default: '
                             '--recon-dir). Outputs are written into its mvg/ '
                             'and mvs/ as <stem>-prefixed siblings of the '
                             'reconstruction and never overwrite recon files.')
    parser.add_argument('--output-mesh', '-o', default=None,
                        help='Exact path + filename of the final textured mesh; '
                             'its extension sets the output format (overrides '
                             '--file-type). The mesh, its .mtl, and the texture '
                             'image are written together there. Default: '
                             '<working-dir>/mvs/<stem>.<file-type>. The <stem> '
                             '(prefixed onto every scratch artifact) is this '
                             'file\'s stem if given, else the modality input '
                             'name plus the capture textured from '
                             '(<scan-dir>_c<capture>).')
    parser.add_argument('--capture', type=int, default=None, metavar='n',
                        help='Capture index in --modality-images to texture from. '
                             'Inferred when the scan holds only one capture; '
                             'required when it holds several. Independent of '
                             'pgs-recon\'s --import-capture: any pair is valid '
                             '(solve from capture 0, texture from capture 3). '
                             'Ignored in --calibration mode.')
    parser.add_argument('--camera-index', '-k', type=int, nargs='+',
                        default=None, metavar='n',
                        help='Restrict texturing to these camera indices '
                             '(e.g. -k 1 3). Default: every camera the capture '
                             'and the solve share. A requested camera the '
                             'capture lacks is warned about and skipped. '
                             'Ignored in --calibration mode.')
    parser.add_argument('--bit-shift', type=int, default=8,
                        help='Right bit-shift applied to 16-bit modality images '
                             'to map to 8-bit (default 8 = divide by 256). '
                             'Applied uniformly to all frames.')
    parser.add_argument('--file-type', '-f', default='obj',
                        choices=['obj', 'ply', 'glb', 'gltf'], type=str.lower,
                        help='Output mesh format')
    parser.add_argument('--texture-resolution-level', type=int, default=None,
                        help='OpenMVS TextureMesh --resolution-level')
    parser.add_argument('--max-texture-size', type=int, default=0,
                        help='OpenMVS TextureMesh --max-texture-size')
    parser.add_argument('--empty-color', type=int, default=None,
                        help='Integer color for faces seen by no image '
                             '(OpenMVS --empty-color; e.g. 0 for black)')
    parser.add_argument('--global-seam-leveling', type=int, default=1,
                        choices=[0, 1],
                        help='OpenMVS global seam leveling. Default 1 (on) '
                             'normalizes patch brightness to hide seams; set 0 '
                             '(off) to preserve source radiometry.')
    parser.add_argument('--local-seam-leveling', type=int, default=0,
                        choices=[0, 1],
                        help='OpenMVS local (Poisson) seam leveling. Default 0 '
                             '(off) to preserve source radiometry.')
    parser.add_argument('--threads', type=int, default=None,
                        help='Threads for openMVG2openMVS')
    # Unset, the install prefix falls through to $PGS_RECON_PREFIX and then to
    # toolchain.DEFAULT_PREFIX, which a default here would shadow.
    parser.add_argument('--path', type=str, default=None,
                        help=configargparse.SUPPRESS)
    parser.add_argument('--log-level', default='INFO', type=str.upper,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    setup_logging(args.log_level)
    global logger
    logger = logging.getLogger('pgs-retexture')

    modality_input = Path(args.modality_images)
    recon_dir = Path(args.recon_dir)
    calibration = Path(args.calibration) if args.calibration else None

    # -r must be a pgs-recon output whatever this run takes from it: the manifest
    # is the run-tracking record, and the mvg/ mvs/ layout below assumes it.
    load_manifest(recon_dir)

    # Beyond that, only what this run actually reads is demanded. The SfM is the
    # scene rebuilt for the rig camera, so it is unused in --calibration mode
    # (the calibration .json is the scene) and replaceable by --sfm-data; the
    # mesh is what gets textured, so it is unused when --mesh names another one.
    # Whatever remains is required, and .require() exits with the manifest's own
    # diagnosis of why it could not be resolved.
    if calibration is not None:
        sfm_data = None
        if args.sfm_data:
            logger.warning('--sfm-data is ignored in --calibration mode; the '
                           'calibration .json is the scene')
    elif args.sfm_data:
        sfm_data = Path(args.sfm_data)
        if not sfm_data.is_file():
            sys.exit(f'--sfm-data: file not found: {sfm_data}')
    else:
        sfm_data = resolve_solved_sfm(recon_dir).require()

    if args.mesh is not None:
        mesh_in = Path(args.mesh)
        if not mesh_in.is_file():
            sys.exit(f'--mesh: file not found: {mesh_in}')
    else:
        mesh_in = resolve_textured_mesh(recon_dir).require()

    sfm_transform = None
    if args.sfm_transform is not None:
        sfm_transform = np.load(args.sfm_transform)
        if sfm_transform.shape != (4, 4):
            sys.exit(f'--sfm-transform: expected a 4x4 matrix, '
                     f'got shape {sfm_transform.shape}')
        if calibration is not None:
            logger.warning('--sfm-transform is ignored in --calibration mode; '
                           'run pgs-calibrate --sfm-transform to embed the '
                           'transform in the calibration instead')
            sfm_transform = None

    # The transformed poses are expressed in the centered mesh frame, so the
    # mesh being textured must be the centered one too. (calibration mode
    # already nulled sfm_transform above, so this only guards legacy mode.)
    if sfm_transform is not None and args.mesh is None:
        sys.exit('--sfm-transform requires --mesh pointing at the centered mesh '
                 '(the transformed camera poses must match a centered mesh)')

    # Index the texturing capture before anything is named after it: the stem
    # carries the capture, so a scan directory's captures cannot collide.
    capture = None
    scan_prefix = None
    img_map = None
    if calibration is None:
        logger.info('Indexing modality images')
        capture, scan_prefix, img_map = index_modality_images(
            modality_input, args.capture, args.camera_index)
        # Pin the capture so a replayed config textures the same one even if the
        # scan later grows another. NOT the camera list: it is derived from the
        # capture, so recording it as if it were requested would silently narrow
        # a replay that overrides --capture. It goes in the manifest instead.
        args.capture = capture
    else:
        for name in ('capture', 'camera_index'):
            if getattr(args, name) is not None:
                logger.warning(f'--{name.replace("_", "-")} is ignored in '
                               f'--calibration mode: the calibration is one '
                               f'camera at one pose')

    # Resolve the final mesh path + format and derive the artifact stem.
    # --output-mesh sets the exact deliverable; its extension wins over
    # --file-type. The stem is prefixed onto every scratch artifact so they
    # coexist with the recon's files (there is no --name flag): the
    # --output-mesh stem if given, else the modality input's name -- the scan
    # directory plus the capture textured from (one directory serves every
    # capture), the image stem in --calibration mode.
    output_mesh = Path(args.output_mesh) if args.output_mesh else None
    if output_mesh is not None:
        ext = output_mesh.suffix.lstrip('.').lower()
        if ext not in ('obj', 'ply', 'glb', 'gltf'):
            sys.exit(f'--output-mesh: unsupported extension '
                     f'{output_mesh.suffix!r} (use .obj/.ply/.glb/.gltf)')
        if ext != args.file_type:
            logger.warning(f'--output-mesh extension .{ext} overrides '
                           f'--file-type {args.file_type}')
        file_format = ext
        stem = output_mesh.stem
    else:
        file_format = args.file_type
        stem = (modality_input.resolve().stem if calibration
                else f'{modality_input.resolve().name}_c{capture}')

    working_dir = Path(args.working_dir) if args.working_dir else recon_dir
    working_dir.mkdir(parents=True, exist_ok=True)

    # Artifacts integrate into the recon's existing mvg/ and mvs/, prefixed by
    # <stem> so they sit beside the recon's files without overwriting them.
    paths: Dict[str, Path] = {'working': working_dir}
    paths['mvg'] = working_dir / 'mvg'
    paths['mvs'] = working_dir / 'mvs'
    paths['mvg'].mkdir(parents=True, exist_ok=True)
    paths['mvs'].mkdir(parents=True, exist_ok=True)
    paths['modality_8bit'] = paths['mvs'] / f'{stem}_modality'
    paths['mvs_scene'] = paths['mvs'] / f'{stem}_scene.mvs'
    paths['mvs_images'] = paths['mvs'] / f'{stem}_undistorted_images'
    paths['sfm_full'] = paths['mvg'] / f'{stem}_sfm_full.json'
    paths['sfm_filtered'] = paths['mvg'] / f'{stem}_sfm.json'

    # Final mesh: --output-mesh if given, else mvs/<stem>.<file_format>.
    paths['output_mesh'] = (output_mesh if output_mesh is not None
                            else paths['mvs'] / f'{stem}.{file_format}')

    # Config + metadata, mirroring pgs-recon conventions (sidecar files; the
    # recon's own manifest is never touched).
    datetime_str = dt.now(tz.utc).strftime('%Y%m%d%H%M%S')
    config_path = working_dir / f'{datetime_str}_{stem}_retexture_config.txt'
    args.config = str(config_path)
    with config_path.open('w') as f:
        for arg in vars(args):
            # Unset arguments are omitted: a literal `path = None` read back
            # through -c would be parsed as the string 'None'.
            if getattr(args, arg) is None:
                continue
            f.write(f"{arg.replace('_', '-')} = {getattr(args, arg)}\n")

    # `parsed` is what was asked for; `capture`/`cameras` are what the run
    # resolved that to. The manifest is a record, never read back as input, so
    # inferred values are safe to state here in a way a config file is not.
    metadata = {'args': ' '.join(sys.argv), 'parsed': vars(args),
                'commands': {}}
    if img_map is not None:
        metadata['capture'] = capture
        metadata['cameras'] = sorted({cam for cam, _pos in img_map})
    paths['manifest'] = working_dir / f'{stem}_retexture.json'

    # Both a recon artifact the stem happened to match and this retexture's own
    # previous output are overwritten -- re-running has to stay legal -- but
    # neither silently.
    existing = sorted(str(p) for k, p in paths.items()
                      if k not in ('working', 'mvg', 'mvs') and p.exists())
    if existing:
        logger.warning('These artifact paths already exist and will be '
                       'overwritten: ' + ', '.join(existing))

    # Where the binaries are and what records their invocations: process-wide, so
    # no wrapper takes either as an argument (ADR 0005).
    recorder = Recorder(metadata['commands'])
    toolchain.configure(prefix=args.path, recorder=recorder)

    @atexit.register
    def write_metadata():
        metadata['paths'] = {k: str(v) for k, v in paths.items()}
        write_manifest(paths['manifest'], metadata)

    write_metadata()

    # 1-2. Build the SfM scene to texture from. Two modes:
    if calibration is not None:
        # New pose from pgs-calibrate: texture from one localized view, swapping
        # in the chosen modality image. No filename convention is needed.
        if not modality_input.is_file():
            sys.exit('--calibration mode expects -i to be a single image file, '
                     f'got: {modality_input}')
        if not args.use_openmvs:
            # Default: project the mesh into the calibrated view so the OBJ's UVs
            # point straight at the image (no OpenMVS atlas resampling; UVs
            # reused across modalities). Projection reads only pose + mesh.
            out_obj = paths['output_mesh']
            if out_obj.suffix.lower() != '.obj':
                logger.warning('Projective UV mapping writes OBJ with an '
                               f'external texture; forcing .obj on '
                               f'{out_obj.name}.')
                out_obj = out_obj.with_suffix('.obj')
            tex_img = modality_input
            if args.convert_texture:
                logger.info('Converting modality image to 8-bit sRGB')
                tex_img = prepare_8bit_image(modality_input,
                                             paths['modality_8bit'])
            logger.info('Projecting mesh into calibrated view for UV mapping')
            project_texture_mesh(calibration, tex_img, mesh_in, out_obj,
                                 backface_cull=not args.no_backface_cull,
                                 recorder=recorder)
            logger.info(f'Done. Re-textured mesh: {out_obj}')
            return
        # OpenMVS path: undistortion reads pixels, so it needs an 8-bit image.
        logger.info('Preparing 8-bit modality image')
        conv = prepare_8bit_image(modality_input, paths['modality_8bit'])
        logger.info('Re-pointing calibration at the modality image')
        repoint_calibration(calibration, conv, paths['sfm_filtered'])
    else:
        # Capture retexture: reuse the rig's solved positions, matched by the
        # PGS-scan filename convention. The capture was indexed before the stem
        # was derived from it, so only the pixels and the scene remain.
        logger.info('Preparing 8-bit modality images')
        key_to_name = convert_modality_images(img_map, paths['modality_8bit'],
                                              args.bit_shift)
        logger.info('Exporting SfM solution to JSON')
        sfm_to_json(sfm_data, paths['sfm_full'])
        logger.info('Filtering SfM scene to the modality images')
        filter_sfm_for_cameras(paths['sfm_full'], scan_prefix,
                               paths['modality_8bit'], key_to_name,
                               paths['sfm_filtered'])
        if sfm_transform is not None:
            logger.info('Transforming SfM extrinsics to centered mesh frame')
            transform_sfm_extrinsics(paths['sfm_filtered'], sfm_transform)

    # 3. MVG -> MVS (undistorts modality images with original intrinsics)
    logger.info('Building MVS scene from modality views')
    mvg_to_mvs(paths['sfm_filtered'], scene=paths['mvs_scene'],
               images_dir=paths['mvs_images'], threads=args.threads)

    # 4. Texture the existing mesh. TextureMesh writes it into the working mvs/
    # beside the scene it textures from; the deliverable may live elsewhere.
    logger.info('Preparing mesh for OpenMVS')
    paths['mesh'] = ensure_ply_mesh(mesh_in, paths['mvs'], out_stem=stem)
    paths['textured_mesh'] = paths['mvs'] / f'{stem}.{file_format}'
    logger.info('Texturing mesh with modality images')
    before = set(paths['mvs'].iterdir())
    mvs_texture(paths['mvs_scene'], mesh=paths['mesh'],
                output=paths['textured_mesh'], export_type=file_format,
                resolution_level=args.texture_resolution_level,
                max_texture_size=args.max_texture_size,
                empty_color=args.empty_color,
                global_seam_leveling=args.global_seam_leveling,
                local_seam_leveling=args.local_seam_leveling)

    # Relocate the deliverable (mesh + .mtl + texture) if --output-mesh points
    # outside the working mvs/. The produced files already carry <stem> ==
    # target stem, so this is a pure directory move with references intact.
    final = paths['textured_mesh']
    target = paths['output_mesh']
    if target.resolve() != final.resolve():
        # Only the mesh and its own sidecars (all <stem>-prefixed); leave
        # unrelated new files such as TextureMesh's own log behind.
        produced = sorted(p for p in paths['mvs'].iterdir()
                          if p.is_file() and p not in before
                          and p.name.startswith(stem))
        final = relocate_textured_mesh(produced, final, target)
    paths['textured_mesh'] = final

    logger.info(f'Done. Re-textured mesh: {final}')


if __name__ == '__main__':
    main()
