"""Re-texture an existing reconstructed mesh using an alternate imaging
modality (e.g. IR940) captured at the same camera positions as one of the
cameras in the original reconstruction.

This tool serves two modes that are different in kind, and only one of them is
inherently single-camera:

  - **Capture retexture** (default mode): the texturing images are another
    *capture* of the same scan -- the same rig, the same capture positions,
    different illumination. Correspondence is by ``(camera, position)``, and
    every camera present in both captures could contribute.
  - **Localized-camera retexture** (``--calibration``): the images come from a
    camera that was never in the solve, placed in the solved frame by
    ``pgs-calibrate``. One camera, one pose, no positional correspondence.

**Both modes are currently restricted to a single camera**, which is inherent
only to the second. For capture retexture it is a limitation, not a design, and
it is what stops you texturing with a capture in which the whole rig fired. See
``docs/multi-camera-capture-retexture.md``.

OpenMVS has no native "texture with a different image set" option (verified
against OpenMVS v2.3.0), so this tool rebuilds a minimal MVS scene from the
original SfM solution restricted to a single camera, with that camera's views
re-pointed at the modality images. The pipeline is:

  1. (optional) Convert 16-bit modality images to 8-bit with a fixed linear
     map (bit-shift), uniform across all frames to preserve relative radiometry
     and keep the merged texture seamless.
  2. Convert the solved OpenMVG SfM_Data to JSON (views/intrinsics/extrinsics
     only) and filter it to the requested camera, re-pointing each view at the
     matching modality image (matched by capture-position index).
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

REQUIRES THE PGS-SCAN FILENAME CONVENTION on both image sets. The correspondence
between a modality image and a camera pose in the SfM solution is established
*entirely by filename* — there is no EXIF, ordering, or geometric fallback. Every
filename must match ``{prefix}_{camera}_{position}_{capture}`` (see ``_NAME_RE``),
and matching is keyed on ``(camera, position)`` (the ``capture`` field is parsed
but ignored). Concretely:

  - Modality images are grouped by ``(camera, position)``; ``--camera-index``
    selects which camera, or it is inferred if the files share one.
  - Each SfM view's stored filename is parsed the same way; views for the chosen
    camera are re-pointed at the modality image with the *same position index*.

This means the original reconstruction must itself have been run on PGS-named
images (e.g. imported via ``pgs-import`` / ``init_sfm_pgs``). If the SfM views
carry arbitrary filenames (e.g. a generic EXIF-based import), none will parse and
the run aborts with "No views matched camera". Modality position indices must
correspond 1:1 to the original camera's positions (the captures must be the same
shots from the same poses).
"""
import atexit
import json
import logging
import re
import shutil
import sys
from datetime import datetime as dt, timezone as tz
from pathlib import Path
from typing import Dict, Optional

import configargparse
import cv2
import numpy as np

from pgs_recon import toolchain
from pgs_recon.openmvg import mvg_to_mvs
from pgs_recon.openmvs import mvs_texture
from pgs_recon.stages import write_manifest
from pgs_recon.toolchain import Recorder, resolve_exe, run
from pgs_recon.utility import ToolFailed
from pgs_recon.utils.apps import setup_logging
from pgs_recon.utils.images import prepare_8bit_image
from pgs_recon.utils.recon_dir import (
    load_manifest,
    resolve_solved_sfm,
    resolve_textured_mesh,
)
from pgs_recon.utils.sfm_json import (
    bare_polymorphic_id,
    camera_from_calibration,
    fix_polymorphic_registration,
    transform_sfm_extrinsics,
)

logger = logging.getLogger(__name__)

# PGS-scan filename convention: {prefix}_{camera}_{position}_{capture}.{ext}.
# This is the ONLY correspondence mechanism between modality images and SfM
# views (see module docstring); both image sets must follow it, and matching is
# keyed on (camera, position).
_NAME_RE = re.compile(r'^(?P<prefix>.*)_(?P<cam>\d+)_(?P<pos>\d+)_(?P<cap>\d+)$')

# Image extensions we treat as modality inputs
_IMG_EXTS = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}


def parse_name(stem: str):
    """Parse a PGS image stem into (camera, position, capture) ints.

    Returns None if the stem does not match the expected pattern.
    """
    m = _NAME_RE.match(stem)
    if m is None:
        return None
    return int(m.group('cam')), int(m.group('pos')), int(m.group('cap'))


def index_modality_images(image_dir: Path, camera_index: Optional[int]):
    """Map capture-position index -> modality image path for a single camera.

    If ``camera_index`` is None it is inferred from the files (they must all
    share one camera index).
    """
    by_cam: Dict[int, Dict[int, Path]] = {}
    for p in sorted(image_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in _IMG_EXTS:
            continue
        parsed = parse_name(p.stem)
        if parsed is None:
            logger.warning(f'Skipping unrecognized filename: {p.name}')
            continue
        cam, pos, _cap = parsed
        existing = by_cam.setdefault(cam, {}).get(pos)
        if existing is not None:
            logger.warning(f'Multiple modality images for camera {cam} '
                           f'position {pos}: keeping {p.name}, '
                           f'dropping {existing.name}')
        by_cam[cam][pos] = p

    if not by_cam:
        sys.exit(f'No modality images found in {image_dir}')

    if camera_index is None:
        if len(by_cam) > 1:
            sys.exit(f'Modality dir contains multiple camera indices '
                     f'{sorted(by_cam)}; specify --camera-index')
        camera_index = next(iter(by_cam))
    elif camera_index not in by_cam:
        sys.exit(f'No modality images for camera {camera_index} in '
                 f'{image_dir} (found {sorted(by_cam)})')

    logger.info(f'Using camera index {camera_index} '
                f'({len(by_cam[camera_index])} modality images)')
    return camera_index, by_cam[camera_index]


def convert_modality_images(pos_map: Dict[int, Path], out_dir: Path,
                            bit_shift: int) -> Dict[int, str]:
    """Ensure every modality image is an 8-bit file usable by openMVG/OpenMVS.

    16-bit images are mapped to 8-bit with a fixed bit-shift (>> bit_shift),
    applied identically to every frame. Float images are scaled from an assumed
    [0, 1] range by a fixed factor. Already-8-bit images are copied through
    unchanged. All conversions are uniform across frames to preserve relative
    radiometry. Returns position index -> output filename (basename).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_names: Dict[int, str] = {}
    for pos, src in sorted(pos_map.items()):
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            sys.exit(f'Could not read modality image: {src}')
        if img.dtype == np.uint16:
            img = (img >> bit_shift).astype(np.uint8)
        elif img.dtype != np.uint8:
            # Float or other depth: assume a [0, 1] range and apply a fixed
            # scale (uniform across all frames, like the 16-bit bit-shift) so
            # relative radiometry is preserved and the merged texture stays
            # consistent. Per-frame normalization would break that.
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        out_name = f'{src.stem}.jpg'
        cv2.imwrite(str(out_dir / out_name), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
        out_names[pos] = out_name
    logger.info(f'Prepared {len(out_names)} 8-bit modality images in {out_dir}')
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

    img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f'Could not read modality image: {image}')
    h, w = img.shape[:2]
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
    cv2.imwrite(str(dummy_img), np.zeros((1, 1, 3), np.uint8))
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


def filter_sfm_for_camera(sfm_json: Path, camera_index: int,
                          modality_dir: Path,
                          pos_to_name: Dict[int, str],
                          out_json: Path) -> int:
    """Rewrite the SfM scene to contain only the requested camera's views,
    each re-pointed at its matching modality image. Returns kept view count."""
    data = json.loads(sfm_json.read_text())
    kept = []
    missing = 0
    kept_intrinsics = set()
    kept_poses = set()
    for v in data['views']:
        vd = v['value']['ptr_wrapper']['data']
        parsed = parse_name(Path(vd['filename']).stem)
        if parsed is None:
            continue
        cam, pos, _cap = parsed
        if cam != camera_index:
            continue
        if pos not in pos_to_name:
            logger.warning(f'No modality image for position {pos} '
                           f'(view {vd["filename"]}); dropping view')
            missing += 1
            continue
        vd['filename'] = pos_to_name[pos]
        vd['local_path'] = ''
        kept_intrinsics.add(vd['id_intrinsic'])
        kept_poses.add(vd['id_pose'])
        kept.append(v)

    if not kept:
        sys.exit(f'No views matched camera {camera_index}; nothing to texture')

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
    logger.info(f'Filtered SfM to {len(kept)} views for camera {camera_index} '
                f'({missing} positions had no modality image)')
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
                        help='Modality image input. Without --calibration: a '
                             'DIRECTORY of images for one rig camera, matched to '
                             'SfM views by the PGS-scan filename convention '
                             '{prefix}_{camera}_{position}_{capture}. With '
                             '--calibration: a SINGLE image file captured from '
                             'the calibrated pose (any filename).')
    parser.add_argument('--calibration', default=None,
                        help='A pgs-calibrate calibration .json (one localized '
                             'view). Textures the mesh from that new pose with '
                             'the single image given by -i, instead of reusing a '
                             'rig camera\'s positions. --camera-index/--sfm-data '
                             'are ignored in this mode.')
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
                             'convert the modality image to 8-bit sRGB via '
                             'ImageMagick before copying it as the texture. '
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
                             'name.')
    parser.add_argument('--camera-index', '-k', type=int, default=None,
                        help='Camera index the modality images correspond to. '
                             'Inferred from filenames if omitted.')
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

    # Resolve the final mesh path + format and derive the artifact stem.
    # --output-mesh sets the exact deliverable; its extension wins over
    # --file-type. The stem is prefixed onto every scratch artifact so they
    # coexist with the recon's files (there is no --name flag): the
    # --output-mesh stem if given, else the modality input's name (the image
    # directory in the default mode, the image stem in --calibration mode).
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
                else modality_input.resolve().name)

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
    if paths['output_mesh'].exists():
        logger.warning(f'Final mesh target already exists and will be '
                       f'overwritten: {paths["output_mesh"]}')

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

    metadata = {'args': ' '.join(sys.argv), 'parsed': vars(args),
                'commands': {}}
    paths['manifest'] = working_dir / f'{stem}_retexture.json'

    # Where the binaries are and what records their invocations: process-wide, so
    # no wrapper takes either as an argument (ADR 0005).
    recorder = Recorder(metadata['commands'])
    toolchain.configure(prefix=args.path, recorder=recorder)

    @atexit.register
    def write_metadata():
        metadata['paths'] = {k: str(v) for k, v in paths.items()}
        write_manifest(paths['manifest'], metadata)

    write_metadata()

    # 1-2. Build the single-camera SfM scene to texture from. Two modes:
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
        # Legacy mode: reuse a rig camera's solved positions, matched by the
        # PGS-scan filename convention.
        logger.info('Indexing modality images')
        camera_index, pos_map = index_modality_images(modality_input,
                                                      args.camera_index)
        logger.info('Preparing 8-bit modality images')
        pos_to_name = convert_modality_images(pos_map, paths['modality_8bit'],
                                              args.bit_shift)
        logger.info('Exporting SfM solution to JSON')
        sfm_to_json(sfm_data, paths['sfm_full'])
        logger.info('Filtering SfM scene to the modality camera')
        filter_sfm_for_camera(paths['sfm_full'], camera_index,
                              paths['modality_8bit'], pos_to_name,
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
