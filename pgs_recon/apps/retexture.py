"""Re-texture an existing reconstructed mesh using an alternate imaging
modality (e.g. IR940) captured at the same camera positions as one of the
cameras in the original reconstruction.

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
``metadata.json`` — the solved SfM fed to openMVG2openMVS (after any
robust/autoscale step, NOT the rig-prior import) and the textured mesh as
output by TextureMesh (before any centering transform). ``--sfm-data`` can
override the SfM if needed.

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

from pgs_recon.openmvg import mvg_to_mvs
from pgs_recon.openmvs import mvs_texture
from pgs_recon.utility import current_timestamp, run_command
from pgs_recon.utils.apps import setup_logging

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


def sfm_to_json(sfm_path: Path, out_json: Path, bin_dir: Path,
                metadata: Dict = None) -> Path:
    """Export an OpenMVG SfM_Data (.bin/.json) to JSON with only views,
    intrinsics, and extrinsics (drops structure/control points). A ``.json``
    input is re-exported anyway to strip structure and normalize."""
    command = [
        str(bin_dir / 'openMVG_main_ConvertSfM_DataFormat'),
        '-i', str(sfm_path.resolve()),
        '-o', str(out_json.resolve()),
        '-V', '-I', '-E',
    ]
    if metadata is not None:
        metadata['commands'][current_timestamp()] = ' '.join(command)
    run_command(command)
    return out_json


_POLY_FLAG = 0x80000000


def _fix_polymorphic_registration(all_items, kept_items):
    """Repair cereal polymorphic-pointer registration after filtering.

    In openMVG's cereal JSON the first instance of each polymorphic type sets
    the high bit on ``polymorphic_id`` and carries a ``polymorphic_name``;
    later instances reference the type by its bare numeric id. If filtering
    drops the registering instance, surviving instances reference an
    unregistered type id and the scene fails to load. This promotes the first
    kept instance of each type back to the registration form.
    """
    # Map type-number -> registered name from the full (original) list.
    registry = {}
    for it in all_items:
        pid = it['value'].get('polymorphic_id', 0)
        if pid & _POLY_FLAG:
            registry[pid & ~_POLY_FLAG] = it['value'].get('polymorphic_name')

    seen = set()
    for it in kept_items:
        val = it['value']
        pid = val.get('polymorphic_id', 0)
        typenum = (pid & ~_POLY_FLAG) if (pid & _POLY_FLAG) else pid
        if typenum not in seen:
            val['polymorphic_id'] = _POLY_FLAG | typenum
            if registry.get(typenum) is not None:
                val['polymorphic_name'] = registry[typenum]
            seen.add(typenum)
        else:
            val.pop('polymorphic_name', None)
            val['polymorphic_id'] = typenum
    return kept_items


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
    data['intrinsics'] = _fix_polymorphic_registration(
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


def ensure_ply_mesh(mesh_path: Path, work_dir: Path) -> Path:
    """Stage a mesh into ``work_dir`` in a form OpenMVS can reliably load.

    OpenMVS' OBJ reader is strict (and mis-resolves a relative ``mtllib`` when
    a working folder is set), so OBJ inputs are converted to a geometry-only
    binary PLY (vertices + triangles). Texture coordinates/materials are
    irrelevant here because TextureMesh regenerates its own UVs. PLY inputs are
    copied through unchanged. Either way the result lives in ``work_dir`` so
    TextureMesh can reference it by basename.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    if mesh_path.suffix.lower() == '.ply':
        out = work_dir / mesh_path.name
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
    out = work_dir / (mesh_path.stem + '.ply')
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


def resolve_recon_inputs(recon_dir: Path):
    """Locate the solved SfM and textured mesh inside a pgs-recon output dir.

    Reads ``<recon_dir>/metadata.json`` but rebuilds every path *relative to*
    ``recon_dir``: the absolute paths recorded there may belong to another
    runtime (e.g. a Docker mount), so they are not trusted directly.

    The SfM that produced the mesh is the one fed to openMVG2openMVS (after any
    robust-triangulation / autoscale step, before colorize); we recover its
    basename from that command and re-root it under ``mvg/recon_dir``. The
    textured mesh is ``mvs/<name>.<file_type>`` from the run's parsed args.
    Returns ``(sfm_path, mesh_path)``.
    """
    meta_path = recon_dir / 'metadata.json'
    if not meta_path.is_file():
        sys.exit(f'No metadata.json in {recon_dir}; '
                 f'is this a pgs-recon output directory?')
    meta = json.loads(meta_path.read_text())

    sfm_name = None
    for cmd in meta.get('commands', {}).values():
        if 'openMVG2openMVS' in cmd:
            toks = cmd.split()
            if '-i' in toks:
                sfm_name = Path(toks[toks.index('-i') + 1]).name
    if sfm_name is None:
        sys.exit(f'{meta_path} records no openMVG2openMVS step; the run had no '
                 f'MVS stage (--no-mvs?) and cannot be re-textured.')
    sfm_path = recon_dir / 'mvg' / 'recon_dir' / sfm_name

    parsed = meta.get('parsed', {})
    name, file_type = parsed.get('name'), parsed.get('file_type')
    if not name or not file_type:
        sys.exit(f'{meta_path} is missing name/file_type; '
                 f'cannot locate the textured mesh.')
    mesh_path = recon_dir / 'mvs' / f'{name}.{file_type}'

    for p in (sfm_path, mesh_path):
        if not p.is_file():
            sys.exit(f'Expected reconstruction artifact not found: {p}')
    logger.info(f'Resolved from {recon_dir}: sfm={sfm_path.name}, '
                f'mesh={mesh_path.name}')
    return sfm_path, mesh_path


def main():
    parser = configargparse.ArgumentParser(
        prog='pgs-retexture',
        description='Re-texture an existing mesh with an alternate imaging '
                    'modality captured at one camera\'s positions.')
    parser.add_argument('--config', '-c', is_config_file=True,
                        help='Config file path')
    parser.add_argument('--modality-images', '-i', required=True,
                        help='Directory of modality images (tif/jpg/png) for '
                             'one camera. 16-bit images are converted to 8-bit. '
                             'Filenames MUST follow the PGS-scan convention '
                             '{prefix}_{camera}_{position}_{capture}; images are '
                             'matched to SfM views by (camera, position).')
    parser.add_argument('--recon-dir', '-r', required=True,
                        help='A completed pgs-recon output directory. The solved '
                             'SfM and the textured mesh are located from its '
                             'metadata.json (override with --sfm-data). The recon '
                             'must have used PGS-named images (its SfM view '
                             'filenames must match the convention above).')
    parser.add_argument('--sfm-data', '-s', default=None,
                        help='Override the solved OpenMVG SfM_Data (.bin/.json) '
                             'to texture from. Defaults to the SfM that produced '
                             'the mesh in --recon-dir. NOT the rig-prior import '
                             'scene.')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: '
                             '<recon-dir>/retexture/<name>)')
    parser.add_argument('--name', '-n', default=None,
                        help='Output mesh basename (default derived from the '
                             'modality directory name)')
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
    parser.add_argument('--path', type=str, default='/usr/local/',
                        help=configargparse.SUPPRESS)
    parser.add_argument('--log-level', default='INFO', type=str.upper,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    setup_logging(args.log_level)
    global logger
    logger = logging.getLogger('pgs-retexture')

    modality_dir = Path(args.modality_images)
    recon_dir = Path(args.recon_dir)
    if args.name is None:
        args.name = modality_dir.resolve().name

    # Locate the SfM solution and mesh from the reconstruction (unless the SfM
    # is overridden); the mesh always comes from the reconstruction.
    sfm_default, mesh_in = resolve_recon_inputs(recon_dir)
    sfm_data = Path(args.sfm_data) if args.sfm_data else sfm_default

    output = Path(args.output) if args.output else recon_dir / 'retexture' / args.name
    output.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {
        'PATH': Path(args.path).resolve(),
    }
    paths['BIN'] = paths['PATH'] / 'bin'
    paths['MVS_BIN'] = paths['BIN'] / 'OpenMVS'

    # Working layout
    paths['output'] = output
    paths['modality_8bit'] = output / f'{args.name}_modality'
    paths['mvs'] = output / 'mvs'
    paths['mvs'].mkdir(parents=True, exist_ok=True)
    paths['mvs_scene'] = paths['mvs'] / f'{args.name}_scene.mvs'
    paths['mvs_images'] = paths['mvs'] / 'undistorted_images'
    paths['sfm_full'] = output / 'sfm_full.json'
    paths['sfm_filtered'] = output / f'{args.name}_sfm.json'

    # Config + metadata, mirroring pgs-recon conventions
    datetime_str = dt.now(tz.utc).strftime('%Y%m%d%H%M%S')
    config_path = output / f'{datetime_str}_{args.name}_retexture_config.txt'
    args.config = str(config_path)
    with config_path.open('w') as f:
        for arg in vars(args):
            f.write(f"{arg.replace('_', '-')} = {getattr(args, arg)}\n")

    metadata = {'args': ' '.join(sys.argv), 'parsed': vars(args),
                'commands': {}}
    paths['metadata'] = output / f'{args.name}_retexture_metadata.json'

    @atexit.register
    def write_metadata():
        metadata['paths'] = {k: str(v) for k, v in paths.items()}
        with paths['metadata'].open('w') as mf:
            mf.write(json.dumps(metadata, indent=4, sort_keys=False))

    write_metadata()

    # 1. Index + convert modality images
    logger.info('Indexing modality images')
    camera_index, pos_map = index_modality_images(modality_dir,
                                                  args.camera_index)
    logger.info('Preparing 8-bit modality images')
    pos_to_name = convert_modality_images(pos_map, paths['modality_8bit'],
                                          args.bit_shift)

    # 2. Export + filter the SfM solution
    logger.info('Exporting SfM solution to JSON')
    sfm_to_json(sfm_data, paths['sfm_full'], paths['BIN'],
                metadata=metadata)
    logger.info('Filtering SfM scene to the modality camera')
    filter_sfm_for_camera(paths['sfm_full'], camera_index,
                          paths['modality_8bit'], pos_to_name,
                          paths['sfm_filtered'])

    # 3. MVG -> MVS (undistorts modality images with original intrinsics)
    logger.info('Building MVS scene from modality views')
    paths['sfm_ir'] = paths['sfm_filtered']
    mvg_to_mvs(paths, sfm_key='sfm_ir', threads=args.threads,
               metadata=metadata)

    # 4. Texture the existing mesh
    logger.info('Preparing mesh for OpenMVS')
    paths['mesh'] = ensure_ply_mesh(mesh_in, paths['mvs'])
    logger.info('Texturing mesh with modality images')
    out_key = mvs_texture(paths, mvs_key='mvs_scene', mesh_key='mesh',
                          file_format=args.file_type,
                          resolution_lvl=args.texture_resolution_level,
                          max_size=args.max_texture_size,
                          empty_color=args.empty_color,
                          global_seam_leveling=args.global_seam_leveling,
                          local_seam_leveling=args.local_seam_leveling,
                          output_name=args.name, metadata=metadata)

    logger.info(f'Done. Re-textured mesh: {paths[out_key]}')


if __name__ == '__main__':
    main()
