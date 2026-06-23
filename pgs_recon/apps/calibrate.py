"""Calibrate (localize) a new camera view against an existing reconstructed
pgs-recon scene, producing a reusable pose + intrinsic that ``pgs-retexture``
can use to texture the mesh from that view.

This is the companion of ``pgs-retexture`` for the case where the texturing
image comes from a camera that was **not** part of the original reconstruction
(e.g. an overhead registration camera), so there is no solved pose to reuse. We
resection the new image into the original solved frame with OpenMVG's image
localization, then extract just that one localized view (its pose and the freshly
estimated intrinsic) into a small ``*_calibration.json`` scene.

The point of separating calibration from texturing is multimodal capture: a
single physical camera position often yields several co-registered images (RGB,
IR, ...). They all share one pose and intrinsic, so we localize **once** (from
the image with the richest features, usually RGB) and reuse the calibration to
texture the mesh with each modality in turn via
``pgs-retexture --calibration <this file>``.

The pipeline is:

  1. Normalize the query image to an 8-bit sRGB file (handles 16-bit and
     non-RGB colorspaces such as CIELab) so OpenMVG can extract features from it.
  2. Resolve the solved SfM (the scene fed to openMVG2openMVS, which carries
     structure and lives in the mesh's frame) and the database regions from a
     completed ``pgs-recon`` output directory.
  3. ``openMVG_main_SfM_Localization`` resections the query image against that
     scene. New query regions go to a private match dir (``-u``) and the
     localized scene to a private output dir, so the original scene is untouched.
  4. Extract the single localized view -> ``<name>_calibration.json`` (one view,
     one intrinsic, one extrinsic, no structure) rooted at the normalized query
     image. This is the reusable calibration artifact.

The localized pose is expressed in the *solved frame* of the SfM that produced
the mesh, so the calibration aligns with the un-centered reconstruction mesh
(the same one ``pgs-retexture`` textures). Unlike ``pgs-retexture``'s
modality-at-same-positions mode, NO filename convention is required here: the
correspondence is geometric (feature matching), so the query image can have any
name.
"""
import atexit
import json
import logging
import shutil
import sys
from datetime import datetime as dt, timezone as tz
from pathlib import Path
from typing import Dict, Optional

import configargparse
import cv2

from pgs_recon.openmvg import mvg_localize, CameraModel, ResectionMethod
from pgs_recon.utils.apps import setup_logging
# Reuse the SfM-JSON surgery and image prep already proven in pgs-retexture.
from pgs_recon.apps.retexture import (
    _fix_polymorphic_registration,
    prepare_8bit_image,
    resolve_recon_inputs,
)

logger = logging.getLogger(__name__)

_IMG_EXTS = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}


def maybe_generate_mask(query_img: Path, mask_arg: Optional[str],
                        generate: bool, open_iterations: int) -> Optional[Path]:
    """Place an OpenMVG per-image mask next to the query image.

    OpenMVG looks for a ``<image_stem>_mask.png`` sibling and restricts feature
    detection to its non-zero region. ``mask_arg`` supplies an existing mask;
    ``generate`` derives one with the same tray segmentation as
    ``pgs-generate-mask``. Masking is opt-in and primarily constrains matching
    to the object (off by default). Returns the mask path or None.
    """
    if mask_arg is None and not generate:
        return None
    mask_path = query_img.with_name(f'{query_img.stem}_mask.png')
    if mask_arg is not None:
        shutil.copy(Path(mask_arg), mask_path)
        logger.info(f'Using provided mask: {mask_path}')
        return mask_path
    # generate==True
    from pgs_recon.utils import educelab
    img = cv2.imread(str(query_img))
    mask = educelab.generate_tray_mask(img, open_iterations=open_iterations)
    cv2.imwrite(str(mask_path), mask)
    logger.info(f'Generated mask: {mask_path}')
    return mask_path


def extract_calibration(expanded_json: Path, query_name: str,
                        query_dir: Path, out_json: Path) -> Dict:
    """Pull the single localized query view out of the expanded scene.

    Keeps only the view whose filename matches ``query_name``, prunes intrinsics
    and extrinsics down to the ones it references, repairs cereal polymorphic
    registration on the surviving intrinsic, drops structure, and roots the
    scene at ``query_dir``. Returns a small summary (focal/center) for logging.
    """
    data = json.loads(expanded_json.read_text())
    kept_view = None
    for v in data['views']:
        vd = v['value']['ptr_wrapper']['data']
        if Path(vd['filename']).name == query_name:
            kept_view = v
            break
    if kept_view is None:
        sys.exit(f'Localized scene does not contain the query view '
                 f'{query_name}; localization may have failed.')

    vd = kept_view['value']['ptr_wrapper']['data']
    id_intrinsic = vd['id_intrinsic']
    id_pose = vd['id_pose']
    if id_pose not in {e['key'] for e in data.get('extrinsics', [])}:
        sys.exit(f'Query view {query_name} was not localized (no pose '
                 f'recovered); try a feature-richer image or relax matching.')
    vd['local_path'] = ''

    intrinsics = [i for i in data.get('intrinsics', [])
                  if i['key'] == id_intrinsic]
    extrinsics = [e for e in data.get('extrinsics', []) if e['key'] == id_pose]

    out = dict(data)
    out['root_path'] = str(query_dir.resolve())
    out['views'] = [kept_view]
    out['intrinsics'] = _fix_polymorphic_registration(
        data.get('intrinsics', []), intrinsics)
    out['extrinsics'] = extrinsics
    out['structure'] = []
    out['control_points'] = []
    out_json.write_text(json.dumps(out, indent=2))

    idata = intrinsics[0]['value']['ptr_wrapper']['data']
    summary = {
        'focal_length': idata.get('focal_length'),
        'principal_point': idata.get('principal_point'),
        'width': idata.get('width'),
        'height': idata.get('height'),
        'center': extrinsics[0]['value']['center'],
    }
    focal = summary['focal_length']
    focal_str = f'{focal:.1f}px' if focal is not None else 'unknown'
    logger.info(f'Calibration written: {out_json}')
    logger.info(f"  focal={focal_str}  "
                f"pp={summary['principal_point']}  "
                f"center={summary['center']}")
    return summary


def main():
    parser = configargparse.ArgumentParser(
        prog='pgs-calibrate',
        description='Localize a new camera image against an existing pgs-recon '
                    'scene and emit a reusable pose+intrinsic calibration for '
                    'pgs-retexture.')
    parser.add_argument('--config', '-c', is_config_file=True,
                        help='Config file path')
    parser.add_argument('--image', '-i', required=True,
                        help='Image to localize against the scene (e.g. the '
                             'overhead RGB). For multimodal capture, localize '
                             'once from the feature-richest modality; the '
                             'resulting calibration is reused for the others.')
    parser.add_argument('--recon-dir', '-r', required=True,
                        help='A completed pgs-recon output directory. The solved '
                             'SfM (with structure) and database regions are '
                             'located from its metadata.json '
                             '(override the SfM with --sfm-data).')
    parser.add_argument('--sfm-data', '-s', default=None,
                        help='Override the solved OpenMVG SfM_Data (.bin/.json) '
                             'to localize against. Must carry structure and be '
                             'the frame the mesh lives in. Defaults to the SfM '
                             'that produced the mesh in --recon-dir.')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: '
                             '<recon-dir>/calibrate/<name>)')
    parser.add_argument('--name', '-n', default=None,
                        help='Calibration name (default derived from the image '
                             'filename)')
    parser.add_argument('--camera-model', type=int,
                        default=int(CameraModel.PINHOLE),
                        choices=[m.value for m in CameraModel],
                        help='OpenMVG camera model for the new (unknown) '
                             'intrinsic: '
                             + ', '.join(f'{m.value}={m.name.lower()}'
                                         for m in CameraModel)
                             + '. Default %(default)s (pinhole): long-focal '
                             'overhead views have negligible distortion and '
                             'higher models tend to overfit it under DLT '
                             'resection.')
    parser.add_argument('--resection-method', type=int,
                        default=int(ResectionMethod.DLT),
                        choices=[m.value for m in ResectionMethod],
                        help='OpenMVG resection method: '
                             + ', '.join(f'{m.value}={m.name.lower()}'
                                         for m in ResectionMethod)
                             + '. Default %(default)s (dlt): recovers focal '
                             'length and works for an uncalibrated camera; '
                             'P3P methods assume a known intrinsic.')
    parser.add_argument('--residual-error', type=float, default=None,
                        help='Upper bound on the resection residual error '
                             '(OpenMVG -r); lower is stricter.')
    parser.add_argument('--single-intrinsics', action='store_true',
                        help='Reuse the scene\'s single intrinsic for the query '
                             'instead of estimating a new one. Only valid if the '
                             'query was taken with a camera already in the scene '
                             'with one shared intrinsic.')
    parser.add_argument('--focal-length', type=float, default=None,
                        help='Known focal length in PIXELS for the camera, if '
                             'calibrated. Recorded for reference; with a known '
                             'focal you may also pick a P3P --resection-method.')
    parser.add_argument('--sensor-width', type=float, default=None,
                        help='Known sensor width in mm (reference only).')
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Known pixel pitch (reference only).')
    parser.add_argument('--mask', default=None,
                        help='Existing mask image to restrict feature matching '
                             'to the object region (opt-in).')
    parser.add_argument('--generate-mask', action='store_true',
                        help='Generate a tray mask for the query image (same '
                             'segmentation as pgs-generate-mask) to restrict '
                             'matching to the object region (opt-in).')
    parser.add_argument('--mask-open-iterations', type=int, default=4,
                        help='Morphological open iterations for --generate-mask.')
    parser.add_argument('--threads', type=int, default=None,
                        help='Threads for localization')
    parser.add_argument('--path', type=str, default='/usr/local/',
                        help=configargparse.SUPPRESS)
    parser.add_argument('--log-level', default='INFO', type=str.upper,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    setup_logging(args.log_level)
    global logger
    logger = logging.getLogger('pgs-calibrate')

    image = Path(args.image)
    if image.suffix.lower() not in _IMG_EXTS:
        sys.exit(f'Unsupported image type: {image}')
    recon_dir = Path(args.recon_dir)
    if args.name is None:
        args.name = image.stem

    # The mesh is not needed for calibration, but reusing resolve_recon_inputs
    # keeps "which SfM produced the mesh" logic in one place; we use only the SfM.
    sfm_default, _mesh = resolve_recon_inputs(recon_dir)
    sfm_data = Path(args.sfm_data) if args.sfm_data else sfm_default

    output = (Path(args.output) if args.output
              else recon_dir / 'calibrate' / args.name)
    output.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {'PATH': Path(args.path).resolve()}
    paths['BIN'] = paths['PATH'] / 'bin'
    paths['matches_dir'] = recon_dir / 'mvg' / 'matches_dir'
    paths['sfm_db'] = sfm_data
    paths['query_dir'] = output / 'query'
    paths['match_out'] = output / 'query_matches'
    paths['loc'] = output / 'localization'
    paths['calibration'] = output / f'{args.name}_calibration.json'
    for k in ('match_out', 'loc'):
        paths[k].mkdir(parents=True, exist_ok=True)

    if not paths['matches_dir'].is_dir():
        sys.exit(f'Database regions not found: {paths["matches_dir"]}')

    # Config + metadata, mirroring pgs-recon / pgs-retexture conventions.
    datetime_str = dt.now(tz.utc).strftime('%Y%m%d%H%M%S')
    config_path = output / f'{datetime_str}_{args.name}_calibrate_config.txt'
    args.config = str(config_path)
    with config_path.open('w') as f:
        for arg in vars(args):
            f.write(f"{arg.replace('_', '-')} = {getattr(args, arg)}\n")

    metadata = {'args': ' '.join(sys.argv), 'parsed': vars(args), 'commands': {}}
    paths['metadata'] = output / f'{args.name}_calibrate_metadata.json'

    @atexit.register
    def write_metadata():
        metadata['paths'] = {k: str(v) for k, v in paths.items()}
        with paths['metadata'].open('w') as mf:
            mf.write(json.dumps(metadata, indent=4, sort_keys=False))

    write_metadata()

    # 1. Normalize the query image and (optionally) mask it.
    logger.info('Preparing query image')
    query_img = prepare_8bit_image(image, paths['query_dir'])
    maybe_generate_mask(query_img, args.mask, args.generate_mask,
                        args.mask_open_iterations)

    # 2. Localize against the solved scene (original scene left untouched).
    logger.info(f'Localizing {query_img.name} against {sfm_data.name}')
    mvg_localize(paths, sfm_key='sfm_db', query_key='query_dir',
                 out_key='loc', match_out_key='match_out',
                 camera_model=args.camera_model,
                 resection_method=args.resection_method,
                 residual_error=args.residual_error,
                 single_intrinsics=args.single_intrinsics,
                 threads=args.threads, metadata=metadata)

    # 3. Extract the single localized view as the reusable calibration.
    logger.info('Extracting calibration (pose + intrinsic)')
    extract_calibration(paths['sfm_expanded'], query_img.name,
                        paths['query_dir'], paths['calibration'])

    logger.info(f'Done. Calibration: {paths["calibration"]}')
    logger.info('Texture each modality via: pgs-retexture --calibration '
                f'{paths["calibration"]} -r {recon_dir} -i <modality-image>')


if __name__ == '__main__':
    main()
