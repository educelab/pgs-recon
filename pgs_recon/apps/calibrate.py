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
     If a known intrinsic is given (``--intrinsic`` for a full precalibrated K +
     distortion, or ``--focal-length`` / ``--focal-length-mm`` for a focal only),
     the query is resectioned with that fixed K via a stable P3P method;
     otherwise an unknown intrinsic is estimated by DLT (which can go degenerate
     on sparse matches -- see ``validate_localized_intrinsic``).
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
import numpy as np

from pgs_recon.openmvg import mvg_localize, CameraModel, ResectionMethod
from pgs_recon.utility import current_timestamp, run_command
from pgs_recon.utils.apps import setup_logging
# Reuse the SfM-JSON surgery and image prep already proven in pgs-retexture.
from pgs_recon.apps.retexture import (
    _camera_from_calibration,
    _fix_polymorphic_registration,
    prepare_8bit_image,
    resolve_recon_inputs,
    transform_extrinsic,
)

logger = logging.getLogger(__name__)

_IMG_EXTS = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}


def save_camera_file(calibration_json: Path, output_path: Path) -> None:
    """Write the calibration in the shared pgs-recon camera-file format.

    This is the same flat key/value format ``--intrinsic`` reads (see
    ``parse_intrinsic_file``): ``fx fy cx cy width height`` (pixels), optional
    radial distortion ``k1 k2 k3`` (only written when non-zero), and ``pose`` as
    the 4x4 world-to-camera matrix in OpenCV convention (x_cam = R*X + t),
    row-major. So a file written here can be fed straight back into
    ``--intrinsic`` (which uses the intrinsic and ignores the pose).
    """
    R, C, f, cx, cy, W, H, disto = _camera_from_calibration(calibration_json)
    t = -R @ C
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = t
    pose_vals = ' '.join([str(v) for v in pose.flatten().tolist()])
    with output_path.open('w') as fh:
        fh.write(f'fx {f}\n')
        fh.write(f'fy {f}\n')
        fh.write(f'cx {cx}\n')
        fh.write(f'cy {cy}\n')
        fh.write(f'width {int(W)}\n')
        fh.write(f'height {int(H)}\n')
        if disto is not None and any(disto):
            for name, val in zip(('k1', 'k2', 'k3'), disto):
                fh.write(f'{name} {val}\n')
        fh.write(f'pose {pose_vals}\n')
    logger.info(f'Saved camera calibration file: {output_path}')


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


def parse_intrinsic_file(path: Path) -> Dict:
    """Parse a registration-toolkit-style key/value intrinsic file.

    Mirrors the camera file written by ``--save-camera-file`` (whitespace-separated
    ``key value`` lines: ``fx fy cx cy width height``) and additionally accepts
    radial distortion coefficients ``k1 k2 k3`` (OpenCV/OpenMVG order). Any other
    keys (e.g. ``pose``) are ignored, so a full rt camera file can be passed
    directly and only its intrinsic is read. Focal and principal point are in
    pixels at the file's ``width`` x ``height`` resolution.

    Returns a dict: ``fx, fy, cx, cy, width, height`` (floats) and ``disto`` (a
    list of up to 3 radial coeffs, or None if no distortion keys were present).
    Tangential terms (``p1``/``p2``) are rejected: the localization camera model
    is radial-only.
    """
    scalars: Dict[str, float] = {}
    disto: Dict[str, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        key = parts[0].lower()
        if key in ('fx', 'fy', 'cx', 'cy', 'width', 'height'):
            scalars[key] = float(parts[1])
        elif key in ('k1', 'k2', 'k3'):
            disto[key] = float(parts[1])
        elif key in ('p1', 'p2') and float(parts[1]) != 0.0:
            sys.exit(f'{path}: tangential distortion ({key}) is not supported '
                     f'by the localization camera model; only radial k1/k2/k3 '
                     f'are honored.')
    missing = [k for k in ('fx', 'cx', 'cy', 'width', 'height')
               if k not in scalars]
    if missing:
        sys.exit(f'{path}: intrinsic file is missing required keys: '
                 f'{", ".join(missing)}')
    fx = scalars['fx']
    disto_list = None
    if disto:
        disto_list = [disto.get('k1', 0.0), disto.get('k2', 0.0),
                      disto.get('k3', 0.0)]
    return {'fx': fx, 'fy': scalars.get('fy', fx),
            'cx': scalars['cx'], 'cy': scalars['cy'],
            'width': scalars['width'], 'height': scalars['height'],
            'disto': disto_list}


def resolve_known_intrinsic(args, query_w: int, query_h: int) -> Dict:
    """Resolve the supplied query intrinsic to a pixel-space spec at the query
    resolution: ``{focal, cx, cy, disto}``. ``cx``/``cy`` None means a centered
    principal point; ``disto`` None means no distortion. Assumes exactly one
    focal source was set (validated in ``main``).

    A full ``--intrinsic`` file carries focal, principal point and radial
    distortion, scaled from the file's resolution to the query's (distortion
    coefficients are dimensionless and copied as-is). ``--focal-length`` is taken
    as pixels directly; ``--focal-length-mm`` is converted to pixels via
    ``--pixel-size`` (mm/px) or ``--sensor-width`` (mm). The focal-only sources
    imply a centered principal point and no distortion -- appropriate for the
    long-focal overhead cameras this targets.
    """
    if args.intrinsic is not None:
        spec = parse_intrinsic_file(Path(args.intrinsic))
        fw, fh = spec['width'], spec['height']
        sx, sy = query_w / fw, query_h / fh
        if abs(sx - sy) / max(sx, sy) > 0.01:
            logger.warning('Intrinsic file resolution %gx%g differs in aspect '
                           'from query %dx%d; scaling focal/pp per axis.',
                           fw, fh, query_w, query_h)
        fx, fy = spec['fx'] * sx, spec['fy'] * sy
        focal = (fx + fy) / 2.0
        if abs(fx - fy) / max(fx, fy) > 0.01:
            logger.warning('Non-square pixels (fx=%.1f, fy=%.1f); OpenMVG uses '
                           'a single focal, averaging to %.1f.', fx, fy, focal)
        return {'focal': focal, 'cx': spec['cx'] * sx, 'cy': spec['cy'] * sy,
                'disto': spec['disto']}
    if args.focal_length is not None:
        return {'focal': args.focal_length, 'cx': None, 'cy': None,
                'disto': None}
    # --focal-length-mm: convert to pixels.
    if args.pixel_size:
        focal = args.focal_length_mm / args.pixel_size
        logger.info('Focal %.3fmm / pixel size %.6fmm/px -> %.1fpx',
                    args.focal_length_mm, args.pixel_size, focal)
    elif args.sensor_width:
        focal = args.focal_length_mm * query_w / args.sensor_width
        logger.info('Focal %.3fmm * %dpx / sensor %.3fmm -> %.1fpx',
                    args.focal_length_mm, query_w, args.sensor_width, focal)
    else:
        sys.exit('--focal-length-mm requires --pixel-size (mm per pixel) or '
                 '--sensor-width (mm) to convert the focal to pixels.')
    return {'focal': focal, 'cx': None, 'cy': None, 'disto': None}


def _apply_distortion(idata: Dict, disto, sfm_path: Path) -> None:
    """Set the lone intrinsic's radial distortion in place.

    ``disto`` None is the calibrated-no-distortion case: every distortion field
    is zeroed (a known focal implies a calibrated camera, and P3P consumes
    undistorted bearings). When real radial coefficients are supplied they are
    written into whichever radial field the templated intrinsic carries
    (``disto_k3`` -> [k1,k2,k3], ``disto_k1`` -> [k1]); OpenMVG then undistorts
    the query features with them before P3P. Exits if distortion is requested but
    the scene's camera model has no radial field to hold it.
    """
    fields = [k for k, v in idata.items()
              if k.startswith('disto') and isinstance(v, list)]
    if not disto:
        for k in fields:
            idata[k] = [0.0] * len(idata[k])
        return
    if 'disto_k3' in idata:
        idata['disto_k3'] = [disto[0],
                             disto[1] if len(disto) > 1 else 0.0,
                             disto[2] if len(disto) > 2 else 0.0]
        for k in fields:
            if k != 'disto_k3':
                idata[k] = [0.0] * len(idata[k])
    elif 'disto_k1' in idata:
        if any(c != 0.0 for c in disto[1:]):
            logger.warning('Scene camera model carries only one radial '
                           'coefficient (disto_k1); higher-order terms %s '
                           'dropped.', disto[1:])
        idata['disto_k1'] = [disto[0]]
    else:
        sys.exit(f'Distortion coefficients were provided but the scene camera '
                 f'model in {sfm_path} has no radial field (disto_k1/disto_k3) '
                 f'to hold them. Re-run without distortion, or against a '
                 f'reconstruction that used a radial camera model.')


def build_single_intrinsic_scene(sfm_path: Path, query_w: int, query_h: int,
                                 focal: float, bin_dir: Path, out_json: Path,
                                 cx: float = None, cy: float = None,
                                 disto=None, metadata: Dict = None) -> Path:
    """Rewrite the solved scene to carry a single known query-camera intrinsic.

    OpenMVG's localizer honors a P3P resection method (and a fixed, known focal)
    only when an intrinsic is supplied for the query, and the stock binary
    exposes that solely through ``--single-intrinsics``: it reuses the scene's
    *one* intrinsic for the query image. The solved scene was built from the rig
    cameras and carries several intrinsics, none matching the (different) query
    camera -- but localization never uses the database views' intrinsics
    (``Init`` builds its retrieval database from the already-solved 3D landmarks
    and their descriptors). So collapsing every intrinsic to the one that
    actually matters for resection -- the query camera's -- is exact, not an
    approximation: the rig intrinsics already did their job when the structure
    was solved.

    The lone intrinsic is rebuilt at the query resolution with the given focal
    (in pixels). By default the principal point is centered and distortion is
    zeroed (the calibrated long-focal case); pass ``cx``/``cy`` and/or ``disto``
    to honor a full precalibrated intrinsic, in which case OpenMVG undistorts the
    query features with those radial coefficients before resection. ``-s`` then
    localizes the query with that fixed K via the chosen P3P method. Every view
    is repointed at it so the binary's "exactly one intrinsic" check passes (the
    rig views are discarded after localization).

    Structure must stay in the scene (the localizer requires landmarks), so the
    full converted scene is loaded; for very large reconstructions this can take
    a minute and a few GB of RAM.
    """
    full = out_json.with_name(f'{out_json.stem}_full.json')
    command = [
        str(bin_dir / 'openMVG_main_ConvertSfM_DataFormat'),
        '-i', str(sfm_path.resolve()), '-o', str(full.resolve()),
        '-V', '-I', '-E', '-S', '-C',
    ]
    if metadata is not None:
        metadata['commands'][current_timestamp()] = ' '.join(command)
    run_command(command)

    data = json.loads(full.read_text())
    intr_list = data.get('intrinsics', [])
    if not intr_list:
        sys.exit(f'Solved scene {sfm_path} has no intrinsic to template from')
    # Reuse the first (registering) intrinsic object so cereal's polymorphic
    # pointer registration stays valid, then override it to the query camera.
    # The key MUST be 0: OpenMVG's --single-intrinsics path reads the query
    # camera via GetIntrinsics().at(0) (main_SfM_Localization.cpp), a hardcoded
    # id-0 lookup that throws if the lone intrinsic is keyed otherwise.
    intr = intr_list[0]
    key = 0
    intr['key'] = key
    idata = intr['value']['ptr_wrapper']['data']
    idata['width'] = int(query_w)
    idata['height'] = int(query_h)
    idata['focal_length'] = float(focal)
    pp_x = query_w / 2.0 if cx is None else float(cx)
    pp_y = query_h / 2.0 if cy is None else float(cy)
    idata['principal_point'] = [pp_x, pp_y]
    _apply_distortion(idata, disto, sfm_path)
    data['intrinsics'] = [intr]
    for v in data['views']:
        v['value']['ptr_wrapper']['data']['id_intrinsic'] = key
    out_json.write_text(json.dumps(data))
    logger.info('Built single-intrinsic localization scene (%dx%d, '
                'focal=%.1fpx, pp=[%.1f, %.1f]%s) -> %s',
                query_w, query_h, focal, pp_x, pp_y,
                '' if not disto else f', disto={disto}', out_json)
    return out_json


def validate_localized_intrinsic(summary: Dict) -> None:
    """Fail loudly if the localized intrinsic is non-physical.

    A degenerate resection (most often DLT recovering an unknown focal from too
    few or ill-distributed correspondences) produces an off-image principal
    point and/or an absurd focal. Such a calibration loads fine but misaligns the
    texture, so reject it here instead of emitting it silently.
    """
    f = summary.get('focal_length')
    pp = summary.get('principal_point') or [None, None]
    W, H = summary.get('width'), summary.get('height')
    if f is None or not np.isfinite(f) or f <= 0:
        sys.exit(f'Localization produced a non-physical focal length ({f}); the '
                 f'resection is degenerate. Provide --focal-length (in pixels) '
                 f'to switch to a stable P3P resection, or use a feature-richer '
                 f'query image / a --mask.')
    cx, cy = pp
    if W is None or H is None or cx is None or cy is None \
            or not (0 <= cx <= W) or not (0 <= cy <= H):
        sys.exit(f'Localization produced a principal point {pp} outside the '
                 f'image bounds ({W}x{H}); the recovered camera is degenerate '
                 f'(typical of DLT resection on sparse matches). Provide '
                 f'--focal-length (in pixels) to switch to a stable P3P '
                 f'resection, or use a feature-richer query image / a --mask.')


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
    validate_localized_intrinsic(summary)
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
                             'resection. Only applies to the uncalibrated DLT '
                             'path; ignored when a known intrinsic is supplied '
                             '(P3P uses the provided/scene K).')
    parser.add_argument('--resection-method', type=int, default=None,
                        choices=[m.value for m in ResectionMethod],
                        help='OpenMVG resection method: '
                             + ', '.join(f'{m.value}={m.name.lower()}'
                                         for m in ResectionMethod)
                             + '. Default: dlt when the camera is uncalibrated '
                             '(recovers focal length, but is unstable on sparse '
                             'matches), or p3p_nordberg when a known intrinsic '
                             'is supplied via --intrinsic / --focal-length / '
                             '--focal-length-mm / --single-intrinsics. P3P/UPnP '
                             'methods REQUIRE a known '
                             'intrinsic (OpenMVG silently falls back to DLT '
                             'without one).')
    parser.add_argument('--residual-error', type=float, default=None,
                        help='Upper bound on the resection residual error '
                             '(OpenMVG -r); lower is stricter.')
    parser.add_argument('--single-intrinsics', action='store_true',
                        help='Reuse the scene\'s single intrinsic for the query '
                             'instead of estimating a new one. Only valid if the '
                             'query was taken with a camera already in the scene '
                             'with one shared intrinsic.')
    parser.add_argument('--intrinsic', default=None,
                        help='Precalibrated intrinsic for the query camera as a '
                             'pgs-recon camera file (the shared format written by '
                             '--save-camera-file: fx fy cx cy width height, plus '
                             'optional radial distortion k1 k2 k3; any pose line '
                             'is ignored). Honors the full K and distortion -- '
                             'OpenMVG undistorts the query before P3P -- and is '
                             'scaled to the query resolution. Mutually exclusive '
                             'with --focal-length / --focal-length-mm.')
    parser.add_argument('--focal-length', type=float, default=None,
                        help='Known focal length in PIXELS for the query camera. '
                             'When given, the query is localized with this fixed '
                             'focal via a P3P resection (centered principal '
                             'point, no distortion) instead of letting DLT '
                             'estimate an unknown -- and often degenerate -- '
                             'intrinsic. Strongly recommended for long-focal '
                             'overhead cameras with few feature matches.')
    parser.add_argument('--focal-length-mm', type=float, default=None,
                        help='Known focal length in MILLIMETERS, converted to '
                             'pixels via --pixel-size (mm/px) or --sensor-width '
                             '(mm). Same P3P path as --focal-length (centered '
                             'principal point, no distortion).')
    parser.add_argument('--sensor-width', type=float, default=None,
                        help='Sensor width in mm. Used with --focal-length-mm to '
                             'convert a mm focal to pixels '
                             '(f_px = f_mm * image_width_px / sensor_width).')
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Pixel pitch in mm per pixel. Used with '
                             '--focal-length-mm to convert a mm focal to pixels '
                             '(f_px = f_mm / pixel_size).')
    parser.add_argument('--sfm-transform', default=None,
                        help='4x4 .npy transform matrix saved by pgs-center '
                             '--save-transform. Re-expresses the localized pose '
                             'in the centered mesh coordinate frame. Use when '
                             'the mesh to be textured was centered by pgs-center '
                             'after reconstruction.')
    parser.add_argument('--save-camera-file', default=None,
                        help='Write the calibrated camera to this path in the '
                             'shared pgs-recon camera-file format (key/value '
                             'text: fx/fy/cx/cy/width/height, radial k1/k2/k3 if '
                             'non-zero, and pose as a 4x4 world-to-camera '
                             'matrix). Consumed by registration-toolkit and '
                             'reusable as an --intrinsic input. Written after '
                             '--sfm-transform is applied if both are set.')
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

    # The query intrinsic can be supplied three mutually exclusive ways: a full
    # --intrinsic file, a focal in pixels (--focal-length), or a focal in mm
    # (--focal-length-mm). Each builds a single-intrinsic localization scene.
    focal_sources = [n for n, v in (('--intrinsic', args.intrinsic),
                                    ('--focal-length', args.focal_length),
                                    ('--focal-length-mm', args.focal_length_mm))
                     if v is not None]
    if len(focal_sources) > 1:
        sys.exit(f'Provide only one of {", ".join(focal_sources)}; they are '
                 f'mutually exclusive ways to specify the query intrinsic.')
    provide_intrinsic = bool(focal_sources)
    if provide_intrinsic and args.single_intrinsics:
        sys.exit('--single-intrinsics reuses the scene intrinsic and cannot be '
                 'combined with --intrinsic/--focal-length/--focal-length-mm.')

    # Resolve the resection method. A known intrinsic (a supplied focal/intrinsic,
    # or reusing the scene's lone intrinsic) lets OpenMVG run a stable P3P;
    # without one it can only run DLT, which estimates the focal and goes
    # degenerate on sparse matches. The stock binary silently ignores a P3P -R
    # when no intrinsic is known, so guard the combination explicitly here.
    known_intrinsic = provide_intrinsic or args.single_intrinsics
    resection_method = args.resection_method
    if resection_method is None:
        resection_method = (int(ResectionMethod.P3P_NORDBERG) if known_intrinsic
                            else int(ResectionMethod.DLT))
    if resection_method != int(ResectionMethod.DLT) and not known_intrinsic:
        sys.exit(f'--resection-method {resection_method} '
                 f'({ResectionMethod(resection_method).name.lower()}) needs a '
                 f'known intrinsic, but none was given. Pass --intrinsic, '
                 f'--focal-length (px), --focal-length-mm, or --single-'
                 f'intrinsics; otherwise OpenMVG silently falls back to DLT.')
    if resection_method == int(ResectionMethod.DLT) and provide_intrinsic:
        sys.exit(f'A known intrinsic was provided ({focal_sources[0]}) but '
                 f'--resection-method is dlt, which ignores it and re-estimates. '
                 f'Choose a P3P method (e.g. 3 = p3p_nordberg) or drop '
                 f'{focal_sources[0]}.')

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

    # 1b. With a supplied intrinsic, synthesize a single-intrinsic localization
    # scene so OpenMVG resections the query with that fixed K via P3P (see
    # build_single_intrinsic_scene). The query JPG keeps the source resolution,
    # which the intrinsic must match for OpenMVG's -s dimension check.
    single_intrinsics = args.single_intrinsics
    if provide_intrinsic:
        qimg = cv2.imread(str(query_img), cv2.IMREAD_UNCHANGED)
        if qimg is None:
            sys.exit(f'Could not read prepared query image: {query_img}')
        qh, qw = qimg.shape[:2]
        spec = resolve_known_intrinsic(args, qw, qh)
        logger.info('Building single-intrinsic scene for known-intrinsic P3P '
                    'localization')
        paths['loc_scene'] = output / f'{args.name}_loc_scene.json'
        build_single_intrinsic_scene(sfm_data, qw, qh, spec['focal'],
                                     paths['BIN'], paths['loc_scene'],
                                     cx=spec['cx'], cy=spec['cy'],
                                     disto=spec['disto'], metadata=metadata)
        paths['sfm_db'] = paths['loc_scene']
        single_intrinsics = True

    # 2. Localize against the solved scene (original scene left untouched).
    # --camera-model only steers the uncalibrated DLT estimate; with a known
    # intrinsic OpenMVG reuses the scene K under -s and ignores -c, so drop it.
    logger.info(f'Localizing {query_img.name} against {paths["sfm_db"].name} '
                f'(resection={ResectionMethod(resection_method).name.lower()})')
    mvg_localize(paths, sfm_key='sfm_db', query_key='query_dir',
                 out_key='loc', match_out_key='match_out',
                 camera_model=None if single_intrinsics else args.camera_model,
                 resection_method=resection_method,
                 residual_error=args.residual_error,
                 single_intrinsics=single_intrinsics,
                 threads=args.threads, metadata=metadata)

    # 3. Extract the single localized view as the reusable calibration.
    logger.info('Extracting calibration (pose + intrinsic)')
    extract_calibration(paths['sfm_expanded'], query_img.name,
                        paths['query_dir'], paths['calibration'])

    # 4. (Optional) Re-express pose in the centered mesh coordinate frame.
    if args.sfm_transform:
        logger.info('Applying SfM transform to calibration pose')
        sfm_transform = np.load(args.sfm_transform)
        if sfm_transform.shape != (4, 4):
            sys.exit(f'--sfm-transform: expected a 4x4 matrix, '
                     f'got shape {sfm_transform.shape}')
        cal = json.loads(paths['calibration'].read_text())
        e = cal['extrinsics'][0]['value']
        R = np.asarray(e['rotation'], dtype=np.float64)
        C = np.asarray(e['center'], dtype=np.float64)
        R_new, C_new = transform_extrinsic(R, C, sfm_transform)
        e['rotation'] = R_new.tolist()
        e['center'] = C_new.tolist()
        paths['calibration'].write_text(json.dumps(cal, indent=2))
        logger.info('Calibration pose re-expressed in centered frame')

    # 5. (Optional) Save the shared-format camera file (for registration-toolkit).
    if args.save_camera_file:
        save_camera_file(paths['calibration'], Path(args.save_camera_file))

    logger.info(f'Done. Calibration: {paths["calibration"]}')
    logger.info('Texture each modality via: pgs-retexture --calibration '
                f'{paths["calibration"]} -r {recon_dir} -i <modality-image>')


if __name__ == '__main__':
    main()
