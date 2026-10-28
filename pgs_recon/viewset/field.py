"""``pgs-localize-field``: how far one view sits from where its calibration
puts it, tile by tile, and what kind of error that is.

The view's image is undistorted with its calibration's K, the reference mesh is
rendered at its calibration's pose (``position-map --output-image``: an
undistorted pinhole render), and the two are phase-correlated on a grid of
tiles. The tile shifts are then fitted with a translation, a scale about the
image centre, and a radial ``r^3`` term -- the three things a wrong pose or a
wrong K leaves behind -- and what the fit cannot explain is the floor the mesh
and the texture set.

A diagnostic, not a phase: ``pgs-localize-set``'s own QA grades the set
(ADR 0013 Decision 6). Comparing against the reference the view was
registered to is circular for the *pose*, but the field's shape still says
whether the K is wrong, which the QA's single shift per pair cannot.
"""
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from pgs_recon import toolchain
from pgs_recon.utility import ToolFailed
from pgs_recon.utils.apps import setup_logging
from pgs_recon.viewset import tools

logger = logging.getLogger('pgs-localize-field')

#: Tile side, in pixels, and the step between tiles (half: they overlap).
TILE = 192

#: Below this standard deviation a tile has no texture to correlate.
MIN_STD = 4

#: Phase-correlation peak response below which a tile's shift is noise.
MIN_RESPONSE = 0.08

#: How much the arrows are lengthened in the drawn field, and the drawing's
#: scale against the image.
ARROW_GAIN = 10
DRAW_SCALE = 0.33


def undistort(calibration: Path, image: Path) -> np.ndarray:
    """``image`` in 8-bit grey, undistorted with ``calibration``'s intrinsic.

    openMVG's ``pinhole_radial_k3`` is OpenCV's model with ``k1, k2, k3`` and
    no tangential terms. The output keeps the same K, so it lines up with a
    render made from the same calibration.
    """
    cal = json.loads(Path(calibration).read_text())
    d = cal['intrinsics'][0]['value']['ptr_wrapper']['data']
    f = d['focal_length']
    cx, cy = d['principal_point']
    k = d.get('disto_k3', [0.0, 0.0, 0.0])
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'cannot read {image}')
    return cv2.undistort(img, K, np.array([k[0], k[1], 0, 0, k[2]]))


def _normalized(a: np.ndarray) -> np.ndarray:
    a = cv2.GaussianBlur(a, (0, 0), 1.0)
    return (a - a.mean()) / (a.std() + 1e-6)


def measure(query: np.ndarray, render: np.ndarray,
            mask: Optional[np.ndarray] = None) -> Dict:
    """The tile shifts that move ``render`` onto ``query``, and their fit.

    Shifts are in ``query``'s pixels. A tile is used only if it lies wholly
    inside ``mask`` (nonzero), has texture in both images, and correlates
    clearly. The fit is iteratively reweighted: three rounds of dropping tiles
    beyond three robust sigmas.
    """
    q = query.astype(np.float32)
    r = render.astype(np.float32)
    if q.shape != r.shape:
        raise ValueError(f'query {q.shape} and render {r.shape} differ in size')
    keep_px = (np.ones(q.shape, bool) if mask is None else mask > 0)
    h, w = q.shape
    window = cv2.createHanningWindow((TILE, TILE), cv2.CV_32F)

    pts, d = [], []
    for y in range(0, h - TILE + 1, TILE // 2):
        for x in range(0, w - TILE + 1, TILE // 2):
            if not keep_px[y:y + TILE, x:x + TILE].all():
                continue
            a, b = q[y:y + TILE, x:x + TILE], r[y:y + TILE, x:x + TILE]
            if a.std() < MIN_STD or b.std() < MIN_STD:
                continue
            (dx, dy), response = cv2.phaseCorrelate(
                _normalized(b), _normalized(a), window)
            if (response < MIN_RESPONSE or abs(dx) > TILE / 4
                    or abs(dy) > TILE / 4):
                continue
            pts.append((x + TILE / 2, y + TILE / 2))
            d.append((dx, dy))
    if len(d) < 4:
        raise ValueError(f'{len(d)} usable tiles: too few to fit a field')
    pts, d = np.array(pts), np.array(d)
    c = pts - [w / 2, h / 2]
    radius = np.linalg.norm(c, axis=1)
    mag = np.linalg.norm(d, axis=1)

    # d = t + s * c + k * c * (r/R)^2, R the half diagonal.
    R = np.hypot(w, h) / 2
    A = np.zeros((2 * len(d), 4))
    A[0::2, 0] = 1
    A[1::2, 1] = 1
    A[0::2, 2] = c[:, 0]
    A[1::2, 2] = c[:, 1]
    A[0::2, 3] = c[:, 0] * (radius / R) ** 2
    A[1::2, 3] = c[:, 1] * (radius / R) ** 2
    b = d.reshape(-1)
    keep = np.ones(len(b), bool)
    for _ in range(3):
        p, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        sigma = 1.4826 * np.median(np.abs((b - A @ p)[keep]))
        keep = np.abs(b - A @ p) < 3 * max(sigma, 0.1)
    residual = np.linalg.norm((b - A @ p).reshape(-1, 2), axis=1)

    zones = {}
    for name, sel in (('centre', radius < R / 3),
                      ('middle', (radius >= R / 3) & (radius < 2 * R / 3)),
                      ('edge', radius >= 2 * R / 3)):
        if sel.any():
            zones[name] = {'tiles': int(sel.sum()),
                           'median_px': float(np.median(mag[sel])),
                           'mean_px': [float(d[sel, 0].mean()),
                                       float(d[sel, 1].mean())]}
    return {
        'tiles': len(d),
        'shift_px': {'median': float(np.median(mag)),
                     'p90': float(np.percentile(mag, 90)),
                     'coherent': [float(np.median(d[:, 0])),
                                  float(np.median(d[:, 1]))]},
        'model': {'translation_px': [float(p[0]), float(p[1])],
                  'scale_permille': float(p[2] * 1e3),
                  'scale_corner_px': float(p[2] * R),
                  'radial': float(p[3]),
                  'radial_corner_px': float(p[3] * R)},
        'residual_px': {'median': float(np.median(residual)),
                        'p90': float(np.percentile(residual, 90))},
        'zones': zones,
        '_points': pts, '_shifts': d,
    }


def draw(query: np.ndarray, field: Dict) -> np.ndarray:
    """``query``, shrunk, with each tile's shift as an arrow, lengthened
    :data:`ARROW_GAIN` times."""
    vis = cv2.cvtColor(cv2.resize(query, None, fx=DRAW_SCALE, fy=DRAW_SCALE),
                       cv2.COLOR_GRAY2BGR)
    for (x, y), (dx, dy) in zip(field['_points'], field['_shifts']):
        p0 = (int(x * DRAW_SCALE), int(y * DRAW_SCALE))
        p1 = (int(x * DRAW_SCALE + dx * ARROW_GAIN),
              int(y * DRAW_SCALE + dy * ARROW_GAIN))
        cv2.arrowedLine(vis, p0, p1, (0, 0, 255), 2, tipLength=0.3)
    return vis


def summary(field: Dict) -> str:
    s, m, res = field['shift_px'], field['model'], field['residual_px']
    lines = [
        f'tiles {field["tiles"]}  |shift| median {s["median"]:.2f} p90 '
        f'{s["p90"]:.2f} px  coherent ({s["coherent"][0]:+.2f}, '
        f'{s["coherent"][1]:+.2f})',
        f'model: t=({m["translation_px"][0]:+.2f},{m["translation_px"][1]:+.2f})'
        f' px  scale {m["scale_permille"]:+.3f} permille (= '
        f'{m["scale_corner_px"]:+.2f} px at the corner)  radial '
        f'{m["radial"]:+.5f} (= {m["radial_corner_px"]:+.2f} px at the corner)',
        f'residual after model: median {res["median"]:.2f} px p90 '
        f'{res["p90"]:.2f} px']
    for name, z in field['zones'].items():
        lines.append(f'  {name:>6}: {z["tiles"]:4d} tiles, median |shift| '
                     f'{z["median_px"]:.2f} px, mean ({z["mean_px"][0]:+.2f}, '
                     f'{z["mean_px"][1]:+.2f})')
    return '\n'.join(lines)


def main():
    """Entry point. A failed binary exits with *its* status, not 1."""
    try:
        _main()
    except ToolFailed as e:
        logger.error(f'{e}')
        sys.exit(e.exit_code)


def _main():
    parser = argparse.ArgumentParser(
        prog='pgs-localize-field',
        description='Measure one view\'s misregistration field against a '
                    'render of the reference mesh at its calibration.')
    parser.add_argument('--calibration', required=True,
                        help='The view\'s one-view calibration .json.')
    parser.add_argument('--image', required=True, help='The view\'s image.')
    parser.add_argument('--mesh', required=True,
                        help='The textured mesh to render: the IR reference.')
    parser.add_argument('--mask', default=None,
                        help='Image whose zero pixels are left out.')
    parser.add_argument('--output', default=None,
                        help='Write the field as arrows on the image (.png).')
    parser.add_argument('--report', default=None,
                        help='Write the numbers as .json.')
    parser.add_argument('--keep-render', default=None, metavar='DIR',
                        help='Keep the undistorted image and the render here.')
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--path', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--log-level', default='INFO', type=str.upper,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()
    setup_logging(args.log_level)
    toolchain.configure(prefix=args.path)

    query = undistort(Path(args.calibration), Path(args.image))
    mask = None
    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            sys.exit(f'cannot read {args.mask}')

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.keep_render or tmp)
        work.mkdir(parents=True, exist_ok=True)
        render_path = work / 'render.png'
        tools.position_map(Path(args.calibration), Path(args.mesh),
                           work / 'position.npy', output_image=render_path,
                           threads=args.threads)
        render = cv2.imread(str(render_path), cv2.IMREAD_GRAYSCALE)
        if args.keep_render:
            cv2.imwrite(str(work / 'undistorted.png'), query)

    try:
        field = measure(query, render, mask)
    except ValueError as e:
        sys.exit(str(e))
    print(summary(field))
    if args.output:
        cv2.imwrite(args.output, draw(query, field))
    if args.report:
        public = {k: v for k, v in field.items() if not k.startswith('_')}
        Path(args.report).write_text(json.dumps(public, indent=1) + '\n')


if __name__ == '__main__':
    main()
