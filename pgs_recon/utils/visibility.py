"""Which faces of a mesh a single camera can actually see.

Projective retexture (``pgs-retexture --calibration``) maps a mesh through one
calibrated view, so every face that projects inside the image takes that image's
pixels whether or not the camera could see it. Wherever the surface overhangs
itself -- a fold, a curled edge, one fragment resting on another -- the
foreground is painted a second time onto whatever is hidden behind it, and
nothing downstream marks it as wrong. Back-face culling does not catch this: the
hidden surface is facing the camera, there is simply something in the way.

This is the depth test OpenMVS does internally and the projective path did not:
rasterize the whole mesh into a z-buffer through the same camera, then ask each
face what fraction of its own pixels it still owns.

The answer is a *fraction* rather than a flag because a face straddling an
occlusion boundary is half-right -- part of it is imaged, part of it is covered
by what is in front -- and a mesh cannot texture half a triangle. The caller
decides what fraction is enough (``pgs-retexture --occlusion-coverage``); the
boundary costs at most one face either way, and the choice is between a
face-wide fringe of foreground smeared onto the hidden surface and a face-wide
fringe with no texture at all.

Two details carry most of the correctness:

- **Every face writes to the z-buffer, even a sub-pixel one.** Faces are sampled
  at the pixel centres they cover, *plus* one guaranteed sample at the face's
  own projected centroid. Without that fallback a mesh finer than the camera's
  pixel grid -- which is the normal case for a camera standing further off than
  the rig that solved the mesh -- would put almost nothing in the depth buffer
  and the test would quietly pass everything.
- **Occlusion is judged with a bias, and it has to be slope-scaled.** A sample
  is occluded only when something is nearer by more than a tolerance. Part of
  that tolerance is relative to depth (float error, mesh noise). The part that
  matters is the surface's own depth gradient: two *different* sub-pixel faces
  landing in one pixel are at genuinely different depths, and without a
  tolerance that follows how fast depth changes per pixel, the farther one reads
  as occluded by its own neighbour and a foreshortened fine mesh speckles. The
  gradient is per pixel rather than per face, so a large face -- which may span
  a lot of depth across many pixels -- keeps a tight tolerance and a real
  occluder in front of it is still caught.

ADR 0012 is where the three constants and the measurements behind them live.
The camera conventions here -- camera-space Z rather than range, samples at
pixel centres, nearest hit -- are deliberately `pgs-localize`'s renderer's
(`localize_render.cpp`, ADR 0011); that one ray-casts where this rasterizes, and
ADR 0012 records why there are two. Changing a convention means changing both.
"""
import logging
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Relative depth tolerance: a sample is occluded only if something is nearer
#: than ``(1 - bias)`` times its own depth. Unit-free, so it holds whatever the
#: scene is scaled in -- 1e-3 is 0.5 mm at a 0.5 m standoff, well under the
#: relief of any true overhang and well over mesh noise and float error.
DEFAULT_DEPTH_BIAS = 1e-3

#: Pixels of slack in the slope-scaled half of the tolerance: two samples in one
#: pixel are up to ~sqrt(2) px apart, so the surface's depth can legitimately
#: differ by that many gradients between them.
_SLOPE_PIXELS = 1.5

#: Floor on a face's projected extent when its depth gradient is measured, in
#: pixels. Below this a face is a sliver or a point, its measured gradient is
#: noise over an arbitrarily small number, and the floor is what stops the
#: tolerance running away with it.
_MIN_EXTENT_PX = 0.1

#: Largest bounding box, in pixels per side, any one rasterizer item covers.
#: Bigger faces are split into tiles first, so a single huge triangle cannot
#: allocate its whole bounding box at once.
_TILE = 32

#: Candidate pixels held in flight at a time. Caps the rasterizer's working set
#: (each candidate costs a few float64 temporaries) independently of mesh size.
_CHUNK = 1 << 20


def project_points(points: np.ndarray, R: np.ndarray, C: np.ndarray,
                   f: float, cx: float, cy: float,
                   disto: Optional[Sequence[float]] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Project world points through a pinhole camera, openMVG's convention.

    ``R`` and ``C`` are the extrinsic as openMVG stores it (``X_cam = R (X -
    C)``) and ``disto`` the radial ``[k1, k2, k3]`` of a ``pinhole_radial_k3``,
    or ``None`` for an ideal pinhole -- i.e. exactly what
    :func:`~pgs_recon.utils.sfm_json.camera_from_calibration` returns.

    Returns ``(uv, depth)``: pixel coordinates with the origin at the *centre*
    of the first pixel (OpenCV/openMVG's convention, so pixel ``i`` spans
    ``[i - 0.5, i + 0.5)``), and camera-space Z -- the depth a z-buffer orders
    on, not the distance to the camera centre.

    A point at or behind the camera plane gets a non-positive depth and a
    meaningless ``uv``; the caller is what must not use it.
    """
    points = np.asarray(points, dtype=np.float64)
    Xc = (np.asarray(R, dtype=np.float64)
          @ (points - np.asarray(C, dtype=np.float64)).T).T
    z = Xc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        x = Xc[:, 0] / z
        y = Xc[:, 1] / z
    if disto is not None:
        r2 = x * x + y * y
        rad = 1.0 + disto[0] * r2 + disto[1] * r2 * r2 + disto[2] * r2 ** 3
        x, y = x * rad, y * rad
    return np.column_stack([f * x + cx, f * y + cy]), z


def _bounding_boxes(tri: np.ndarray, width: int, height: int):
    """Integer pixel-centre ranges each projected triangle can cover.

    Returns ``(ix0, iy0, bw, bh)``; a box with ``bw`` or ``bh`` <= 0 covers no
    pixel centre at all (off-image, or smaller than the gap between two).
    """
    x, y = tri[:, :, 0], tri[:, :, 1]
    # Clipped *before* the cast: a projection near the camera plane runs to
    # 1e20, which int64 does not hold.
    ix0 = np.clip(np.ceil(x.min(axis=1)), 0, width).astype(np.int64)
    ix1 = np.clip(np.floor(x.max(axis=1)), -1, width - 1).astype(np.int64)
    iy0 = np.clip(np.ceil(y.min(axis=1)), 0, height).astype(np.int64)
    iy1 = np.clip(np.floor(y.max(axis=1)), -1, height - 1).astype(np.int64)
    return ix0, iy0, ix1 - ix0 + 1, iy1 - iy0 + 1


def _tiles(ix0, iy0, bw, bh):
    """Split bounding boxes into at most ``_TILE``-square pieces.

    Returns ``(src, x0, y0, w, h)``, ``src`` indexing back into the boxes. This
    is what bounds the rasterizer's per-item allocation: without it one triangle
    spanning the image would ask for its whole bounding box in one go.
    """
    ntx = -(-bw // _TILE)
    counts = ntx * -(-bh // _TILE)
    src = np.repeat(np.arange(len(counts)), counts)
    # Position of each tile within its own box, from the flat item index.
    pos = np.arange(int(counts.sum())) - (np.cumsum(counts) - counts)[src]
    tx = pos % ntx[src]
    ty = pos // ntx[src]
    return (src, ix0[src] + tx * _TILE, iy0[src] + ty * _TILE,
            np.minimum(_TILE, bw[src] - tx * _TILE),
            np.minimum(_TILE, bh[src] - ty * _TILE))


def _rasterize(tri: np.ndarray, z: np.ndarray, face: np.ndarray,
               width: int, height: int):
    """Sample every projected triangle at the pixel centres it covers.

    ``tri`` is (N, 3, 2) pixel coordinates, ``z`` (N, 3) camera-space depths and
    ``face`` the face index each row belongs to. Returns
    ``(pixel, depth, face)``, flat and unordered, one entry per covered pixel
    centre: ``pixel`` is ``row * width + column``, ``depth`` the
    perspective-correct camera-space Z at that pixel centre.
    """
    ix0, iy0, bw, bh = _bounding_boxes(tri, width, height)
    covers = (bw > 0) & (bh > 0)
    src, x0, y0, w, h = _tiles(ix0[covers], iy0[covers], bw[covers], bh[covers])
    # _tiles indexed into the covering subset; carry that back to the
    # input rows.
    src = np.flatnonzero(covers)[src]

    pix_out, dep_out, fid_out = [], [], []
    span = np.maximum(w, h)
    size = 1 << np.ceil(np.log2(span)).astype(np.int64)
    for s in np.unique(size):
        items = np.flatnonzero(size == s)
        # The candidate grid is s*s wide, so the chunk is however many items'
        # grids fit the budget -- always at least one.
        step = max(1, _CHUNK // int(s * s))
        off = np.arange(s * s)
        dx, dy = (off % s)[None, :], (off // s)[None, :]
        for lo in range(0, len(items), step):
            m = items[lo:lo + step]
            px = x0[m][:, None] + dx
            py = y0[m][:, None] + dy
            inside = (dx < w[m][:, None]) & (dy < h[m][:, None])

            t = tri[src[m]]
            ax, ay = t[:, 0, 0][:, None], t[:, 0, 1][:, None]
            bx, by = t[:, 1, 0][:, None], t[:, 1, 1][:, None]
            cx, cy = t[:, 2, 0][:, None], t[:, 2, 1][:, None]
            # Edge functions: wi is twice the area of the sub-triangle opposite
            # vertex i, so they sum to twice the signed area of the whole.
            w0 = (bx - px) * (cy - py) - (by - py) * (cx - px)
            w1 = (cx - px) * (ay - py) - (cy - py) * (ax - px)
            w2 = (ax - px) * (by - py) - (ay - py) * (bx - px)
            area = w0 + w1 + w2
            sign = np.where(area < 0, -1.0, 1.0)
            inside &= (area != 0) & (w0 * sign >= 0) & (w1 * sign >= 0) \
                & (w2 * sign >= 0)
            if not inside.any():
                continue

            # Depth interpolates in 1/z across the image, not in z: the
            # barycentrics are screen-space, so w0/z0 + w1/z1 + w2/z2 over the
            # doubled area is the inverse depth at the pixel centre.
            zz = z[src[m]]
            with np.errstate(divide='ignore', invalid='ignore'):
                inv = (w0 / zz[:, 0][:, None] + w1 / zz[:, 1][:, None]
                       + w2 / zz[:, 2][:, None]) / area
                d = 1.0 / inv
            inside &= np.isfinite(d) & (d > 0)

            pix_out.append(py[inside].astype(np.int64) * width + px[inside])
            dep_out.append(d[inside].astype(np.float32))
            fid_out.append(np.broadcast_to(face[src[m]][:, None],
                                           inside.shape)[inside])

    empty_i = np.empty(0, dtype=np.int64)
    if not pix_out:
        return empty_i, np.empty(0, dtype=np.float32), empty_i
    return (np.concatenate(pix_out), np.concatenate(dep_out),
            np.concatenate(fid_out))


def _depth_buffer(pixel: np.ndarray, depth: np.ndarray, size: int):
    """Nearest sampled depth per pixel, ``inf`` where nothing was sampled.

    Done by sorting one uint64 key per sample -- the pixel in the high word, the
    depth's float32 bit pattern in the low one, which orders as the float does
    for positive depths. One sort beats ``np.minimum.at`` by an order of
    magnitude at the sample counts a full-resolution mesh produces.
    """
    key = (pixel.astype(np.uint64) << np.uint64(32)) \
        | depth.view(np.uint32).astype(np.uint64)
    key.sort()
    at = key >> np.uint64(32)
    first = np.empty(at.shape, dtype=bool)
    first[0] = True
    np.not_equal(at[1:], at[:-1], out=first[1:])
    buf = np.full(size, np.inf, dtype=np.float32)
    buf[at[first]] = (key[first] & np.uint64(0xFFFFFFFF)) \
        .astype(np.uint32).view(np.float32)
    return buf


def visible_fraction(uv: np.ndarray, depth: np.ndarray, faces: np.ndarray,
                     width: int, height: int, face_uv: np.ndarray,
                     face_depth: np.ndarray,
                     bias: float = DEFAULT_DEPTH_BIAS) -> np.ndarray:
    """Fraction of each face's sampled pixels that no other face covers.

    ``uv``/``depth`` are the projected mesh vertices (:func:`project_points`)
    and ``face_uv``/``face_depth`` the projection of each face's 3D centroid --
    the guaranteed sample that keeps a sub-pixel face in the depth buffer. Both
    come from the same camera; projecting the centroid rather than averaging the
    projected vertices is what keeps that sample exact under perspective.

    The whole mesh occludes: a face is tested against everything that projects
    in front of the camera, back-facing and out-of-view faces included, because
    a solid surface is occluded by its own far side too.

    Returns one fraction per face, in [0, 1]. A face that does not project in
    front of the camera at all gets 0.0 -- it is not visible, which is the same
    answer the caller wants for an occluded one.
    """
    faces = np.asarray(faces, dtype=np.int64)
    tri = uv[faces]
    z = depth[faces]
    # A face crossing the camera plane projects to nonsense, so it neither
    # occludes nor is tested; it fails the caller's own in-front test anyway.
    ahead = np.isfinite(tri).all(axis=(1, 2)) & (z > 1e-9).all(axis=1)
    rows = np.flatnonzero(ahead)
    pix, dep, fid = _rasterize(tri[rows], z[rows], rows, width, height)

    # One sample per face at its projected centroid, whether or not the face
    # covered a pixel centre. Faces whose centroid lands off-image are dropped
    # from the buffer, not from the mesh: they cannot occlude what they do not
    # cover, and a face fully off-image is the caller's to reject.
    cu = np.rint(face_uv[:, 0]).astype(np.int64)
    cv = np.rint(face_uv[:, 1]).astype(np.int64)
    on = (ahead & np.isfinite(face_uv).all(axis=1) & (face_depth > 1e-9)
          & (cu >= 0) & (cu < width) & (cv >= 0) & (cv < height))
    keys = np.flatnonzero(on)
    pix = np.concatenate([pix, cv[keys] * width + cu[keys]])
    dep = np.concatenate([dep, face_depth[keys].astype(np.float32)])
    fid = np.concatenate([fid, keys])

    frac = np.zeros(len(faces), dtype=np.float64)
    if len(pix) == 0:
        return frac
    logger.debug(f'Depth test: {len(pix)} samples over {len(rows)} projected '
                 f'faces into a {width}x{height} buffer')

    # How fast depth changes per pixel across each face, which is the scale of
    # the depth difference between two faces of one surface that land in the
    # same pixel. Note this is the face's *gradient*, not its depth span: a
    # sub-pixel face has a small span and can still be steep, which is exactly
    # the case that needs the slack, while a face spanning a lot of depth over
    # many pixels keeps a tight tolerance and stays occludable.
    extent = np.maximum(np.ptp(tri[:, :, 0], axis=1),
                        np.ptp(tri[:, :, 1], axis=1))
    with np.errstate(invalid='ignore'):
        slope = np.ptp(z, axis=1) / np.maximum(extent, _MIN_EXTENT_PX)
    tol = (_SLOPE_PIXELS * np.where(np.isfinite(slope), slope, 0.0)) \
        .astype(np.float32)

    nearest = _depth_buffer(pix, dep, width * height)
    seen = nearest[pix] >= dep * (1.0 - bias) - tol[fid]
    total = np.bincount(fid, minlength=len(faces))
    np.divide(np.bincount(fid[seen], minlength=len(faces)), total, out=frac,
              where=total > 0)
    return frac
