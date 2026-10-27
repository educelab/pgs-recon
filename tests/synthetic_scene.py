"""A synthetic localization scene with a known answer.

A textured plane at z = 0 and pinhole views of it rendered by inverse
ray-casting onto that plane. The plane is what makes the rendering exact
without any renderer of our own: nothing here shares code with the tools under
test, only the conventions they document -- pixel centres at +0.5, OBJ ``vt``
bottom-left, a camera file's ``pose`` world-to-camera.

Not a test module (no ``test_`` prefix): the tests that use it import it.
Needs ``numpy`` and ``cv2``; callers skip when either is missing.
"""
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

#: The plane spans [-HALF, HALF]^2.
HALF = 50.0


class View(NamedTuple):
    """A camera: world-to-camera ``R``, centre ``C``, square pixels, and
    openMVG's ``pinhole_radial_k3`` distortion ``k`` (zero by default)."""
    R: np.ndarray
    C: np.ndarray
    f: float
    cx: float
    cy: float
    width: int
    height: int
    k: tuple = (0.0, 0.0, 0.0)


def _distortion(k, r2):
    return 1.0 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3


def project(view: View, X: np.ndarray) -> np.ndarray:
    """World points (N, 3) to pixels (N, 2): openMVG's radial-k3 model, which
    distorts the normalized coordinates before K."""
    cam = (np.asarray(X, float) - view.C) @ view.R.T
    xy = cam[:, :2] / cam[:, 2:]
    xy = xy * _distortion(view.k, (xy ** 2).sum(1))[:, None]
    return view.f * xy + [view.cx, view.cy]


def rot_x(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t), np.cos(t)]])


def rot_y(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0],
                     [-np.sin(t), 0, np.cos(t)]])


def rot_z(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0],
                     [0, 0, 1]])


#: Looking straight down at the plane: camera +y is world -y, +z is world -z.
DOWN = np.diag([1.0, -1.0, -1.0])


def make_texture(size: int = 1024, seed: int = 7) -> np.ndarray:
    """Multi-scale blobs in [0, 1], so a describer finds features at every
    scale."""
    rng = np.random.default_rng(seed)
    tex = np.zeros((size, size), np.float32)
    for sigma, weight in ((size / 32, 1.0), (size / 128, 0.6),
                          (size / 512, 0.35), (1.5, 0.2)):
        noise = rng.standard_normal((size, size)).astype(np.float32)
        tex += weight * cv2.GaussianBlur(noise, (0, 0), sigma) * sigma
    return (tex - tex.min()) / (tex.max() - tex.min())


def write_plane(out: Path, texture: np.ndarray, grid: int = 20) -> Path:
    """The textured plane as OBJ + MTL + PNG in ``out``; returns the OBJ."""
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / 'texture.png'), (texture * 255).astype(np.uint8))
    (out / 'plane.mtl').write_text('newmtl material0\nmap_Kd texture.png\n')
    xs = np.linspace(-HALF, HALF, grid + 1)
    lines = ['mtllib plane.mtl']
    lines += [f'v {x} {y} 0' for y in xs for x in xs]
    # u left to right, v bottom-up (OBJ); the image's top row is world +y.
    lines += [f'vt {(x + HALF) / (2 * HALF)} {(y + HALF) / (2 * HALF)}'
              for y in xs for x in xs]
    lines.append('usemtl material0')
    for j in range(grid):
        for i in range(grid):
            a = j * (grid + 1) + i + 1
            b, c, d = a + 1, a + grid + 2, a + grid + 1
            lines.append(f'f {a}/{a} {b}/{b} {c}/{c}')
            lines.append(f'f {a}/{a} {c}/{c} {d}/{d}')
    obj = out / 'plane.obj'
    obj.write_text('\n'.join(lines) + '\n')
    return obj


def plane_points(view: View) -> np.ndarray:
    """World point on z = 0 behind every pixel centre, (H, W, 3); NaN where
    the ray misses the plane's extent."""
    u, v = np.meshgrid(np.arange(view.width) + 0.5,
                       np.arange(view.height) + 0.5)
    # Each pixel is a *distorted* location; undistort it by fixed-point
    # iteration to find the ray it images.
    xd = (u - view.cx) / view.f
    yd = (v - view.cy) / view.f
    x, y = xd.copy(), yd.copy()
    if any(view.k):
        for _ in range(20):
            d = _distortion(view.k, x * x + y * y)
            x, y = xd / d, yd / d
    d_cam = np.stack([x, y, np.ones_like(u)], -1)
    d_world = d_cam @ view.R          # R^T d, row-wise
    s = -view.C[2] / d_world[..., 2]
    pts = view.C + s[..., None] * d_world
    off = (np.abs(pts[..., 0]) > HALF) | (np.abs(pts[..., 1]) > HALF)
    pts[off] = np.nan
    return pts


def render(view: View, texture: np.ndarray, gamma: float = 0.8,
           noise: float = 0.01, seed: int = 11) -> np.ndarray:
    """The plane as ``view`` sees it, 8-bit, with a tone curve and sensor noise
    so it is not the texture's own processing."""
    pts = plane_points(view)
    size = texture.shape[0]
    map_x = ((pts[..., 0] + HALF) / (2 * HALF) * size - 0.5).astype(np.float32)
    map_y = ((HALF - pts[..., 1]) / (2 * HALF) * size - 0.5).astype(np.float32)
    img = cv2.remap(texture, np.nan_to_num(map_x, nan=-1),
                    np.nan_to_num(map_y, nan=-1), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rng = np.random.default_rng(seed)
    img = np.clip(img ** gamma + rng.normal(0, noise, img.shape), 0, 1)
    return (img * 255).astype(np.uint8)


def camera_file(view: View, with_pose: bool = True) -> str:
    """``view`` in ``localize_camera.hpp``'s flat format."""
    k1, k2, k3 = view.k
    text = (f'fx {view.f}\nfy {view.f}\ncx {view.cx}\ncy {view.cy}\n'
            f'width {view.width}\nheight {view.height}\n'
            f'k1 {k1}\nk2 {k2}\nk3 {k3}\n')
    if with_pose:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = view.R, -view.R @ view.C
        text += 'pose ' + ' '.join(f'{x:.12g}' for x in m.ravel()) + '\n'
    return text


def nudged(view: View, dC=(1.5, -1.0, 3.0), tilt_deg: float = 1.0) -> View:
    """``view`` moved a few units and tilted a degree: a prior, not the truth."""
    return view._replace(R=rot_x(tilt_deg) @ view.R,
                         C=view.C + np.asarray(dC, float))


def tie_scene(views, names, points: np.ndarray, frame_scale: float = 0.5,
              frame_offset=(7.0, -3.0, 2.0)) -> dict:
    """An openMVG ``sfm_data.json`` holding ``views`` and the tracks of
    ``points`` through them -- the shape a group's SfM solve has.

    The structure is written in a *different* frame (scaled and shifted), as a
    group solve's own arbitrary frame would be, so a consumer that trusted the
    stored coordinates rather than re-triangulating would be visibly wrong.
    Only tracks seen by two or more views are written.
    """
    out = {'sfm_data_version': '0.3', 'root_path': '', 'views': [],
           'intrinsics': [], 'extrinsics': [], 'structure': [],
           'control_points': []}
    ptr = 0x80000000
    for i, (view, name) in enumerate(zip(views, names)):
        ptr += 1
        out['views'].append({'key': i, 'value': {
            'polymorphic_id': 0x40000000,
            'ptr_wrapper': {'id': ptr, 'data': {
                'local_path': '', 'filename': name, 'width': view.width,
                'height': view.height, 'id_view': i, 'id_intrinsic': 0,
                'id_pose': i}}}})
        out['extrinsics'].append({'key': i, 'value': {
            'rotation': view.R.tolist(), 'center': view.C.tolist()}})
    v0 = views[0]
    out['intrinsics'].append({'key': 0, 'value': {
        'polymorphic_id': 0x80000001, 'polymorphic_name': 'pinhole_radial_k3',
        'ptr_wrapper': {'id': ptr + 1, 'data': {
            'width': v0.width, 'height': v0.height, 'focal_length': v0.f,
            'principal_point': [v0.cx, v0.cy], 'disto_k3': list(v0.k)}}}})
    uvs = [project(v, points) for v in views]
    for j, X in enumerate(points):
        obs = [{'key': i, 'value': {'id_feat': j, 'x': uv[j].tolist()}}
               for i, (v, uv) in enumerate(zip(views, uvs))
               if 0 <= uv[j][0] < v.width and 0 <= uv[j][1] < v.height]
        if len(obs) >= 2:
            stored = frame_scale * X + np.asarray(frame_offset)
            out['structure'].append({'key': j, 'value': {
                'X': stored.tolist(), 'observations': obs}})
    return out
