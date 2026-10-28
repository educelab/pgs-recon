"""A virtual overhead: the reference mesh seen by a camera straight above it.

For a plate with no registered overhead capture (ADR 0013 Decision 2). The
prior only needs *an* image of the plate whose every pixel has a known world
position, and a render of the IR reference mesh is one -- ``pgs-localize-joint
position-map --output-image`` writes the position map and the image in one
pass. This module writes the calibration that render is taken from.

The camera sits above the mesh's bounding box in the canonical overhead
orientation, ``R = diag(1, -1, -1)``, looking down ``-Z``, so the mesh must be
Z-up -- as ``pgs-recon``'s autoscaled, centered output is. It frames the box
with a margin, at a resolution chosen by the caller.

Stdlib-only.
"""
from pathlib import Path
from typing import Dict, NamedTuple

#: Fraction of the box's footprint left as border on every side, so the
#: plate's edge is never on the image edge.
MARGIN = 1.05

#: Camera height above the mesh's top, in the mesh's units. Large against a
#: plate's relief, so the render is near-orthographic; 150 is what CR04 was
#: localized through, in its units (apparently centimetres).
STANDOFF = 150.0

#: Render width in pixels: about twice a hand-held view's 3072, so the prior
#: is limited by the views rather than by the render.
WIDTH = 6000

#: cereal's polymorphic ids for openMVG's ``View`` and ``pinhole_radial_k3``
#: (the first use of each type in a file), and the shared-pointer ids of the
#: one view and the one intrinsic.
_VIEW_TYPE = 0x40000000
_INTRINSIC_TYPE = 0x80000001
_VIEW_PTR = 0x80000001
_INTRINSIC_PTR = 0x80000002


class Bounds(NamedTuple):
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float


def obj_bounds(path: Path) -> Bounds:
    """The axis-aligned bounding box of an OBJ's vertices.

    Streams the ``v`` lines rather than loading the mesh: a reference mesh
    runs to millions of vertices, and nothing else of it is needed.
    """
    lo = [float('inf')] * 3
    hi = [float('-inf')] * 3
    with open(path, 'rb') as fh:
        for line in fh:
            if not line.startswith(b'v '):
                continue
            for i, value in enumerate(line.split()[1:4]):
                v = float(value)
                if v < lo[i]:
                    lo[i] = v
                if v > hi[i]:
                    hi[i] = v
    if lo[0] > hi[0]:
        raise ValueError(f'{path} has no vertices')
    return Bounds(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])


def calibration(bounds: Bounds, image_name: str, width: int = WIDTH,
                root: Path = Path('.')) -> Dict:
    """A one-view openMVG calibration framing ``bounds`` from above.

    The image's aspect follows the box's, and the focal length makes the
    framed box exactly ``width`` pixels across at the mesh's top.
    """
    half_x = (bounds.xmax - bounds.xmin) / 2 * MARGIN
    half_y = (bounds.ymax - bounds.ymin) / 2 * MARGIN
    if half_x <= 0 or half_y <= 0:
        raise ValueError(f'degenerate footprint: {bounds}')
    height = int(round(width * half_y / half_x))
    focal = (width / 2) * STANDOFF / half_x
    centre = [(bounds.xmin + bounds.xmax) / 2,
              (bounds.ymin + bounds.ymax) / 2,
              bounds.zmax + STANDOFF]
    return {
        'sfm_data_version': '0.3', 'root_path': str(root),
        'views': [{'key': 0, 'value': {
            'polymorphic_id': _VIEW_TYPE,
            'ptr_wrapper': {'id': _VIEW_PTR, 'data': {
                'local_path': '', 'filename': image_name, 'width': width,
                'height': height, 'id_view': 0, 'id_intrinsic': 0,
                'id_pose': 0}}}}],
        'intrinsics': [{'key': 0, 'value': {
            'polymorphic_id': _INTRINSIC_TYPE,
            'polymorphic_name': 'pinhole_radial_k3',
            'ptr_wrapper': {'id': _INTRINSIC_PTR, 'data': {
                'width': width, 'height': height, 'focal_length': focal,
                'principal_point': [width / 2, height / 2],
                'disto_k3': [0.0, 0.0, 0.0]}}}}],
        'extrinsics': [{'key': 0, 'value': {
            'rotation': [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            'center': centre}}],
        'structure': [], 'control_points': [],
    }
