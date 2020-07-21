from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot

RADIUS_AROUND_PIVOT = 33.54101966249684
PIVOT_OFFSET_ANGLE = 0.46364760900080615


def pivot_offset(alpha: float) -> Tuple[float, float]:
    a = alpha + PIVOT_OFFSET_ANGLE
    return np.cos(a) * RADIUS_AROUND_PIVOT, -31.75 + (np.sin(a) * RADIUS_AROUND_PIVOT)


# Calculates position relative to the bottom left of the test rig mount
def testrig_calc_positions():
    # Camera pos in test board
    hole_coord = [5.0, 0.0, 0.0]

    # alpha is CW around X (cam mount elbow)
    # beta is CW around Z (mount-to-board pivot)
    alpha = -140
    beta = -45

    pos = [0.0, 0.0, 0.0]
    pos[1], pos[2] = pivot_offset(np.deg2rad(alpha))
    rotation = Rot.from_euler('z', beta, degrees=True)
    pos = np.array(pos) @ rotation.as_matrix().round(15)

    hole_offset = np.multiply(hole_coord, 19.05)
    pos = np.add(pos, hole_offset)

    print(pos)


if __name__ == '__main__':
    testrig_calc_positions()
