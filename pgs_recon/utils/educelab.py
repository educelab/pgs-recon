from itertools import combinations

import cv2.aruco as ar
import numpy as np

import pgs_recon.utils.charuco as char

# Physical positions of known keypoints relative to top-left corner of sample
# square. See the Sample Square v1 Placement Guide in:
# https://gitlab.com/educelab/acquisition-workflow
_SAMPLE_SQUARE_V1_KP_POS_CM = [
    # M0,0 -> M0,3
    np.array((0.966666666666667, 0.3)),
    np.array((0.3, 0.966666666666667)),
    np.array((1.633333333333334, 0.966666666666667)),
    np.array((0.966666666666667, 1.633333333333334)),
    # C0,0 -> C0,3
    np.array((0.866623541666667, 0.866623541666667)),
    np.array((1.533290208333333, 0.866623541666667)),
    np.array((0.866623541666667, 1.533290208333333)),
    np.array((1.533290208333333, 1.533290208333333)),
    # M1,0 -> M1,3
    np.array((0.966666666667, 13.4)),
    np.array((0.3, 14.066666666666667)),
    np.array((1.633333333333334, 14.066666666666667)),
    np.array((0.966666666666667, 14.733333333333334)),
    # C1,0 -> C1,3
    np.array((0.866623541666667, 13.966623541666667)),
    np.array((1.533290208333333, 13.966623541666667)),
    np.array((0.866623541666667, 14.633290208333333)),
    np.array((1.533290208333333, 14.633290208333333))
]

# Pre-calculated distances between each keypoint in cm. This list does not
# store duplicates. For example, the value for D(0, 10) and D(10, 0) is only
# stored in list[0][10]. Use keypoint_distance(a, b) to easily get distances
# between arbitrary keys.
_SAMPLE_SQUARE_V1_KP_DIST_CM = [
    [np.linalg.norm(b - a) for b in _SAMPLE_SQUARE_V1_KP_POS_CM[idx + 1:]] for
    idx, a in enumerate(_SAMPLE_SQUARE_V1_KP_POS_CM[:-1])]


# Get the distance between two keypoints
def keypoint_distance(a, b):
    if a == b:
        return 0.
    if b < a:
        a, b = b, a
    return _SAMPLE_SQUARE_V1_KP_DIST_CM[a][b - a - 1]


# Detect the EduceLab sample square in an image
def detect_sample_square(img):
    # Results
    boards = []
    kp_ids = []
    kp_pos = []

    # Try to detect both boards
    for idx in range(2):
        board = char.generate_board(offset=idx * 512)

        # Detect board
        b = char.detect_board(img, board)

        # Shift IDs to [0, 15]
        if b.marker_corners is not None and len(b.marker_corners) > 0:
            # can't update directly through += so update reference
            ids = b.marker_ids
            ids += idx * 8
            kp_ids.extend(ids.flatten().tolist())
            kp_pos.extend(c[:, 0, :].squeeze() for c in b.marker_corners)
        if b.board_corners is not None and len(b.board_corners) > 0:
            # can't update directly through += so update reference
            ids = b.board_ids
            ids += (idx * 8) + 4
            kp_ids.extend(ids.flatten().tolist())
            kp_pos.extend(c[0, :].squeeze() for c in b.board_corners)

        boards.append(b)

    # Make sure we have at least landmarks
    num_ldms = sum((b.marker_cnt + b.board_cnt for b in boards))
    detected = num_ldms > 1

    # Calculate pixels-per-cm
    ppcm = 0.
    if num_ldms > 1:
        ppc_samples = []
        for ids, pts in zip(combinations(kp_ids, r=2),
                            combinations(kp_pos, r=2)):
            dist_px = np.linalg.norm(pts[1] - pts[0])
            dist_cm = keypoint_distance(ids[0], ids[1])
            ppc_samples.append(dist_px / dist_cm)
        ppcm = np.mean(ppc_samples)

    return detected, boards, ppcm, kp_ids, kp_pos


def main():
    # local imports
    import cv2
    import argparse

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image', '-i', required=True,
                        help='Input image')
    parser.add_argument('--output-image', '-o',
                        help='Draw detected markers onto the input image and '
                             'save to the provided file path')
    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.input_image)

    # Detect boards
    detected, boards, ppcm, *_ = detect_sample_square(img)

    # Draw detected markers
    if detected:
        # Print results
        num_markers = sum((b.marker_cnt for b in boards))
        num_boards = sum((b.board_cnt for b in boards))
        print(f'Detected:\n'
              f' - Markers: {num_markers}\n'
              f' - Board corners: {num_boards}\n'
              f' - Texture resolution (pixels/cm): {ppcm}')

        # Draw each board
        if args.output_image is not None:
            for b in boards:
                # Draw markers and board corners
                if b.marker_corners is not None and len(b.marker_corners) > 0:
                    ar.drawDetectedMarkers(img, b.marker_corners, b.marker_ids)
                if b.board_corners is not None and len(b.board_corners) > 0:
                    ar.drawDetectedCornersCharuco(img, b.board_corners,
                                                  b.board_ids)
            cv2.imwrite(args.output_image, img)
    else:
        print('No markers detected.')


if __name__ == '__main__':
    main()
