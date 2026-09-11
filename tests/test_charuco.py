"""``utils.charuco``' detection contract, and the two consumers that index it.

``CharucoDetector.detectBoard`` moved its return shapes between OpenCV 4 and 5
-- board corners ``(N, 1, 2)`` -> ``(N, 2)``, IDs ``(N, 1)`` -> ``(N,)`` -- and
nothing in the pipeline noticed, because ``DetectedBoard`` passed them straight
through to code that indexed the OpenCV 4 layout. The result was a hard crash
in ``detect_sample_square`` under OpenCV 5, latent only because no pipeline
calls ``pgs-detect-boards``/``pgs-center`` today. So the shapes are the
contract, and they are asserted here rather than left to whichever OpenCV the
image happens to carry.

The third consumer is OpenCV's own drawing helper, which is why the
normalization goes toward version 4: ``drawDetectedCornersCharuco`` refuses
``(N, 2)`` corners even in OpenCV 5, so passing the detector's own output back
to it fails.

The fixture is a *synthesized* sample square rather than a captured frame: the
board layout and the physical keypoint table are both in the repo, so the image
can be drawn at an exactly known pixels-per-cm and ``ppcm`` checked against it
-- no archive volume, no multi-megabyte blob in git.
"""
import importlib.util
import unittest

MISSING = [d for d in ('numpy', 'cv2') if importlib.util.find_spec(d) is None]

if not MISSING:
    import cv2
    import cv2.aruco as ar
    import numpy as np

    import pgs_recon.utils.charuco as char
    import pgs_recon.utils.educelab as el

#: Pixels per cm the fixture is drawn at. The real MegaVision frames land at
#: ~141.68 px/cm, so this is the right order of magnitude for the detector's
#: perimeter-rate heuristics without making the fixture huge.
PPCM = 150.

#: Where the sample square's boards sit, read back out of
#: ``educelab._SAMPLE_SQUARE_V1_KP_POS_CM``: a board is 30 units == 2 cm, its
#: interpolated corners are 10 units in from its origin, so C0,0's 0.8666 cm
#: puts board 0's top-left at 0.2 cm and C1,0's 13.9666 puts board 1's at
#: 13.2999 -- 13.1 cm apart.
BOARD_CM = 2.
ORIGIN_CM = 0.199956875
BOARD_GAP_CM = 13.1

#: Every marker corner and interpolated ChArUco corner of both boards.
EXPECTED_KEYPOINTS = 16


def sample_square(ppcm=PPCM):
    """Draw the EduceLab sample square's two boards at a known scale."""
    px = int(round(BOARD_CM * ppcm))
    canvas = np.full((int(round(18 * ppcm)), int(round(6 * ppcm))), 255,
                     np.uint8)
    for idx in range(2):
        board = char.generate_board(offset=idx * 512)
        x = int(round(ORIGIN_CM * ppcm))
        y = int(round((ORIGIN_CM + idx * BOARD_GAP_CM) * ppcm))
        canvas[y:y + px, x:x + px] = board.generateImage((px, px))
    return canvas


@unittest.skipIf(MISSING, f'missing {MISSING}')
class DetectedBoardShapes(unittest.TestCase):
    """What ``detect_board`` promises, whatever OpenCV returned."""

    def setUp(self):
        self.img = sample_square()

    def test_detected_board_has_opencv4_layout(self):
        for offset in (0, 512):
            with self.subTest(offset=offset):
                b = char.detect_board(self.img,
                                      char.generate_board(offset=offset))
                self.assertEqual(4, b.marker_cnt)
                self.assertEqual(4, b.board_cnt)
                self.assertEqual((4, 1), b.marker_ids.shape)
                self.assertEqual((4, 1), b.board_ids.shape)
                self.assertEqual((4, 1, 2), b.board_corners.shape)
                self.assertEqual(4, len(b.marker_corners))
                for c in b.marker_corners:
                    self.assertEqual((1, 4, 2), np.asarray(c).shape)

    def test_a_board_corner_indexes_as_a_point(self):
        """``board_corners[i][0]`` is the (x, y) the consumers rotate.

        Under OpenCV 5's own ``(N, 2)`` it is a scalar, which is the shape
        assumption both ``educelab`` sites were written against.
        """
        b = char.detect_board(self.img, char.generate_board())
        for corner in b.board_corners:
            self.assertEqual((2,), np.asarray(corner[0]).shape)

    def test_results_are_sorted_by_id(self):
        b = char.detect_board(self.img, char.generate_board())
        self.assertEqual([0, 1, 2, 3], b.marker_ids.flatten().tolist())
        self.assertEqual([0, 1, 2, 3], b.board_ids.flatten().tolist())

    def test_nothing_detected_is_empty_not_malformed(self):
        b = char.detect_board(np.full((600, 600), 255, np.uint8),
                              char.generate_board())
        self.assertEqual(0, b.marker_cnt)
        self.assertEqual(0, b.board_cnt)
        self.assertEqual((), b.marker_corners)
        self.assertEqual((), b.board_corners)
        self.assertIsNone(b.marker_ids)
        self.assertIsNone(b.board_ids)

    def test_opencv_will_draw_what_detect_board_returned(self):
        """``educelab.main --output-image``, which OpenCV 5 refuses unnormalized."""
        img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        b = char.detect_board(self.img, char.generate_board())
        ar.drawDetectedMarkers(img, b.marker_corners, b.marker_ids)
        ar.drawDetectedCornersCharuco(img, b.board_corners, b.board_ids)


@unittest.skipIf(MISSING, f'missing {MISSING}')
class SampleSquareDetection(unittest.TestCase):
    """``detect_sample_square``: the resolution and pose ``pgs-center`` reads."""

    def setUp(self):
        self.img = sample_square()

    def assertSquareFound(self, result, msg=None):
        detected, boards, ppcm, kp_ids, kp_pos, _, _ = result
        self.assertTrue(detected, msg)
        self.assertEqual(EXPECTED_KEYPOINTS, len(kp_ids), msg)
        self.assertEqual(set(range(EXPECTED_KEYPOINTS)), set(kp_ids), msg)
        self.assertEqual(EXPECTED_KEYPOINTS, len(kp_pos), msg)
        # Drawn at PPCM, so ppcm has to come back to it. The tolerance is the
        # board's rasterization, not the estimator: it is stable to 0.04% on
        # real captures.
        self.assertAlmostEqual(PPCM, ppcm, delta=PPCM * 0.01, msg=msg)
        for b in boards:
            self.assertEqual((4, 1, 2), b.board_corners.shape, msg)

    def corners(self, result):
        """Every board and marker corner of a run, in detection order."""
        boards = result[1]
        return (np.concatenate([b.board_corners for b in boards]),
                np.concatenate([np.asarray(m) for b in boards
                                for m in b.marker_corners]))

    def test_upright_square(self):
        result = self.assertNoCrash(self.img)
        self.assertSquareFound(result)
        self.assertIsNone(result[5], 'upright frame should need no flip')
        self.assertIsNone(result[6], 'upright frame should need no rotation')

    def test_rotated_square_corrects_its_corners(self):
        """The rotation branch, which indexes ``board_corners`` to *write* it.

        Counts and shapes do not cover this: dropping ``rotate_kp`` from both
        writes leaves those green, because ``kp_pos`` is corrected in a
        separate loop. What the writes are for is ``center_mesh``' orientation
        solve, which reads ``marker_corners`` directly -- so the assertion has
        to be on the values. The fixture is pixel-aligned, so a rotated frame
        must reproduce the upright frame's corners exactly; the tolerance is
        here only for subpixel refinement, and is orders of magnitude tighter
        than an uncorrected corner would land.
        """
        upright = self.corners(self.assertNoCrash(self.img))
        for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                     cv2.ROTATE_90_COUNTERCLOCKWISE):
            with self.subTest(rotation=code):
                result = self.assertNoCrash(cv2.rotate(self.img, code))
                self.assertSquareFound(result)
                self.assertIsNotNone(result[6], 'rotation should be reported')
                for got, want, what in zip(self.corners(result), upright,
                                           ('board', 'marker')):
                    self.assertEqual(want.shape, got.shape, what)
                    self.assertLess(np.abs(got - want).max(), 1.,
                                    f'{what} corners were not rotated back '
                                    f'onto the upright frame')

    def test_flipped_square(self):
        """Both flips are found -- though not necessarily as the axis given.

        ``detect_sample_square`` probes axis 1 first and keeps the first
        orientation that yields more keypoints, and a board is still
        detectable upside down, so a vertically flipped frame comes back as
        the horizontal flip plus the 180 degree rotation that completes it.
        Asserting the pair rather than "not None" is what keeps that from
        silently becoming some other pair.
        """
        for axis, expected in ((0, (1, 1)), (1, (1, None))):
            with self.subTest(axis=axis):
                result = self.assertNoCrash(cv2.flip(self.img, axis))
                self.assertSquareFound(result)
                self.assertEqual(expected, (result[5], result[6]))

    def assertNoCrash(self, img):
        try:
            return el.detect_sample_square(img)
        except IndexError as e:
            self.fail(f'detect_sample_square indexed a moved shape: {e}')
