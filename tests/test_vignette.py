"""``pgs_recon.viewset.vignette``: the lens vignette's mask.

Needs ``cv2`` and ``numpy``; skips without them.
"""
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
    from pgs_recon.viewset import vignette
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, 'needs cv2 and numpy')
class TestVignetteMask(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, img) -> Path:
        path = self.dir / 'view.png'
        cv2.imwrite(str(path), img)
        return path

    def test_masks_the_dark_corners_and_keeps_the_dark_middle(self):
        h, w = 600, 800
        y, x = np.mgrid[0:h, 0:w]
        img = np.full((h, w), 150, np.uint8)
        # A vignette: dark beyond a circle that clips the corners.
        img[np.hypot(x - w / 2, y - h / 2) > 420] = 10
        # Dark ink in the middle, not connected to any corner.
        img[280:320, 380:420] = 10
        mask = vignette.vignette_mask(self.write(img))
        for corner in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            self.assertEqual(mask[corner], 0, corner)
        self.assertEqual(mask[300, 400], 255)
        self.assertEqual(mask[300, 100], 255)
        # Grown inward past the circle's edge, where the falloff is.
        edge = int(w / 2 + (420 ** 2 - 290 ** 2) ** 0.5)
        self.assertEqual(mask[10, edge - vignette.GROW // 4], 0)

    def test_an_image_without_a_vignette_masks_nothing(self):
        img = np.full((300, 400), 150, np.uint8)
        masked = vignette.write_vignette_mask(self.write(img),
                                              self.dir / 'out' / 'mask.png')
        self.assertEqual(masked, 0.0)
        self.assertTrue((self.dir / 'out' / 'mask.png').is_file())

    def test_an_unreadable_image_is_refused(self):
        path = self.dir / 'broken.png'
        path.write_text('not an image')
        with self.assertRaisesRegex(ValueError, 'cannot read'):
            vignette.vignette_mask(path)


if __name__ == '__main__':
    unittest.main()
