"""``pgs_recon.viewset.vignette``: the lens vignette's masks, voted per lens
setup.

Needs ``cv2`` and ``numpy``; skips without them.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import cv2
    import numpy as np
    from pgs_recon.viewset import vignette
except ImportError:
    cv2 = None

try:
    from pgs_recon.viewset import layout, pipeline
except ImportError:     # the pipeline's own image dependencies
    pipeline = None

H, W = 600, 800
CORNERS = ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1))


def lens(radius=420, grey=150):
    """A plain scene through a lens dark beyond ``radius`` from the centre."""
    y, x = np.mgrid[0:H, 0:W]
    img = np.full((H, W), grey, np.uint8)
    img[np.hypot(x - W / 2, y - H / 2) > radius] = 10
    return img


def with_dark_object_at_left(img):
    """Dark papyrus along the left edge, touching the vignette there."""
    img = img.copy()
    img[150:450, 0:160] = 10
    return img


@unittest.skipIf(cv2 is None, 'needs cv2 and numpy')
class TestOneImage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, img, name='view.png') -> Path:
        path = self.dir / name
        cv2.imwrite(str(path), img)
        return path

    def test_masks_the_dark_corners_and_keeps_the_dark_middle(self):
        img = lens()
        # Dark ink in the middle, not connected to any corner.
        img[280:320, 380:420] = 10
        mask = vignette.vignette_mask(self.write(img))
        for corner in CORNERS:
            self.assertEqual(mask[corner], 0, corner)
        self.assertEqual(mask[300, 400], 255)
        self.assertEqual(mask[300, 100], 255)
        # Grown inward past the circle's edge, where the falloff is.
        edge = int(W / 2 + (420 ** 2 - 290 ** 2) ** 0.5)
        self.assertEqual(mask[10, edge - vignette.GROW // 4], 0)

    def test_an_unreadable_image_is_refused(self):
        path = self.dir / 'broken.png'
        path.write_text('not an image')
        with self.assertRaisesRegex(ValueError, 'cannot read'):
            vignette.vignette_mask(path)

    def test_corner_reach_is_zero_without_a_vignette(self):
        raw = vignette.raw_vignette(np.full((H, W), 150, np.uint8))
        self.assertEqual(vignette.corner_reaches(raw), (0.0, 0.0, 0.0, 0.0))

    def test_corner_reach_follows_the_vignette_not_a_side_object(self):
        plain = vignette.corner_reaches(vignette.raw_vignette(lens()))
        dark = vignette.corner_reaches(
            vignette.raw_vignette(with_dark_object_at_left(lens())))
        half_diagonal = (H ** 2 + W ** 2) ** 0.5 / 2
        self.assertAlmostEqual(plain[0], half_diagonal - 420, delta=3)
        self.assertEqual(plain, dark)


@unittest.skipIf(cv2 is None, 'needs cv2 and numpy')
class TestVote(unittest.TestCase):
    def raws(self, imgs):
        return [vignette.raw_vignette(i) for i in imgs]

    def test_dark_scene_in_one_image_is_outvoted(self):
        raws = self.raws([with_dark_object_at_left(lens())]
                         + [lens()] * 6)
        consensus = vignette.vote(raws, len(raws))
        self.assertTrue(consensus[0, 0])
        self.assertFalse(consensus[300, 80])
        np.testing.assert_array_equal(consensus, raws[1])
        self.assertLess(vignette.agreement(raws[0], consensus),
                        vignette.AGREE)
        self.assertEqual(vignette.agreement(raws[1], consensus), 1.0)

    def test_one_overexposed_image_does_not_shrink_the_mask(self):
        bright = np.full((H, W), 150, np.uint8)     # corners cleared DARK
        raws = self.raws([bright] + [lens()] * 6)
        consensus = vignette.vote(raws, len(raws))
        np.testing.assert_array_equal(consensus, raws[1])

    def test_a_lone_image_decides_for_itself(self):
        raws = self.raws([with_dark_object_at_left(lens())])
        np.testing.assert_array_equal(vignette.vote(raws, 1), raws[0])

    def test_no_images_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'no images'):
            vignette.vote([], 0)


class TestGroupByReach(unittest.TestCase):
    """Pure: no images needed, but the module imports cv2."""

    @unittest.skipIf(cv2 is None, 'needs cv2 and numpy')
    def test_two_setups_split_and_one_long_corner_does_not(self):
        narrow = [(168, 140, 158, 124), (170, 136, 160, 122),
                  (171, 186, 165, 140)]      # one corner touched by papyrus
        wide = [(224, 195, 209, 179), (226, 196, 210, 180)]
        groups = vignette.group_by_reach([narrow[0], wide[0], narrow[1],
                                          wide[1], narrow[2]])
        self.assertEqual(groups, [[0, 2, 4], [1, 3]])

    @unittest.skipIf(cv2 is None, 'needs cv2 and numpy')
    def test_tiny_vignettes_are_held_together_by_the_pixel_floor(self):
        self.assertEqual(vignette.group_by_reach([(0, 0, 0, 0), (4, 3, 5, 2)]),
                         [[0, 1]])


@unittest.skipIf(cv2 is None, 'needs cv2 and numpy')
class TestVignetteMasks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, img) -> Path:
        path = self.dir / name
        cv2.imwrite(str(path), img)
        return path

    def test_each_setup_votes_its_own_mask(self):
        narrow = [self.write(f'n{i}.png', lens(420)) for i in range(4)]
        dark = self.write('n_dark.png', with_dark_object_at_left(lens(420)))
        wide = [self.write(f'w{i}.png', lens(360)) for i in range(3)]
        groups = vignette.vignette_masks([dark] + narrow + wide)
        self.assertEqual([sorted(p.name for p in g.members) for g in groups],
                         [['n0.png', 'n1.png', 'n2.png', 'n3.png',
                           'n_dark.png'],
                          ['w0.png', 'w1.png', 'w2.png']])
        np.testing.assert_array_equal(groups[0].mask,
                                      vignette.vignette_mask(narrow[0]))
        np.testing.assert_array_equal(groups[1].mask,
                                      vignette.vignette_mask(wide[0]))
        self.assertLess(groups[0].agreement[dark], vignette.AGREE)
        self.assertEqual(groups[0].agreement[narrow[0]], 1.0)

    def test_sizes_never_share_a_group(self):
        a = self.write('a.png', lens())
        b = self.write('b.png', cv2.resize(lens(), (W // 2, H // 2),
                                           interpolation=cv2.INTER_NEAREST))
        groups = vignette.vignette_masks([a, b])
        self.assertEqual([[p.name for p in g.members] for g in groups],
                         [['a.png'], ['b.png']])

    def test_no_images_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'no images'):
            vignette.vignette_masks([])

    def test_identical_pixels_are_not_rewritten(self):
        target = self.dir / 'out' / 'mask.png'
        mask = vignette.grow(vignette.raw_vignette(lens()))
        self.assertTrue(vignette.write_mask(mask, target))
        os.utime(target, ns=(1, 1))
        self.assertFalse(vignette.write_mask(mask.copy(), target))
        self.assertEqual(target.stat().st_mtime_ns, 1)
        self.assertTrue(vignette.write_mask(255 - mask, target))


@unittest.skipIf(cv2 is None or pipeline is None,
                 'needs cv2, numpy and the pipeline\'s dependencies')
class TestPipelineMasks(unittest.TestCase):
    """What the run does with the masks: one per view, remade when an image
    changes, and part of each view's localization inputs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.out = self.dir / 'out'
        images = self.dir / 'images'
        images.mkdir()
        self.images = []
        for i in range(4):
            path = images / f'v{i}.png'
            cv2.imwrite(str(path), lens())
            self.images.append(path)
        self.plate = SimpleNamespace(images=self.images)

    def tearDown(self):
        self.tmp.cleanup()

    def opts(self, **kw):
        return pipeline.Options(**kw)

    def test_no_mask_asked_for(self):
        self.assertEqual(pipeline.view_masks(self.plate, self.out,
                                             self.opts()), {})

    def test_a_given_mask_serves_every_view(self):
        given = self.dir / 'square.png'
        masks = pipeline.view_masks(self.plate, self.out,
                                    self.opts(mask=given))
        self.assertEqual(masks, {p.stem: given for p in self.images})

    def test_vignette_masks_are_voted_and_reported(self):
        masks = pipeline.view_masks(self.plate, self.out,
                                    self.opts(vignette_mask=True))
        target = layout.vignette_group_mask(self.out, 0)
        self.assertEqual(masks, {p.stem: target for p in self.images})
        report = json.loads(layout.vignette_report(self.out).read_text())
        self.assertEqual(report['groups'][0]['members'],
                         [p.stem for p in self.images])
        self.assertEqual(report['groups'][0]['disagreeing'], [])

    def test_current_masks_are_reused_and_a_changed_image_revotes(self):
        pipeline.make_vignette_masks(self.plate, self.out)
        target = layout.vignette_group_mask(self.out, 0)
        os.utime(target, ns=(1, 1))
        with mock.patch.object(pipeline.logger, 'info') as info:
            pipeline.make_vignette_masks(self.plate, self.out)
        self.assertIn('current, reusing', info.call_args_list[0][0][0])

        # The first image now has dark papyrus at its edge. The vote is
        # retaken, outvotes it, and leaves the mask file -- and so every
        # view's fingerprint of it -- untouched.
        cv2.imwrite(str(self.images[0]),
                    with_dark_object_at_left(lens()))
        with self.assertLogs(pipeline.logger, 'WARNING') as logs:
            pipeline.make_vignette_masks(self.plate, self.out)
        self.assertTrue(any('v0' in m and 'overlaps' in m
                            for m in logs.output))
        self.assertEqual(target.stat().st_mtime_ns, 1)

    def test_a_changed_mask_relocalizes_the_view(self):
        files = {}
        for name in ('reference.obj', 'position.npy', 'camera.txt',
                     'mask_a.png', 'mask_b.png'):
            files[name] = self.dir / name
            files[name].write_text(name)
        plate = SimpleNamespace(images=self.images[:1], overhead=SimpleNamespace(
            image=self.dir / 'o.png', calibration=self.dir / 'o.json'))
        meta = {'views': {}}
        meta_path = self.dir / 'manifest.json'

        def localize(masks):
            with mock.patch.object(pipeline.tools, 'localize_view') as tool:
                done = pipeline.localize_views(
                    plate, files['reference.obj'], files['position.npy'],
                    files['camera.txt'], masks, self.out, self.opts(), meta,
                    meta_path)
            self.assertEqual(done, ['v0'])
            return tool

        tool = localize({'v0': files['mask_a.png']})
        self.assertEqual(tool.call_args.kwargs['mask'], files['mask_a.png'])
        self.assertEqual(meta['views']['v0']['inputs']['mask']['path'],
                         str(files['mask_a.png']))
        self.assertFalse(localize({'v0': files['mask_a.png']}).called)
        self.assertTrue(localize({'v0': files['mask_b.png']}).called)
        self.assertTrue(localize({}).called)


if __name__ == '__main__':
    unittest.main()
