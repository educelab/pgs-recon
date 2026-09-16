"""``pgs-retexture``'s capture-retexture correspondence: which images it picks
out of a scan directory, and which solved views it re-points at them.

Everything here is keyed on ``(camera, position)``, and *only* on that -- there is
no EXIF, ordering or geometric fallback (ADR 0001), so a mis-keyed image is a
mis-textured mesh with nothing to catch it downstream. That is what these tests
hold: the capture selection, the camera-set arithmetic between the solve and the
texturing capture, and the asymmetry between the two ways the two sets can
differ. An image the solve cannot place is loud (it was never calibrated); a
solved view the capture does not cover is quiet (the capture simply fired fewer
cameras).

The other half is the conversion those images go through: ``read_srgb`` replaced
OpenCV here, and ``convert_modality_images`` maps a *set* of frames to 8-bit with
one fixed scale, so relative radiometry survives and the merged atlas stays
seamless. That contract is why it cannot simply call ``prepare_8bit_image``.

No binary runs -- ``index_modality_images`` reads a directory and
``filter_sfm_for_cameras`` rewrites a JSON scene, so the fixtures are files.
"""
import importlib.util
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

DEPS = ('configargparse', 'cv2', 'numpy', 'sfm_utils', 'exiftool', 'imageio',
        'skimage', 'tifffile')
MISSING = [d for d in DEPS if importlib.util.find_spec(d) is None]

#: cereal sets this bit on the *first* instance of a polymorphic type.
POLY_FLAG = 0x80000000


def scan_dir(root: Path, images, prefix='PGS_', ext='tif', captures=None):
    """A PGS scan directory: ``metadata.json`` plus ``images``, each a
    ``(camera, position, capture)`` triple or an explicit filename."""
    root.mkdir(parents=True, exist_ok=True)
    scan = {'file_prefix': prefix, 'format': ext}
    if captures is not None:
        scan['capture_settings'] = captures
    (root / 'metadata.json').write_text(json.dumps({'scan': scan}))
    for img in images:
        name = img if isinstance(img, str) else \
            f'{prefix}{img[0]}_{img[1]}_{img[2]}.{ext}'
        (root / name).write_bytes(b'')
    return root


def sfm_scene(views, prefix='PGS_', ext='tif'):
    """A minimal OpenMVG SfM_Data JSON: one intrinsic and one pose per view,
    ``views`` being ``(camera, position)`` pairs. The first intrinsic carries the
    cereal type registration, so dropping it is what
    ``fix_polymorphic_registration`` has to repair."""
    data = {'root_path': '/nowhere', 'views': [], 'intrinsics': [],
            'extrinsics': [], 'structure': [], 'control_points': []}
    for key, (cam, pos) in enumerate(views):
        data['views'].append({'key': key, 'value': {'ptr_wrapper': {
            'id': 2000 + key,
            'data': {'local_path': 'sub/', 'filename': f'{prefix}{cam}_{pos}_0.{ext}',
                     'id_view': key, 'id_intrinsic': cam, 'id_pose': key,
                     'width': 4, 'height': 4}}}})
        data['extrinsics'].append(
            {'key': key, 'value': {'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   'center': [0, 0, float(key)]}})
    for n, cam in enumerate(sorted({c for c, _p in views})):
        value = {'polymorphic_id': 1 if n else POLY_FLAG | 1,
                 'ptr_wrapper': {'id': 3000 + n,
                                 'data': {'width': 4, 'height': 4,
                                          'focal_length': 1.0,
                                          'principal_point': [2, 2]}}}
        if n == 0:
            value['polymorphic_name'] = 'pinhole_radial_k3'
        data['intrinsics'].append({'key': cam, 'value': value})
    return data


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class RetextureCase(unittest.TestCase):
    #: What the tests assert log records against. The app logs through a module
    #: global that ``_main()`` rebinds to its own name, so which logger the
    #: functions use depends on whether anything has run ``_main()`` in this
    #: process. Substituting one removes that coupling.
    LOGGER = 'test-retexture'

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch('pgs_recon.apps.retexture.logger',
                             logging.getLogger(self.LOGGER))
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_sfm(self, views, **kwargs):
        path = self.tmp / 'sfm.json'
        path.write_text(json.dumps(sfm_scene(views, **kwargs)))
        return path


class TestIndexingSelectsOneCapture(RetextureCase):
    """The scan directory is the input, so ``metadata.json`` decides which files
    are images and ``--capture`` decides which of them are *this* run's."""

    def test_it_keys_every_camera_of_the_requested_capture(self):
        from pgs_recon.apps.retexture import index_modality_images
        root = scan_dir(self.tmp / 'scan',
                        [(0, 0, 0), (1, 0, 0), (0, 0, 3), (1, 0, 3), (1, 1, 3)])
        capture, prefix, images = index_modality_images(root, 3, None)
        self.assertEqual(3, capture)
        self.assertEqual('PGS_', prefix)
        # Keyed on (camera, position) across every camera -- the whole point of
        # the change. Capture 0's images, which share those keys, stay out.
        self.assertEqual({(0, 0), (1, 0), (1, 1)}, set(images))
        self.assertEqual('PGS_1_1_3.tif', images[(1, 1)].name)

    def test_a_lone_capture_is_inferred_and_several_are_refused(self):
        from pgs_recon.apps.retexture import index_modality_images
        one = scan_dir(self.tmp / 'one', [(0, 0, 2), (1, 0, 2)])
        self.assertEqual(2, index_modality_images(one, None, None)[0])
        # Ambiguous: texturing from an arbitrary capture is silently wrong, so it
        # is the user's call. The message has to name the choices.
        several = scan_dir(self.tmp / 'several', [(0, 0, 0), (0, 0, 3)])
        with self.assertRaises(SystemExit) as ctx:
            index_modality_images(several, None, None)
        self.assertIn('[0, 3]', str(ctx.exception))

    def test_a_capture_with_no_images_is_fatal_and_names_what_exists(self):
        from pgs_recon.apps.retexture import index_modality_images
        root = scan_dir(self.tmp / 'scan', [(0, 0, 0), (0, 0, 3)])
        with self.assertRaises(SystemExit) as ctx:
            index_modality_images(root, 5, None)
        self.assertIn('[0, 3]', str(ctx.exception))

    def test_an_undeclared_capture_is_a_warning_not_an_error(self):
        """A *derived* capture need not appear in ``capture_settings`` (CONTEXT.md,
        **Capture**), and filenames are what selection runs on -- so this matches
        ``import_pgs_scan``: warn, then texture from it anyway."""
        from pgs_recon.apps.retexture import index_modality_images
        root = scan_dir(self.tmp / 'scan', [(0, 0, 4)],
                        captures=[{'name': 'White'}])
        with self.assertLogs(self.LOGGER, 'WARNING') as logs:
            capture, _prefix, images = index_modality_images(root, 4, None)
        self.assertEqual(4, capture)
        self.assertEqual(1, len(images))
        self.assertIn('declares 1 capture', '\n'.join(logs.output))

    def test_the_metadata_format_is_what_resolves_two_files_per_slot(self):
        """``PGS_0_0_0.tif`` and ``PGS_0_0_0.jpg`` are one image in two formats.
        ``scan.format`` says which one this scan is made of, so the collision the
        old extension-agnostic scan produced cannot arise."""
        from pgs_recon.apps.retexture import index_modality_images
        root = scan_dir(self.tmp / 'scan', [(0, 0, 0)])
        (root / 'PGS_0_0_0.jpg').write_bytes(b'')
        _capture, _prefix, images = index_modality_images(root, 0, None)
        self.assertEqual(['PGS_0_0_0.tif'], [p.name for p in images.values()])

    def test_a_directory_without_metadata_is_refused(self):
        """Capture retexture reads a scan directory, always -- the metadata is
        what says how to read it, so guessing is not the fallback."""
        from pgs_recon.apps.retexture import index_modality_images
        bare = self.tmp / 'ir'
        bare.mkdir()
        (bare / 'PGS_0_0_3.tif').write_bytes(b'')
        with self.assertRaises(SystemExit) as ctx:
            index_modality_images(bare, 3, None)
        self.assertIn('metadata.json', str(ctx.exception))

    def test_an_unreadable_metadata_is_reported_against_the_file(self):
        """The metadata is the premise of the whole read, so a truncated or older
        one names itself rather than surfacing as a KeyError from inside."""
        from pgs_recon.apps.retexture import index_modality_images
        for text in ('{not json', '{}', '{"scan": {"file_prefix": "PGS_"}}'):
            root = self.tmp / f'scan{abs(hash(text))}'
            root.mkdir()
            (root / 'metadata.json').write_text(text)
            with self.assertRaises(SystemExit) as ctx:
                index_modality_images(root, 3, None)
            self.assertIn('metadata.json', str(ctx.exception))

    def test_matching_files_that_do_not_parse_blame_the_convention(self):
        """Nothing selected has two causes with different fixes: the metadata's
        prefix/format matched nothing, or it matched files whose names do not
        follow the convention. The second must not read as the first."""
        from pgs_recon.apps.retexture import index_modality_images
        root = scan_dir(self.tmp / 'scan', ['PGS_alpha.tif', 'PGS_beta.tif'])
        with self.assertRaises(SystemExit) as ctx:
            index_modality_images(root, 3, None)
        self.assertIn('naming convention', str(ctx.exception))
        empty = scan_dir(self.tmp / 'empty', [])
        with self.assertRaises(SystemExit) as ctx:
            index_modality_images(empty, 3, None)
        self.assertIn('No images matching', str(ctx.exception))


class TestCameraIndexRestrictsRatherThanAsserts(RetextureCase):
    """``--camera-index`` narrows what is available; it does not claim what
    exists. Only an empty result stops the run, because a tool that isn't running
    yet must not create a hard dependency on cameras a capture happens to hold."""

    def test_it_keeps_the_requested_cameras(self):
        from pgs_recon.apps.retexture import index_modality_images
        root = scan_dir(self.tmp / 'scan',
                        [(0, 0, 3), (1, 0, 3), (2, 0, 3), (3, 0, 3)])
        _capture, _prefix, images = index_modality_images(root, 3, [1, 3])
        self.assertEqual({(1, 0), (3, 0)}, set(images))

    def test_a_camera_the_capture_lacks_is_loud_but_survivable(self):
        from pgs_recon.apps.retexture import index_modality_images
        root = scan_dir(self.tmp / 'scan', [(1, 0, 3)])
        with self.assertLogs(self.LOGGER, 'WARNING') as logs:
            _capture, _prefix, images = index_modality_images(root, 3, [1, 3])
        self.assertEqual({(1, 0)}, set(images))
        self.assertIn('[3]', '\n'.join(logs.output))

    def test_only_an_empty_selection_is_fatal(self):
        """``-k 1 3`` against ``{1}`` warns; ``-k 1 3`` and ``-k 13`` against
        nothing are the same condition, and it is the fatal one."""
        from pgs_recon.apps.retexture import index_modality_images
        root = scan_dir(self.tmp / 'scan', [(0, 0, 3), (2, 0, 3)])
        for requested in ([1, 3], [13]):
            with self.assertRaises(SystemExit) as ctx:
                index_modality_images(root, 3, requested)
            self.assertIn('[0, 2]', str(ctx.exception))


class TestConversionPreservesWhatIsAlreadyUsable(RetextureCase):
    """``-i`` is a scan directory, so the source may already be 8-bit JPEG. The
    conversion exists to make images readable, not to re-encode them (ADR 0001,
    Radiometry): a second lossy generation would change the pixels it claims to
    be a faithful copy of."""

    def images(self, names, dtype='uint8'):
        import numpy as np
        import cv2
        out = {}
        for key, name in enumerate(names):
            path = self.tmp / name
            depth = np.uint16 if dtype == 'uint16' else np.uint8
            img = np.full((4, 4, 3), 40, dtype=depth)
            cv2.imwrite(str(path), img)
            out[(key, 0)] = path
        return out

    def test_an_8bit_jpeg_is_copied_and_a_16bit_tiff_is_converted(self):
        from pgs_recon.apps.retexture import convert_modality_images
        img_map = self.images(['PGS_0_0_3.jpg'])
        img_map.update({(1, 0): self.images(['PGS_1_0_3.tif'],
                                            'uint16')[(0, 0)]})
        out_dir = self.tmp / 'out'
        names = convert_modality_images(img_map, out_dir, 8)
        self.assertEqual('PGS_0_0_3.jpg', names[(0, 0)])
        self.assertEqual('PGS_1_0_3.jpg', names[(1, 0)])
        # The copy is byte-for-byte; the conversion is not a copy.
        self.assertEqual((self.tmp / 'PGS_0_0_3.jpg').read_bytes(),
                         (out_dir / 'PGS_0_0_3.jpg').read_bytes())
        self.assertTrue((out_dir / 'PGS_1_0_3.jpg').is_file())

    def test_a_16bit_rgba_png_loses_its_alpha_rather_than_the_run(self):
        """Only 8-bit images are copied through, so an RGBA one reaches the JPEG
        writer -- which refuses four channels, aborting the whole retexture."""
        from pgs_recon.apps.retexture import convert_modality_images
        import imageio.v3 as iio
        import numpy as np
        import cv2
        src = self.tmp / 'PGS_0_0_3.png'
        cv2.imwrite(str(src), np.full((4, 4, 4), 40, dtype=np.uint16))
        out_dir = self.tmp / 'out'
        names = convert_modality_images({(0, 0): src}, out_dir, 8)
        self.assertEqual(iio.imread(out_dir / names[(0, 0)]).shape, (4, 4, 3))

    def test_a_16bit_png_is_converted_despite_its_extension(self):
        """The extension says nothing about bit depth, so the decode is what
        decides -- a 16-bit PNG copied through would reach OpenMVS unusable."""
        from pgs_recon.apps.retexture import convert_modality_images
        img_map = self.images(['PGS_0_0_3.png'], 'uint16')
        out_dir = self.tmp / 'out'
        names = convert_modality_images(img_map, out_dir, 8)
        self.assertEqual('PGS_0_0_3.jpg', names[(0, 0)])


class TestConversionArithmeticIsFixedPerSet(RetextureCase):
    """What the conversion does to the pixels it cannot copy through.

    Two behaviours changed when it moved off OpenCV onto ``read_srgb`` and both
    are asserted here: a CIELab TIFF is now decoded rather than taken
    channel-for-channel as BGR, and a shift small enough to leave values above
    255 clips instead of wrapping (``(v >> 6).astype(uint8)`` turned the
    brightest pixels black).
    """

    def convert(self, samples, bit_shift=8, photometric='minisblack'):
        """Run one image through and read back what was written."""
        import imageio.v3 as iio
        import tifffile
        from pgs_recon.apps.retexture import convert_modality_images
        src = self.tmp / 'IR_000_00000_00.tif'
        tifffile.imwrite(src, samples, photometric=photometric)
        out = self.tmp / f'out{bit_shift}'
        names = convert_modality_images({(0, 0): src}, out, bit_shift)
        return iio.imread(out / names[(0, 0)])

    def expected(self, pixels):
        """A reference array through the same JPEG encoder.

        ``convert_modality_images`` writes a lossy file, so comparing its output
        to a raw array would test libjpeg, not the conversion. Encoding the
        reference identically cancels that: equal inputs give equal files, so
        any difference that survives is arithmetic.
        """
        import imageio.v3 as iio
        path = self.tmp / 'expected.jpg'
        iio.imwrite(path, pixels, quality=100)
        return iio.imread(path)

    def test_16bit_shift_matches_the_old_bit_shift(self):
        """The default path must be bit-identical to what OpenCV produced."""
        import numpy as np
        src = (np.arange(256, dtype=np.uint16) * 257).reshape(16, 16)
        for shift in (8, 10, 12):
            with self.subTest(bit_shift=shift):
                np.testing.assert_array_equal(
                    self.convert(src, shift),
                    self.expected((src >> shift).astype(np.uint8)))

    def test_small_shift_clips_instead_of_wrapping(self):
        """``(v >> 6).astype(uint8)`` wrapped, turning bright pixels black."""
        import numpy as np
        # >> 6 sends these to 0, 256, 512 and 1023: everything but the first
        # overflows a uint8, and the middle two wrap to exactly 0.
        src = np.tile(np.array([0, 16384, 32768, 65535], dtype=np.uint16),
                      (16, 4))
        np.testing.assert_array_equal(
            self.convert(src, 6),
            self.expected(np.clip(src >> 6, 0, 255).astype(np.uint8)))
        wrapped = (src >> 6).astype(np.uint8)
        self.assertEqual(int(wrapped[0, 1]), 0, 'the bug this replaces')
        self.assertEqual(int(wrapped[0, 2]), 0)

    def test_scale_is_uniform_across_frames(self):
        """Per-frame normalization would break the merged texture."""
        import numpy as np
        dim = (np.arange(256, dtype=np.uint16) * 40).reshape(16, 16)
        bright = (np.arange(256, dtype=np.uint16) * 257).reshape(16, 16)
        # the dim frame stays dim -- it is not stretched to fill the range
        self.assertLessEqual(int(self.convert(dim, 8).max()),
                             int((dim >> 8).max()) + 1)
        self.assertGreaterEqual(int(self.convert(bright, 8).max()), 254)

    def test_8bit_input_passes_through_unchanged(self):
        import numpy as np
        src = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        np.testing.assert_array_equal(self.convert(src, photometric='rgb'),
                                      self.expected(src))

    def test_lab_is_decoded_not_taken_as_bgr(self):
        """OpenCV read L/a/b as B/G/R; a decoded neutral ramp proves it does not."""
        import numpy as np
        from pgs_recon.utils.images import CIELAB
        from tests.test_images import encode
        from skimage.color import rgb2lab
        # a neutral grey ramp: decoded it must stay grey (R == G == B)
        grey = np.repeat(np.arange(20, 240, 20, dtype=np.uint8), 3)
        grey = grey.reshape(1, 11, 3)
        samples = encode(rgb2lab(grey / 255.), CIELAB, np.uint8)
        got = self.convert(samples, photometric='cielab').astype(int)
        self.assertEqual(got.shape[-1], 3)
        self.assertLessEqual(np.abs(got[..., 0] - got[..., 1]).max(), 2)
        self.assertLessEqual(np.abs(got[..., 1] - got[..., 2]).max(), 2)
        np.testing.assert_allclose(got[0, :, 0], grey[0, :, 0], atol=3)


class TestViewNamesAreParsedAgainstTheScanPrefix(RetextureCase):
    def test_the_prefix_anchors_and_the_extension_does_not(self):
        """A solve may have been imported from converted copies, so the suffix
        carries no information. The prefix anchor is what makes the optional
        capture field safe: unanchored, ``PGS_003_00047_02`` also reads as camera
        47, position 2."""
        from pgs_recon.apps.retexture import parse_view_name
        self.assertEqual((3, 47), parse_view_name('PGS_3_47_2.tif', 'PGS_'))
        self.assertEqual((3, 47), parse_view_name('PGS_3_47_2.jpg', 'PGS_'))
        self.assertEqual((3, 47), parse_view_name('PGS_3_47.png', 'PGS_'))
        self.assertIsNone(parse_view_name('IMG_3_47_2.tif', 'PGS_'))
        self.assertIsNone(parse_view_name('PGS_3_47_2', 'PGS_'))


class TestFilteringSpansTheCapturesCameras(RetextureCase):
    """The scene surgery of ADR 0001, now over a camera *set*: keep the views
    that have an image, re-point them, and leave nothing orphaned behind."""

    def test_it_repoints_every_camera_and_prunes_what_no_view_uses(self):
        from pgs_recon.apps.retexture import filter_sfm_for_cameras
        sfm = self.write_sfm([(0, 0), (1, 0), (2, 0)])
        out = self.tmp / 'filtered.json'
        kept = filter_sfm_for_cameras(
            sfm, 'PGS_', self.tmp / 'modality',
            {(0, 0): 'PGS_0_0_3.jpg', (2, 0): 'PGS_2_0_3.jpg'}, out)
        self.assertEqual(2, kept)
        data = json.loads(out.read_text())
        # Both cameras' views survive, re-pointed, rooted at the modality dir.
        self.assertEqual(['PGS_0_0_3.jpg', 'PGS_2_0_3.jpg'],
                         [v['value']['ptr_wrapper']['data']['filename']
                          for v in data['views']])
        self.assertEqual([''], list({v['value']['ptr_wrapper']['data']['local_path']
                                     for v in data['views']}))
        self.assertEqual(str((self.tmp / 'modality').resolve()),
                         data['root_path'])
        # openMVG2openMVS rejects a scene carrying an intrinsic or pose no view
        # references, so camera 1's must be gone -- both of them.
        self.assertEqual([0, 2], [i['key'] for i in data['intrinsics']])
        self.assertEqual([0, 2], [e['key'] for e in data['extrinsics']])
        # Camera 0 registered the cereal type and stayed, so it still does.
        self.assertEqual(POLY_FLAG | 1,
                         data['intrinsics'][0]['value']['polymorphic_id'])
        self.assertNotIn('polymorphic_name', data['intrinsics'][1]['value'])

    def test_the_registration_moves_when_the_first_camera_drops_out(self):
        """Texturing from cameras ``{1, 2}`` of a solve of ``{0, 1, 2}`` drops the
        intrinsic that registered the type; without the promotion the scene fails
        to load with "Could not find type id"."""
        from pgs_recon.apps.retexture import filter_sfm_for_cameras
        sfm = self.write_sfm([(0, 0), (1, 0), (2, 0)])
        out = self.tmp / 'filtered.json'
        filter_sfm_for_cameras(sfm, 'PGS_', self.tmp / 'modality',
                               {(1, 0): 'a.jpg', (2, 0): 'b.jpg'}, out)
        intrinsics = json.loads(out.read_text())['intrinsics']
        self.assertEqual([1, 2], [i['key'] for i in intrinsics])
        self.assertEqual(POLY_FLAG | 1, intrinsics[0]['value']['polymorphic_id'])
        self.assertEqual('pinhole_radial_k3',
                         intrinsics[0]['value']['polymorphic_name'])
        self.assertEqual(1, intrinsics[1]['value']['polymorphic_id'])

    def test_an_uncalibrated_image_is_loud(self):
        """Images for a ``(camera, position)`` the solve never reconstructed
        cannot be placed at all -- the one absence that means something is
        wrong."""
        from pgs_recon.apps.retexture import filter_sfm_for_cameras
        sfm = self.write_sfm([(0, 0)])
        with self.assertLogs(self.LOGGER, 'WARNING') as logs:
            filter_sfm_for_cameras(
                sfm, 'PGS_', self.tmp / 'modality',
                {(0, 0): 'a.jpg', (0, 9): 'b.jpg', (4, 0): 'c.jpg'},
                self.tmp / 'filtered.json')
        message = '\n'.join(logs.output)
        self.assertIn('2 modality image(s) have no solved view', message)
        self.assertIn('[0, 4]', message)

    def test_a_view_the_capture_does_not_cover_is_quiet(self):
        """The mirror case, and not a fault: a capture that fired two of five
        cameras is a capture, not an error. Nothing at WARNING or above."""
        from pgs_recon.apps.retexture import filter_sfm_for_cameras
        sfm = self.write_sfm([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
        with self.assertLogs(self.LOGGER, 'INFO') as logs:
            kept = filter_sfm_for_cameras(
                sfm, 'PGS_', self.tmp / 'modality',
                {(1, 0): 'a.jpg', (3, 0): 'b.jpg'}, self.tmp / 'filtered.json')
        self.assertEqual(2, kept)
        self.assertEqual([], [r for r in logs.records
                              if r.levelno >= logging.WARNING])

    def test_views_that_do_not_parse_are_named_as_the_reason(self):
        """A recon imported generically (no PGS names) can never be capture
        retextured, and the old message blamed the camera index for it."""
        from pgs_recon.apps.retexture import filter_sfm_for_cameras
        sfm = self.write_sfm([(0, 0), (1, 0)], prefix='IMG_')
        with self.assertRaises(SystemExit) as ctx:
            filter_sfm_for_cameras(sfm, 'PGS_', self.tmp / 'modality',
                                   {(0, 0): 'a.jpg'}, self.tmp / 'filtered.json')
        self.assertIn("prefix 'PGS_'", str(ctx.exception))

    def test_a_shared_convention_with_no_shared_slot_is_its_own_message(self):
        """Names parse, cameras and positions simply do not meet -- e.g. capture 3
        holds camera 4 alone and the solve never had it."""
        from pgs_recon.apps.retexture import filter_sfm_for_cameras
        sfm = self.write_sfm([(0, 0), (1, 0)])
        with self.assertRaises(SystemExit) as ctx:
            filter_sfm_for_cameras(sfm, 'PGS_', self.tmp / 'modality',
                                   {(4, 0): 'a.jpg'}, self.tmp / 'filtered.json')
        self.assertIn('No solved view shares', str(ctx.exception))


class TestProjectiveTexturingTexturesOnlyWhatIsSeen(RetextureCase):
    """``--calibration`` without ``--use-openmvs`` textures by projecting the
    mesh through the one calibrated view, and a face that projects inside the
    image is not the same thing as a face the camera saw: anything behind a
    nearer part of the mesh would otherwise take the foreground's pixels.

    The kernel is tested in ``test_visibility``; what is held here is the wiring
    and the OBJ it writes -- an unseen face keeps its geometry and loses its
    UVs, which is the same thing this file's out-of-view faces already do.
    """
    def calibration(self):
        """``test_visibility``'s camera, written as a one-view calibration:
        the kernel's tests and these have to be aimed at the same camera, and a
        second copy of a principal point is how they stop being. Only the lone
        intrinsic and extrinsic are read (``camera_from_calibration``), so only
        those are written."""
        from tests.test_visibility import CAM, H, W
        path = self.tmp / 'cal.json'
        path.write_text(json.dumps({
            'intrinsics': [{'key': 0, 'value': {
                'polymorphic_id': POLY_FLAG | 1,
                'polymorphic_name': 'pinhole',
                'ptr_wrapper': {'id': 1, 'data': {
                    'width': W, 'height': H, 'focal_length': CAM['f'],
                    'principal_point': [CAM['cx'], CAM['cy']]}}}}],
            'extrinsics': [{'key': 0, 'value': {
                'rotation': CAM['R'].tolist(),
                'center': CAM['C'].tolist()}}]}))
        return path

    def write_mesh(self, *quads):
        """An OBJ of flat quads, each given as ``(depth, x0, x1, y0, y1)`` with
        the extents in *pixels* off the principal point -- what a patch covers
        in the image is what matters here, and it is not its size in the scene.
        Wound so every quad faces the camera, so back-face culling keeps them
        all and only the depth test can drop one."""
        from tests.test_visibility import CAM
        path = self.tmp / 'mesh.obj'
        verts, faces = [], []
        for z, x0, x1, y0, y1 in quads:
            wx0, wx1, wy0, wy1 = (p * z / CAM['f'] for p in (x0, x1, y0, y1))
            n = len(verts)
            verts += [(wx0, wy0), (wx0, wy1), (wx1, wy1), (wx1, wy0)]
            faces += [(n + 1, n + 2, n + 3), (n + 1, n + 3, n + 4)]
            for i in range(n, len(verts)):
                verts[i] = verts[i] + (z,)
        path.write_text(
            '\n'.join([f'v {x} {y} {z}' for x, y, z in verts]
                      + [f'f {a} {b} {c}' for a, b, c in faces]) + '\n')
        return path

    def texture(self, mesh, **kwargs):
        """Project ``mesh`` through the calibration; returns the counts of
        textured (``f v/vt ...``) and untextured (``f v ...``) faces."""
        from pgs_recon.apps.retexture import project_texture_mesh
        image = self.tmp / 'ir940.jpg'
        image.write_bytes(b'not read: the projective path copies it')
        out = self.tmp / 'out' / 'textured.obj'
        project_texture_mesh(self.calibration(), image, mesh, out, **kwargs)
        faces = [ln for ln in out.read_text().splitlines()
                 if ln.startswith('f ')]
        self.assertTrue((out.parent / 'textured.jpg').is_file())
        return (sum('/' in ln for ln in faces),
                sum('/' not in ln for ln in faces))

    #: A 40x40 px patch at depth 5, a small quad hidden right behind it, and a
    #: third quad off to the side that nothing covers.
    HIDDEN = ((5.0, -20, 20, -20, 20), (10.0, -5, 5, -5, 5),
              (10.0, 20, 40, -10, 10))

    def test_a_hidden_face_keeps_its_geometry_and_loses_its_uvs(self):
        textured, plain = self.texture(self.write_mesh(*self.HIDDEN))
        self.assertEqual((textured, plain), (4, 2))

    def test_without_the_depth_test_the_hidden_face_is_textured_too(self):
        # Same mesh, same in-view and front-facing tests: the difference is the
        # depth test alone, which is what --no-occlusion-cull turns off.
        textured, plain = self.texture(self.write_mesh(*self.HIDDEN),
                                       occlusion_coverage=None)
        self.assertEqual((textured, plain), (6, 0))

    def test_a_partly_hidden_face_is_the_coverage_threshold_s_to_decide(self):
        # A quad running from the middle of the occluder to well outside it.
        # The diagonal splits the covered part unevenly, leaving one of its two
        # faces a quarter visible and the other three quarters, so the threshold
        # can be read off the count: below both, between them, above both.
        mesh = self.write_mesh((5.0, -20, 20, -20, 20),
                               (10.0, 0, 40, -10, 10))
        self.assertEqual(self.texture(mesh, occlusion_coverage=0.2), (4, 0))
        self.assertEqual(self.texture(mesh, occlusion_coverage=0.5), (3, 1))
        self.assertEqual(self.texture(mesh), (2, 2))

    def test_the_mesh_is_never_missing_a_face(self):
        """Unseen faces are omitted from the texture, not from the mesh: this
        is a retexture, and dropping geometry would make it a different mesh."""
        # The default and --no-occlusion-cull are pinned face-for-face above,
        # at (4, 2) and (6, 0); a partial threshold is the case they miss.
        textured, plain = self.texture(self.write_mesh(*self.HIDDEN),
                                       occlusion_coverage=0.25)
        self.assertEqual(textured + plain, 6)

    def run_main(self, *extra):
        """Drive the app itself, so what is under test is the wiring and not
        just ``project_texture_mesh``. Returns the working dir."""
        from pgs_recon.apps import retexture
        recon = self.tmp / 'recon'
        recon.mkdir()
        (recon / 'pgs-recon.json').write_text('{"stages": {}}')
        image = self.tmp / 'ir940.jpg'
        image.write_bytes(b'not read: the projective path copies it')
        work = self.tmp / 'work'
        argv = ['pgs-retexture', '-i', str(image),
                '--calibration', str(self.calibration()),
                '--recon-dir', str(recon),
                '--mesh', str(self.write_mesh(*self.HIDDEN)),
                '-w', str(work), '--log-level', 'ERROR', *extra]
        # The manifest hook is an atexit in the app; run it now instead, or it
        # fires against a deleted temp dir once the interpreter exits.
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(retexture.atexit, 'register', lambda f: f):
            retexture._main()
        return work

    def test_it_leaves_no_empty_mvs_scaffolding(self):
        """``mvg/`` and ``mvs/`` hold an MVS scene, and this path builds none:
        naming those directories must not be what creates them."""
        work = self.run_main('-o', str(self.tmp / 'out' / 'textured.obj'))
        self.assertTrue((self.tmp / 'out' / 'textured.obj').is_file())
        self.assertFalse((work / 'mvg').exists())
        self.assertFalse((work / 'mvs').exists())

    def test_the_default_output_still_lands_in_mvs(self):
        """Created where written, not never: with no --output-mesh the
        deliverable's own home is ``mvs/``, so that one does appear -- holding
        something. ``mvg/`` has no reason to exist either way."""
        work = self.run_main()
        self.assertTrue((work / 'mvs' / 'ir940.obj').is_file())
        self.assertFalse((work / 'mvg').exists())


if __name__ == '__main__':
    unittest.main()
