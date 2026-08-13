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

No binary runs -- ``index_modality_images`` reads a directory and
``filter_sfm_for_cameras`` rewrites a JSON scene, so the fixtures are files.
"""
import importlib.util
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

DEPS = ('configargparse', 'cv2', 'numpy', 'sfm_utils', 'exiftool')
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

    def test_a_16bit_png_is_converted_despite_its_extension(self):
        """The extension says nothing about bit depth, so the decode is what
        decides -- a 16-bit PNG copied through would reach OpenMVS unusable."""
        from pgs_recon.apps.retexture import convert_modality_images
        img_map = self.images(['PGS_0_0_3.png'], 'uint16')
        out_dir = self.tmp / 'out'
        names = convert_modality_images(img_map, out_dir, 8)
        self.assertEqual('PGS_0_0_3.jpg', names[(0, 0)])


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


if __name__ == '__main__':
    unittest.main()
