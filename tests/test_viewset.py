"""``pgs_recon.viewset``: the parts of ADR 0013's orchestrator that are logic.

What is tested is what the component *decides* -- which reference and prior a
plate gets, how a match graph splits, when a view's results may be reused, what
each artifact is called, what argv the C++ tool is handed. Nothing here runs a
binary or reads an image; the binary is mocked at ``tools.run``.

Stdlib-only, like the planner tests.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pgs_recon.viewset import (combined, groups, inputs, layout, manifest,
                               overhead, tools)
from pgs_recon.viewset.inputs import (NoPath, Overhead, PlateInputs, Prior,
                                      Reference, RigCapture)

OUT = Path('/sets/plate1')


class TestLayout(unittest.TestCase):
    def test_everything_lands_under_the_output_root(self):
        names = [layout.manifest(OUT), layout.ir_reference_mesh(OUT),
                 layout.overhead_position_map(OUT), layout.overhead_features(OUT),
                 layout.surviving_pairs(OUT),
                 layout.group_dir(OUT, 3), layout.joint_scene(OUT),
                 layout.joint_camera(OUT), layout.joint_report(OUT),
                 layout.qa_report(OUT), layout.starting_camera(OUT),
                 layout.group_scene(OUT, 0), layout.group_pairs(OUT, 0),
                 layout.group_putative_matches(OUT, 0),
                 layout.group_filtered_matches(OUT, 0),
                 layout.combined_sources(OUT), layout.combined_sfm(OUT),
                 layout.combined_undistorted(OUT), layout.combined_scene(OUT),
                 layout.combined_mesh(OUT),
                 layout.inputs_record(layout.overhead_position_map(OUT)),
                 layout.inputs_record(layout.virtual_overhead_image(OUT))]
        for stem in ('IMG_0001',):
            names += [layout.view_calibration(OUT, stem),
                      layout.view_adjusted_calibration(OUT, stem),
                      layout.view_camera(OUT, stem),
                      layout.view_correspondences(OUT, stem),
                      layout.view_report(OUT, stem),
                      layout.view_qa_render(OUT, stem),
                      layout.view_textured_mesh(OUT, stem)]
        for p in names:
            self.assertEqual(p.parts[:3], OUT.parts, p)
        self.assertEqual(len(set(names)), len(names), 'two artifacts collide')

    def test_mesh_only_pose_survives_the_adjustment(self):
        # Decision 5 needs both: the divergence is measured between them.
        self.assertNotEqual(layout.view_calibration(OUT, 'a'),
                            layout.view_adjusted_calibration(OUT, 'a'))

    def test_deliverable_keeps_the_calibration_suffix(self):
        self.assertEqual(layout.view_adjusted_calibration(OUT, 'a').name,
                         'a_calibration.json')

    def test_views_do_not_share_a_directory(self):
        self.assertNotEqual(layout.view_dir(OUT, 'a'), layout.view_dir(OUT, 'b'))


OVERHEAD = Overhead(Path('/o/overhead_calibration.json'), Path('/o/940.tif'))
RIG = RigCapture(Path('/scan'), 4)


def plate(**kw) -> PlateInputs:
    base = dict(recon_dir=Path('/recon'), mesh=Path('/recon/mvs/m.obj'),
                images=[Path('/h/a.jpg')])
    base.update(kw)
    return PlateInputs(**base)


class TestChoose(unittest.TestCase):
    def test_rig_ir940_is_preferred_over_the_overhead_texture(self):
        s = inputs.choose(plate(rig_ir_capture=RIG, overhead=OVERHEAD))
        self.assertIs(s.reference, Reference.RIG_IR940)
        self.assertIs(s.prior, Prior.OVERHEAD)

    def test_a_provided_mesh_wins(self):
        s = inputs.choose(plate(ir_reference_mesh=Path('/r.obj'),
                                rig_ir_capture=RIG, overhead=OVERHEAD))
        self.assertIs(s.reference, Reference.PROVIDED)

    def test_overhead_texture_is_a_named_fallback(self):
        s = inputs.choose(plate(overhead=OVERHEAD))
        self.assertIs(s.reference, Reference.OVERHEAD_940)
        self.assertTrue(any('FALLBACK' in r for r in s.reasons))

    def test_no_reference_is_refused(self):
        with self.assertRaises(NoPath):
            inputs.choose(plate())

    def test_no_overhead_is_refused_rather_than_seeded(self):
        # Manual seeding is named in the ADR but not built.
        with self.assertRaisesRegex(NoPath, 'Manual seeding'):
            inputs.choose(plate(rig_ir_capture=RIG))

    def test_virtual_overhead_stands_in_for_a_missing_one(self):
        s = inputs.choose(plate(ir_reference_mesh=Path('/r.obj'),
                                virtual_overhead=True))
        self.assertIs(s.reference, Reference.PROVIDED)
        self.assertIs(s.prior, Prior.VIRTUAL_OVERHEAD)
        self.assertTrue(any('not run' in r for r in s.reasons))

    def test_a_registered_overhead_beats_a_virtual_one(self):
        s = inputs.choose(plate(rig_ir_capture=RIG, overhead=OVERHEAD,
                                virtual_overhead=True))
        self.assertIs(s.prior, Prior.OVERHEAD)

    def test_virtual_overhead_cannot_be_the_reference(self):
        # It is rendered from the reference mesh, so it cannot supply one.
        with self.assertRaisesRegex(NoPath, 'No IR reference'):
            inputs.choose(plate(virtual_overhead=True))


class TestCollectImages(unittest.TestCase):
    def test_sorted_images_only(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('b.JPG', 'a.tif', 'notes.txt'):
                (Path(d) / name).touch()
            (Path(d) / 'sub').mkdir()
            got = [p.name for p in inputs.collect_images(Path(d))]
        self.assertEqual(got, ['a.tif', 'b.JPG'])

    def test_hidden_files_are_skipped(self):
        # macOS's AppleDouble files carry the image's suffix but no image.
        with tempfile.TemporaryDirectory() as d:
            for name in ('a.tif', '._a.tif', '.hidden.jpg'):
                (Path(d) / name).touch()
            got = [p.name for p in inputs.collect_images(Path(d))]
        self.assertEqual(got, ['a.tif'])

    def test_shared_stem_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'a.jpg').touch()
            (Path(d) / 'a.tif').touch()
            with self.assertRaisesRegex(ValueError, 'share the stem'):
                inputs.collect_images(Path(d))

    def test_a_rotated_image_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'a.tif').write_bytes(_tiff(1))
            (Path(d) / 'b.tif').write_bytes(_tiff(3))
            with self.assertRaisesRegex(ValueError,
                                        r'b\.tif \(Orientation = 3\)'):
                inputs.collect_images(Path(d))

    def test_orientation_one_and_untagged_pass(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'a.tif').write_bytes(_tiff(1))
            (Path(d) / 'b.tif').write_bytes(_tiff(None))
            got = [p.name for p in inputs.collect_images(Path(d))]
        self.assertEqual(got, ['a.tif', 'b.tif'])


def _tiff(orientation, order='<'):
    """A TIFF header and IFD0 with one SHORT Orientation entry, or none."""
    import struct
    entries = [] if orientation is None else [(274, 3, 1, orientation)]
    out = (b'II' if order == '<' else b'MM') + struct.pack(order + 'HI', 42, 8)
    out += struct.pack(order + 'H', len(entries) + 1)
    for tag, kind, n, value in [(256, 3, 1, 4)] + entries:
        out += struct.pack(order + 'HHIH2x', tag, kind, n, value)
    return out + struct.pack(order + 'I', 0)


class TestImageOrientation(unittest.TestCase):
    def read(self, data, suffix):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f'x{suffix}'
            p.write_bytes(data)
            return inputs.image_orientation(p)

    def test_tiff_either_byte_order(self):
        self.assertEqual(self.read(_tiff(3, '<'), '.tif'), 3)
        self.assertEqual(self.read(_tiff(6, '>'), '.tif'), 6)
        self.assertIsNone(self.read(_tiff(None), '.tif'))

    def test_jpeg_exif(self):
        import struct
        exif = b'Exif\x00\x00' + _tiff(3, '>')
        app0 = b'\xff\xe0' + struct.pack('>H', 2 + 5) + b'JFIF\x00'
        app1 = b'\xff\xe1' + struct.pack('>H', 2 + len(exif)) + exif
        self.assertEqual(self.read(b'\xff\xd8' + app0 + app1 + b'\xff\xda',
                                   '.jpg'), 3)
        self.assertIsNone(self.read(b'\xff\xd8' + app0 + b'\xff\xda', '.jpg'))

    def test_png_exif(self):
        import struct

        def chunk(kind, body):
            return struct.pack('>I', len(body)) + kind + body + b'\0' * 4
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr = chunk(b'IHDR', b'\0' * 13)
        self.assertEqual(self.read(sig + ihdr + chunk(b'eXIf', _tiff(8)) +
                                   chunk(b'IDAT', b''), '.png'), 8)
        self.assertIsNone(self.read(sig + ihdr + chunk(b'IDAT', b''), '.png'))

    def test_unparseable_files_are_left_to_the_reader(self):
        self.assertIsNone(self.read(b'', '.tif'))
        self.assertIsNone(self.read(b'II*\x00\xff\xff\xff\x00', '.tif'))
        self.assertIsNone(self.read(b'not an image', '.jpg'))


class TestGroups(unittest.TestCase):
    def test_islands_and_singletons(self):
        g = groups.split('abcdefg', [('a', 'b'), ('b', 'c'), ('e', 'f')])
        self.assertEqual(g.groups, (('a', 'b', 'c'), ('e', 'f')))
        self.assertEqual(g.singletons, ('d', 'g'))

    def test_order_is_size_then_first_member(self):
        g = groups.split('abcdef', [('e', 'f'), ('c', 'd'), ('a', 'b'),
                                    ('a', 'c')])
        self.assertEqual(g.groups, (('a', 'b', 'c', 'd'), ('e', 'f')))

    def test_pairs_order_does_not_change_the_answer(self):
        pairs = [('a', 'b'), ('d', 'c'), ('b', 'c'), ('e', 'f')]
        self.assertEqual(groups.split('abcdef', pairs),
                         groups.split('abcdef', list(reversed(pairs))))

    def test_self_pair_does_not_make_a_group(self):
        g = groups.split('ab', [('a', 'a')])
        self.assertEqual(g.groups, ())
        self.assertEqual(g.singletons, ('a', 'b'))

    def test_pair_outside_the_set_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'not in the view set'):
            groups.split('ab', [('a', 'z')])

    def test_no_pairs_means_all_singletons(self):
        g = groups.split([0, 1, 2], [])
        self.assertEqual((g.groups, g.singletons), ((), (0, 1, 2)))

    def test_read_pairs_both_spellings(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pairs.txt'
            p.write_text('0 1\n\n2 3 4\n')
            self.assertEqual(groups.read_pairs(p),
                             [(0, 1), (2, 3), (2, 4)])

    def test_read_pairs_rejects_a_lone_id(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pairs.txt'
            p.write_text('0 1\n7\n')
            with self.assertRaisesRegex(ValueError, ':2:'):
                groups.read_pairs(p)


FP_A = {'path': '/a', 'size': 1, 'mtime_ns': 1}
FP_A2 = {'path': '/a', 'size': 1, 'mtime_ns': 2}
FP_B = {'path': '/b', 'size': 5, 'mtime_ns': 9}


class TestManifest(unittest.TestCase):
    def test_current_view_is_reused(self):
        record = {'status': 'done', 'inputs': {'image': FP_A, 'reference': FP_B}}
        self.assertIsNone(manifest.stale_reason(
            record, {'image': FP_A, 'reference': FP_B}))

    def test_reasons(self):
        done = {'status': 'done', 'inputs': {'image': FP_A}}
        cases = [
            (None, {'image': FP_A}, 'never run'),
            ({'status': 'failed', 'inputs': {'image': FP_A}}, {'image': FP_A},
             'failed'),
            (done, {'image': FP_A2}, 'image changed'),
            (done, {'image': FP_A, 'camera': FP_B}, 'camera is a new input'),
            ({'status': 'done', 'inputs': {'image': FP_A, 'camera': FP_B}},
             {'image': FP_A}, 'camera is no longer an input'),
        ]
        for record, current, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, manifest.stale_reason(record, current))

    def test_provenance_drift(self):
        self.assertEqual(manifest.provenance_drift({'mesh': FP_A}, {'mesh': FP_A}),
                         {})
        self.assertEqual(
            manifest.provenance_drift({'mesh': FP_A, 'old': FP_B},
                                      {'mesh': FP_A2, 'new': FP_B}),
            {'mesh': 'changed', 'old': 'removed', 'new': 'added'})

    def test_round_trip_and_format_check(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'sub' / 'm.json'
            fresh = manifest.load(path)
            self.assertEqual(fresh['views'], {})
            fresh['views']['a'] = {'status': 'done'}
            manifest.write(path, fresh)
            self.assertEqual(manifest.load(path), fresh)
            self.assertFalse(path.with_suffix('.json.tmp').exists())
            path.write_text(json.dumps({'format': 999}))
            with self.assertRaisesRegex(ValueError, 'format 999'):
                manifest.load(path)

    def test_artifact_follows_its_reference_not_its_age(self):
        # A run pointed at another reference mesh keeps every output path, and
        # that mesh is routinely older than the render it should replace.
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            render, record = d / 'overhead.png', d / 'overhead.png.inputs.json'
            rgb, ir = d / 'final.obj', d / 'ir940.obj'
            rgb.write_text('rgb')
            ir.write_text('ir940')
            used = {'reference': manifest.fingerprint(rgb)}
            self.assertIn('missing', manifest.artifact_stale_reason(
                (render,), record, used))
            render.touch()
            self.assertIn('no record', manifest.artifact_stale_reason(
                (render,), record, used))
            manifest.record_artifact(record, used)
            self.assertIsNone(manifest.artifact_stale_reason(
                (render,), record, used))
            self.assertIn('reference changed', manifest.artifact_stale_reason(
                (render,), record,
                {'reference': manifest.fingerprint(ir)}))

    def test_fingerprint_sees_a_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'f'
            p.write_text('x')
            before = manifest.fingerprint(p)
            p.write_text('xy')
            self.assertNotEqual(before, manifest.fingerprint(p))


def project(cal: dict, point) -> tuple:
    """``point`` in ``cal``'s pixels, by its pinhole model (no distortion)."""
    d = cal['intrinsics'][0]['value']['ptr_wrapper']['data']
    pose = cal['extrinsics'][0]['value']
    rel = [p - c for p, c in zip(point, pose['center'])]
    x, y, z = (sum(r * v for r, v in zip(row, rel)) for row in pose['rotation'])
    return (d['focal_length'] * x / z + d['principal_point'][0],
            d['focal_length'] * y / z + d['principal_point'][1])


class TestVirtualOverhead(unittest.TestCase):
    BOUNDS = overhead.Bounds(-4.0, 1.0, -0.5, 6.0, 7.0, 0.5)

    def test_bounds_read_only_the_vertices(self):
        with tempfile.TemporaryDirectory() as d:
            obj = Path(d) / 'm.obj'
            obj.write_text('mtllib m.mtl\nv 1 2 3\nvt 9 9\nvn 0 0 1\n'
                           'v -1 5 0.5\nv 0 -2 7\nf 1/1/1 2/1/1 3/1/1\n')
            self.assertEqual(overhead.obj_bounds(obj),
                             overhead.Bounds(-1, -2, 0.5, 1, 5, 7))

    def test_a_mesh_without_vertices_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            obj = Path(d) / 'm.obj'
            obj.write_text('mtllib m.mtl\n')
            with self.assertRaisesRegex(ValueError, 'no vertices'):
                overhead.obj_bounds(obj)

    def test_the_box_fills_the_frame_less_the_margin(self):
        cal = overhead.calibration(self.BOUNDS, 'o.png', width=1000)
        view = cal['views'][0]['value']['ptr_wrapper']['data']
        w, h = view['width'], view['height']
        self.assertEqual(w, 1000)
        self.assertEqual(h, 600)  # the box's 10 x 6 aspect
        b = self.BOUNDS
        x0, y0 = project(cal, (b.xmin, b.ymin, b.zmax))
        x1, y1 = project(cal, (b.xmax, b.ymax, b.zmax))
        border = w * (1 - 1 / overhead.MARGIN) / 2
        self.assertAlmostEqual(min(x0, x1), border, places=6)
        self.assertAlmostEqual(max(x0, x1), w - border, places=6)
        self.assertAlmostEqual(min(y0, y1), h * (1 - 1 / overhead.MARGIN) / 2,
                               places=6)

    def test_looking_down_with_x_right_and_y_down_the_image(self):
        # The canonical overhead orientation: world +x is image right, world
        # +y is image *up*, and everything is in front of the camera.
        cal = overhead.calibration(self.BOUNDS, 'o.png', width=1000)
        b = self.BOUNDS
        left = project(cal, (b.xmin, 4.0, 0.0))
        right = project(cal, (b.xmax, 4.0, 0.0))
        low = project(cal, (1.0, b.ymin, 0.0))
        high = project(cal, (1.0, b.ymax, 0.0))
        self.assertLess(left[0], right[0])
        self.assertGreater(low[1], high[1])
        pose = cal['extrinsics'][0]['value']
        self.assertGreater(pose['center'][2], b.zmax)

    def test_names_the_image_it_will_be_rendered_to(self):
        cal = overhead.calibration(self.BOUNDS, 'virtual_overhead.png')
        self.assertEqual(
            cal['views'][0]['value']['ptr_wrapper']['data']['filename'],
            'virtual_overhead.png')

    def test_a_flat_footprint_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            overhead.calibration(overhead.Bounds(0, 0, 0, 0, 1, 1), 'o.png')


class TestMergeCalibrations(unittest.TestCase):
    """The scene ``--combined-texture`` hands openMVG2openMVS."""

    def write(self, d: Path, name: str, focal: float,
              model: str = 'pinhole_radial_k3') -> Path:
        cal = overhead.calibration(overhead.Bounds(0, 0, 0, 4, 3, 1),
                                   f'{name}.tif', width=400)
        intrinsic = cal['intrinsics'][0]['value']
        intrinsic['polymorphic_name'] = model
        intrinsic['ptr_wrapper']['data']['focal_length'] = focal
        cal['extrinsics'][0]['value']['center'][0] = focal
        path = d / f'{name}_calibration.json'
        path.write_text(json.dumps(cal))
        return path

    def merge(self, models=('pinhole_radial_k3',) * 3):
        with tempfile.TemporaryDirectory() as d:
            cals = [self.write(Path(d), f'v{i}', 1000.0 + i, m)
                    for i, m in enumerate(models)]
            return combined.merge_calibrations(
                cals, [f'v{i}.jpg' for i in range(len(cals))], Path('/src'))

    def test_every_view_keeps_its_own_pose_and_intrinsic(self):
        scene = self.merge()
        self.assertEqual(scene['root_path'], '/src')
        for key in range(3):
            view = scene['views'][key]['value']['ptr_wrapper']['data']
            self.assertEqual(view['filename'], f'v{key}.jpg')
            self.assertEqual((view['id_view'], view['id_pose'],
                              view['id_intrinsic']), (key, key, key))
            intrinsic = scene['intrinsics'][key]['value']['ptr_wrapper']
            self.assertEqual(intrinsic['data']['focal_length'], 1000.0 + key)
            self.assertEqual(scene['extrinsics'][key]['value']['center'][0],
                             1000.0 + key)

    def test_cereal_ids_are_declared_once_each(self):
        scene = self.merge()
        pointers = ([v['value']['ptr_wrapper']['id'] for v in scene['views']]
                    + [i['value']['ptr_wrapper']['id']
                       for i in scene['intrinsics']])
        self.assertEqual(pointers, [0x80000000 | i for i in range(1, 7)])
        # The camera model is registered by name on its first use only.
        first, *rest = [i['value'] for i in scene['intrinsics']]
        self.assertEqual(first['polymorphic_name'], 'pinhole_radial_k3')
        self.assertEqual(first['polymorphic_id'], 0x80000001)
        for value in rest:
            self.assertNotIn('polymorphic_name', value)
            self.assertEqual(value['polymorphic_id'], 1)
        self.assertEqual(list(scene)[0], 'sfm_data_version')

    def test_mixed_camera_models_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'one camera model'):
            self.merge(('pinhole_radial_k3', 'pinhole'))

    def test_one_view_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'at least two'):
            self.merge(('pinhole_radial_k3',))

    def test_counts_must_match(self):
        with self.assertRaisesRegex(ValueError, 'one each per view'):
            combined.merge_calibrations([Path('a'), Path('b')], ['a.jpg'],
                                        Path('/src'))


class TestTools(unittest.TestCase):
    """The argv each call hands ``pgs-localize-joint``."""

    def call(self, fn, **kw):
        with mock.patch.object(tools, 'resolve_exe',
                               return_value=Path('/p/bin/pgs-localize-joint')), \
                mock.patch.object(tools, 'run') as run:
            fn(**kw)
        return [str(t) for t in run.call_args.args[0]]

    def test_unset_optionals_are_omitted(self):
        argv = self.call(tools.position_map, calibration=Path('c.json'),
                         mesh=Path('m.obj'), output=Path('p.npy'))
        self.assertEqual(argv, ['/p/bin/pgs-localize-joint', 'position-map',
                                '--calibration', 'c.json', '--mesh', 'm.obj',
                                '--output', 'p.npy'])

    def test_position_map_can_also_write_the_render(self):
        argv = self.call(tools.position_map, calibration=Path('c.json'),
                         mesh=Path('m.obj'), output=Path('p.npy'),
                         output_image=Path('o.png'))
        self.assertEqual(argv[-2:], ['--output-image', 'o.png'])

    def test_view_carries_every_required_flag(self):
        argv = self.call(
            tools.localize_view, image=Path('a.jpg'), mesh=Path('r.obj'),
            overhead_image=Path('o.tif'), overhead_calibration=Path('o.json'),
            overhead_position_map=Path('p.npy'),
            output_calibration=Path('a_mesh_calibration.json'),
            output_correspondences=Path('a.npy'), report=Path('a.json'),
            camera=Path('k.txt'))
        self.assertEqual(argv[1], 'view')
        for flag in ('--image', '--mesh', '--overhead-image',
                     '--overhead-calibration', '--overhead-position-map',
                     '--output-calibration', '--output-correspondences',
                     '--report', '--camera'):
            self.assertIn(flag, argv)
        self.assertNotIn('--mask', argv)

    def test_overhead_features(self):
        argv = self.call(tools.overhead_features,
                         overhead_image=Path('o.tif'),
                         overhead_position_map=Path('p.npy'),
                         output=Path('f.npy'))
        self.assertEqual(argv, ['/p/bin/pgs-localize-joint', 'overhead-features',
                                '--overhead-image', 'o.tif',
                                '--overhead-position-map', 'p.npy',
                                '--output', 'f.npy'])

    def test_view_takes_the_overheads_features(self):
        argv = self.call(
            tools.localize_view, image=Path('a.jpg'), mesh=Path('r.obj'),
            overhead_image=Path('o.tif'), overhead_calibration=Path('o.json'),
            overhead_position_map=Path('p.npy'),
            output_calibration=Path('a_mesh_calibration.json'),
            output_correspondences=Path('a.npy'), report=Path('a.json'),
            camera=Path('k.txt'), overhead_features=Path('f.npy'))
        i = argv.index('--overhead-features')
        self.assertEqual(argv[i + 1], 'f.npy')

    def test_adjust_interleaves_each_views_triple(self):
        argv = self.call(
            tools.adjust, calibrations=[Path('a.json'), Path('b.json')],
            correspondences=[Path('a.npy'), Path('b.npy')],
            output_calibrations=[Path('A.json'), Path('B.json')],
            output_scene=Path('s.json'), output_camera=Path('k.txt'),
            report=Path('r.json'), fix_intrinsic=True)
        i = argv.index('--calibration')
        self.assertEqual(argv[i:i + 12], [
            '--calibration', 'a.json', '--correspondences', 'a.npy',
            '--output-calibration', 'A.json',
            '--calibration', 'b.json', '--correspondences', 'b.npy',
            '--output-calibration', 'B.json'])
        self.assertEqual(argv[-1], '--fix-intrinsic')

    def test_adjust_refuses_mismatched_counts(self):
        with self.assertRaisesRegex(ValueError, 'one each per view'):
            tools.adjust(calibrations=[Path('a')], correspondences=[],
                         output_calibrations=[Path('A')],
                         output_scene=Path('s'), output_camera=Path('k'),
                         report=Path('r'))


try:
    from pgs_recon.viewset import pipeline
except ImportError:  # its image helpers need numpy and friends
    pipeline = None


@unittest.skipIf(pipeline is None, 'pgs_recon.viewset.pipeline does not import')
class TestImportSet(unittest.TestCase):
    """The hand-held import follows the images, not its own existence: an output
    root reused for another image set must not keep the old scene."""

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.root = Path(self._d.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.output = self.root / 'out'

    def tearDown(self):
        self._d.cleanup()

    def plate(self, *names):
        for name in names:
            (self.images / name).write_text(name)
        return mock.Mock(images=[self.images / n for n in names])

    def run_import(self, plate):
        def init(images_dir, output_dir, **kw):
            (output_dir / 'sfm_data.json').write_text(json.dumps(
                sorted(p.name for p in plate.images)))
        with mock.patch.object(pipeline.openmvg, 'init_sfm_generic',
                               side_effect=init) as call, \
                mock.patch.object(pipeline, 'cam_db', return_value='db'):
            imported = pipeline.import_set(plate, self.output)
        return imported, call.call_count

    def test_the_same_images_are_not_reimported(self):
        plate = self.plate('a.tif', 'b.tif')
        self.assertEqual(self.run_import(plate)[1], 1)
        self.assertEqual(self.run_import(plate)[1], 0)

    def test_another_image_set_is_reimported_from_a_clean_sfm_dir(self):
        self.run_import(self.plate('a.tif', 'b.tif'))
        stale = layout.filtered_matches(self.output)
        stale.parent.mkdir(parents=True)
        stale.write_text('old matches')
        imported, calls = self.run_import(self.plate('c.tif', 'd.tif'))
        self.assertEqual(calls, 1)
        self.assertEqual(json.loads(imported.read_text()), ['c.tif', 'd.tif'])
        self.assertFalse(stale.exists())

    def test_a_scene_with_no_record_is_reimported(self):
        # What a run from before the record existed left behind.
        plate = self.plate('a.tif')
        sfm = layout.sfm_dir(self.output)
        sfm.mkdir(parents=True)
        (sfm / 'sfm_data.json').write_text('[]')
        self.assertEqual(self.run_import(plate)[1], 1)


@unittest.skipIf(pipeline is None, 'pgs_recon.viewset.pipeline does not import')
class TestOverheadFeatures(unittest.TestCase):
    """The overhead is described once per run, for every view that needs it,
    and only when one does."""

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        d = self.dir = Path(self._d.name)
        self.output = d / 'out'
        self.files = {}
        for name in ('o.png', 'o.json', 'reference.obj', 'position.npy',
                     'camera.txt', 'pgs-localize-joint', 'a.png', 'b.png'):
            self.files[name] = d / name
            self.files[name].write_text(name)
        self.plate = mock.Mock(
            images=[self.files['a.png'], self.files['b.png']],
            overhead=Overhead(self.files['o.json'], self.files['o.png']))
        self.meta = {'views': {}}

    def tearDown(self):
        self._d.cleanup()

    def localize(self, describe=None):
        def write(overhead_image, overhead_position_map, output):
            output.write_text('features')
        with mock.patch.object(pipeline.toolchain, 'resolve_exe',
                               return_value=self.files['pgs-localize-joint']), \
                mock.patch.object(pipeline.tools, 'overhead_features',
                                  side_effect=describe or write) as describer, \
                mock.patch.object(pipeline.tools, 'localize_view') as view:
            done = pipeline.localize_views(
                self.plate, self.files['reference.obj'],
                self.files['position.npy'], self.files['camera.txt'], {},
                self.output, pipeline.Options(), self.meta,
                self.dir / 'manifest.json')
        return done, describer, view

    def test_described_once_and_handed_to_every_view(self):
        done, describer, view = self.localize()
        self.assertEqual(done, ['a', 'b'])
        self.assertEqual(describer.call_count, 1)
        self.assertEqual(
            [c.kwargs['overhead_features'] for c in view.call_args_list],
            [layout.overhead_features(self.output)] * 2)

    def test_reused_across_runs_and_not_described_with_every_view_current(self):
        self.localize()
        del self.meta['views']['b']
        _, describer, view = self.localize()
        self.assertEqual(describer.call_count, 0)
        self.assertEqual(view.call_count, 1)
        _, describer, view = self.localize()
        self.assertEqual((describer.call_count, view.call_count), (0, 0))

    def test_a_failed_describe_leaves_each_view_to_describe_itself(self):
        def fail(*a, **kw):
            raise pipeline.ToolFailed(['pgs-localize-joint'], 1)
        with self.assertLogs(pipeline.logger, 'WARNING'):
            done, _, view = self.localize(describe=fail)
        self.assertEqual(done, ['a', 'b'])
        self.assertEqual(
            [c.kwargs['overhead_features'] for c in view.call_args_list],
            [None, None])


@unittest.skipIf(pipeline is None, 'pgs_recon.viewset.pipeline does not import')
class TestViewsSideBySide(unittest.TestCase):
    """``--jobs``: views localized and textured at once, recorded as if one
    at a time."""

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        d = self.dir = Path(self._d.name)
        self.output = d / 'out'
        self.files = {}
        for name in ('o.png', 'o.json', 'reference.obj', 'position.npy',
                     'camera.txt', 'mesh.obj', 'a.png', 'b.png', 'c.png',
                     'd.png'):
            self.files[name] = d / name
            self.files[name].write_text(name)
        self.stems = ['a', 'b', 'c', 'd']
        self.plate = mock.Mock(
            images=[self.files[f'{s}.png'] for s in self.stems],
            overhead=Overhead(self.files['o.json'], self.files['o.png']),
            mesh=self.files['mesh.obj'], recon_dir=d)
        self.meta = {'views': {}}
        self.meta_path = d / 'manifest.json'

    def tearDown(self):
        self._d.cleanup()

    def localize(self, view, jobs=4, threads=None):
        with mock.patch.object(pipeline, 'describe_overhead',
                               return_value=None), \
                mock.patch.object(pipeline.tools, 'localize_view',
                                  side_effect=view) as tool:
            done = pipeline.localize_views(
                self.plate, self.files['reference.obj'],
                self.files['position.npy'], self.files['camera.txt'], {},
                self.output, pipeline.Options(jobs=jobs, threads=threads),
                self.meta, self.meta_path)
        return done, tool

    def test_views_overlap_and_return_in_the_plates_order(self):
        import threading
        import time
        running, peak = set(), []
        lock = threading.Lock()

        def view(image, *a, **kw):
            with lock:
                running.add(image.stem)
                peak.append(len(running))
            # The first view finishes last.
            time.sleep(0.2 if image.stem == 'a' else 0.05)
            with lock:
                running.discard(image.stem)

        done, _ = self.localize(view)
        self.assertEqual(self.stems, done)
        self.assertGreater(max(peak), 1, 'the views ran one at a time')
        self.assertEqual({'done'}, {self.meta['views'][s]['status']
                                    for s in self.stems})
        on_disk = json.loads(self.meta_path.read_text())
        self.assertEqual('done', on_disk['views']['a']['status'])

    def test_a_failed_view_does_not_stop_the_others(self):
        def view(image, *a, **kw):
            if image.stem == 'b':
                raise pipeline.ToolFailed(['pgs-localize-joint'], 1)
        done, _ = self.localize(view)
        self.assertEqual(['a', 'c', 'd'], done)
        self.assertEqual('failed', self.meta['views']['b']['status'])

    def test_each_view_has_its_own_log_and_a_share_of_the_threads(self):
        _, tool = self.localize(lambda *a, **kw: None, jobs=4, threads=16)
        calls = {c.args[0].stem: c.kwargs for c in tool.call_args_list}
        self.assertEqual({4}, {kw['threads'] for kw in calls.values()})
        self.assertEqual(layout.view_log(self.output, 'c'), calls['c']['log'])

    def test_one_job_keeps_threads_as_given(self):
        _, tool = self.localize(lambda *a, **kw: None, jobs=1)
        self.assertEqual({None}, {c.kwargs['threads']
                                  for c in tool.call_args_list})

    def test_texturing_side_by_side_records_each_view(self):
        for stem in self.stems:
            cal = layout.view_adjusted_calibration(self.output, stem)
            cal.parent.mkdir(parents=True, exist_ok=True)
            cal.write_text('{}')
            self.meta['views'][stem] = {'status': 'done'}

        def retexture(argv, log=None):
            if '-i' in argv and Path(argv[argv.index('-i') + 1]).stem == 'c':
                raise pipeline.ToolFailed(argv, 1)
        with mock.patch.object(pipeline, 'run_tool',
                               side_effect=retexture) as tool:
            pipeline.texture_views(self.plate, self.stems, self.output,
                                   self.meta, self.meta_path, jobs=3)
        self.assertEqual(4, tool.call_count)
        self.assertEqual(
            {'a': 'done', 'b': 'done', 'c': 'failed', 'd': 'done'},
            {s: self.meta['views'][s]['texture']['status']
             for s in self.stems})


try:
    from pgs_recon.viewset import cli
except ImportError:  # configargparse
    cli = None


@unittest.skipIf(cli is None, 'needs configargparse')
class TestPlateMesh(unittest.TestCase):
    """Which mesh is textured when --mesh is not given."""

    def plate(self, *extra):
        args = cli._parser().parse_args(
            ['-r', '/recon', '-i', '/images', '-o', '/out', *extra])
        with mock.patch.object(cli, 'load_manifest'), \
                mock.patch.object(cli, 'collect_images', return_value=()), \
                mock.patch.object(cli, 'resolve_textured_mesh') as resolve:
            resolve.return_value.require.return_value = Path('/recon/final.obj')
            return cli._plate(args)

    def test_the_reconstructions_mesh_by_default(self):
        self.assertEqual(self.plate().mesh, Path('/recon/final.obj'))

    def test_a_provided_reference_is_textured(self):
        plate = self.plate('--ir-reference-mesh', '/recon/ir.obj')
        self.assertEqual(plate.mesh, Path('/recon/ir.obj'))

    def test_mesh_still_wins(self):
        plate = self.plate('--ir-reference-mesh', '/recon/ir.obj',
                           '--mesh', '/recon/centered.obj')
        self.assertEqual(plate.mesh, Path('/recon/centered.obj'))


@unittest.skipIf(pipeline is None, 'pgs_recon.viewset.pipeline does not import')
class TestGroupScene(unittest.TestCase):
    """The scene and pair file one group's tie points are matched from."""

    def solve(self, members):
        write = TestMergeCalibrations().write
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            cals = []
            for i in range(5):
                path = layout.view_calibration(out, f'v{i}')
                path.parent.mkdir(parents=True, exist_ok=True)
                cals.append(write(path.parent, f'v{i}', 1000.0 + i))
                cals[-1].rename(path)
            imported = out / 'imported.json'
            imported.write_text(json.dumps(combined.merge_calibrations(
                [layout.view_calibration(out, f'v{i}') for i in range(5)],
                [f'v{i}.tif' for i in range(5)], Path('/src'))))
            stem_of = {i: f'v{i}' for i in range(5)}
            pairs = [(a, b) for a in sorted(members) for b in sorted(members)
                     if a < b]
            with mock.patch.object(pipeline, 'openmvg') as mvg:
                pipeline._solve_group(imported, set(members), stem_of, pairs,
                                      0, out)
            scene = json.loads(layout.group_scene(out, 0).read_text())
            pair_ids = [tuple(map(int, line.split())) for line in
                        layout.group_pairs(out, 0).read_text().splitlines()]
            pairs_arg = mvg.compute_matches.call_args.kwargs['pairs_file']
            self.assertEqual(pairs_arg, layout.group_pairs(out, 0))
            self.residual = mvg.mvg_compute_known.call_args.kwargs.get(
                'residual_threshold')
        return scene, pair_ids

    def test_a_gap_in_the_set_is_closed_in_the_group(self):
        # View 2 did not localize. OpenMVG refuses a pair id that is not below
        # the scene's view count, which on PHerc118 Cr07 cost every tie point.
        scene, pair_ids = self.solve({0, 1, 3, 4})
        views = [v['value']['ptr_wrapper']['data'] for v in scene['views']]
        self.assertEqual([v['filename'] for v in views],
                         ['v0.tif', 'v1.tif', 'v3.tif', 'v4.tif'])
        self.assertEqual([v['key'] for v in scene['views']], [0, 1, 2, 3])
        self.assertEqual([(v['id_view'], v['id_pose']) for v in views],
                         [(k, k) for k in range(4)])
        self.assertTrue(all(a < 4 and b < 4 for a, b in pair_ids))
        # (1, 3) in the set is (1, 2) here: the same two images.
        self.assertIn((1, 2), pair_ids)
        self.assertEqual(len(pair_ids), 6)

    def test_each_view_keeps_its_own_pose(self):
        scene, _ = self.solve({0, 1, 3, 4})
        centers = {e['key']: e['value']['center'][0]
                   for e in scene['extrinsics']}
        self.assertEqual(centers, {0: 1000.0, 1: 1001.0, 2: 1003.0,
                                   3: 1004.0})

    def test_triangulation_cut_is_loose(self):
        # OpenMVG's own 4 px, under mesh-only poses, removed the ties between
        # exactly the views that disagreed most; adjust cuts strictly later.
        self.solve({0, 1, 3, 4})
        self.assertEqual(self.residual, pipeline.TIE_TRIANGULATION_PX)
        self.assertGreater(self.residual, 4.0)

    def test_every_intrinsic_a_view_names_is_present(self):
        scene, _ = self.solve({0, 3})
        named = {v['value']['ptr_wrapper']['data']['id_intrinsic']
                 for v in scene['views']}
        self.assertEqual(named, {i['key'] for i in scene['intrinsics']})


if __name__ == '__main__':
    unittest.main()
