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
                 layout.overhead_position_map(OUT), layout.surviving_pairs(OUT),
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


if __name__ == '__main__':
    unittest.main()
