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

from pgs_recon.viewset import groups, inputs, layout, manifest, tools
from pgs_recon.viewset.inputs import (NoPath, Overhead, PlateInputs, Prior,
                                      Reference, RigCapture)

OUT = Path('/sets/plate1')


class TestLayout(unittest.TestCase):
    def test_everything_lands_under_the_output_root(self):
        names = [layout.manifest(OUT), layout.ir_reference_mesh(OUT),
                 layout.overhead_position_map(OUT), layout.surviving_pairs(OUT),
                 layout.group_dir(OUT, 3), layout.joint_scene(OUT),
                 layout.joint_camera(OUT), layout.joint_report(OUT)]
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


class TestCollectImages(unittest.TestCase):
    def test_sorted_images_only(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('b.JPG', 'a.tif', 'notes.txt'):
                (Path(d) / name).touch()
            (Path(d) / 'sub').mkdir()
            got = [p.name for p in inputs.collect_images(Path(d))]
        self.assertEqual(got, ['a.tif', 'b.JPG'])

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

    def test_fingerprint_sees_a_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'f'
            p.write_text('x')
            before = manifest.fingerprint(p)
            p.write_text('xy')
            self.assertNotEqual(before, manifest.fingerprint(p))


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


if __name__ == '__main__':
    unittest.main()
