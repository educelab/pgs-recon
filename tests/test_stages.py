"""Planner tests for staged, resumable runs.

``pgs_recon.stages`` imports only the standard library and never touches the
filesystem, so this suite needs no binaries, no fixtures and no third-party
packages:

    python3 -m unittest discover -s tests

Everything here builds an argument ``Namespace`` and a dict of fake stage
records, then asserts on the resolved plan. ``build_records()`` is a small model
of ``run_pipeline``'s bookkeeping -- it reproduces the artifact *names* the
OpenMVG/OpenMVS wrappers derive, which is what the graph's rebinding rule
compares.
"""
import logging
import unittest
from argparse import Namespace
from pathlib import Path

from pgs_recon.stages import (STAGE_ARGS, STAGE_IO, STAGES, StageError,
                              StageTracker, drifted_stages, pipeline_shape,
                              resolve_range, revert_out_of_range)

ROOT = Path('/recon')

# Parser defaults, so a test only states what it is actually varying.
DEFAULTS = dict(
    # import
    input='/images', focal_length=None, new_importer=False,
    import_pgs_scan=False, import_calib=None, matching_pairs_radius=2,
    # features / matches / filter
    describer_method='SIFT', describer_preset='HIGH', describer_upright=False,
    matching_method='FASTCASCADEHASHINGL2', matching_ratio=None,
    matching_pairs_file='auto', matching_geometric_model=None,
    # sfm / robust / autoscale
    mvg_recon_method='global', mvg_priors=False, mvg_refine_intrinsics=None,
    mvg_initializer=None, sfm_ba=False, mvg_robust=False, robust_ba=False,
    mvg_autoscale=None, autoscale_method='markers', autoscale_marker_pix=None,
    autoscale_include_from=None, autoscale_exclude_from=None,
    # mvs
    mvs_densify=False, densify_resolution_level=None, mask_value=0,
    free_space_support=False, mvs_smooth=2,
    mvs_refine=True, decimation_factor=None, refine_resolution_level=None,
    refine_min_resolution=None, refine_scales=3, refine_scale_step=None,
    name='obj', file_type='obj', texture_resolution_level=None,
    texture_max_size=0,
    # flow control
    from_stage=None, to_stage=None, rerun=False, dry_run=False,
)


def make_args(**overrides) -> Namespace:
    values = dict(DEFAULTS)
    values.update(overrides)
    return Namespace(**values)


def build_records(args, shape=None, status='complete') -> dict:
    """A manifest for a run of ``shape`` that finished successfully.

    The paths mirror what the wrappers actually write, because that is what the
    planner compares: every output name is derived from its input's stem, so
    enabling densify renames the whole mesh chain.
    """
    shape = shape or pipeline_shape(args)
    records = {}

    def rec(stage, inputs, outputs):
        records[stage] = {
            'status': status,
            'inputs': {r: p for r, p in inputs.items() if p is not None},
            'outputs': {r: p for r, p in outputs.items() if p is not None},
            'args': {d: getattr(args, d, None) for d in STAGE_ARGS[stage]},
        }

    sfm = 'mvg/sfm_data.json'
    features = 'mvg/matches_dir'
    matches = 'mvg/matches_dir/matches.bin'
    filtered = 'mvg/matches_dir/matches_filtered.bin'
    pairs = 'mvg/view_pairs.txt' if args.import_pgs_scan else None

    rec('import', {'images': 'input'}, {'sfm': sfm, 'view_pairs': pairs})
    rec('features', {'sfm': sfm}, {'features': features})
    rec('matches', {'sfm': sfm, 'features': features, 'view_pairs': pairs},
        {'matches': matches})
    rec('filter', {'sfm': sfm, 'matches': matches},
        {'matches_filtered': filtered})

    sfm_inputs = {'sfm': sfm, 'features': features}
    if args.mvg_recon_method == 'direct':
        # Triangulating known poses reads the unfiltered matches, and writes
        # beside its input rather than to the engine's fixed name.
        sfm_inputs['matches'] = matches
        solved = f'mvg/recon_dir/{Path(sfm).stem}_structured.bin'
    else:
        sfm_inputs['matches_filtered'] = filtered
        solved = 'mvg/recon_dir/sfm_data.bin'
    rec('sfm', sfm_inputs, {'sfm': solved})
    if 'robust' in shape:
        nxt = f'mvg/recon_dir/{Path(solved).stem}_structured.bin'
        rec('robust', {'sfm': solved, 'features': features, 'matches': matches},
            {'sfm': nxt})
        solved = nxt
    if 'autoscale' in shape:
        nxt = f'mvg/recon_dir/{Path(solved).stem}_scaled.bin'
        rec('autoscale', {'sfm': solved}, {'sfm': nxt})
        solved = nxt
    rec('colorize', {'sfm': solved},
        {'colorized': f'mvg/recon_dir/{Path(solved).stem}_colorized.ply'})

    scene = 'mvs/scene.mvs'
    rec('convert', {'sfm': solved}, {'scene': scene})
    cloud = None
    if 'densify' in shape:
        cloud = 'mvs/scene_dense.ply'
        dense = 'mvs/scene_dense.mvs'
        rec('densify', {'scene': scene}, {'scene': dense, 'cloud': cloud})
        scene = dense
    mesh = f'mvs/{Path(scene).stem}_mesh.ply'
    # reconstruct and refine hand the scene back untouched, so neither records
    # it as an output -- see STAGE_IO on pass-through roles.
    rec('reconstruct', {'scene': scene, 'cloud': cloud}, {'mesh': mesh})
    if 'refine' in shape:
        refined = f'mvs/{Path(scene).stem}_refine.ply'
        rec('refine', {'scene': scene, 'mesh': mesh}, {'mesh': refined})
        mesh = refined
    rec('texture', {'scene': scene, 'mesh': mesh},
        {'mesh': f'mvs/{args.name}.{args.file_type}'})
    return {s: records[s] for s in shape}


def quiet_logger():
    log = logging.getLogger('test-stages')
    log.handlers = [logging.NullHandler()]
    log.propagate = False
    log.setLevel(logging.CRITICAL)
    return log


class Captured(logging.Handler):
    """Collects records so a test can assert on what the user was told."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append((record.levelno, record.getMessage()))

    def text(self, level=logging.WARNING):
        return '\n'.join(m for lvl, m in self.messages if lvl >= level)


def capturing_logger():
    log = logging.getLogger('test-stages-capture')
    handler = Captured()
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.DEBUG)
    return log, handler


def make_tracker(args, records=None, explicit=(), shape=None, logger=None):
    meta = {'stages': dict(records or {})}
    paths = {'output': ROOT, 'mvs': ROOT / 'mvs'}
    shape = shape or pipeline_shape(args)
    from_stage, to_stage = resolve_range(args, shape)
    drift = drifted_stages(args, meta['stages'], set(explicit), shape)
    return StageTracker(meta, ROOT / 'metadata.json', paths, args, shape,
                        from_stage, to_stage, rerun=args.rerun, drift=drift,
                        logger=logger or quiet_logger())


def runs(tracker):
    return [s for s, st in tracker.plan() if st == 'run']


class TestTables(unittest.TestCase):
    """The two hand-maintained tables have to agree with each other."""

    def test_every_stage_declares_io_and_args(self):
        self.assertEqual(set(STAGES), set(STAGE_IO))
        self.assertEqual(set(STAGES), set(STAGE_ARGS))

    def test_every_consumed_role_is_produced_by_some_stage(self):
        produced = {r for io in STAGE_IO.values() for r in io.makes}
        for stage, io in STAGE_IO.items():
            for role in io.needs + io.may:
                self.assertIn(role, produced, f'{stage} consumes {role}')

    def test_sfm_declares_both_matches_roles(self):
        # The engine decides which one it reads, so the required role is the one
        # every engine has (filter is unconditionally part of the shape) and the
        # direct engine's unfiltered matches are optional.
        self.assertIn('matches_filtered', STAGE_IO['sfm'].needs)
        self.assertIn('matches', STAGE_IO['sfm'].may)

    def test_pass_through_roles_are_not_declared_as_produced(self):
        # mvs_reconstruct/mvs_refine return their input scene unchanged;
        # declaring it produced would create a phantom edge.
        self.assertNotIn('scene', STAGE_IO['reconstruct'].makes)
        self.assertNotIn('scene', STAGE_IO['refine'].makes)


class TestRange(unittest.TestCase):
    def test_from_a_disabled_stage_errors(self):
        args = make_args(mvs_densify=False, from_stage='densify')
        with self.assertRaises(StageError) as ctx:
            resolve_range(args, pipeline_shape(args))
        self.assertIn('not part of this pipeline', str(ctx.exception))

    def test_to_a_disabled_stage_errors(self):
        args = make_args(mvs_refine=False, to_stage='refine')
        with self.assertRaises(StageError):
            resolve_range(args, pipeline_shape(args))

    def test_inverted_range_errors(self):
        args = make_args(from_stage='texture', to_stage='refine')
        with self.assertRaises(StageError) as ctx:
            resolve_range(args, pipeline_shape(args))
        self.assertIn('comes after', str(ctx.exception))

    def test_defaults_span_the_shape(self):
        args = make_args(mvs_densify=True)
        self.assertEqual(('import', 'texture'),
                         resolve_range(args, pipeline_shape(args)))


class TestOverrides(unittest.TestCase):
    def test_out_of_range_override_warns_and_reverts(self):
        args = make_args(refine_scales=9, from_stage='import',
                         to_stage='convert')
        stored = dict(DEFAULTS)
        log, handler = capturing_logger()
        revert_out_of_range(args, stored, {'refine_scales'}, 'import',
                            'convert', log)
        self.assertEqual(3, args.refine_scales)
        self.assertIn('--refine-scales', handler.text())
        self.assertIn('outside the range', handler.text())

    def test_in_range_override_is_left_alone(self):
        args = make_args(refine_scales=9, from_stage='refine')
        stored = dict(DEFAULTS)
        log, handler = capturing_logger()
        revert_out_of_range(args, stored, {'refine_scales'}, 'refine',
                            'texture', log)
        self.assertEqual(9, args.refine_scales)
        self.assertEqual('', handler.text())

    def test_global_override_is_silent(self):
        args = make_args(from_stage='refine')
        args.path = '/opt/other'
        stored = dict(DEFAULTS, path='/usr/local/')
        log, handler = capturing_logger()
        revert_out_of_range(args, stored, {'path'}, 'refine', 'texture', log)
        self.assertEqual('/opt/other', args.path)
        self.assertEqual('', handler.text())

    def test_first_run_keeps_out_of_range_overrides(self):
        # Nothing is stored yet, so a --to convert job may still carry the
        # --refine-* arguments a later job will need. submit_recon_pipeline.sh
        # depends on this.
        args = make_args(refine_scales=9, to_stage='convert')
        log, handler = capturing_logger()
        revert_out_of_range(args, {}, {'refine_scales'}, 'import', 'convert',
                            log)
        self.assertEqual(9, args.refine_scales)
        self.assertEqual('', handler.text())


class TestPlanning(unittest.TestCase):
    def test_fresh_directory_runs_everything(self):
        args = make_args()
        tracker = make_tracker(args)
        self.assertEqual(list(pipeline_shape(args)), runs(tracker))
        self.assertEqual('never run', tracker.dirty['import'])

    def test_complete_run_is_a_noop(self):
        args = make_args()
        tracker = make_tracker(args, build_records(args))
        self.assertEqual([], runs(tracker))
        self.assertIsNone(tracker.first_to_run())
        self.assertEqual([], tracker.prereq_errors())

    def test_disabled_stages_are_off(self):
        args = make_args(mvs_densify=False, mvg_robust=False)
        tracker = make_tracker(args, build_records(args))
        self.assertEqual('off', tracker.status_of('densify'))
        self.assertEqual('off', tracker.status_of('robust'))
        self.assertEqual('off', tracker.status_of('autoscale'))

    def test_in_range_drift_cascades_to_texture(self):
        args = make_args()
        records = build_records(args)
        args.refine_scales = 5
        tracker = make_tracker(args, records, explicit={'refine_scales'})
        self.assertEqual(['refine', 'texture'], runs(tracker))
        self.assertIn('--refine-scales', tracker.dirty['refine'])
        self.assertIn('inputs rebuilt by refine', tracker.dirty['texture'])

    def test_cascade_stops_at_to(self):
        args = make_args(from_stage='refine', to_stage='refine',
                         refine_scales=5)
        records = build_records(make_args())
        tracker = make_tracker(args, records, explicit={'refine_scales'})
        self.assertEqual(['refine'], runs(tracker))
        self.assertEqual('after', tracker.status_of('texture'))
        self.assertEqual(['texture'], tracker.stale_after_range())

    def test_downstream_staleness_warns_and_does_not_error(self):
        log, handler = capturing_logger()
        args = make_args(from_stage='refine', to_stage='refine',
                         refine_scales=5)
        tracker = make_tracker(args, build_records(make_args()),
                               explicit={'refine_scales'}, logger=log)
        tracker.log_plan()
        self.assertEqual([], tracker.prereq_errors())
        self.assertIn('texture', handler.text())
        self.assertIn('left stale', handler.text())

    def test_unfinished_stages_past_to_are_not_reported_stale(self):
        # submit_recon_pipeline.sh job 1 is exactly this: a fresh directory with
        # --to convert. The MVS tail has not run, which is what the range says,
        # not something this run invalidated.
        log, handler = capturing_logger()
        args = make_args(mvs_densify=True, to_stage='convert')
        tracker = make_tracker(args, logger=log)
        tracker.log_plan()
        self.assertEqual([], tracker.stale_after_range())
        self.assertEqual('', handler.text())

    def test_rerun_forces_the_range_only(self):
        args = make_args(from_stage='reconstruct', rerun=True)
        tracker = make_tracker(args, build_records(args))
        self.assertEqual(['reconstruct', 'refine', 'texture'], runs(tracker))
        self.assertEqual('--rerun', tracker.dirty['reconstruct'])
        self.assertEqual('done', tracker.status_of('convert'))

    def test_missing_prereqs_error_naming_each(self):
        args = make_args(from_stage='refine')
        records = build_records(args)
        records['reconstruct']['status'] = 'failed'
        del records['convert']
        tracker = make_tracker(args, records)
        errors = tracker.prereq_errors()
        self.assertEqual(2, len(errors))
        self.assertTrue(any(e.startswith('convert: never run') for e in errors))
        self.assertTrue(any("reconstruct: recorded as 'failed'" in e
                            for e in errors))
        self.assertEqual('blocked', tracker.status_of('convert'))

    def test_failed_stage_in_range_reruns(self):
        args = make_args()
        records = build_records(args)
        records['refine']['status'] = 'failed'
        tracker = make_tracker(args, records)
        self.assertEqual(['refine', 'texture'], runs(tracker))

    def test_producer_rebuild_cascades_when_the_path_is_unchanged(self):
        # sfm rewrites recon_dir/sfm_data.bin at the same path every time, so
        # only the producer's dirtiness reveals that robust is stale.
        args = make_args(mvg_robust=True)
        records = build_records(args)
        args.mvg_recon_method = 'stellar'
        tracker = make_tracker(args, records, explicit={'mvg_recon_method'})
        self.assertEqual(['sfm', 'robust', 'colorize', 'convert', 'reconstruct',
                          'refine', 'texture'], runs(tracker))
        self.assertEqual(records['robust']['inputs']['sfm'],
                         records['sfm']['outputs']['sfm'])
        self.assertIn('inputs rebuilt by sfm', tracker.dirty['robust'])

    def test_the_direct_engine_consumes_the_unfiltered_matches(self):
        args = make_args(mvg_recon_method='direct')
        records = build_records(args)
        self.assertIn('matches', records['sfm']['inputs'])
        self.assertNotIn('matches_filtered', records['sfm']['inputs'])
        records['matches']['status'] = 'failed'
        tracker = make_tracker(args, records)
        self.assertIn('sfm', runs(tracker))

    def test_key_refuses_an_artifact_that_does_not_exist_yet(self):
        tracker = make_tracker(make_args())
        with self.assertRaises(StageError):
            tracker.key('sfm')

    def test_rehydrated_inputs_come_from_the_records(self):
        args = make_args(from_stage='refine')
        tracker = make_tracker(args, build_records(args))
        self.assertEqual(ROOT / 'mvs/scene.mvs', tracker.path('scene'))
        self.assertEqual(ROOT / 'mvs/scene_mesh.ply', tracker.path('mesh'))
        self.assertFalse(tracker.has('cloud'))

    def test_view_pairs_rehydrate_into_a_later_job(self):
        args = make_args(import_pgs_scan=True, from_stage='matches')
        tracker = make_tracker(args, build_records(args))
        self.assertEqual(ROOT / 'mvg/view_pairs.txt',
                         tracker.path('view_pairs'))

    def test_a_generic_import_binds_no_view_pairs(self):
        args = make_args(from_stage='matches')
        tracker = make_tracker(args, build_records(args))
        self.assertIsNone(tracker.path('view_pairs'))
        self.assertEqual([], runs(tracker))


class TestShapeChanges(unittest.TestCase):
    """The two headline regressions: a shrunk shape and an isolated leaf."""

    def test_removing_densify_rebuilds_the_mesh_chain(self):
        # Finding #1. The dense run's reconstruct recorded scene_dense.mvs;
        # without densify the live binding is convert's scene.mvs, so the mesh
        # stages must re-run against the non-dense names even though nothing is
        # incomplete. (Pre-fix: all thirteen stages skipped.)
        dense = make_args(mvs_densify=True)
        records = build_records(dense)
        self.assertEqual('mvs/scene_dense.mvs',
                         records['reconstruct']['inputs']['scene'])
        args = make_args(mvs_densify=False)
        tracker = make_tracker(args, records, explicit={'mvs_densify'})
        self.assertEqual(['reconstruct', 'refine', 'texture'], runs(tracker))
        self.assertEqual('inputs changed: scene',
                         tracker.dirty['reconstruct'])
        self.assertEqual('skip', tracker.status_of('convert'))
        self.assertEqual('off', tracker.status_of('densify'))
        self.assertEqual([], tracker.prereq_errors())

    def test_adding_densify_rebuilds_the_mesh_chain(self):
        args = make_args(mvs_densify=True)
        records = build_records(make_args(mvs_densify=False))
        tracker = make_tracker(args, records, explicit={'mvs_densify'})
        self.assertEqual(['densify', 'reconstruct', 'refine', 'texture'],
                         runs(tracker))
        self.assertEqual('never run', tracker.dirty['densify'])
        self.assertEqual('skip', tracker.status_of('convert'))

    def test_adding_densify_with_a_short_range_warns(self):
        args = make_args(mvs_densify=True, to_stage='reconstruct')
        records = build_records(make_args(mvs_densify=False))
        log, handler = capturing_logger()
        tracker = make_tracker(args, records, explicit={'mvs_densify'},
                               logger=log)
        tracker.log_plan()
        self.assertEqual(['densify', 'reconstruct'], runs(tracker))
        self.assertEqual(['refine', 'texture'], tracker.stale_after_range())
        self.assertEqual([], tracker.prereq_errors())
        self.assertIn('left stale', handler.text())

    def test_invalidating_colorize_cascades_to_nothing(self):
        # Finding #2. colorize is a leaf: convert consumes the pre-colorize SfM,
        # not the colorized ply. (Pre-fix: convert, reconstruct, refine and
        # texture all re-ran because colorize sat earlier in STAGES.)
        args = make_args()
        records = build_records(args)
        records['colorize']['status'] = 'failed'
        tracker = make_tracker(args, records)
        self.assertEqual(['colorize'], runs(tracker))
        for stage in ('convert', 'reconstruct', 'refine', 'texture'):
            self.assertEqual('skip', tracker.status_of(stage))

    def test_disabling_refine_rebuilds_texture(self):
        args = make_args(mvs_refine=False)
        records = build_records(make_args(mvs_refine=True))
        tracker = make_tracker(args, records, explicit={'mvs_refine'})
        self.assertEqual(['texture'], runs(tracker))
        self.assertEqual('inputs changed: mesh', tracker.dirty['texture'])
        self.assertEqual('off', tracker.status_of('refine'))


if __name__ == '__main__':
    unittest.main()
