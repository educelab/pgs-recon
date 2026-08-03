"""Execution tests for ``StageTracker``: the records it writes, not the plan.

``test_stages`` covers planning, which is pure logic over two dicts. This covers
the other half -- ``begin()``, ``end()``, ``abort()``, and the manifest they
write -- so it needs a temporary directory. Still stdlib only, and still runs no
binary: every artifact here is a path that nothing creates.

``TestRoundTrip`` is the case that matters most. It drives a whole pipeline
through the real begin/end machinery, reloads the manifest that produced, and
plans against it -- which is what pins ``test_stages.build_records()``, a
hand-written model of these records, to what the code actually writes.
"""
import json
import tempfile
import unittest
from pathlib import Path

from pgs_recon import layout
from pgs_recon.stages import (StageError, StageTracker, drifted_stages,
                              find_manifest, load_manifest, pipeline_shape,
                              resolve_range)

from test_stages import build_records, make_args, quiet_logger


def fake_run(tracker, args) -> None:
    """Drive every stage in ``tracker``'s range through ``begin()``/``end()``.

    Mirrors ``run_pipeline``'s role wiring and its ``layout`` calls exactly,
    minus the binary each stage would launch. ``test_pipeline`` is what checks
    that the real thing wires the same roles; this exercises the records.
    """
    root = tracker.root

    if tracker.begin('import'):
        outputs = {'sfm': layout.imported_sfm(root)}
        if args.import_pgs_scan:
            outputs['view_pairs'] = layout.view_pairs(root)
        tracker.end('import', inputs={'images': Path(args.input)},
                    outputs=outputs)

    if tracker.begin('features'):
        sfm_in = tracker.require('sfm')
        tracker.end('features', inputs={'sfm': sfm_in},
                    outputs={'features': layout.matches_dir(root)})

    if tracker.begin('matches'):
        sfm_in, features_in = tracker.require('sfm'), tracker.require('features')
        pairs = tracker.path('view_pairs') if args.import_pgs_scan else None
        tracker.end('matches', inputs={'sfm': sfm_in, 'features': features_in,
                                       'view_pairs': pairs},
                    outputs={'matches': layout.matches(root)})

    if tracker.begin('filter'):
        sfm_in, matches_in = tracker.require('sfm'), tracker.require('matches')
        tracker.end('filter', inputs={'sfm': sfm_in, 'matches': matches_in},
                    outputs={'matches_filtered':
                             layout.matches_filtered(matches_in)})

    if tracker.begin('sfm'):
        sfm_in, features_in = tracker.require('sfm'), tracker.require('features')
        inputs = {'sfm': sfm_in, 'features': features_in}
        if args.mvg_recon_method == 'direct':
            inputs['matches'] = tracker.require('matches')
        else:
            inputs['matches_filtered'] = tracker.require('matches_filtered')
        # Either engine writes the solve to the same place.
        tracker.end('sfm', inputs=inputs,
                    outputs={'sfm': layout.solved_sfm(root)})

    if tracker.begin('robust'):
        sfm_in = tracker.require('sfm')
        features_in, matches_in = (tracker.require('features'),
                                   tracker.require('matches'))
        tracker.end('robust',
                    inputs={'sfm': sfm_in, 'features': features_in,
                            'matches': matches_in},
                    outputs={'sfm': layout.robust_sfm(root)})

    if tracker.begin('autoscale'):
        sfm_in = tracker.require('sfm')
        tracker.end('autoscale', inputs={'sfm': sfm_in},
                    outputs={'sfm': layout.autoscale_sfm(root)})

    if tracker.begin('colorize'):
        sfm_in = tracker.require('sfm')
        tracker.end('colorize', inputs={'sfm': sfm_in},
                    outputs={'colorized': layout.colorize_sfm(root)})

    if tracker.begin('convert'):
        sfm_in = tracker.require('sfm')
        tracker.end('convert', inputs={'sfm': sfm_in},
                    outputs={'scene': layout.convert_scene(root)})

    if tracker.begin('densify'):
        scene_in = tracker.require('scene')
        tracker.end('densify', inputs={'scene': scene_in},
                    outputs={'scene': layout.densify_scene(root),
                             'cloud': layout.densify_cloud(root)})

    if tracker.begin('reconstruct'):
        scene_in, cloud = tracker.require('scene'), tracker.path('cloud')
        tracker.end('reconstruct', inputs={'scene': scene_in, 'cloud': cloud},
                    outputs={'mesh': layout.reconstruct_mesh(root)})

    if tracker.begin('refine'):
        scene_in, mesh_in = tracker.require('scene'), tracker.require('mesh')
        tracker.end('refine', inputs={'scene': scene_in, 'mesh': mesh_in},
                    outputs={'mesh': layout.refine_mesh(root)})

    if tracker.begin('texture'):
        scene_in, mesh_in = tracker.require('scene'), tracker.require('mesh')
        tracker.end('texture', inputs={'scene': scene_in, 'mesh': mesh_in},
                    outputs={'mesh': layout.final_mesh(root, args.name,
                                                       args.file_type)})


def modelled_part(records: dict) -> dict:
    """The fields of a stage record that ``build_records()`` claims to model.

    Timings, host, pid and rusage are not modelled. Neither is import's image
    directory, which sits outside the output dir and so is recorded absolute.
    """
    return {
        stage: {'status': rec['status'],
                'inputs': {r: p for r, p in rec['inputs'].items()
                           if r != 'images'},
                'outputs': rec['outputs'],
                'args': rec['args']}
        for stage, rec in records.items()
    }


class TrackerCase(unittest.TestCase):
    """A real output directory, and a tracker built from whatever is in it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Resolved: rel_to() resolves before making paths relative, and macOS
        # hands out /var/... symlinks to /private/var/....
        top = Path(tmp.name).resolve()
        self.root = top / 'recon'
        self.root.mkdir()
        self.images = top / 'images'
        self.manifest = layout.manifest(self.root)
        self.sfm = layout.imported_sfm(self.root)
        self.scene = layout.convert_scene(self.root)

    def tracker(self, args, explicit=(), manifest=None,
                read_from=None) -> StageTracker:
        """A tracker over what is on disk now, as a fresh process would build it.

        Each call builds its own, from the manifest alone, so one simulated job
        cannot see another's live bindings.

        ``manifest`` is the file written, ``read_from`` the file read. They are
        the same unless a caller is modelling ``_main()``'s pre-1.8 fallback,
        where the records come from ``metadata.json`` and the writes land on
        ``pgs-recon.json`` -- which is what moves such a directory onto the new
        name.
        """
        path = manifest or self.manifest
        meta = load_manifest(read_from or path)
        shape = pipeline_shape(args)
        from_stage, to_stage = resolve_range(args, shape)
        drift = drifted_stages(args, meta.get('stages') or {}, set(explicit),
                              shape)
        return StageTracker(meta, path, self.root, args, shape,
                            from_stage, to_stage, rerun=args.rerun, drift=drift,
                            logger=quiet_logger())

    def records(self) -> dict:
        return load_manifest(self.manifest).get('stages') or {}

    def runs(self, tracker):
        return [s for s, st in tracker.plan() if st == 'run']


class TestRecords(TrackerCase):
    def test_begin_writes_a_running_record(self):
        tracker = self.tracker(make_args())
        self.assertTrue(tracker.begin('import'))
        record = self.records()['import']
        self.assertEqual('running', record['status'])
        self.assertIn('pid', record)
        self.assertIn('host', record)
        self.assertIn('started', record)
        self.assertEqual('/images', record['args']['input'])

    def test_end_writes_a_complete_record(self):
        args = make_args()
        tracker = self.tracker(args)
        tracker.begin('import')
        tracker.end('import', inputs={'images': self.images},
                    outputs={'sfm': self.sfm})
        record = self.records()['import']
        self.assertEqual('complete', record['status'])
        self.assertIn('finished', record)
        self.assertIsInstance(record['elapsed_s'], float)
        self.assertEqual({'sfm': 'mvg/sfm_data.json'}, record['outputs'])

    def test_recorded_paths_are_relative_to_the_output_dir(self):
        tracker = self.tracker(make_args())
        tracker.begin('convert')
        tracker.end('convert', inputs={'sfm': self.sfm},
                    outputs={'scene': self.scene})
        record = self.records()['convert']
        self.assertEqual({'sfm': 'mvg/sfm_data.json'}, record['inputs'])
        self.assertEqual({'scene': 'mvs/convert_scene.mvs'}, record['outputs'])

    def test_an_artifact_outside_the_output_dir_is_recorded_absolute(self):
        # A relative -i would otherwise be recorded bare and re-rooted under the
        # output dir on read.
        tracker = self.tracker(make_args())
        tracker.begin('import')
        tracker.end('import', inputs={'images': Path('images')},
                    outputs={'sfm': self.sfm})
        recorded = self.records()['import']['inputs']['images']
        self.assertTrue(Path(recorded).is_absolute(), recorded)
        self.assertEqual(str(Path('images').resolve()), recorded)

    def test_none_valued_roles_are_not_recorded(self):
        tracker = self.tracker(make_args())
        tracker.begin('reconstruct')
        tracker.end('reconstruct',
                    inputs={'scene': self.scene,
                            'cloud': None},
                    outputs={'mesh': self.root / 'mvs' / 'reconstruct_mesh.ply'})
        self.assertEqual(['scene'], list(self.records()['reconstruct']['inputs']))

    def test_commands_are_attributed_to_the_stage_that_ran_them(self):
        tracker = self.tracker(make_args())
        tracker.meta.setdefault('commands', {})['t0'] = 'before any stage'
        tracker.begin('convert')
        tracker.meta['commands']['t1'] = 'openMVG2openMVS -i ...'
        tracker.end('convert', outputs={'scene': self.scene})
        self.assertEqual(['openMVG2openMVS -i ...'],
                         self.records()['convert']['commands'])

    def test_rerunning_a_stage_replaces_its_previous_record(self):
        # What retexture relies on when it reports a stage's status instead of
        # guessing why the artifact is missing.
        args = make_args()
        fake_run(self.tracker(args), args)
        self.assertIn('outputs', self.records()['texture'])
        tracker = self.tracker(make_args(rerun=True, from_stage='texture'))
        self.assertTrue(tracker.begin('texture'))
        record = self.records()['texture']
        self.assertEqual('running', record['status'])
        self.assertNotIn('outputs', record)


class TestChain(TrackerCase):
    def test_outputs_rebind_the_chain_for_later_stages(self):
        tracker = self.tracker(make_args())
        tracker.begin('import')
        tracker.end('import', outputs={'sfm': self.sfm})
        self.assertEqual(self.sfm, tracker.path('sfm'))
        tracker.begin('features')
        tracker.end('features', outputs={'features': self.root / 'mvg/md'})
        solved = layout.solved_sfm(self.root)
        tracker.begin('sfm')
        tracker.end('sfm', outputs={'sfm': solved})
        self.assertEqual(solved, tracker.path('sfm'))

    def test_a_skipped_stage_absorbs_its_recorded_outputs(self):
        args = make_args()
        fake_run(self.tracker(args), args)
        # Nothing to do, but convert's recorded scene still has to reach the
        # stages after it.
        tracker = self.tracker(args)
        self.assertFalse(tracker.begin('convert'))
        self.assertEqual(self.root / 'mvs/convert_scene.mvs', tracker.path('scene'))

    def test_a_later_job_requires_the_mesh_the_last_one_recorded(self):
        args = make_args()
        fake_run(self.tracker(args), args)
        tracker = self.tracker(make_args(from_stage='texture', rerun=True))
        self.assertTrue(tracker.begin('texture'))
        # The mesh role is bound to what refine wrote, not to what
        # reconstruct did.
        self.assertEqual(self.root / 'mvs/refine_mesh.ply',
                         tracker.require('mesh'))


class TestAbort(TrackerCase):
    def test_abort_marks_the_active_stage_failed(self):
        tracker = self.tracker(make_args())
        tracker.begin('import')
        tracker.end('import', outputs={'sfm': self.sfm})
        tracker.begin('features')
        tracker.abort()
        records = self.records()
        self.assertEqual('failed', records['features']['status'])
        self.assertEqual('complete', records['import']['status'])

    def test_abort_without_an_active_stage_writes_nothing(self):
        tracker = self.tracker(make_args())
        tracker.abort()
        self.assertFalse(self.manifest.exists())

    def test_a_failed_stage_is_re_run_by_the_next_job(self):
        args = make_args()
        tracker = self.tracker(args)
        tracker.begin('import')
        tracker.abort()
        self.assertEqual('import', self.tracker(args).first_to_run())


class TestManifestFailures(TrackerCase):
    """A stage transition that cannot be recorded has to stop the run."""

    def broken(self) -> Path:
        return self.root / 'gone' / layout.manifest(self.root).name

    def test_begin_refuses_to_start_a_stage_it_cannot_record(self):
        tracker = self.tracker(make_args(), manifest=self.broken())
        with self.assertRaises(StageError) as ctx:
            tracker.begin('import')
        self.assertIn('Cannot write the manifest', str(ctx.exception))

    def test_end_refuses_to_lose_a_completion(self):
        tracker = self.tracker(make_args())
        tracker.begin('import')
        tracker.manifest_path = self.broken()  # the directory goes away
        with self.assertRaises(StageError):
            tracker.end('import', outputs={'sfm': self.sfm})

    def test_abort_tolerates_a_manifest_it_cannot_write(self):
        # It runs while an exception is propagating; raising here would replace
        # the failure being reported.
        tracker = self.tracker(make_args())
        tracker.begin('import')
        tracker.manifest_path = self.broken()
        with self.assertLogs('pgs_recon.stages', 'ERROR'):
            tracker.abort()

    def test_a_partial_write_is_never_left_behind(self):
        tracker = self.tracker(make_args())
        tracker.begin('import')
        self.assertEqual([], list(self.root.glob('pgs-recon.json.tmp*')))


class TestRoundTrip(TrackerCase):
    """Records written by end() must plan as complete, and match the model."""

    maxDiff = None  # the whole-manifest comparison below is unreadable truncated

    def assert_round_trips(self, args):
        """Run the whole shape, then plan against the records it wrote."""
        fake_run(self.tracker(args), args)
        written = self.records()
        self.assertEqual(list(pipeline_shape(args)), list(written))
        self.assertEqual(modelled_part(build_records(args)),
                         modelled_part(written),
                         'build_records() no longer models what end() writes')
        replan = self.tracker(args)
        self.assertEqual([], self.runs(replan))
        self.assertIsNone(replan.first_to_run())
        self.assertEqual([], replan.prereq_errors())

    def test_default_pipeline(self):
        self.assert_round_trips(make_args())

    def test_every_optional_stage_enabled(self):
        self.assert_round_trips(make_args(mvg_robust=True, mvg_autoscale=0.47,
                                         mvs_densify=True, import_pgs_scan=True))

    def test_direct_engine(self):
        self.assert_round_trips(make_args(mvg_recon_method='direct'))


class TestPre18ManifestName(TrackerCase):
    """1.8 renamed the manifest ``metadata.json`` -> ``pgs-recon.json``.

    Unlike an artifact name, this one is *located* by name, so a directory
    written by an earlier version would look empty and re-run the whole
    pipeline. ``find_manifest`` is the fallback that stops that.
    """

    def legacy(self) -> Path:
        return layout.legacy_manifest(self.root)

    def test_a_fresh_directory_resolves_to_the_current_name(self):
        self.assertEqual(self.manifest, find_manifest(self.root))

    def test_an_old_directory_resolves_to_the_name_it_has(self):
        args = make_args()
        fake_run(self.tracker(args, manifest=self.legacy()), args)
        self.assertTrue(self.legacy().is_file())
        self.assertFalse(self.manifest.exists())
        self.assertEqual(self.legacy(), find_manifest(self.root))

    def test_the_current_name_wins_when_both_are_present(self):
        # What a directory looks like after one run under 1.8: the old file is
        # left in place and must never be read again.
        args = make_args()
        fake_run(self.tracker(args, manifest=self.legacy()), args)
        self.manifest.write_text('{}')
        self.assertEqual(self.manifest, find_manifest(self.root))

    def test_an_old_directory_resumes_with_nothing_re_run(self):
        args = make_args()
        fake_run(self.tracker(args, manifest=self.legacy()), args)
        # The next job reads through find_manifest, as _main() does...
        resumed = self.tracker(args, manifest=find_manifest(self.root))
        self.assertEqual([], self.runs(resumed))
        self.assertEqual([], resumed.prereq_errors())

    def test_a_1_7_manifest_records_nothing_to_resume_from(self):
        """The older case, and the one that is *not* free.

        1.7 predates ADR 0004's stage records: four keys, no ``stages`` and no
        ``effective_args``. Finding the file is all the fallback can do -- there
        is nothing in it to resume from, so every stage runs and the arguments
        have to be given again. Only a 1.8-alpha directory (records, old
        filename) resumes untouched.
        """
        self.legacy().write_text(json.dumps({
            'args': 'pgs-recon -i /images -o recon --name obj',
            'parsed': {'name': 'obj', 'file_type': 'obj'},
            'paths': {'output': str(self.root)},
            'commands': {'08/01/2026, 00:00:00.000000 UTC':
                         'openMVG_main_SfM -i sfm_data.json'},
        }))
        found = find_manifest(self.root)
        self.assertEqual(self.legacy(), found)

        meta = load_manifest(found)
        self.assertEqual({}, meta.get('effective_args', {}))
        args = make_args()
        tracker = self.tracker(args, manifest=found)
        self.assertEqual(list(pipeline_shape(args)), self.runs(tracker))
        self.assertEqual('never run', tracker.dirty['import'])

    def test_the_records_a_resumed_old_directory_writes_land_on_the_new_name(self):
        args = make_args()
        fake_run(self.tracker(args, manifest=self.legacy()), args)
        legacy_before = self.legacy().read_text()

        # ...and writes to layout.manifest, which is what moves the directory.
        rerun = make_args(from_stage='texture', rerun=True)
        fake_run(self.tracker(rerun, read_from=find_manifest(self.root)), rerun)

        self.assertEqual('complete', self.records()['texture']['status'])
        self.assertEqual(legacy_before, self.legacy().read_text(),
                         'the pre-1.8 manifest must be left untouched')


class TestStagedJobs(TrackerCase):
    def test_a_later_job_resumes_from_the_records_the_first_wrote(self):
        args = make_args(mvs_densify=True)
        fake_run(self.tracker(make_args(mvs_densify=True, to_stage='convert')),
                 args)
        self.assertEqual(['import', 'features', 'matches', 'filter', 'sfm',
                          'colorize', 'convert'], list(self.records()))

        job2 = self.tracker(args)
        self.assertEqual(['densify', 'reconstruct', 'refine', 'texture'],
                         self.runs(job2))
        self.assertEqual('skip', job2.status_of('convert'))
        fake_run(job2, args)

        # densify consumed the scene convert recorded in the previous job, which
        # it can only have reached by absorbing the skipped stage's outputs.
        self.assertEqual('mvs/convert_scene.mvs',
                         self.records()['densify']['inputs']['scene'])
        self.assertEqual([], self.runs(self.tracker(args)))
        self.assertEqual('mvs/obj.obj',
                         self.records()['texture']['outputs']['mesh'])

    def test_an_sfm_only_run_records_only_its_range(self):
        args = make_args(to_stage='colorize')
        fake_run(self.tracker(args), args)
        self.assertEqual(['import', 'features', 'matches', 'filter', 'sfm',
                          'colorize'], list(self.records()))
        # The MVS tail is unfinished, which is what the range said -- not stale.
        again = self.tracker(args)
        self.assertEqual([], self.runs(again))
        self.assertEqual([], again.stale_after_range())

    def test_a_job_before_its_prerequisites_refuses_to_run(self):
        tracker = self.tracker(make_args(from_stage='refine'))
        self.assertEqual(['import', 'features', 'matches', 'filter', 'sfm',
                          'colorize', 'convert', 'reconstruct'],
                         [e.split(':')[0] for e in tracker.prereq_errors()])
        self.assertFalse(self.manifest.exists())

    def test_view_pairs_rehydrate_into_a_later_job(self):
        args = make_args(import_pgs_scan=True)
        fake_run(self.tracker(make_args(import_pgs_scan=True,
                                        to_stage='import')), args)
        job2 = self.tracker(make_args(import_pgs_scan=True,
                                      from_stage='features'))
        self.assertEqual(layout.view_pairs(self.root),
                         job2.path('view_pairs'))

    def test_dropping_densify_rebuilds_the_mesh_chain_against_real_records(self):
        dense = make_args(mvs_densify=True)
        fake_run(self.tracker(dense), dense)
        self.assertEqual('mvs/densify.mvs',
                         self.records()['reconstruct']['inputs']['scene'])

        plain = make_args(mvs_densify=False)
        tracker = self.tracker(plain, explicit={'mvs_densify'})
        self.assertEqual(['reconstruct', 'refine', 'texture'],
                         self.runs(tracker))
        self.assertEqual('inputs changed: scene', tracker.dirty['reconstruct'])
        fake_run(tracker, plain)
        # The mesh keeps its name across the shape change (ADR 0006) and is
        # rebuilt in place: what moved is reconstruct's *input*, back to the
        # scene convert wrote, which is what clause 4 saw.
        self.assertEqual('mvs/convert_scene.mvs',
                         self.records()['reconstruct']['inputs']['scene'])
        self.assertEqual('mvs/reconstruct_mesh.ply',
                         self.records()['reconstruct']['outputs']['mesh'])


if __name__ == '__main__':
    unittest.main()
