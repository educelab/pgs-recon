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
import tempfile
import unittest
from pathlib import Path

from pgs_recon.stages import (StageError, StageTracker, drifted_stages,
                              load_manifest, pipeline_shape, resolve_range)

from test_stages import build_records, make_args, quiet_logger


def fake_run(tracker, args) -> None:
    """Drive every stage in ``tracker``'s range through ``begin()``/``end()``.

    Mirrors ``run_pipeline``'s role wiring exactly, but derives each output name
    the way the OpenMVG/OpenMVS wrappers do -- from its input path's stem --
    instead of launching the binary that would write it.
    """
    paths = tracker.paths

    if tracker.begin('import'):
        outputs = {'sfm': paths['sfm']}
        if args.import_pgs_scan:
            outputs['view_pairs'] = paths['mvg'] / 'view_pairs.txt'
        tracker.end('import', inputs={'images': paths['input']},
                    outputs=outputs)

    if tracker.begin('features'):
        sfm_in = tracker.key('sfm')
        tracker.end('features', inputs={'sfm': paths[sfm_in]},
                    outputs={'features': paths['matches_dir']})

    if tracker.begin('matches'):
        sfm_in, features_in = tracker.key('sfm'), tracker.key('features')
        pairs = tracker.path('view_pairs') if args.import_pgs_scan else None
        tracker.end('matches', inputs={'sfm': paths[sfm_in],
                                       'features': paths[features_in],
                                       'view_pairs': pairs},
                    outputs={'matches': paths['matches_file']})

    if tracker.begin('filter'):
        sfm_in, matches_in = tracker.key('sfm'), tracker.key('matches')
        m = paths[matches_in]
        tracker.end('filter', inputs={'sfm': paths[sfm_in], 'matches': m},
                    outputs={'matches_filtered':
                             m.parent / (m.stem + '_filtered' + m.suffix)})

    if tracker.begin('sfm'):
        in_key, features_in = tracker.key('sfm'), tracker.key('features')
        inputs = {'sfm': paths[in_key], 'features': paths[features_in]}
        if args.mvg_recon_method == 'direct':
            matches_in = tracker.key('matches')
            inputs['matches'] = paths[matches_in]
            out = paths['recon_dir'] / (paths[in_key].stem + '_structured.bin')
        else:
            filtered_in = tracker.key('matches_filtered')
            inputs['matches_filtered'] = paths[filtered_in]
            out = paths['recon_dir'] / 'sfm_data.bin'
        tracker.end('sfm', inputs=inputs, outputs={'sfm': out})

    if tracker.begin('robust'):
        in_key = tracker.key('sfm')
        features_in, matches_in = tracker.key('features'), tracker.key('matches')
        tracker.end('robust',
                    inputs={'sfm': paths[in_key],
                            'features': paths[features_in],
                            'matches': paths[matches_in]},
                    outputs={'sfm': paths['recon_dir']
                             / (paths[in_key].stem + '_structured.bin')})

    if tracker.begin('autoscale'):
        in_key = tracker.key('sfm')
        tracker.end('autoscale', inputs={'sfm': paths[in_key]},
                    outputs={'sfm': paths['recon_dir']
                             / (paths[in_key].stem + '_scaled.bin')})

    if tracker.begin('colorize'):
        p = paths[tracker.key('sfm')]
        tracker.end('colorize', inputs={'sfm': p},
                    outputs={'colorized':
                             p.parent / (p.stem + '_colorized.ply')})

    if tracker.begin('convert'):
        in_key = tracker.key('sfm')
        tracker.end('convert', inputs={'sfm': paths[in_key]},
                    outputs={'scene': paths['mvs_scene']})

    if tracker.begin('densify'):
        p = paths[tracker.key('scene')]
        dense = p.parent / (p.stem + '_dense.mvs')
        tracker.end('densify', inputs={'scene': p},
                    outputs={'scene': dense, 'cloud': dense.with_suffix('.ply')})

    if tracker.begin('reconstruct'):
        p = paths[tracker.key('scene')]
        cloud = tracker.path('cloud') if tracker.has('cloud') else None
        tracker.end('reconstruct', inputs={'scene': p, 'cloud': cloud},
                    outputs={'mesh': p.parent / (p.stem + '_mesh.ply')})

    if tracker.begin('refine'):
        p, mesh = paths[tracker.key('scene')], paths[tracker.key('mesh')]
        tracker.end('refine', inputs={'scene': p, 'mesh': mesh},
                    outputs={'mesh': p.parent / (p.stem + '_refine.ply')})

    if tracker.begin('texture'):
        p, mesh = paths[tracker.key('scene')], paths[tracker.key('mesh')]
        tracker.end('texture', inputs={'scene': p, 'mesh': mesh},
                    outputs={'mesh': p.parent
                             / f'{args.name}.{args.file_type}'})


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
        self.manifest = self.root / 'metadata.json'
        mvg = self.root / 'mvg'
        self.base_paths = {
            'output': self.root,
            'input': self.images,
            'mvg': mvg,
            'matches_dir': mvg / 'matches_dir',
            'matches_file': mvg / 'matches_dir' / 'matches.bin',
            'recon_dir': mvg / 'recon_dir',
            'sfm': mvg / 'sfm_data.json',
            'mvs': self.root / 'mvs',
            'mvs_scene': self.root / 'mvs' / 'scene.mvs',
        }

    def tracker(self, args, explicit=(), manifest=None) -> StageTracker:
        """A tracker over what is on disk now, as a fresh process would build it.

        Each call gets its own ``paths`` copy, so one simulated job cannot see
        another's rehydrated ``resume_*`` keys.
        """
        path = manifest or self.manifest
        meta = load_manifest(path)
        shape = pipeline_shape(args)
        from_stage, to_stage = resolve_range(args, shape)
        drift = drifted_stages(args, meta.get('stages') or {}, set(explicit),
                              shape)
        return StageTracker(meta, path, dict(self.base_paths), args, shape,
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
                    outputs={'sfm': self.base_paths['sfm']})
        record = self.records()['import']
        self.assertEqual('complete', record['status'])
        self.assertIn('finished', record)
        self.assertIsInstance(record['elapsed_s'], float)
        self.assertEqual({'sfm': 'mvg/sfm_data.json'}, record['outputs'])

    def test_recorded_paths_are_relative_to_the_output_dir(self):
        tracker = self.tracker(make_args())
        tracker.begin('convert')
        tracker.end('convert', inputs={'sfm': self.base_paths['sfm']},
                    outputs={'scene': self.base_paths['mvs_scene']})
        record = self.records()['convert']
        self.assertEqual({'sfm': 'mvg/sfm_data.json'}, record['inputs'])
        self.assertEqual({'scene': 'mvs/scene.mvs'}, record['outputs'])

    def test_an_artifact_outside_the_output_dir_is_recorded_absolute(self):
        # A relative -i would otherwise be recorded bare and re-rooted under the
        # output dir on read.
        tracker = self.tracker(make_args())
        tracker.begin('import')
        tracker.end('import', inputs={'images': Path('images')},
                    outputs={'sfm': self.base_paths['sfm']})
        recorded = self.records()['import']['inputs']['images']
        self.assertTrue(Path(recorded).is_absolute(), recorded)
        self.assertEqual(str(Path('images').resolve()), recorded)

    def test_none_valued_roles_are_not_recorded(self):
        tracker = self.tracker(make_args())
        tracker.begin('reconstruct')
        tracker.end('reconstruct',
                    inputs={'scene': self.base_paths['mvs_scene'],
                            'cloud': None},
                    outputs={'mesh': self.root / 'mvs' / 'scene_mesh.ply'})
        self.assertEqual(['scene'], list(self.records()['reconstruct']['inputs']))

    def test_commands_are_attributed_to_the_stage_that_ran_them(self):
        tracker = self.tracker(make_args())
        tracker.meta.setdefault('commands', {})['t0'] = 'before any stage'
        tracker.begin('convert')
        tracker.meta['commands']['t1'] = 'openMVG2openMVS -i ...'
        tracker.end('convert', outputs={'scene': self.base_paths['mvs_scene']})
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
        tracker.end('import', outputs={'sfm': self.base_paths['sfm']})
        self.assertEqual(self.base_paths['sfm'], tracker.path('sfm'))
        tracker.begin('features')
        tracker.end('features', outputs={'features': self.root / 'mvg/md'})
        solved = self.base_paths['recon_dir'] / 'sfm_data.bin'
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
        self.assertEqual(self.root / 'mvs/scene.mvs', tracker.path('scene'))

    def test_key_rehydrates_under_a_stable_synthetic_key(self):
        args = make_args()
        fake_run(self.tracker(args), args)
        tracker = self.tracker(make_args(from_stage='texture', rerun=True))
        self.assertTrue(tracker.begin('texture'))
        self.assertEqual('resume_mesh', tracker.key('mesh'))
        # RefineMesh names its output after the scene it refined against, not
        # after the mesh it consumed.
        self.assertEqual(self.root / 'mvs/scene_refine.ply',
                         tracker.paths['resume_mesh'])


class TestAbort(TrackerCase):
    def test_abort_marks_the_active_stage_failed(self):
        tracker = self.tracker(make_args())
        tracker.begin('import')
        tracker.end('import', outputs={'sfm': self.base_paths['sfm']})
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
        return self.root / 'gone' / 'metadata.json'

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
            tracker.end('import', outputs={'sfm': self.base_paths['sfm']})

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
        self.assertEqual([], list(self.root.glob('metadata.json.tmp*')))


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
        self.assertEqual('mvs/scene.mvs',
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
        self.assertEqual(self.root / 'mvg/view_pairs.txt',
                         job2.path('view_pairs'))

    def test_dropping_densify_rebuilds_the_mesh_chain_against_real_records(self):
        dense = make_args(mvs_densify=True)
        fake_run(self.tracker(dense), dense)
        self.assertEqual('mvs/scene_dense.mvs',
                         self.records()['reconstruct']['inputs']['scene'])

        plain = make_args(mvs_densify=False)
        tracker = self.tracker(plain, explicit={'mvs_densify'})
        self.assertEqual(['reconstruct', 'refine', 'texture'],
                         self.runs(tracker))
        self.assertEqual('inputs changed: scene', tracker.dirty['reconstruct'])
        fake_run(tracker, plain)
        self.assertEqual('mvs/scene_mesh.ply',
                         self.records()['reconstruct']['outputs']['mesh'])


if __name__ == '__main__':
    unittest.main()
