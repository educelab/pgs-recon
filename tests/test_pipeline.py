"""``pgs-recon`` end to end, with the binaries faked at the last inch.

Everything above ``run_command`` is the real thing: the parser, the planner, the
stage records, ``layout``'s naming, and ``resolve_exe`` against a real prefix of
real (empty, executable) files. Only the spawning of the child process is
replaced -- :func:`fake_binary` reads an argv and creates whatever the binary it
names would have written.

The fake learns each output path **from argv**, never from ``layout``. A fixture
that consulted ``layout`` would reimplement the code under test and could not
catch a wrapper passing the wrong path. For the same reason it *checks its
inputs*: a stage handed a path nothing wrote fails here rather than producing an
empty artifact that the next stage happily consumes.

Four things are pinned that nothing pinned before:

* **Both ADR 0003 invariants.** ``--archive-type -1`` on all four MVS stages, and
  ``-p`` whenever a dense cloud exists -- including one rehydrated from the
  manifest by a later job, which is the case that actually broke meshes.
* **ADR 0004's stated acceptance test**: one reconstruction split across three
  jobs must leave the same artifact tree as a single-shot run of the same shape.
* **The prefix a later job runs against is its own.** ``--path`` is not inherited
  from the manifest, so ``$PGS_RECON_PREFIX`` stays reachable on job 2..n -- the
  nodes a staged run splits onto, whose install prefix is the whole reason that
  tier exists.
* **One ``-S``.** ``openMVG_main_SfM`` keeps the last it is given, so the priors
  initializer must not be emitted alongside a caller's own.
"""
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pgs_recon import toolchain
from pgs_recon.toolchain import MVG_BIN, MVS_BIN
from pgs_recon.utility import ToolFailed

from test_toolchain import make_fake_prefix

DEPS = ('configargparse', 'sfm_utils', 'exiftool')
MISSING = [d for d in DEPS if importlib.util.find_spec(d) is None]
#: ``pgs-retexture``/``pgs-calibrate`` additionally read pixels.
APP_MISSING = MISSING + [d for d in ('cv2',)
                         if importlib.util.find_spec(d) is None]

#: Every binary a reconstruction can reach, by tool family.
MVG_TOOLS = ('openMVG_main_SfMInit_ImageListing', 'openMVG_main_ComputeFeatures',
             'openMVG_main_ComputeMatches', 'openMVG_main_GeometricFilter',
             'openMVG_main_SfM', 'openMVG_main_ComputeStructureFromKnownPoses',
             'openMVG_main_ComputeSfM_DataColor', 'openMVG_main_openMVG2openMVS',
             'openMVG_main_ConvertSfM_DataFormat',
             'openMVG_main_SfM_Localization', 'pgs-global-scaler')
MVS_TOOLS = ('DensifyPointCloud', 'ReconstructMesh', 'RefineMesh',
             'TextureMesh')


def flag(argv, name: str):
    """The value following ``name`` in ``argv``, or None if it is absent."""
    for i, token in enumerate(argv[:-1]):
        if token == name:
            return argv[i + 1]
    return None


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'fake {path.name}\n')


def _need(path: Path, argv) -> Path:
    if not Path(path).exists():
        raise ToolFailed(argv, 1, detail=f'input does not exist: {path}')
    return Path(path)


def fake_binary(argv, cwd=None) -> None:
    """Do what the binary named by ``argv[0]`` would do, as far as files go.

    Output *locations* come from the argv the wrapper built; only the fact that,
    say, ``DensifyPointCloud`` writes a ``.ply`` beside its scene is knowledge
    about the binary, which is what a fake is for.
    """
    tool = Path(argv[0]).name
    out = flag(argv, '-o')

    if tool == 'openMVG_main_SfMInit_ImageListing':
        _need(flag(argv, '-i'), argv)
        _write(Path(out) / 'sfm_data.json')
    elif tool == 'openMVG_main_ComputeFeatures':
        _need(flag(argv, '-i'), argv)
        Path(out).mkdir(parents=True, exist_ok=True)
    elif tool in ('openMVG_main_ComputeMatches', 'openMVG_main_GeometricFilter',
                  'openMVG_main_ComputeSfM_DataColor',
                  'openMVG_main_ConvertSfM_DataFormat'):
        _need(flag(argv, '-i'), argv)
        if flag(argv, '-m') is not None:
            _need(flag(argv, '-m'), argv)
        _write(Path(out))
    elif tool == 'openMVG_main_SfM':
        _need(flag(argv, '-i'), argv)
        # -M is resolved *relative to* the regions directory (-m), which is a
        # join, so `../` in it is legitimate and has to resolve here too.
        _need(_need(flag(argv, '-m'), argv) / Path(flag(argv, '-M')), argv)
        _write(Path(out) / 'sfm_data.bin')
    elif tool == 'openMVG_main_SfM_Localization':
        for f in ('-i', '-m', '-q'):
            _need(flag(argv, f), argv)
        _write(Path(out) / 'sfm_data_expanded.json')
    elif tool == 'openMVG_main_ComputeStructureFromKnownPoses':
        for f in ('-i', '-m', '-f'):
            _need(flag(argv, f), argv)
        _write(Path(out))
    elif tool == 'pgs-global-scaler':
        _need(flag(argv, '-i'), argv)
        _write(Path(out))
        for f in ('--save-landmarks', '--save-scaled-landmarks'):
            if flag(argv, f) is not None:
                _write(Path(flag(argv, f)))
    elif tool == 'openMVG_main_openMVG2openMVS':
        # Writes relative to cwd, by basename.
        _need(flag(argv, '-i'), argv)
        _write(Path(cwd) / out)
        (Path(cwd) / flag(argv, '-d')).mkdir(parents=True, exist_ok=True)
    elif tool in MVS_TOOLS:
        # Every MVS stage names its files against the working directory.
        work = Path(flag(argv, '-w'))
        for f in ('-i', '-m', '-p'):
            if flag(argv, f) is not None:
                _need(work / flag(argv, f), argv)
        _write(work / out)
        if tool == 'DensifyPointCloud':
            # The dense cloud OpenMVS pairs with the scene by name, not by flag.
            _write((work / out).with_suffix('.ply'))
    else:  # pragma: no cover - a new binary reached this fake unannounced
        raise AssertionError(f'no fake for {tool}: {argv}')


def tree(root: Path) -> list:
    """Every file under ``root``, relative and sorted -- the artifact tree."""
    return sorted(str(p.relative_to(root)) for p in root.rglob('*')
                  if p.is_file())


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class PipelineCase(unittest.TestCase):
    """A real prefix of fake binaries, and real output directories."""

    def setUp(self):
        # ``_main()`` configures the process-wide toolchain and never unsets it,
        # so without this every test here leaves ``_prefix`` pointing at a
        # deleted temp dir for whatever module runs next.
        self.addCleanup(toolchain.configure, prefix=None, recorder=None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.prefix = make_fake_prefix(self.tmp / 'prefix', mvg=MVG_TOOLS,
                                       mvs=MVS_TOOLS)
        self.images = self.tmp / 'images'
        self.images.mkdir()
        # The camera db is passed to a binary but never read by us; the fake
        # importer ignores it. Given explicitly so nothing looks under /usr/local.
        self.cam_db = self.tmp / 'camdb.txt'
        self.cam_db.write_text('Fake Camera;36\n')
        self.commands = []

    def fake(self, argv, cwd=None):
        self.commands.append((list(argv), cwd))
        fake_binary(argv, cwd=cwd)

    def run_recon(self, out: Path, *argv) -> dict:
        """One ``pgs-recon`` invocation over the fake toolchain.

        The ``atexit`` flush is collected and run here instead of at interpreter
        exit, so it lands while the output directory still exists.
        """
        from pgs_recon.apps import reconstruct
        full = ['pgs-recon', '-i', str(self.images), '-o', str(out),
                '--name', 'obj', '--path', str(self.prefix),
                '--cam-db', str(self.cam_db), '--log-level', 'ERROR', *argv]
        hooks = []

        def remember(fn):
            hooks.append(fn)
            return fn  # atexit.register returns its argument; the caller calls it

        with mock.patch.object(sys, 'argv', full), \
                mock.patch('pgs_recon.toolchain.run_command', self.fake), \
                mock.patch('atexit.register', remember):
            reconstruct._main()
        for hook in hooks:
            hook()
        return json.loads((out / 'metadata.json').read_text())

    def argv_for(self, tool: str) -> list:
        """The argv of the single invocation of ``tool``."""
        found = [c for c, _ in self.commands if Path(c[0]).name == tool]
        self.assertEqual(1, len(found), f'{tool} ran {len(found)} times')
        return found[0]

    def ran(self) -> list:
        return [Path(c[0]).name for c, _ in self.commands]


class TestFullRun(PipelineCase):
    def test_a_default_run_produces_the_expected_tree(self):
        out = self.tmp / 'recon'
        self.run_recon(out)
        self.assertEqual([
            'metadata.json',
            'mvg/matches_dir/matches.bin',
            'mvg/matches_dir/matches_filtered.bin',
            'mvg/recon_dir/sfm_data.bin',
            'mvg/recon_dir/sfm_data_colorized.ply',
            'mvg/sfm_data.json',
            'mvs/obj.obj',
            'mvs/scene.mvs',
            'mvs/scene_mesh.ply',
            # Refine names its output after the *scene* it refined against, not
            # after the mesh it consumed (ADR 0003; ADR 0006 ends this).
            'mvs/scene_refine.ply',
            'obj_recon_config.txt',
        ], tree(out))

    def test_the_binaries_run_in_pipeline_order(self):
        self.run_recon(self.tmp / 'recon', '--mvs-densify', '--mvg-robust',
                       '--mvg-autoscale', '0.47')
        self.assertEqual([
            'openMVG_main_SfMInit_ImageListing',
            'openMVG_main_ComputeFeatures',
            'openMVG_main_ComputeMatches',
            'openMVG_main_GeometricFilter',
            'openMVG_main_SfM',
            'openMVG_main_ComputeStructureFromKnownPoses',  # robust
            'pgs-global-scaler',                            # autoscale
            'openMVG_main_ComputeSfM_DataColor',
            'openMVG_main_openMVG2openMVS',
            'DensifyPointCloud',
            'ReconstructMesh',
            'RefineMesh',
            'TextureMesh',
        ], self.ran())

    def test_stopping_at_colorize_runs_no_mvs_binary(self):
        out = self.tmp / 'recon'
        self.run_recon(out, '--to', 'colorize')
        self.assertEqual([], [t for t in self.ran() if t in MVS_TOOLS])
        self.assertFalse((out / 'mvs' / 'scene.mvs').exists())

    def test_a_missing_binary_names_the_prefix_that_was_searched(self):
        # resolve_exe is exercised for real, so a prefix without the tool fails
        # before anything runs -- with the tier that chose the prefix named.
        (self.prefix / MVS_BIN / 'RefineMesh').unlink()
        with self.assertRaises(ToolFailed) as ctx:
            self.run_recon(self.tmp / 'recon')
        self.assertIn('RefineMesh', str(ctx.exception))
        self.assertIn('configure(prefix=...)', str(ctx.exception))
        self.assertEqual(127, ctx.exception.exit_code)


class TestSfMEngineFlags(PipelineCase):
    """``openMVG_main_SfM`` keeps the *last* ``-S``, so only one may be emitted."""

    def occurrences(self, argv, name: str) -> list:
        return [argv[i + 1] for i, tok in enumerate(argv[:-1]) if tok == name]

    def test_priors_on_incrementalv2_seed_from_the_existing_poses(self):
        self.run_recon(self.tmp / 'recon', '-m', 'incrementalv2', '--mvg-priors')
        argv = self.argv_for('openMVG_main_SfM')
        self.assertIn('-P', argv)
        self.assertEqual(['EXISTING_POSE'], self.occurrences(argv, '-S'))

    def test_an_explicit_initializer_is_not_duplicated_by_the_priors_default(self):
        # Both would otherwise be emitted, and the binary would silently take
        # the second -- so the flag the caller asked for won by accident.
        self.run_recon(self.tmp / 'recon', '-m', 'incrementalv2',
                       '--mvg-priors', '--mvg-initializer', 'MAX_PAIR')
        argv = self.argv_for('openMVG_main_SfM')
        self.assertEqual(['MAX_PAIR'], self.occurrences(argv, '-S'))

    def test_priors_on_another_engine_add_no_initializer(self):
        self.run_recon(self.tmp / 'recon', '--mvg-priors')
        argv = self.argv_for('openMVG_main_SfM')
        self.assertIn('-P', argv)
        self.assertEqual([], self.occurrences(argv, '-S'))

    def test_the_filtered_matches_are_passed_relative_to_the_regions_dir(self):
        # In the pipeline they are siblings, so the relative spelling is the
        # bare basename -- the argv a caller reading main_SfM.cpp would expect.
        self.run_recon(self.tmp / 'recon')
        argv = self.argv_for('openMVG_main_SfM')
        self.assertEqual('matches_filtered.bin', flag(argv, '-M'))
        self.assertEqual(str(self.tmp / 'recon' / 'mvg' / 'matches_dir'),
                         flag(argv, '-m'))

    def test_a_matches_file_outside_the_regions_dir_is_supported(self):
        """The library surface must not be narrower than the binary.

        ``-M`` is joined onto ``-m`` and that join resolves ``../``, so a matches
        file kept somewhere else is perfectly usable -- e.g. one set of regions
        matched several ways, with the variants stored apart from them. The
        wrapper spells it relatively rather than refusing it; passing it through
        as an absolute path is what would break, since stlplus concatenates.
        """
        from pgs_recon.openmvg import mvg_sfm
        regions = self.tmp / 'mvg' / 'matches_dir'
        regions.mkdir(parents=True)
        matches = self.tmp / 'mvg' / 'variants' / 'matches_strict.bin'
        matches.parent.mkdir(parents=True)
        matches.write_text('fake')
        sfm = self.tmp / 'mvg' / 'sfm_data.json'
        sfm.write_text('{}')

        toolchain.configure(prefix=self.prefix, recorder=None)
        with mock.patch('pgs_recon.toolchain.run_command', self.fake):
            mvg_sfm(sfm, features_dir=regions, matches=matches,
                    output_dir=self.tmp / 'out', engine='global')
        argv = self.argv_for('openMVG_main_SfM')
        self.assertEqual('../variants/matches_strict.bin', flag(argv, '-M'))
        # And the fake, which joins the two exactly as main_SfM.cpp does, found
        # it -- so the spelling is one the binary could actually resolve.
        self.assertTrue((self.tmp / 'out' / 'sfm_data.bin').is_file())


class TestArchiveType(PipelineCase):
    """ADR 0003: portable intermediates, so a `.mvs` survives another container."""

    def test_every_mvs_stage_is_told_to_write_an_interface_scene(self):
        self.run_recon(self.tmp / 'recon', '--mvs-densify')
        for tool in MVS_TOOLS:
            argv = self.argv_for(tool)
            self.assertEqual('-1', flag(argv, '--archive-type'),
                             f'{tool} did not pass --archive-type -1')


class TestDenseCloudIsHandedOver(PipelineCase):
    """ADR 0003: without ``-p``, ReconstructMesh meshes the *sparse* cloud.

    The scene densify writes still holds the sparse cloud, so omitting the flag
    is silent: a mesh comes out, just not the one the densification paid for.
    """

    def test_reconstruct_is_given_the_dense_cloud(self):
        self.run_recon(self.tmp / 'recon', '--mvs-densify')
        argv = self.argv_for('ReconstructMesh')
        self.assertEqual('scene_dense.ply', flag(argv, '-p'))
        self.assertEqual('scene_dense.mvs', flag(argv, '-i'))

    def test_reconstruct_is_given_the_dense_cloud_a_job_later(self):
        # The binding comes from the manifest here, not from a densify that ran
        # in this process -- which is the case a staged run actually hits.
        out = self.tmp / 'recon'
        self.run_recon(out, '--mvs-densify')
        self.commands.clear()
        self.run_recon(out, '--from', 'reconstruct', '--rerun')
        self.assertEqual('scene_dense.ply',
                         flag(self.argv_for('ReconstructMesh'), '-p'))

    def test_no_densify_means_no_point_cloud_flag(self):
        self.run_recon(self.tmp / 'recon')
        self.assertIsNone(flag(self.argv_for('ReconstructMesh'), '-p'))


class TestStagedRunsMatchSingleShot(PipelineCase):
    """ADR 0004's acceptance test: a split run must leave the same tree.

    The names are derived from the artifacts a stage consumes, so a job that
    rehydrates its inputs from the manifest has to reproduce them exactly. This
    is the regression that a rename (ADR 0006) could plausibly break, and the
    reason it is automated rather than checked by eye.
    """

    def one_shot(self, *shape) -> Path:
        out = self.tmp / 'single'
        self.run_recon(out, *shape)
        return out

    def staged(self, *shape) -> Path:
        out = self.tmp / 'staged'
        # Shape flags go on the first job, as submit_recon_pipeline.sh does; the
        # rest are recovered from the manifest.
        self.run_recon(out, *shape, '--to', 'convert')
        self.run_recon(out, '--from', 'densify', '--to', 'reconstruct')
        self.run_recon(out, '--from', 'refine')
        return out

    def assert_same_tree(self, *shape):
        single, staged = self.one_shot(*shape), self.staged(*shape)
        self.assertEqual(tree(single), tree(staged))
        # And every stage really did run exactly once in the staged case.
        stages = json.loads((staged / 'metadata.json').read_text())['stages']
        self.assertTrue(all(r['status'] == 'complete' for r in stages.values()),
                        stages)
        return tree(single)

    def test_three_windows_leave_the_same_artifacts_as_one_run(self):
        files = self.assert_same_tree('--mvs-densify')
        # Spot-check that the tree compared is the real one, not two empties.
        for expected in ('mvs/scene_dense.mvs', 'mvs/scene_dense.ply',
                         'mvs/scene_dense_mesh.ply',
                         'mvs/scene_dense_refine.ply', 'mvs/obj.obj'):
            self.assertIn(expected, files)

    def test_the_same_holds_for_a_scaled_and_re_triangulated_shape(self):
        self.assert_same_tree('--mvs-densify', '--mvg-robust',
                              '--mvg-autoscale', '0.47')

    def test_a_window_starting_at_sfm_reproduces_the_mvg_chain(self):
        """The MVG half split as well -- where ``-M``'s contract is live.

        ``openMVG_main_SfM`` takes the filtered matches by *basename*, resolved
        against the regions directory, and on a resume both of those come back
        from the manifest rather than from a stage that just ran. Every other
        staged case here splits inside MVS, so this is the one that exercises a
        rehydrated pair having to agree about where it lives.
        """
        single = self.tmp / 'single'
        self.run_recon(single)

        staged = self.tmp / 'staged'
        self.run_recon(staged, '--to', 'filter')
        self.commands.clear()
        self.run_recon(staged, '--from', 'sfm')

        self.assertEqual(tree(single), tree(staged))
        argv = self.argv_for('openMVG_main_SfM')
        self.assertEqual('matches_filtered.bin', flag(argv, '-M'))
        self.assertEqual(str(staged / 'mvg' / 'matches_dir'), flag(argv, '-m'))


class TestResumeIsIdempotent(PipelineCase):
    def test_a_verbatim_re_run_runs_nothing_and_changes_nothing(self):
        out = self.tmp / 'recon'
        self.run_recon(out, '--mvs-densify')
        before = tree(out)
        self.commands.clear()
        self.run_recon(out, '--mvs-densify')
        self.assertEqual([], self.ran())
        self.assertEqual(before, tree(out))

    def test_deleting_densify_from_the_shape_rebuilds_the_mesh_chain(self):
        out = self.tmp / 'recon'
        self.run_recon(out, '--mvs-densify')
        self.commands.clear()
        self.run_recon(out, '--no-mvs-densify')
        self.assertEqual(['ReconstructMesh', 'RefineMesh', 'TextureMesh'],
                         self.ran())
        # The mesh chain is back on the non-dense names, and the dense
        # intermediates are left where they were: nothing deletes.
        self.assertEqual('scene.mvs', flag(self.argv_for('ReconstructMesh'), '-i'))
        self.assertTrue((out / 'mvs' / 'scene_mesh.ply').is_file())


class TestPrefixResolution(PipelineCase):
    """``--path`` unset is what makes ``$PGS_RECON_PREFIX`` reachable."""

    def run_with_env_prefix(self, prefix: Path, *argv) -> None:
        """One ``pgs-recon`` run with no ``--path``, ``$PGS_RECON_PREFIX`` set."""
        from pgs_recon.apps import reconstruct
        full = ['pgs-recon', '--log-level', 'ERROR', *argv]
        with mock.patch.dict('os.environ',
                             {toolchain.PREFIX_ENV: str(prefix)}), \
                mock.patch.object(sys, 'argv', full), \
                mock.patch('pgs_recon.toolchain.run_command', self.fake), \
                mock.patch('atexit.register', lambda fn: fn):
            reconstruct._main()

    def test_the_environment_supplies_the_prefix_when_path_is_not_given(self):
        out = self.tmp / 'recon'
        self.run_with_env_prefix(self.prefix, '-i', str(self.images), '-o',
                                 str(out), '--name', 'obj', '--cam-db',
                                 str(self.cam_db), '--to', 'import')
        self.assertEqual(str(self.prefix / MVG_BIN
                             / 'openMVG_main_SfMInit_ImageListing'),
                         self.argv_for('openMVG_main_SfMInit_ImageListing')[0])

    def test_a_later_job_reads_the_environment_rather_than_the_recorded_path(self):
        """A recorded ``--path`` must not shadow ``$PGS_RECON_PREFIX``.

        Job 1 of a staged run is not usually the node that matters: the whole
        point of splitting is that refine lands somewhere else, and that node's
        install prefix legitimately differs. An inherited ``--path`` would make
        the environment tier unreachable there -- the same shadowing that cost
        ``--path`` its parser default, one level up, via the manifest.
        """
        out = self.tmp / 'recon'
        self.run_recon(out, '--to', 'import')          # records --path
        self.assertNotIn('path', json.loads(
            (out / 'metadata.json').read_text())['effective_args'])
        self.commands.clear()

        elsewhere = make_fake_prefix(self.tmp / 'other-prefix', mvg=MVG_TOOLS,
                                     mvs=MVS_TOOLS)
        self.run_with_env_prefix(elsewhere, '-o', str(out), '--to', 'features')
        self.assertEqual(str(elsewhere / MVG_BIN / 'openMVG_main_ComputeFeatures'),
                         self.argv_for('openMVG_main_ComputeFeatures')[0])

    def test_the_prefix_a_run_used_is_still_recoverable_from_the_manifest(self):
        # Dropping --path from effective_args must not cost the provenance: the
        # recorded argv[0] is absolute, so which prefix ran is still on record.
        out = self.tmp / 'recon'
        meta = self.run_recon(out, '--to', 'import')
        self.assertEqual(str(self.prefix), meta['parsed']['path'])
        self.assertTrue(any(str(self.prefix) in cmd
                            for cmd in meta['commands'].values()))

    def test_a_config_file_written_by_a_run_is_loadable_by_the_next(self):
        # Every unset argument is omitted from the config, so nothing comes back
        # as the string 'None' -- which for --path would send the next run
        # looking for its binaries under ./None.
        out = self.tmp / 'recon'
        self.run_recon(out, '--to', 'colorize')
        config = out / 'obj_recon_config.txt'
        self.assertNotIn('None', config.read_text())
        self.commands.clear()
        second = self.tmp / 'from-config'
        self.run_recon(second, '-c', str(config), '--to', 'import')
        self.assertEqual(['openMVG_main_SfMInit_ImageListing'], self.ran())


@unittest.skipIf(APP_MISSING, f'requires {", ".join(APP_MISSING)}')
class TestCalibrateLocalizesAgainstTheRecon(PipelineCase):
    """``pgs-calibrate``'s one wrapper call, whose five paths are easy to swap.

    The rest of the app (image prep, extracting the localized view) is stubbed:
    what MR2 changed here is the call, and a transposed ``-q``/``-o`` would
    otherwise only show up against a real OpenMVG.
    """

    def test_localization_is_given_the_recon_regions_and_a_private_output(self):
        from pgs_recon.apps import calibrate
        recon = self.tmp / 'recon'
        self.run_recon(recon)
        query = self.tmp / 'query.jpg'
        query.write_bytes(b'')
        self.commands.clear()

        out = self.tmp / 'calib'
        argv = ['pgs-calibrate', '-i', str(query), '-r', str(recon),
                '-o', str(out), '--path', str(self.prefix),
                '--log-level', 'ERROR']
        extracted = []
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch('pgs_recon.toolchain.run_command', self.fake), \
                mock.patch('atexit.register', lambda fn: fn), \
                mock.patch.object(calibrate, 'prepare_8bit_image',
                                  side_effect=self._stub_prepare), \
                mock.patch.object(calibrate, 'extract_calibration',
                                  side_effect=lambda p, *a: extracted.append(p)):
            calibrate._main()

        argv = self.argv_for('openMVG_main_SfM_Localization')
        self.assertEqual(str(recon / 'mvg' / 'matches_dir'), flag(argv, '-m'))
        self.assertEqual(str(out / 'query'), flag(argv, '-q'))
        self.assertEqual(str(out / 'query_matches'), flag(argv, '-u'))
        self.assertEqual(str(out / 'localization'), flag(argv, '-o'))
        self.assertEqual(str(recon / 'mvg' / 'recon_dir' / 'sfm_data.bin'),
                         flag(argv, '-i'))
        # The scene the wrapper reported is the one the calibration is read from.
        self.assertEqual([out / 'localization' / 'sfm_data_expanded.json'],
                         extracted)

    @staticmethod
    def _stub_prepare(src, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        prepared = Path(out_dir) / f'{Path(src).stem}.jpg'
        prepared.write_bytes(b'')
        return prepared


@unittest.skipIf(APP_MISSING, f'requires {", ".join(APP_MISSING)}')
class TestRetextureUsesTheSameToolchain(PipelineCase):
    """``pgs-retexture`` reaches the wrappers without fabricating a ``paths``
    dict, which is the reuse ADR 0005 exists for."""

    def test_it_textures_a_recon_directory_it_did_not_build(self):
        from pgs_recon.apps import retexture
        recon = self.tmp / 'recon'
        self.run_recon(recon)
        # A modality image set following the PGS naming convention, and an SfM
        # whose views match it. The retexture-specific machinery (image
        # conversion, SfM filtering) is not what is under test here, so it is
        # stubbed; the wrapper calls are not.
        modality = self.tmp / 'ir'
        modality.mkdir()
        (modality / 'ir_0_0_0.tif').write_bytes(b'')
        self.commands.clear()
        argv = ['pgs-retexture', '-i', str(modality), '-r', str(recon),
                '--path', str(self.prefix), '--log-level', 'ERROR']
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch('pgs_recon.toolchain.run_command', self.fake), \
                mock.patch('atexit.register', lambda fn: fn), \
                mock.patch.object(retexture, 'convert_modality_images',
                                  return_value={0: 'ir_0_0_0.jpg'}), \
                mock.patch.object(retexture, 'filter_sfm_for_camera',
                                  side_effect=self._stub_filter), \
                mock.patch.object(retexture, 'ensure_ply_mesh',
                                  side_effect=self._stub_mesh):
            retexture._main()
        # The scene it built and the mesh it staged both sit in the recon's mvs/,
        # which is what lets TextureMesh reference them by basename.
        argv = self.argv_for('TextureMesh')
        self.assertEqual(str(recon / 'mvs'), flag(argv, '-w'))
        self.assertEqual('ir_scene.mvs', flag(argv, '-i'))
        self.assertEqual('ir.obj', flag(argv, '-o'))
        self.assertTrue((recon / 'mvs' / 'ir.obj').is_file())

    @staticmethod
    def _stub_filter(sfm_json, camera_index, modality_dir, pos_to_name,
                     out_json):
        out_json.write_text('{}')
        return 1

    @staticmethod
    def _stub_mesh(mesh_path, work_dir, out_stem=None):
        staged = Path(work_dir) / f'{out_stem}_input.ply'
        shutil.copy(mesh_path, staged)
        return staged


if __name__ == '__main__':
    unittest.main()
