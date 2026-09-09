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
import contextlib
import importlib.util
import io
import json
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pgs_recon import layout, toolchain
from pgs_recon.toolchain import MVG_BIN, MVS_BIN
from pgs_recon.stages import StageError
from pgs_recon.utility import ToolFailed

from test_toolchain import flag, make_fake_prefix

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
             'openMVG_main_SfM_Localization', 'pgs-global-scaler',
             'pgs-decimate')
MVS_TOOLS = ('DensifyPointCloud', 'ReconstructMesh', 'RefineMesh',
             'TextureMesh')


#: Faces a fake PLY declares when the binary writing it was given no count.
#: An awkward number on purpose: ``coarsen`` takes ``COARSEN_RATIO`` of its
#: input's declared face count, so a round one could be matched by accident.
FAKE_FACES = 26003288


def _write(path: Path, faces: int = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != '.ply':
        path.write_text(f'fake {path.name}\n')
        return
    # A header a reader can get a face count out of. ``coarsen`` reads its
    # input's ``element face N`` to size its own target, so a fake mesh that
    # declares none puts that stage out of reach of these tests entirely.
    n = FAKE_FACES if faces is None else int(faces)
    path.write_text(f'ply\n'
                    f'format ascii 1.0\n'
                    f'comment fake {path.name}\n'
                    f'element vertex {max(1, n // 2)}\n'
                    f'property float x\n'
                    f'element face {n}\n'
                    f'property list uchar int vertex_indices\n'
                    f'end_header\n')


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
    elif tool == 'pgs-decimate':
        # Ours, and the one MVS-side binary that resolves absolute paths of its
        # own rather than basenames against -w.
        _need(flag(argv, '-i'), argv)
        # A face budget is met exactly, so `coarsen`'s achieved count matches
        # what it asked for and its drift check has something real to compare.
        _write(Path(flag(argv, '-o')), faces=flag(argv, '--max-faces'))
        if flag(argv, '--report') is not None:
            _write(Path(flag(argv, '--report')))
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
        return json.loads(layout.manifest(out).read_text())

    def argv_for(self, tool: str) -> list:
        """The argv of the single invocation of ``tool``."""
        found = [c for c, _ in self.commands if Path(c[0]).name == tool]
        self.assertEqual(1, len(found), f'{tool} ran {len(found)} times')
        return found[0]

    def argv_writing(self, name: str) -> list:
        """The argv of the one invocation whose ``-o`` names ``name``.

        Two stages reach ``pgs-decimate`` -- ``coarsen`` on the way into refine
        and ``decimate`` on the way out -- so the tool name no longer picks out
        a call, and the artifact it writes does.
        """
        found = [c for c, _ in self.commands
                 if flag(c, '-o') is not None
                 and Path(str(flag(c, '-o'))).name == name]
        self.assertEqual(1, len(found),
                         f'{name} was written by {len(found)} invocation(s)')
        return found[0]

    def ran(self) -> list:
        return [Path(c[0]).name for c, _ in self.commands]

    def captured_warnings(self) -> list:
        """Warnings the run logs, collected into a list that fills as it goes.

        ``assertLogs`` cannot express "and nothing was warned about", which is
        half of what the coarsen and decimate warnings have to get right.
        """
        collected = []

        class Collect(logging.Handler):
            def emit(self, record):
                collected.append(record.getMessage())

        logger = logging.getLogger('pgs-recon')
        handler = Collect(level=logging.WARNING)
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        return collected


class TestFullRun(PipelineCase):
    def test_a_default_run_produces_the_expected_tree(self):
        out = self.tmp / 'recon'
        self.run_recon(out)
        self.assertEqual([
            'mvg/matches_dir/matches.bin',
            'mvg/matches_dir/matches_filtered.bin',
            'mvg/recon_dir/colorize_sfm.ply',
            'mvg/recon_dir/sfm_data.bin',
            'mvg/sfm_data.json',
            # coarsen is in a default shape: refine's own CGAL preparation is
            # ours to do now (ADR 0009).
            'mvs/coarsen_mesh.ply',
            'mvs/convert_scene.mvs',
            'mvs/obj.obj',
            'mvs/reconstruct_mesh.ply',
            # Every intermediate is <stage>_<role> (ADR 0006): refine's output
            # is a mesh named for the stage that made it, not for the scene it
            # was refined against.
            'mvs/refine_mesh.ply',
            'obj_recon_config.txt',
            # The manifest is named for the tool, not 'metadata.json', which is
            # what a PGS *scan* directory calls its own descriptor.
            'pgs-recon.json',
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
            'pgs-decimate',                                 # coarsen
            'RefineMesh',
            'TextureMesh',
        ], self.ran())

    def test_stopping_at_colorize_runs_no_mvs_binary(self):
        out = self.tmp / 'recon'
        self.run_recon(out, '--to', 'colorize')
        self.assertEqual([], [t for t in self.ran() if t in MVS_TOOLS])
        self.assertFalse((out / 'mvs' / 'convert_scene.mvs').exists())

    def test_a_missing_binary_names_the_prefix_that_was_searched(self):
        # resolve_exe is exercised for real, so a prefix without the tool fails
        # before anything runs -- with the tier that chose the prefix named.
        (self.prefix / MVS_BIN / 'RefineMesh').unlink()
        with self.assertRaises(ToolFailed) as ctx:
            self.run_recon(self.tmp / 'recon')
        self.assertIn('RefineMesh', str(ctx.exception))
        self.assertIn('configure(prefix=...)', str(ctx.exception))
        self.assertEqual(127, ctx.exception.exit_code)


class TestDecimateStage(PipelineCase):
    """The stage between refine and texture, and what enabling it is.

    ADR 0008: the budget *is* the enable flag, the coarse mesh is what gets
    textured, and the report is recorded so a finished run states the deviation
    its deliverable is within.
    """

    def test_no_budget_means_no_decimate_stage(self):
        out = self.tmp / 'recon'
        self.run_recon(out)
        self.assertFalse((out / 'mvs' / 'decimate_mesh.ply').exists())
        self.assertFalse((out / 'mvs' / 'decimate_report.json').exists())
        # `pgs-decimate` still runs -- coarsen drives the same binary -- so the
        # absence to check is the stage's, not the tool's.
        self.assertEqual(['coarsen_mesh.ply'],
                         [Path(str(flag(c, '-o'))).name
                          for c, _ in self.commands
                          if Path(c[0]).name == 'pgs-decimate'])

    def test_a_budget_puts_it_between_refine_and_texture(self):
        self.run_recon(self.tmp / 'recon', '--decimate-max-error', '0.2')
        ran = self.ran()
        self.assertEqual(['RefineMesh', 'pgs-decimate', 'TextureMesh'],
                         ran[-3:])

    def test_the_budget_and_its_paths_reach_argv(self):
        out = self.tmp / 'recon'
        self.run_recon(out, '--decimate-max-error', '0.2')
        argv = self.argv_writing('decimate_mesh.ply')
        self.assertEqual('0.2', flag(argv, '--max-error'))
        self.assertEqual('error', flag(argv, '--prefer'))
        self.assertEqual(str(out / 'mvs' / 'refine_mesh.ply'), flag(argv, '-i'))
        self.assertEqual(str(out / 'mvs' / 'decimate_mesh.ply'),
                         flag(argv, '-o'))
        self.assertEqual(str(out / 'mvs' / 'decimate_report.json'),
                         flag(argv, '--report'))
        # No face budget was given, so the binary's own default governs.
        self.assertIsNone(flag(argv, '--max-faces'))

    def test_the_search_levers_reach_argv(self):
        self.run_recon(self.tmp / 'recon', '--decimate-max-error', '0.2',
                       '--decimate-quadric-seed', '1e-7',
                       '--decimate-min-gain', '0.2')
        argv = self.argv_writing('decimate_mesh.ply')
        self.assertEqual('1e-07', flag(argv, '--quadric-seed'))
        self.assertEqual('0.2', flag(argv, '--min-gain'))

    def test_the_search_levers_are_absent_unasked(self):
        """A run that says nothing about the search gets the binary's own
        defaults, not a spelling of them from here."""
        self.run_recon(self.tmp / 'recon', '--decimate-max-error', '0.2')
        argv = self.argv_writing('decimate_mesh.ply')
        self.assertIsNone(flag(argv, '--quadric-seed'))
        self.assertIsNone(flag(argv, '--min-gain'))

    def test_a_seed_alone_does_not_turn_the_stage_on(self):
        """It tunes the search; the budgets are still the only enable flag
        (ADR 0008 s6), and a seed with nothing to search for is a no-op."""
        self.run_recon(self.tmp / 'recon', '--decimate-quadric-seed', '1e-7')
        # As above, coarsen drives the same binary, so what says the stage is
        # off is that nothing wrote its output -- not that the tool never ran.
        self.assertEqual(['coarsen_mesh.ply'],
                         [Path(str(flag(c, '-o'))).name
                          for c, _ in self.commands
                          if Path(c[0]).name == 'pgs-decimate'])

    def test_texture_is_handed_the_coarse_mesh(self):
        """The deliverable is born coarse: texturing faces that are about to be
        thrown away is the thing this ordering exists to avoid."""
        out = self.tmp / 'recon'
        self.run_recon(out, '--decimate-max-error', '0.2')
        argv = self.argv_for('TextureMesh')
        self.assertEqual('decimate_mesh.ply', flag(argv, '-m'))
        self.assertEqual(str(out / 'mvs'), flag(argv, '-w'))

    def test_the_mesh_and_the_report_are_both_recorded(self):
        out = self.tmp / 'recon'
        meta = self.run_recon(out, '--decimate-max-error', '0.2')
        record = meta['stages']['decimate']
        self.assertEqual('complete', record['status'])
        self.assertEqual({'mesh': 'mvs/refine_mesh.ply'}, record['inputs'])
        self.assertEqual({'mesh': 'mvs/decimate_mesh.ply',
                          'deviation': 'mvs/decimate_report.json'},
                         record['outputs'])
        self.assertTrue((out / 'mvs' / 'decimate_report.json').is_file())

    def test_a_face_budget_alone_enables_it(self):
        """No autoscale, no world units -- and a face budget is still a target,
        which is the point of having two."""
        self.run_recon(self.tmp / 'recon', '--decimate-max-faces', '2000000')
        argv = self.argv_writing('decimate_mesh.ply')
        self.assertEqual('2000000', flag(argv, '--max-faces'))
        self.assertIsNone(flag(argv, '--max-error'))

    def test_it_coarsens_the_reconstructed_mesh_when_refine_is_off(self):
        out = self.tmp / 'recon'
        self.run_recon(out, '--no-mvs-refine', '--decimate-max-error', '0.2')
        self.assertEqual(str(out / 'mvs' / 'reconstruct_mesh.ply'),
                         flag(self.argv_for('pgs-decimate'), '-i'))

    def test_a_zero_budget_on_a_resume_drops_the_stage_and_reruns_texture(self):
        """The documented way back from a budget that was wrong.

        Omitting the flag would inherit the recorded budget, so ``0`` is the
        off switch -- and dropping the stage costs one re-run, not a pipeline,
        because texture's recorded mesh stops matching its binding.
        """
        out = self.tmp / 'recon'
        self.run_recon(out, '--decimate-max-error', '0.2')
        self.commands.clear()
        meta = self.run_recon(out, '--decimate-max-error', '0')
        self.assertEqual(['TextureMesh'], self.ran())
        self.assertEqual('refine_mesh.ply',
                         flag(self.argv_for('TextureMesh'), '-m'))
        self.assertNotIn('decimate', meta['shape'])
        # The orphaned record stays; nothing deletes the coarse mesh either.
        self.assertEqual('complete', meta['stages']['decimate']['status'])
        self.assertTrue((out / 'mvs' / 'decimate_mesh.ply').is_file())

    def test_a_world_unit_budget_without_autoscale_warns(self):
        """The binary cannot detect this -- it only ever sees a mesh -- so the
        pipeline is the only place that knows the units are arbitrary."""
        with self.assertLogs('pgs-recon', level='WARNING') as captured:
            self.run_recon(self.tmp / 'recon', '--decimate-max-error', '0.2')
        self.assertTrue(any('autoscale' in m for m in captured.output),
                        captured.output)

    def test_a_scaled_run_is_not_warned_at(self):
        warnings = self.captured_warnings()
        self.run_recon(self.tmp / 'recon', '--decimate-max-error', '0.2',
                       '--mvg-autoscale', '0.47')
        self.assertEqual([], [m for m in warnings if 'autoscale' in m])

    def test_a_face_budget_is_not_warned_at(self):
        """A face budget has no units to be meaningless in."""
        warnings = self.captured_warnings()
        self.run_recon(self.tmp / 'recon', '--decimate-max-faces', '2000000')
        self.assertEqual([], [m for m in warnings if 'autoscale' in m])

    def test_omitting_the_budget_on_a_resume_inherits_it(self):
        """Which is why zero exists. The effective arguments are defaults on the
        next run, so a bare resume keeps decimating."""
        out = self.tmp / 'recon'
        self.run_recon(out, '--decimate-max-error', '0.2')
        self.commands.clear()
        meta = self.run_recon(out)
        self.assertEqual([], self.ran())
        self.assertIn('decimate', meta['shape'])
        self.assertEqual(0.2, meta['effective_args']['decimate_max_error'])


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


class TestDirectAndRobustAreDistinctStages(PipelineCase):
    """Both run ``ComputeStructureFromKnownPoses``, and both can be enabled.

    The names used to tell them apart by chaining (``sfm_data_structured.bin``
    then ``sfm_data_structured_structured.bin``). Under ADR 0006 the ``sfm``
    stage writes the solve's own name whichever engine implements it, so robust
    still has a distinct output rather than reading and writing one file.
    """

    def triangulations(self) -> list:
        return [c for c, _ in self.commands
                if Path(c[0]).name == 'openMVG_main_ComputeStructureFromKnownPoses']

    def test_direct_writes_the_solve_and_robust_writes_its_own(self):
        out = self.tmp / 'recon'
        self.run_recon(out, '--mvg-recon-method', 'direct', '--mvg-robust')
        direct, robust = self.triangulations()
        solved = str(out / 'mvg' / 'recon_dir' / 'sfm_data.bin')
        self.assertIn('-d', direct)
        self.assertEqual(str(out / 'mvg' / 'sfm_data.json'), flag(direct, '-i'))
        self.assertEqual(solved, flag(direct, '-o'))
        self.assertNotIn('-d', robust)
        self.assertEqual(solved, flag(robust, '-i'))
        self.assertEqual(str(out / 'mvg' / 'recon_dir' / 'robust_sfm.bin'),
                         flag(robust, '-o'))

    def test_no_openmvg_solve_runs_for_the_direct_method(self):
        out = self.tmp / 'recon'
        self.run_recon(out, '--mvg-recon-method', 'direct')
        self.assertNotIn('openMVG_main_SfM', self.ran())
        self.assertTrue((out / 'mvg' / 'recon_dir' / 'sfm_data.bin').is_file())


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
        self.assertEqual('densify.ply', flag(argv, '-p'))
        self.assertEqual('densify.mvs', flag(argv, '-i'))

    def test_reconstruct_is_given_the_dense_cloud_a_job_later(self):
        # The binding comes from the manifest here, not from a densify that ran
        # in this process -- which is the case a staged run actually hits.
        out = self.tmp / 'recon'
        self.run_recon(out, '--mvs-densify')
        self.commands.clear()
        self.run_recon(out, '--from', 'reconstruct', '--rerun')
        self.assertEqual('densify.ply',
                         flag(self.argv_for('ReconstructMesh'), '-p'))

    def test_no_densify_means_no_point_cloud_flag(self):
        self.run_recon(self.tmp / 'recon')
        self.assertIsNone(flag(self.argv_for('ReconstructMesh'), '-p'))


class TestReconstructCleanFlags(PipelineCase):
    """ReconstructMesh's clean block is reachable, so it can be bisected.

    All of it runs at OpenMVS defaults, and the whole block sits between the ROI
    trim and the mesh being written -- which is where a run that dies in clean
    dies. Omitted means the binary's default, not ours.
    """

    def test_clean_flags_are_omitted_by_default(self):
        self.run_recon(self.tmp / 'recon')
        argv = self.argv_for('ReconstructMesh')
        for name in ('--remove-spurious', '--remove-spikes', '--close-holes'):
            self.assertIsNone(flag(argv, name), f'{name} was not left to OpenMVS')

    def test_clean_flags_reach_argv(self):
        self.run_recon(self.tmp / 'recon', '--mvs-smooth', '0',
                       '--mvs-remove-spurious', '0', '--no-mvs-remove-spikes',
                       '--mvs-close-holes', '0')
        argv = self.argv_for('ReconstructMesh')
        self.assertEqual('0', flag(argv, '--smooth'))
        # float, per OpenMVS's own declaration -- boost parses '0.0' fine.
        self.assertEqual('0.0', flag(argv, '--remove-spurious'))
        self.assertEqual('0', flag(argv, '--remove-spikes'))
        self.assertEqual('0', flag(argv, '--close-holes'))


class TestCoarsenStage(PipelineCase):
    """The stage between reconstruct and refine (ADR 0009).

    ``RefineMesh``'s own pre-refinement decimation is a CGAL pass whose cost at
    a fixed target varies 135x between meshes; this drives ``pgs-decimate`` to
    the same target instead and tells the binary to skip both of its own
    preparation passes. What is checked here is the whole coupling: the target
    computed from the input mesh's header, the two flags that must reach
    ``RefineMesh`` together, and the face counts recorded in place of the log
    line that disabling the pass removes.
    """

    def test_the_target_is_the_ratio_of_the_input_mesh(self):
        out = self.tmp / 'recon'
        self.run_recon(out)
        argv = self.argv_writing('coarsen_mesh.ply')
        self.assertEqual(str(round(FAKE_FACES * 0.375)),
                         flag(argv, '--max-faces'))
        self.assertEqual(str(out / 'mvs' / 'reconstruct_mesh.ply'),
                         flag(argv, '-i'))
        self.assertEqual(str(out / 'mvs' / 'coarsen_mesh.ply'),
                         flag(argv, '-o'))

    def test_it_is_a_face_budget_and_not_a_deviation_one(self):
        # `--max-error 0` is what collapses the search to a single round, which
        # is the whole reason this is affordable against the pass it replaces.
        self.run_recon(self.tmp / 'recon')
        argv = self.argv_writing('coarsen_mesh.ply')
        self.assertEqual('0', flag(argv, '--max-error'))
        self.assertIsNone(flag(argv, '--prefer'))

    def test_the_measurement_is_turned_down_and_no_report_is_written(self):
        # `pgs-decimate` measures on every round whatever the target, so the
        # cheapest it can be is the floor; nothing consumes a deviation here,
        # refine reworking this surface anyway.
        out = self.tmp / 'recon'
        self.run_recon(out)
        argv = self.argv_writing('coarsen_mesh.ply')
        self.assertEqual('1', flag(argv, '--samples-per-face'))
        self.assertEqual('0', flag(argv, '--curvature-samples'))
        self.assertIsNone(flag(argv, '--report'))
        self.assertFalse((out / 'mvs' / 'coarsen_report.json').exists())

    def test_an_explicit_ratio_reaches_the_target(self):
        self.run_recon(self.tmp / 'recon', '--coarsen-ratio', '0.1')
        self.assertEqual(str(round(FAKE_FACES * 0.1)),
                         flag(self.argv_writing('coarsen_mesh.ply'),
                              '--max-faces'))

    def test_an_explicit_face_count_overrides_the_ratio(self):
        self.run_recon(self.tmp / 'recon', '--coarsen-max-faces', '9761507')
        self.assertEqual('9761507',
                         flag(self.argv_writing('coarsen_mesh.ply'),
                              '--max-faces'))

    def test_a_face_count_at_or_above_the_input_warns(self):
        """The no-op the drift check cannot see: the target is met before the
        first collapse, so achieved == target and the miss is 0, while refine
        skips its own preparation anyway and nothing reduces the mesh."""
        with self.assertLogs('pgs-recon', level='WARNING') as captured:
            self.run_recon(self.tmp / 'recon',
                           '--coarsen-max-faces', str(FAKE_FACES))
        self.assertTrue(any('write the mesh through unchanged' in m
                            for m in captured.output), captured.output)

    def test_a_face_count_below_the_input_does_not_warn(self):
        self.run_recon(self.tmp / 'recon',
                       '--coarsen-max-faces', str(FAKE_FACES - 1))
        self.assertEqual([], [m for m in self.captured_warnings()
                              if 'unchanged' in m])

    def test_refine_gets_both_preparation_flags(self):
        out = self.tmp / 'recon'
        self.run_recon(out)
        argv = self.argv_for('RefineMesh')
        self.assertEqual('1.0', flag(argv, '--decimate'))
        self.assertEqual('2', flag(argv, '--ensure-edge-size'))
        self.assertEqual('coarsen_mesh.ply', flag(argv, '-m'))

    def test_dropping_the_stage_leaves_refine_at_its_own_defaults(self):
        self.run_recon(self.tmp / 'recon', '--no-mvs-coarsen')
        argv = self.argv_for('RefineMesh')
        self.assertIsNone(flag(argv, '--decimate'))
        self.assertIsNone(flag(argv, '--ensure-edge-size'))
        self.assertEqual('reconstruct_mesh.ply', flag(argv, '-m'))
        self.assertNotIn('pgs-decimate', self.ran())

    def test_an_explicit_refine_flag_still_wins(self):
        self.run_recon(self.tmp / 'recon', '--refine-decimate', '0.5')
        argv = self.argv_for('RefineMesh')
        self.assertEqual('0.5', flag(argv, '--decimate'))
        # And the other half of the pair is still coupled to the stage.
        self.assertEqual('2', flag(argv, '--ensure-edge-size'))

    def test_the_face_counts_are_recorded_in_place_of_the_lost_log_line(self):
        meta = self.run_recon(self.tmp / 'recon')
        record = meta['stages']['coarsen']
        self.assertEqual('complete', record['status'])
        self.assertEqual({'mesh': 'mvs/reconstruct_mesh.ply'},
                         record['inputs'])
        self.assertEqual({'mesh': 'mvs/coarsen_mesh.ply'}, record['outputs'])
        self.assertEqual(FAKE_FACES, record['input_faces'])
        self.assertEqual(round(FAKE_FACES * 0.375), record['target_faces'])
        self.assertEqual(record['target_faces'], record['achieved_faces'])
        self.assertAlmostEqual(0.375, record['achieved_ratio'], places=4)

    def test_a_face_count_it_could_not_hit_warns(self):
        """The drift canary firing: a stalled collapse, which non-manifold
        geometry under --preserve-topology is the usual cause of."""
        def stalling(argv, cwd=None):
            self.commands.append((list(argv), cwd))
            if Path(argv[0]).name == 'pgs-decimate':
                # Half the faces it was told to reach.
                _write(Path(flag(argv, '-o')),
                       faces=int(flag(argv, '--max-faces')) * 2)
                return
            fake_binary(argv, cwd=cwd)

        with self.assertLogs('pgs-recon', level='WARNING') as captured:
            with mock.patch.object(self, 'fake', stalling):
                meta = self.run_recon(self.tmp / 'recon')
        self.assertTrue(any('coarsen asked for' in m for m in captured.output),
                        captured.output)
        record = meta['stages']['coarsen']
        self.assertEqual(record['target_faces'] * 2, record['achieved_faces'])

    def test_a_mesh_with_no_readable_face_count_fails_the_stage(self):
        """Better an error than a target invented from nothing: the stage
        cannot size itself, and says which two flags recover.
        """
        def textfile(argv, cwd=None):
            self.commands.append((list(argv), cwd))
            fake_binary(argv, cwd=cwd)
            if Path(argv[0]).name == 'ReconstructMesh':
                (Path(flag(argv, '-w')) / flag(argv, '-o')).write_text('nope\n')

        with mock.patch.object(self, 'fake', textfile):
            with self.assertRaises(StageError) as ctx:
                self.run_recon(self.tmp / 'recon')
        self.assertIn('--coarsen-max-faces', str(ctx.exception))
        self.assertIn('--no-mvs-coarsen', str(ctx.exception))

    def test_a_mesh_with_no_faces_fails_the_stage(self):
        """`element face 0` is a point cloud, and every ratio of it is zero."""
        def empty(argv, cwd=None):
            self.commands.append((list(argv), cwd))
            fake_binary(argv, cwd=cwd)
            if Path(argv[0]).name == 'ReconstructMesh':
                _write(Path(flag(argv, '-w')) / flag(argv, '-o'), faces=0)

        with mock.patch.object(self, 'fake', empty):
            with self.assertRaises(StageError) as ctx:
                self.run_recon(self.tmp / 'recon')
        self.assertIn('declares no faces', str(ctx.exception))

    def test_it_can_be_a_job_of_its_own(self):
        """The cheapest window in the shape: no scene, no images, one mesh.

        Which is the point of it being a stage rather than something refine's
        block does -- a coarsen job needs neither a big-memory node nor the
        undistorted images.
        """
        out = self.tmp / 'recon'
        self.run_recon(out, '--to', 'reconstruct')
        self.commands.clear()
        meta = self.run_recon(out, '--from', 'coarsen', '--to', 'coarsen')
        self.assertEqual(['pgs-decimate'], self.ran())
        self.assertEqual('complete', meta['stages']['coarsen']['status'])
        self.assertNotIn('refine', meta['stages'])

    def test_the_decimate_stage_still_coarsens_the_refined_mesh(self):
        """Both stages in one shape, each with its own artifact and role.

        They drive the same binary to different ends -- a face budget into
        refine, a deviation budget out of it -- so what has to hold is that
        neither reads the other's mesh.
        """
        out = self.tmp / 'recon'
        self.run_recon(out, '--decimate-max-error', '0.2')
        self.assertEqual(['ReconstructMesh', 'pgs-decimate', 'RefineMesh',
                          'pgs-decimate', 'TextureMesh'], self.ran()[-5:])
        coarsen = self.argv_writing('coarsen_mesh.ply')
        decimate = self.argv_writing('decimate_mesh.ply')
        self.assertEqual(str(out / 'mvs' / 'reconstruct_mesh.ply'),
                         flag(coarsen, '-i'))
        self.assertEqual(str(out / 'mvs' / 'refine_mesh.ply'),
                         flag(decimate, '-i'))
        self.assertEqual('decimate_mesh.ply',
                         flag(self.argv_for('TextureMesh'), '-m'))

    def test_a_warning_when_it_is_asked_for_without_a_refine(self):
        with self.assertLogs('pgs-recon', level='WARNING') as captured:
            self.run_recon(self.tmp / 'recon', '--no-mvs-refine',
                           '--mvs-coarsen')
        self.assertTrue(any('--mvs-coarsen has no effect' in m
                            for m in captured.output), captured.output)

    def test_no_such_warning_on_the_default(self):
        # Coarsening is on by default, so --no-mvs-refine on its own must not
        # be scolded about a flag the caller never touched.
        warnings = self.captured_warnings()
        self.run_recon(self.tmp / 'recon', '--no-mvs-refine')
        self.assertEqual([], [m for m in warnings if '--mvs-coarsen' in m])


class TestRefineDecimateIsNotTheDecimateStage(PipelineCase):
    """``--decimation-factor`` is gone, and what replaced it is RefineMesh's.

    Two flags whose names both say "decimate", one of them silently meaning
    another stage's, is the confusion the rename exists to end -- so this pins
    that each reaches its own binary.
    """

    def test_refine_decimate_reaches_refine_mesh(self):
        self.run_recon(self.tmp / 'recon', '--refine-decimate', '1')
        self.assertEqual('1.0', flag(self.argv_for('RefineMesh'), '--decimate'))

    def test_the_old_spelling_is_refused(self):
        # Deleted, not aliased: a script still passing it fails at parsing
        # rather than silently decimating differently.
        with self.assertRaises(SystemExit) as ctx, \
                contextlib.redirect_stderr(io.StringIO()) as usage:
            self.run_recon(self.tmp / 'recon', '--decimation-factor', '0.5')
        self.assertEqual(2, ctx.exception.code)
        self.assertIn('--decimation-factor', usage.getvalue())

    def test_the_two_do_not_reach_each_other(self):
        self.run_recon(self.tmp / 'recon', '--refine-decimate', '1',
                       '--decimate-max-error', '0.2')
        self.assertIsNone(flag(self.argv_writing('decimate_mesh.ply'),
                               '--decimate'))
        self.assertIsNone(flag(self.argv_for('RefineMesh'), '--max-error'))


class TestStagedRunsMatchSingleShot(PipelineCase):
    """ADR 0004's acceptance test: a split run must leave the same tree.

    A later job rehydrates its inputs from the manifest rather than recomputing
    them, which is what has to hold for the tree to match. That is the invariant
    the rename (ADR 0006) rides on, and the reason this is automated rather than
    checked by eye.
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
        self.run_recon(out, '--from', 'coarsen')
        return out

    def assert_same_tree(self, *shape):
        single, staged = self.one_shot(*shape), self.staged(*shape)
        self.assertEqual(tree(single), tree(staged))
        # And every stage really did run exactly once in the staged case.
        stages = json.loads(layout.manifest(staged).read_text())['stages']
        self.assertTrue(all(r['status'] == 'complete' for r in stages.values()),
                        stages)
        return tree(single)

    def test_three_windows_leave_the_same_artifacts_as_one_run(self):
        files = self.assert_same_tree('--mvs-densify')
        # Spot-check that the tree compared is the real one, not two empties.
        for expected in ('mvs/densify.mvs', 'mvs/densify.ply',
                         'mvs/reconstruct_mesh.ply', 'mvs/refine_mesh.ply',
                         'mvs/obj.obj'):
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
        self.assertEqual(['ReconstructMesh', 'pgs-decimate', 'RefineMesh',
                          'TextureMesh'], self.ran())
        # The mesh chain is back on the non-dense names, and the dense
        # intermediates are left where they were: nothing deletes.
        self.assertEqual('convert_scene.mvs',
                         flag(self.argv_for('ReconstructMesh'), '-i'))
        self.assertTrue((out / 'mvs' / 'reconstruct_mesh.ply').is_file())


class TestPre20ManifestName(PipelineCase):
    """A directory whose manifest is still called ``metadata.json``.

    The artifact rename was safe because names are only ever written; the
    manifest is the one thing located *by* name, so this is the case that would
    silently re-run an entire finished reconstruction.
    """

    def make_legacy(self, out: Path, *shape) -> None:
        """A finished run as pgs-recon before 2.0 left it."""
        self.run_recon(out, *shape)
        layout.manifest(out).rename(layout.legacy_manifest(out))
        self.commands.clear()

    def test_a_pre_1_8_directory_resumes_without_running_anything(self):
        out = self.tmp / 'recon'
        self.make_legacy(out, '--mvs-densify')
        meta = self.run_recon(out, '--mvs-densify')
        self.assertEqual([], self.ran())
        self.assertTrue(all(r['status'] == 'complete'
                            for r in meta['stages'].values()), meta['stages'])

    def test_the_arguments_of_a_pre_1_8_run_are_still_recovered(self):
        # --name and --input come back out of the old file, so a later job needs
        # neither -- the property the whole staged workflow rests on.
        out = self.tmp / 'recon'
        self.make_legacy(out)
        meta = self.run_recon(out, '--from', 'texture', '--rerun')
        self.assertEqual(['TextureMesh'], self.ran())
        self.assertEqual('obj', meta['effective_args']['name'])

    def test_the_next_run_records_to_the_new_name_and_leaves_the_old_alone(self):
        out = self.tmp / 'recon'
        self.make_legacy(out)
        legacy = layout.legacy_manifest(out)
        before = legacy.read_text()
        self.run_recon(out, '--from', 'texture', '--rerun')
        self.assertTrue(layout.manifest(out).is_file())
        self.assertEqual(before, legacy.read_text())
        # And from then on the new file is the record: the stale one is ignored,
        # not merged, so a second resume still runs only what was asked.
        self.commands.clear()
        self.run_recon(out)
        self.assertEqual([], self.ran())


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
            layout.manifest(out).read_text())['effective_args'])
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
        # A PGS scan directory holding two captures, which is what capture
        # retexture reads: its metadata.json says how to select the images. The
        # retexture-specific machinery (image conversion, SfM filtering) is not
        # what is under test here, so it is stubbed; the wrapper calls are not.
        scan = self.tmp / 'ir'
        scan.mkdir()
        (scan / 'metadata.json').write_text(json.dumps(
            {'scan': {'file_prefix': 'ir_', 'format': 'tif'}}))
        for cap in (0, 3):
            for cam in (0, 1):
                (scan / f'ir_{cam}_0_{cap}.tif').write_bytes(b'')
        self.commands.clear()
        argv = ['pgs-retexture', '-i', str(scan), '-r', str(recon),
                '--capture', '3', '--path', str(self.prefix),
                '--log-level', 'ERROR']
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch('pgs_recon.toolchain.run_command', self.fake), \
                mock.patch('atexit.register', lambda fn: fn), \
                mock.patch.object(retexture, 'convert_modality_images',
                                  return_value={(0, 0): 'ir_0_0_3.jpg',
                                                (1, 0): 'ir_1_0_3.jpg'}), \
                mock.patch.object(retexture, 'filter_sfm_for_cameras',
                                  side_effect=self._stub_filter), \
                mock.patch.object(retexture, 'ensure_ply_mesh',
                                  side_effect=self._stub_mesh):
            retexture._main()
        # The scene it built and the mesh it staged both sit in the recon's mvs/,
        # which is what lets TextureMesh reference them by basename. The stem
        # names the capture, so another capture of this scan cannot overwrite it.
        argv = self.argv_for('TextureMesh')
        self.assertEqual(str(recon / 'mvs'), flag(argv, '-w'))
        self.assertEqual('ir_c3_scene.mvs', flag(argv, '-i'))
        self.assertEqual('ir_c3.obj', flag(argv, '-o'))
        self.assertTrue((recon / 'mvs' / 'ir_c3.obj').is_file())

        # What the run resolved is recorded, but only what the user chose is
        # replayable. The camera set is derived from the capture, so pinning it
        # into the config would silently narrow a replay that overrides
        # --capture to a capture with more cameras.
        manifest = json.loads((recon / 'ir_c3_retexture.json').read_text())
        self.assertEqual(3, manifest['capture'])
        self.assertEqual([0, 1], manifest['cameras'])
        self.assertIsNone(manifest['parsed']['camera_index'])
        config = next(recon.glob('*_ir_c3_retexture_config.txt')).read_text()
        self.assertNotIn('camera-index', config)
        self.assertIn('capture = 3', config)

    @staticmethod
    def _stub_filter(sfm_json, prefix, modality_dir, key_to_name, out_json):
        out_json.write_text('{}')
        return 1

    @staticmethod
    def _stub_mesh(mesh_path, work_dir, out_stem=None):
        staged = Path(work_dir) / f'{out_stem}_input.ply'
        shutil.copy(mesh_path, staged)
        return staged


if __name__ == '__main__':
    unittest.main()
