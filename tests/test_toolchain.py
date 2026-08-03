"""``toolchain``: which binary gets run, where it runs, and what gets recorded.

Five invariants worth the test file. Resolution is **late** -- it reads the
configuration at call time, so a ``configure()`` after import still takes effect
and no filesystem lookup happens on the import path (CI runs this suite before it
builds ``dependencies/``, so the module has to import on a machine with no
OpenMVG). It is **tiered** -- argument over ``configure()`` over environment over
default -- and a failure says *which* tier chose the prefix, because "not found
at /usr/local/bin/X" is only actionable once you know whether /usr/local was
asked for or merely assumed. Whichever tier wins, the prefix is made
**absolute** on the way in, because the two stages that run with ``cwd`` set
would otherwise re-interpret a relative argv[0] against the wrong directory.
Files a binary resolves **against a directory** are handled where the argv is
built rather than left to fail inside it -- refused when they genuinely must be
co-located, translated when the binary's own join is more forgiving. And
:func:`~pgs_recon.toolchain.run` **records before it executes**, so a binary that
dies still leaves behind the invocation that killed it.

The prefixes here are real directories with real executable bits, so
``resolve_exe`` is exercised rather than mocked; :func:`make_fake_prefix` is the
seed of the end-to-end fixture MR2 needs. Nothing here spawns a process:
``run_command`` is patched at the module global, which is the seam every argv
assertion downstream will use.
"""
import os
import shutil
import tempfile
import unittest
from enum import IntEnum
from pathlib import Path
from unittest import mock

from pgs_recon import toolchain
from pgs_recon.toolchain import (MVG_BIN, MVS_BIN, ArtifactsNotColocated,
                                Recorder, ToolNotFound, cam_db, configure,
                                effective_prefix, relative_to_dir, resolve_exe,
                                run, using, work_dir)
from pgs_recon.utility import ToolFailed


#: A tool name no install prefix contains, for the not-found paths.
#:
#: Load-bearing: the tiers fall through to ``/usr/local``, so a not-found test
#: naming a *real* tool passes only where OpenMVG is absent. That is true of CI
#: (which runs this suite before it builds ``dependencies/``) and false inside
#: our own Docker image, where these tests failed while CI stayed green.
ABSENT_TOOL = 'openMVG_main_NoSuchTool'


def make_fake_prefix(root: Path, mvg=(), mvs=()) -> Path:
    """Build an install prefix holding empty but executable stand-ins.

    Real files with the executable bit set, because that is exactly what
    ``resolve_exe`` inspects; a mocked ``Path.is_file`` would test the mock.
    """
    for subdir, names in ((MVG_BIN, mvg), (MVS_BIN, mvs)):
        target = root / subdir
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            exe = target / name
            exe.touch()
            exe.chmod(0o755)
    return root


def flag(argv, name: str):
    """The value following ``name`` in ``argv``, or None if it is absent.

    The argv-inspection convention every wrapper test shares, so it is defined
    beside :func:`make_fake_prefix` rather than once per module. ``argv`` is
    stringified on the way in: ``toolchain.run`` has already done that by the time
    the ``run_command`` seam sees it, but a caller inspecting a half-built command
    has not.
    """
    argv = [str(a) for a in argv]
    return argv[argv.index(name) + 1] if name in argv else None


class ToolchainCase(unittest.TestCase):
    """Isolates the process-wide configuration and the environment.

    Both are global state: a leaked ``configure()`` would make these tests order
    dependent, and a developer with ``$PGS_RECON_PREFIX`` set would otherwise see
    different results than CI.
    """

    def setUp(self):
        env = mock.patch.dict(os.environ, {})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(toolchain.PREFIX_ENV, None)

        configure(prefix=None, recorder=None)
        self.addCleanup(configure, prefix=None, recorder=None)

        # Resolved: on macOS mkdtemp hands back a path under /var, which is a
        # symlink to /private/var. Since a prefix is absolutised on the way in,
        # an unresolved fixture would not compare equal to what comes back out.
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.tmp)


class TestResolutionTiers(ToolchainCase):

    def test_the_default_is_the_container_layout(self):
        self.assertEqual(Path('/usr/local'), effective_prefix())

    def test_environment_beats_the_default(self):
        os.environ[toolchain.PREFIX_ENV] = str(self.tmp)
        self.assertEqual(self.tmp, effective_prefix())

    def test_configure_beats_the_environment(self):
        os.environ[toolchain.PREFIX_ENV] = '/env/prefix'
        configure(prefix=self.tmp)
        self.assertEqual(self.tmp, effective_prefix())

    def test_the_argument_beats_configure(self):
        configure(prefix='/configured')
        found = make_fake_prefix(self.tmp, mvg=['openMVG_main_SfM'])
        self.assertEqual(found / 'bin/openMVG_main_SfM',
                         resolve_exe('openMVG_main_SfM', prefix=found))

    def test_an_empty_environment_variable_is_not_a_prefix(self):
        os.environ[toolchain.PREFIX_ENV] = ''
        self.assertEqual(Path('/usr/local'), effective_prefix())

    def test_configure_none_clears_back_to_the_environment(self):
        os.environ[toolchain.PREFIX_ENV] = str(self.tmp)
        configure(prefix='/configured')
        configure(prefix=None)
        self.assertEqual(self.tmp, effective_prefix())

    def test_configure_leaves_unmentioned_settings_alone(self):
        recorder = Recorder({})
        configure(prefix=self.tmp, recorder=recorder)
        configure(prefix='/elsewhere')
        self.assertIs(recorder, toolchain._recorder)

    def test_a_string_prefix_is_accepted(self):
        configure(prefix=str(self.tmp))
        self.assertEqual(self.tmp, effective_prefix())


class TestAbsolutePrefixes(ToolchainCase):
    """A prefix is absolutised as it is accepted, from whichever tier.

    Not cosmetic. ``mvg_to_mvs`` and ``mvs_texture`` run with ``cwd`` set to
    ``mvs/``, and ``subprocess`` resolves a relative argv[0] against ``cwd``
    rather than against the directory it was launched from -- so a relative
    prefix would satisfy ``resolve_exe``'s check and then fail to exec,
    reporting a tool that had just been confirmed to exist as missing.
    """

    def _in_tmp(self):
        """Run the rest of the test from inside the fixture directory."""
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.tmp)

    def test_a_relative_prefix_resolves_against_the_launch_directory(self):
        make_fake_prefix(self.tmp / 'installed', mvg=['openMVG_main_SfM'])
        self._in_tmp()
        exe = resolve_exe('openMVG_main_SfM', prefix='installed')
        self.assertTrue(exe.is_absolute(), exe)
        self.assertEqual(self.tmp / 'installed/bin/openMVG_main_SfM', exe)

    def test_a_relative_prefix_from_configure_is_absolutised(self):
        self._in_tmp()
        configure(prefix='installed')
        self.assertEqual(self.tmp / 'installed', effective_prefix())

    def test_a_relative_prefix_from_the_environment_is_absolutised(self):
        self._in_tmp()
        os.environ[toolchain.PREFIX_ENV] = 'installed'
        self.assertEqual(self.tmp / 'installed', effective_prefix())

    def test_configure_absolutises_once_not_at_every_lookup(self):
        # A stage that changes directory must not change where the toolchain is.
        self._in_tmp()
        configure(prefix='installed')
        os.chdir(self.tmp.parent)
        self.assertEqual(self.tmp / 'installed', effective_prefix())

    def test_a_tilde_is_expanded(self):
        # A shell expands ~ before --path ever sees it; $PGS_RECON_PREFIX can
        # reach us with a literal tilde, which would otherwise be searched for
        # as a directory named '~'.
        os.environ['HOME'] = str(self.tmp)
        os.environ[toolchain.PREFIX_ENV] = '~/installed'
        self.assertEqual(self.tmp / 'installed', effective_prefix())

    def test_the_default_needs_no_absolutising(self):
        # Left as the literal constant, so the default tier's error messages
        # stay predictable on a host where /usr/local is itself a symlink.
        self.assertEqual(Path('/usr/local'), effective_prefix())


class TestResolveExe(ToolchainCase):

    def test_the_two_tool_families_live_in_different_directories(self):
        make_fake_prefix(self.tmp, mvg=['openMVG_main_SfM', 'pgs-global-scaler'],
                         mvs=['DensifyPointCloud'])
        configure(prefix=self.tmp)
        self.assertEqual(self.tmp / 'bin/openMVG_main_SfM',
                         resolve_exe('openMVG_main_SfM'))
        self.assertEqual(self.tmp / 'bin/pgs-global-scaler',
                         resolve_exe('pgs-global-scaler'))
        self.assertEqual(self.tmp / 'bin/OpenMVS/DensifyPointCloud',
                         resolve_exe('DensifyPointCloud', MVS_BIN))

    def test_resolution_is_late_bound(self):
        # The whole reason configuration is process-wide rather than defaulted
        # into signatures: this import already happened, and configure() still
        # decides where the tool comes from.
        make_fake_prefix(self.tmp, mvs=[ABSENT_TOOL])
        with self.assertRaises(ToolNotFound):
            resolve_exe(ABSENT_TOOL, MVS_BIN)
        configure(prefix=self.tmp)
        self.assertEqual(self.tmp / 'bin/OpenMVS' / ABSENT_TOOL,
                         resolve_exe(ABSENT_TOOL, MVS_BIN))

    def test_a_missing_tool_names_the_tier_that_chose_the_prefix(self):
        with self.assertRaises(ToolNotFound) as ctx:
            resolve_exe(ABSENT_TOOL)
        message = str(ctx.exception)
        self.assertIn(f'/usr/local/bin/{ABSENT_TOOL}', message)
        self.assertIn('the built-in default', message)
        self.assertEqual('the built-in default', ctx.exception.tier)

    def test_each_tier_is_named_in_its_own_words(self):
        os.environ[toolchain.PREFIX_ENV] = '/from/env'
        with self.assertRaises(ToolNotFound) as ctx:
            resolve_exe('openMVG_main_SfM')
        self.assertIn(f'${toolchain.PREFIX_ENV}', str(ctx.exception))

        configure(prefix='/from/configure')
        with self.assertRaises(ToolNotFound) as ctx:
            resolve_exe('openMVG_main_SfM')
        self.assertIn('configure(prefix=...)', str(ctx.exception))

        with self.assertRaises(ToolNotFound) as ctx:
            resolve_exe('openMVG_main_SfM', prefix='/from/argument')
        self.assertIn('the prefix argument', str(ctx.exception))
        self.assertEqual(Path('/from/argument/bin/openMVG_main_SfM'),
                         ctx.exception.expected)

    def test_a_missing_tool_reads_as_127_to_the_shell(self):
        # A ToolFailed on purpose, so every main()'s existing handler covers it,
        # and 127 because that is what a shell reports for a command it could
        # not find.
        with self.assertRaises(ToolFailed) as ctx:
            resolve_exe(ABSENT_TOOL)
        self.assertEqual(127, ctx.exception.exit_code)
        self.assertIsNone(ctx.exception.returncode)

    def test_a_directory_is_not_a_tool(self):
        (self.tmp / 'bin/DensifyPointCloud').mkdir(parents=True)
        configure(prefix=self.tmp)
        with self.assertRaises(ToolNotFound):
            resolve_exe('DensifyPointCloud')

    def test_a_file_without_the_executable_bit_says_so(self):
        # The symptom of a half-finished install, or of a build artifact copied
        # without its mode. Distinguished because the fix is different.
        (self.tmp / 'bin').mkdir(parents=True)
        (self.tmp / 'bin/openMVG_main_SfM').touch()
        configure(prefix=self.tmp)
        with self.assertRaises(ToolNotFound) as ctx:
            resolve_exe('openMVG_main_SfM')
        self.assertIn('not executable', str(ctx.exception))


class TestNothingResolvesAtImport(unittest.TestCase):

    def test_importing_the_module_touches_no_filesystem(self):
        # CI runs the test suite before it builds dependencies/, so a module-level
        # discovery would break the build on a machine with no OpenMVG -- and it
        # would also freeze the prefix before main() could configure one.
        import importlib.util

        def forbidden(*_args, **_kwargs):
            raise AssertionError('resolution ran at import time')

        # Executed as a private copy rather than reloaded: a reload would swap
        # the live module's classes out from under every other test in this file.
        spec = importlib.util.spec_from_file_location('_toolchain_reimport',
                                                     toolchain.__file__)
        fresh = importlib.util.module_from_spec(spec)
        with mock.patch.object(Path, 'is_file', forbidden), \
                mock.patch.object(os, 'access', forbidden):
            spec.loader.exec_module(fresh)
        self.assertIsNone(fresh._prefix)
        self.assertIsNone(fresh._recorder)


class TestCamDb(ToolchainCase):

    def test_prefix_relative_by_default(self):
        configure(prefix=self.tmp)
        self.assertEqual(
            self.tmp / 'lib/openMVG/sensor_width_camera_database.txt',
            cam_db())

    def test_an_explicit_path_wins(self):
        configure(prefix=self.tmp)
        self.assertEqual(Path('/data/camdb.txt'), cam_db('/data/camdb.txt'))

    def test_an_explicit_path_is_made_absolute(self):
        # The one importer that passes this to a binary passes it as an
        # argument, so it has to mean the same thing from any working directory.
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.tmp)
        self.assertEqual(self.tmp / 'calib/camdb.txt',
                         cam_db('calib/camdb.txt'))

    def test_existence_is_not_checked(self):
        # Unlike resolve_exe: whoever opens it reports a missing file with more
        # context than a check here could, and only one importer needs it at all.
        configure(prefix='/nowhere')
        self.assertEqual(
            Path('/nowhere/lib/openMVG/sensor_width_camera_database.txt'),
            cam_db())


class TestDirectoryRelativeAddressing(ToolchainCase):
    """``work_dir``/``relative_to_dir``: binaries that resolve names against a dir.

    Two different constraints, deliberately handled differently. Every OpenMVS
    stage addresses its scene and geometry as bare filenames against ``-w``, and
    densify pairs its dense cloud with its scene *by name*, so those artifacts
    genuinely have to be co-located -- ``work_dir`` refuses when they are not,
    raising ``ArtifactsNotColocated`` (a ``ToolFailed``, so every ``main()``'s
    existing handler reports it rather than a traceback).

    ``openMVG_main_SfM``'s ``-M`` is not that: it is joined onto ``-m``, and the
    join resolves ``..``, so any location is reachable given the right spelling.
    ``relative_to_dir`` produces that spelling instead of restricting the input.
    """

    def test_the_shared_directory_is_returned(self):
        mvs = self.tmp / 'mvs'
        self.assertEqual(mvs, work_dir(mvs / 'scene.mvs', mvs / 'mesh.ply'))

    def test_a_none_artifact_is_ignored(self):
        # So an optional input (the dense cloud) needs no special case.
        mvs = self.tmp / 'mvs'
        self.assertEqual(mvs, work_dir(mvs / 'scene.mvs', None))

    def test_artifacts_in_two_directories_are_refused(self):
        with self.assertRaises(ArtifactsNotColocated) as ctx:
            work_dir(self.tmp / 'mvs' / 'scene.mvs',
                     self.tmp / 'elsewhere' / 'mesh.ply')
        # The message has to name both, since which one is misplaced is the
        # question the reader has.
        self.assertIn('scene.mvs', str(ctx.exception))
        self.assertIn('elsewhere', str(ctx.exception))

    def test_a_refusal_is_a_tool_failure_the_apps_already_handle(self):
        with self.assertRaises(ToolFailed) as ctx:
            work_dir(self.tmp / 'a' / 'scene.mvs', self.tmp / 'b' / 'mesh.ply')
        # Not the child's status and not 127: nothing was spawned.
        self.assertEqual(1, ctx.exception.exit_code)

    def test_no_artifacts_at_all_is_refused_rather_than_guessed(self):
        with self.assertRaises(ArtifactsNotColocated):
            work_dir(None)

    def test_relative_and_absolute_spellings_of_one_directory_agree(self):
        # ``.resolve()`` is what makes this hold; on macOS the temp dir is also
        # reached through a /var -> /private/var symlink.
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.tmp)
        self.assertEqual(self.tmp, work_dir('scene.mvs', self.tmp / 'mesh.ply'))

    def test_a_file_in_the_directory_is_spelled_as_its_basename(self):
        regions = self.tmp / 'matches_dir'
        self.assertEqual('matches_filtered.bin',
                         relative_to_dir(regions / 'matches_filtered.bin',
                                         regions))

    def test_a_file_elsewhere_is_spelled_relative_to_the_directory(self):
        # openMVG_main_SfM joins -M onto -m, and that join resolves `..`, so a
        # matches file outside the regions directory is reachable. Verified
        # against stlplus create_filespec, which is what does the joining.
        regions = self.tmp / 'mvg' / 'matches_dir'
        self.assertEqual('../other/matches.bin',
                         relative_to_dir(self.tmp / 'mvg' / 'other' / 'matches.bin',
                                         regions))

    def test_a_file_in_a_subdirectory_keeps_its_subdirectory(self):
        regions = self.tmp / 'matches_dir'
        self.assertEqual('nested/matches.bin',
                         relative_to_dir(regions / 'nested' / 'matches.bin',
                                         regions))

    def test_the_result_is_never_absolute(self):
        # The one spelling the binary cannot use: stlplus concatenates, so an
        # absolute -M becomes `/regions//abs/path`. Translating rather than
        # passing through is what makes that unreachable.
        for artifact in (self.tmp / 'matches_dir' / 'm.bin',
                         Path('/var/tmp/m.bin'), Path('m.bin')):
            spelled = relative_to_dir(artifact, self.tmp / 'matches_dir')
            self.assertFalse(Path(spelled).is_absolute(), spelled)

    def test_mixed_relative_and_absolute_spellings_still_work(self):
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.tmp)
        (self.tmp / 'matches_dir').mkdir()
        self.assertEqual('m.bin',
                         relative_to_dir('matches_dir/m.bin',
                                         self.tmp / 'matches_dir'))


class TestUsing(ToolchainCase):

    def test_the_override_applies_inside_and_is_restored_after(self):
        configure(prefix='/outer')
        with using(prefix='/inner'):
            self.assertEqual(Path('/inner'), effective_prefix())
        self.assertEqual(Path('/outer'), effective_prefix())

    def test_restoration_survives_an_exception(self):
        configure(prefix='/outer')
        with self.assertRaises(ToolNotFound):
            with using(prefix='/inner'):
                resolve_exe('openMVG_main_SfM')
        self.assertEqual(Path('/outer'), effective_prefix())

    def test_an_unmentioned_setting_is_not_disturbed(self):
        recorder = Recorder({})
        configure(prefix='/outer', recorder=recorder)
        with using(prefix='/inner'):
            self.assertIs(recorder, toolchain._recorder)
        self.assertIs(recorder, toolchain._recorder)


class Lying(int):
    """An ``int`` subclass whose ``str()`` is not its value.

    The narrowing is invisible to a value assertion on a modern interpreter --
    ``str()`` of an ``IntEnum`` is already its value from Python 3.11 on -- so an
    enum cannot prove the narrowing happened here. This can: without the
    ``int()``, ``str()`` returns ``'BAD'`` on every Python, so the test fails on
    the runner in front of you rather than only on a 3.9 one.
    """

    def __str__(self):
        return 'BAD'


class Method(IntEnum):
    L2 = 2


class TestArgvNarrowing(ToolchainCase):
    """``run`` is the one place argv becomes strings, so it is the one place an
    ``int`` subclass is narrowed. Everything the wrappers hand over -- flags from
    a ``_optional`` helper and flags they assemble by hand -- goes through it."""

    def setUp(self):
        super().setUp()
        patch = mock.patch('pgs_recon.toolchain.run_command')
        self.run_command = patch.start()
        self.addCleanup(patch.stop)

    def argv(self, *command):
        run(list(command))
        return self.run_command.call_args[0][0]

    def test_an_int_subclass_is_narrowed_before_stringifying(self):
        self.assertEqual(['-x', '5'], self.argv('-x', Lying(5)))

    def test_an_enum_reaches_argv_as_its_value(self):
        """What OpenMVG parses. On 3.9/3.10 an unnarrowed member would arrive as
        ``'Method.L2'``."""
        self.assertEqual(['-R', '2'], self.argv('-R', Method.L2))

    def test_a_bool_becomes_zero_or_one(self):
        """The binaries declare these as options over a bool, not as presence
        switches, so ``'True'`` would not parse."""
        self.assertEqual(['-u', '1', '-f', '0'],
                         self.argv('-u', True, '-f', False))

    def test_a_float_is_left_alone(self):
        """The narrowing keys on ``int``, so a ratio must not be truncated."""
        self.assertEqual(['-r', '0.8'], self.argv('-r', 0.8))

    def test_a_hand_built_flag_is_narrowed_too(self):
        """The reason this lives at the chokepoint: ``--archive-type`` and
        ``--max-texture-size`` never pass through a ``_optional`` helper."""
        self.assertEqual(['--archive-type', '-1'],
                         self.argv('--archive-type', Lying(-1)))


class TestRun(ToolchainCase):

    def setUp(self):
        super().setUp()
        patch = mock.patch('pgs_recon.toolchain.run_command')
        self.run_command = patch.start()
        self.addCleanup(patch.stop)
        self.commands = {}
        configure(recorder=Recorder(self.commands))

    def test_the_argv_is_passed_through_as_strings(self):
        # Wrappers hand over Paths; subprocess would take them, but the recorded
        # line has to be joinable and argv assertions have to compare like with
        # like.
        run(['DensifyPointCloud', '-i', Path('mvs/scene.mvs'), '--archive-type',
             -1])
        self.run_command.assert_called_once_with(
            ['DensifyPointCloud', '-i', 'mvs/scene.mvs', '--archive-type', '-1'],
            cwd=None)

    def test_the_working_directory_is_forwarded(self):
        run(['TextureMesh'], cwd=self.tmp)
        self.assertEqual(self.tmp, self.run_command.call_args.kwargs['cwd'])

    def test_running_records_the_command(self):
        run(['openMVG_main_SfM', '-i', 'sfm_data.json'])
        self.assertEqual(['openMVG_main_SfM -i sfm_data.json'],
                         list(self.commands.values()))

    def test_a_failing_command_is_recorded_anyway(self):
        # Recorded before it runs, so the manifest names the invocation that
        # killed the run -- which is the whole point of keeping the log.
        self.run_command.side_effect = ToolFailed(['RefineMesh'], -9)
        with self.assertRaises(ToolFailed):
            run(['RefineMesh', '-i', 'scene.mvs'])
        self.assertEqual(['RefineMesh -i scene.mvs'],
                         list(self.commands.values()))

    def test_an_unconfigured_recorder_still_runs(self):
        configure(recorder=None)
        run(['openMVG_main_SfM'])
        self.run_command.assert_called_once()
        self.assertEqual({}, self.commands)

    def test_an_explicit_recorder_overrides_the_configured_one(self):
        elsewhere = {}
        run(['openMVG_main_SfM'], recorder=Recorder(elsewhere))
        self.assertEqual({}, self.commands)
        self.assertEqual(['openMVG_main_SfM'], list(elsewhere.values()))


class TestRecorder(unittest.TestCase):
    """The manifest's ``commands``: timestamp -> one line.

    A compatibility surface, not an internal detail --
    ``recon_dir._sfm_from_commands`` greps the values for ``openMVG2openMVS`` and
    reads the token after ``-i`` when resolving a legacy manifest.
    """

    def test_a_command_is_recorded_as_a_rerunnable_line(self):
        commands = {}
        Recorder(commands).command(
            ['openMVG_main_openMVG2openMVS', '-i', Path('/o/mvg/sfm_data.bin')])
        line, = commands.values()
        self.assertEqual('openMVG_main_openMVG2openMVS -i /o/mvg/sfm_data.bin',
                         line)

    def test_a_python_step_is_recorded_as_a_call(self):
        # The two importers are not binary invocations, but a run that cannot say
        # how its scene was imported is no more reproducible for it.
        commands = {}
        Recorder(commands).step('init_sfm_generic2', scan_dir=Path('/in'),
                                sfm_file=Path('/o/mvg/sfm_data.json'))
        self.assertEqual(['init_sfm_generic2(scan_dir=/in, '
                          'sfm_file=/o/mvg/sfm_data.json)'],
                         list(commands.values()))

    def test_positional_and_keyword_arguments_render_in_order(self):
        commands = {}
        Recorder(commands).step('load_cam_calib', Path('/in/calib.json'))
        self.assertEqual(['load_cam_calib(/in/calib.json)'],
                         list(commands.values()))

    def test_a_note_is_recorded_verbatim(self):
        commands = {}
        Recorder(commands).note('Processing complete')
        self.assertEqual(['Processing complete'], list(commands.values()))

    def test_entries_accumulate_in_order(self):
        commands = {}
        recorder = Recorder(commands)
        recorder.command(['first'])
        recorder.command(['second'])
        self.assertEqual(['first', 'second'], list(commands.values()))

    def test_it_appends_to_an_existing_log(self):
        # Commands accumulate across the jobs of a staged run, so a Recorder is
        # handed the manifest's existing dict rather than a fresh one.
        commands = {'01/01/1970, 00:00:00.000000 UTC': 'from an earlier job'}
        Recorder(commands).command(['openMVG_main_SfM'])
        self.assertEqual(2, len(commands))
        self.assertIn('from an earlier job', commands.values())

    def test_a_timestamp_collision_does_not_lose_an_entry(self):
        # Microsecond stamps make this vanishingly rare rather than impossible,
        # and losing a record silently is worse than an odd-looking key. Nothing
        # reads the keys: consumers walk the values.
        commands = {}
        with mock.patch('pgs_recon.toolchain.current_timestamp',
                        return_value='the same instant'):
            recorder = Recorder(commands)
            recorder.command(['first'])
            recorder.command(['second'])
        self.assertEqual(['first', 'second'], list(commands.values()))


if __name__ == '__main__':
    unittest.main()
