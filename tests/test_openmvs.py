"""``openmvs``: the four wrappers are the binaries' complete flag surface.

ADR 0005 promises every flag is reachable and every omission deliberate. The
promise was never checkable, and had stopped being true -- ``mvs_refine`` exposed
six of nineteen options, and two of the missing thirteen turn off the CGAL remesh
a real reconstruction spent forty minutes inside.

:data:`SURFACES` tabulates each binary's flags at the pinned revision, and
:data:`NOT_MIRRORED` the deliberate omissions, so a flag upstream adds or we drop
is one failing test away from being noticed.

The table is keyword-argument *names*, because OpenMVS derives ``--max-face-area``
from ``max_face_area`` mechanically -- which means most of what follows compares
the wrappers against a list of the names they already have, and would not notice
upstream moving. :class:`TestSurfaceMatchesTheInstalledBinaries` is what closes
that gap, by parsing the binaries' own generated ``--help``. It needs the real
toolchain, so it skips everywhere the binaries are absent and runs in CI's
``test:in-image`` job. ``test_openmvg`` needs no equivalent: OpenMVG is a git
checkout, so its table is transcribed from the pinned source and is already an
independent record.
"""
import inspect
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pgs_recon import openmvs, toolchain
from pgs_recon.toolchain import (MVS_BIN, ArtifactsNotColocated, ToolNotFound,
                                resolve_exe)

from test_toolchain import ToolchainCase, flag, make_fake_prefix

#: Every flag each binary documents in its own options groups, as keyword-argument
#: names. Transcribed from ``apps/<App>/<App>.cpp`` at the pinned revision
#: (``ca991d5``, ``dependencies/cmake/BuildOpenMVS.cmake``). ``--archive-type``
#: and ``--max-threads`` are here as the two generic options that describe the
#: run; the rest of each "Generic options" group is in :data:`NOT_MIRRORED`.
SURFACES = {
    'mvs_densify': (
        'point_cloud', 'mask_path', 'view_neighbors_file',
        'output_view_neighbors_file', 'resolution_level', 'max_resolution',
        'min_resolution', 'sub_resolution_levels', 'number_views',
        'number_views_fuse', 'ignore_mask_label', 'iters', 'geometric_iters',
        'estimate_colors', 'estimate_normals', 'estimate_scale',
        'estimate_segmentation', 'sub_scene_area', 'sample_mesh',
        'fusion_mode', 'fusion_filter', 'fusion_depth_diff_threshold',
        'fusion_reprojection_threshold', 'postprocess_dmaps',
        'filter_point_cloud', 'export_number_views', 'roi_border',
        'estimate_roi', 'crop_to_roi', 'up_axis', 'remove_dmaps', 'tower_mode',
        'normalize_coordinates', 'archive_type', 'max_threads',
    ),
    'mvs_reconstruct': (
        'point_cloud', 'min_point_distance', 'integrate_only_roi',
        'constant_weight', 'free_space_support', 'thickness_factor',
        'quality_factor', 'decimate', 'target_face_num', 'remove_spurious',
        'remove_spikes', 'close_holes', 'smooth', 'edge_length', 'roi_border',
        'crop_to_roi', 'export_type', 'archive_type', 'max_threads',
    ),
    'mvs_refine': (
        'resolution_level', 'min_resolution', 'max_views', 'decimate',
        'close_holes', 'ensure_edge_size', 'max_face_area', 'scales',
        'scale_step', 'alternate_pair', 'regularity_weight',
        'rigidity_elasticity_ratio', 'gradient_step', 'planar_vertex_ratio',
        'reduce_memory', 'export_type', 'archive_type', 'max_threads',
    ),
    'mvs_texture': (
        'export_type', 'decimate', 'close_holes', 'resolution_level',
        'min_resolution', 'outlier_threshold', 'cost_smoothness_ratio',
        'virtual_face_images', 'global_seam_leveling', 'local_seam_leveling',
        'texture_size_multiple', 'empty_color', 'sharpness_weight',
        'orthographic_image_resolution', 'ignore_mask_label',
        'max_texture_size', 'archive_type', 'max_threads',
    ),
}

#: Flags deliberately absent from the wrappers, with the reason: these configure
#: the process rather than the reconstruction. Each app's undocumented "Hidden
#: options" group is excluded wholesale, as upstream excludes it from ``--help``.
NOT_MIRRORED = {
    'help': 'not a run parameter',
    'config_file': 'would let a config file contradict the recorded argv',
    'process_priority': 'a property of the host, not the reconstruction',
    'verbosity': 'the run log is ours to control, not a stage argument',
    'cuda_device': 'device selection belongs to the job, not the pipeline',
}

#: The positional artifacts each wrapper takes before its flags.
ARTIFACTS = {
    'mvs_densify': ('scene', 'output'),
    'mvs_reconstruct': ('scene', 'output'),
    'mvs_refine': ('scene', 'mesh', 'output'),
    'mvs_texture': ('scene', 'mesh', 'output'),
}

#: Which binary each wrapper resolves, for the fake prefix.
BINARIES = {
    'mvs_densify': 'DensifyPointCloud',
    'mvs_reconstruct': 'ReconstructMesh',
    'mvs_refine': 'RefineMesh',
    'mvs_texture': 'TextureMesh',
}

#: A value to pass for each flag whose argv spelling is not just ``str(value)``,
#: with what argv should then hold. A ``bool`` becomes ``0``/``1``, because
#: OpenMVS declares these as ``value<bool>`` rather than as presence switches;
#: ``free_space_support`` is the older spelling of the same idea and emits ``1``
#: only when true.
PROBES = {
    'crop_to_roi': (True, '1'),
    'remove_dmaps': (False, '0'),
    'remove_spikes': (False, '0'),
    'integrate_only_roi': (True, '1'),
    'constant_weight': (False, '0'),
    'free_space_support': (True, '1'),
    'export_type': ('obj', 'obj'),
}

#: Flags OpenMVS gives a single-letter name and this module passes as an
#: artifact rather than as an option. Asserted by name in their own tests
#: (basenames for ``-p``, absolute for ``-m``), so the sweep skips them.
SHORT_FLAGS = {'point_cloud': '-p', 'mask_path': '-m'}

#: Path-valued flags, probed with a real path so ``resolve()`` has something to
#: work on rather than an integer.
PATH_FLAGS = ('view_neighbors_file', 'output_view_neighbors_file')

#: Flags every invocation carries, per wrapper, and so the ones a
#: "``None`` omits it" sweep cannot assert against. ``archive_type`` is ADR
#: 0003's; the rest are this pipeline's own defaults, kept because changing what
#: a default run passes is a behaviour change and this module's job is not to
#: make one.
ALWAYS_EMITTED = {
    'mvs_densify': ('archive_type',),
    'mvs_reconstruct': ('archive_type', 'smooth'),
    'mvs_refine': ('archive_type', 'scales'),
    'mvs_texture': ('archive_type', 'max_texture_size', 'export_type'),
}

#: Wrapper argument -> the binary's own long flag, for the few that are not a
#: mechanical underscore-to-dash of the keyword. Only
#: :class:`TestSurfaceMatchesTheInstalledBinaries` needs these: everything else
#: in this file compares keyword names, which is exactly the gap that class
#: closes.
LONG_FLAGS = {
    'scene': 'input-file',
    'mesh': 'mesh-file',
    'output': 'output-file',
    'point_cloud': 'pointcloud-file',
}

#: Flags the wrapper supplies with no parameter of its own, and why. ``-w`` is
#: derived from the artifacts themselves (:func:`toolchain.work_dir`) rather than
#: taken as an argument, because it is also the frame the scene's *image* paths
#: resolve against -- see the module docstring of :mod:`pgs_recon.openmvs`.
DERIVED = {'working-folder': 'derived from the artifacts by toolchain.work_dir'}

#: One declaration line of ``boost::program_options`` help output, in either of
#: the two shapes it prints: ``--long-name arg (=default)`` or
#: ``-s [ --long-name ] arg``. Anchored so a wrapped description line, which is
#: indented past the flag column, cannot match.
HELP_FLAG = re.compile(r'^ {2}(?:-\S+ \[ )?--([a-z][a-z0-9-]*)')


def advertised_flags(binary: str) -> set:
    """Every long flag ``binary --help`` prints, across all of its groups.

    The generated help rather than a transcription, which is the whole point of
    the class that uses it. OpenMVS prints its banner and its options to stdout
    together; the parse keys on the declaration column, so the banner lines fall
    out on their own.
    """
    exe = resolve_exe(binary, MVS_BIN)
    with tempfile.TemporaryDirectory() as quiet:
        # Its own directory: these binaries write a log beside their cwd.
        out = subprocess.run([str(exe), '--help'], cwd=quiet,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True).stdout
    found = {m.group(1) for m in
             (HELP_FLAG.match(line) for line in out.splitlines()) if m}
    if not found:
        raise AssertionError(f'{binary} --help printed no recognisable option '
                             f'declarations; the parse, not the wrapper, is '
                             f'what broke:\n{out}')
    return found


class OpenMVSCase(ToolchainCase):
    """Each wrapper called over a fake prefix, with ``run_command`` captured.

    ``ToolchainCase`` supplies the resolved temp dir and the isolation of the two
    pieces of process-wide state (``configure()`` and ``$PGS_RECON_PREFIX``), so a
    developer with a prefix exported does not see different results than CI.
    """

    def setUp(self):
        super().setUp()
        toolchain.configure(
            prefix=make_fake_prefix(self.tmp / 'prefix', mvs=BINARIES.values()))
        # Co-located, because that is what every one of these binaries requires
        # of the artifacts it names by basename.
        self.work = self.tmp / 'mvs'
        self.work.mkdir()

    def call(self, name, **kwargs):
        """Invoke wrapper ``name`` and return the argv it built."""
        artifacts = {a: self.work / f'{a}.ply' for a in ARTIFACTS[name]}
        with mock.patch('pgs_recon.toolchain.run_command') as ran:
            getattr(openmvs, name)(**artifacts, **kwargs)
        return [str(a) for a in ran.call_args[0][0]]


class TestSurfaceIsComplete(OpenMVSCase):
    """Every documented flag is reachable, and every absence is on purpose."""

    def test_every_documented_flag_has_a_keyword_argument(self):
        for name, flags in SURFACES.items():
            params = inspect.signature(getattr(openmvs, name)).parameters
            for kwarg in flags:
                with self.subTest(wrapper=name, flag=kwarg):
                    self.assertIn(
                        kwarg, params,
                        f'{name} cannot reach --{kwarg.replace("_", "-")}; '
                        f'ADR 0005 says a wrapper is the binary\'s full surface')

    def test_no_keyword_argument_is_undeclared(self):
        """The table and the signature are edited together, or neither is
        authoritative."""
        for name, flags in SURFACES.items():
            params = inspect.signature(getattr(openmvs, name)).parameters
            extra = set(params) - set(flags) - set(ARTIFACTS[name])
            with self.subTest(wrapper=name):
                self.assertEqual(set(), extra,
                                 f'{name} takes arguments SURFACES does not '
                                 f'list; add them or explain them in '
                                 f'NOT_MIRRORED')

    def test_the_omissions_are_named(self):
        """A tabulated omission stays an omission: name it, or mirror it."""
        for name in SURFACES:
            params = inspect.signature(getattr(openmvs, name)).parameters
            for kwarg, reason in NOT_MIRRORED.items():
                with self.subTest(wrapper=name, flag=kwarg):
                    self.assertNotIn(kwarg, params, reason)


class TestFlagsReachArgv(OpenMVSCase):
    """A keyword argument is spelled in argv the way the binary declares it."""

    def test_each_flag_lands_with_its_value(self):
        for name, flags in SURFACES.items():
            for kwarg in flags:
                if kwarg in ARTIFACTS[name] or kwarg in SHORT_FLAGS:
                    continue
                if kwarg in PATH_FLAGS:
                    probe = self.tmp / f'{kwarg}.txt'
                    expected = str(probe)
                else:
                    probe, expected = PROBES.get(kwarg, (7, '7'))
                with self.subTest(wrapper=name, flag=kwarg):
                    argv = self.call(name, **{kwarg: probe})
                    spelled = f'--{kwarg.replace("_", "-")}'
                    self.assertIn(spelled, argv)
                    self.assertEqual(expected, flag(argv, spelled))

    def test_omitting_a_flag_omits_it(self):
        """``None`` means the binary's own default wins, not ``--flag None``."""
        for name, flags in SURFACES.items():
            argv = self.call(name)
            for kwarg in flags:
                spelled = f'--{kwarg.replace("_", "-")}'
                if kwarg in ALWAYS_EMITTED[name]:
                    continue
                with self.subTest(wrapper=name, flag=kwarg):
                    self.assertNotIn(spelled, argv)
            self.assertNotIn('None', argv)


class TestRefineRemeshKnobs(OpenMVSCase):
    """The two flags a real reconstruction needed and could not reach -- and

    which must stay unset by default, since a mesh that comes out different is not
    a rename.
    """

    def test_ensure_edge_size_zero_is_reachable(self):
        argv = self.call('mvs_refine', ensure_edge_size=0)
        self.assertEqual('0', flag(argv, '--ensure-edge-size'))

    def test_max_face_area_is_reachable(self):
        argv = self.call('mvs_refine', max_face_area=0)
        self.assertEqual('0', flag(argv, '--max-face-area'))

    def test_neither_is_emitted_by_default(self):
        """Exposing the knobs must not move them: upstream's defaults are what
        make the remesh run, and leaving all three unset keeps output identical."""
        argv = self.call('mvs_refine')
        for spelled in ('--ensure-edge-size', '--max-face-area', '--decimate'):
            self.assertNotIn(spelled, argv)


class TestAuxiliaryPathsGoAbsolute(OpenMVSCase):
    """A path that is not a basename-addressed artifact goes absolute, so
    ``MAKE_PATH_SAFE`` cannot re-root it against ``-w``."""

    def test_mask_path_is_normalised(self):
        # Spelled with a ``..`` rather than relatively, so the assertion is about
        # what the wrapper does and not about the test's working directory.
        argv = self.call('mvs_densify',
                         mask_path=self.work / '..' / 'masks')
        self.assertEqual(str(self.tmp / 'masks'), flag(argv, '-m'))

    def test_view_neighbors_files_are_normalised(self):
        argv = self.call('mvs_densify',
                         view_neighbors_file=self.work / '..' / 'in.txt',
                         output_view_neighbors_file=self.work / '..' / 'out.txt')
        self.assertEqual(str(self.tmp / 'in.txt'),
                         flag(argv, '--view-neighbors-file'))
        self.assertEqual(str(self.tmp / 'out.txt'),
                         flag(argv, '--output-view-neighbors-file'))

    def test_a_mask_folder_outside_the_working_directory_is_allowed(self):
        """``work_dir`` is not consulted for these, and must not be: a folder's
        ``parent`` is not the folder."""
        masks = self.tmp / 'elsewhere' / 'masks'
        argv = self.call('mvs_densify', mask_path=masks)
        self.assertEqual(str(masks), flag(argv, '-m'))
        self.assertEqual(str(self.work), flag(argv, '-w'))


class TestArtifactsStayBasenames(OpenMVSCase):
    """The surface got bigger; what ``-w`` means did not -- the directory
    holding the scene, being the frame its image paths were written against."""

    def test_scene_and_output_are_basenames(self):
        for name in SURFACES:
            with self.subTest(wrapper=name):
                argv = self.call(name)
                self.assertEqual('scene.ply', flag(argv, '-i'))
                self.assertEqual('output.ply', flag(argv, '-o'))
                self.assertEqual(str(self.work), flag(argv, '-w'))

    def test_the_dense_cloud_is_a_basename(self):
        for name in ('mvs_densify', 'mvs_reconstruct'):
            with self.subTest(wrapper=name):
                argv = self.call(name, point_cloud=self.work / 'dense.ply')
                self.assertEqual('dense.ply', flag(argv, '-p'))

    def test_a_cloud_from_elsewhere_is_refused(self):
        """Still refused, but as our invariant now rather than an assumption:
        ``MAKE_PATH_SAFE`` would accept ``../elsewhere/dense.ply``; ``-w`` is what
        does not relax. See ADR 0005."""
        elsewhere = self.tmp / 'elsewhere'
        elsewhere.mkdir()
        with self.assertRaises(ArtifactsNotColocated):
            self.call('mvs_reconstruct', point_cloud=elsewhere / 'dense.ply')


class TestArchiveTypeSurvivedTheRewrite(OpenMVSCase):
    """ADR 0003's flag is still always emitted, on all four stages."""

    def test_archive_type_is_always_minus_one(self):
        for name in SURFACES:
            with self.subTest(wrapper=name):
                self.assertEqual('-1', flag(self.call(name), '--archive-type'))

    def test_it_is_overridable(self):
        argv = self.call('mvs_densify', archive_type=2)
        self.assertEqual('2', flag(argv, '--archive-type'))


class TestResolvedBinaries(OpenMVSCase):
    """argv[0] is the absolute path to the right binary under MVS_BIN."""

    def test_each_wrapper_runs_its_own_binary(self):
        for name, binary in BINARIES.items():
            with self.subTest(wrapper=name):
                argv = self.call(name)
                self.assertEqual(binary, Path(argv[0]).name)
                self.assertIn(MVS_BIN, argv[0])


class TestSurfaceMatchesTheInstalledBinaries(unittest.TestCase):
    """:data:`SURFACES` checked against the binaries instead of against itself.

    Every other class here compares a wrapper's signature to a list of keyword
    names, and the flag is derived from the keyword mechanically -- so the
    comparison is the table against itself. That catches a parameter someone
    drops; it cannot notice a flag upstream *adds*, or one upstream renames,
    which is precisely the drift ADR 0005 is worried about. Unlike
    ``test_openmvg``, whose table is transcribed from the pinned OpenMVG source
    and so is an independent record, OpenMVS is fetched as a tarball with no
    source tree to read -- the binaries' own generated help is the only authority
    available, and it is a better one than a transcription anyway.

    So this needs the real toolchain, and skips wherever it is absent. That is
    what ``test:in-image`` is for: it is the one CI job where these binaries
    exist, and the only place this class actually runs.

    Deliberately *not* a :class:`OpenMVSCase`: that fixture points the prefix at
    a directory of empty stand-ins, which is the opposite of what is wanted here.
    """

    @classmethod
    def setUpClass(cls):
        """Parse each binary's help once, or skip the class as a whole.

        Skipping here rather than per assertion keeps the absence of a toolchain
        one line of output instead of one per wrapper per test, and it is the
        honest report: without the binaries there is no partial version of this
        check to run.
        """
        cls.advertised = {}
        for wrapper, binary in BINARIES.items():
            try:
                cls.advertised[wrapper] = advertised_flags(binary)
            except ToolNotFound as absent:
                raise unittest.SkipTest(
                    f'{binary} is not installed under this prefix, so there is '
                    f'nothing to compare against: {absent}')

    @staticmethod
    def spelled(kwargs) -> set:
        """Wrapper keyword names as the long flags they reach argv as."""
        return {LONG_FLAGS.get(k, k.replace('_', '-')) for k in kwargs}

    def offered(self, wrapper: str) -> set:
        """The flags the wrapper actually passes: its options and its
        artifacts. ``NOT_MIRRORED`` is not here -- those are the ones it
        deliberately never passes."""
        return self.spelled(set(SURFACES[wrapper]) | set(ARTIFACTS[wrapper]))

    def test_no_flag_the_binary_advertises_is_unaccounted_for(self):
        """The half a transcription cannot check: upstream adding an option.

        Accounted for means mirrored, named in ``NOT_MIRRORED``, or derived --
        the point being that a new upstream flag is none of those until someone
        decides which it is.
        """
        for wrapper in SURFACES:
            with self.subTest(wrapper=wrapper):
                accounted = (self.offered(wrapper) | self.spelled(NOT_MIRRORED)
                             | set(DERIVED))
                unreachable = sorted(self.advertised[wrapper] - accounted)
                self.assertEqual(
                    [], unreachable,
                    f'{BINARIES[wrapper]} advertises flags {wrapper} cannot '
                    f'reach; add them to the wrapper and to SURFACES, or name '
                    f'them in NOT_MIRRORED')

    def test_no_flag_the_wrapper_offers_has_gone_away(self):
        """And the other half: upstream removing or renaming one, which reaches
        the binary as an unrecognised option and kills the stage.

        Only what the wrapper passes is checked. ``NOT_MIRRORED`` is exempt
        because an omission stays correct however upstream spells it, and
        ``--cuda-device`` really is absent from a non-CUDA build.
        """
        for wrapper in SURFACES:
            with self.subTest(wrapper=wrapper):
                gone = sorted(self.offered(wrapper)
                              - self.advertised[wrapper])
                self.assertEqual(
                    [], gone,
                    f'{wrapper} passes flags {BINARIES[wrapper]} no longer '
                    f'advertises')


if __name__ == '__main__':
    unittest.main()
