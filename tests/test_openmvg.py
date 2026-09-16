"""``openmvg``: the wrappers are the binaries' complete registered flag surface.

ADR 0005 promises every flag is reachable and every omission deliberate.
`test_openmvs` made that checkable for the MVS half; this is the other half, and
the same audit found the same kind of gap -- ``mvg_sfm`` reached eight of
``openMVG_main_SfM``'s eighteen options, and the ten it missed include *every*
tuning knob of the GLOBAL and STELLAR engines, both of which ``pgs-recon
--mvg-recon-method`` will happily select.

:data:`SURFACES` tabulates each binary's options at the pinned revision and
:data:`NOT_MIRRORED` the deliberate omissions, so a flag upstream adds or we drop
is one failing test away from being noticed.

The table carries the argv spelling as well as the keyword name, which the MVS
one does not have to: OpenMVS derives ``--max-face-area`` from ``max_face_area``
mechanically, while OpenMVG's flags are per-binary (the same letter means
different things in different apps) and its long spellings are a mix of
conventions. Transcribed from the ``cmd.add`` calls rather than from the usage
text each binary prints, because the two disagree and only the registrations
decide what parses.
"""
import inspect
import unittest
from pathlib import Path
from unittest import mock

from pgs_recon import openmvg, toolchain
from pgs_recon.toolchain import MVG_BIN

from test_toolchain import ToolchainCase, flag, make_fake_prefix

#: Every option each binary registers, as ``{keyword argument: argv flag}``.
#: Transcribed from the ``cmd.add`` calls at the pinned revision (``c92ed1b``,
#: ``dependencies/cmake/BuildOpenMVG.cmake``); ``mvg_autoscale``'s come from
#: ``dependencies/utilities/src/global_scaler.cpp``, which is in this tree. The
#: required arguments are here too -- unlike OpenMVS, every one of these is a
#: flag rather than a basename positioned against a working directory, so there
#: is no second category and the table can be total.
SURFACES = {
    # openMVG_main_SfMInit_ImageListing
    'init_sfm_generic': {
        'images': '-i', 'cam_db': '-d', 'output_dir': '-o',
        'focal_length': '-f', 'intrinsics': '-k', 'camera_model': '-c',
        'group_camera_model': '-g', 'use_pose_prior': '-P',
        'prior_weights': '-W', 'gps_to_xyz_method': '-m',
    },
    # openMVG_main_ComputeFeatures
    'compute_features': {
        'sfm': '-i', 'output_dir': '-o', 'method': '-m', 'upright': '-u',
        'force': '-f', 'preset': '-p', 'threads': '-n',
    },
    # openMVG_main_ComputeMatches
    'compute_matches': {
        'sfm': '-i', 'output': '-o', 'pairs_file': '-p', 'ratio': '-r',
        'method': '-n', 'force': '-f', 'cache_size': '-c',
        'preemptive_feature_count': '-P',
    },
    # openMVG_main_GeometricFilter
    'geometric_filter': {
        'sfm': '-i', 'output': '-o', 'matches': '-m', 'pairs_file': '-p',
        'output_pairs': '-s', 'model': '-g', 'force': '-f',
        'guided_matching': '-r', 'max_iteration': '-I', 'cache_size': '-c',
    },
    # openMVG_main_SfM
    'mvg_sfm': {
        'sfm': '-i', 'features_dir': '-m', 'matches': '-M',
        'output_dir': '-o', 'engine': '-s', 'refine_intrinsics': '-f',
        'refine_extrinsics': '-e', 'use_priors': '-P',
        'triangulation_method': '-t', 'resection_method': '-r',
        'camera_model': '-c', 'initializer': '-S', 'initial_pair_a': '-a',
        'initial_pair_b': '-b', 'rotation_averaging': '-R',
        'translation_averaging': '-T', 'graph_simplification': '-G',
        'graph_simplification_value': '-g',
    },
    # openMVG_main_ComputeStructureFromKnownPoses
    'mvg_compute_known': {
        'sfm': '-i', 'features_dir': '-m', 'matches': '-f',
        'pairs_file': '-p', 'output': '-o', 'bundle_adjustment': '-b',
        'residual_threshold': '-r', 'cache_size': '-c', 'direct': '-d',
        'triangulation_method': '-t', 'sfm_data_tracks': '-T',
    },
    # openMVG_main_ComputeSfM_DataColor -- the whole surface, two flags.
    'mvg_colorize_sfm': {'sfm': '-i', 'output': '-o'},
    # openMVG_main_SfM_Localization
    'mvg_localize': {
        'sfm': '-i', 'features_dir': '-m', 'output_dir': '-o',
        'match_out_dir': '-u', 'query_dir': '-q', 'residual_error': '-r',
        'camera_model': '-c', 'single_intrinsics': '-s',
        'export_structure': '-e', 'resection_method': '-R', 'threads': '-n',
    },
    # openMVG_main_openMVG2openMVS
    'mvg_to_mvs': {
        'sfm': '-i', 'scene': '-o', 'images_dir': '-d', 'threads': '-n',
    },
    # pgs-global-scaler (ours, dependencies/utilities/)
    'mvg_autoscale': {
        'sfm': '-i', 'output': '-o', 'scale_method': '--scale-method',
        'input_mesh': '--input-mesh', 'output_mesh': '--output-mesh',
        'histogram_out': '--histogram-out', 'marker_size': '-s',
        'detection_method': '-m', 'sfm_root': '--sfm-root',
        'include_from': '--include-from', 'exclude_from': '--exclude-from',
        'undistort_images': '--undistort-images',
        'min_marker_pix': '--min-marker-pix',
        'detect_inverted': '--detect-inverted', 'no_ransac': '--no-ransac',
        'save_debug_images': '--save-debug-images',
        'landmarks': '--save-landmarks',
        'scaled_landmarks': '--save-scaled-landmarks',
    },
}

#: Flags deliberately absent from the wrappers, with the reason. Both are
#: ``pgs-global-scaler``'s: OpenMVG's ``CmdLine`` registers no equivalent, which
#: is why this list is so much shorter than the MVS one. ``--force`` is *not*
#: here -- it changes whether the binary recomputes, which is the computation
#: rather than the process, so it is mirrored and left unset (see
#: :class:`TestForceIsMirroredButUnset`).
NOT_MIRRORED = {
    'help': 'not a run parameter',
    'progress': 'the run log is ours to control, not a stage argument',
}

#: Which binary each wrapper resolves, for the fake prefix. All under ``bin/``:
#: ``pgs-global-scaler`` is one of ours but lives beside OpenMVG's.
BINARIES = {
    'init_sfm_generic': 'openMVG_main_SfMInit_ImageListing',
    'compute_features': 'openMVG_main_ComputeFeatures',
    'compute_matches': 'openMVG_main_ComputeMatches',
    'geometric_filter': 'openMVG_main_GeometricFilter',
    'mvg_sfm': 'openMVG_main_SfM',
    'mvg_compute_known': 'openMVG_main_ComputeStructureFromKnownPoses',
    'mvg_colorize_sfm': 'openMVG_main_ComputeSfM_DataColor',
    'mvg_localize': 'openMVG_main_SfM_Localization',
    'mvg_to_mvs': 'openMVG_main_openMVG2openMVS',
    'mvg_autoscale': 'pgs-global-scaler',
}

#: ``make_switch`` (and ``po::bool_switch``) flags: presence-only, so ``True``
#: emits a bare flag and there is no ``0`` spelling to omit. These stay plain
#: ``bool`` for that reason -- ``False`` and absent are the same argv.
SWITCHES = {
    'init_sfm_generic': ('use_pose_prior',),
    'mvg_sfm': ('use_priors',),
    'mvg_compute_known': ('direct', 'bundle_adjustment'),
    'mvg_localize': ('single_intrinsics', 'export_structure'),
    'mvg_autoscale': ('undistort_images', 'detect_inverted', 'no_ransac'),
}

#: ``make_option`` over a ``bool``: the flag takes a value, so these are
#: tri-state -- ``None`` omits, ``False`` emits ``0``, ``True`` emits ``1``.
#: ``group_camera_model`` is why the distinction is load-bearing: OpenMVG's own
#: default is *true*, so omitting it is not the same as passing ``False``.
TRISTATE = ('group_camera_model', 'force', 'guided_matching')

#: A value to pass for each flag whose argv spelling is not ``str(value)``, with
#: what argv should then hold.
PROBES = {kwarg: (False, '0') for kwarg in TRISTATE}

#: Flags whose argv value is a *translation* of the argument, asserted by name in
#: their own tests rather than by the sweep: a resolved or basenamed path, a
#: matches file spelled relative to the regions directory, a case fold.
TRANSLATED = ('images', 'sfm', 'scene', 'images_dir', 'output_dir',
              'features_dir', 'query_dir', 'match_out_dir', 'matches',
              'engine', 'model', 'upright')

#: Flags every invocation carries, and so the ones a "``None`` omits it" sweep
#: cannot assert against. ``detection_method`` is this pipeline's own default,
#: kept because changing what a default run passes is a behaviour change and this
#: module's job is not to make one -- the same footing as MVS's ``archive_type``.
ALWAYS_EMITTED = {'mvg_autoscale': ('detection_method',)}


class OpenMVGCase(ToolchainCase):
    """Each wrapper called over a fake prefix, with ``run_command`` captured.

    ``ToolchainCase`` supplies the resolved temp dir and the isolation of the two
    pieces of process-wide state (``configure()`` and ``$PGS_RECON_PREFIX``), so a
    developer with a prefix exported does not see different results than CI.
    """

    def setUp(self):
        super().setUp()
        toolchain.configure(
            prefix=make_fake_prefix(self.tmp / 'prefix', mvg=BINARIES.values()))
        # openMVG2openMVS names its outputs by basename against a cwd, so its
        # scene and image directory have to share one.
        self.work = self.tmp / 'mvs'
        self.work.mkdir()

    def args(self, name):
        """The arguments ``name`` cannot be called without.

        Values rather than a shared map, because the same keyword means
        different things in different binaries: ``method`` is a describer for
        ``compute_features`` and a matcher for ``compute_matches``.
        """
        required = {
            'init_sfm_generic': dict(images=self.tmp / 'images',
                                     output_dir=self.tmp / 'mvg',
                                     cam_db=self.tmp / 'cameras.txt'),
            'compute_features': dict(sfm=self.tmp / 'sfm.json',
                                     output_dir=self.tmp / 'features',
                                     method='SIFT', preset='HIGH'),
            'compute_matches': dict(sfm=self.tmp / 'sfm.json',
                                    output=self.tmp / 'matches.bin',
                                    method='FASTCASCADEHASHINGL2'),
            'geometric_filter': dict(sfm=self.tmp / 'sfm.json',
                                     matches=self.tmp / 'matches.bin',
                                     output=self.tmp / 'matches.f.bin'),
            'mvg_sfm': dict(sfm=self.tmp / 'sfm.json',
                            features_dir=self.tmp / 'features',
                            matches=self.tmp / 'matches.f.bin',
                            output_dir=self.tmp / 'recon', engine='global'),
            'mvg_compute_known': dict(sfm=self.tmp / 'sfm.json',
                                      features_dir=self.tmp / 'features',
                                      matches=self.tmp / 'matches.bin',
                                      output=self.tmp / 'robust.bin'),
            'mvg_colorize_sfm': dict(sfm=self.tmp / 'sfm.json',
                                     output=self.tmp / 'colorized.ply'),
            'mvg_localize': dict(sfm=self.tmp / 'sfm.json',
                                 features_dir=self.tmp / 'features',
                                 query_dir=self.tmp / 'query',
                                 output_dir=self.tmp / 'localized',
                                 match_out_dir=self.tmp / 'query_regions'),
            'mvg_to_mvs': dict(sfm=self.tmp / 'sfm.json',
                               scene=self.work / 'scene.mvs',
                               images_dir=self.work / 'undistorted'),
            'mvg_autoscale': dict(sfm=self.tmp / 'sfm.json',
                                  output=self.tmp / 'scaled.bin'),
        }
        return required[name]

    def call(self, name, **kwargs):
        """Invoke wrapper ``name`` and return the argv it built.

        ``kwargs`` overrides the required arguments rather than colliding with
        them, so a test can vary ``engine`` without restating the paths.
        """
        with mock.patch('pgs_recon.toolchain.run_command') as ran:
            getattr(openmvg, name)(**{**self.args(name), **kwargs})
        return [str(a) for a in ran.call_args[0][0]]


class TestSurfaceIsComplete(OpenMVGCase):
    """Every registered option is reachable, and every absence is on purpose."""

    def test_every_registered_flag_has_a_keyword_argument(self):
        for name, flags in SURFACES.items():
            params = inspect.signature(getattr(openmvg, name)).parameters
            for kwarg, spelled in flags.items():
                with self.subTest(wrapper=name, flag=spelled):
                    self.assertIn(
                        kwarg, params,
                        f'{name} cannot reach {spelled}; ADR 0005 says a '
                        f'wrapper is the binary\'s full surface')

    def test_no_keyword_argument_is_undeclared(self):
        """The table and the signature are edited together, or neither is
        authoritative."""
        for name, flags in SURFACES.items():
            params = inspect.signature(getattr(openmvg, name)).parameters
            with self.subTest(wrapper=name):
                self.assertEqual(
                    set(), set(params) - set(flags),
                    f'{name} takes arguments SURFACES does not list; add them '
                    f'or explain them in NOT_MIRRORED')

    def test_the_omissions_are_named(self):
        """A tabulated omission stays an omission: name it, or mirror it."""
        for name in SURFACES:
            params = inspect.signature(getattr(openmvg, name)).parameters
            for kwarg, reason in NOT_MIRRORED.items():
                with self.subTest(wrapper=name, flag=kwarg):
                    self.assertNotIn(kwarg, params, reason)

    def test_no_binary_is_handed_one_flag_twice(self):
        """Two keyword arguments sharing a flag is the transcription error this
        table is most likely to contain, and OpenMVG's ``CmdLine`` would take
        the last one silently rather than complain."""
        for name, flags in SURFACES.items():
            with self.subTest(wrapper=name):
                spellings = list(flags.values())
                self.assertEqual(sorted(set(spellings)), sorted(spellings),
                                 f'{name} maps two arguments to one flag')


class TestFlagsReachArgv(OpenMVGCase):
    """A keyword argument is spelled in argv the way the binary declares it."""

    def test_each_flag_lands_with_its_value(self):
        for name, flags in SURFACES.items():
            switches = SWITCHES.get(name, ())
            for kwarg, spelled in flags.items():
                if (kwarg in self.args(name) or kwarg in switches
                        or kwarg in TRANSLATED):
                    continue
                probe, expected = PROBES.get(kwarg, (7, '7'))
                with self.subTest(wrapper=name, flag=spelled):
                    argv = self.call(name, **{kwarg: probe})
                    self.assertIn(spelled, argv)
                    self.assertEqual(expected, flag(argv, spelled))

    def test_omitting_a_flag_omits_it(self):
        """``None`` means the binary's own default wins, not ``--flag None``."""
        for name, flags in SURFACES.items():
            argv = self.call(name)
            always = ALWAYS_EMITTED.get(name, ())
            for kwarg, spelled in flags.items():
                if kwarg in self.args(name) or kwarg in always:
                    continue
                with self.subTest(wrapper=name, flag=spelled):
                    self.assertNotIn(spelled, argv)
            self.assertNotIn('None', argv)

    def test_a_switch_is_emitted_bare(self):
        """``make_switch`` takes no value: the binary reads ``cmd.used()``."""
        for name, switches in SWITCHES.items():
            for kwarg in switches:
                spelled = SURFACES[name][kwarg]
                with self.subTest(wrapper=name, flag=spelled):
                    argv = self.call(name, **{kwarg: True})
                    self.assertIn(spelled, argv)
                    # Nothing follows it but the next flag, if any.
                    following = argv[argv.index(spelled) + 1:]
                    self.assertTrue(all(a.startswith('-') for a in following),
                                    f'{spelled} was given a value')

    def test_a_switch_left_false_is_absent(self):
        for name, switches in SWITCHES.items():
            argv = self.call(name)
            for kwarg in switches:
                with self.subTest(wrapper=name, flag=kwarg):
                    self.assertNotIn(SURFACES[name][kwarg], argv)


class TestBooleanOptionsAreTriState(OpenMVGCase):
    """``make_option`` over a bool is not a switch, and conflating the two loses
    the only spelling that turns an on-by-default flag off."""

    def test_false_is_emitted_as_zero(self):
        self.assertEqual('0', flag(self.call('init_sfm_generic',
                                             group_camera_model=False), '-g'))
        self.assertEqual('0', flag(self.call('compute_features', force=False),
                                   '-f'))

    def test_true_is_emitted_as_one(self):
        self.assertEqual('1', flag(self.call('init_sfm_generic',
                                             group_camera_model=True), '-g'))
        self.assertEqual('1', flag(self.call('geometric_filter',
                                             guided_matching=True), '-r'))

    def test_none_omits_it(self):
        """The case that matters: ``group_camera_model``'s upstream default is
        ``true``, so omitting the flag is not the same as passing ``False``, and
        a wrapper that only had a ``bool`` could not express both."""
        self.assertNotIn('-g', self.call('init_sfm_generic'))

    def test_upright_stays_a_plain_bool(self):
        """It predates this module and ``0`` is already OpenMVG's default, so
        ``False`` and omitted agree and the argv is unchanged from before."""
        self.assertEqual('1', flag(self.call('compute_features', upright=True),
                                   '-u'))
        self.assertNotIn('-u', self.call('compute_features', upright=False))


class TestForceIsMirroredButUnset(OpenMVGCase):
    """``--force`` is reachable and off, which is what makes a killed stage
    resumable: OpenMVG skips every image whose regions it already finds."""

    def test_the_three_binaries_that_have_it_can_reach_it(self):
        for name in ('compute_features', 'compute_matches',
                     'geometric_filter'):
            with self.subTest(wrapper=name):
                self.assertEqual('1', flag(self.call(name, force=True), '-f'))

    def test_it_is_absent_by_default(self):
        for name in ('compute_features', 'compute_matches',
                     'geometric_filter'):
            with self.subTest(wrapper=name):
                self.assertNotIn('-f', self.call(name))


class TestEngineSpecificFlagsReachTheirEngine(OpenMVGCase):
    """The gap this audit found: ``pgs-recon --mvg-recon-method global`` and
    ``stellar`` were selectable while every knob either engine reads was
    unreachable. OpenMVG accepts a flag its engine ignores in silence, so
    nothing failed -- the tuning simply could not be expressed."""

    def test_global_averaging_methods_are_reachable(self):
        argv = self.call(
            'mvg_sfm', engine='GLOBAL',
            rotation_averaging=openmvg.RotationAveraging.L1,
            translation_averaging=openmvg.TranslationAveraging.LIGT)
        self.assertEqual('1', flag(argv, '-R'))
        self.assertEqual('4', flag(argv, '-T'))

    def test_stellar_graph_simplification_is_reachable(self):
        argv = self.call('mvg_sfm', engine='STELLAR',
                         graph_simplification='STAR_X',
                         graph_simplification_value=5)
        self.assertEqual('STAR_X', flag(argv, '-G'))
        self.assertEqual('5', flag(argv, '-g'))

    def test_incremental_initial_pair_is_reachable(self):
        argv = self.call('mvg_sfm', engine='INCREMENTAL',
                         initial_pair_a='left.jpg', initial_pair_b='right.jpg')
        self.assertEqual('left.jpg', flag(argv, '-a'))
        self.assertEqual('right.jpg', flag(argv, '-b'))

    def test_none_of_them_is_emitted_by_default(self):
        """Reaching them must not move them: a default run's argv is what it
        was before this table existed."""
        argv = self.call('mvg_sfm')
        for spelled in ('-R', '-T', '-G', '-g', '-a', '-b', '-e', '-t', '-r',
                        '-c'):
            self.assertNotIn(spelled, argv)


class TestEnumsMatchTheBinary(unittest.TestCase):
    """The enums are transcribed values, and a wrong one is a silently different
    reconstruction rather than an error.

    Checked member by member rather than by spot-check. These used to leak past
    this module -- ``pgs-calibrate`` built two ``choices`` lists straight off
    :class:`~pgs_recon.openmvg.CameraModel` and
    :class:`~pgs_recon.openmvg.ResectionMethod`, so a member added here for
    ``openMVG_main_SfM``'s benefit silently became a value offered for
    ``openMVG_main_SfM_Localization``, a different binary with its own
    validation. ``CameraModel.SPHERICAL`` arrived exactly that way. That app is
    gone (ADR 0011) and its replacement spells its own names in C++, so the enums
    now have one consumer; the per-member check is what keeps them honest to the
    pinned headers regardless.
    """

    #: Each enum's full membership, transcribed from the pinned OpenMVG headers.
    #: Sentinels and ``DEFAULT`` aliases are deliberately *absent*: upstream's
    #: default is expressed by omitting the flag, never by naming a value.
    VALUES = {
        # Camera_Common.hpp. PINHOLE_CAMERA_START (0) and PINHOLE_CAMERA_END (6)
        # bracket the pinhole range and are rejected by isValid(), which is why
        # CAMERA_SPHERICAL is 7 rather than 6.
        'CameraModel': {'PINHOLE': 1, 'RADIAL_1': 2, 'RADIAL_3': 3,
                        'RADIAL_3_TANGENTIAL': 4, 'FISHEYE': 5,
                        'SPHERICAL': 7},
        # solver_resection.hpp. DEFAULT aliases P3P_DING_CVPR23.
        'ResectionMethod': {'DLT': 0, 'P3P_KE': 1, 'P3P_KNEIP': 2,
                            'P3P_NORDBERG': 3, 'P3P_DING': 4, 'UP2P': 5},
        # triangulation_method.hpp. DEFAULT aliases the last member.
        'TriangulationMethod': {'DIRECT_LINEAR_TRANSFORM': 0, 'L1_ANGULAR': 1,
                                'LINFINITY_ANGULAR': 2,
                                'INVERSE_DEPTH_WEIGHTED_MIDPOINT': 3},
        # GlobalSfM_rotation_averaging.hpp -- one-based, no zero member.
        'RotationAveraging': {'L1': 1, 'L2': 2},
        # GlobalSfM_translation_averaging.hpp -- likewise.
        'TranslationAveraging': {'L1': 1, 'L2_DISTANCE_CHORDAL': 2,
                                 'SOFTL1': 3, 'LIGT': 4},
    }

    def test_every_member_has_its_upstream_value(self):
        for enum_name, members in self.VALUES.items():
            enum = getattr(openmvg, enum_name)
            with self.subTest(enum=enum_name):
                self.assertEqual(
                    members, {m.name: m.value for m in enum},
                    f'{enum_name} no longer matches the pinned header; a wrong '
                    f'value is a different reconstruction, not an error')

    def test_no_enum_spells_a_sentinel_or_a_default_alias(self):
        """The two ways a transcription goes wrong. A sentinel is not a model --
        ``--camera-model 6`` is "Invalid camera type" -- and an alias for the
        upstream default would put a value in argv where omitting the flag is
        what "upstream's choice" means."""
        for enum_name in self.VALUES:
            enum = getattr(openmvg, enum_name)
            with self.subTest(enum=enum_name):
                named = [n for n in enum.__members__
                         if n == 'DEFAULT' or n.endswith(('_END', '_START'))]
                self.assertEqual([], named)
                # An IntEnum silently aliases a repeated value onto the first
                # member, which would hide exactly that mistake.
                self.assertEqual(len(enum.__members__),
                                 len({m.value for m in enum}))


class TestEnumsSurviveStringification(OpenMVGCase):
    """An enum must reach argv as its *value* on every Python we support.

    ``toolchain.run`` stringifies argv, and plain ``str()`` of an ``IntEnum``
    returns the value only from Python 3.11 on; 3.9 and 3.10 return
    ``'RotationAveraging.L1'``, which OpenMVG cannot parse. The narrowing that
    makes this version independent lives in ``toolchain._argv_token``, and
    ``test_toolchain`` is what asserts it happened rather than the interpreter
    being new enough -- an assertion these end-to-end checks cannot make, since
    they read argv after stringification.
    """

    def test_an_enum_reaches_argv_as_its_value(self):
        for kwarg, spelled, member in (
                ('rotation_averaging', '-R', openmvg.RotationAveraging.L2),
                ('translation_averaging', '-T',
                 openmvg.TranslationAveraging.SOFTL1),
                ('camera_model', '-c', openmvg.CameraModel.SPHERICAL),
                ('resection_method', '-r', openmvg.ResectionMethod.P3P_DING),
                ('triangulation_method', '-t',
                 openmvg.TriangulationMethod.L1_ANGULAR)):
            with self.subTest(flag=spelled):
                argv = self.call('mvg_sfm', **{kwarg: member})
                self.assertEqual(str(member.value), flag(argv, spelled))

    def test_a_float_is_left_alone(self):
        """The narrowing keys on ``int``, so a ratio or a residual must not be
        truncated on its way through."""
        self.assertEqual('0.8', flag(self.call('compute_matches', ratio=0.8),
                                     '-r'))


class TestAutoscaleSurface(OpenMVGCase):
    """``pgs-global-scaler`` is ours, and was the wrapper missing the most: ten
    of nineteen options, including the scale method and the mesh pair."""

    def test_marker_size_is_optional_so_sample_square_is_reachable(self):
        """The tool requires ``--marker-size`` *unless* the detection method has
        a known fixed size, which a required argument could not express."""
        argv = self.call('mvg_autoscale', detection_method='sample-square')
        self.assertNotIn('-s', argv)
        self.assertEqual('sample-square', flag(argv, '-m'))

    def test_the_mesh_pair_is_reachable(self):
        argv = self.call('mvg_autoscale', input_mesh=self.tmp / 'in.ply',
                         output_mesh=self.tmp / 'out.ply')
        self.assertEqual(str(self.tmp / 'in.ply'), flag(argv, '--input-mesh'))
        self.assertEqual(str(self.tmp / 'out.ply'), flag(argv, '--output-mesh'))

    def test_scale_method_is_reachable(self):
        argv = self.call('mvg_autoscale', scale_method='edge')
        self.assertEqual('edge', flag(argv, '--scale-method'))

    def test_detection_method_is_always_emitted(self):
        self.assertEqual('markers', flag(self.call('mvg_autoscale'), '-m'))


class TestTranslatedArgumentsAreUnchanged(OpenMVGCase):
    """The surface got bigger; what the path-shaped arguments mean did not."""

    def test_the_importer_resolves_its_image_directory(self):
        argv = self.call('init_sfm_generic')
        self.assertEqual(str(self.tmp / 'images'), flag(argv, '-i'))

    def test_the_engine_name_is_upper_cased(self):
        self.assertEqual('GLOBAL', flag(self.call('mvg_sfm'), '-s'))

    def test_the_geometric_model_is_lower_cased(self):
        self.assertEqual('f', flag(self.call('geometric_filter', model='F'),
                                   '-g'))

    def test_the_matches_file_is_relative_to_the_regions_directory(self):
        """``-M`` is joined onto ``-m`` unconditionally, and absolute is the one
        spelling the binary cannot resolve (ADR 0005)."""
        self.assertEqual('../matches.f.bin', flag(self.call('mvg_sfm'), '-M'))

    def test_the_converter_still_names_its_outputs_by_basename(self):
        argv = self.call('mvg_to_mvs')
        self.assertEqual(str(self.tmp / 'sfm.json'), flag(argv, '-i'))
        self.assertEqual('scene.mvs', flag(argv, '-o'))
        self.assertEqual('undistorted', flag(argv, '-d'))

    def test_localize_resolves_all_five_of_its_paths(self):
        argv = self.call('mvg_localize')
        for spelled, expected in (('-i', 'sfm.json'), ('-m', 'features'),
                                  ('-u', 'query_regions'),
                                  ('-o', 'localized'), ('-q', 'query')):
            with self.subTest(flag=spelled):
                self.assertEqual(str(self.tmp / expected), flag(argv, spelled))


class TestPriorsStillDecideTheInitializer(OpenMVGCase):
    """Untouched by the expansion, and the reason ``-S`` is not just another
    tabulated flag: exactly one may be emitted, and the caller's wins."""

    def test_priors_on_incrementalv2_seed_from_existing_poses(self):
        argv = self.call('mvg_sfm', engine='incrementalv2', use_priors=True)
        self.assertIn('-P', argv)
        self.assertEqual('EXISTING_POSE', flag(argv, '-S'))

    def test_an_explicit_initializer_wins(self):
        argv = self.call('mvg_sfm', engine='incrementalv2', use_priors=True,
                         initializer='MAX_PAIR')
        self.assertEqual('MAX_PAIR', flag(argv, '-S'))
        self.assertEqual(1, argv.count('-S'))

    def test_other_engines_get_no_initializer(self):
        self.assertNotIn('-S', self.call('mvg_sfm', engine='global',
                                         use_priors=True))


class TestResolvedBinaries(OpenMVGCase):
    """argv[0] is the absolute path to the right binary under MVG_BIN."""

    def test_each_wrapper_runs_its_own_binary(self):
        for name, binary in BINARIES.items():
            with self.subTest(wrapper=name):
                argv = self.call(name)
                self.assertEqual(binary, Path(argv[0]).name)
                self.assertIn(MVG_BIN, argv[0])


if __name__ == '__main__':
    unittest.main()
