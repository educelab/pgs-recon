"""One function per OpenMVG binary, plus our own ``pgs-global-scaler``.

Each is a straight translation of a Python call into a single binary invocation
(`ADR 0005 <../docs/adr/0005-wrappers-mirror-the-binary.md>`_): assemble an argv,
hand it to :func:`toolchain.run`, return nothing. The exceptions are the two
binaries that name their own output -- :func:`mvg_sfm` and :func:`mvg_localize`
are handed an output *directory* and pick the filename inside it, so they return
the path they wrote. Everywhere else the caller named the output and already
holds it.

Consequently a wrapper takes no ``prefix`` and no ``metadata``: the install
prefix and the command log are process-wide configuration
(:func:`toolchain.configure`), which is what makes these usable from outside
``pgs-recon`` without fabricating a dict, and what makes it impossible for a
wrapper to forget to record what it ran. Output *naming* is
:mod:`pgs_recon.layout`'s job, not theirs.

``None`` means **omit the flag**, leaving the binary's own default in force. That
is load-bearing rather than tidiness: a default spelled out here would silently
diverge from OpenMVG's the day OpenMVG changes one.

Each function is its binary's **complete** registered option surface, transcribed
from the ``cmd.add`` calls at the pinned revision (``c92ed1b``,
``dependencies/cmake/BuildOpenMVG.cmake``), which is the authority rather than the
hand-written usage block a binary prints -- the two disagree, and only the
registrations decide what parses. ``tests/test_openmvg.SURFACES`` tabulates that
inventory so ADR 0005's "every flag reachable" is a failing test rather than a
claim. The only omissions are ``pgs-global-scaler``'s ``--help`` and
``--progress``; OpenMVG's own ``CmdLine`` registers no equivalents.

Unlike :mod:`pgs_recon.openmvs`, the argv flag cannot be derived from the keyword
argument's name, so each wrapper carries the mapping literally. Two reasons: the
same letter means different things in different binaries (``-f`` is ``--force``
in ``ComputeFeatures``, ``--focal`` in ``SfMInit_ImageListing``, ``--match_file``
in ``ComputeStructureFromKnownPoses`` and ``--refine_intrinsic_config`` in
``SfM``), and the long spellings are an unpredictable mix of ``snake_case``,
``camelCase`` and single words (``--input_file``, ``--describerMethod``,
``--outdir``). The short flags are what these wrappers have always passed and
what the recorded argv holds.

How a flag is spelled depends on how OpenMVG registered it, and the three cases
are not interchangeable:

- ``make_option`` over a string or number: ``None`` omits it.
- ``make_option`` over a **bool** takes a value, so it is a tri-state here --
  ``None`` omits, ``False`` emits ``0``, ``True`` emits ``1``. That matters
  wherever the binary's own default is *true*: ``group_camera_model=False`` is
  the only way to turn grouping off, and omitting it leaves it on.
- ``make_switch`` has no value at all -- the binary reads ``cmd.used('P')`` -- so
  ``False`` and omitted are the same thing and these stay plain ``bool``.

``threads`` reaches ``-n`` only in an OpenMP build. Upstream registers that flag
inside ``#ifdef OPENMVG_USE_OPENMP`` (``ComputeFeatures``, ``SfM_Localization``,
``openMVG2openMVS``), and a build without OpenMP rejects it as unknown rather
than ignoring it. Ours is built with it; a library user's may not be.
"""
from enum import IntEnum
from pathlib import Path

from pgs_recon.toolchain import relative_to_dir, resolve_exe, run, work_dir


class CameraModel(IntEnum):
    # Values mirror openMVG::cameras::EINTRINSIC (Camera_Common.hpp). 6 is
    # PINHOLE_CAMERA_END, a sentinel rather than a model, which is why SPHERICAL
    # is 7 and not 6.
    PINHOLE = 1
    RADIAL_1 = 2
    RADIAL_3 = 3
    RADIAL_3_TANGENTIAL = 4
    FISHEYE = 5
    SPHERICAL = 7


class ResectionMethod(IntEnum):
    # Values mirror openMVG::resection::SolverType (solver_resection.hpp).
    # Only DLT estimates the focal; the rest require a known intrinsic.
    DLT = 0
    P3P_KE = 1
    P3P_KNEIP = 2
    P3P_NORDBERG = 3
    P3P_DING = 4
    UP2P = 5


class TriangulationMethod(IntEnum):
    # Values mirror openMVG::ETriangulationMethod (triangulation_method.hpp).
    # Upstream's DEFAULT is an alias for INVERSE_DEPTH_WEIGHTED_MIDPOINT rather
    # than a sixth value, so it is not spelled here -- pass ``None`` for it.
    DIRECT_LINEAR_TRANSFORM = 0
    L1_ANGULAR = 1
    LINFINITY_ANGULAR = 2
    INVERSE_DEPTH_WEIGHTED_MIDPOINT = 3


class RotationAveraging(IntEnum):
    # openMVG::sfm::ERotationAveragingMethod (GlobalSfM_rotation_averaging.hpp).
    L1 = 1
    L2 = 2


class TranslationAveraging(IntEnum):
    # openMVG::sfm::ETranslationAveragingMethod
    # (GlobalSfM_translation_averaging.hpp).
    L1 = 1
    L2_DISTANCE_CHORDAL = 2
    SOFTL1 = 3
    LIGT = 4


def _optional(*flags) -> list:
    """``flag value`` for each ``(flag, value)`` pair whose value is not ``None``.

    Values are passed through as given. How one becomes an argv string -- a
    ``bool`` as ``0``/``1``, an :class:`~enum.IntEnum` as its value rather than
    its member name -- is :func:`toolchain._argv_token`'s job, at the one
    chokepoint that stringifies argv, so the rule holds for the flags assembled
    by hand here too. Callers used to have to remember ``int()``.
    """
    command = []
    for flag, value in flags:
        if value is None:
            continue
        command.append(flag)
        command.append(value)
    return command


def _switches(*flags) -> list:
    """The bare flag for each ``(flag, value)`` pair that is true.

    ``make_switch`` options carry no value, so there is no ``0`` to emit and
    ``False`` cannot be distinguished from absence.
    """
    return [flag for flag, value in flags if value]


def init_sfm_generic(images: Path, output_dir: Path, cam_db: Path,
                     focal_length: int = None, intrinsics: str = None,
                     camera_model: int = None,
                     group_camera_model: bool = None,
                     use_pose_prior: bool = False, prior_weights: str = None,
                     gps_to_xyz_method: int = None) -> None:
    """Import a directory of images as an SfM scene.

    OpenMVG writes ``sfm_data.json`` into ``output_dir`` under a name of its own
    choosing (:func:`layout.imported_sfm`).

    ``focal_length`` keeps its name rather than the binary's ``--focal``: it is
    what the flag has always been called here and out at ``pgs-recon
    --focal-length``. ``intrinsics`` is the whole calibration instead, as
    OpenMVG's ``"f;0;ppx;0;f;ppy;0;0;1"`` K-matrix string.

    ``group_camera_model`` is the one flag here whose upstream default is *true*,
    so ``False`` is meaningful and ``None`` is not the same thing: omitting it
    lets views keep sharing one intrinsic. ``use_pose_prior`` reads the EXIF GPS
    pose, weighted per axis by ``prior_weights`` (``"x;y;z"``) and projected by
    ``gps_to_xyz_method`` (0 ECEF, 1 UTM); the solve then needs
    :func:`mvg_sfm`'s ``use_priors`` to act on them.
    """
    command = [
        resolve_exe('openMVG_main_SfMInit_ImageListing'),
        '-i', Path(images).resolve(),
        '-o', output_dir,
        '-d', cam_db,
    ]
    command += _optional(('-f', focal_length), ('-k', intrinsics),
                         ('-c', camera_model), ('-g', group_camera_model))
    command += _switches(('-P', use_pose_prior))
    command += _optional(('-W', prior_weights), ('-m', gps_to_xyz_method))
    run(command)


def compute_features(sfm: Path, output_dir: Path, method: str, preset: str,
                     upright: bool = False, force: bool = None,
                     threads: int = None) -> None:
    """Detect and describe image features.

    Regions land in ``output_dir`` under names OpenMVG derives from each image,
    so the directory is the artifact.

    ``force`` recomputes regions that are already on disk. Leaving it unset is
    what makes a killed features stage resumable -- OpenMVG skips every image
    whose ``.feat``/``.desc`` it finds -- so this defaults to omitted for the
    same reason ADR 0004 wants stages restartable.

    ``upright`` stays a plain ``bool`` rather than joining the tri-state
    convention: it predates this module, and ``0`` is already OpenMVG's default,
    so ``False`` and omitted agree.
    """
    command = [
        resolve_exe('openMVG_main_ComputeFeatures'),
        '-i', sfm,
        '-o', output_dir,
        '-m', method,
        '-p', preset,
    ]
    if upright:
        command.extend(['-u', '1'])
    command += _optional(('-f', force), ('-n', threads))
    run(command)


def compute_matches(sfm: Path, output: Path, method: str, ratio: float = None,
                    pairs_file: Path = None, force: bool = None,
                    cache_size: int = None,
                    preemptive_feature_count: int = None) -> None:
    """Match image features, writing putative matches to ``output``.

    ``pairs_file`` limits matching to the listed view pairs -- a grid scan's
    spatial neighbours (see :mod:`pgs_recon.pgs_data`).

    ``cache_size`` bounds how many images' regions are held in memory at once;
    unset loads them all, which is the fast path and the one that runs out of RAM
    on a large scan. ``preemptive_feature_count`` matches on that many features
    first and drops pairs that look unpromising, which is the cheaper way to cut
    an exhaustive match down.
    """
    command = [
        resolve_exe('openMVG_main_ComputeMatches'),
        '-i', sfm,
        '-o', output,
        '-n', method,
    ]
    command += _optional(('-r', ratio), ('-p', pairs_file), ('-f', force),
                         ('-c', cache_size),
                         ('-P', preemptive_feature_count))
    run(command)


def geometric_filter(sfm: Path, matches: Path, output: Path, model: str = None,
                     pairs_file: Path = None, output_pairs: Path = None,
                     force: bool = None, guided_matching: bool = None,
                     max_iteration: int = None,
                     cache_size: int = None) -> None:
    """Geometrically filter putative matches into ``output``.

    ``output_pairs`` writes out the pairs that survived, which is the input
    ``pairs_file`` of a later run rather than anything this one consumes.
    ``guided_matching`` re-matches each pair using the model just estimated for
    it, recovering correspondences the putative pass missed.
    """
    command = [
        resolve_exe('openMVG_main_GeometricFilter'),
        '-i', sfm,
        '-m', matches,
        '-o', output,
    ]
    command += _optional(
        ('-g', model.lower() if model is not None else None),
        ('-p', pairs_file), ('-s', output_pairs), ('-f', force),
        ('-r', guided_matching), ('-I', max_iteration), ('-c', cache_size),
    )
    run(command)


def mvg_sfm(sfm: Path, features_dir: Path, matches: Path, output_dir: Path,
            engine: str, use_priors: bool = False,
            refine_intrinsics: str = None, refine_extrinsics: str = None,
            initializer: str = None, triangulation_method: int = None,
            resection_method: int = None, camera_model: int = None,
            initial_pair_a: str = None, initial_pair_b: str = None,
            rotation_averaging: int = None, translation_averaging: int = None,
            graph_simplification: str = None,
            graph_simplification_value: int = None) -> Path:
    """Solve the scene, returning the ``sfm_data.bin`` OpenMVG names itself.

    ``matches`` is the filtered matches file. OpenMVG resolves ``-M`` **relative
    to** ``features_dir`` -- it joins the two unconditionally -- so the argument
    is translated into that form (:func:`toolchain.relative_to_dir`) rather than
    passed through. It need not live in ``features_dir``: a matches file
    elsewhere is spelled ``../…`` and resolves fine. Passing an absolute path is
    what the binary cannot do, and translating here is what keeps a caller from
    trying.

    Most of the surface below is **engine specific**, and OpenMVG accepts the
    flags it does not use in silence rather than rejecting them -- so passing one
    to the wrong ``engine`` does nothing at all, which is why each is documented
    by the engine that reads it:

    - ``INCREMENTAL``: ``initial_pair_a``/``initial_pair_b`` (image *filenames*,
      no path) seed the two-view reconstruction; ``camera_model``,
      ``triangulation_method`` and ``resection_method`` apply here and to
      ``INCREMENTALV2``.
    - ``INCREMENTALV2``: ``initializer`` picks the seed instead.
    - ``GLOBAL``: ``rotation_averaging`` (:class:`RotationAveraging`) and
      ``translation_averaging`` (:class:`TranslationAveraging`).
    - ``STELLAR``: ``graph_simplification`` (``NONE``, ``MST_X``, ``STAR_X``) and
      ``graph_simplification_value``, which the binary requires to be > 1.

    ``refine_intrinsics`` and ``refine_extrinsics`` are the bundle adjustment's,
    on every engine: ``ADJUST_ALL`` (upstream's default), ``NONE``, or for
    intrinsics a ``|``-joined combination of ``ADJUST_FOCAL_LENGTH``,
    ``ADJUST_PRINCIPAL_POINT`` and ``ADJUST_DISTORTION``.
    """
    command = [
        resolve_exe('openMVG_main_SfM'),
        '-i', sfm,
        '-s', engine.upper(),
        '-m', features_dir,
        '-o', output_dir,
        '-M', relative_to_dir(matches, features_dir),
    ]
    if use_priors:
        command.append('-P')
    # Priors on the INCREMENTALV2 engine need the pose-seeded initializer, but
    # exactly one ``-S`` may be emitted: OpenMVG keeps the *last* it is given
    # (verified against the v2.1 binary -- `-S BOGUS -S MAX_PAIR` is accepted,
    # `-S MAX_PAIR -S BOGUS` is rejected). Emitting both left the argv claiming
    # an initializer the run did not use, so the recorded command lied about the
    # reconstruction. Deciding here instead keeps the caller's choice winning --
    # which it already did, by accident of ordering.
    if initializer is None and use_priors and engine.lower() == 'incrementalv2':
        initializer = 'EXISTING_POSE'
    command += _optional(('-f', refine_intrinsics), ('-S', initializer),
                         ('-e', refine_extrinsics),
                         ('-t', triangulation_method),
                         ('-r', resection_method), ('-c', camera_model),
                         ('-a', initial_pair_a), ('-b', initial_pair_b),
                         ('-R', rotation_averaging),
                         ('-T', translation_averaging),
                         ('-G', graph_simplification),
                         ('-g', graph_simplification_value))
    run(command)
    return Path(output_dir) / 'sfm_data.bin'


def mvg_autoscale(sfm: Path, output: Path, marker_size: float = None,
                  detection_method: str = 'markers', scale_method: str = None,
                  min_marker_pix: int = None, include_from: Path = None,
                  exclude_from: Path = None, landmarks: Path = None,
                  scaled_landmarks: Path = None, input_mesh: Path = None,
                  output_mesh: Path = None, histogram_out: Path = None,
                  sfm_root: Path = None, save_debug_images: Path = None,
                  undistort_images: bool = False,
                  detect_inverted: bool = False,
                  no_ransac: bool = False) -> None:
    """Rescale a scene to physical units from detected markers.

    Our own tool (``dependencies/utilities/``) rather than OpenMVG's, so its flags
    are long-form and its source is in this repository -- but it is a binary
    behind :func:`toolchain.run` like any other, and ADR 0005 applies to it the
    same way.

    ``landmarks``/``scaled_landmarks``, when given, save the markers as found and
    after rescaling -- the check that autoscale did what was asked.
    ``histogram_out`` writes the distribution of per-observation scale estimates
    as SVG, which is how a bimodal fit (two marker sizes in one scan, or a false
    positive) shows itself.

    ``marker_size`` is required *unless* ``detection_method='sample-square'``,
    whose size the tool already knows -- hence the ``None`` default rather than a
    positional argument. ``scale_method`` chooses how the observations are
    combined: ``umeyama`` (the tool's default, a weighted median) or ``edge``, the
    median of marker edge lengths.

    ``input_mesh``/``output_mesh`` apply the same scale factor to a mesh, and only
    do anything together -- the tool checks for both. ``no_ransac`` *disables*
    the RANSAC that makes marker triangulation tolerant of false-positive
    matches, matching the flag's polarity rather than its name.
    """
    command = [
        resolve_exe('pgs-global-scaler'),
        '-i', sfm,
        '-o', output,
    ]
    command += _optional(('-s', marker_size), ('-m', detection_method))
    command += _optional(('--save-landmarks', landmarks),
                         ('--save-scaled-landmarks', scaled_landmarks),
                         ('--min-marker-pix', min_marker_pix),
                         ('--include-from', include_from),
                         ('--exclude-from', exclude_from),
                         ('--scale-method', scale_method),
                         ('--input-mesh', input_mesh),
                         ('--output-mesh', output_mesh),
                         ('--histogram-out', histogram_out),
                         ('--sfm-root', sfm_root),
                         ('--save-debug-images', save_debug_images))
    command += _switches(('--undistort-images', undistort_images),
                         ('--detect-inverted', detect_inverted),
                         ('--no-ransac', no_ransac))
    run(command)


def mvg_compute_known(sfm: Path, features_dir: Path, matches: Path,
                      output: Path, direct: bool = False,
                      bundle_adjustment: bool = False,
                      pairs_file: Path = None,
                      residual_threshold: float = None,
                      cache_size: int = None,
                      triangulation_method: int = None,
                      sfm_data_tracks: Path = None) -> None:
    """Triangulate structure from known poses.

    Reads the **unfiltered** matches (``-f``), unlike :func:`mvg_sfm`, and by
    full path rather than basename. ``direct`` is the reconstruction method that
    triangulates the imported scene's rig priors; without it this is the robust
    re-triangulation of an already solved scene.

    The binary has three mutually exclusive modes and the flags select between
    them, which is why they are not independent knobs: guided epipolar matching
    over ``pairs_file`` or ``matches`` (the default), robust triangulation of
    ``matches``' tracks (``direct``), or tracks read as landmark observations from
    ``sfm_data_tracks``.

    ``residual_threshold`` is the reprojection error in pixels above which a
    triangulation is discarded (upstream's default is 4.0), so it is the knob for
    a scene that comes back sparser or noisier than its poses deserve.
    """
    command = [
        resolve_exe('openMVG_main_ComputeStructureFromKnownPoses'),
        '-i', sfm,
        '-m', features_dir,
        '-o', output,
        '-f', matches,
    ]
    command += _switches(('-d', direct), ('-b', bundle_adjustment))
    command += _optional(('-p', pairs_file), ('-r', residual_threshold),
                         ('-c', cache_size), ('-t', triangulation_method),
                         ('-T', sfm_data_tracks))
    run(command)


def mvg_colorize_sfm(sfm: Path, output: Path) -> None:
    """Colour a scene's sparse cloud from the images.

    ``-i`` and ``-o`` are the whole of ``ComputeSfM_DataColor``'s surface, so
    this wrapper has no flags to omit and nothing to keep in step with upstream.
    """
    command = [
        resolve_exe('openMVG_main_ComputeSfM_DataColor'),
        '-i', sfm,
        '-o', output,
    ]
    run(command)


def mvg_localize(sfm: Path, features_dir: Path, query_dir: Path,
                 output_dir: Path, match_out_dir: Path,
                 camera_model: int = None, resection_method: int = None,
                 residual_error: float = None, single_intrinsics: bool = False,
                 export_structure: bool = False, threads: int = None) -> Path:
    """Localize new image(s) into an existing reconstruction.

    Resections each image in ``query_dir`` against the database scene ``sfm``
    (which must carry structure) using the database regions in ``features_dir``.
    New query regions go to ``match_out_dir`` so the original regions are left
    untouched. Returns the ``sfm_data_expanded.json`` OpenMVG writes into
    ``output_dir`` -- the database views plus the localized query views.

    For an uncalibrated camera leave ``single_intrinsics`` off so a fresh
    intrinsic is estimated. ``resection_method=0`` (DLT) does not require known
    intrinsics and so recovers focal length as part of the pose; the P3P methods
    assume a calibrated camera.
    """
    command = [
        resolve_exe('openMVG_main_SfM_Localization'),
        '-i', Path(sfm).resolve(),
        '-m', Path(features_dir).resolve(),
        '-u', Path(match_out_dir).resolve(),
        '-o', Path(output_dir).resolve(),
        '-q', Path(query_dir).resolve(),
    ]
    command += _optional(('-c', camera_model), ('-R', resection_method),
                         ('-r', residual_error))
    command += _switches(('-s', single_intrinsics), ('-e', export_structure))
    command += _optional(('-n', threads))
    run(command)
    return Path(output_dir) / 'sfm_data_expanded.json'


def mvg_to_mvs(sfm: Path, scene: Path, images_dir: Path,
               threads: int = None) -> None:
    """Convert a solved OpenMVG scene to an OpenMVS one, undistorting as it goes.

    Runs in the scene's directory and names both outputs by basename, so the
    undistorted images land beside the scene where every MVS stage expects them
    (:func:`toolchain.work_dir`). The input is passed absolute for that reason.

    The image paths it writes into the scene are *relative to the scene file's own
    directory*, and OpenMVS resolves them against ``-w``; those agree because the
    MVS stages derive ``-w`` from the scene. Nothing here checks that the images
    exist, because relative paths make the whole ``mvs/`` directory movable
    together -- see ADR 0005.
    """
    work = work_dir(scene, images_dir)
    command = [
        resolve_exe('openMVG_main_openMVG2openMVS'),
        '-i', Path(sfm).resolve(),
        '-o', Path(scene).name,
        '-d', Path(images_dir).name,
    ]
    command += _optional(('-n', threads))
    run(command, cwd=work)
