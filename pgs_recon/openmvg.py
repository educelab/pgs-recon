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
"""
from enum import IntEnum
from pathlib import Path

from pgs_recon.toolchain import relative_to_dir, resolve_exe, run, work_dir


class CameraModel(IntEnum):
    PINHOLE = 1
    RADIAL_1 = 2
    RADIAL_3 = 3
    RADIAL_3_TANGENTIAL = 4
    FISHEYE = 5


class ResectionMethod(IntEnum):
    # Values mirror openMVG::resection::SolverType (solver_resection.hpp).
    # Only DLT estimates the focal; the rest require a known intrinsic.
    DLT = 0
    P3P_KE = 1
    P3P_KNEIP = 2
    P3P_NORDBERG = 3
    P3P_DING = 4
    UP2P = 5


def init_sfm_generic(images: Path, output_dir: Path, cam_db: Path,
                     focal_length: int = None) -> None:
    """Import a directory of images as an SfM scene.

    OpenMVG writes ``sfm_data.json`` into ``output_dir`` under a name of its own
    choosing (:func:`layout.imported_sfm`).
    """
    command = [
        resolve_exe('openMVG_main_SfMInit_ImageListing'),
        '-i', Path(images).resolve(),
        '-o', output_dir,
        '-d', cam_db,
    ]
    if focal_length is not None:
        command.extend(['-f', focal_length])
    run(command)


def compute_features(sfm: Path, output_dir: Path, method: str, preset: str,
                     upright: bool = False, threads: int = None) -> None:
    """Detect and describe image features.

    Regions land in ``output_dir`` under names OpenMVG derives from each image,
    so the directory is the artifact.
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
    if threads is not None:
        command.extend(['-n', threads])
    run(command)


def compute_matches(sfm: Path, output: Path, method: str, ratio: float = None,
                    pairs_file: Path = None) -> None:
    """Match image features, writing putative matches to ``output``.

    ``pairs_file`` limits matching to the listed view pairs -- a grid scan's
    spatial neighbours (see :mod:`pgs_recon.pgs_data`).
    """
    command = [
        resolve_exe('openMVG_main_ComputeMatches'),
        '-i', sfm,
        '-o', output,
        '-n', method,
    ]
    if ratio is not None:
        command.extend(['-r', ratio])
    if pairs_file is not None:
        command.extend(['-p', pairs_file])
    run(command)


def geometric_filter(sfm: Path, matches: Path, output: Path, model: str = None,
                     pairs_file: Path = None) -> None:
    """Geometrically filter putative matches into ``output``."""
    command = [
        resolve_exe('openMVG_main_GeometricFilter'),
        '-i', sfm,
        '-m', matches,
        '-o', output,
    ]
    if model is not None:
        command.extend(['-g', model.lower()])
    if pairs_file is not None:
        command.extend(['-p', pairs_file])
    run(command)


def mvg_sfm(sfm: Path, features_dir: Path, matches: Path, output_dir: Path,
            engine: str, use_priors: bool = False,
            refine_intrinsics: str = None,
            initializer: str = None) -> Path:
    """Solve the scene, returning the ``sfm_data.bin`` OpenMVG names itself.

    ``matches`` is the filtered matches file. OpenMVG resolves ``-M`` **relative
    to** ``features_dir`` -- it joins the two unconditionally -- so the argument
    is translated into that form (:func:`toolchain.relative_to_dir`) rather than
    passed through. It need not live in ``features_dir``: a matches file
    elsewhere is spelled ``../…`` and resolves fine. Passing an absolute path is
    what the binary cannot do, and translating here is what keeps a caller from
    trying.
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
    if refine_intrinsics is not None:
        command.extend(['-f', refine_intrinsics])
    if initializer is not None:
        command.extend(['-S', initializer])
    run(command)
    return Path(output_dir) / 'sfm_data.bin'


def mvg_autoscale(sfm: Path, output: Path, marker_size: float,
                  detection_method: str = 'markers', min_marker_pix: int = None,
                  include_from: Path = None, exclude_from: Path = None,
                  landmarks: Path = None,
                  scaled_landmarks: Path = None) -> None:
    """Rescale a scene to physical units from detected markers.

    ``landmarks``/``scaled_landmarks``, when given, save the markers as found and
    after rescaling -- the check that autoscale did what was asked.
    """
    command = [
        resolve_exe('pgs-global-scaler'),
        '-i', sfm,
        '-o', output,
        '-s', marker_size,
        '-m', detection_method,
    ]
    if landmarks is not None:
        command.extend(['--save-landmarks', landmarks])
    if scaled_landmarks is not None:
        command.extend(['--save-scaled-landmarks', scaled_landmarks])
    if min_marker_pix is not None:
        command.extend(['--min-marker-pix', min_marker_pix])
    if include_from is not None:
        command.extend(['--include-from', include_from])
    if exclude_from is not None:
        command.extend(['--exclude-from', exclude_from])
    run(command)


def mvg_compute_known(sfm: Path, features_dir: Path, matches: Path,
                      output: Path, direct: bool = False,
                      bundle_adjustment: bool = False) -> None:
    """Triangulate structure from known poses.

    Reads the **unfiltered** matches (``-f``), unlike :func:`mvg_sfm`, and by
    full path rather than basename. ``direct`` is the reconstruction method that
    triangulates the imported scene's rig priors; without it this is the robust
    re-triangulation of an already solved scene.
    """
    command = [
        resolve_exe('openMVG_main_ComputeStructureFromKnownPoses'),
        '-i', sfm,
        '-m', features_dir,
        '-o', output,
        '-f', matches,
    ]
    if direct:
        command.append('-d')
    if bundle_adjustment:
        command.append('-b')
    run(command)


def mvg_colorize_sfm(sfm: Path, output: Path) -> None:
    """Colour a scene's sparse cloud from the images."""
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
    if camera_model is not None:
        command.extend(['-c', camera_model])
    if resection_method is not None:
        command.extend(['-R', resection_method])
    if residual_error is not None:
        command.extend(['-r', residual_error])
    if single_intrinsics:
        command.append('-s')
    if export_structure:
        command.append('-e')
    if threads is not None:
        command.extend(['-n', threads])
    run(command)
    return Path(output_dir) / 'sfm_data_expanded.json'


def mvg_to_mvs(sfm: Path, scene: Path, images_dir: Path,
               threads: int = None) -> None:
    """Convert a solved OpenMVG scene to an OpenMVS one, undistorting as it goes.

    Runs in the scene's directory and names both outputs by basename, so the
    undistorted images land beside the scene where every MVS stage expects them
    (:func:`toolchain.work_dir`). The input is passed absolute for that reason.
    """
    work = work_dir(scene, images_dir)
    command = [
        resolve_exe('openMVG_main_openMVG2openMVS'),
        '-i', Path(sfm).resolve(),
        '-o', Path(scene).name,
        '-d', Path(images_dir).name,
    ]
    if threads is not None:
        command.extend(['-n', threads])
    run(command, cwd=work)
