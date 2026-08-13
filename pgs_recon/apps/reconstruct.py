"""Run the photogrammetry pipeline on a set of input images."""
import argparse
import atexit
import logging
import socket
import sys
from datetime import datetime as dt, timezone as tz
from pathlib import Path

import exiftool
import configargparse
import sfm_utils as sfm

from pgs_recon import layout, toolchain
from pgs_recon.openmvg import (compute_features, compute_matches,
                               geometric_filter, init_sfm_generic,
                               mvg_colorize_sfm, mvg_compute_known, mvg_sfm,
                               mvg_autoscale, mvg_to_mvs)
from pgs_recon.openmvs import (mvs_densify, mvs_reconstruct, mvs_refine,
                               mvs_texture)
from pgs_recon.pgs_data import init_sfm_pgs, get_tag_option
from pgs_recon.stages import (CONTROL_ARGS, NO_PERSIST, STAGES, StageError,
                              StageTracker, apply_stored, drifted_stages,
                              explicit_dests, find_manifest, load_manifest,
                              pipeline_shape, resolve_range,
                              revert_out_of_range, utc_now, validate_arg_map,
                              write_manifest)
from pgs_recon.toolchain import Recorder
from pgs_recon.utility import ToolFailed
from pgs_recon.utils.apps import setup_logging


def init_sfm_generic2(scan_dir: Path, sfm_file: Path, camdb_path: Path,
                      recorder: Recorder = None):
    """Init an SfM scene from the given directory"""
    logger = logging.getLogger(__name__)
    if recorder is not None:
        recorder.step('init_sfm_generic2', scan_dir=scan_dir,
                      sfm_file=sfm_file, camdb_path=camdb_path)
    # Load the camera db
    cam_db = sfm.openmvg_load_camdb(camdb_path)

    # Get a list of files
    extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}
    extensions = extensions.union([ext.upper() for ext in extensions])
    all_files = list(scan_dir.glob(f'*.*'))
    all_files.sort()
    images = []
    for f in all_files:
        if 'mask.png' in str(f):
            logger.warning(f'{str(f.name)} is a mask image')
        elif Path(f).suffix in extensions:
            images.append(f)
        else:
            logger.debug(f'Ignoring file: {str(f.name)}')
    if len(images) == 0:
        logger.error(
            'Provided scan metadata specifies file pattern, but no files match.')
        raise RuntimeError()

    # Get image metadata
    with exiftool.ExifToolHelper() as et:
        img_metadata = et.get_metadata([str(i) for i in images])

    # Setup sfm
    scene = sfm.Scene()
    scene.root_dir = scan_dir

    # Fill out sfm with data
    for img in images:
        # Lookup this images tags
        tags = next((i for i in img_metadata if i['File:FileName'] == img.name),
                    None)
        if tags is None:
            logger.error(f'No tags loaded for image: {str(img.name)}')
            continue

        # Setup view
        view = sfm.View()
        view.path = img
        view.width = get_tag_option(tags, ['File:ImageWidth', 'EXIF:ImageWidth'])
        view.height = get_tag_option(tags, ['File:ImageHeight', 'EXIF:ImageHeight'])
        view.make = tags['EXIF:Make']
        view.model = tags['EXIF:Model']

        # Setup intrinsic
        intrinsic = sfm.IntrinsicRadialK3()
        intrinsic.width = view.width
        intrinsic.height = view.height
        intrinsic.focal_length = tags['EXIF:FocalLength']
        if f'{view.make} {view.model}' in cam_db.keys():
            intrinsic.sensor_width = cam_db[f'{view.make} {view.model}']
        elif f'{view.model}' in cam_db.keys():
            intrinsic.sensor_width = cam_db[f'{view.model}']
        else:
            logger.warning(
                f'Camera not in database: {view.make} {view.model}. Ignoring file: {img.name}')
            continue

        # Only add everything to the SfM at the end
        scene.add_view(view)
        view.intrinsic = scene.add_intrinsic(intrinsic)
        view.pose = scene.add_pose(sfm.Pose())

    sfm.export_scene(path=sfm_file, scene=scene)


def build_parser() -> configargparse.ArgumentParser:
    """Build the pgs-recon parser.

    Factored out of ``main()`` so a second, sentinel-defaulted copy can be
    parsed to find which arguments the user actually supplied (see
    ``stages.explicit_dests``). Any argument added here must also be classified
    in ``stages.STAGE_ARGS``/``GLOBAL_ARGS``; ``validate_arg_map()`` enforces it.
    """
    parser = configargparse.ArgumentParser(prog='pgs-recon')

    # Generic options
    parser.add_argument('--config', '-c', is_config_file=True,
                        help='Config file path')
    parser.add_argument('--input', '-i',
                        help='directory of input images. Required for a new '
                             'reconstruction; recovered from the output '
                             'directory\'s manifest when resuming.')
    parser.add_argument('--output', '-o', required=True,
                        help='directory for output files')
    parser.add_argument('--name', '-n', type=str,
                        help='Experiment name. Recovered from the output '
                             'directory\'s manifest when resuming.')
    parser.add_argument('--file-type', choices=['ply', 'obj'],
                        default='obj', type=str.lower,
                        help='Output format for final textured mesh')
    parser.add_argument('--focal-length', '-f', type=int, default=None,
                        help='focal length in pixels', metavar='n')
    parser.add_argument('--new-importer', default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--import-pgs-scan', '-p', default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Input directory is assumed to be a PGS Scan '
                             'directory')
    parser.add_argument('--import-calib', type=str,
                        help='When importing a PGS Scan, merge provided PGS '
                             'calibration file with the imported camera '
                             'configurations.')
    parser.add_argument('--import-capture', type=int, default=0, metavar='n',
                        help='When importing a PGS Scan, the capture to '
                             'reconstruct from, as it appears in the '
                             '{prefix}{camera}_{position}_{capture} filename. '
                             'A scan captures every position once per capture, '
                             'each with its own lighting and camera set; this '
                             'selects one of them (default: %(default)s)')
    parser.add_argument('--log-level', default='INFO', type=str.upper,
                        choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])

    # Hidden opts. --path has no default of its own: unset, the install prefix
    # falls through to $PGS_RECON_PREFIX and then to toolchain.DEFAULT_PREFIX,
    # which giving it a default here would shadow.
    parser.add_argument('--path', type=str, default=None,
                        help=configargparse.SUPPRESS)
    parser.add_argument('--threads', type=int, help=configargparse.SUPPRESS)

    opts_desc = parser.add_argument_group('describer options')
    opts_desc.add_argument('--describer-method',
                           choices=['SIFT', 'AKAZE_FLOAT', 'AKAZE_MLDB'],
                           default='SIFT',
                           type=str.upper,
                           help="Set the feature descriptors method.")
    opts_desc.add_argument('--describer-preset',
                           choices=['NORMAL', 'HIGH', 'ULTRA'], default='HIGH',
                           type=str.upper,
                           help='Set the description detail level.')
    opts_desc.add_argument('--describer-upright', '-u', default=False,
                           action=argparse.BooleanOptionalAction,
                           help='Disable rotational invariance for feature '
                                'detection step. Useful if the camera is '
                                'always "upright" w.r.t the ground plane.')

    opts_matcher = parser.add_argument_group('matcher options')
    # These are the matchers openMVG_main_ComputeMatches builds at the pinned
    # revision, and nothing else parses -- an unrecognized name is a hard failure
    # in the binary, not a fallback. ANNL2 was offered here long after upstream
    # replaced it with the HNSW matchers, so choosing it failed the stage.
    opts_matcher.add_argument('--matching-method',
                              choices=['AUTO',
                                       'BRUTEFORCEL2',
                                       'HNSWL2',
                                       'HNSWL1',
                                       'CASCADEHASHINGL2',
                                       'FASTCASCADEHASHINGL2',
                                       'BRUTEFORCEHAMMING',
                                       'HNSWHAMMING'],
                              default='FASTCASCADEHASHINGL2', type=str.upper,
                              help='Feature matching method. BRUTEFORCEHAMMING '
                                   'and HNSWHAMMING are for binary descriptors '
                                   '(--describer-method AKAZE_MLDB), the rest '
                                   'for scalar ones; AUTO chooses from the '
                                   'descriptor type. HNSWL1 is tuned for '
                                   'quantized/histogram descriptors.')
    opts_matcher.add_argument('--matching-geometric-model',
                              choices=['f', 'e', 'h', 'a', 'u', 'o'],
                              type=str.lower,
                              help='Geometric model for robust putative '
                                   'matches filtering: f: Fundamental, '
                                   'e: Essential, h: Homography, a: essential '
                                   'matrix with angular parameterization, u: '
                                   'upright essential matrix with angular '
                                   'parameterization, o: orthographic '
                                   'essential matrix')
    opts_matcher.add_argument('--matching-ratio', type=float, default=None,
                              help='Nearest-Neighbor distance ratio')
    opts_matcher.add_argument('--matching-pairs-file', type=str, default='auto',
                              help='NONE, AUTO, or path to an OpenMVG view '
                                   'pairs file. If AUTO (default), use the '
                                   'view pairs file created when importing a '
                                   'PGS scan. If NONE, do not use a view pairs '
                                   'file.')
    opts_matcher.add_argument('--matching-pairs-radius', type=int, default=2,
                              help='The neighbor search radius when '
                                   'automatically generating a view pairs file')

    opts_mvg = parser.add_argument_group('mvg reconstruction options')
    opts_mvg.add_argument('--mvg-recon-method', '-m',
                          choices=['global',
                                   'stellar',
                                   'incremental',
                                   'incrementalv2',
                                   'direct'],
                          type=str.lower, default='global',
                          help='MVG scene reconstruction method. Note: direct '
                               'requires an sfm file with camera poses.')
    opts_mvg.add_argument('--mvg-priors', default=False,
                          action=argparse.BooleanOptionalAction,
                          help='Use pose priors with SfM reconstruction')
    opts_mvg.add_argument('--mvg-refine-intrinsics', type=str.upper,
                          help='SfM intrinsic refinement options: NONE, '
                               'ADJUST_FOCAL_LENGTH, ADJUST_PRINCIPAL_POINT, '
                               'ADJUST_DISTORTION, ADJUST_ALL. '
                               'Note: Quoted options can be combined with '
                               '\'|\' (e.g. '
                               '\'ADJUST_FOCAL_LENGTH|ADJUST_DISTORTION\')')
    opts_mvg.add_argument('--mvg-robust', '-r', default=False,
                          action=argparse.BooleanOptionalAction,
                          help='robustly triangulate reconstructed scene')
    opts_mvg.add_argument('--mvg-autoscale', type=float,
                          help='use pgs-global-scaler to automatically scale '
                               'aruco markers to the provided size in '
                               'real world units. see the pgs-global-scaler '
                               '--marker-size flag for more information')
    opts_mvg.add_argument('--autoscale-method', default='markers',
                          choices=['markers', 'sample-square'],
                          help='marker detection method. see the '
                               'pgs-global-scaler --detection-method flag '
                               'for more information'
                          )
    opts_mvg.add_argument('--autoscale-marker-pix', type=int,
                          help="Minimum marker size in pixels. see the "
                               "pgs-global-scaler --min-marker-pix flag "
                               "for more information")
    opts_mvg.add_argument('--autoscale-include-from',
                          help='text file containing a list of scene image '
                               'files to be exclusively considered during '
                               'auto-scaling. see the pgs-global-scaler '
                               '--include-from flag for more information')
    opts_mvg.add_argument('--autoscale-exclude-from',
                          help='text file containing a list of scene image '
                               'files to be excluded during auto-scaling. '
                               'see the pgs-global-scaler --exclude-from flag '
                               'for more information')

    # MVG hidden opts
    opts_mvg.add_argument('--cam-db', type=str, help=configargparse.SUPPRESS)
    opts_mvg.add_argument('--sfm-ba', default=False,
                          action=argparse.BooleanOptionalAction,
                          help=configargparse.SUPPRESS)
    opts_mvg.add_argument('--robust-ba', default=False,
                          action=argparse.BooleanOptionalAction,
                          help=configargparse.SUPPRESS)
    opts_mvg.add_argument('--mvg-initializer',
                          choices=['EXISTING_POSE', 'MAX_PAIR', 'AUTO_PAIR',
                                   'STELLAR'],
                          type=str.upper, help=configargparse.SUPPRESS)

    opts_mvs = parser.add_argument_group('openmvs options')
    # Deprecated alias for --to colorize. It only ever truncated the pipeline, so
    # it is flow control rather than configuration -- which also means it is no
    # longer recorded in the manifest, and so no longer sticky across resumes.
    opts_mvs.add_argument('--mvs', action=argparse.BooleanOptionalAction,
                          default=None, help=configargparse.SUPPRESS)
    opts_mvs.add_argument('--free-space-support', default=False,
                          action=argparse.BooleanOptionalAction,
                          help='use free-space support in ReconstructMesh')
    opts_mvs.add_argument('--mvs-densify', default=False,
                          action=argparse.BooleanOptionalAction,
                          help='Enable point cloud densification step')
    opts_mvs.add_argument('--mvs-refine', default=True,
                          action=argparse.BooleanOptionalAction,
                          help='Enable MVS mesh refinement step')
    opts_mvs.add_argument('--mvs-smooth', type=int, default=2,
                          help='Number of smoothing iterations after initial '
                               'surface reconstruction. 0 is disabled.')
    opts_mvs.add_argument('--densify-resolution-level', default=None, type=int,
                          help='how many times to scale down images before '
                               'DensifyPointCloud')
    opts_mvs.add_argument('--refine-resolution-level', default=None, type=int,
                          help='scale input images down N times before')
    opts_mvs.add_argument('--refine-min-resolution', default=None, type=int,
                          help='do not scale images\' max dimension smaller '
                               'than this value when using '
                               '--refine-resolution-level')
    opts_mvs.add_argument('--refine-scales', default=3, type=int,
                          help='number of mesh optimization iterations on '
                               'multi-scale images')
    opts_mvs.add_argument('--refine-scale-step', default=None, type=float,
                          help='image scale factor used at each mesh '
                               'optimization step')
    # Refine's mesh *preparation*, where a large mesh now spends its wall clock.
    opts_mvs.add_argument('--refine-ensure-edge-size', default=None, type=int,
                          help='improve edge sizes and vertex valence before '
                               'refinement (0 - disabled, 1 - auto, 2 - force). '
                               'Pass 0 if refine stalls in preparation.')
    opts_mvs.add_argument('--refine-max-face-area', default=None, type=int,
                          help='maximum projected face area left unsubdivided '
                               'before refinement (0 - disabled)')
    opts_mvs.add_argument('--texture-resolution-level', default=None, type=int,
                          help='how many times to scale down images before '
                               'TextureMesh')
    opts_mvs.add_argument('--decimation-factor', type=float,
                          help='Decimation factor in range [0..1] to be '
                               'applied to the input surface before mesh '
                               'refinement (0 - auto, 1 - disabled)')
    opts_mvs.add_argument('--mask-value', type=int, default=0,
                          help='Label value in the image mask to ignore during '
                               'mesh densification. Set to a value < 0 to '
                               'ignore masks during this step.')
    opts_mvs.add_argument('--texture-max-size', type=int, default=0,
                          help='Limits the maximum size (edge length) of the'
                               'output texture image. If set to 0 (default), '
                               'the edge length is unbounded.')

    opts_stage = parser.add_argument_group(
        'staged run options',
        'Split one reconstruction across several jobs, each sized for the '
        'stages it runs. State lives in <output>/pgs-recon.json; re-running the '
        'original command verbatim resumes where the last job stopped.')
    opts_stage.add_argument('--from', dest='from_stage', choices=STAGES,
                            metavar='STAGE',
                            help=f'First stage to run (inclusive). Defaults to '
                                 f'the first stage of the pipeline. Every '
                                 f'earlier stage must already be complete. '
                                 f'One of: {", ".join(STAGES)}')
    opts_stage.add_argument('--to', dest='to_stage', choices=STAGES,
                            metavar='STAGE',
                            help='Last stage to run (inclusive). Defaults to '
                                 'the last stage of the pipeline.')
    opts_stage.add_argument('--rerun', default=False, action='store_true',
                            help='Re-run every stage in the range, even ones '
                                 'already recorded complete.')
    opts_stage.add_argument('--dry-run', default=False, action='store_true',
                            help='Resolve the plan (loaded arguments, pipeline '
                                 'shape, skip-vs-run per stage, rehydrated '
                                 'inputs, prerequisites), print it, and exit '
                                 'without running a binary.')
    return parser


def main():
    """Entry point. Planning and tool failures both surface here.

    Both are raised rather than exiting in place so they can be tested without a
    subprocess. A ``ToolFailed`` exits with the *binary's* status, not 1, so a
    stage killed by the OOM reaper is distinguishable from a bad argument in a
    batch scheduler's log.
    """
    try:
        _main()
    except StageError as e:
        sys.exit(f'ERROR: {e}')
    except ToolFailed as e:
        # A binary can only fail after setup_logging, so the reason for the exit
        # status lands in the run's log next to the stage lines that led to it.
        logging.getLogger('pgs-recon').error(f'{e}')
        sys.exit(e.exit_code)


def _main():
    parser = build_parser()
    validate_arg_map(parser)
    args = parser.parse_args()
    explicit = explicit_dests(build_parser, sys.argv[1:])

    out_dir = Path(args.output).resolve()
    manifest_path = layout.manifest(out_dir)
    # Read from wherever this directory's manifest actually is; write to the
    # current name, which is what moves a pre-2.0 directory onto it.
    read_from = find_manifest(out_dir)

    # Load the previous run(s) in this directory. Recorded effective arguments
    # become defaults, so --input/--name are not needed to resume.
    metadata = load_manifest(read_from)
    stored = metadata.get('effective_args') or {}
    apply_stored(args, stored, explicit)

    setup_logging(args.log_level)
    logger = logging.getLogger("pgs-recon")
    if stored:
        logger.info(f'Loaded arguments from {read_from}')
    if read_from != manifest_path:
        logger.warning(f'Resuming from {read_from.name}, written by pgs-recon '
                       f'before 2.0. This run records to {manifest_path.name}; '
                       f'the old file is left in place and goes stale.')

    if args.mvs is False:
        if args.to_stage not in (None, 'colorize'):
            raise StageError(f'--no-mvs and --to {args.to_stage} conflict. '
                             f'--no-mvs is a deprecated alias for --to '
                             f'colorize; drop it and use --to on its own.')
        logger.warning('--no-mvs is deprecated and will be removed; it is '
                       'equivalent to --to colorize. Stopping after colorize.')
        args.to_stage = 'colorize'
    elif args.mvs is True:
        logger.warning('--mvs is deprecated and will be removed; it already '
                       'does nothing, as the MVS stages run unless you stop '
                       'the range before them (e.g. --to colorize).')

    # Enable flags declare the pipeline shape; --from/--to select a window of it
    shape = pipeline_shape(args)
    from_stage, to_stage = resolve_range(args, shape)
    revert_out_of_range(args, stored, explicit, from_stage, to_stage, logger)
    # Reverting an out-of-range override can change the shape back
    shape = pipeline_shape(args)
    from_stage, to_stage = resolve_range(args, shape)

    if args.mask_value is not None and args.mask_value < 0:
        args.mask_value = None

    # Only the PGS importer reads capture indices. Warn rather than fail, so a
    # shared config carrying PGS settings still drives a generic run.
    if args.import_capture != 0 and not args.import_pgs_scan:
        logger.warning(f'--import-capture={args.import_capture} is ignored '
                       f'without --import-pgs-scan; the generic importers take '
                       f'every image in the input directory.')

    # Where the binaries are and what records their invocations: process-wide, so
    # no stage takes either as an argument (ADR 0005). The command log has to
    # exist before the recorder wraps it, and the recorder before any stage runs.
    recorder = Recorder(metadata.setdefault('commands', {}))
    toolchain.configure(prefix=args.path, recorder=recorder)

    # The output layout, kept only for the manifest's own record of where a run
    # put things: nothing reads it back, and artifact names come from ``layout``.
    logger.info('Setting up output directories')
    paths = {
        'output': out_dir,
        'mvg': layout.mvg_dir(out_dir),
        'matches_dir': layout.matches_dir(out_dir),
        'recon_dir': layout.recon_dir(out_dir),
        'mvs': layout.mvs_dir(out_dir),
        'undistorted_images': layout.undistorted_images(out_dir),
        'manifest': manifest_path,
    }
    if args.input is not None:
        paths['input'] = Path(args.input)
    if args.import_calib:
        paths['input_calib'] = Path(args.import_calib)

    # Setup experiment
    experiment_start = dt.now(tz.utc)
    datetime_str = experiment_start.strftime('%Y%m%d%H%M%S')
    if args.name is None:
        if args.input is None:
            raise StageError(f'--name could not be recovered from '
                             f'{manifest_path} and --input was not given, so '
                             f'there is no name to derive it from.')
        args.name = datetime_str + '_' + str(Path(args.input).stem)
    # One config per reconstruction, not per job; the manifest's 'runs' has the
    # per-invocation history.
    config = layout.config(out_dir, args.name)
    paths['config'] = config

    # Resolve the plan. The tracker reads the manifest and the merged arguments
    # and touches nothing else, so everything up to here is safe under --dry-run.
    drift = drifted_stages(args, metadata.get('stages') or {}, explicit, shape)
    tracker = StageTracker(metadata, manifest_path, out_dir, args, shape,
                           from_stage, to_stage, rerun=args.rerun, drift=drift,
                           logger=logger)
    tracker.log_plan()

    # Prerequisites are never auto-backfilled: a job sized for texturing must
    # not silently start refining.
    errors = tracker.prereq_errors()
    if tracker.status_of('import') == 'run' and args.input is None:
        errors.append('import: --input was neither given nor recorded in '
                      f'{manifest_path}')
    if errors:
        for err in errors:
            logger.error(err)
        raise StageError(f'cannot start at {from_stage}; {len(errors)} '
                         f'prerequisite problem(s) above.')
    logger.info('prereqs OK.')

    # --dry-run stops before the first write of any kind. Persisting effective
    # args here would make a dry run's overrides sticky -- `--dry-run
    # --no-mvs-refine` would silently drop refine from every later run.
    if args.dry_run:
        logger.info('nothing executed (--dry-run).')
        return

    # Create output folders
    for d in layout.directories(out_dir):
        d.mkdir(exist_ok=True, parents=True)

    # Write config after all arguments have been changed. Flow-control flags are
    # left out so the file stays usable as a -c config for another run, and so
    # are unset ones: a literal `focal-length = None` read back would be parsed
    # as the string 'None' rather than as the default it stands for.
    args.config = str(config)
    with config.open(mode='w') as file:
        for arg in vars(args):
            if arg in CONTROL_ARGS:
                continue
            attr = getattr(args, arg)
            if attr is None:
                continue
            arg = arg.replace('_', '-')
            file.write(f'{arg} = {attr}\n')

    # Init metadata. Merge into any existing manifest rather than clobbering it:
    # commands accumulate across the runs that share this directory. ``parsed``
    # is a copy, not an alias of args.__dict__: the atexit flush would otherwise
    # serialise it as of exit while effective_args is a snapshot from here.
    metadata['args'] = " ".join(sys.argv)
    metadata['parsed'] = dict(vars(args))
    metadata['effective_args'] = {k: v for k, v in vars(args).items()
                                  if k not in NO_PERSIST}
    metadata['shape'] = list(shape)
    metadata.setdefault('runs', []).append({
        'argv': " ".join(sys.argv),
        'started': utc_now(),
        'host': socket.gethostname(),
        'range': f'{from_stage}..{to_stage}',
    })

    # Register a manifest write whenever the program closes. Stage transitions
    # write it too, so a killed job still leaves an accurate record.
    @atexit.register
    def flush_manifest():
        metadata['paths'] = {key: str(val) for key, val in paths.items()}
        write_manifest(paths['manifest'], metadata)

    flush_manifest()

    try:
        run_pipeline(tracker, args, out_dir, recorder, logger)
    except BaseException:
        # A failed binary raises ToolFailed past here; anything not marked
        # complete is re-runnable, but recording 'failed' makes the reason
        # legible. BaseException so a Ctrl-C is recorded too.
        tracker.abort()
        raise

    recorder.note('Processing complete')
    logger.info(f'Processing complete. Results saved to: {out_dir}')


def run_pipeline(tracker: StageTracker, args, output: Path,
                 recorder: Recorder, logger):
    """Run the stages in the tracker's range, in pipeline order.

    Each block asks the tracker whether its stage runs, takes its inputs from the
    artifact chain (``require`` for what a stage cannot run without, ``path`` for
    what may legitimately be absent), names its outputs through ``layout``, and
    reports them back under semantic roles. Because a name is a function of the
    output root and nothing else (ADR 0006), a staged run produces exactly the
    filenames a single-shot run would.

    The invariants that are ours rather than the binaries' live here, not in the
    wrappers (ADR 0005): notably that ``reconstruct`` is handed the dense cloud
    whenever densify ran.
    """
    images = Path(args.input) if args.input is not None else None
    cam_db = toolchain.cam_db(args.cam_db)

    if tracker.begin('import'):
        logger.info('Importing dataset')
        imported = layout.imported_sfm(output)
        outputs = {'sfm': imported}
        if args.import_pgs_scan:
            pairs = init_sfm_pgs(
                images, sfm_file=imported, cam_db=cam_db,
                view_pairs_file=layout.view_pairs(output),
                calib_file=args.import_calib,
                pairs_file_radius=args.matching_pairs_radius,
                capture=args.import_capture,
                recorder=recorder)
            if pairs is not None:
                outputs['view_pairs'] = pairs
        elif args.new_importer:
            init_sfm_generic2(images.resolve(), sfm_file=imported,
                              camdb_path=cam_db, recorder=recorder)
        else:
            init_sfm_generic(images, output_dir=layout.mvg_dir(output),
                             cam_db=cam_db, focal_length=args.focal_length)
        tracker.end('import', inputs={'images': images}, outputs=outputs)

    if tracker.begin('features'):
        logger.info('Computing image features')
        sfm_in = tracker.require('sfm')
        features = layout.matches_dir(output)
        compute_features(sfm_in, output_dir=features,
                         method=args.describer_method,
                         preset=args.describer_preset,
                         upright=args.describer_upright, threads=args.threads)
        tracker.end('features', inputs={'sfm': sfm_in},
                    outputs={'features': features})

    if tracker.begin('matches'):
        logger.info('Matching image features')
        sfm_in = tracker.require('sfm')
        features_in = tracker.require('features')
        pairs_file = args.matching_pairs_file
        if pairs_file is None or pairs_file.lower() == 'none':
            pairs_file = None
        elif pairs_file.lower() == 'auto':
            # The importer's view pairs file, from this run or an earlier job
            pairs_file = (tracker.path('view_pairs')
                          if args.import_pgs_scan else None)
        else:
            pairs_file = Path(pairs_file)
        matches = layout.matches(output)
        compute_matches(sfm_in, output=matches, method=args.matching_method,
                        ratio=args.matching_ratio, pairs_file=pairs_file)
        tracker.end('matches', inputs={'sfm': sfm_in, 'features': features_in,
                                       'view_pairs': pairs_file},
                    outputs={'matches': matches})

    if tracker.begin('filter'):
        logger.info('Filtering image features')
        sfm_in = tracker.require('sfm')
        matches_in = tracker.require('matches')
        filtered = layout.matches_filtered(matches_in)
        geometric_filter(sfm_in, matches=matches_in, output=filtered,
                         model=args.matching_geometric_model)
        tracker.end('filter', inputs={'sfm': sfm_in, 'matches': matches_in},
                    outputs={'matches_filtered': filtered})

    if tracker.begin('sfm'):
        sfm_in = tracker.require('sfm')
        features_in = tracker.require('features')
        inputs = {'sfm': sfm_in, 'features': features_in}
        if args.mvg_recon_method == 'direct':
            # Triangulating known poses uses the unfiltered matches
            logger.info('Computing structure from known poses')
            matches_in = tracker.require('matches')
            inputs['matches'] = matches_in
            # Same name the openMVG engines produce: this is the solve, however
            # it was computed, and only one of the two branches ever runs.
            solved = layout.solved_sfm(output)
            mvg_compute_known(sfm_in, features_dir=features_in,
                              matches=matches_in, output=solved, direct=True,
                              bundle_adjustment=args.sfm_ba)
        else:
            logger.info(f'Running SfM (Engine: {args.mvg_recon_method})')
            filtered_in = tracker.require('matches_filtered')
            inputs['matches_filtered'] = filtered_in
            # openMVG_main_SfM names its own output, so the wrapper reports it.
            solved = mvg_sfm(sfm_in, features_dir=features_in,
                             matches=filtered_in,
                             output_dir=layout.recon_dir(output),
                             engine=args.mvg_recon_method,
                             use_priors=args.mvg_priors,
                             refine_intrinsics=args.mvg_refine_intrinsics,
                             initializer=args.mvg_initializer)
        tracker.end('sfm', inputs=inputs, outputs={'sfm': solved})

    if tracker.begin('robust'):
        logger.info('Performing robust triangulation')
        sfm_in = tracker.require('sfm')
        features_in = tracker.require('features')
        matches_in = tracker.require('matches')
        robust = layout.robust_sfm(output)
        mvg_compute_known(sfm_in, features_dir=features_in, matches=matches_in,
                          output=robust, bundle_adjustment=args.robust_ba)
        tracker.end('robust', inputs={'sfm': sfm_in, 'features': features_in,
                                      'matches': matches_in},
                    outputs={'sfm': robust})

    if tracker.begin('autoscale'):
        logger.info('Auto-scaling SfM scene')
        sfm_in = tracker.require('sfm')
        scaled = layout.autoscale_sfm(output)
        mvg_autoscale(sfm_in, output=scaled, marker_size=args.mvg_autoscale,
                      detection_method=args.autoscale_method,
                      min_marker_pix=args.autoscale_marker_pix,
                      include_from=args.autoscale_include_from,
                      exclude_from=args.autoscale_exclude_from,
                      landmarks=layout.landmarks(output),
                      scaled_landmarks=layout.scaled_landmarks(output))
        tracker.end('autoscale', inputs={'sfm': sfm_in},
                    outputs={'sfm': scaled})

    # colorize is a leaf: it produces a side artifact, so its output is recorded
    # under its own role and the ``sfm`` role stays bound to what it coloured.
    if tracker.begin('colorize'):
        logger.info('Colorizing SfM scene')
        sfm_in = tracker.require('sfm')
        colorized = layout.colorize_sfm(output)
        mvg_colorize_sfm(sfm_in, output=colorized)
        tracker.end('colorize', inputs={'sfm': sfm_in},
                    outputs={'colorized': colorized})

    if tracker.begin('convert'):
        logger.info('Converting MVG scene to MVS scene')
        sfm_in = tracker.require('sfm')
        scene = layout.convert_scene(output)
        mvg_to_mvs(sfm_in, scene=scene,
                   images_dir=layout.undistorted_images(output),
                   threads=args.threads)
        tracker.end('convert', inputs={'sfm': sfm_in},
                    outputs={'scene': scene})

    if tracker.begin('densify'):
        logger.info('Densifying point cloud')
        scene_in = tracker.require('scene')
        scene = layout.densify_scene(output)
        cloud = layout.densify_cloud(output)
        mvs_densify(scene_in, output=scene,
                    resolution_level=args.densify_resolution_level,
                    ignore_mask_label=args.mask_value)
        tracker.end('densify', inputs={'scene': scene_in},
                    outputs={'scene': scene, 'cloud': cloud})

    if tracker.begin('reconstruct'):
        logger.info('Reconstructing mesh')
        scene_in = tracker.require('scene')
        # ADR 0003: whenever densify ran, its dense cloud MUST be handed over --
        # the scene it wrote still holds the sparse one, so omitting -p meshes
        # that instead, silently.
        cloud_in = tracker.path('cloud')
        mesh = layout.reconstruct_mesh(output)
        mvs_reconstruct(scene_in, output=mesh, point_cloud=cloud_in,
                        free_space_support=args.free_space_support,
                        smooth=args.mvs_smooth)
        # ReconstructMesh hands the scene back untouched, so only the mesh is
        # recorded as produced: see STAGE_IO on pass-through roles.
        tracker.end('reconstruct',
                    inputs={'scene': scene_in, 'cloud': cloud_in},
                    outputs={'mesh': mesh})

    if tracker.begin('refine'):
        logger.info('Refining mesh')
        scene_in = tracker.require('scene')
        mesh_in = tracker.require('mesh')
        refined = layout.refine_mesh(output)
        mvs_refine(scene_in, mesh=mesh_in, output=refined,
                   decimate=args.decimation_factor,
                   resolution_level=args.refine_resolution_level,
                   min_resolution=args.refine_min_resolution,
                   scales=args.refine_scales,
                   scale_step=args.refine_scale_step,
                   ensure_edge_size=args.refine_ensure_edge_size,
                   max_face_area=args.refine_max_face_area)
        tracker.end('refine', inputs={'scene': scene_in, 'mesh': mesh_in},
                    outputs={'mesh': refined})

    if tracker.begin('texture'):
        logger.info('Texturing mesh')
        scene_in = tracker.require('scene')
        mesh_in = tracker.require('mesh')
        final = layout.final_mesh(output, args.name, args.file_type)
        mvs_texture(scene_in, mesh=mesh_in, output=final,
                    export_type=args.file_type,
                    resolution_level=args.texture_resolution_level,
                    max_texture_size=args.texture_max_size)
        tracker.end('texture', inputs={'scene': scene_in, 'mesh': mesh_in},
                    outputs={'mesh': final})


if __name__ == '__main__':
    main()
