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

from pgs_recon.openmvg import (compute_features, compute_matches,
                               geometric_filter, init_sfm_generic,
                               mvg_colorize_sfm, mvg_compute_known, mvg_sfm,
                               mvg_autoscale, mvg_to_mvs)
from pgs_recon.openmvs import (mvs_densify, mvs_reconstruct, mvs_refine,
                               mvs_texture)
from pgs_recon.pgs_data import init_sfm_pgs, get_tag_option
from pgs_recon.stages import (CONTROL_ARGS, NO_PERSIST, STAGES, StageError,
                              StageTracker, apply_stored, drifted_stages,
                              explicit_dests, load_manifest,
                              pipeline_shape, resolve_range,
                              revert_out_of_range, utc_now, validate_arg_map,
                              write_manifest)
from pgs_recon.utility import current_timestamp
from pgs_recon.utils.apps import setup_logging


def init_sfm_generic2(scan_dir: Path, sfm_file: Path, camdb_path: Path):
    """Init an SfM scene from the given directory"""
    logger = logging.getLogger(__name__)
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
                             'directory\'s metadata.json when resuming.')
    parser.add_argument('--output', '-o', required=True,
                        help='directory for output files')
    parser.add_argument('--name', '-n', type=str,
                        help='Experiment name. Recovered from the output '
                             'directory\'s metadata.json when resuming.')
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
    parser.add_argument('--log-level', default='INFO', type=str.upper,
                        choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])

    # Hidden opts
    parser.add_argument('--path', type=str, default='/usr/local/',
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
    opts_matcher.add_argument('--matching-method',
                              choices=['AUTO',
                                       'BRUTEFORCEL2',
                                       'ANNL2',
                                       'CASCADEHASHINGL2',
                                       'FASTCASCADEHASHINGL2',
                                       'BRUTEFORCEHAMMING'],
                              default='FASTCASCADEHASHINGL2', type=str.upper,
                              help='Feature matching method.')
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
                               "pgs-global-scaler --min-marker-size flag "
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
        'stages it runs. State lives in <output>/metadata.json; re-running the '
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
    """Entry point. Planning failures arrive as ``StageError`` and exit here.

    The planner raises rather than calling ``sys.exit`` so it can be tested
    without a subprocess; ``utility.run_command`` still exits directly.
    """
    try:
        _main()
    except StageError as e:
        sys.exit(f'ERROR: {e}')


def _main():
    parser = build_parser()
    validate_arg_map(parser)
    args = parser.parse_args()
    explicit = explicit_dests(build_parser, sys.argv[1:])

    out_dir = Path(args.output).resolve()
    manifest_path = out_dir / 'metadata.json'

    # Load the previous run(s) in this directory. Recorded effective arguments
    # become defaults, so --input/--name are not needed to resume.
    metadata = load_manifest(manifest_path)
    stored = metadata.get('effective_args') or {}
    apply_stored(args, stored, explicit)

    setup_logging(args.log_level)
    logger = logging.getLogger("pgs-recon")
    if stored:
        logger.info(f'Loaded arguments from {manifest_path}')

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

    # Structure for storing important paths
    logger.info('Setting up output directories')
    paths = {
        'PATH': Path(args.path).resolve(),
        'output': out_dir,
    }
    if args.input is not None:
        paths['input'] = Path(args.input)
    if args.import_calib:
        paths['input_calib'] = Path(args.import_calib)
    paths['BIN'] = paths['PATH'] / 'bin'
    paths['MVS_BIN'] = paths['BIN'] / 'OpenMVS'
    if args.cam_db is None:
        db_path = 'lib/openMVG/sensor_width_camera_database.txt'
        paths['CAM_DB'] = paths['PATH'] / db_path
    else:
        paths['CAM_DB'] = Path(args.cam_db).resolve()

    # Setup output directory names
    paths['mvg'] = paths['output'] / 'mvg'
    paths['matches_dir'] = paths['mvg'] / 'matches_dir'
    paths['matches_file'] = paths['matches_dir'] / 'matches.bin'
    # The filtered matches are not pre-seeded here: geometric_filter derives the
    # name from its input and the sfm stage reads it back through the
    # matches_filtered binding, so there is only one place it is spelled.
    paths['recon_dir'] = paths['mvg'] / 'recon_dir'
    paths['sfm'] = paths['mvg'] / 'sfm_data.json'
    paths['mvs'] = paths['output'] / 'mvs'
    paths['mvs_scene'] = paths['mvs'] / 'scene.mvs'
    paths['mvs_images'] = paths['mvs'] / 'undistorted_images'
    paths['metadata'] = manifest_path

    # Setup experiment
    experiment_start = dt.now(tz.utc)
    datetime_str = experiment_start.strftime('%Y%m%d%H%M%S')
    if args.name is None:
        if args.input is None:
            raise StageError(f'--name could not be recovered from '
                             f'{manifest_path} and --input was not given, so '
                             f'there is no name to derive it from.')
        args.name = datetime_str + '_' + str(Path(args.input).stem)
    # One config per reconstruction, not per job; metadata.json's 'runs' has the
    # per-invocation history.
    config = paths['output'] / f'{args.name}_recon_config.txt'

    # Resolve the plan. The tracker reads the manifest and the merged arguments
    # and touches nothing else, so everything up to here is safe under --dry-run.
    drift = drifted_stages(args, metadata.get('stages') or {}, explicit, shape)
    tracker = StageTracker(metadata, manifest_path, paths, args, shape,
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
    for d in 'output', 'mvg', 'matches_dir', 'recon_dir', 'mvs':
        paths[d].mkdir(exist_ok=True, parents=True)

    # Write config after all arguments have been changed. Flow-control flags are
    # left out so the file stays usable as a -c config for another run.
    args.config = str(config)
    with config.open(mode='w') as file:
        for arg in vars(args):
            if arg in CONTROL_ARGS:
                continue
            attr = getattr(args, arg)
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
    metadata.setdefault('commands', {})
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
        write_manifest(paths['metadata'], metadata)

    flush_manifest()

    try:
        run_pipeline(tracker, paths, args, metadata, logger)
    except BaseException:
        # run_command sys.exit()s on failure; anything not marked complete is
        # re-runnable, but recording 'failed' makes the reason legible.
        tracker.abort()
        raise

    metadata['commands'][current_timestamp()] = "Processing complete"
    logger.info(f'Processing complete. Results saved to: {paths["output"]}')


def run_pipeline(tracker: StageTracker, paths, args, metadata, logger):
    """Run the stages in the tracker's range, in pipeline order.

    Each block asks the tracker whether its stage runs, rehydrates its inputs
    from the artifact chain, and reports the outputs back under semantic roles.
    Because the OpenMVG/OpenMVS wrappers derive output filenames from the input
    ``Path``'s stem and not from the ``paths`` key, a staged run produces exactly
    the filenames a single-shot run would.
    """
    if tracker.begin('import'):
        logger.info('Importing dataset')
        outputs = {'sfm': paths['sfm']}
        if args.import_pgs_scan:
            init_sfm_pgs(paths, pairs_file_radius=args.matching_pairs_radius,
                         metadata=metadata)
            if paths.get('view_pairs') is not None:
                outputs['view_pairs'] = paths['view_pairs']
        elif args.new_importer:
            metadata['commands'][current_timestamp()] = (
                f'init_sfm_generic2(scan_dir={paths["input"]}, '
                f'sfm_file={paths["sfm"]}, camdb_path={paths["CAM_DB"]})')
            init_sfm_generic2(paths['input'].resolve(),
                              sfm_file=paths['sfm'],
                              camdb_path=paths['CAM_DB'])
        else:
            init_sfm_generic(paths, focal_length=args.focal_length,
                             metadata=metadata)
        tracker.end('import', inputs={'images': paths.get('input')},
                    outputs=outputs)

    if tracker.begin('features'):
        logger.info('Computing image features')
        sfm_in = tracker.key('sfm')
        out_key = compute_features(paths, sfm_key=sfm_in,
                                   method=args.describer_method,
                                   preset=args.describer_preset,
                                   upright=args.describer_upright,
                                   metadata=metadata, threads=args.threads)
        tracker.end('features', inputs={'sfm': paths[sfm_in]},
                    outputs={'features': paths[out_key]})

    if tracker.begin('matches'):
        logger.info('Matching image features')
        sfm_in = tracker.key('sfm')
        features_in = tracker.key('features')
        pairs_file = args.matching_pairs_file
        if pairs_file is None or pairs_file.lower() == 'none':
            pairs_file = None
        elif pairs_file.lower() == 'auto':
            # The importer's view pairs file, from this run or an earlier job
            pairs_file = (tracker.path('view_pairs')
                          if args.import_pgs_scan else None)
        paths['view_pairs'] = pairs_file
        out_key = compute_matches(paths, sfm_key=sfm_in,
                                  method=args.matching_method,
                                  ratio=args.matching_ratio,
                                  pairs_file=pairs_file, metadata=metadata)
        tracker.end('matches', inputs={'sfm': paths[sfm_in],
                                       'features': paths[features_in],
                                       'view_pairs': pairs_file},
                    outputs={'matches': paths[out_key]})

    if tracker.begin('filter'):
        logger.info('Filtering image features')
        sfm_in = tracker.key('sfm')
        matches_in = tracker.key('matches')
        out_key = geometric_filter(paths, sfm_key=sfm_in,
                                   matches_key=matches_in,
                                   model=args.matching_geometric_model,
                                   metadata=metadata)
        tracker.end('filter', inputs={'sfm': paths[sfm_in],
                                      'matches': paths[matches_in]},
                    outputs={'matches_filtered': paths[out_key]})

    if tracker.begin('sfm'):
        in_key = tracker.key('sfm')
        features_in = tracker.key('features')
        inputs = {'sfm': paths[in_key], 'features': paths[features_in]}
        if args.mvg_recon_method == 'direct':
            # Triangulating known poses uses the unfiltered matches
            logger.info('Computing structure from known poses')
            matches_in = tracker.key('matches')
            inputs['matches'] = paths[matches_in]
            out_key = mvg_compute_known(paths, sfm_key=in_key,
                                        features_key=features_in,
                                        matches_key=matches_in, direct=True,
                                        bundle_adjustment=args.sfm_ba,
                                        metadata=metadata)
        else:
            logger.info(f'Running SfM (Engine: {args.mvg_recon_method})')
            filtered_in = tracker.key('matches_filtered')
            inputs['matches_filtered'] = paths[filtered_in]
            out_key = mvg_sfm(paths, sfm_key=in_key, features_key=features_in,
                              matches_key=filtered_in,
                              engine=args.mvg_recon_method,
                              use_priors=args.mvg_priors,
                              refine_intrinsics=args.mvg_refine_intrinsics,
                              initializer=args.mvg_initializer,
                              metadata=metadata)
        tracker.end('sfm', inputs=inputs, outputs={'sfm': paths[out_key]})

    if tracker.begin('robust'):
        logger.info('Performing robust triangulation')
        in_key = tracker.key('sfm')
        features_in = tracker.key('features')
        matches_in = tracker.key('matches')
        out_key = mvg_compute_known(paths, sfm_key=in_key,
                                    features_key=features_in,
                                    matches_key=matches_in,
                                    bundle_adjustment=args.robust_ba,
                                    metadata=metadata)
        tracker.end('robust', inputs={'sfm': paths[in_key],
                                      'features': paths[features_in],
                                      'matches': paths[matches_in]},
                    outputs={'sfm': paths[out_key]})

    if tracker.begin('autoscale'):
        logger.info('Auto-scaling SfM scene')
        in_key = tracker.key('sfm')
        out_key = mvg_autoscale(paths=paths, sfm_key=in_key,
                                marker_size=args.mvg_autoscale,
                                detection_method=args.autoscale_method,
                                marker_pix=args.autoscale_marker_pix,
                                include_from=args.autoscale_include_from,
                                exclude_from=args.autoscale_exclude_from,
                                metadata=metadata)
        tracker.end('autoscale', inputs={'sfm': paths[in_key]},
                    outputs={'sfm': paths[out_key]})

    # colorize is a leaf: it produces a side artifact, so its output is recorded
    # under its own role and the pre-colorize SfM stays the chained one.
    if tracker.begin('colorize'):
        logger.info('Colorizing SfM scene')
        in_key = tracker.key('sfm')
        out_key = mvg_colorize_sfm(paths, sfm_key=in_key, metadata=metadata)
        tracker.end('colorize', inputs={'sfm': paths[in_key]},
                    outputs={'colorized': paths[out_key]})

    if tracker.begin('convert'):
        logger.info('Converting MVG scene to MVS scene')
        in_key = tracker.key('sfm')
        out_key = mvg_to_mvs(paths, sfm_key=in_key, metadata=metadata,
                             threads=args.threads)
        tracker.end('convert', inputs={'sfm': paths[in_key]},
                    outputs={'scene': paths[out_key]})

    if tracker.begin('densify'):
        logger.info('Densifying point cloud')
        in_key = tracker.key('scene')
        scene_key, cloud_key = mvs_densify(
            paths, mvs_key=in_key,
            resolution_lvl=args.densify_resolution_level,
            mask_value=args.mask_value, metadata=metadata)
        tracker.end('densify', inputs={'scene': paths[in_key]},
                    outputs={'scene': paths[scene_key],
                             'cloud': paths[cloud_key]})

    if tracker.begin('reconstruct'):
        logger.info('Reconstructing mesh')
        in_key = tracker.key('scene')
        cloud_key = tracker.key('cloud') if tracker.has('cloud') else None
        # Both builders hand the scene back untouched, so only the mesh is
        # recorded as produced: see STAGE_IO on pass-through roles.
        _, mesh_key = mvs_reconstruct(
            paths, mvs_key=in_key, free_space=args.free_space_support,
            smooth=args.mvs_smooth, pointcloud_key=cloud_key,
            metadata=metadata)
        tracker.end('reconstruct',
                    inputs={'scene': paths[in_key],
                            'cloud': paths[cloud_key] if cloud_key else None},
                    outputs={'mesh': paths[mesh_key]})

    if tracker.begin('refine'):
        logger.info('Refining mesh')
        scene_in = tracker.key('scene')
        mesh_in = tracker.key('mesh')
        _, mesh_key = mvs_refine(
            paths, mvs_key=scene_in, mesh_key=mesh_in,
            decimation_factor=args.decimation_factor,
            resolution_lvl=args.refine_resolution_level,
            min_resolution=args.refine_min_resolution,
            scales=args.refine_scales, scale_step=args.refine_scale_step,
            metadata=metadata)
        tracker.end('refine',
                    inputs={'scene': paths[scene_in], 'mesh': paths[mesh_in]},
                    outputs={'mesh': paths[mesh_key]})

    if tracker.begin('texture'):
        logger.info('Texturing mesh')
        scene_in = tracker.key('scene')
        mesh_in = tracker.key('mesh')
        out_key = mvs_texture(paths, mvs_key=scene_in, mesh_key=mesh_in,
                              file_format=args.file_type,
                              resolution_lvl=args.texture_resolution_level,
                              max_size=args.texture_max_size,
                              metadata=metadata, output_name=args.name)
        tracker.end('texture',
                    inputs={'scene': paths[scene_in], 'mesh': paths[mesh_in]},
                    outputs={'mesh': paths[out_key]})


if __name__ == '__main__':
    main()
