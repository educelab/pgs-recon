"""Run the photogrammetry pipeline on a set of input images."""
import argparse
import atexit
import json
import logging
import sys
from datetime import datetime as dt, timezone as tz
from pathlib import Path

import configargparse

from pgs_recon.openmvg import (compute_features, compute_matches,
                               geometric_filter, init_sfm_generic,
                               mvg_colorize_sfm, mvg_compute_known, mvg_sfm,
                               mvg_to_mvs)
from pgs_recon.openmvs import (mvs_densify, mvs_reconstruct, mvs_refine,
                               mvs_texture)
from pgs_recon.pgs_data import init_sfm_pgs
from pgs_recon.utility import current_timestamp


def main():
    parser = configargparse.ArgumentParser(prog='pgs-recon')

    # Generic options
    parser.add_argument('--config', '-c', is_config_file=True,
                        help='Config file path')
    parser.add_argument('--input', '-i', required=True,
                        help='directory of input images')
    parser.add_argument('--output', '-o', required=True,
                        help='directory for output files')
    parser.add_argument('--name', '-n', type=str, help='Experiment name.')
    parser.add_argument('--file-type', choices=['ply', 'obj'],
                        default='obj', type=str.lower,
                        help='Output format for final textured mesh')
    parser.add_argument('--focal-length', '-f', type=int, default=None,
                        help='focal length in pixels', metavar='n')
    parser.add_argument('--import-pgs-scan', '-p',
                        action=argparse.BooleanOptionalAction,
                        help='Input directory is assumed to be a PGS Scan '
                             'directory')
    parser.add_argument('--import-calib', type=str,
                        help='When importing a PGS Scan, merge provided PGS '
                             'calibration file with the imported camera '
                             'configurations.')

    # Hidden opts
    parser.add_argument('--path', type=str, default='/usr/local/',
                        help=configargparse.SUPPRESS)
    parser.add_argument('--threads', type=int, help=configargparse.SUPPRESS)

    opts_desc = parser.add_argument_group('describer options')
    opts_desc.add_argument('--describer-method',
                           choices=['SIFT', 'AKAZE_FLOAT', 'AKAZE_MLDB'],
                           default="SIFT",
                           type=str.upper,
                           help="Set the feature descriptors method.")
    opts_desc.add_argument('--describer-preset',
                           choices=['NORMAL', 'HIGH', 'ULTRA'], default='HIGH',
                           type=str.upper,
                           help='Set the description detail level.')
    opts_desc.add_argument('--describer-upright', '-u',
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
                              default="FASTCASCADEHASHINGL2", type=str.upper,
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

    opts_mvg = parser.add_argument_group('mvg reconstruction options')
    opts_mvg.add_argument('--mvg-recon-method', '-m',
                          choices=['global',
                                   'incremental',
                                   'incrementalv2',
                                   'direct'],
                          type=str.lower, default='global',
                          help='MVG scene reconstruction method. Note: direct '
                               'requires an sfm file with camera poses.')
    opts_mvg.add_argument('--mvg-priors', action=argparse.BooleanOptionalAction,
                          help='Use pose priors with SfM reconstruction')
    opts_mvg.add_argument('--mvg-refine-intrinsics', type=str.upper,
                          help='SfM intrinsic refinement options: NONE, '
                               'ADJUST_FOCAL_LENGTH, ADJUST_PRINCIPAL_POINT, '
                               'ADJUST_DISTORTION, ADJUST_ALL. '
                               'Note: Quoted options can be combined with '
                               '\'|\' (e.g. '
                               '\'ADJUST_FOCAL_LENGTH|ADJUST_DISTORTION\')')
    opts_mvg.add_argument('--mvg-robust', '-r',
                          action=argparse.BooleanOptionalAction,
                          help='robustly triangulate reconstructed scene')

    # MVG hidden opts
    opts_mvg.add_argument('--sfm-ba', action=argparse.BooleanOptionalAction,
                          help=configargparse.SUPPRESS)
    opts_mvg.add_argument('--robust-ba', action=argparse.BooleanOptionalAction,
                          help=configargparse.SUPPRESS)
    opts_mvg.add_argument('--mvg-initializer',
                          choices=['EXISTING_POSE', 'MAX_PAIR', 'AUTO_PAIR',
                                   'STELLAR'],
                          type=str.upper, help=configargparse.SUPPRESS)

    opts_mvs = parser.add_argument_group('openmvs options')
    opts_mvs.add_argument('--free-space-support',
                          action=argparse.BooleanOptionalAction,
                          help='use free-space support in ReconstructMesh')
    opts_mvs.add_argument('--mvs-densify',
                          action=argparse.BooleanOptionalAction,
                          help='Enable point cloud densification step')
    opts_mvs.add_argument('--mvs-smooth', type=int, default=2,
                          help='Number of smoothing iterations after initial '
                               'surface reconstruction. 0 is disabled.')
    opts_mvs.add_argument('--densify-resolution-level', default=None, type=int,
                          help='how many times to scale down images before '
                               'DensifyPointCloud')
    opts_mvs.add_argument('--refine-resolution-level', default=None, type=int,
                          help='how many times to scale down images before '
                               'RefineMesh')
    opts_mvs.add_argument('--texture-resolution-level', default=None, type=int,
                          help='how many times to scale down images before '
                               'TextureMesh')
    opts_mvs.add_argument('--decimation-factor', type=float,
                          help='Decimation factor in range [0..1] to be '
                               'applied to the input surface before mesh '
                               'refinement (0 - auto, 1 - disabled)')
    args = parser.parse_args()

    logger = logging.getLogger("pgs-recon")

    # Structure for storing important paths
    logger.info('Setting up output directories')
    paths = {
        'PATH': Path(args.path).resolve(),
        'input': Path(args.input),
        'output': Path(args.output)
    }
    if args.import_calib:
        paths['input_calib'] = Path(args.import_calib)
    paths['BIN'] = paths['PATH'] / 'bin'
    paths['MVS_BIN'] = paths['BIN'] / 'OpenMVS'
    db_path = 'share/openMVG/sensor_width_camera_database.txt'
    paths['CAM_DB'] = paths['PATH'] / db_path

    # Setup output directory names
    paths['mvg'] = paths['output'] / 'mvg'
    paths['matches_dir'] = paths['mvg'] / 'matches_dir'
    paths['matches_file'] = paths['matches_dir'] / 'matches.bin'
    paths['recon_dir'] = paths['mvg'] / 'recon_dir'
    paths['sfm'] = paths['mvg'] / 'sfm_data.json'
    paths['mvs'] = paths['output'] / 'mvs'
    paths['mvs_scene'] = paths['mvs'] / 'scene.mvs'
    paths['mvs_images'] = paths['mvs'] / 'undistorted_images'

    # Create output folders
    for d in 'output', 'mvg', 'matches_dir', 'recon_dir', 'mvs':
        paths[d].mkdir(exist_ok=True, parents=True)

    # Setup experiment
    experiment_start = dt.now(tz.utc)
    datetime_str = experiment_start.strftime('%Y%m%d%H%M%S')
    if args.name is None:
        args.name = datetime_str + '_' + str(Path(args.input).stem)

    # Write config after all arguments have been changed
    config = paths['output'] / f'{datetime_str}_{args.name}_recon_config.txt'
    args.config = str(config)
    with config.open(mode='w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            arg = arg.replace('_', '-')
            file.write(f'{arg} = {attr}\n')

    # Init metadata
    metadata = {'args': " ".join(sys.argv),
                'parsed': vars(args),
                'paths': {key: str(val) for key, val in paths.items()},
                'commands': {}}
    paths['metadata'] = paths['output'] / 'metadata.json'

    # Register a metadata write whenever the program closes
    @atexit.register
    def write_metadata():
        metadata['paths'] = {key: str(val) for key, val in paths.items()}
        metadata['commands'][current_timestamp()] = "Writing metadata"
        with paths['metadata'].open('w') as meta_file:
            meta_file.write(json.dumps(metadata, indent=4, sort_keys=False))

    # Write metadata once
    write_metadata()

    # Initialize the mvg project
    logger.info('Importing dataset')
    if args.import_pgs_scan:
        init_sfm_pgs(paths, metadata=metadata)
    else:
        init_sfm_generic(paths, focal_length=args.focal_length,
                         metadata=metadata)

    # Compute features
    logger.info('Computing image features')
    compute_features(paths, method=args.describer_method,
                     preset=args.describer_preset,
                     upright=args.describer_upright,
                     metadata=metadata, threads=args.threads)

    # Match features/Compute Structure
    logger.info('Matching image features')
    compute_matches(paths, method=args.matching_method,
                    ratio=args.matching_ratio, metadata=metadata)

    # Filter matches
    logger.info('Filtering image features')
    geometric_filter(paths, model=args.matching_geometric_model,
                     metadata=metadata)

    # MVG scene computation
    if args.mvg_recon_method == 'direct':
        logger.info('Computing structure from known poses')
        sfm_key = mvg_compute_known(paths, sfm_key='sfm', direct=True,
                                    bundle_adjustment=args.sfm_ba,
                                    metadata=metadata)
    else:
        logger.info(f'Running SfM (Engine: {args.mvg_recon_method}')
        sfm_key = mvg_sfm(paths, sfm_key='sfm', engine=args.mvg_recon_method,
                          use_priors=args.mvg_priors,
                          refine_intrinsics=args.mvg_refine_intrinsics,
                          initializer=args.mvg_initializer,
                          metadata=metadata)

    # Colorize reconstructed scene
    logger.info('Colorizing SfM scene')
    mvg_colorize_sfm(paths, sfm_key=sfm_key, metadata=metadata)

    # Robust SfM
    if args.mvg_robust:
        logger.info('Performing robust triangulation')
        sfm_key = mvg_compute_known(paths, sfm_key=sfm_key,
                                    bundle_adjustment=args.robust_ba,
                                    metadata=metadata)
        logger.info('Colorizing SfM scene')
        mvg_colorize_sfm(paths, sfm_key=sfm_key, metadata=metadata)

    # Convert MVG -> MVS
    logger.info('Converting MVG scene to MVS scene')
    mvs_key = mvg_to_mvs(paths, sfm_key=sfm_key, metadata=metadata,
                         threads=args.threads)

    # Densify MVS Scene
    if args.mvs_densify:
        logger.info('Densifying point cloud')
        mvs_key = mvs_densify(paths, mvs_key=mvs_key,
                              resolution_lvl=args.densify_resolution_level,
                              metadata=metadata)

    # Reconstruct Scene Mesh
    logger.info('Reconstructing mesh')
    mvs_key = mvs_reconstruct(paths, mvs_key=mvs_key,
                              free_space=args.free_space_support,
                              smooth=args.mvs_smooth,
                              metadata=metadata)

    # Refine Mesh
    logger.info('Refining mesh')
    mvs_key = mvs_refine(paths, mvs_key=mvs_key,
                         decimation_factor=args.decimation_factor,
                         resolution_lvl=args.refine_resolution_level,
                         metadata=metadata)

    # Texture Mesh
    logger.info('Texturing mesh')
    mvs_texture(paths, mvs_key=mvs_key, file_format=args.file_type,
                resolution_lvl=args.texture_resolution_level,
                metadata=metadata, output_name=args.name)

    # Add final timestamp
    current_time = current_timestamp()
    metadata['commands'][current_time] = "Processing complete"
    logger.info(f'Processing complete. Results saved to: {paths["output"]}')


if __name__ == '__main__':
    main()
