"""Run the photogrammetry pipeline on a set of input images."""

import argparse
import atexit
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Dict, List

from pgs_recon.sfm import import_pgs_scan, load_cam_calib, load_cam_db, SfMFormat


def _current_timestamp_str() -> str:
    return datetime.now(timezone.utc).strftime("%m/%d/%Y, %H:%M:%S.%f %Z")


def _run_command(cmd: List[str], cwd=None):
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except OSError as e:
        print(f'Error: Failed to start command: {" ".join(cmd)}')
        sys.exit(f'{e.args}')
    except subprocess.SubprocessError as e:
        print(f'Error: Command failed: {" ".join(cmd)}')
        sys.exit(f'{e.args}')
    except:
        sys.exit(f'Unexpected error: {sys.exc_info()[0]}')


# Init an SfM from a PGS Scan
def init_sfm_pgs(paths: Dict[str, Path], metadata: Dict = None):
    # Load the camera db
    cam_db = load_cam_db(paths['CAM_DB'])

    # Load the calib if provided
    calib = None
    if 'input_calib' in paths.keys():
        print('Loading camera calibrations...')
        if metadata is not None:
            metadata['commands'][_current_timestamp_str()] = 'load_cam_calib ' + str(paths['input_calib'])
        calib = load_cam_calib(paths['input_calib'])

    # Load the pgs file
    sfm = import_pgs_scan(paths['input'].resolve(), cam_db=cam_db, cam_calib=calib)
    if metadata is not None:
        cmd = ' '.join(['import_pgs_scan', str(paths["input"]), str(paths['CAM_DB'])])
        if calib:
            cmd += ' ' + str(paths['input_calib'])
        metadata['commands'][_current_timestamp_str()] = cmd

    # Write the SFM
    sfm.save(str(paths['sfm']), fmt=SfMFormat.OPEN_MVG)


# Init sfm scene from dir of images
def init_sfm_generic(paths: Dict[str, Path], focal_length=None, metadata: Dict = None):
    command = [
        str(paths['BIN'] / 'openMVG_main_SfMInit_ImageListing'),
        '-i', str(paths['input'].resolve()),
        '-o', str(paths['mvg']),
        '-d', str(paths['CAM_DB']),
    ]
    if focal_length is not None:
        command.extend(['-f', str(focal_length)])
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command)


# MVG: Compute image features
def compute_features(paths: Dict[str, Path], method: str, preset: str, upright=False, threads: int = None,
                     metadata: Dict = None):
    # Compute features
    command = [
        str(paths['BIN'] / 'openMVG_main_ComputeFeatures'),
        '-i', str(paths['sfm']),
        '-o', str(paths['matches_dir']),
        '-m', method,
        '-p', preset,
    ]
    if upright:
        command.extend(['-u', '1'])
    if threads is not None:
        command.extend(['-n', str(threads)])
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command)


# MVG: Compute image feature matches
def compute_matches(paths: Dict[str, Path], method: str, model: str = None, video_frames: int = None,
                    metadata: Dict = None):
    command = [
        str(paths['BIN'] / 'openMVG_main_ComputeMatches'),
        '-i', str(paths['sfm']),
        '-o', str(paths['matches_dir']),
        '-n', method,
    ]
    if model is not None:
        command.extend(['-g', model])
    if video_frames is not None:
        command.extend(['-v', str(video_frames)])
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command)


# MVG: Run SfM
def mvg_sfm(paths: Dict[str, Path], sfm_key: str, method: str, use_priors=False, refine_intrinsics: str = None, initializer: str= None,
            metadata: Dict = None) -> str:
    if method == 'global':
        sfm_binary = 'openMVG_main_GlobalSfM'
    elif method == 'incremental':
        sfm_binary = 'openMVG_main_IncrementalSfM'
    elif method == 'incremental2':
        sfm_binary = 'openMVG_main_IncrementalSfM2'
    else:
        raise ValueError(f'Invalid SfM reconstruction method: {method}')

    command = [
        str(paths['BIN'] / sfm_binary),
        '-i', str(paths[sfm_key]),
        '-m', str(paths['matches_dir']),
        '-o', str(paths['recon_dir']),
        '-M', str(paths['matches_file']),
    ]
    if use_priors:
        command.append('-P')
        if method == 'incremental2':
            command.extend(['-S', 'EXISTING_POSE'])
    if refine_intrinsics is not None:
        command.extend(['-f', refine_intrinsics])
    if initializer is not None:
        command.extend(['-S', initializer])
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command)
    paths['sfm_recon'] = paths['recon_dir'] / 'sfm_data.bin'
    return 'sfm_recon'


# MVG: Compute structure from known poses (direct/robust)
def mvg_compute_known(paths: Dict[str, Path], sfm_key: str, direct: bool = False, bundle_adjustment: bool = False,
                      metadata: Dict = None) -> str:
    out_key = sfm_key + '_structured'
    in_path = paths[sfm_key]
    paths[out_key] = paths['recon_dir'] / (in_path.stem + '_structured.bin')
    command = [
        str(paths['BIN'] / 'openMVG_main_ComputeStructureFromKnownPoses'),
        '-i', str(paths[sfm_key]),
        '-m', str(paths['matches_dir']),
        '-o', str(paths[out_key]),
        '-f', str(paths['matches_file']),
    ]
    if direct:
        command.append('-d')
    if bundle_adjustment:
        command.append('-b')
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command)
    return out_key


# MVG: Colorize SfM file
def mvg_colorize_sfm(paths: Dict[str, Path], sfm_key: str, metadata: Dict = None) -> str:
    sfm_colorized_key = sfm_key + '_colorized'
    in_path = paths[sfm_key]
    paths[sfm_colorized_key] = in_path.parent / (in_path.stem + '_colorized.ply')
    command = [
        str(paths['BIN'] / 'openMVG_main_ComputeSfM_DataColor'),
        '-i', str(paths[sfm_key]),
        '-o', str(paths[sfm_colorized_key]),
    ]
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command)
    return sfm_colorized_key


# Convert OpenMVG SfM to OpenMVS Scene
def mvg_to_mvs(paths: Dict[str, Path], sfm_key: str, threads: int = None, metadata: Dict = None) -> str:
    command = [
        str(paths['BIN'] / 'openMVG_main_openMVG2openMVS'),
        '-i', str(paths[sfm_key].resolve()),
        '-o', str(paths['mvs_scene'].name),
        '-d', str(paths['mvs_images'].name)
    ]
    if threads is not None:
        command.extend(['-n', str(threads)])
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command, cwd=paths['mvs'])
    return 'mvs_scene'


# MVS: Densify a point cloud
def mvs_densify(paths: Dict[str, Path], mvs_key: str, resolution_lvl: int = None, metadata: Dict = None) -> str:
    out_key = mvs_key + '_dense'
    in_path = paths[mvs_key]
    paths[out_key] = in_path.parent / (in_path.stem + '_dense.mvs')
    command = [
        str(paths['MVS_BIN'] / 'DensifyPointCloud'),
        '-i', str(paths[mvs_key].name),
        '-o', str(paths[out_key].name),
        '-w', str(paths['mvs']),
    ]
    if resolution_lvl is not None:
        command.extend(['--resolution-level', str(resolution_lvl)])
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command)
    return out_key


# MVS: Reconstruct an MVS scene
def mvs_reconstruct(paths: Dict[str, Path], mvs_key: str, free_space=False, metadata: Dict = None) -> str:
    out_key = mvs_key + '_mesh'
    in_path = paths[mvs_key]
    paths[out_key] = in_path.parent / (in_path.stem + '_mesh.mvs')
    command = [
        str(paths['MVS_BIN'] / 'ReconstructMesh'),
        '-i', str(paths[mvs_key].name),
        '-o', str(paths[out_key].name),
        '-w', str(paths['mvs'])
    ]
    if free_space:
        command.extend(['--free-space-support', '1'])
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command)
    return out_key


# MVS: Refine a reconstructed mesh
def mvs_refine(paths: Dict[str, Path], mvs_key: str, decimation_factor: float = None, resolution_lvl: int = None,
               metadata: Dict = None) -> str:
    out_key = mvs_key + '_refine'
    in_path = paths[mvs_key]
    paths[out_key] = in_path.parent / (in_path.stem + '_refine.mvs')
    command = [
        str(paths['MVS_BIN'] / 'RefineMesh'),
        '-i', str(paths[mvs_key].name),
        '-o', str(paths[out_key].name),
        '-w', str(paths['mvs'])
    ]
    if decimation_factor is not None:
        command.extend(['--decimate', str(decimation_factor)])
    if resolution_lvl is not None:
        command.extend(['--resolution-level', str(resolution_lvl)])
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command)
    return out_key


# MVS: Texture a mesh
def mvs_texture(paths: Dict[str, Path], mvs_key: str, file_format: str = 'ply', resolution_lvl: int = None,
                metadata: Dict = None) -> str:
    out_key = mvs_key + '_texture'
    in_path = paths[mvs_key]
    paths[out_key] = in_path.parent / (in_path.stem + f'_texture.{file_format.lower()}')
    command = [
        str(paths['MVS_BIN'] / 'TextureMesh'),
        '-i', str(paths[mvs_key].name),
        '-o', str(paths[out_key].name),
        '--export-type', file_format.lower(),
        '-w', str(paths['mvs'])
    ]
    if resolution_lvl is not None:
        command.extend(['--resolution-level', str(resolution_lvl)])
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command)
    return out_key


# Copy a file or directory to an rclone remote
def rclone_copy(source: Path, target: PurePath, metadata: Dict = None):
    if source.is_dir():
        target_path = target / source.name
    else:
        target_path = target
    command = [
        'rclone',
        'copy',
        '-P',
        str(source),
        str(target_path),
    ]
    if metadata is not None:
        metadata['commands'][_current_timestamp_str()] = (str(' ').join(command))
    _run_command(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', '-i', required=True, help='directory of input images')
    parser.add_argument('--output', '-o', required=True, help='directory for output files')
    parser.add_argument('--focal-length', '-f', type=int, default=None, help='focal length in pixels', metavar='n')
    parser.add_argument('--import-pgs-scan', '-p', action='store_true',
                        help='Input directory is assumed to be a PGS Scan directory')
    parser.add_argument('--import-calib', type=str, help='When importing a PGS Scan, merge provided PGS calibration '
                                                         'file with the imported camera configurations.')

    # Hidden opts
    parser.add_argument('--path', type=str, default='/usr/local/', help=argparse.SUPPRESS)
    parser.add_argument('--threads', type=int, help=argparse.SUPPRESS)

    opts_desc = parser.add_argument_group('describer options')
    opts_desc.add_argument('--describer-method', choices=['SIFT', 'AKAZE_FLOAT', 'AKAZE_MLDB'], default="SIFT",
                           type=str.upper,
                           help="Set the feature descriptors method.")
    opts_desc.add_argument('--describer-preset', choices=['NORMAL', 'HIGH', 'ULTRA'], default='HIGH', type=str.upper,
                           help='Set the description detail level.')
    opts_desc.add_argument('--describer-upright', '-u', action='store_true',
                           help='Disable rotational invariance for feature detection step. Useful if '
                                'the camera is always "upright" w.r.t the ground plane.')

    opts_matcher = parser.add_argument_group('matcher options')
    opts_matcher.add_argument('--matching-method',
                              choices=['AUTO', 'BRUTEFORCEL2', 'ANNL2', 'CASCADEHASHINGL2', 'FASTCASCADEHASHINGL2',
                                       'BRUTEFORCEHAMMING'], default="FASTCASCADEHASHINGL2", type=str.upper,
                              help='Feature matching method.')
    opts_matcher.add_argument('--matching-geometric-model', choices=['f', 'e', 'h'], default='f', type=str.lower,
                              help='Geometric model for robust putative matches filtering: f: Fundamental, '
                                   'e: Essential, h: Homography')
    opts_matcher.add_argument('--matching-video-mode', '-v', type=int, default=None,
                              help='sequence matching with an overlap of X images')

    opts_mvg = parser.add_argument_group('mvg reconstruction options')
    opts_mvg.add_argument('--mvg-recon-method', '-m', choices=['global', 'incremental', 'incremental2', 'direct'],
                          type=str.lower, default='global',
                          help='MVG scene reconstruction method. Note: "direct" requires an sfm file '
                               'with camera poses.')
    opts_mvg.add_argument('--mvg-priors', action='store_true', help='Use pose priors with SfM reconstruction')
    opts_mvg.add_argument('--mvg-refine-intrinsics', type=str.upper,
                          help='SfM intrinsic refinement options: NONE, ADJUST_FOCAL_LENGTH, ADJUST_PRINCIPAL_POINT, '
                               'ADJUST_DISTORTION, ADJUST_ALL. Note: Quoted options can be combined with \'|\' (e.g. '
                               '\'ADJUST_FOCAL_LENGTH|ADJUST_DISTORTION\')')
    opts_mvg.add_argument('--mvg-robust', '-r', action='store_true',
                          help='robustly triangulate reconstructed scene')

    # MVG hidden opts
    opts_mvg.add_argument('--sfm-ba', action='store_true', help=argparse.SUPPRESS)
    opts_mvg.add_argument('--robust-ba', action='store_true', help=argparse.SUPPRESS)
    opts_mvg.add_argument('--mvg-initializer', choices=['EXISTING_POSE', 'MAX_PAIR', 'AUTO_PAIR', 'STELLAR'], type=str.upper, help=argparse.SUPPRESS)

    opts_mvs = parser.add_argument_group('openmvs options')
    opts_mvs.add_argument('--free-space-support', action='store_true', help='use free-space support in ReconstructMesh')
    opts_mvs.add_argument('--mvs-densify', action='store_true', help='Enable point cloud densification step')
    opts_mvs.add_argument('--densify-resolution-level', default=None, type=int,
                          help='how many times to scale down images before DensifyPointCloud')
    opts_mvs.add_argument('--refine-resolution-level', default=None, type=int,
                          help='how many times to scale down images before RefineMesh')
    opts_mvs.add_argument('--texture-resolution-level', default=None, type=int,
                          help='how many times to scale down images before TextureMesh')
    opts_mvs.add_argument('--decimation-factor', type=float,
                          help='Decimation factor in range [0..1] to be applied '
                               'to the input surface before mesh refinement '
                               '(0 - auto, 1 - disabled)')

    opts_adv = parser.add_argument_group('advanced options')
    opts_adv.add_argument('--rclone-copy', metavar='remote', default=None,
                          help='Uses rclone copy to transfer the generated project to an rclone remote. '
                               'Must be in the format \'remote:/path/\'.')
    args = parser.parse_args()

    # SETUP
    # Structure for storing important paths
    paths = {
        'PATH': Path(args.path).resolve(),
        'input': Path(args.input),
        'output': Path(args.output)
    }
    if args.import_calib:
        paths['input_calib'] = Path(args.import_calib)
    paths['BIN'] = paths['PATH'] / 'bin'
    paths['MVS_BIN'] = paths['BIN'] / 'OpenMVS'
    paths['CAM_DB'] = paths['PATH'] / 'share/openMVG/sensor_width_camera_database.txt'

    # Setup output directory names
    paths['mvg'] = paths['output'] / 'mvg'
    paths['matches_dir'] = paths['mvg'] / 'matches_dir'
    paths['recon_dir'] = paths['mvg'] / 'recon_dir'
    paths['sfm'] = paths['mvg'] / 'sfm_data.json'
    paths['mvs'] = paths['output'] / 'mvs'
    paths['mvs_scene'] = paths['mvs'] / 'scene.mvs'
    paths['mvs_images'] = paths['mvs'] / 'undistorted_images'

    # Generate the matches file path from the geometric model
    model = args.matching_geometric_model.lower()
    if model in ("f", "a"):
        matches_file = 'matches.f.bin'
    else:
        matches_file = f'matches.{model}.bin'
    paths['matches_file'] = paths['matches_dir'] / matches_file

    # Create output folders
    for d in 'output', 'mvg', 'matches_dir', 'recon_dir', 'mvs':
        paths[d].mkdir(exist_ok=True, parents=True)

    # Init metadata
    metadata = {'args': " ".join(sys.argv), 'parsed': vars(args),
                'paths': {key: str(val) for key, val in paths.items()}, 'commands': {}}
    paths['metadata'] = paths['output'] / 'metadata.json'

    # Register a metadata write whenever the program closes
    @atexit.register
    def write_metadata():
        metadata['paths'] = {key: str(val) for key, val in paths.items()}
        metadata['commands'][_current_timestamp_str()] = "Writing metadata"
        with paths['metadata'].open('w') as meta_file:
            meta_file.write(json.dumps(metadata, indent=4, sort_keys=False))

    # Write metadata once
    write_metadata()

    # Initialize the mvg project
    if args.import_pgs_scan:
        init_sfm_pgs(paths, metadata=metadata)
    else:
        init_sfm_generic(paths, focal_length=args.focal_length, metadata=metadata)

    # Compute features
    compute_features(paths, method=args.describer_method, preset=args.describer_preset, upright=args.describer_upright,
                     metadata=metadata, threads=args.threads)

    # Match features/Compute Structure
    compute_matches(paths, method=args.matching_method, model=args.matching_geometric_model,
                    video_frames=args.matching_video_mode, metadata=metadata)

    # MVG scene computation
    if args.mvg_recon_method == 'direct':
        sfm_key = mvg_compute_known(paths, sfm_key='sfm', direct=True, bundle_adjustment=args.sfm_ba, metadata=metadata)
    else:
        sfm_key = mvg_sfm(paths, sfm_key='sfm', method=args.mvg_recon_method, use_priors=args.mvg_priors,
                          refine_intrinsics=args.mvg_refine_intrinsics, initializer=args.mvg_initializer, metadata=metadata)

    # Colorize reconstructed scene
    mvg_colorize_sfm(paths, sfm_key=sfm_key, metadata=metadata)

    # Robust SfM
    if args.mvg_robust:
        sfm_key = mvg_compute_known(paths, sfm_key=sfm_key, bundle_adjustment=args.robust_ba, metadata=metadata)
        mvg_colorize_sfm(paths, sfm_key=sfm_key, metadata=metadata)

    # Convert MVG -> MVS
    mvs_key = mvg_to_mvs(paths, sfm_key=sfm_key, metadata=metadata, threads=args.threads)

    # Densify MVS Scene
    if args.mvs_densify:
        mvs_key = mvs_densify(paths, mvs_key=mvs_key, resolution_lvl=args.densify_resolution_level, metadata=metadata)

    # Reconstruct Scene Mesh
    mvs_key = mvs_reconstruct(paths, mvs_key=mvs_key, free_space=args.free_space_support, metadata=metadata)

    # Refine Mesh
    mvs_key = mvs_refine(paths, mvs_key=mvs_key, decimation_factor=args.decimation_factor,
                         resolution_lvl=args.refine_resolution_level, metadata=metadata)

    # Texture Mesh
    mvs_texture(paths, mvs_key=mvs_key, file_format='ply', resolution_lvl=args.texture_resolution_level,
                metadata=metadata)

    # Transfer via rclone if requested
    if args.rclone_copy is not None:
        rclone_copy(paths['output'], PurePath(args.rclone_copy), metadata=metadata)

    # Add final timestamp
    metadata['commands'][_current_timestamp_str()] = "Processing complete"


if __name__ == '__main__':
    main()
