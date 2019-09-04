"""Run the photogrammetry pipeline on a set of input images."""

import argparse
import datetime
import json
import os
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.realpath(__file__))
OPENMVG_SFM_BIN = os.path.join(REPO_DIR, 'installed/bin')
OPENMVS_BIN = os.path.join(REPO_DIR, 'installed/bin/OpenMVS')
CAMERA_FILE_PARAMS = os.path.join(REPO_DIR, 'installed/share/openMVG/sensor_width_camera_database.txt')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='directory of input images')
    parser.add_argument('output', help='directory for output files')
    parser.add_argument('--focal-length', '-f', type=int, default=None, help='focal length in pixels', metavar='n')

    optsDescriber = parser.add_argument_group('describer options')
    optsDescriber.add_argument('--describer-method', default="SIFT", help="Set the describer method: SIFT, AKAZE_FLOAT, or AKAZE_MLDB")
    optsDescriber.add_argument('--describer-preset', default='HIGH', help='Set the describer detail level: NORMAL, HIGH, or ULTRA')
    optsDescriber.add_argument('--describer-upright', '-u', action='store_true',
                               help='Disable rotational invariance for feature detection step. Useful if '
                               'the camera is always "upright" w.r.t the ground plane.')

    optsMatcher = parser.add_argument_group('matcher options')
    optsMatcher.add_argument('--matching-method', default="FASTCASCADEHASHINGL2",
                             help='OpenMVG matching method: '
                             'AUTO, BRUTEFORCEL2, ANNL2, CASCADEHASHINGL2, '
                             'FASTCASCADEHASHINGL2, or BRUTEFORCEHAMMING')
    optsMatcher.add_argument('--matching-geometric-model', default=None, help='type of model used for robust estimation from the photometric putative matches')
    optsMatcher.add_argument('--matching-video-mode', '-v', type=int, default=None, help='sequence matching with an overlap of X images')

    optsSFM = parser.add_argument_group('sfm reconstruction options')
    optsSFM.add_argument('--sfm-use-incremental', '-i', action='store_true', help='use incremental SfM instead of global')
    optsSFM.add_argument('--sfm-use-robust', '-r', action='store_true', help='robustly triangulate corresponding features')

    optsMVS = parser.add_argument_group('openmvs options')
    optsMVS.add_argument('--free-space-support', action='store_true', help='use free-space support in ReconstructMesh')
    optsMVS.add_argument('--disable-densify', action='store_true', help='Disable point cloud densification step')
    optsMVS.add_argument('--densify-resolution-level', default=None, type=int, help='how many times to scale down images before DensifyPointCloud')
    optsMVS.add_argument('--refine-resolution-level', default=None, type=int, help='how many times to scale down images before RefineMesh')
    optsMVS.add_argument('--texture-resolution-level', default=None, type=int, help='how many times to scale down images before TextureMesh')
    optsMVS.add_argument('--decimation-factor', default='0', type=float,
                         help='Decimation factor in range [0..1] to be applied '
                         'to the input surface before mesh refinement '
                         '(0 - auto, 1 - disabled)')

    optsAdv = parser.add_argument_group('advanced options')
    optsAdv.add_argument('--rclone-transfer-remote', metavar='remote', default=None,
                        help='if specified, and if matches the name of one of the directories in '
                        'the output path, transfer the results to that rclone remote into the '
                        'subpath following the remote name')
    args = parser.parse_args()

    ### Setup ###
    # Create a timestamped output directory
    output_path = os.path.join(
        args.output,
        datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
    )
    os.makedirs(output_path)

    # Write cmd line parameters to file
    metadata = {}
    metadata['Arguments'] = vars(args)

    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4, sort_keys=False))

    # Setup output directory names
    mvg_dir = os.path.join(output_path, 'openMVG')
    matches_dir = os.path.join(mvg_dir, 'matches')
    reconstruction_dir = os.path.join(mvg_dir, 'reconstruction')
    mvs_dir = os.path.join(output_path, 'openMVS')

    # Create the matches file path from the geometric model
    matches_file = os.path.join(matches_dir, 'matches.f.bin')
    if args.matching_geometric_model is not None:
        model = args.matching_geometric_model.lower()
        if model in ("f", "a"):
            matches_file = os.path.join(matches_dir, 'matches.f.bin')
        else:
            matches_file = os.path.join(matches_dir, 'matches.' + model + '.bin')

    # Create output folders
    if not os.path.exists(matches_dir):
        os.makedirs(matches_dir)
    if not os.path.exists(reconstruction_dir):
        os.makedirs(reconstruction_dir)
    if not os.path.exists(mvs_dir):
        os.makedirs(mvs_dir)

    # Detect CUDA support
    built_with_cuda = False
    if os.path.isfile('build/openMVS-prefix/src/openMVS-build/CMakeCache.txt'):
        s = None
        with open('build/openMVS-prefix/src/openMVS-build/CMakeCache.txt') as f:
            for line in f:
                if 'OpenMVS_USE_CUDA:BOOL=' in line:
                    s = line
        if s is not None:
            if s.endswith('ON'):
                built_with_cuda = True

    ### Command Generation ###
    commands = []
    # List images
    # https://openmvg.readthedocs.io/en/latest/software/SfM/SfM/
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_SfMInit_ImageListing'),
        '-i', args.input,
        '-o', matches_dir,
        '-d', CAMERA_FILE_PARAMS,
    ])
    if args.focal_length is not None:
        commands[-1] += ['-f', str(args.focal_length)]

    # Compute features
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeFeatures'),
        '-i', os.path.join(matches_dir, 'sfm_data.json'),
        '-o', matches_dir,
        '-m', args.describer_method,
        '-p', args.describer_preset,
    ])
    if args.describer_upright:
        commands[-1] += ['-u', '1']

    # Match features
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeMatches'),
        '-i', os.path.join(matches_dir, 'sfm_data.json'),
        '-o', matches_dir,
        '-n', args.matching_method,
    ])
    if args.matching_geometric_model is not None:
        commands[-1] += ['-g', args.matching_geometric_model]
    if args.matching_video_mode is not None:
        commands[-1] += ['-v', str(args.matching_video_mode)]

    # SfM Computation
    if args.sfm_use_incremental:
        sfm_binary = 'openMVG_main_IncrementalSfM'
    else:
        sfm_binary = 'openMVG_main_GlobalSfM'
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, sfm_binary),
        '-i', os.path.join(matches_dir, 'sfm_data.json'),
        '-m', matches_dir,
        '-o', reconstruction_dir,
        '-M', matches_file,
    ])

    # Colorize SfM Result
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeSfM_DataColor'),
        '-i', os.path.join(reconstruction_dir, 'sfm_data.bin'),
        '-o', os.path.join(reconstruction_dir, 'colorized.ply'),
    ])

    # Robust SfM
    if args.sfm_use_robust:
        commands.append([
            os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeStructureFromKnownPoses'),
            '-i', os.path.join(reconstruction_dir, 'sfm_data.bin'),
            '-m', matches_dir,
            '-f', matches_file,
            '-o', os.path.join(reconstruction_dir,'robust.bin'),
        ])

        commands.append([
            os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeSfM_DataColor'),
            '-i', os.path.join(reconstruction_dir, 'robust.bin'),
            '-o', os.path.join(reconstruction_dir, 'robust_colorized.ply'),
        ])
        reconFile = 'robust.bin'
    else:
        reconFile = 'sfm_data.bin'

    # Convert MVG -> MVS
    # https://github.com/cdcseacave/openMVS/wiki/Usage
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_openMVG2openMVS'),
        '-i', os.path.join(reconstruction_dir, reconFile),
        '-o', os.path.join(output_path, 'openMVS', 'scene.mvs'),
        '-d', os.path.join(output_path, 'openMVS', 'undistorted_images'),
    ])

    # Densify MVS Scene
    if args.disable_densify:
        mvsFile = 'scene'
    else:
        commands.append([
            os.path.join(OPENMVS_BIN, 'DensifyPointCloud'),
            os.path.join(output_path, 'openMVS', 'scene.mvs'),
            # '-w', os.path.join(output_path, 'openMVS', 'working'),
        ])
        if args.densify_resolution_level is not None:
            commands[-1] += ['--resolution-level', str(args.densify_resolution_level)]
        mvsFile = 'scene_dense'

    # Reconstruct Scene Mesh
    commands.append([
        os.path.join(OPENMVS_BIN, 'ReconstructMesh'),
        os.path.join(output_path, 'openMVS', mvsFile + '.mvs'),
        # '-w', os.path.join(output_path, 'openMVS', 'working'),
    ])
    if args.free_space_support:
        commands[-1] += ['--free-space-support', '1']
    mvsFile += '_mesh'

    # Refine Mesh
    commands.append([
        os.path.join(OPENMVS_BIN, 'RefineMesh'),
        os.path.join(output_path, 'openMVS', mvsFile + '.mvs'),
        '--decimate', str(args.decimation_factor),
        # '-w', os.path.join(output_path, 'openMVS', 'working'),
    ])
    if built_with_cuda:
        commands[-1] += ['--use-cuda', '0']  # https://github.com/cdcseacave/openMVS/issues/230
    if args.refine_resolution_level is not None:
        commands[-1] += ['--resolution-level', str(args.refine_resolution_level)]
    mvsFile += '_refine'

    # Texture Mesh
    commands.append([
        os.path.join(OPENMVS_BIN, 'TextureMesh'),
        os.path.join(output_path, 'openMVS', mvsFile + '.mvs'),
        # '-w', os.path.join(output_path, 'openMVS', 'working'),
    ])
    if args.texture_resolution_level is not None:
        commands[-1] += ['--resolution-level', str(args.texture_resolution_level)]
    mvsFile += '_texture'

    # Transfer via rclone if requested
    if args.rclone_transfer_remote is not None:
        folders = []
        path = os.path.abspath(output_path)
        while True:
            path, folder = os.path.split(path)
            if folder != "":
                folders.append(folder)
            else:
                if path != "":
                    folders.append(path)
                break
        folders.reverse()

        if args.rclone_transfer_remote not in folders:
            print('Provided rclone transfer remote was not a directory '
                  'name in the output path, so it is not clear where in the '
                  'remote to put the files. Transfer canceled.')
        else:
            while folders.pop(0) != args.rclone_transfer_remote:
                continue

            commands.append([
                'rclone',
                'move',
                '-v',
                '--delete-empty-src-dirs',
                output_path,
                args.rclone_transfer_remote + ':' + os.path.join(*folders)
            ])

    ### Execute commands ###
    for command in commands:
        print(' '.join(command))
        subprocess.run(command, check=True)

if __name__ == '__main__':
    main()
