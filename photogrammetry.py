"""Run the photogrammetry pipeline on a set of input images."""

import argparse
import datetime
import json
import os
import subprocess
import sys


REPO_DIR = os.path.dirname(os.path.realpath(__file__))
if sys.platform == 'darwin':
    OPENMVG_SFM_BIN = os.path.join(REPO_DIR, 'build/openMVG-prefix/src/openMVG-build/Darwin-x86_64-Release')
else:
    OPENMVG_SFM_BIN = os.path.join(REPO_DIR, 'build/openMVG-prefix/src/openMVG-build/Linux-x86_64-Release')
CAMERA_SENSOR_WIDTH_DIRECTORY = os.path.join(REPO_DIR, 'build/openMVG-prefix/src/openMVG/src/openMVG/exif/sensor_width_database')
OPENMVS_BIN = os.path.join(REPO_DIR, 'build/openMVS-prefix/src/openMVS-build/bin')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='directory of input images')
    parser.add_argument('output', help='directory for output files')
    parser.add_argument('--focal-length', '-f', type=int, default=None, help='focal length in pixels', metavar='n')
    parser.add_argument('--video-mode-matching', '-v', type=int, default=None, help='sequence matching with an overlap of X images')
    parser.add_argument('--rclone-transfer-remote', metavar='remote', default=None,
                        help='if specified, and if matches the name of one of the directories in '
                        'the output path, transfer the results to that rclone remote into the '
                        'subpath following the remote name')
    parser.add_argument('--incremental-sfm', '-i', action='store_true', help='use incremental SfM instead of global')
    parser.add_argument('--free-space-support', action='store_true', help='use free-space support in ReconstructMesh')
    parser.add_argument('--densify-resolution-level', default=None, type=int, help='how many times to scale down images before DensifyPointCloud')
    parser.add_argument('--refine-resolution-level', default=None, type=int, help='how many times to scale down images before RefineMesh')
    parser.add_argument('--texture-resolution-level', default=None, type=int, help='how many times to scale down images before TextureMesh')
    parser.add_argument('--matching-geometric-model', default=None, help='type of model used for robust estimation from the photometric putative matches')
    args = parser.parse_args()

    output_path = os.path.join(
        args.output,
        datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
    )
    os.makedirs(output_path)

    metadata = {}
    metadata['Arguments'] = vars(args)

    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4, sort_keys=False))    

    mvg_dir = os.path.join(output_path, 'openMVG')
    matches_dir = os.path.join(mvg_dir, 'matches')
    reconstruction_dir = os.path.join(mvg_dir, 'reconstruction_global')
    mvs_dir = os.path.join(output_path, 'openMVS')

    if not os.path.exists(matches_dir):
        os.makedirs(matches_dir)
    if not os.path.exists(reconstruction_dir):
        os.makedirs(reconstruction_dir)
    if not os.path.exists(mvs_dir):
        os.makedirs(mvs_dir)

    camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, 'sensor_width_camera_database.txt')

    commands = []
    # https://openmvg.readthedocs.io/en/latest/software/SfM/SfM/
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_SfMInit_ImageListing'),
        '-i', args.input,
        '-o', matches_dir,
        '-d', camera_file_params,
    ])
    if args.focal_length is not None:
        commands[-1] += ['-f', str(args.focal_length)]
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeFeatures'),
        '-i', os.path.join(matches_dir, 'sfm_data.json'),
        '-o', matches_dir,
        '-m', 'SIFT',
    ])
    if not args.incremental_sfm:
        commands[-1] += ['-p', 'HIGH']  # https://openmvg.readthedocs.io/en/latest/software/SfM/GlobalSfM/?highlight=please%20use

    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeMatches'),
        '-i', os.path.join(matches_dir, 'sfm_data.json'),
        '-o', matches_dir,
    ])
    if args.matching_geometric_model is not None:
        commands[-1] += ['-g', args.matching_geometric_model]
    if args.video_mode_matching is not None:
        commands[-1] += ['-v', str(args.video_mode_matching)]

    if args.incremental_sfm:
        sfm_binary = 'openMVG_main_IncrementalSfM'
    else:
        sfm_binary = 'openMVG_main_GlobalSfM'
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, sfm_binary),
        '-i', os.path.join(matches_dir, 'sfm_data.json'),
        '-m', matches_dir,
        '-o', reconstruction_dir,
    ])
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeSfM_DataColor'),
        '-i', os.path.join(reconstruction_dir, 'sfm_data.bin'),
        '-o', os.path.join(reconstruction_dir, 'colorized.ply'),
    ])
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeStructureFromKnownPoses'),
        '-i', os.path.join(reconstruction_dir, 'sfm_data.bin'),
        '-m', matches_dir,
        '-f', os.path.join(matches_dir, 'matches.e.bin'),
        '-o', os.path.join(reconstruction_dir,'robust.bin'),
    ])
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeSfM_DataColor'),
        '-i', os.path.join(reconstruction_dir, 'robust.bin'),
        '-o', os.path.join(reconstruction_dir, 'robust_colorized.ply'),
    ])

    # https://github.com/cdcseacave/openMVS/wiki/Usage
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_openMVG2openMVS'),
        '-i', os.path.join(output_path, 'openMVG', 'reconstruction_global', 'sfm_data.bin'),
        '-o', os.path.join(output_path, 'openMVS', 'scene.mvs'),
        '-d', os.path.join(output_path, 'openMVG', 'undistorted_images'),
    ])
    
    commands.append([
        os.path.join(OPENMVS_BIN, 'DensifyPointCloud'),
        os.path.join(output_path, 'openMVS', 'scene.mvs'),
        '-w', os.path.join(output_path, 'openMVS', 'working'),
    ])
    if args.densify_resolution_level is not None:
        commands[-1] += ['--resolution-level', str(args.densify_resolution_level)]
        
    commands.append([
        os.path.join(OPENMVS_BIN, 'ReconstructMesh'),
        os.path.join(output_path, 'openMVS', 'scene_dense.mvs'),
        '-w', os.path.join(output_path, 'openMVS', 'working'),
    ])
    if args.free_space_support:
        commands[-1] += ['--free-space-support', '1']

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

    commands.append([
        os.path.join(OPENMVS_BIN, 'RefineMesh'),
        os.path.join(output_path, 'openMVS', 'scene_dense_mesh.mvs'),
        '-w', os.path.join(output_path, 'openMVS', 'working'),
    ])
    if built_with_cuda:
        commands[-1] += ['--use-cuda', '0']  # https://github.com/cdcseacave/openMVS/issues/230
    if args.refine_resolution_level is not None:
        commands[-1] += ['--resolution-level', str(args.refine_resolution_level)]

    commands.append([
        os.path.join(OPENMVS_BIN, 'TextureMesh'),
        os.path.join(output_path, 'openMVS', 'scene_dense_mesh_refine.mvs'),
        '-w', os.path.join(output_path, 'openMVS', 'working'),
    ])
    if args.texture_resolution_level is not None:
        commands[-1] += ['--resolution-level', str(args.texture_resolution_level)]

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

    for command in commands:
        print(' '.join(command))
        subprocess.run(command)


if __name__ == '__main__':
    main()
