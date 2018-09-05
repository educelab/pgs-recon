"""Run the photogrammetry pipeline on a set of input images."""

import argparse
import os
import subprocess


OPENMVG_SFM_BIN = 'build/openMVG-prefix/src/openMVG-build/linux-x86_64-Release'
CAMERA_SENSOR_WIDTH_DIRECTORY = 'build/openMVG-prefix/src/openMVG/src/openMVG/exif/sensor_width_database'
OPENMVS_BIN = 'build/openMVS-prefix/src/openMVS-build/bin'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='directory of input images')
    parser.add_argument('output', help='directory for output files')
    args = parser.parse_args()

    mvg_dir = os.path.join(args.output, 'openMVG')
    matches_dir = os.path.join(mvg_dir, 'matches')
    reconstruction_dir = os.path.join(mvg_dir, 'reconstruction_global')
    mvs_dir = os.path.join(args.output, 'openMVS')

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
        '-i', input_dir,
        '-o', matches_dir,
        '-d', camera_file_params,
    ])
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeFeatures'),
        '-i', os.path.join(matches_dir, 'sfm_data.json'),
        '-o', matches_dir,
        '-m', 'SIFT',
    ])
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_ComputeMatches'),
        '-i', os.path.join(matches_dir, 'sfm_data.json'),
        '-o', matches_dir,
        '-g', 'e',
    ])
    commands.append([
        os.path.join(OPENMVG_SFM_BIN, 'openMVG_main_GlobalSfM'),
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
        'build/openMVG-prefix/src/openMVG-build/Linux-x86_64-Release/openMVG_main_openMVG2openMVS',
        '-i', os.path.join(args.output, 'openMVG', 'reconstruction_global', 'sfm_data.bin'),
        '-o', os.path.join(args.output, 'openMVS', 'scene.mvs'),
        '-d', os.path.join(args.output, 'openMVG', 'undistorted_images'),
    ])
    commands.append([
        os.path.join(OPENMVS_BIN, 'DensifyPointCloud'),
        os.path.join(args.output, 'openMVS', 'scene.mvs'),
        '-w', os.path.join(args.output, 'openMVS', 'working')
    ])
    commands.append([
        os.path.join(OPENMVS_BIN, 'ReconstructMesh'),
        os.path.join(args.output, 'openMVS', 'scene_dense.mvs'),
        '-w', os.path.join(args.output, 'openMVS', 'working'),
    ])
    commands.append([
        os.path.join(OPENMVS_BIN, 'RefineMesh'),
        os.path.join(args.output, 'openMVS', 'scene_dense_mesh.mvs'),
        '-w', os.path.join(args.output, 'openMVS', 'working'),
        '--use-cuda', '0',  # https://github.com/cdcseacave/openMVS/issues/230
    ])
    commands.append([
        os.path.join(OPENMVS_BIN, 'TextureMesh'),
        os.path.join(args.output, 'openMVS', 'scene_dense_mesh_refine.mvs'),
        '-w', os.path.join(args.output, 'openMVS', 'working'),
    ])

    for command in commands:
        print(' '.join(command))
        subprocess.run(command)


if __name__ == '__main__':
    main()
