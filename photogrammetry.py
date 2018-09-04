"""Run the photogrammetry pipeline on a set of input images."""

import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='directory of input images')
    parser.add_argument('output', help='directory for output files')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output, 'openMVS')):
        os.makedirs(os.path.join(args.output, 'openMVS'))

    commands = [
        # https://openmvg.readthedocs.io/en/latest/software/SfM/SfM/
        [
            'python',
            'build/openMVG-prefix/src/openMVG-build/software/SfM/SfM_GlobalPipeline.py',
            args.input,
            os.path.join(args.output, 'openMVG')
        ],

        # https://github.com/cdcseacave/openMVS/wiki/Usage
        [
            'build/openMVG-prefix/src/openMVG-build/Linux-x86_64-Release/openMVG_main_openMVG2openMVS',
            '-i', os.path.join(args.output, 'openMVG', 'reconstruction_global', 'sfm_data.bin'),
            '-o', os.path.join(args.output, 'openMVS', 'scene.mvs'),
            '-d', os.path.join(args.output, 'openMVG', 'undistorted_images'),
        ],
        [
            'build/openMVS-prefix/src/openMVS-build/bin/DensifyPointCloud',
            os.path.join(args.output, 'openMVS', 'scene.mvs'),
            '-w', os.path.join(args.output, 'openMVS', 'working')
        ],
        [
            'build/openMVS-prefix/src/openMVS-build/bin/ReconstructMesh',
            os.path.join(args.output, 'openMVS', 'scene_dense.mvs'),
            '-w', os.path.join(args.output, 'openMVS', 'working'),
        ],
        [
            'build/openMVS-prefix/src/openMVS-build/bin/RefineMesh',
            os.path.join(args.output, 'openMVS', 'scene_dense_mesh.mvs'),
            '-w', os.path.join(args.output, 'openMVS', 'working'),
            '--use-cuda', '0',  # https://github.com/cdcseacave/openMVS/issues/230
        ],
        [
            'build/openMVS-prefix/src/openMVS-build/bin/TextureMesh',
            os.path.join(args.output, 'openMVS', 'scene_dense_mesh_refine.mvs'),
            '-w', os.path.join(args.output, 'openMVS', 'working'),
        ]
    ]

    for command in commands:
        print(command)
        subprocess.run(command)


if __name__ == '__main__':
    main()
