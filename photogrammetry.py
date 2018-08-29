"""Run the photogrammetry pipeline on a set of input images."""

import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='directory of input images')
    parser.add_argument('output', help='directory for output files')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    subprocess.run([
        'python3',
        'build/openMVG-prefix/src/openMVG-build/software/SfM/SfM_GlobalPipeline.py',
        args.input,
        os.path.join(args.output, 'openMVG')
    ])


if __name__ == '__main__':
    main()
