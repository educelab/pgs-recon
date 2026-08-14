import argparse
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt, timezone as tz
from pathlib import Path

import configargparse
import imageio.v3 as iio
import numpy as np
from educelab import imgproc
from educelab.imgproc import pipeline
from tqdm import tqdm

from pgs_recon.utility import ToolFailed, run_command
from pgs_recon.utils.images import lab_encoding, read_srgb


def write_config(args, config_path=None):
    # Setup experiment
    experiment_start = dt.now(tz.utc)
    datetime_str = experiment_start.strftime('%Y%m%d_%H%M%S')
    args.name = datetime_str + '_' + str(Path(args.input).stem)

    # Write config after all arguments have been changed
    if config_path is None:
        config_path = Path(
            args.output) / f'{datetime_str}_{args.name}_convert_config.txt'
    args.config = str(config_path)
    with config_path.open(mode='w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            arg = arg.replace('_', '-')
            file.write(f'{arg} = {attr}\n')


def main():
    """Entry point. A failed binary exits with *its* status, not 1."""
    try:
        _main()
    except ToolFailed as e:
        logging.getLogger('pgs-convert').error(f'{e}')
        sys.exit(e.exit_code)


def _main():
    parser = configargparse.ArgumentParser(prog='pgs-convert')
    parser.add_argument('--config', '-c', is_config_file=True,
                        help='Config file path')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input PGS dataset directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output PGS dataset directory')
    parser.add_argument('--log-level', type=str.upper,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='WARNING', help='Logging level')

    convert_opts = parser.add_argument_group('conversion options')
    convert_opts.add_argument('--file-type', '-f', choices=['jpg', 'tif'],
                              default='jpg', type=str.lower,
                              help='Output image format')
    convert_opts.add_argument('--if-same-type', default='copy',
                              choices=['skip', 'copy', 'convert'],
                              help='Behavior to use when the input file type '
                                   'matches the target: (skip) the dataset, '
                                   '(copy) directly to the output directory, '
                                   '(convert) files anyway. Files are always '
                                   'converted if one of the enhancement '
                                   'options is provided, and CIE L*a*b* files '
                                   'are always converted to sRGB regardless.')
    convert_opts.add_argument('--force-copy', default=False,
                              action=argparse.BooleanOptionalAction,
                              help='When performing a dataset copy, ignore '
                                   'files which would be overwritten in the '
                                   'output directory')
    convert_opts.add_argument('--quality', '-q', type=int,
                              help='Output image quality. Range depends on '
                                   '--file-type')

    file_opts = parser.add_argument_group('file filter options')
    file_opts.add_argument('--filter-cam', type=int, metavar='INT',
                           help='Filter by camera index')
    file_opts.add_argument('--filter-pos', type=int, metavar='INT',
                           help='Filter by position index')
    file_opts.add_argument('--filter-cap', type=int, metavar='INT',
                           help='Filter by capture index')

    perf_opts = parser.add_argument_group('performance options')
    perf_opts.add_argument('--threads', '-t', type=int,
                           help="Maximum number of threads to use when "
                                "converting images")

    # add the enhancement pipeline options
    pipeline.add_parser_enhancement_group(parser)

    # parse arguments and commands
    args = parser.parse_args()
    apply_pipeline, cmds = pipeline.parse_and_build(args.commands)

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger('pgs-convert')

    # If we're on a SLURM node, os.cpu_count returns the hardware CPUs, which is
    # not necessarily what's available to the job. To avoid deadlocks, override
    # the default with what's actually usable.
    if args.threads is None and 'SLURM_JOB_CPUS_PER_NODE' in os.environ.keys():
        args.threads = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    logger.debug(f'Max worker threads: {"auto" if args.threads is None else args.threads}')

    # Validate the input directory
    scan_dir = Path(args.input)
    if not scan_dir.exists():
        logger.error(f'Input directory does not exist: {str(scan_dir)}')
        sys.exit(1)

    # Get output directory
    output_dir = Path(args.output)

    # Load the scan metadata
    meta_path = scan_dir / 'metadata.json'
    if not meta_path.exists():
        logger.error(f'File not found: {str(meta_path)}')
        sys.exit(1)
    with meta_path.open(encoding='utf-8') as f:
        meta = json.loads(f.read())

    # Get file name info
    prefix = meta['scan']['file_prefix']
    ext = meta['scan']['format'].lower()

    # File filter
    cam_f = f'{args.filter_cam:03}' if args.filter_cam is not None else '*'
    pos_f = f'_{args.filter_pos:05}' if args.filter_pos is not None else '_*'
    cap_f = f'_{args.filter_cap:02}' if args.filter_cap is not None else '_*'
    suffix = f'{cam_f}{pos_f}{cap_f}'

    # Get a list of images. Listed before the pass-through branches below only
    # so colorspaces can be probed; an empty set is still their business.
    images = list(scan_dir.glob(f'{prefix}{suffix}.{ext}'))
    images.sort()

    # Whether a file already sRGB in the requested format can simply be moved,
    # unread and un-re-encoded. Gated on the pipeline that was built rather than
    # on whether --commands was given, so a --commands that parses to nothing
    # passes through too.
    fmt_match = ext == args.file_type
    pass_through = fmt_match and not cmds and args.if_same_type != 'convert'

    # OpenMVG reads sRGB, not L*a*b*, so no Lab file may reach the output
    # untouched -- which disqualifies the whole-dataset shortcuts below. The
    # colorspace is decided per file (a set can mix the two), so this only asks
    # which files force the slow path; the rest are still passed through. One
    # header read each, threaded: a scan lives on a share often enough that the
    # latency, not the parsing, is what this costs.
    lab_images = []
    if pass_through:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            lab_images = [p for p, enc in zip(images,
                                              executor.map(lab_encoding, images))
                          if enc is not None]
    if lab_images:
        logger.info(f'Input contains {len(lab_images)} CIE L*a*b* image(s). '
                    f'Converting those to sRGB and copying the rest.')

    def copy_dataset(hold_back=frozenset()):
        """The scan tree wholesale, minus ``hold_back``. Exits on a populated
        output directory, which is what --force-copy waives."""
        try:
            shutil.copytree(scan_dir, output_dir,
                            ignore=lambda _d, names: hold_back & set(names),
                            dirs_exist_ok=args.force_copy)
        except FileExistsError as e:
            logger.error(e)
            sys.exit(1)

    # Handle matching format
    if pass_through and not lab_images:
        # Format matches and not copying
        if args.if_same_type == 'skip':
            logger.info('Input dataset matches requested format. '
                        'Data will not be copied or converted.')
            sys.exit(0)

        # Format matches and copying directly
        elif args.if_same_type == 'copy':
            logger.info('Input dataset matches requested format. '
                        'Copying to the output directory.')
            copy_dataset()
            write_config(args)
            sys.exit(0)

    if len(images) == 0:
        logger.error('No images found in directory.')
        sys.exit(1)

    # A mixed set still owes --if-same-type copy what a whole-dataset copy
    # promises: the sidecar files a per-image loop never looks at, and the
    # refusal to write over a populated output directory. So everything but the
    # Lab images is copied here, wholesale, and only those are converted below.
    # A filtered run stays file by file, though: it asked for a subset, and the
    # tree would bring back everything it excluded -- including Lab images this
    # pass never probed, which is the one thing that must not reach the output.
    filtered = any(f is not None for f in (args.filter_cam, args.filter_pos,
                                           args.filter_cap))
    copy_rest = (bool(lab_images) and args.if_same_type == 'copy'
                 and not filtered)
    if copy_rest:
        copy_dataset({p.name for p in lab_images})
    to_convert = lab_images if copy_rest else images
    lab_paths = set(lab_images)

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write config before convert
    write_config(args)

    # Modify the metadata and save to out dir
    meta['scan']['output_dir'] = str(output_dir.resolve())
    meta['scan']['format'] = args.file_type.upper()
    if len(cmds):
        meta['enhancements'] = cmds
    meta_path = output_dir / 'metadata.json'
    with meta_path.open('w', encoding='utf8') as f:
        json.dump(meta, f, indent=4)

    # Define conversion function
    def convert_image(p) -> tuple[bool, Path]:
        # Already sRGB in the requested format, and this run is only opening
        # files for the sake of the Lab ones -- the --if-same-type skip case,
        # where no copy ran above. Move the bytes, keeping the original quality.
        if pass_through and p not in lab_paths:
            shutil.copy2(p, output_dir / p.name)
            return True, p.name

        # Load image, as [0, 1] floats whatever the file stored. A reader hands
        # back L*a*b* samples undecoded, so scaling them as if they were RGB is
        # what miscolors the output; read_srgb decodes instead, landing in the
        # same floats the enhancement pipeline expects.
        try:
            image = read_srgb(p)
        except (OSError, ValueError) as e:
            logger.error(f'Failed to read {str(p)} as sRGB: {e}')
            return False, p.name
        in_dtype, img = image.dtype, image.pixels

        # Process the image
        img = apply_pipeline(img)

        # Determine output format
        kwargs = {}

        # Type conversion
        if args.file_type == 'jpg':
            out_dtype = np.uint8
        else:
            out_dtype = in_dtype
        img = np.clip(img, 0., 1.)
        img = imgproc.as_dtype(img, out_dtype)

        # Format specific opts
        if args.file_type == 'jpg':
            kwargs[
                'quality'] = args.quality if args.quality is not None else 100
        elif args.file_type == 'tif':
            kwargs['compression'] = 'zlib'
            kwargs['compressionargs'] = {'level': 9}

        # Save the image to disk
        out_file = p.with_suffix(f'.{args.file_type}').name
        out_path = output_dir / out_file
        iio.imwrite(out_path, img, **kwargs)
        return True, p.name

    # Convert images (single-threaded)
    results = []
    if args.threads == 1:
        for p in tqdm(to_convert, desc='Converting images'):
            results.append(convert_image(p))
    # Convert images (multithreaded)
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = executor.map(convert_image, to_convert)
            results = list(
                tqdm(futures, total=len(to_convert), desc='Converting images'))
    # Report success. A dataset where nothing converted is a failed run, not a
    # quiet one: the output would otherwise be an empty (or copy-only) directory
    # that only the next stage of the pipeline discovers is unusable.
    failed_files = [r[1] for r in results if not r[0]]
    if len(failed_files) == len(results):
        logger.error(f'All {len(results)} images failed to convert.')
        sys.exit(1)
    if len(failed_files) > 0:
        logger.warning(f'{len(failed_files)} images failed to convert.')
        meta['conversion'] = {'failed': failed_files}
        with meta_path.open('w', encoding='utf8') as f:
            json.dump(meta, f, indent=4)

    # Setup metadata copy
    cmd = ['exiftool', '-q', '-P', '-overwrite_original']

    # Original tags from original files
    # Use dummy _ to get OS separator then strip dummy _
    meta_dir = str(scan_dir / '_')[:-1]
    cmd.extend(['-TagsFromFile', f'{meta_dir}%f.{ext}'])

    # Map all the other tags
    cmd.append('-all:all')

    # Everything below excludes tags from that copy (--TAG) rather than deleting
    # them after it (-TAG=): exiftool performs deletions *before* it copies, so
    # the -XMP-tiff:all= that used to sit ahead of -TagsFromFile removed nothing
    # the copy then put back. Excluding also leaves a passed-through file's own
    # tags alone, which deleting would not.

    # The output is sRGB whatever the source was, so nothing describing the
    # source's colorspace may ride along -- a Lab file's white point on an sRGB
    # one says the pixels are something they are not. (PhotometricInterpretation
    # itself exiftool protects, and will not write.)
    for tag in ('PhotometricInterpretation', 'WhitePoint',
                'PrimaryChromaticities', 'ReferenceBlackWhite'):
        cmd.extend([f'--IFD0:{tag}', f'--XMP-tiff:{tag}'])

    # Skip tags that don't make sense in JPGs
    if args.file_type == 'jpg':
        cmd.extend(['--XMP-tiff:all', '--ExifIFD:BitsPerSample',
                    '--IFD0:BitsPerSample'])

    # Say so positively, since writing EXIF at all makes exiftool fill in its
    # mandatory ColorSpace, whose default is "Uncalibrated".
    cmd.append('-ExifIFD:ColorSpace=sRGB')

    # Iterate over the output dir
    cmd.append(str(output_dir))

    # Copy metadata
    logger.info('Copying metadata...')
    logger.debug(f'Metadata args: {cmd}')
    run_command(cmd)


if __name__ == '__main__':
    main()
