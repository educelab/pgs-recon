import argparse
import json
import logging
import shutil
import sys
from datetime import datetime as dt, timezone as tz
from pathlib import Path

import configargparse

from pgs_recon.utility import run_command


def write_config(args, config_path=None):
    # Setup experiment
    experiment_start = dt.now(tz.utc)
    datetime_str = experiment_start.strftime('%Y%m%d%H%M%S')
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


def has_group_opt(args, grp):
    return any([getattr(args, b.dest) is not None for b in grp._group_actions])


def main():
    parser = configargparse.ArgumentParser(prog='pgs-convert')
    parser.add_argument('--config', '-c', is_config_file=True,
                        help='Config file path')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input PGS dataset directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output PGS dataset directory')

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
                                   'options is provided.')
    convert_opts.add_argument('--force-copy', default=False,
                              action=argparse.BooleanOptionalAction,
                              help='When performing a dataset copy, ignore '
                                   'files which would be overwritten in the '
                                   'output directory')
    convert_opts.add_argument('--quality', '-q', type=int,
                              help='Output image quality. Range depends on '
                                   '--file-type')

    file_opts = parser.add_argument_group('file filter options')
    file_opts.add_argument('--filter-cam', type=int, metavar='INT', help='Filter by camera index')
    file_opts.add_argument('--filter-pos', type=int, metavar='INT', help='Filter by position index')
    file_opts.add_argument('--filter-cap', type=int, metavar='INT', help='Filter by capture index')

    enhance_opts = parser.add_argument_group('enhancement options')
    enhance_opts.add_argument('--exposure', type=float,
                              help='Exposure adjustment +/-')
    enhance_opts.add_argument('--shadows', type=float,
                              help='Shadow adjustment +/-')

    parser.add_argument('--log-level', type=str.upper,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='WARNING', help='Logging level')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger('pgs-convert')

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

    # Handle matching format
    fmt_match = ext == args.file_type
    has_enhance_opt = has_group_opt(args, enhance_opts)
    if fmt_match and not has_enhance_opt:
        # Format matches and not copying
        if args.if_same_type == 'skip':
            logger.info('Input dataset matches requested format. '
                        'Data will not be copied or converted.')
            sys.exit(0)

        # Format matches and copying directly
        elif args.if_same_type == 'copy':
            logger.info('Input dataset matches requested format. '
                        'Copying to the output directory.')
            try:
                shutil.copytree(scan_dir, output_dir,
                                dirs_exist_ok=args.force_copy)
            except FileExistsError as e:
                logger.error(e)
                sys.exit(1)
            write_config(args)
            sys.exit(0)

    # File filter
    cam_f = f'{args.filter_cam:03}' if args.filter_cam else '*'
    pos_f = f'_{args.filter_pos:05}' if args.filter_pos else '_*'
    cap_f = f'_{args.filter_cap:02}' if args.filter_cap else '_*'
    suffix = f'{cam_f}{pos_f}{cap_f}'

    # Get a list of images
    images = list(scan_dir.glob(f'{prefix}{suffix}.{ext}'))
    images.sort()
    if len(images) == 0:
        logger.error('No images found in directory.')
        sys.exit(1)

    # Construct command
    cmd = ['mogrify']

    # Conversion metadata
    conv_meta = {}

    # Bump exposure
    # Map stop factor -> % (e.g. +2.0 -> +200%)
    if args.exposure is not None:
        conv_meta['exposure'] = args.exposure
        exp_val = 100.0 + args.exposure * 10
        cmd.extend(['+level', f'0x{exp_val}%'])

    # Simulate Photoshop shadows with Magick's sigmoidal-contrast
    # Photoshop shadows [-100, 100] -> Magick contrast [-inf, inf]
    if args.shadows is not None:
        conv_meta['shadows'] = args.shadows
        prefix = '-'
        if args.shadows < 0:
            prefix = '+'
        shadow_val = abs(args.shadows / 10.0)
        cmd.extend([f'{prefix}sigmoidal-contrast', f'{shadow_val}x0%'])

    # File format
    cmd.extend(['-format', args.file_type])

    # Quality
    if args.quality is not None and args.file_type in ['jpg', 'png']:
        conv_meta['quality'] = args.quality
        cmd.extend(['-quality', str(args.quality)])

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(['-path', str(output_dir)])

    # Add input file list
    cmd.extend([str(i) for i in images])

    # Write config before convert
    write_config(args)

    # Modify the metadata and save to out dir
    meta['adjustment'] = conv_meta
    meta['scan']['output_dir'] = str(output_dir.resolve())
    meta['scan']['format'] = args.file_type.upper()
    meta_path = output_dir / 'metadata.json'
    with meta_path.open('w', encoding='utf8') as f:
        json.dump(meta, f, indent=4)

    # Convert images
    logger.info('Converting data...')
    logger.debug(f'Convert args: {cmd}')
    run_command(cmd)

    # Setup metadata copy
    cmd = ['exiftool', '-q', '-P', '-overwrite_original']

    # Skip tags that don't make sense in JPGs
    if args.file_type in ['jpg', 'png']:
        cmd.extend(['-XMP-tiff:all=', '-ExifIFD:BitsPerSample=',
                    '-IFD0:BitsPerSample='])

    # Original tags from original files
    # Use dummy _ to get OS separator then strip dummy _
    meta_dir = str(scan_dir / '_')[:-1]
    cmd.extend(['-TagsFromFile', f'{meta_dir}%f.{ext}'])

    # Map all the other tags
    cmd.append('-all:all')

    # Iterate over the output dir
    cmd.append(str(output_dir))

    # Copy metadata
    logger.info('Copying metadata...')
    logger.debug(f'Metadata args: {cmd}')
    run_command(cmd)


if __name__ == '__main__':
    main()
