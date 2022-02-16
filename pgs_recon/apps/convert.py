import json
import logging
from pathlib import Path

import configargparse

from pgs_recon.utility import run_command


def main():
    parser = configargparse.ArgumentParser(prog='pgs-convert')
    parser.add_argument('--config', '-c', is_config_file=True,
                        help='Config file path')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input PGS dataset directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output PGS dataset directory')

    parser.add_argument('--file-type', '-f', choices=['jpg', 'tif'],
                        default='jpg', type=str.lower,
                        help='Output image format')
    parser.add_argument('--quality', '-q', type=int,
                        help='Output image quality. Range depends on '
                             '--file-type.')

    parser.add_argument('--exposure', type=float,
                        help='Exposure adjustment +/-')
    parser.add_argument('--shadows', type=float, help='Shadow adjustment +/-')

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
        return 1

    # Load the scan metadata
    meta_path = scan_dir / 'metadata.json'
    if not meta_path.exists():
        logger.error(f'File not found: {str(meta_path)}')
        return 1
    with meta_path.open() as f:
        meta = json.loads(f.read())

    # Get a list of images
    prefix = meta['scan']['file_prefix']
    ext = meta['scan']['format'].lower()
    images = list(scan_dir.glob(f'{prefix}*.{ext}'))
    images.sort()
    if len(images) == 0:
        logger.error('No images found in directory.')

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
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(['-path', str(output_dir)])

    # Add input file list
    cmd.extend([str(i) for i in images])

    # Modify the metadata and save to out dir
    meta['adjustment'] = conv_meta
    meta['scan']['output_dir'] = str(output_dir.resolve())
    meta['scan']['format'] = args.file_type.upper()
    meta_path = output_dir / 'metadata.json'
    with meta_path.open('w') as f:
        json.dump(meta, f, indent=4)

    # Convert images
    logger.info('Converting data...')
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
    run_command(cmd)


if __name__ == '__main__':
    main()
