import argparse
import sys

import imageio.v3 as iio
import matplotlib
import numpy as np
from educelab import imgproc

DEFAULT_COLORMAP = 'viridis'
DEFAULT_CLIP = (2.0, 98.0)


def main():
    parser = argparse.ArgumentParser(
        description='Render a colormapped, 8-bit preview of a floating-point '
                    'depth map for human inspection. The source depth map '
                    'remains the artifact to measure against: the preview is '
                    'stretched per-image, so the same color in two previews '
                    'does not mean the same depth.')
    parser.add_argument('--input-file', '-i', required=True,
                        help='Input depth map (floating-point image)')
    parser.add_argument('--output-file', '-o', required=True,
                        help='Output preview image. Use a format with an alpha '
                             'channel (e.g. png) to preserve the no-data mask.')
    parser.add_argument('--colormap', default=DEFAULT_COLORMAP,
                        help=f'Matplotlib colormap name. Prefer a perceptually '
                             f'uniform map, since the ramp is read as depth '
                             f'(default: {DEFAULT_COLORMAP})')
    parser.add_argument('--clip', nargs=2, type=float, default=DEFAULT_CLIP,
                        metavar=('LOW', 'HIGH'),
                        help='Percentile range stretched across the colormap. '
                             'Taken over the valid pixels only, so a handful of '
                             'outliers cannot flatten the useful range '
                             f'(default: {DEFAULT_CLIP[0]} {DEFAULT_CLIP[1]})')
    args = parser.parse_args()

    if args.colormap not in matplotlib.colormaps:
        sys.exit(f'unknown colormap: {args.colormap}')

    # Depth maps are commonly LZW-compressed TIFFs, which imageio can only
    # decode via imagecodecs — hence that package being a hard requirement
    # rather than an optional extra.
    print('Loading depth map...')
    try:
        depth = iio.imread(args.input_file)
    except (OSError, ValueError) as e:
        sys.exit(f'could not read depth map: {args.input_file}: {e}')
    if depth.ndim == 3:
        depth = depth[..., 0]

    # Depth maps carry a non-finite value wherever a pixel has no surface (for
    # rt_reorder_texture that is NaN, and it is a large share of a typical map).
    valid = np.isfinite(depth)
    if not valid.any():
        sys.exit(f'depth map has no finite values: {args.input_file}')

    print('Stretching depth range...')
    low, high = np.percentile(depth[valid], args.clip)
    # Neutralize the no-data pixels before stretching so they cannot poison the
    # rescale; the alpha channel below is what actually marks them.
    depth = np.where(valid, depth, low)
    if high > low:
        depth = imgproc.stretch(depth, low, high, out_range=(0.0, 1.0))
    else:
        # Degenerate (flat) depth map: put everything at the bottom of the ramp
        # rather than dividing by zero.
        depth = np.zeros(depth.shape, dtype=np.float32)

    print(f'Applying {args.colormap} colormap...')
    rgba = matplotlib.colormaps[args.colormap](depth, bytes=True)
    # No-data is carried in alpha instead of being colormapped, so that it does
    # not read as the shallowest measured depth.
    rgba[..., 3] = np.where(valid, 255, 0)

    print('Saving preview...')
    iio.imwrite(args.output_file, rgba)
    print(f'Wrote {rgba.shape[1]}x{rgba.shape[0]} preview: '
          f'{args.clip[0]}-{args.clip[1]}% maps [{low:.6g}, {high:.6g}], '
          f'{100 * float((~valid).mean()):.2f}% no-data')


if __name__ == '__main__':
    main()
