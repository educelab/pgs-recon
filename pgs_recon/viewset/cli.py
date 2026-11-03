"""``pgs-localize-set``: localize a hand-held view set and texture each view.

See ADR 0013. Reads a finished ``pgs-recon`` output and writes only under
``--output``.
"""
import logging
import sys
from pathlib import Path

import configargparse

from pgs_recon import toolchain
from pgs_recon.utility import ToolFailed
from pgs_recon.utils.apps import setup_logging
from pgs_recon.utils.recon_dir import load_manifest, resolve_textured_mesh
from pgs_recon.viewset import pipeline
from pgs_recon.viewset.inputs import (NoPath, Overhead, PlateInputs,
                                      RigCapture, choose, collect_images)

logger = logging.getLogger('pgs-localize-set')


def main():
    """Entry point. A failed binary exits with *its* status, not 1."""
    try:
        _main()
    except ToolFailed as e:
        logger.error(f'{e}')
        sys.exit(e.exit_code)


def _parser() -> configargparse.ArgumentParser:
    parser = configargparse.ArgumentParser(
        prog='pgs-localize-set',
        description='Localize a set of hand-held images against a finished '
                    'reconstruction -- one calibration per view, solved with '
                    'a shared intrinsic -- and texture the mesh from each '
                    '(ADR 0013).')
    parser.add_argument('--config', '-c', is_config_file=True,
                        help='Config file path')

    plate = parser.add_argument_group('plate')
    plate.add_argument('--recon-dir', '-r', required=True,
                       help='A pgs-recon output directory. Read only.')
    plate.add_argument('--images', '-i', required=True,
                       help='Directory of hand-held images, one view each.')
    plate.add_argument('--output', '-o', required=True,
                       help='Where this run writes. Not --recon-dir.')
    plate.add_argument('--mesh', default=None,
                       help='Override the mesh to texture (default: '
                            '--ir-reference-mesh if given, else the '
                            'reconstruction\'s, from its manifest). Must be in '
                            'the frame of the overhead calibration.')
    plate.add_argument('--sfm-transform', default=None,
                       help='4x4 .npy into the centered frame, when --mesh is '
                            'the centered mesh (see pgs-retexture).')

    ref = parser.add_argument_group(
        'reference', 'What views are matched against (ADR 0013 Decision 2). '
                     'In order of preference; the first given is used.')
    ref.add_argument('--ir-reference-mesh', default=None,
                     help='A prebuilt IR940 capture-retexture mesh.')
    ref.add_argument('--rig-scan-dir', default=None,
                     help='The rig\'s PGS scan directory, to build the IR '
                          'reference by capture retexture.')
    ref.add_argument('--rig-ir-capture', type=int, default=None, metavar='n',
                     help='The IR940 capture index in --rig-scan-dir.')

    overhead = parser.add_argument_group(
        'overhead', 'The registered overhead spectral view: the prior source, '
                    'and the reference of last resort.')
    overhead.add_argument('--overhead-calibration', default=None,
                          help='Its pgs-localize one-view calibration .json.')
    overhead.add_argument('--overhead-image', default=None,
                          help='Its 940 nm band image.')
    overhead.add_argument('--virtual-overhead', action='store_true',
                          help='No registered overhead: render one from the IR '
                               'reference mesh, from straight above (the mesh '
                               'must be Z-up). Needs a reference mesh source; '
                               'the overhead QA comparison is not run.')

    solve = parser.add_argument_group('solve')
    solve.add_argument('--camera', default=None,
                       help='Starting K and distortion as a flat camera file: '
                            'from a board calibration or a previous run.')
    solve.add_argument('--fix-intrinsic', action='store_true',
                       help='Hold K at --camera (a board-calibrated camera).')
    solve.add_argument('--focal-sweep', default=None, metavar='lo:hi:steps',
                       help='The focal cost curve\'s range, in pixels.')
    solve.add_argument('--mask', default=None,
                       help='Image whose zero pixels are never matched (the '
                            'sample square).')
    solve.add_argument('--vignette-mask', action='store_true',
                       help='Make --mask from the lens vignette: the dark '
                            'region at the corners of the set\'s first image.')

    parser.add_argument('--skip-texture', action='store_true',
                        help='Stop after the calibrations.')
    parser.add_argument('--combined-texture', action='store_true',
                        help='Also texture one mesh from every localized view '
                             'at once (OpenMVS TextureMesh, a resampled atlas), '
                             'beside the per-view meshes. Keeps the --mask\'s '
                             'pixels out of it.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Resolve the inputs, say which reference and prior '
                             'would be used, and stop.')
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--path', type=str, default=None,
                        help=configargparse.SUPPRESS)
    parser.add_argument('--log-level', default='INFO', type=str.upper,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser


def _plate(args) -> PlateInputs:
    recon_dir = Path(args.recon_dir)
    load_manifest(recon_dir)  # -r must be a pgs-recon output
    # A provided reference is the frame every view is localized into (and the
    # one a virtual overhead is rendered from), so it is the mesh to texture
    # unless told otherwise; the manifest's mesh is only right when the
    # reference was retextured from it.
    if args.mesh:
        mesh = Path(args.mesh)
    elif args.ir_reference_mesh:
        mesh = Path(args.ir_reference_mesh)
    else:
        mesh = resolve_textured_mesh(recon_dir).require()

    if (args.rig_scan_dir is None) != (args.rig_ir_capture is None):
        sys.exit('--rig-scan-dir and --rig-ir-capture go together.')
    if (args.overhead_calibration is None) != (args.overhead_image is None):
        sys.exit('--overhead-calibration and --overhead-image go together.')
    if args.virtual_overhead and args.overhead_calibration:
        sys.exit('--virtual-overhead replaces --overhead-calibration and '
                 '--overhead-image; give one or the other.')

    return PlateInputs(
        recon_dir=recon_dir,
        mesh=mesh,
        images=collect_images(Path(args.images)),
        sfm_transform=Path(args.sfm_transform) if args.sfm_transform else None,
        ir_reference_mesh=(Path(args.ir_reference_mesh)
                           if args.ir_reference_mesh else None),
        rig_ir_capture=(RigCapture(Path(args.rig_scan_dir),
                                   args.rig_ir_capture)
                        if args.rig_scan_dir else None),
        overhead=(Overhead(Path(args.overhead_calibration),
                           Path(args.overhead_image))
                  if args.overhead_calibration else None),
        virtual_overhead=args.virtual_overhead,
    )


def _main():
    args = _parser().parse_args()
    setup_logging(args.log_level)
    toolchain.configure(prefix=args.path)

    output = Path(args.output).resolve()
    if output == Path(args.recon_dir).resolve():
        sys.exit('--output must not be --recon-dir: this run only reads the '
                 'reconstruction.')

    if args.fix_intrinsic and not args.camera:
        sys.exit('--fix-intrinsic holds K at --camera; give one.')
    if args.vignette_mask and args.mask:
        sys.exit('--vignette-mask makes the --mask; give one or the other.')
    if args.combined_texture and args.skip_texture:
        sys.exit('--combined-texture is a texture; it cannot run with '
                 '--skip-texture.')

    plate = _plate(args)
    if not plate.images:
        sys.exit(f'No images in {args.images}.')
    try:
        strategy = choose(plate)
    except NoPath as e:
        sys.exit(str(e))

    logger.info(f'{len(plate.images)} hand-held views')
    for line in strategy.reasons:
        logger.info(line)
    if args.dry_run:
        return

    pipeline.run(plate, strategy, output, pipeline.Options(
        camera=Path(args.camera) if args.camera else None,
        fix_intrinsic=args.fix_intrinsic,
        focal_sweep=args.focal_sweep,
        mask=Path(args.mask) if args.mask else None,
        vignette_mask=args.vignette_mask,
        threads=args.threads,
        skip_texture=args.skip_texture,
        combined_texture=args.combined_texture))


if __name__ == '__main__':
    main()
