"""Image normalization shared by the texturing tools.

Kept out of the apps because both ``pgs-retexture`` and ``pgs-calibrate`` have to
hand OpenMVG/OpenMVS something they can actually read, and the capture formats
they are pointed at (16-bit, CIELab TIFFs) are not it.
"""
import logging
import sys
from pathlib import Path

from pgs_recon.utility import run_command

logger = logging.getLogger(__name__)


def prepare_8bit_image(src: Path, out_dir: Path) -> Path:
    """Write an 8-bit sRGB copy of a single image via ImageMagick ``convert``.

    Unlike ``pgs-retexture``'s ``convert_modality_images`` (which keeps a *set*
    of frames mutually consistent with a uniform bit-shift for atlas texturing),
    this handles ONE standalone image that becomes its own texture, or the query
    image ``pgs-calibrate`` extracts features from. ImageMagick reads the
    embedded colorspace and bit depth, so it correctly handles 16-bit and non-RGB
    inputs such as the EduceLab CIELab TIFFs (which OpenCV would misread
    channel-for-channel). Per-image tone mapping is fine here because each image
    is textured (or localized) independently. Returns the output path.

    A failed ``convert`` propagates as ``ToolFailed``: both callers
    (``pgs-retexture``, ``pgs-calibrate``) catch it in ``main()``, so nothing is
    gained by handling it here and the child's exit status stays intact.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'{src.stem}.jpg'
    command = [
        'convert', str(src.resolve()),
        '-colorspace', 'sRGB', '-depth', '8', '-type', 'TrueColor',
        '-quality', '100', str(out.resolve()),
    ]
    run_command(command)
    if not out.is_file():
        sys.exit(f'Failed to prepare 8-bit image from {src}')
    logger.info(f'Prepared 8-bit image: {out}')
    return out
