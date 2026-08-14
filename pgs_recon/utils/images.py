"""The one place a capture file on disk becomes pixels.

``pgs-convert``, ``pgs-calibrate`` and ``pgs-retexture`` all have to hand
OpenMVG/OpenMVS something it can read, and the formats they are pointed at
(16-bit greyscale, CIELab TIFFs) are not it. They used to disagree about how:
``pgs-convert`` read in process, while the other two shelled out to ImageMagick
``convert``, which ignores a Lab file's WhitePoint tag and decodes every one of
them as D65. The EduceLab captures carry no such tag and are D50, so the same
file textured differently depending on which tool read it.

:func:`read_srgb` is now that single answer. It replaces ImageMagick exactly
where ImageMagick was right -- a 16-bit greyscale becomes ``round(v / 257)`` and
an 8-bit RGB passes through byte for byte, both verified against it -- and
differs only where it was wrong, on the white point. Nothing here shells out.
"""
import logging
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import tifffile
from skimage import img_as_float
from skimage.color import xyz2rgb

logger = logging.getLogger(__name__)

#: TIFF PhotometricInterpretation values whose samples are CIE L*a*b*, not RGB.
CIELAB, ICCLAB, ITULAB = 8, 9, 10
LAB_PHOTOMETRICS = frozenset((CIELAB, ICCLAB, ITULAB))

#: TIFF 6.0's default WhitePoint, used when a Lab file does not carry the tag.
D50 = (0.34567, 0.35850)

#: ICCLab's 16-bit L* reaches 100 at 0xFF00, not at full scale (ICC.1 6.3.4.2).
ICCLAB_L_MAX_16 = 0xff00

#: What a reader hands back for a Lab TIFF that declares SampleFormat = 2.
_UNSIGNED = {np.dtype(np.int8): np.dtype(np.uint8),
             np.dtype(np.int16): np.dtype(np.uint16)}

#: sRGB's white, and the one :func:`skimage.color.xyz2rgb` assumes.
D65_XYZ = np.array([0.95047, 1., 1.08883])

#: Bradford cone response, for adapting a file's white to D65.
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                      [-0.7502, 1.7135, 0.0367],
                      [0.0389, -0.0685, 1.0296]])


class LabEncoding(NamedTuple):
    """How one file stores L*a*b*: its photometric, and the white it is against."""
    photometric: int
    white_point: Tuple[float, float]


class SRGBImage(NamedTuple):
    """A capture read as sRGB, plus what it took to get there.

    ``dtype`` is what the *file* stored, not what ``pixels`` is; a caller
    choosing an output bit depth wants the former. ``decoded`` says a colorspace
    conversion ran, so ``pixels`` are no longer the stored samples rescaled --
    which is the difference between "drop the low bits of a 16-bit sample" and
    "these are already sRGB".
    """
    pixels: np.ndarray
    dtype: np.dtype
    decoded: bool


def read_srgb(path: Path) -> SRGBImage:
    """Read a capture as float sRGB in [0, 1], whatever the file stores.

    Lab TIFFs are decoded against their own white point (see :func:`lab_to_rgb`);
    everything else is a plain rescale from the stored sample range. That
    mapping is fixed per dtype and never per image, so a set of frames stays
    mutually comparable -- which is what texturing from a modality series
    depends on, and what any autolevel here would quietly destroy.

    Channel count is preserved: a greyscale capture stays greyscale, because
    only some callers want three channels. Those call :func:`as_truecolor`.
    """
    samples = _read_samples(path)
    encoding = lab_encoding(path)
    if encoding is not None:
        # A signed container (SampleFormat = 2, for a*/b*) is not a depth a
        # caller can write back out, and the decoded image is not signed.
        return SRGBImage(lab_to_rgb(samples, encoding),
                         _UNSIGNED.get(samples.dtype, samples.dtype), True)
    return SRGBImage(img_as_float(samples), samples.dtype, False)


def _read_samples(path: Path) -> np.ndarray:
    """A file's stored samples, at the depth it stored them.

    imageio reads PNG through Pillow, which cannot represent 16-bit multichannel
    and quietly hands back 8-bit -- so the bit depth a caller keys its
    requantization on would be wrong. OpenCV keeps it, and a PNG has no
    orientation or colorspace of its own to disagree about.
    """
    if path.suffix.lower() == '.png':
        samples = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if samples is None:
            raise OSError(f'Could not read image: {path}')
        if samples.ndim == 3:  # BGR(A)
            samples = samples[..., [2, 1, 0] + list(range(3, samples.shape[-1]))]
        return samples
    return iio.imread(path)


def drop_alpha(pixels: np.ndarray) -> np.ndarray:
    """``pixels`` without its alpha channel, which no writer here accepts.

    A greyscale pair comes back 2D rather than as one trailing channel, which
    is the shape a plain greyscale capture already has.
    """
    if pixels.ndim == 3 and pixels.shape[-1] in (2, 4):
        return pixels[..., 0] if pixels.shape[-1] == 2 else pixels[..., :3]
    return pixels


def as_truecolor(pixels: np.ndarray) -> np.ndarray:
    """Three channels, which is what OpenMVG and OpenMVS want. Alpha is dropped."""
    pixels = drop_alpha(pixels)
    if pixels.ndim == 2:
        return np.dstack([pixels] * 3)
    if pixels.shape[-1] == 1:
        return np.dstack([pixels[..., 0]] * 3)
    return pixels[..., :3]


def to_uint8(pixels: np.ndarray) -> np.ndarray:
    """Float sRGB in [0, 1] to 8-bit, the range every binary downstream reads.

    Clipped, since an enhancement pipeline may overshoot. In place past the
    first multiply: these arrays are whole captures.
    """
    out = pixels * 255.
    np.round(out, out=out)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def to_uint8_shifted(pixels: np.ndarray, bit_shift: int) -> np.ndarray:
    """A 16-bit capture to 8-bit as ``v >> bit_shift``, from ``read_srgb``'s floats.

    The stored samples are recovered exactly -- :func:`read_srgb` divides a
    16-bit file by 65535 and does nothing else to it -- so the shift stays
    integer arithmetic and the scale stays fixed across a set of frames, which
    is what relative radiometry in a merged texture depends on. Clipped, not
    wrapped: a shift under 8 leaves values above 255, and a plain uint8 cast
    turned those bright pixels dark.
    """
    samples = np.round(pixels * 65535.).astype(np.uint16)
    return np.clip(samples >> bit_shift, 0, 255).astype(np.uint8)


def prepare_8bit_image(src: Path, out_dir: Path) -> Path:
    """Write an 8-bit sRGB copy of a single image. Returns the output path.

    Unlike ``pgs-retexture``'s ``convert_modality_images`` (which keeps a *set*
    of frames mutually consistent for atlas texturing), this handles ONE
    standalone image that becomes its own texture, or the query image
    ``pgs-calibrate`` extracts features from.

    This was ImageMagick, and differs from it only on Lab files; see the module
    docstring.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'{src.stem}.jpg'
    image = read_srgb(src)
    # Narrowed before it is tripled: a capture is large, and the two commute.
    iio.imwrite(out, as_truecolor(to_uint8(image.pixels)), quality=100)
    logger.info(f'Prepared 8-bit image: {out}')
    return out


def _white_point(page) -> Tuple[float, float]:
    """A TIFF page's WhitePoint chromaticity, defaulting to D50.

    The tag is two RATIONALs, which tifffile spells either as four flat ints or
    as two pairs -- both ravel to four numerators and denominators. Anything
    else is not worth guessing at, so it falls back to the spec default rather
    than raising in the middle of a scan.
    """
    try:
        value = page.tags.valueof(318)  # WhitePoint
        flat = [float(n) for n in np.ravel(np.asarray(value, dtype=float))]
        if len(flat) == 4:
            return flat[0] / flat[1], flat[2] / flat[3]
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        pass
    return D50


def lab_encoding(path: Path) -> Optional[LabEncoding]:
    """How ``path`` stores L*a*b*, or ``None`` if it is not a Lab file.

    Colorspace is a property of the file, not of the scan holding it: a capture
    set can mix Lab and RGB, so every file is asked. TIFF is the only format
    here that records the answer, and readers hand back its samples undecoded
    either way, which is what silently turns a Lab image into a
    channel-for-channel garbage RGB one. Anything that is not a Lab TIFF --
    another format, an RGB TIFF, a file that will not open -- is ``None``, i.e.
    "treat as RGB"; an unreadable file fails later, in the read, where the error
    belongs.
    """
    if path.suffix.lower() not in ('.tif', '.tiff'):
        return None
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            photometric = int(page.photometric)
            if photometric not in LAB_PHOTOMETRICS:
                return None
            return LabEncoding(photometric, _white_point(page))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def lab_to_rgb(samples: np.ndarray, encoding: LabEncoding) -> np.ndarray:
    """Decode raw CIE L*a*b* TIFF samples to float sRGB in [0, 1].

    ``samples`` is the array a reader returns for a Lab file: stored channels,
    undecoded. TIFF 6.0 s23 fixes their encoding per bit depth -- L* spans the
    unsigned range as 0-100, while a*/b* are signed and, at 16 bits, scaled by
    256. ICCLab (9) stores that same pair unsigned, biased by half its range,
    and puts L* = 100 at 0xff00 rather than at full scale.
    ITULab (10) has no published encoding, so it is refused, not guessed.

    L*a*b* is only meaningful against a white, so the file's own WhitePoint tag
    is honored and the result Bradford-adapted to D65, which is the white sRGB
    is defined against. Note that ImageMagick -- and so
    :func:`prepare_8bit_image` -- ignores that tag and always reads D65, so the
    two agree on a D65 file and diverge on any other.
    """
    if encoding.photometric == ITULAB:
        raise ValueError('ITULab has no defined sample encoding')
    if samples.ndim != 3 or samples.shape[-1] != 3:
        raise ValueError(f'Expected 3-channel L*a*b*, got shape {samples.shape}')

    cielab = encoding.photometric == CIELAB
    if samples.dtype in _UNSIGNED:
        # A file may declare SampleFormat = 2 for its signed a*/b* channels,
        # which makes the reader hand back the whole array signed. Same bits
        # either way: L* is unsigned over the range, and the casts below recover
        # the chroma's sign regardless of which way it arrived.
        samples = samples.view(_UNSIGNED[samples.dtype])
    if np.issubdtype(samples.dtype, np.floating):
        lab = samples.astype(np.float64)  # already in L*a*b* units
    elif samples.dtype == np.uint8:
        light = samples[..., 0] * (100. / 255.)
        chroma = (samples[..., 1:].astype(np.int8) if cielab
                  else samples[..., 1:].astype(np.int16) - 128)
        lab = np.dstack([light, chroma.astype(np.float64)])
    elif samples.dtype == np.uint16:
        l_max = 65535. if cielab else ICCLAB_L_MAX_16
        light = samples[..., 0] * (100. / l_max)
        chroma = (samples[..., 1:].astype(np.int16) if cielab
                  else samples[..., 1:].astype(np.int32) - 32768)
        lab = np.dstack([light, chroma / 256.])
    else:
        raise ValueError(f'Unsupported L*a*b* sample type: {samples.dtype}')

    white = _xyz_of(encoding.white_point)
    xyz = _lab_to_xyz(lab, white) @ _bradford_to_d65(white).T
    return np.clip(xyz2rgb(xyz), 0., 1.)


def _xyz_of(chromaticity: Tuple[float, float]) -> np.ndarray:
    """A white point's XYZ from its (x, y), normalized to Y = 1."""
    x, y = chromaticity
    return np.array([x / y, 1., (1. - x - y) / y])


def _bradford_to_d65(white: np.ndarray) -> np.ndarray:
    """The chromatic adaptation matrix from ``white`` to D65.

    A no-op for a D65 file, so honoring the tag costs nothing in the common case.
    """
    source = _BRADFORD @ white
    target = _BRADFORD @ D65_XYZ
    return np.linalg.solve(_BRADFORD, np.diag(target / source) @ _BRADFORD)


def _lab_to_xyz(lab: np.ndarray, white: np.ndarray) -> np.ndarray:
    """CIE L*a*b* to XYZ against ``white``.

    Spelled out rather than taken from scikit-image, whose ``lab2xyz`` only
    accepts illuminants by name and cannot be handed a file's measured white.
    """
    lightness = (lab[..., 0] + 16.) / 116.
    f = np.stack([lightness + lab[..., 1] / 500., lightness,
                  lightness - lab[..., 2] / 200.], axis=-1)
    # The linear arm applies below L* ~ 8, so it is evaluated on those samples
    # only; np.where would run it over a whole capture to discard nearly all.
    delta = 6. / 29.
    xyz = f ** 3
    low = f <= delta
    xyz[low] = 3 * delta ** 2 * (f[low] - 4. / 29.)
    xyz *= white
    return xyz
