"""What a plate supplies, and which path through ADR 0013 that makes available.

Decision 2 splits a hand-held view's outside needs in two: a **reference**
texture to match against in its own band, and a **prior** to start from. Each
has a preferred source and a fallback, and the fallbacks cost accuracy, so the
choice is made here, once, explicitly, and recorded -- never discovered halfway
through a run by whichever input happened to be missing.

Stdlib-only.
"""
import io
import struct
from enum import Enum
from pathlib import Path
from typing import BinaryIO, List, NamedTuple, Optional, Sequence

#: Image extensions a hand-held set is collected from. Matches what
#: ``openMVG_main_SfMInit_ImageListing`` will import.
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


class Reference(Enum):
    """Where the IR reference mesh's texture comes from."""
    #: Handed in already built. Trusted as an IR940 capture retexture.
    PROVIDED = 'provided'
    #: Built here by capture retexture from the rig's IR940 capture. Preferred.
    RIG_IR940 = 'rig-ir940'
    #: Built here from the overhead 940 band. Inherits the overhead
    #: registration: ~0.15 mm floor, one bias shared by every view, one
    #: viewpoint's coverage (Decision 2, *Fallbacks*).
    OVERHEAD_940 = 'overhead-940'


class Prior(Enum):
    """Where each view's starting pose comes from."""
    #: Matched against the overhead 940 image, lifted through its position map.
    OVERHEAD = 'overhead'
    #: As ``OVERHEAD``, but the overhead is rendered here from the IR reference
    #: mesh by a virtual camera straight above it (:mod:`.overhead`), for a
    #: plate with no registered overhead capture. The prior is as good; what
    #: is lost is the overhead QA comparison, which becomes circular.
    VIRTUAL_OVERHEAD = 'virtual-overhead'
    #: Supplied per group by hand. Not implemented; named so a plate without
    #: an overhead view is refused with a reason rather than a traceback.
    MANUAL_SEED = 'manual-seed'


class Overhead(NamedTuple):
    """The registered overhead spectral view (ADR 0011's output)."""
    calibration: Path  # one-view calibration .json
    image: Path        # its 940 nm band
    camera: Optional[Path] = None  # the flat camera file, if kept


class RigCapture(NamedTuple):
    """The rig scan's IR940 capture, for capture retexture (ADR 0001)."""
    scan_dir: Path
    capture: int


class PlateInputs(NamedTuple):
    """Everything one view-set run reads. Nothing here is written to."""
    recon_dir: Path
    mesh: Path
    images: Sequence[Path]
    sfm_transform: Optional[Path] = None
    ir_reference_mesh: Optional[Path] = None
    rig_ir_capture: Optional[RigCapture] = None
    overhead: Optional[Overhead] = None
    #: Render the overhead from the IR reference mesh instead of reading one.
    #: Only consulted when ``overhead`` is absent.
    virtual_overhead: bool = False


class Strategy(NamedTuple):
    reference: Reference
    prior: Prior
    #: One line per choice, for the log and the manifest.
    reasons: tuple


class NoPath(Exception):
    """The plate supplies no usable combination. The message says what is
    missing and what would supply it."""


def choose(plate: PlateInputs) -> Strategy:
    """Pick the reference and prior sources ADR 0013 Decision 2 prefers.

    Raises :class:`NoPath` rather than falling back to anything the ADR does not
    name. The manual seed is named so its absence is a clear refusal; it is not
    yet a path a run can take.
    """
    reasons: List[str] = []

    if plate.ir_reference_mesh is not None:
        reference = Reference.PROVIDED
        reasons.append(f'reference: provided mesh {plate.ir_reference_mesh}')
    elif plate.rig_ir_capture is not None:
        reference = Reference.RIG_IR940
        reasons.append(f'reference: capture retexture of rig capture '
                       f'{plate.rig_ir_capture.capture} in '
                       f'{plate.rig_ir_capture.scan_dir}')
    elif plate.overhead is not None:
        reference = Reference.OVERHEAD_940
        reasons.append('reference: FALLBACK to the overhead 940 band -- no rig '
                       'IR940 capture given. Every view inherits the overhead '
                       'registration (ADR 0013 Decision 2, Fallbacks)')
    else:
        raise NoPath('No IR reference: pass a prebuilt IR reference mesh, the '
                     'rig scan directory and its IR940 capture index, or a '
                     'registered overhead view. A virtual overhead is rendered '
                     'from the reference mesh, so it cannot supply one.')

    if plate.overhead is not None:
        prior = Prior.OVERHEAD
        reasons.append(f'prior: overhead view {plate.overhead.calibration}')
    elif plate.virtual_overhead:
        prior = Prior.VIRTUAL_OVERHEAD
        reasons.append('prior: virtual overhead rendered from the IR reference '
                       'mesh. The overhead QA comparison would only compare '
                       'the reference with itself, so it is not run')
    else:
        raise NoPath('No prior source: a registered overhead view is required. '
                     'Manual seeding (ADR 0013 Decision 2) is not implemented.')

    return Strategy(reference, prior, tuple(reasons))


def collect_images(directory: Path) -> List[Path]:
    """The hand-held images in ``directory``, sorted, non-recursive.

    Hidden files are skipped: macOS writes an AppleDouble ``._name.tif`` beside
    every file it copies to a non-Apple volume, with an image's suffix and no
    image in it. The name is checked before ``is_file()``: through a Docker
    Desktop mount even a ``stat`` of one of those fails with ``EPERM``.

    Refuses a set where two images share a stem: per-view artifacts are keyed by
    stem (:mod:`.layout`), so the second would overwrite the first's results.

    Refuses a set where any image carries an orientation tag other than 1 --
    see :func:`check_orientation`.
    """
    images = sorted(p for p in Path(directory).iterdir()
                    if not p.name.startswith('.') and p.is_file()
                    and p.suffix.lower() in IMAGE_SUFFIXES)
    seen = {}
    for p in images:
        if p.stem in seen:
            raise ValueError(f'{p.name} and {seen[p.stem].name} share the stem '
                             f'{p.stem!r}; per-view outputs are keyed by stem')
        seen[p.stem] = p
    check_orientation(images)
    return images


def check_orientation(images: Sequence[Path]) -> None:
    """Refuse images whose orientation tag asks a reader to rotate or flip.

    The readers in a view-set run disagree about that tag. OpenCV applies it
    -- for a palette TIFF even under ``IMREAD_UNCHANGED``, because libtiff's
    RGBA path does the rotating -- so ``pgs-localize-joint`` solves the pose,
    and its QA checks it, in the *displayed* frame. ``pgs-retexture`` reads
    with imageio and copies the file as the texture, i.e. the *stored* frame.
    A view with ``Orientation = 3`` therefore passes QA and is textured upside
    down and displaced. Nothing downstream can see the mismatch, so it is
    refused here, before anything reads a pixel.

    The fix is to set the tag to 1 and leave the pixels alone: the stored
    pixels are the sensor's frame, which the shared intrinsic is solved in,
    and the solver does not care about in-plane rotation.
    """
    bad = []
    for p in images:
        value = image_orientation(p)
        if value not in (None, 1):
            bad.append(f'{p.name} (Orientation = {value})')
    if bad:
        raise ValueError(
            'These images carry an orientation tag, which OpenCV applies and '
            'the texturing step does not, so they would be registered in one '
            'frame and textured in another: ' + ', '.join(bad) + '. Set the '
            'tag to 1 without changing the pixels, e.g. '
            '`exiftool -n -IFD0:Orientation=1 -overwrite_original <image>`, '
            'and run again.')


#: The TIFF/EXIF Orientation tag.
_ORIENTATION = 274


def image_orientation(path: Path) -> Optional[int]:
    """The orientation tag of ``path``, or ``None`` if it has none.

    Reads the TIFF IFD0 of a TIFF, or the EXIF block of a JPEG or PNG -- the
    places OpenCV takes orientation from. A file that is not one of those, or
    is too damaged to parse, is ``None``: it fails later, in the read, where
    that error belongs.
    """
    try:
        with open(path, 'rb') as f:
            head = f.read(8)
            f.seek(0)
            if head[:4] in (b'II*\x00', b'MM\x00*'):
                return _tiff_orientation(f)
            if head[:2] == b'\xff\xd8':
                exif = _jpeg_exif(f)
            elif head == b'\x89PNG\r\n\x1a\n':
                exif = _png_exif(f)
            else:
                return None
            return _tiff_orientation(io.BytesIO(exif)) if exif else None
    except (OSError, struct.error, ValueError):
        return None


def _tiff_orientation(f: BinaryIO) -> Optional[int]:
    """The Orientation entry of IFD0 in the TIFF stream ``f`` (offset 0)."""
    order = {b'II': '<', b'MM': '>'}[f.read(2)]
    magic, ifd = struct.unpack(order + 'HI', f.read(6))
    if magic != 42:
        return None
    f.seek(ifd)
    (count,) = struct.unpack(order + 'H', f.read(2))
    for _ in range(count):
        tag, kind, n, value = struct.unpack(order + 'HHI4s', f.read(12))
        if tag == _ORIENTATION and n == 1 and kind == 3:  # one SHORT
            return struct.unpack(order + 'H', value[:2])[0]
    return None


def _jpeg_exif(f: BinaryIO) -> Optional[bytes]:
    """The TIFF stream inside a JPEG's ``Exif`` APP1 segment, if any."""
    f.read(2)  # SOI
    while True:
        marker = f.read(2)
        if len(marker) < 2 or marker[0] != 0xFF or marker[1] in (0xD9, 0xDA):
            return None  # EOI or start of scan: no EXIF precedes the data
        (length,) = struct.unpack('>H', f.read(2))
        body = f.read(length - 2)
        if marker[1] == 0xE1 and body.startswith(b'Exif\x00\x00'):
            return body[6:]


def _png_exif(f: BinaryIO) -> Optional[bytes]:
    """The TIFF stream in a PNG's ``eXIf`` chunk, if it precedes the data."""
    f.read(8)  # signature
    while True:
        header = f.read(8)
        if len(header) < 8:
            return None
        length, kind = struct.unpack('>I4s', header)
        if kind in (b'IDAT', b'IEND'):
            return None
        body = f.read(length)
        f.read(4)  # CRC
        if kind == b'eXIf':
            return body
