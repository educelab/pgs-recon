"""What a plate supplies, and which path through ADR 0013 that makes available.

Decision 2 splits a hand-held view's outside needs in two: a **reference**
texture to match against in its own band, and a **prior** to start from. Each
has a preferred source and a fallback, and the fallbacks cost accuracy, so the
choice is made here, once, explicitly, and recorded -- never discovered halfway
through a run by whichever input happened to be missing.

Stdlib-only.
"""
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

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
    return images
