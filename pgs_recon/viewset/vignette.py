"""Mask a hand-held lens's dark vignette, so no feature is taken from it.

The vignette is the lens's, not the scene's: it sits in the same pixels in
every shot taken with one lens setup, while the object moves between shots. So
the mask is a vote. Each image's own vignette is the dark region connected to
its corners; a pixel is vignette for the set when nearly every image says so.
Dark papyrus that happens to touch one image's corner is outvoted by the
images in which that pixel was paper -- one image deciding alone is how a
first image with dark papyrus at its edge once masked a quarter of every view.

One mask does not fit a set shot with two lens setups (a different aperture or
zoom moves the vignette), so the images are grouped first, by image size and
then by how far the vignette reaches in along each corner's diagonal, and each
group votes on its own mask. The diagonal is measured per corner and compared
by the median of the four differences: a dark object touching one corner
lengthens that corner's reach and nothing else, so it cannot move an image into
another group.

Every image's own vignette is scored against its group's mask
(intersection-over-union), so an image that disagrees -- the dark-papyrus case,
or an overexposed shot whose corners cleared the threshold -- is reported
rather than silently decided for.

Needs OpenCV, unlike the rest of the planner; imported only when asked for.
"""
import math
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Sequence, Tuple

import cv2
import numpy as np

#: 8-bit grey level below which a smoothed pixel counts as vignette.
DARK = 60

#: Side of the square the vignette is grown by, in pixels.
GROW = 61

#: Fraction of a group's images that must call a pixel vignette for the
#: group's mask to. Short of all of them, so that one or two overexposed shots
#: cannot shrink the mask; well past half, so that an object at the same frame
#: edge in most shots is still outvoted by the rest.
QUORUM = 0.85

#: Two images belong to different lens setups when the median of their four
#: corner reaches differs by more than this fraction of the group's reach...
SPLIT = 0.15

#: ...or by more than this many pixels, whichever is larger: a lens with
#: almost no vignette has reaches of a few pixels, where a fraction is noise.
SPLIT_MIN_PX = 10.0

#: An image whose own vignette overlaps its group's by less than this
#: (intersection-over-union) is reported as disagreeing with it.
AGREE = 0.7


def read_grey(image: Path) -> np.ndarray:
    img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'cannot read {image}')
    return img


def raw_vignette(img: np.ndarray) -> np.ndarray:
    """The dark region connected to ``img``'s corners, not grown: True where
    vignette."""
    h, w = img.shape
    dark = (cv2.GaussianBlur(img, (0, 0), 3) < DARK).astype(np.uint8)
    _, labels = cv2.connectedComponents(dark)
    corners = {labels[0, 0], labels[0, w - 1], labels[h - 1, 0],
               labels[h - 1, w - 1]} - {0}
    return np.isin(labels, list(corners))


def corner_reaches(raw: np.ndarray) -> Tuple[float, float, float, float]:
    """How far ``raw`` reaches in from each corner along its diagonal toward the
    centre, in pixels: top-left, top-right, bottom-left, bottom-right. Zero for
    a corner that is not vignette at all."""
    h, w = raw.shape
    steps = max(h, w) // 2
    t = np.arange(steps) / steps
    length = math.hypot(h / 2, w / 2)
    reaches = []
    for y0, x0, sy, sx in ((0, 0, 1, 1), (0, w - 1, 1, -1),
                           (h - 1, 0, -1, 1), (h - 1, w - 1, -1, -1)):
        ys = np.round(y0 + sy * t * (h / 2)).astype(int)
        xs = np.round(x0 + sx * t * (w / 2)).astype(int)
        along = raw[ys, xs]
        first_clear = steps if along.all() else int(np.argmin(along))
        reaches.append(first_clear / steps * length)
    return tuple(reaches)


def _reach_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.median(np.abs(np.subtract(a, b))))


def group_by_reach(reaches: Sequence[Sequence[float]],
                   split: float = SPLIT,
                   split_min_px: float = SPLIT_MIN_PX) -> List[List[int]]:
    """Indices of ``reaches`` grouped into lens setups.

    Images are taken in order of their median reach, and each joins the first
    group whose median reach vector is within tolerance, else starts one. The
    groups come back ordered by their first index, members ascending.
    """
    order = sorted(range(len(reaches)),
                   key=lambda i: (float(np.median(reaches[i])), i))
    groups: List[List[int]] = []
    for i in order:
        for g in groups:
            centre = np.median([reaches[j] for j in g], axis=0)
            tolerance = max(split * float(np.median(centre)), split_min_px)
            if _reach_distance(reaches[i], centre) <= tolerance:
                g.append(i)
                break
        else:
            groups.append([i])
    return sorted((sorted(g) for g in groups), key=lambda g: g[0])


def vote(raws: Iterable[np.ndarray], count: int,
         quorum: float = QUORUM) -> np.ndarray:
    """The pixels at least ``quorum`` of ``count`` raw vignettes agree on, kept
    only where connected to a corner (the vote can leave islands)."""
    votes = None
    for raw in raws:
        votes = raw.astype(np.uint16) if votes is None else votes + raw
    if votes is None:
        raise ValueError('no images to vote')
    agreed = votes >= max(1, math.ceil(quorum * count))
    h, w = agreed.shape
    _, labels = cv2.connectedComponents(agreed.astype(np.uint8))
    corners = {labels[0, 0], labels[0, w - 1], labels[h - 1, 0],
               labels[h - 1, w - 1]} - {0}
    return np.isin(labels, list(corners))


def agreement(raw: np.ndarray, consensus: np.ndarray) -> float:
    """Intersection-over-union of one image's vignette with its group's; 1.0
    when both are empty."""
    union = np.count_nonzero(raw | consensus)
    if union == 0:
        return 1.0
    return np.count_nonzero(raw & consensus) / union


def grow(raw: np.ndarray) -> np.ndarray:
    """The mask: 0 in the vignette grown by :data:`GROW`, 255 where features
    may be taken. The falloff at the vignette's edge is too dim to match
    reliably but bright enough to pass the threshold."""
    grown = cv2.dilate(raw.astype(np.uint8), np.ones((GROW, GROW), np.uint8))
    return np.where(grown > 0, 0, 255).astype(np.uint8)


def vignette_mask(image: Path) -> np.ndarray:
    """One image's own mask, decided by that image alone."""
    return grow(raw_vignette(read_grey(image)))


class GroupMask(NamedTuple):
    """One lens setup's images and the mask they voted for."""
    members: List[Path]
    mask: np.ndarray               # 0 = vignette, 255 = usable
    reach: List[float]             # the group's median corner reaches, px
    agreement: Dict[Path, float]   # each member's own vignette against it


def vignette_masks(images: Sequence[Path],
                   quorum: float = QUORUM) -> List[GroupMask]:
    """Every lens setup in ``images`` and its voted mask.

    Images of different sizes never share a group. Each raw vignette is held
    bit-packed (an eighth of a byte a pixel), so a set of hundreds of images
    costs tens of megabytes, not gigabytes.
    """
    if not images:
        raise ValueError('no images to make a vignette mask from')
    packed: List[np.ndarray] = []
    shapes: List[Tuple[int, int]] = []
    reaches: List[Tuple[float, ...]] = []
    for image in images:
        raw = raw_vignette(read_grey(image))
        packed.append(np.packbits(raw, axis=None))
        shapes.append(raw.shape)
        reaches.append(corner_reaches(raw))

    def unpack(i: int) -> np.ndarray:
        h, w = shapes[i]
        return np.unpackbits(packed[i], count=h * w).reshape(h, w).astype(bool)

    by_shape: Dict[Tuple[int, int], List[int]] = {}
    for i, shape in enumerate(shapes):
        by_shape.setdefault(shape, []).append(i)
    groups = []
    for indices in by_shape.values():
        for sub in group_by_reach([reaches[i] for i in indices]):
            groups.append([indices[k] for k in sub])
    groups.sort(key=lambda g: g[0])

    out = []
    for g in groups:
        consensus = vote((unpack(i) for i in g), len(g), quorum)
        out.append(GroupMask(
            members=[images[i] for i in g],
            mask=grow(consensus),
            reach=[round(float(r), 1)
                   for r in np.median([reaches[i] for i in g], axis=0)],
            agreement={images[i]: round(agreement(unpack(i), consensus), 3)
                       for i in g}))
    return out


def write_mask(mask: np.ndarray, output: Path) -> bool:
    """Write ``mask`` to ``output`` unless the file already holds exactly these
    pixels -- left alone, it keeps its fingerprint, so the views localized
    through it stay current. True when written."""
    if output.exists():
        current = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
        if current is not None and np.array_equal(current, mask):
            return False
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), mask):
        raise OSError(f'cannot write {output}')
    return True

