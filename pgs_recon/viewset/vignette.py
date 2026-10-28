"""Mask a hand-held lens's dark vignette, so no feature is taken from it.

The vignette is the lens's, not the scene's, so one mask made from any view of
the set serves every view. It is the dark region connected to each image
corner, grown by a margin: the falloff at its edge is too dim to match
reliably but bright enough to pass a plain threshold.

Needs OpenCV, unlike the rest of the planner; imported only when asked for.
"""
from pathlib import Path

import cv2
import numpy as np

#: 8-bit grey level below which a smoothed pixel counts as vignette.
DARK = 60

#: Side of the square the vignette is grown by, in pixels.
GROW = 61


def vignette_mask(image: Path) -> np.ndarray:
    """``image``'s mask: 0 in the vignette, 255 where features may be taken."""
    img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'cannot read {image}')
    h, w = img.shape
    dark = (cv2.GaussianBlur(img, (0, 0), 3) < DARK).astype(np.uint8)
    _, labels = cv2.connectedComponents(dark)
    corners = {labels[0, 0], labels[0, w - 1], labels[h - 1, 0],
               labels[h - 1, w - 1]} - {0}
    vignette = np.isin(labels, list(corners)).astype(np.uint8)
    vignette = cv2.dilate(vignette, np.ones((GROW, GROW), np.uint8))
    return np.where(vignette > 0, 0, 255).astype(np.uint8)


def write_vignette_mask(image: Path, output: Path) -> float:
    """Write ``image``'s mask to ``output``; the fraction of it masked."""
    mask = vignette_mask(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), mask):
        raise OSError(f'cannot write {output}')
    return float((mask == 0).mean())
