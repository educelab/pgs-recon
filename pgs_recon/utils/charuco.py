from collections import namedtuple

import cv2.aruco as ar
import numpy as np

DetectedBoard = namedtuple('DetectedBoard',
                           ['marker_corners',
                            'marker_ids',
                            'marker_cnt',
                            'board_corners',
                            'board_ids',
                            'board_cnt'])
DetectedBoard.__doc__ = """One board's detection, in shapes that do not move.

``CharucoDetector.detectBoard`` changed what it hands back between OpenCV 4
and OpenCV 5: the board corners went from ``(N, 1, 2)`` to ``(N, 2)`` and both
ID arrays from ``(N, 1)`` to ``(N,)``. ``detect_board`` normalizes all of them
back to the ``(N, 1, ...)`` layout, so a consumer indexes one shape whatever
OpenCV is installed:

- ``marker_corners``: a tuple of ``(1, 4, 2)`` arrays, one per marker. The one
  value that did *not* change across versions.
- ``marker_ids``, ``board_ids``: ``(N, 1)``.
- ``board_corners``: ``(N, 1, 2)``, so ``board_corners[i]`` is a point rather
  than a scalar.

Normalizing toward OpenCV 4 rather than 5 is not nostalgia: it is the only
layout ``aruco.drawDetectedCornersCharuco`` accepts, under *either* version.
OpenCV 5 rejects its own detector's output there.

An undetected board is ``()`` corners, ``None`` IDs and a count of 0 --
enforced here rather than inherited, so an empty-but-not-``None`` result from
some other OpenCV build cannot reach a caller as a third spelling of "none".
"""


# Create a 3x3 Charuco board containing 4x Aruco markers.
# Board is 30 units x 30 units. Markers are 7 units x 7 units.
def generate_board(dictionary=ar.DICT_ARUCO_ORIGINAL, offset=0):
    aruco_dict = ar.getPredefinedDictionary(dictionary)
    aruco_dict.bytesList = aruco_dict.bytesList[offset:offset + 4]
    board = ar.CharucoBoard((3, 3), squareLength=10, markerLength=7, dictionary=aruco_dict)
    return board


# Detect a Charuco board. Returned results are sorted by marker and board IDs.
def detect_board(img, board) -> DetectedBoard:
    # Account for markers being small relative to max dimension for large area
    # scans
    detectorParams = ar.DetectorParameters()
    if max(img.shape) > 14000:
        detectorParams.minMarkerPerimeterRate = 0.015

    # Detect Aruco markers
    detector = ar.CharucoDetector(board, detectorParams=detectorParams)
    board_corners, board_ids, marker_corners, marker_ids = detector.detectBoard(img)

    # Normalize away the OpenCV 4 -> 5 shape change before anything indexes
    # these. See DetectedBoard for the layout every caller may rely on.
    if marker_ids is not None and len(marker_ids) > 0:
        marker_ids = np.asarray(marker_ids).reshape(-1, 1)
        marker_corners = tuple(np.asarray(c).reshape(1, -1, 2)
                               for c in marker_corners)
        marker_cnt = len(marker_ids)
    else:
        marker_corners = ()
        marker_ids = None
        marker_cnt = 0
    if board_ids is not None and len(board_ids) > 0:
        board_ids = np.asarray(board_ids).reshape(-1, 1)
        board_corners = np.asarray(board_corners).reshape(-1, 1, 2)
        board_cnt = len(board_ids)
    else:
        board_corners = ()
        board_ids = None
        board_cnt = 0

    # Sort the results
    if marker_ids is not None:
        p = np.argsort(marker_ids, axis=0)
        marker_ids = np.take_along_axis(marker_ids, p, axis=0)
        marker_corners = tuple(marker_corners[i] for i in p.flat)

    if board_ids is not None:
        p = np.argsort(board_ids, axis=0)
        board_ids = np.take_along_axis(board_ids, p, axis=0)
        board_corners = np.take_along_axis(board_corners,
                                           np.expand_dims(p, axis=-1), axis=0)

    return DetectedBoard(marker_corners,
                         marker_ids,
                         marker_cnt,
                         board_corners,
                         board_ids,
                         board_cnt)
