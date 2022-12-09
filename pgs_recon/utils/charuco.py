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


# Create a 3x3 Charuco board containing 4x Aruco markers.
# Board is 30 units x 30 units. Markers are 7 units x 7 units.
def generate_board(dictionary=ar.DICT_ARUCO_ORIGINAL, offset=0):
    aruco_dict = ar.getPredefinedDictionary(dictionary)
    aruco_dict.bytesList = aruco_dict.bytesList[offset:offset + 4]
    board = ar.CharucoBoard_create(squaresX=3, squaresY=3, squareLength=10,
                                   markerLength=7, dictionary=aruco_dict)
    return board


# Detect a Charuco board. Returned results are sorted by marker and board IDs.
def detect_board(img, board) -> DetectedBoard:
    # Account for markers being small relative to max dimension for large area
    # scans
    params = ar.DetectorParameters_create()
    if max(img.shape) > 14000:
        params.minMarkerPerimeterRate = 0.015

    # Detect Aruco markers
    marker_corners, marker_ids, _ = ar.detectMarkers(img, board.dictionary, parameters=params)
    if marker_ids is not None:
        marker_cnt = len(marker_ids)
    else:
        marker_cnt = 0

    # Interpolate Charuco corners
    board_corners = None
    board_ids = None
    board_cnt = 0
    if marker_ids is not None and len(marker_ids) > 0:
        _, board_corners, board_ids = \
            ar.interpolateCornersCharuco(marker_corners,
                                         marker_ids,
                                         img,
                                         board)
        if board_ids is not None:
            board_cnt = len(board_ids)

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
