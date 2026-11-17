"""Detección de marcadores ArUco (DICT_4X4_50, IDs 0-3). La corrección está en pipeline.rectify."""

import cv2
import numpy as np

from fenotit import log

_log = log.get("corrections.aruco")


# ── Detección de ArUcos ───────────────────────────────────────────────────────

def _get_aruco_dict():
    """Retorna el diccionario ArUco compatible con la versión de OpenCV."""
    try:
        # OpenCV 4.7+
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:
        return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


def _get_aruco_params():
    try:
        return cv2.aruco.DetectorParameters()
    except AttributeError:
        return cv2.aruco.DetectorParameters_create()


def detect_aruco_corners(image: np.ndarray) -> dict[int, np.ndarray]:
    """
    Detecta marcadores ArUco en la imagen.
    Retorna dict {id: center_point} para IDs 0-3.
    """
    aruco_dict   = _get_aruco_dict()
    aruco_params = _get_aruco_params()

    try:
        # OpenCV 4.7+
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(image)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image, aruco_dict, parameters=aruco_params)

    result = {}
    if ids is None:
        return result

    for i, marker_id in enumerate(ids.flatten()):
        if 0 <= marker_id <= 3:
            # Centro del marcador
            marker_corners = corners[i][0]
            result[int(marker_id)] = marker_corners.mean(axis=0)  # [x, y]

    return result
