"""Detección de marcadores ArUco (DICT_4X4_50, IDs 0-3). La corrección está en pipeline.rectify."""

import cv2
import numpy as np
from pathlib import Path

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
            c = corners[i][0]
            result[int(marker_id)] = c.mean(axis=0)  # [x, y]

    return result


def preview_detection(image: np.ndarray) -> np.ndarray:
    """
    Genera imagen de preview mostrando los ArUcos detectados.
    Útil para verificar antes de aplicar la corrección.
    """
    preview = image.copy()
    corners_dict = detect_aruco_corners(image)

    colors = {
        0: (0, 200, 80),    # verde  — TL
        1: (0, 120, 255),   # azul   — TR
        2: (0, 60, 200),    # azul oscuro — BR
        3: (200, 160, 0),   # amarillo — BL
    }
    labels = {0: "ID0 TL", 1: "ID1 TR", 2: "ID2 BR", 3: "ID3 BL"}

    for mid, center in corners_dict.items():
        cx, cy = int(center[0]), int(center[1])
        color  = colors.get(mid, (128, 128, 128))
        cv2.circle(preview, (cx, cy), 10, color, -1)
        cv2.circle(preview, (cx, cy), 12, (255, 255, 255), 2)
        cv2.putText(preview, labels.get(mid, f"ID{mid}"),
                    (cx + 14, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (30, 30, 30), 2, cv2.LINE_AA)

    # Dibujar línea conectando los 4 si están todos presentes
    if len(corners_dict) == 4:
        pts = np.int32([corners_dict[i] for i in range(4)])
        cv2.polylines(preview, [pts], isClosed=True,
                      color=(100, 100, 100), thickness=1,
                      lineType=cv2.LINE_AA)

    detected = len(corners_dict)
    msg = (f"ArUcos detectados: {detected}/4"
           if detected < 4
           else "✓ 4 ArUcos detectados — listo para corregir")
    cv2.putText(preview, msg, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 150, 60) if detected == 4 else (0, 60, 200),
                2, cv2.LINE_AA)

    return preview
