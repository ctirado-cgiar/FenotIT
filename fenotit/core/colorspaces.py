"""Espacios de color disponibles para segmentar."""
import cv2
import numpy as np

CHANNELS = {
    "BGR":   ["B", "G", "R"],
    "HSV":   ["H", "S", "V"],
    "LAB":   ["L", "A", "B"],
    "YCrCb": ["Y", "Cr", "Cb"],
    "HLS":   ["H", "L", "S"],
    "XYZ":   ["X", "Y", "Z"],
    "YUV":   ["Y", "U", "V"],
    "LUV":   ["L", "U", "V"],
}

# color para dibujar cada canal (histograma): el del propio canal cuando lo tiene
# (rojo, verde, azul), gris para luminosidad, y pares opuestos para los ejes de color
_RED, _GREEN, _BLUE, _GRAY = "#D73027", "#1A9850", "#2166AC", "#636363"
_HUE, _SAT, _YEL, _MAG = "#7B3294", "#E08214", "#C9A400", "#C51B7D"
CHANNEL_COLORS = {
    "BGR":   [_BLUE, _GREEN, _RED],
    "HSV":   [_HUE, _SAT, _GRAY],
    "LAB":   [_GRAY, _MAG, _YEL],      # a*: verde–magenta, b*: azul–amarillo
    "YCrCb": [_GRAY, _RED, _BLUE],
    "HLS":   [_HUE, _GRAY, _SAT],
    "XYZ":   [_RED, _GRAY, _BLUE],     # Y = luminancia
    "YUV":   [_GRAY, _BLUE, _RED],     # U: diferencia de azul, V: diferencia de rojo
    "LUV":   [_GRAY, _MAG, _YEL],
}

CODES = {
    "HSV":   cv2.COLOR_BGR2HSV,
    "LAB":   cv2.COLOR_BGR2LAB,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
    "LUV":   cv2.COLOR_BGR2LUV,
    "HLS":   cv2.COLOR_BGR2HLS,
    "XYZ":   cv2.COLOR_BGR2XYZ,
    "YUV":   cv2.COLOR_BGR2YUV,
}


def channel(image: np.ndarray, space: str, idx: int) -> np.ndarray:
    conv = cv2.cvtColor(image, CODES[space]) if space in CODES else image
    return conv[:, :, min(int(idx), conv.shape[2] - 1)] if conv.ndim == 3 else conv


def channels(image: np.ndarray, space: str) -> list[np.ndarray]:
    """Los tres canales de la imagen en ese espacio (una sola conversión)."""
    conv = cv2.cvtColor(image, CODES[space]) if space in CODES else image
    return [conv[:, :, i] for i in range(3)] if conv.ndim == 3 else [conv]
