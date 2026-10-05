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
