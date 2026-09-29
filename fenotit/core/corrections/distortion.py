"""
corrections/distortion.py
Calibración óptica con tablero de ajedrez y corrección de distorsión de lente.
Adaptado de scripts 01/02. Tablero configurable (filas x columnas esquinas).
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
from dataclasses import dataclass

from fenotit import log

_log = log.get("corrections.distortion")


@dataclass
class CalibrationResult:
    success: bool
    mtx: np.ndarray | None = None
    dist: np.ndarray | None = None
    rms_error: float = 0.0
    n_images_used: int = 0
    n_images_total: int = 0
    message: str = ""


def calibrate_from_folder(
    folder: str,
    inner_cols: int = 7,
    inner_rows: int = 6,
    output_params_dir: str | None = None,
    progress_cb=None,
) -> CalibrationResult:
    """
    Detecta esquinas en todas las imágenes del tablero y calibra la cámara.

    inner_cols, inner_rows: esquinas interiores del tablero
        (cuadros - 1 en cada dirección)
    output_params_dir: si se da, guarda calibracion_params.npz ahí
    progress_cb: callback(current, total, filename) para progreso
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((inner_rows * inner_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(folder, ext)))
        images.extend(glob.glob(os.path.join(folder, ext.upper())))
    images = sorted(set(images))

    if not images:
        return CalibrationResult(
            success=False,
            message="No se encontraron imágenes en la carpeta.")

    gray_shape = None
    n_found = 0

    for i, fname in enumerate(images):
        if progress_cb:
            progress_cb(i + 1, len(images), Path(fname).name)

        # Carga Unicode-safe
        raw  = Path(fname).read_bytes()
        arr  = np.frombuffer(raw, dtype=np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape

        ret, corners = cv2.findChessboardCorners(
            gray, (inner_cols, inner_rows), None)

        if ret:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            n_found += 1

    if n_found < 3:
        return CalibrationResult(
            success=False,
            n_images_used=n_found,
            n_images_total=len(images),
            message=f"Solo {n_found} imágenes con esquinas detectadas. "
                    f"Se necesitan mínimo 3.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape[::-1], None, None)

    if output_params_dir:
        Path(output_params_dir).mkdir(parents=True, exist_ok=True)
        np.savez(os.path.join(output_params_dir, "calibracion_params.npz"), mtx=mtx, dist=dist)

    return CalibrationResult(
        success=True,
        mtx=mtx,
        dist=dist,
        rms_error=round(float(ret), 4),
        n_images_used=n_found,
        n_images_total=len(images),
        message=f"Calibración exitosa. RMS error: {ret:.4f} px")
