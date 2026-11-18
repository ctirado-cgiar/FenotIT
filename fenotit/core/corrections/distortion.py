"""
Calibración óptica con tablero de ajedrez y corrección de distorsión de lente.
Tablero configurable (esquinas interiores: columnas × filas).

detect()  busca el tablero en una foto: primero en una copia reducida (rápido aun con
          fotos de 20+ MP) y afina las esquinas en la foto completa.
solve()   calibra con las fotos donde se encontró y da el error de cada una.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Registro de mensajes: anota las fotos del tablero que no se pudieron leer o calibrar.
from fenotit import log

_log = log.get("corrections.distortion")

MAX_PHOTOS = 40          # más fotos no mejoran la calibración y la hacen lenta
DETECT_SIDE = 1600       # lado mayor de la copia donde se busca el tablero
QUICK_SIDE = 400         # revisión rápida de muchas fotos (¿hay tablero?)


@dataclass
class Detection:
    path: str
    found: bool
    corners: np.ndarray | None = None      # en px de la foto completa
    shape: tuple[int, int] | None = None   # (alto, ancho)
    preview: np.ndarray | None = None      # miniatura BGR con las esquinas dibujadas
    error_px: float | None = None          # error de reproyección de esta foto (tras calibrar)


@dataclass
class CalibrationResult:
    success: bool
    mtx: np.ndarray | None = None
    dist: np.ndarray | None = None
    rms_error: float = 0.0
    n_images_used: int = 0
    n_images_total: int = 0
    message: str = ""
    detections: list[Detection] = field(default_factory=list)
    size: tuple[int, int] | None = None     # (ancho, alto) de las fotos usadas


def spread(paths: list[str], count: int = MAX_PHOTOS) -> list[str]:
    """Hasta n fotos repartidas a lo largo de la lista (no solo las primeras)."""
    if len(paths) <= count:
        return list(paths)
    idx = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[i] for i in dict.fromkeys(idx)]


def _read(path: str):
    data = np.frombuffer(Path(path).read_bytes(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def has_board(path: str, cols: int, rows: int) -> bool:
    """¿Hay tablero? Rápido, para revisar cientos de fotos: el JPEG se decodifica ya
    reducido (1/2, 1/4 u 1/8) y solo se busca, sin afinar esquinas."""
    try:
        from PIL import Image
        with Image.open(path) as picture:
            side = max(picture.size)
        data = np.frombuffer(Path(path).read_bytes(), np.uint8)
    except Exception:
        return False
    flag = (cv2.IMREAD_REDUCED_GRAYSCALE_8 if side > 8000 else cv2.IMREAD_REDUCED_GRAYSCALE_4 if side > 3200
            else cv2.IMREAD_REDUCED_GRAYSCALE_2 if side > 1600 else cv2.IMREAD_GRAYSCALE)
    gray = cv2.imdecode(data, flag)
    if gray is None:
        return False
    shrink = QUICK_SIDE / max(gray.shape)
    if shrink < 1:                                   # en fotos sin tablero la búsqueda tarda más con el tamaño
        gray = cv2.resize(gray, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, _ = cv2.findChessboardCorners(gray, (cols, rows), flags)
    return bool(found)


def detect(path: str, cols: int, rows: int, preview_side: int = 360) -> Detection:
    try:
        img = _read(path)
    except OSError:
        img = None
    if img is None:
        return Detection(path, False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    shrink = min(1.0, DETECT_SIDE / max(height, width))
    small = cv2.resize(gray, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA) if shrink < 1 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(small, (cols, rows), flags)
    if found:
        corners = (corners / shrink).astype(np.float32)
        win = max(5, int(round(5 / shrink)))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), crit)
    preview_scale = preview_side / max(height, width)
    prev = cv2.resize(img, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
    if found:
        cv2.drawChessboardCorners(prev, (cols, rows), corners * preview_scale, True)
    return Detection(path, bool(found), corners if found else None, (height, width), prev)


def solve(detections: list[Detection], cols: int, rows: int) -> CalibrationResult:
    used = [detection for detection in detections if detection.found]
    total = len(detections)
    if len(used) < 3:
        return CalibrationResult(False, n_images_used=len(used), n_images_total=total, detections=detections,
                                 message=f"Board found in {len(used)} of {total} photos (need 3).")
    sizes = {detection.shape for detection in used}
    if len(sizes) > 1:
        size = max(sizes, key=lambda shape: sum(detection.shape == shape for detection in used))
        used = [detection for detection in used if detection.shape == size]
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    height, width = used[0].shape
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp] * len(used), [detection.corners for detection in used],
                                                       (width, height), None, None)
    for detection, rvec, tvec in zip(used, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
        detection.error_px = float(np.sqrt(np.mean(np.sum((proj - detection.corners) ** 2, axis=2))))
    return CalibrationResult(True, mtx, dist, round(float(rms), 4), len(used), total,
                             f"RMS {rms:.4f} px", detections, (width, height))


def calibrate(images: list[str], inner_cols: int = 7, inner_rows: int = 6,
              output_params_dir: str | None = None, progress_cb=None) -> CalibrationResult:
    """Sin interfaz: detectar en cada foto y calibrar. progress_cb(i, n, nombre)."""
    dets = []
    for i, path in enumerate(images, 1):
        if progress_cb:
            progress_cb(i, len(images), Path(path).name)
        dets.append(detect(path, inner_cols, inner_rows))
    res = solve(dets, inner_cols, inner_rows)
    if res.success and output_params_dir:
        Path(output_params_dir).mkdir(parents=True, exist_ok=True)
        np.savez(os.path.join(output_params_dir, "calibracion_params.npz"), mtx=res.mtx, dist=res.dist)
    return res


def calibrate_from_folder(folder: str, inner_cols: int = 7, inner_rows: int = 6,
                          output_params_dir: str | None = None, progress_cb=None) -> CalibrationResult:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(str(path) for path in Path(folder).iterdir() if path.suffix.lower() in exts)
    return calibrate(files, inner_cols, inner_rows, output_params_dir, progress_cb)
