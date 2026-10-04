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

from fenotit import log

_log = log.get("corrections.distortion")

MAX_PHOTOS = 40          # más fotos no mejoran la calibración y la hacen lenta
DETECT_SIDE = 1600       # lado mayor de la copia donde se busca el tablero


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


def spread(paths: list[str], n: int = MAX_PHOTOS) -> list[str]:
    """Hasta n fotos repartidas a lo largo de la lista (no solo las primeras)."""
    if len(paths) <= n:
        return list(paths)
    idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    return [paths[i] for i in dict.fromkeys(idx)]


def _read(path: str):
    data = np.frombuffer(Path(path).read_bytes(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detect(path: str, cols: int, rows: int, preview_side: int = 360) -> Detection:
    try:
        img = _read(path)
    except OSError:
        img = None
    if img is None:
        return Detection(path, False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = min(1.0, DETECT_SIDE / max(h, w))
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(small, (cols, rows), flags)
    if found:
        corners = (corners / k).astype(np.float32)
        win = max(5, int(round(5 / k)))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), crit)
    kp = preview_side / max(h, w)
    prev = cv2.resize(img, None, fx=kp, fy=kp, interpolation=cv2.INTER_AREA)
    if found:
        cv2.drawChessboardCorners(prev, (cols, rows), corners * kp, True)
    return Detection(path, bool(found), corners if found else None, (h, w), prev)


def solve(detections: list[Detection], cols: int, rows: int) -> CalibrationResult:
    used = [d for d in detections if d.found]
    total = len(detections)
    if len(used) < 3:
        return CalibrationResult(False, n_images_used=len(used), n_images_total=total, detections=detections,
                                 message=f"Board found in {len(used)} of {total} photos (need 3).")
    sizes = {d.shape for d in used}
    if len(sizes) > 1:
        size = max(sizes, key=lambda s: sum(d.shape == s for d in used))
        used = [d for d in used if d.shape == size]
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    h, w = used[0].shape
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp] * len(used), [d.corners for d in used],
                                                       (w, h), None, None)
    for d, rv, tv in zip(used, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rv, tv, mtx, dist)
        d.error_px = float(np.sqrt(np.mean(np.sum((proj - d.corners) ** 2, axis=2))))
    return CalibrationResult(True, mtx, dist, round(float(rms), 4), len(used), total,
                             f"RMS {rms:.4f} px", detections)


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
    files = sorted(str(p) for p in Path(folder).iterdir() if p.suffix.lower() in exts)
    return calibrate(files, inner_cols, inner_rows, output_params_dir, progress_cb)
