"""Correcciones opcionales aplicadas en memoria: distorsión → perspectiva → color."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fenotit import log
from fenotit.core.corrections import aruco

_log = log.get("corrections")


@dataclass
class Distortion:
    enabled: bool = False
    mtx: list | None = None
    dist: list | None = None
    rms: float | None = None
    source: str = ""
    size: list | None = None          # [ancho, alto] de las fotos del tablero (None = desconocido, p. ej. .npz)
    photos: list = field(default_factory=list)   # fotos del tablero que están entre las del proyecto: no se analizan


@dataclass
class Perspective:
    enabled: bool = False
    margin_px: int = 20
    width_mm: float | None = None     # distancia real entre centros ID0-ID1
    height_mm: float | None = None    # distancia real entre centros ID0-ID3


@dataclass
class ColorCard:
    enabled: bool = False
    card: str = "colorchecker24"      # colorchecker24 | custom | white
    rows: int = 4                     # custom
    cols: int = 6
    values: list | None = None        # custom: [R, G, B] 0-255 de cada cuadrito, fila por fila (de la marca)
    reference: str = "values"         # values = colores reales de la tarjeta | photo = igualar a una foto
    target: list | None = None        # photo: colores medidos en la foto de referencia (filas, cols, 3) 0-1
    target_photo: str = ""
    mode: str = "per_image"           # per_image = buscar la tarjeta en cada foto | fixed = siempre en el mismo lugar
    place: dict | None = None         # fixed: colorcard.Card.to_dict() + "size" [ancho, alto]
    region: list | None = None        # white: [x0, y0, x1, y1] (0-1) del área blanca o gris

    @classmethod
    def from_dict(cls, d: dict) -> "ColorCard":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in (d or {}).items() if k in known})   # proyectos viejos (pos, radius, mask)

    def grid(self) -> tuple[int, int]:
        return (4, 6) if self.card == "colorchecker24" else (int(self.rows), int(self.cols))


@dataclass
class Corrections:
    distortion: Distortion = field(default_factory=Distortion)
    perspective: Perspective = field(default_factory=Perspective)
    color: ColorCard = field(default_factory=ColorCard)

    def active(self) -> list[str]:
        return [n for n in ("distortion", "perspective", "color") if getattr(self, n).enabled]

    def to_dict(self) -> dict:
        return {"distortion": asdict(self.distortion),
                "perspective": asdict(self.perspective),
                "color": asdict(self.color)}

    @classmethod
    def from_dict(cls, d: dict | None, folder: Path | None = None) -> "Corrections":
        d = d or {}
        return cls(Distortion(**d.get("distortion", {})),
                   Perspective(**d.get("perspective", {})),
                   ColorCard.from_dict(d.get("color", {})))

    def save_files(self, folder: Path) -> None:
        """Todo va en el YAML del proyecto (antes, una máscara de la tarjeta en calibration/)."""


@dataclass
class CorrectionInfo:
    applied: list[str] = field(default_factory=list)
    mm_per_px: float | None = None    # escala obtenida de ArUco con medidas reales
    warnings: list[str] = field(default_factory=list)
    aruco_missing: list[int] = field(default_factory=list)
    color: dict | None = None          # tarjeta encontrada, cuadritos descartados, error


# ── Distorsión ────────────────────────────────────────────────────────────────

def load_npz(path: str | Path) -> Distortion:
    data = np.load(path)
    return Distortion(True, data["mtx"].tolist(), data["dist"].tolist(), None, Path(path).name)


def camera_matrix(cfg: Distortion, w: int, h: int) -> np.ndarray:
    """La matriz de la cámara vale para el tamaño de las fotos del tablero. Otra resolución
    con la misma proporción (p. ej. la cámara en 50 MP y en 200 MP) se escala; otra
    proporción (recortada o girada) no se puede corregir con esta calibración."""
    mtx = np.array(cfg.mtx, dtype=np.float64)
    if not cfg.size or tuple(cfg.size) == (w, h):
        return mtx
    cw, ch = cfg.size
    if abs(w / h - cw / ch) > 0.01:
        raise ValueError(f"calibration made for {cw}x{ch} photos, this one is {w}x{h}")
    k = np.diag([w / cw, h / ch, 1.0])
    return k @ mtx


def undistort(image: np.ndarray, cfg: Distortion) -> np.ndarray:
    h, w = image.shape[:2]
    mtx, dist = camera_matrix(cfg, w, h), np.array(cfg.dist, dtype=np.float64)
    new_mtx, (x, y, rw, rh) = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    out = cv2.undistort(image, mtx, dist, None, new_mtx)
    return out[y:y + rh, x:x + rw] if rw > 0 and rh > 0 else out


# ── Perspectiva ───────────────────────────────────────────────────────────────

def rectify(image: np.ndarray, cfg: Perspective) -> tuple[np.ndarray | None, float | None, str]:
    """Endereza con ArUco 0-3 (TL, TR, BR, BL) sin deformar. Devuelve (imagen, mm/px, error)."""
    centers = aruco.detect_aruco_corners(image)
    missing = [i for i in range(4) if i not in centers]
    if missing:
        return None, None, f"missing ArUco IDs {missing}"
    tl, tr, br, bl = (np.asarray(centers[i], dtype=np.float64) for i in range(4))
    w_px = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h_px = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2

    mm_per_px = None
    if cfg.width_mm and cfg.height_mm:
        px_per_mm = (w_px / cfg.width_mm + h_px / cfg.height_mm) / 2
        w_px, h_px = cfg.width_mm * px_per_mm, cfg.height_mm * px_per_mm
        mm_per_px = 1 / px_per_mm

    m = cfg.margin_px
    dst = np.float32([[m, m], [m + w_px, m], [m + w_px, m + h_px], [m, m + h_px]])
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), dst)
    size = (int(round(w_px + 2 * m)), int(round(h_px + 2 * m)))
    out = cv2.warpPerspective(image, M, size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return out, mm_per_px, ""


# ── Color ─────────────────────────────────────────────────────────────────────

def color_correct(image: np.ndarray, cfg: ColorCard) -> tuple[np.ndarray | None, dict]:
    """Corrección de color con la tarjeta (core.corrections.colorcard)."""
    from fenotit.core.corrections import colorcard as cc
    return cc.apply_config(image, cfg)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def apply(image: np.ndarray, corr: Corrections) -> tuple[np.ndarray, CorrectionInfo]:
    info = CorrectionInfo()
    out = image

    if corr.distortion.enabled and corr.distortion.mtx:
        try:
            out = undistort(out, corr.distortion)
            info.applied.append("distortion")
        except ValueError as e:                       # foto de otro tamaño que el tablero
            _log.warning("Distorsión: %s", e)
            info.warnings.append(f"distortion: {e}")
        except Exception as e:
            _log.exception("Distorsión falló")
            info.warnings.append(f"distortion: {e}")

    if corr.perspective.enabled:
        fixed, mm_per_px, err = rectify(out, corr.perspective)
        if fixed is None:
            info.warnings.append(f"perspective: {err}")
            found = aruco.detect_aruco_corners(out)
            info.aruco_missing = [i for i in range(4) if i not in found]
        else:
            out, info.mm_per_px = fixed, mm_per_px
            info.applied.append("perspective")

    if corr.color.enabled:
        try:
            fixed, cinfo = color_correct(out, corr.color)
            if fixed is None:
                info.warnings.append(f"color: {cinfo.get('error')}")
            else:
                out = fixed
                info.applied.append("color")
                info.color = cinfo
        except Exception as e:
            _log.exception("Corrección de color falló")
            info.warnings.append(f"color: {e}")

    for w in info.warnings:
        _log.warning(w)
    return out, info
