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


@dataclass
class Perspective:
    enabled: bool = False
    margin_px: int = 20
    width_mm: float | None = None     # distancia real entre centros ID0-ID1
    height_mm: float | None = None    # distancia real entre centros ID0-ID3


@dataclass
class ColorCard:
    enabled: bool = False
    mode: str = "per_image"           # per_image | fixed
    pos: int | str = "auto"          # 0-3 u "auto" (elige la orientación con menor error)
    radius: int = 20
    mask_file: str | None = None      # relativo a la carpeta del proyecto
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class Corrections:
    distortion: Distortion = field(default_factory=Distortion)
    perspective: Perspective = field(default_factory=Perspective)
    color: ColorCard = field(default_factory=ColorCard)

    def active(self) -> list[str]:
        return [n for n in ("distortion", "perspective", "color") if getattr(self, n).enabled]

    def to_dict(self) -> dict:
        color = {k: v for k, v in asdict(self.color).items() if k != "mask"}
        return {"distortion": asdict(self.distortion),
                "perspective": asdict(self.perspective),
                "color": color}

    @classmethod
    def from_dict(cls, d: dict | None, folder: Path | None = None) -> "Corrections":
        d = d or {}
        c = cls(Distortion(**d.get("distortion", {})),
                Perspective(**d.get("perspective", {})),
                ColorCard(**d.get("color", {})))
        if c.color.mask_file and folder:
            p = folder / c.color.mask_file
            if p.exists():
                c.color.mask = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        return c

    def save_files(self, folder: Path) -> None:
        if self.color.mask is not None:
            rel = "calibration/colorcard_mask.png"
            ok, buf = cv2.imencode(".png", self.color.mask)
            if ok:
                (folder / rel).parent.mkdir(parents=True, exist_ok=True)
                (folder / rel).write_bytes(buf.tobytes())
                self.color.mask_file = rel


@dataclass
class CorrectionInfo:
    applied: list[str] = field(default_factory=list)
    mm_per_px: float | None = None    # escala obtenida de ArUco con medidas reales
    warnings: list[str] = field(default_factory=list)
    aruco_missing: list[int] = field(default_factory=list)


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

def plantcv_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("plantcv") is not None


def detect_card(image: np.ndarray, radius: int = 6) -> np.ndarray | None:
    from plantcv import plantcv as pcv
    pcv.params.debug = None
    return pcv.transform.detect_color_card(rgb_img=image, radius=radius)


def _fit_error(card: np.ndarray, std: np.ndarray) -> float:
    X = np.hstack([card[:, 1:], np.ones((len(card), 1))])
    A, *_ = np.linalg.lstsq(X, std[:, 1:], rcond=None)
    return float(np.abs(X @ A - std[:, 1:]).mean())


def color_correct(image: np.ndarray, mask: np.ndarray, pos: int | str = "auto") -> tuple[np.ndarray | None, str]:
    from plantcv import plantcv as pcv
    pcv.params.debug = None
    _, card = pcv.transform.get_color_matrix(rgb_img=image, mask=mask)
    options = range(4) if pos == "auto" else [int(pos)]
    stds = {p: pcv.transform.std_color_matrix(pos=p) for p in options}
    stds = {p: m for p, m in stds.items() if m.shape == card.shape}
    if not stds:
        return None, f"card has {len(card)} chips, expected 24"
    best = min(stds, key=lambda p: _fit_error(card, stds[p]))
    std = stds[best]
    out = pcv.transform.affine_color_correction(rgb_img=image, source_matrix=card, target_matrix=std)
    return out, ""


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
        if not plantcv_available():
            info.warnings.append("color: PlantCV not installed")
        else:
            try:
                mask = detect_card(out, corr.color.radius) if corr.color.mode == "per_image" else corr.color.mask
                if mask is None:
                    info.warnings.append("color: card not found")
                elif mask.shape[:2] != out.shape[:2]:
                    info.warnings.append("color: reference mask size differs from image")
                else:
                    fixed, err = color_correct(out, mask, corr.color.pos)
                    if fixed is None:
                        info.warnings.append(f"color: {err}")
                    else:
                        out = fixed
                        info.applied.append("color")
            except Exception as e:
                _log.exception("Corrección de color falló")
                info.warnings.append(f"color: {e}")

    for w in info.warnings:
        _log.warning(w)
    return out, info
