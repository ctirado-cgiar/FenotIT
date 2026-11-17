"""
Tarjeta de color sin dependencias externas (OpenCV + NumPy).

detect()    busca los cuadritos (zonas lisas, casi cuadradas, del mismo tamaño y juntas),
            arma la cuadrícula con una homografía (sirve girada a cualquier ángulo y algo
            inclinada) y completa los que no se vieron (p. ej. el negro sobre borde negro).
measure()   color medio del centro de cada cuadrito.
orient()    gira la cuadrícula medida hasta la orientación que mejor ajusta la referencia.
correct()   corrección afín en RGB 0-1 (la misma que PlantCV `affine_color_correction`).

Referencia por defecto: ColorChecker de 24 colores (X-Rite Classic / Mini / Passport y la
24ColorCard de CameraTrax, que tiene los mismos colores girados 90°), valores sRGB
publicados (los mismos que usa PlantCV).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 4 filas × 6 columnas, como la ColorChecker clásica (fila de grises abajo)
COLORCHECKER24 = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
    [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
    [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
], float).reshape(4, 6, 3) / 255.0

DETECT_SIDE = 1600


@dataclass
class Card:
    rows: int
    cols: int
    centers: np.ndarray        # (rows, cols, 2) en px de la foto, en el orden de la cuadrícula vista
    chip: float                # lado aproximado de un cuadrito (px)
    found: int                 # cuadritos vistos (el resto se completó con la cuadrícula)

    def to_dict(self) -> dict:
        return {"rows": self.rows, "cols": self.cols, "centers": self.centers.round(1).tolist(),
                "chip": round(float(self.chip), 1), "found": self.found}

    @classmethod
    def from_dict(cls, data: dict) -> "Card":
        return cls(data["rows"], data["cols"], np.array(data["centers"], float), data["chip"], data.get("found", 0))


def _candidates(small: np.ndarray):
    """Zonas lisas rodeadas de bordes, casi cuadradas: (cx, cy, área, lado)."""
    blur = cv2.GaussianBlur(small, (3, 3), 0)
    edges = np.zeros(small.shape[:2], np.uint8)
    for channel in cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)):
        edges |= cv2.Canny(channel, 20, 50)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    n_regions, lab, region_stats, _ = cv2.connectedComponentsWithStats((edges == 0).astype(np.uint8), connectivity=4)
    out = []
    for i in range(1, n_regions):
        x, y, box_width, box_height, area = region_stats[i]
        if area < 40 or area > 0.01 * edges.size or max(box_width, box_height) > 2.5 * max(1, min(box_width, box_height)):
            continue
        contours, _ = cv2.findContours((lab[y:y + box_height, x:x + box_width] == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        (cx, cy), (rect_width, rect_height), _ = cv2.minAreaRect(contours[0])
        if min(rect_width, rect_height) == 0 or max(rect_width, rect_height) / min(rect_width, rect_height) > 1.35 or area / (rect_width * rect_height) < 0.85:
            continue
        out.append((cx + x, cy + y, area, (rect_width + rect_height) / 2))
    return np.array(out, float).reshape(-1, 4)


def _cluster(candidates: np.ndarray, pitch: float) -> np.ndarray:
    """El grupo más grande de cuadritos vecinos (la tarjeta)."""
    pts = candidates[:, :2]
    distances = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    near = distances < 1.7 * pitch
    seen, best = set(), []
    for start in range(len(pts)):
        if start in seen:
            continue
        group, todo = [], [start]
        seen.add(start)
        while todo:
            i = todo.pop()
            group.append(i)
            for j in np.nonzero(near[i])[0]:
                if j not in seen:
                    seen.add(j)
                    todo.append(j)
        if len(group) > len(best):
            best = group
    return candidates[sorted(best)]


def _grid_indices(pts: np.ndarray, pitch: float):
    """Índices (i, j) de cada punto en la cuadrícula, sin importar el giro."""
    offsets = pts[:, None] - pts[None]
    dist = np.linalg.norm(offsets, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest_offset = offsets[np.arange(len(pts)), dist.argmin(1)]
    ang = np.arctan2(nearest_offset[:, 1], nearest_offset[:, 0])
    theta = np.angle(np.exp(4j * ang).mean()) / 4               # dirección de la cuadrícula (mód. 90°)
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    rotated = (pts - pts.mean(0)) @ rot.T
    idx = np.zeros_like(rotated)
    for axis in (0, 1):
        coords = rotated[:, axis]
        spacing = pitch
        origin = coords.min()
        for _ in range(3):                                      # paso y origen por mínimos cuadrados
            steps = np.round((coords - origin) / spacing)
            if np.ptp(steps) > 0:
                spacing, origin = np.polyfit(steps, coords, 1)
            else:
                break
        idx[:, axis] = np.round((coords - origin) / spacing)
    idx -= idx.min(0)
    return idx.astype(int)


def detect(image: np.ndarray, rows: int = 4, cols: int = 6) -> Card | None:
    """La tarjeta de rows × cols cuadritos (en cualquier giro) o None."""
    height, width = image.shape[:2]
    shrink = min(1.0, DETECT_SIDE / max(height, width))
    small = cv2.resize(image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA) if shrink < 1 else image
    candidates = _candidates(small)
    if len(candidates) < 4:
        return None
    side = np.median(candidates[:, 3])
    candidates = candidates[(candidates[:, 3] > 0.6 * side) & (candidates[:, 3] < 1.6 * side)]
    if len(candidates) < 4:
        return None
    pts = candidates[:, :2]
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    pitch = float(np.median(dist.min(1)))
    candidates = _cluster(candidates, pitch)
    need = max(4, int(0.5 * rows * cols))
    if len(candidates) < need:
        return None
    idx = _grid_indices(candidates[:, :2], pitch)
    grid_rows, grid_cols = idx[:, 1].max() + 1, idx[:, 0].max() + 1           # filas (v) y columnas (u) vistas
    if sorted((grid_rows, grid_cols)) != sorted((rows, cols)):
        return None
    if len({tuple(i) for i in idx}) < len(idx):                  # dos puntos en la misma celda
        return None
    homography, inl = cv2.findHomography(idx.astype(np.float32), candidates[:, :2].astype(np.float32), cv2.RANSAC, 0.25 * pitch)
    if homography is None:
        return None
    grid_y, grid_x = np.mgrid[0:grid_rows, 0:grid_cols]
    grid = np.stack([grid_x, grid_y], -1).reshape(-1, 1, 2).astype(np.float32)
    centers = cv2.perspectiveTransform(grid, homography).reshape(grid_rows, grid_cols, 2) / shrink
    return Card(int(grid_rows), int(grid_cols), centers, float(np.median(candidates[:, 3])) / shrink, len(candidates))


def measure(image: np.ndarray, card: Card, frac: float = 0.4) -> np.ndarray:
    """Color medio (RGB 0-1) del centro de cada cuadrito: (filas, cols, 3). Un cuadrito
    que no queda entero dentro de la foto (tarjeta cortada por el borde) queda en NaN."""
    half = max(1, int(card.chip * frac / 2))
    height, width = image.shape[:2]
    out = np.full((card.rows, card.cols, 3), np.nan)
    for j in range(card.rows):
        for i in range(card.cols):
            x, y = card.centers[j, i].round().astype(int)
            if half <= x < width - half and half <= y < height - half:
                out[j, i] = image[y - half:y + half + 1, x - half:x + half + 1].reshape(-1, 3).mean(0)[::-1] / 255.0
    return out


def visible(measured: np.ndarray) -> int:
    """Cuadritos medidos (dentro de la foto)."""
    return int(np.isfinite(measured.reshape(-1, 3)).all(1).sum())


def _affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    source_matrix = np.hstack([src.reshape(-1, 3), np.ones((src.size // 3, 1))])
    return np.linalg.pinv(source_matrix) @ dst.reshape(-1, 3)                    # (4, 3)


def _fit_error(src, dst) -> float:
    src, dst = src.reshape(-1, 3), dst.reshape(-1, 3)
    ok = np.isfinite(src).all(1)
    affine = _affine(src[ok], dst[ok])
    source_matrix = np.hstack([src[ok], np.ones((int(ok.sum()), 1))])
    return float(np.abs(source_matrix @ affine - dst[ok]).mean())


def orient(measured: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, int]:
    """Gira la medición (0, 90, 180, 270°) a la forma de la referencia; elige la que mejor
    ajusta. Devuelve (medición girada, vueltas)."""
    options = [(np.rot90(measured, turns), turns) for turns in range(4)]
    options = [(rotated, turns) for rotated, turns in options if rotated.shape == reference.shape]
    if not options:
        raise ValueError(f"card {measured.shape[0]}x{measured.shape[1]} does not match reference "
                         f"{reference.shape[0]}x{reference.shape[1]}")
    return min(options, key=lambda option: _fit_error(option[0], reference))


MAX_MISSING = 4                                    # cuadritos que pueden faltar (fuera de la foto)


def matrix(src: np.ndarray, reference: np.ndarray, max_drop: int = 4) -> tuple[np.ndarray, list[int]]:
    """Matriz afín (4×3) que lleva los colores medidos a la referencia (RGB 0-1). Los
    cuadritos que no ajustan (reflejo, sombra, tapados) se dejan fuera: hasta max_drop."""
    source, target = src.reshape(-1, 3), reference.reshape(-1, 3)
    keep = np.isfinite(source).all(1)
    dropped: list[int] = list(np.nonzero(~keep)[0])
    for _ in range(max_drop):
        affine = _affine(source[keep], target[keep])
        res = np.linalg.norm(np.hstack([source, np.ones((len(source), 1))]) @ affine - target, axis=1)
        res[~keep] = 0
        worst = int(res.argmax())
        if res[worst] <= max(3 * np.median(res[keep]), 0.08):
            break
        keep[worst] = False
        dropped.append(worst)
    return _affine(source[keep], target[keep]), sorted(int(i) for i in dropped)


def apply_matrix(image: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Aplica la corrección afín a una foto BGR uint8, por bloques de filas."""
    height, width = image.shape[:2]
    out = np.empty_like(image)
    flat_in, flat_out = image.reshape(-1, 3), out.reshape(-1, 3)
    linear, offset = affine[:3, ::-1][::-1].astype(np.float32), affine[3, ::-1].astype(np.float32)     # en orden BGR
    step = 1 << 20
    for start in range(0, height * width, step):
        pixels = flat_in[start:start + step].astype(np.float32) / 255.0
        flat_out[start:start + step] = np.clip((pixels @ linear + offset) * 255.0, 0, 255).astype(np.uint8)
    return out


def correct(image: np.ndarray, card: Card | None = None, reference: np.ndarray = COLORCHECKER24):
    """Detecta (si hace falta), mide, orienta y corrige. Devuelve (foto, info)."""
    card = card or detect(image, *reference.shape[:2])
    if card is None:
        return None, {"error": "card not found"}
    measured = measure(image, card)
    if visible(measured) < card.rows * card.cols - MAX_MISSING:
        return None, {"error": "card incomplete", "card": card, "visible": visible(measured),
                      "chips": card.rows * card.cols}
    src, turns = orient(measured, reference)
    affine, dropped = matrix(src, reference)
    fixed = apply_matrix(image, affine)
    after = np.rot90(measure(fixed, card), turns).reshape(-1, 3)
    ok = np.ones(len(after), bool)
    ok[dropped] = False
    err = float(np.abs(after[ok] - reference.reshape(-1, 3)[ok]).mean() * 255)
    return fixed, {"card": card, "turns": turns, "matrix": affine, "found": card.found,
                   "chips": card.rows * card.cols, "dropped": dropped, "mean_error_rgb": round(err, 2)}


# ── Configuración del proyecto (pipeline.ColorCard) ─────────────────────────────

def reference_of(cfg) -> np.ndarray:
    """Colores a los que se lleva la tarjeta: los de la marca o los de la foto de referencia."""
    rows, cols = cfg.grid()
    if cfg.reference == "photo":
        if not cfg.target:
            raise ValueError("no reference photo measured")
        return np.array(cfg.target, float)
    if cfg.card == "colorchecker24":
        return COLORCHECKER24
    if not cfg.values or len(cfg.values) != rows * cols:
        raise ValueError(f"custom card needs {rows * cols} reference colors")
    return np.array(cfg.values, float).reshape(rows, cols, 3) / 255.0


def place_card(cfg, image: np.ndarray) -> Card | None:
    """La tarjeta en esta foto: buscada, o en el lugar guardado (escalado a su tamaño)."""
    rows, cols = cfg.grid()
    if cfg.mode == "fixed" and cfg.place:
        card = Card.from_dict(cfg.place)
        saved_width, saved_height = cfg.place.get("size", (image.shape[1], image.shape[0]))
        factor = np.array([image.shape[1] / saved_width, image.shape[0] / saved_height])
        card.centers = card.centers * factor
        card.chip *= float(factor.mean())
        return card
    return detect(image, rows, cols)


def region_mean(image: np.ndarray, region) -> np.ndarray:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = region
    patch = image[int(min(y0, y1) * height):int(max(y0, y1) * height) + 1, int(min(x0, x1) * width):int(max(x0, x1) * width) + 1]
    if patch.size == 0:
        raise ValueError("empty white/gray area")
    return patch.reshape(-1, 3).mean(0)[::-1] / 255.0          # RGB 0-1


def white_balance(image: np.ndarray, cfg) -> tuple[np.ndarray | None, dict]:
    """Ganancia por canal para que el área blanca/gris quede neutra (sin cambiar su
    brillo) o igual a la de la foto de referencia."""
    if not cfg.region:
        return None, {"error": "no white/gray area marked"}
    measured = region_mean(image, cfg.region)
    if cfg.reference == "photo" and cfg.target:
        goal = np.array(cfg.target, float).reshape(-1)[:3]
    else:
        goal = np.full(3, measured.mean())
    gains = goal / np.maximum(measured, 1e-3)
    affine = np.vstack([np.diag(gains), np.zeros(3)])
    return apply_matrix(image, affine), {"gains": gains.round(4).tolist(), "measured": measured.round(4).tolist()}


def apply_config(image: np.ndarray, cfg) -> tuple[np.ndarray | None, dict]:
    if cfg.card == "white":
        return white_balance(image, cfg)
    ref = reference_of(cfg)
    card = place_card(cfg, image)
    if card is None:
        return None, {"error": "card not found"}
    return correct(image, card, ref)


def measure_reference(image: np.ndarray, cfg) -> list:
    """Colores de la foto de referencia (para `reference = photo`), en el orden de la
    cuadrícula; con tarjeta de 24 colores se ordenan como la ColorChecker."""
    if cfg.card == "white":
        return region_mean(image, cfg.region).tolist()
    card = place_card(cfg, image)
    if card is None:
        raise ValueError("card not found")
    measured = measure(image, card)
    if visible(measured) < measured.shape[0] * measured.shape[1]:
        raise ValueError("card incomplete")
    if cfg.card == "colorchecker24":
        measured, _ = orient(measured, COLORCHECKER24)
    elif measured.shape[:2] != (cfg.rows, cfg.cols):
        measured = np.rot90(measured)
    return measured.round(5).tolist()


def preview(image: np.ndarray, card: Card, dropped=(), turns: int = 0, side: int = 520) -> np.ndarray:
    """Recorte de la tarjeta con cada cuadrito numerado (y tachado si se descartó)."""
    pts = card.centers.reshape(-1, 2)
    pad = card.chip
    x0, y0 = np.maximum(pts.min(0) - pad, 0).astype(int)
    x1, y1 = np.minimum(pts.max(0) + pad, [image.shape[1], image.shape[0]]).astype(int)
    crop = image[y0:y1, x0:x1].copy()
    zoom = side / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
    order = np.rot90(np.arange(card.rows * card.cols).reshape(card.rows, card.cols), turns).reshape(-1)
    number = {int(grid_index): reference_index for reference_index, grid_index in enumerate(order)}                # cuadrícula vista -> nº en la referencia
    radius = max(3, int(card.chip * zoom * 0.2))
    for grid_index, (x, y) in enumerate(pts):
        cx, cy = int((x - x0) * zoom), int((y - y0) * zoom)
        reference_index = number.get(grid_index, grid_index)
        bad = reference_index in dropped
        cv2.circle(crop, (cx, cy), radius, (0, 0, 255) if bad else (255, 255, 255), 2, cv2.LINE_AA)
        if bad:
            cv2.line(crop, (cx - radius, cy - radius), (cx + radius, cy + radius), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(crop, str(reference_index + 1), (cx - radius, cy - radius - 3), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, radius / 22),
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(crop, str(reference_index + 1), (cx - radius, cy - radius - 3), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, radius / 22),
                    (0, 0, 0), 1, cv2.LINE_AA)
    return crop
