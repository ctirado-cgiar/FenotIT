"""Segmentación por ejemplos: el usuario pinta trazos de cada clase y un bosque aleatorio
(random forest) clasifica cada píxel con rasgos de color y textura a varias escalas.

- La clasificación se hace en una copia de `detail` megapíxeles (1 MP por defecto); las
  probabilidades se amplían a la foto con interpolación bilineal.
- Los rasgos de cada foto se calculan una vez y se guardan; al cambiar los trazos solo se
  vuelve a entrenar.
- Bosque de 40 árboles de profundidad 12.
- Dos modelos: uno de contexto (rasgos suavizados con σ = 1, 3 y 7 px) decide en toda la
  foto; otro con rasgos locales (color del píxel y σ = 1) decide en una franja alrededor
  de los bordes que encuentra el primero.

Los ejemplos (rasgos de los píxeles pintados) se guardan en un .npz en la carpeta de datos
de la app; la cadena del análisis lleva su ruta y su huella (hash), y cada proceso del
lote entrena una vez con esos datos."""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

SIGMAS = (1.0, 3.0, 7.0)
LOCAL = 7                  # rasgos locales: L*a*b* y saturación del píxel + L*a*b* con σ = 1
BAND = int(2 * SIGMAS[-1])
TREES, DEPTH = 40, 12
MAX_SAMPLES_PER_CLASS = 6000


def work_scale(shape, detail_mpx: float) -> float:
    height, width = shape[:2]
    return min(1.0, (detail_mpx * 1e6 / (height * width)) ** 0.5) if detail_mpx else 1.0


def work_copy(img: np.ndarray, detail_mpx: float) -> np.ndarray:
    factor = work_scale(img.shape, detail_mpx)
    return img if factor >= 1 else cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def features(img: np.ndarray) -> np.ndarray:
    """Rasgos por píxel (float32, H×W×22): L*a*b*, saturación y, a 3 escalas, el color
    suavizado, el gradiente y el laplaciano de la luminosidad (textura y bordes)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    out = [lab, sat[..., None]]
    for sigma in SIGMAS:
        blurred = cv2.GaussianBlur(lab, (0, 0), sigma)
        lightness = blurred[..., 0]
        grad_x, grad_y = cv2.Sobel(lightness, cv2.CV_32F, 1, 0), cv2.Sobel(lightness, cv2.CV_32F, 0, 1)
        out += [blurred, cv2.magnitude(grad_x, grad_y)[..., None], cv2.Laplacian(lightness, cv2.CV_32F)[..., None]]
    return np.concatenate(out, axis=-1)


def rasterize(strokes: list[dict], shape) -> np.ndarray:
    """Trazos (coordenadas 0-1, ancho relativo al ancho de la foto) → mapa de etiquetas
    (0 = sin pintar, clase + 1 donde se pintó). El último trazo manda."""
    image_height, image_width = shape[:2]
    lab = np.zeros((image_height, image_width), np.uint8)
    for stroke in strokes or []:
        pts = np.asarray(stroke.get("pts") or [], np.float64)
        if not len(pts):
            continue
        pts = np.round(pts * [image_width - 1, image_height - 1]).astype(np.int32)
        width = max(1, int(round(float(stroke.get("w", 0.01)) * image_width)))
        class_value = int(stroke["c"]) + 1
        if len(pts) == 1:
            cv2.circle(lab, tuple(pts[0]), max(1, width // 2), class_value, -1)
        else:
            cv2.polylines(lab, [pts.reshape(-1, 1, 2)], False, class_value, width, lineType=cv2.LINE_8)
    return lab


def samples(feat: np.ndarray, strokes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    lab = rasterize(strokes, feat.shape)
    painted = lab > 0
    return feat[painted], (lab[painted] - 1).astype(np.int16)


def balance(feature_rows: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A lo más MAX_SAMPLES_PER_CLASS por clase (un trazo enorme no tapa a los demás)."""
    rng = np.random.default_rng(seed)
    keep = []
    for class_value in np.unique(y):
        idx = np.flatnonzero(y == class_value)
        if len(idx) > MAX_SAMPLES_PER_CLASS:
            idx = np.sort(rng.choice(idx, MAX_SAMPLES_PER_CLASS, replace=False))
        keep.append(idx)
    keep = np.concatenate(keep) if keep else np.zeros(0, int)
    return feature_rows[keep], y[keep]


def fingerprint(feature_rows: np.ndarray, y: np.ndarray) -> str:
    hasher = hashlib.blake2b(digest_size=12)
    hasher.update(np.ascontiguousarray(feature_rows).tobytes())
    hasher.update(np.ascontiguousarray(y).tobytes())
    return hasher.hexdigest()


def store(feature_rows: np.ndarray, y: np.ndarray, folder: Path) -> tuple[str, str]:
    key = fingerprint(feature_rows, y)
    path = Path(folder) / f"examples_{key}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=feature_rows.astype(np.float32), y=y.astype(np.int16))
    return str(path), key


_MODELS: "OrderedDict[str, object]" = OrderedDict()


def _forest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(TREES, max_depth=DEPTH, n_jobs=-1, random_state=0, class_weight="balanced")


def train(feature_rows: np.ndarray, y: np.ndarray, key: str | None = None):
    """Los dos bosques (contexto y local); guarda los 3 últimos pares en memoria."""
    key = key or fingerprint(feature_rows, y)
    model = _MODELS.get(key)
    if model is None:
        model = (_forest().fit(feature_rows, y), _forest().fit(feature_rows[:, :LOCAL], y))
        _MODELS[key] = model
        while len(_MODELS) > 3:
            _MODELS.popitem(last=False)
    _MODELS.move_to_end(key)
    return model


def load_model(path: str, key: str):
    if key in _MODELS:
        _MODELS.move_to_end(key)
        return _MODELS[key]
    with np.load(path) as data:
        feature_rows, y = data["X"], data["y"]
    return train(feature_rows, y, key)


def predict(model, feat: np.ndarray, out_shape=None, region: np.ndarray | None = None) -> np.ndarray:
    """Mapa de clases (uint8). Con `out_shape` las probabilidades se amplían a ese tamaño
    antes de elegir la clase (bordes suaves, sin escalones). `region` (en el tamaño de
    `feat`) limita el cálculo al área de análisis."""
    ctx_model, local = model
    height, width = feat.shape[:2]
    flat = feat.reshape(-1, feat.shape[-1])
    classes = ctx_model.classes_
    idx = np.flatnonzero(region.reshape(-1)) if region is not None and region.any() and not region.all() else None
    prob = np.zeros((height * width, len(classes)), np.float32)
    prob[:, 0] = 1.0
    if idx is None:
        prob[:] = ctx_model.predict_proba(flat)
    else:
        prob[idx] = ctx_model.predict_proba(flat[idx])
    # franja de los bordes: decide el modelo local
    first = classes[np.argmax(prob, axis=1)].reshape(height, width).astype(np.uint8)
    edge = cv2.morphologyEx(first, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    band = cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND + 1,) * 2))
    if region is not None:
        band &= region.astype(np.uint8)
    band_index = np.flatnonzero(band.reshape(-1))
    if len(band_index):
        prob[band_index] = local.predict_proba(flat[band_index, :LOCAL])
    prob = prob.reshape(height, width, len(classes))
    if out_shape is not None and (out_shape[0], out_shape[1]) != (height, width):
        prob = np.dstack([cv2.resize(prob[..., i], (out_shape[1], out_shape[0]), interpolation=cv2.INTER_LINEAR)
                          for i in range(len(classes))]) if len(classes) > 1 else \
            cv2.resize(prob, (out_shape[1], out_shape[0]))[..., None]
    return classes[np.argmax(prob, axis=-1)].astype(np.uint8)


def colorize(class_map: np.ndarray, colors: list[str]) -> np.ndarray:
    """Mapa de clases pintado con el color de cada clase (BGR)."""
    lut = np.zeros((256, 3), np.uint8)
    for i, hexcol in enumerate(colors):
        hex_digits = hexcol.lstrip("#")
        lut[i] = (int(hex_digits[4:6], 16), int(hex_digits[2:4], 16), int(hex_digits[0:2], 16))
    return lut[class_map]


# ── Pieza del pipeline ──────────────────────────────────────────────────────

from fenotit.core.pipeline.base import step  # noqa: E402


@step("classify", "segmenter", provides=("mask", "classes"), params=[
    {"key": "samples", "type": "str", "default": ""},
    {"key": "hash", "type": "str", "default": ""},
    {"key": "objects", "type": "list", "default": [1]},
    {"key": "colors", "type": "list", "default": []},
    {"key": "detail", "type": "float", "default": 1.0},
])
def classify(ctx, params):
    """Segmentación por ejemplos: cada píxel recibe la clase que el bosque aleatorio
    aprendió de los trazos; la máscara = las clases marcadas como objeto."""
    if not params["samples"]:
        raise ValueError("no examples: paint at least two classes")
    model = load_model(params["samples"], params["hash"])
    small = work_copy(ctx.image, float(params["detail"]))
    full = ctx.analysis_region(ctx.image.shape)
    region = None if full is None else cv2.resize(full, (small.shape[1], small.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST) > 0
    classes = predict(model, features(small), ctx.image.shape, region)
    if full is not None:
        classes[full == 0] = 0
    ctx.classes = classes
    ctx.mask = np.isin(classes, [int(class_id) for class_id in params["objects"]]).astype(np.uint8) * 255
    ctx.images["mask"] = ctx.mask
    if params["colors"]:
        ctx.images["classes"] = colorize(classes, list(params["colors"]))
    ctx.extra["class_colors"] = list(params["colors"])
