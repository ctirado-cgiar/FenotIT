"""Segmentación por ejemplos (como ilastik): el usuario pinta trazos de cada clase y un
bosque aleatorio aprende a clasificar cada píxel con rasgos de color y textura a varias
escalas.

Para que responda rápido en un computador común:
  - se trabaja en una copia de `detail` megapíxeles (1 MP por defecto) y las
    probabilidades se amplían a la foto con interpolación bilineal;
  - los rasgos de la foto se calculan una vez y se guardan (cambiar los trazos solo
    vuelve a entrenar, que tarda décimas de segundo);
  - 40 árboles de profundidad 12 (más no mejora con pocos trazos y es más lento).

Bordes: los rasgos suavizados (σ = 3 y 7 px) dan contexto, pero cerca del borde de un
objeto mezclan objeto y fondo, valores que no aparecen en los trazos (pintados lejos del
borde): solo con ellos los objetos salían ~25 % más grandes en una escena de prueba. Por
eso hay dos modelos: el de contexto decide en general y, en una franja alrededor de los
bordes que encuentra, decide un segundo modelo con rasgos locales (color del píxel y
σ = 1). En la escena de prueba el error del área baja de +25 % a ~2 %.

Los ejemplos (rasgos de los píxeles pintados) se guardan en un .npz en la carpeta de
datos de la app; la cadena del análisis solo lleva su ruta y su huella (hash), así el
lote en procesos aparte entrena una vez por proceso con exactamente los mismos datos."""
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
    h, w = shape[:2]
    return min(1.0, (detail_mpx * 1e6 / (h * w)) ** 0.5) if detail_mpx else 1.0


def work_copy(img: np.ndarray, detail_mpx: float) -> np.ndarray:
    k = work_scale(img.shape, detail_mpx)
    return img if k >= 1 else cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


def features(img: np.ndarray) -> np.ndarray:
    """Rasgos por píxel (float32, H×W×22): L*a*b*, saturación y, a 3 escalas, el color
    suavizado, el gradiente y el laplaciano de la luminosidad (textura y bordes)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    out = [lab, sat[..., None]]
    for s in SIGMAS:
        b = cv2.GaussianBlur(lab, (0, 0), s)
        g = b[..., 0]
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        out += [b, cv2.magnitude(gx, gy)[..., None], cv2.Laplacian(g, cv2.CV_32F)[..., None]]
    return np.concatenate(out, axis=-1)


def rasterize(strokes: list[dict], shape) -> np.ndarray:
    """Trazos (coordenadas 0-1, ancho relativo al ancho de la foto) → mapa de etiquetas
    (0 = sin pintar, clase + 1 donde se pintó). El último trazo manda."""
    h, w = shape[:2]
    lab = np.zeros((h, w), np.uint8)
    for s in strokes or []:
        pts = np.asarray(s.get("pts") or [], np.float64)
        if not len(pts):
            continue
        pts = np.round(pts * [w - 1, h - 1]).astype(np.int32)
        width = max(1, int(round(float(s.get("w", 0.01)) * w)))
        c = int(s["c"]) + 1
        if len(pts) == 1:
            cv2.circle(lab, tuple(pts[0]), max(1, width // 2), c, -1)
        else:
            cv2.polylines(lab, [pts.reshape(-1, 1, 2)], False, c, width, lineType=cv2.LINE_8)
    return lab


def samples(feat: np.ndarray, strokes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    lab = rasterize(strokes, feat.shape)
    m = lab > 0
    return feat[m], (lab[m] - 1).astype(np.int16)


def balance(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A lo más MAX_SAMPLES_PER_CLASS por clase (un trazo enorme no tapa a los demás)."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) > MAX_SAMPLES_PER_CLASS:
            idx = np.sort(rng.choice(idx, MAX_SAMPLES_PER_CLASS, replace=False))
        keep.append(idx)
    keep = np.concatenate(keep) if keep else np.zeros(0, int)
    return X[keep], y[keep]


def fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def store(X: np.ndarray, y: np.ndarray, folder: Path) -> tuple[str, str]:
    key = fingerprint(X, y)
    path = Path(folder) / f"examples_{key}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int16))
    return str(path), key


_MODELS: "OrderedDict[str, object]" = OrderedDict()


def _forest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(TREES, max_depth=DEPTH, n_jobs=-1, random_state=0, class_weight="balanced")


def train(X: np.ndarray, y: np.ndarray, key: str | None = None):
    """Los dos bosques (contexto y local); guarda los 3 últimos pares en memoria."""
    key = key or fingerprint(X, y)
    model = _MODELS.get(key)
    if model is None:
        model = (_forest().fit(X, y), _forest().fit(X[:, :LOCAL], y))
        _MODELS[key] = model
        while len(_MODELS) > 3:
            _MODELS.popitem(last=False)
    _MODELS.move_to_end(key)
    return model


def load_model(path: str, key: str):
    if key in _MODELS:
        _MODELS.move_to_end(key)
        return _MODELS[key]
    with np.load(path) as d:
        X, y = d["X"], d["y"]
    return train(X, y, key)


def predict(model, feat: np.ndarray, out_shape=None, region: np.ndarray | None = None) -> np.ndarray:
    """Mapa de clases (uint8). Con `out_shape` las probabilidades se amplían a ese tamaño
    antes de elegir la clase (bordes suaves, sin escalones). `region` (en el tamaño de
    `feat`) limita el cálculo al área de análisis."""
    ctx_model, local = model
    h, w = feat.shape[:2]
    flat = feat.reshape(-1, feat.shape[-1])
    classes = ctx_model.classes_
    idx = np.flatnonzero(region.reshape(-1)) if region is not None and region.any() and not region.all() else None
    prob = np.zeros((h * w, len(classes)), np.float32)
    prob[:, 0] = 1.0
    if idx is None:
        prob[:] = ctx_model.predict_proba(flat)
    else:
        prob[idx] = ctx_model.predict_proba(flat[idx])
    # franja de los bordes: decide el modelo local
    first = classes[np.argmax(prob, axis=1)].reshape(h, w).astype(np.uint8)
    edge = cv2.morphologyEx(first, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    band = cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND + 1,) * 2))
    if region is not None:
        band &= region.astype(np.uint8)
    b = np.flatnonzero(band.reshape(-1))
    if len(b):
        prob[b] = local.predict_proba(flat[b, :LOCAL])
    prob = prob.reshape(h, w, len(classes))
    if out_shape is not None and (out_shape[0], out_shape[1]) != (h, w):
        prob = np.dstack([cv2.resize(prob[..., i], (out_shape[1], out_shape[0]), interpolation=cv2.INTER_LINEAR)
                          for i in range(len(classes))]) if len(classes) > 1 else \
            cv2.resize(prob, (out_shape[1], out_shape[0]))[..., None]
    return classes[np.argmax(prob, axis=-1)].astype(np.uint8)


def colorize(class_map: np.ndarray, colors: list[str]) -> np.ndarray:
    """Mapa de clases pintado con el color de cada clase (BGR)."""
    lut = np.zeros((256, 3), np.uint8)
    for i, hexcol in enumerate(colors):
        h = hexcol.lstrip("#")
        lut[i] = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
    return lut[class_map]
