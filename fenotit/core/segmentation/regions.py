"""Categorías de las regiones del modelo aprendidas de los clics del usuario (como la
clasificación de objetos de ilastik). En las fotos donde el usuario clasificó regiones,
cada región da un ejemplo: sus rasgos de color, tamaño y forma y su categoría (0 = sin
usar). Un bosque aleatorio aprende de todos los ejemplos y clasifica las regiones de las
fotos donde el usuario no hizo clic.

Rasgos (no dependen de la resolución de la foto): L*a*b* medio y su desviación, textura
(gradiente medio de L*), área como fracción de la foto (log), compacidad, solidez,
extensión, alargamiento y si toca el borde de la foto."""
from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np
from scipy.ndimage import find_objects

SIDE = 1024
NAMES = ("L", "a", "b", "sd_L", "sd_a", "sd_b", "texture", "log_area", "compactness",
         "solidity", "extent", "elongation", "border")


def features(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Una fila por región (etiqueta 1..n; las que no existen quedan en NaN)."""
    shrink = min(1.0, SIDE / max(img.shape[:2]))
    if shrink < 1:
        img = cv2.resize(img, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        labels = cv2.resize(labels.astype(np.int32), (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0)
    grad = cv2.magnitude(cv2.Sobel(lightness, cv2.CV_32F, 1, 0), cv2.Sobel(lightness, cv2.CV_32F, 0, 1))
    height, width = labels.shape
    n_regions = int(labels.max())
    feature_rows = np.full((n_regions, len(NAMES)), np.nan, np.float32)
    for i, bounds in enumerate(find_objects(labels), 1):
        if bounds is None:
            continue
        mask = labels[bounds] == i
        area = int(mask.sum())
        if area == 0:
            continue
        pixels = lab[bounds][mask]
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(contour, True)
        hull = cv2.contourArea(cv2.convexHull(contour))
        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
        box_height, box_width = mask.shape
        feature_rows[i - 1] = (*pixels.mean(0), *pixels.std(0), grad[bounds][mask].mean(), np.log10(area / (height * width)),
                    4 * np.pi * area / max(per * per, 1), area / max(hull, 1), area / (box_height * box_width),
                    min(rect_width, rect_height) / max(rect_width, rect_height, 1e-6),
                    float(bounds[0].start == 0 or bounds[1].start == 0 or bounds[0].stop == height or bounds[1].stop == width))
    return feature_rows


def examples(img: np.ndarray, labels: np.ndarray, cats: np.ndarray):
    """Rasgos y categoría de cada región de una foto clasificada por el usuario."""
    feature_rows = features(img, labels)
    y = np.asarray(cats[1:len(feature_rows) + 1], np.int16)
    ok = ~np.isnan(feature_rows).any(1)
    return feature_rows[ok], y[ok]


class _OneCategory:
    """Modelo para ejemplos de una sola categoría: todas las regiones van a esa."""

    def __init__(self, category: int):
        self.category = int(category)

    def predict(self, feature_rows: np.ndarray) -> np.ndarray:
        return np.full(len(feature_rows), self.category, np.int32)


# modelos entrenados en esta sesión (los 3 últimos), por huella de los ejemplos
_MODELS: "OrderedDict[str, object]" = OrderedDict()


def _cached(key: str):
    if key in _MODELS:
        _MODELS.move_to_end(key)
    return _MODELS.get(key)


def train(feature_rows: np.ndarray, y: np.ndarray, key: str):
    """Bosque aleatorio entrenado con los ejemplos (X, y); `key` = su huella."""
    model = _cached(key)
    if model is not None:
        return model
    if len(np.unique(y)) < 2:
        model = _OneCategory(y[0] if len(y) else 1)
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(200, min_samples_leaf=1, n_jobs=-1, random_state=0,
                                       class_weight="balanced").fit(feature_rows, y)
    _MODELS[key] = model
    while len(_MODELS) > 3:
        _MODELS.popitem(last=False)
    return model


def load(path: str, key: str):
    """El modelo de los ejemplos guardados en `path` (.npz con X e y)."""
    model = _cached(key)
    if model is not None:
        return model
    with np.load(path) as data:
        return train(data["X"], data["y"], key)


def predict(model, img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Categoría de cada región (índice = etiqueta; 0 = sin usar)."""
    n_regions = int(labels.max())
    cats = np.zeros(n_regions + 1, np.int32)
    if n_regions == 0:
        return cats
    feature_rows = features(img, labels)
    ok = ~np.isnan(feature_rows).any(1)
    if ok.any():
        cats[1:][ok] = model.predict(feature_rows[ok])
    return cats
