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
    k = min(1.0, SIDE / max(img.shape[:2]))
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        labels = cv2.resize(labels.astype(np.int32), (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    h, w = labels.shape
    n = int(labels.max())
    X = np.full((n, len(NAMES)), np.nan, np.float32)
    for i, sl in enumerate(find_objects(labels), 1):
        if sl is None:
            continue
        m = labels[sl] == i
        a = int(m.sum())
        if a == 0:
            continue
        px = lab[sl][m]
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        hull = cv2.contourArea(cv2.convexHull(c))
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        bh, bw = m.shape
        X[i - 1] = (*px.mean(0), *px.std(0), grad[sl][m].mean(), np.log10(a / (h * w)),
                    4 * np.pi * a / max(per * per, 1), a / max(hull, 1), a / (bh * bw),
                    min(rw, rh) / max(rw, rh, 1e-6),
                    float(sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w))
    return X


def examples(img: np.ndarray, labels: np.ndarray, cats: np.ndarray):
    """Rasgos y categoría de cada región de una foto clasificada por el usuario."""
    X = features(img, labels)
    y = np.asarray(cats[1:len(X) + 1], np.int16)
    ok = ~np.isnan(X).any(1)
    return X[ok], y[ok]


_MODELS: "OrderedDict[str, object]" = OrderedDict()


def train(X: np.ndarray, y: np.ndarray, key: str):
    if key in _MODELS:
        _MODELS.move_to_end(key)
        return _MODELS[key]
    if len(np.unique(y)) < 2:
        model = int(y[0]) if len(y) else 1               # una sola categoría: todas a esa
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(200, min_samples_leaf=1, n_jobs=-1, random_state=0,
                                       class_weight="balanced").fit(X, y)
    _MODELS[key] = model
    while len(_MODELS) > 3:
        _MODELS.popitem(last=False)
    return model


def load(path: str, key: str):
    if key in _MODELS:
        return train(None, None, key)
    with np.load(path) as d:
        return train(d["X"], d["y"], key)


def predict(model, img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Categoría de cada región (índice = etiqueta; 0 = sin usar)."""
    n = int(labels.max())
    cats = np.zeros(n + 1, np.int32)
    if n == 0:
        return cats
    if isinstance(model, int):
        cats[1:] = model
        return cats
    X = features(img, labels)
    ok = ~np.isnan(X).any(1)
    if ok.any():
        cats[1:][ok] = model.predict(X[ok])
    return cats
