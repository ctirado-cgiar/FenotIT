"""Clasificación de regiones aprendida de los clics: con regiones verdes (planta) y
cafés (suelo) clasificadas en una imagen, clasifica bien las de otra imagen."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fenotit.core.segmentation import regions  # noqa: E402

GREEN, BROWN, SOIL = (60, 170, 60), (40, 80, 130), (90, 110, 120)


def _scene(seed: int):
    """Imagen BGR con regiones verdes y cafés sobre suelo; etiquetas y categoría real
    de cada región (1 = planta, 0 = sin usar)."""
    rng = np.random.default_rng(seed)
    img = np.full((400, 600, 3), SOIL, np.uint8)
    labels = np.zeros(img.shape[:2], np.int32)
    truth = [0]
    for n in range(1, 21):
        plant = n % 2 == 0
        cx, cy = rng.integers(40, 560), rng.integers(40, 360)
        ax, ay = (rng.integers(14, 30), rng.integers(8, 16)) if plant else (rng.integers(8, 14), rng.integers(8, 14))
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), float(rng.integers(0, 180)), 0, 360, 1, -1)
        color = np.array(GREEN if plant else BROWN) + rng.integers(-15, 15, 3)
        img[m > 0] = np.clip(color, 0, 255)
        labels[m > 0] = n
        truth.append(1 if plant else 0)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8), labels, np.array(truth)


def test_learns_from_other_image():
    img, lab, truth = _scene(1)
    X, y = regions.examples(img, lab, truth)
    model = regions.train(X, y, "test")
    img2, lab2, truth2 = _scene(2)
    ok = np.unique(lab2)[1:]
    pred = regions.predict(model, img2, lab2)
    assert (pred[ok] == truth2[ok]).mean() >= 0.95, (pred[ok], truth2[ok])


def test_single_category():
    img, lab, _ = _scene(3)
    X, y = regions.examples(img, lab, np.ones(lab.max() + 1, int))
    model = regions.train(X, y, "one")
    assert (regions.predict(model, img, lab)[1:] == 1).all()


if __name__ == "__main__":
    for fn in (test_learns_from_other_image, test_single_category):
        fn()
        print("ok ", fn.__name__)
