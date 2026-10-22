"""Segmentación por ejemplos: con unos trazos de fondo, sano y lesión se recupera el %
de lesión de cada objeto (escena sintética con ruido)."""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fenotit.core.analysis import objects
from fenotit.core.segmentation import classifier as C


def scene():
    rng = np.random.default_rng(3)
    img = np.full((600, 900, 3), (120, 118, 115), np.uint8)
    truth = []
    for (cx, cy), r_les in (((230, 300), 40), ((640, 300), 70)):
        leaf = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(leaf, (cx, cy), (170, 110), 15, 0, 360, 1, -1)
        les = np.zeros_like(leaf)
        cv2.circle(les, (cx + 30, cy - 10), r_les, 1, -1)
        les &= leaf
        img[leaf > 0] = (40, 150, 70)
        img[les > 0] = (30, 70, 190)
        truth.append(100 * les.sum() / leaf.sum())
    noise = rng.normal(0, 9, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8), truth


def line(c, a, b, w=0.012):
    W, H = 900, 600
    return {"c": c, "w": w, "pts": [[a[0] / W, a[1] / H], [b[0] / W, b[1] / H]]}


def test_lesion_percent():
    img, truth = scene()
    strokes = [line(0, (20, 40), (880, 40)), line(0, (20, 560), (880, 560)), line(0, (450, 100), (450, 500)),
               line(1, (120, 300), (180, 330)), line(1, (560, 400), (720, 400)),
               line(2, (250, 290), (270, 290)), line(2, (650, 280), (700, 300))]
    small = C.work_copy(img, 1.0)
    X, y = C.balance(*C.samples(C.features(small), strokes))
    with tempfile.TemporaryDirectory() as tmp:
        path, key = C.store(X, y, Path(tmp))
        seg = {"method": "classifier", "refine": [{"op": "fill_holes"}],
               "classifier": {"samples": path, "hash": key, "objects": [1, 2],
                              "names": ["Background", "Healthy", "Lesion"],
                              "colors": ["#888888", "#2E7D32", "#C62828"]}}
        res = objects.run(img, {"segmentation": seg, "exclude_border": True, "measure_size": True})
    rows = sorted(res.extra["tables"]["objects"], key=lambda r: r["centroid_x_px"])
    assert len(rows) == 2, rows
    for r, tr in zip(rows, truth):
        assert abs(r["lesion_pct"] - tr) < 2.0, (r["lesion_pct"], tr)
    assert "lesion_pct" in res.stats


if __name__ == "__main__":
    test_lesion_percent(); print("ok  test_lesion_percent")
