"""MobileSAM: con un clic en una semilla sale una máscara de su tamaño, y el análisis de
objetos usa los objetos que separa el modelo. Se salta si el modelo no está instalado."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fenotit.core.segmentation import sam


def test_click_and_objects():
    if not sam.available():
        print("    (sin modelo: se salta)")
        return
    img = cv2.imread(str(Path(__file__).parent / "images" / "01.jpg"))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 2]
    _, m = cv2.threshold(lab, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, cc, st, cen = cv2.connectedComponentsWithStats(m)
    i = 1 + int(np.argsort(-st[1:, 4])[len(st) // 2])     # una semilla de tamaño típico
    h, w = img.shape[:2]
    labels = sam.from_points(img, [[cen[i][0] / w, cen[i][1] / h, 1]])
    assert labels.max() == 1
    a = (labels == 1).sum()
    assert 0.5 < a / st[i, 4] < 2.0, (a, st[i, 4])
    from fenotit.core.analysis import objects
    seg = {"method": "model", "model": {"mode": "points", "points": [[cen[i][0] / w, cen[i][1] / h, 1]]},
           "refine": []}
    res = objects.run(img, {"segmentation": seg})
    assert res.stats["n_objects"] == 1


if __name__ == "__main__":
    test_click_and_objects(); print("ok  test_click_and_objects")
