"""Conteo en fotos reales con objetos que se tocan. 23455 verificado por el usuario;
el resto revisado a ojo (el método anterior daba 200, 381, 98 y 634)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit.core import pipeline

IMAGES = Path(__file__).parent / "images"
CHAIN = [{"step": "otsu", "params": {"color_space": "LAB", "channel": 2}},
         {"step": "clean", "params": {"open_iterations": 1}},
         {"step": "separate"},
         {"step": "filter", "params": {"area_min": 300}},
         {"step": "count"}]
EXPECTED = {"23379.jpg": 204, "23455.jpg": 399, "23505.jpg": 89, "23507.jpg": 699}   # sin objetos de borde
TOLERANCE = 0.01


def test_touching_counts():
    for name, expected in EXPECTED.items():
        img = cv2.imdecode(np.fromfile(str(IMAGES / name), np.uint8), cv2.IMREAD_COLOR)
        n = pipeline.run(img, CHAIN).image_row()["n_objects"]
        assert abs(n - expected) <= max(1, TOLERANCE * expected), f"{name}: {n} (esperado {expected})"
        print(f"    {name}: {n}")


if __name__ == "__main__":
    test_touching_counts()
    print("ok  test_touching_counts")
