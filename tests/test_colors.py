"""Puntos de color para el gráfico 3D: niveles, torta y espacios."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit.core.stats import colors

TABLES = {"object_colors": [
    {"object_id": 1, "R": 200, "G": 0, "B": 0, "hex": "#c80000", "pct": 40.0},
    {"object_id": 1, "R": 0, "G": 0, "B": 200, "hex": "#0000c8", "pct": 60.0},
    {"object_id": 2, "R": 0, "G": 150, "B": 0, "hex": "#009600", "pct": 100.0}],
    "image_colors": [{"R": 100, "G": 100, "B": 100, "hex": "#646464", "pct": 100.0}]}


def test_levels_and_pies():
    pts = colors.points([("a/01.jpg", 1, "01.jpg", TABLES)], colors.LEVELS, "Lab")
    by = {lv: [p for p in pts if p.level == lv] for lv in colors.LEVELS}
    assert len(by["object_colors"]) == 3 and len(by["objects"]) == 2 and len(by["images"]) == 1
    pie = next(p for p in by["objects"] if p.object_id == 1)
    assert [pct for _, pct in pie.sectors] == [40.0, 60.0] and pie.label_id == "1-1"
    assert pie.rgb[0] > 0.3 and pie.rgb[2] > 0.3 and pie.rgb[1] < 0.1   # mezcla de rojo y azul


def test_spaces():
    p = colors.to_space(np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]), "Lab")
    assert abs(p[0, 0] - 100) < 0.5 and abs(p[0, 1]) < 0.5 and p[1, 1] > 50
    assert np.allclose(colors.to_space(np.array([[1.0, 0.0, 0.0]]), "HSV")[0], [0, 100, 100], atol=0.5)
    assert np.allclose(colors.to_space(np.array([[0.5, 0.5, 0.5]]), "RGB")[0], [127.5] * 3)


if __name__ == "__main__":
    test_levels_and_pies()
    test_spaces()
    print("ok")
