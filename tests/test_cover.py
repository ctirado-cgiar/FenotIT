"""Clasificar áreas: % y área de cada clase dentro del área de análisis."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fenotit.core.pipeline.base import REGISTRY, Context, discover  # noqa: E402


def _ctx(roi=None, exclusions=None, mm_per_px=None):
    img = np.zeros((100, 200, 3), np.uint8)
    classes = np.zeros((100, 200), np.uint8)
    classes[:, 100:] = 1                     # mitad derecha: clase 1
    classes[:50, 150:] = 2                   # un cuarto de la mitad derecha: clase 2
    return Context(image=img, classes=classes, roi=roi, exclusions=exclusions or [], mm_per_px=mm_per_px)


def _run(ctx):
    discover()
    REGISTRY["cover"].func(ctx, {"names": ["fondo", "sano", "lesión"], "colors": ["#000000", "#00ff00", "#ff0000"]})
    return {r["class"]: r for r in ctx.tables["classes"]}


def test_whole_photo():
    rows = _run(_ctx())
    assert rows["fondo"]["pct"] == 50 and rows["sano"]["pct"] == 37.5 and rows["lesión"]["pct"] == 12.5
    assert rows["sano"]["area_px2"] == 7500


def test_analysis_area_and_exclusion():
    roi = np.zeros((100, 200), np.uint8)
    roi[:, 100:] = 1                         # solo la mitad derecha
    excl = [(np.array([[150, 0], [199, 0], [199, 49], [150, 49]]), (0, 0, 255))]
    rows = _run(_ctx(roi=roi, exclusions=excl))
    assert rows["fondo"]["pct"] == 0 and rows["lesión"]["pct"] == 0 and rows["sano"]["pct"] == 100


def test_scale_and_view():
    ctx = _ctx(mm_per_px=0.5)
    rows = _run(ctx)
    assert rows["sano"]["area_mm2"] == 7500 * 0.25
    assert ctx.images["cover"].shape == (100, 200, 3)
    assert ctx.image_row()["lesion_pct"] == 12.5


if __name__ == "__main__":
    for fn in (test_whole_photo, test_analysis_area_and_exclusion, test_scale_and_view):
        fn()
        print("ok ", fn.__name__)
