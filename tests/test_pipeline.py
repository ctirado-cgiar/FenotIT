import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit.core import pipeline

SEG = {"step": "threshold", "params": {"color_space": "HSV", "channel": 2, "min_val": 0, "max_val": 250}}


def _scene():
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.circle(img, (150, 150), 60, (0, 0, 200), -1)                   # círculo rojo r=60
    cv2.rectangle(img, (400, 100), (600, 150), (0, 200, 0), -1)        # rectángulo verde 201x51
    cv2.ellipse(img, (300, 420), (70, 40), 0, 0, 360, (200, 0, 0), -1)  # dos elipses azules pegadas
    cv2.ellipse(img, (420, 420), (70, 40), 0, 0, 360, (200, 0, 0), -1)
    cv2.circle(img, (795, 300), 30, (0, 0, 0), -1)                     # objeto en el borde
    return img


def _run(chain, **kw):
    return pipeline.run(_scene(), [SEG, *chain], **kw)


def test_morphometry_known_shapes():
    ctx = _run([{"step": "label"}, {"step": "filter", "params": {"area_min": 100}}, {"step": "morphometry"}],
               mm_per_px=0.1)
    rows = ctx.tables["objects"]
    assert len(rows) == 3, len(rows)                    # borde excluido, elipses pegadas = 1 objeto
    circle = min(rows, key=lambda r: abs(r["centroid_x_px"] - 150))
    assert abs(circle["area_mm2"] - np.pi * 6 ** 2) / (np.pi * 36) < 0.02
    assert circle["circularity"] > 0.9 and circle["convexity"] > 0.98 and circle["solidity"] <= 1
    rect = min(rows, key=lambda r: abs(r["centroid_x_px"] - 500))
    assert abs(rect["length_mm"] - 20.0) < 0.3 and abs(rect["width_mm"] - 5.0) < 0.3
    assert ctx.image_row()["n_objects"] == 3


def test_separate_touching():
    ctx = _run([{"step": "separate"},
                {"step": "filter", "params": {"area_min": 100}}, {"step": "count"}])
    assert ctx.image_row()["n_objects"] == 4


def test_border_kept_when_disabled():
    ctx = _run([{"step": "label"}, {"step": "filter", "params": {"area_min": 100, "exclude_border": False}},
                {"step": "count"}])
    assert ctx.image_row()["n_objects"] == 4


def test_roi_and_exclusion():
    roi = np.zeros((600, 800), np.uint8)
    roi[:300, :] = 255
    excl = [(np.array([[380, 80], [620, 80], [620, 170], [380, 170]]), (255, 255, 255))]
    ctx = _run([{"step": "roi"}, {"step": "label"}, {"step": "filter", "params": {"area_min": 100}},
                {"step": "count"}], roi=roi, exclusions=excl)
    assert ctx.image_row()["n_objects"] == 1          # solo el círculo


def test_color_per_object():
    ctx = _run([{"step": "label"}, {"step": "filter", "params": {"area_min": 100}},
                {"step": "color", "params": {"n_colors": 1}}])
    circle = min(ctx.tables["objects"], key=lambda r: abs(r.get("mean_R", 0) - 200))
    assert circle["mean_R"] == 200 and circle["mean_G"] == 0
    assert {r["object_id"] for r in ctx.tables["object_colors"]} == {1, 2, 3}


def test_otsu():
    ctx = pipeline.run(_scene(), [{"step": "otsu", "params": {"color_space": "BGR", "channel": 1, "invert": True}},
                                  {"step": "label"}, {"step": "filter", "params": {"area_min": 100}},
                                  {"step": "count"}])
    assert ctx.image_row()["n_objects"] >= 2


def test_missing_requirement():
    try:
        pipeline.run(_scene(), [{"step": "morphometry"}])
    except pipeline.PipelineError as e:
        assert "labels" in str(e)
    else:
        raise AssertionError("debió fallar")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
