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


def test_split_side_by_side():
    img = np.full((500, 900, 3), 255, np.uint8)
    for x in (100, 250, 400, 550):
        cv2.ellipse(img, (x, 120), (55, 30), 0, 0, 360, (0, 0, 200), -1)
    cv2.ellipse(img, (300, 330), (55, 30), 0, 0, 360, (0, 0, 200), -1)   # dos pegadas a lo largo
    cv2.ellipse(img, (300, 388), (55, 30), 0, 0, 360, (0, 0, 200), -1)
    ctx = pipeline.run(img, [SEG, {"step": "separate"}, {"step": "filter", "params": {"area_min": 100}},
                             {"step": "count"}])
    assert ctx.image_row()["n_objects"] == 6, ctx.image_row()
    assert ctx.image_row()["n_touching"] == 2


def test_morphometry_skips_touching():
    ctx = _run([{"step": "separate"}, {"step": "filter", "params": {"area_min": 100}}, {"step": "morphometry"}])
    row = ctx.image_row()
    assert (row["n_objects"], row["n_touching"], row["n_measured"]) == (4, 2, 2), row
    assert all(("area_px2" in r) == (not r["touching"]) for r in ctx.tables["objects"])


def test_shape_invariant_to_size_rotation_mirror():
    from fenotit.core.efd import efd, normalize
    from fenotit.core.stats.shape import align
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    r = 1 + 0.15 * np.cos(2 * t) + 0.08 * np.sin(3 * t)          # contorno asimétrico
    base = np.column_stack([r * np.cos(t), r * np.sin(t)])

    def pose(scale, ang, mirror):
        pts = base * [1, -1 if mirror else 1]
        c, s = np.cos(ang), np.sin(ang)
        pts = pts @ [[c, -s], [s, c]] * scale + 500
        return pts[::-1] if mirror else pts
    coeffs = [normalize(efd(pose(*args), 10)) for args in ((80, 0, False), (200, 1.1, False), (50, 2.5, True))]
    a, b, c = align(coeffs)
    assert np.abs(a - b).max() < 0.02 and np.abs(a - c).max() < 0.02


def test_shape_extraction_only_per_object():
    ctx = _run([{"step": "label"}, {"step": "filter", "params": {"area_min": 100}}, {"step": "shape"}])
    rows = ctx.tables["object_shape"]
    assert len(rows) == 3 and "image_shape" not in ctx.tables
    assert all(abs(r["efd_a1"] - 1) < 1e-6 and "efd_d20" in r for r in rows)


def test_stats_by_image_and_by_set():
    from fenotit.core import stats
    from fenotit.core.stats.shape import mean_shapes
    chain = [{"step": "label"}, {"step": "filter", "params": {"area_min": 100}},
             {"step": "morphometry"}, {"step": "shape"}]
    results = [(name, _run(chain).tables) for name in ("a.jpg", "b.jpg")]
    objs = stats.combine(results)
    assert [r["Image_ID"] for r in objs] == [1, 1, 1, 2, 2, 2] and objs[0]["Image_name"] == "a.jpg"
    per_image = stats.summarize(objs, by="Image_ID", columns=["area_px2"])
    whole_set = stats.summarize(objs, columns=["area_px2"])
    assert [g["n"] for g in per_image] == [3, 3] and whole_set[0]["n"] == 6
    assert whole_set[0]["area_px2_mean"] == per_image[0]["area_px2_mean"]
    shapes = stats.combine(results, "object_shape")
    assert len(mean_shapes(shapes, by="Image_ID")) == 2 and mean_shapes(shapes)[0]["n"] == 6


def test_cache_reuses_objects_and_other_measurements():
    full = [{"step": "separate"}, {"step": "filter", "params": {"area_min": 100}},
            {"step": "morphometry"}, {"step": "color", "params": {"n_colors": 2}}, {"step": "count"}]
    cache = pipeline.ChainCache()
    a = cache.run(_scene(), [SEG, *full])
    ref = _run(full)
    assert a.tables["objects"] == ref.tables["objects"] and a.tables["image"] == ref.tables["image"]
    assert a.tables["object_colors"] == ref.tables["object_colors"]
    assert list(a.tables["objects"][0]) == list(ref.tables["objects"][0])
    changed = [*full[:3], {"step": "color", "params": {"n_colors": 3}}, full[4]]
    b = cache.run(_scene(), [SEG, *changed])
    assert cache.hits == ["objects", "morphometry", "count"]
    assert b.tables["objects"] == _run(changed).tables["objects"]
    cache.run(_scene(), [SEG, {"step": "label"}, *changed[1:]])
    assert cache.hits == []


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


def test_color_pooled():
    """Paleta común: todos los objetos usan los mismos colores y sus % suman 100."""
    img = np.zeros((200, 300, 3), np.uint8)
    cv2.circle(img, (70, 100), 40, (0, 0, 200), -1)
    cv2.circle(img, (220, 100), 40, (0, 200, 0), -1)
    cv2.circle(img, (220, 100), 15, (0, 0, 200), -1)
    ctx = pipeline.run(img, [{"step": "threshold", "params": {"color_space": "HSV", "channel": 2, "min_val": 50}},
                             {"step": "label"}, {"step": "color", "params": {"n_colors": 2, "mode": "pooled",
                                                                             "edge_trim": 0}}])
    rows = ctx.tables["object_colors"]
    shared = {(c["R"], c["G"], c["B"]) for c in ctx.tables["image_colors"]}
    assert {(r["R"], r["G"], r["B"]) for r in rows} == shared
    for oid in {r["object_id"] for r in rows}:
        assert abs(sum(r["pct"] for r in rows if r["object_id"] == oid) - 100) < 0.1


def test_distances():
    """Tres círculos de radio 20 en fila, centros a 100 px: borde a borde = 60, centro = 100."""
    img = np.zeros((200, 400, 3), np.uint8)
    for x in (100, 200, 300):
        cv2.circle(img, (x, 100), 20, (255, 255, 255), -1)
    base = [{"step": "threshold", "params": {"color_space": "HSV", "channel": 2, "min_val": 50}}, {"step": "label"}]
    for measure, expected in (("edge_line", 60), ("edge_nearest", 60), ("center", 100)):
        ctx = pipeline.run(img, base + [{"step": "distances", "params": {"k": 1, "measure": measure}}])
        ds = sorted(r["distance_px"] for r in ctx.tables["distances"])
        assert len(ds) == 2 and all(abs(d - expected) <= 2.5 for d in ds), (measure, ds)
    assert ctx.image_row()["n_pairs"] == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
