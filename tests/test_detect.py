"""Detectores ONNX (cajas): lectura de los formatos YOLOv8, YOLOv5 y RF-DETR, agregar un
modelo, cadena de Objetos y Distancias con cajas, exclusiones y cajas pasadas a máscaras
con MobileSAM. Los modelos de tests/models tienen salida fija (los genera make_models.py)."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOCALAPPDATA", tempfile.mkdtemp())
os.environ.setdefault("APPDATA", os.environ["LOCALAPPDATA"])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "models"))

pytest.importorskip("onnxruntime")

from make_models import OBJECTS, scene                    # noqa: E402

from fenotit.core.segmentation import detect, models     # noqa: E402

MODELS = ROOT / "tests" / "models"


def _added(name: str) -> dict:
    found = [m for m in models.available() if m["source"].endswith(f"{name}.onnx")]
    return found[0] if found else models.add(str(MODELS / f"{name}.onnx"))


def _seg(model, **kw):
    return {"method": "model", "model": {"name": model["id"], "output": "boxes", "classes": model["classes"],
                                         "conf": 0.25, "colors": ["#2E9E44", "#D62728"], **kw}}


@pytest.mark.parametrize("name,fmt", [("yolov8_test", "yolo"), ("yolov5_test", "yolo"), ("rfdetr_test", "detr")])
def test_formats(name, fmt):
    m = _added(name)
    assert m["format"] == fmt and m["output"] == "boxes"
    assert m["classes"] == ["seed", "pod"]                  # metadatos del ONNX o data.yaml
    dets = detect.run(scene(), m, 0.25)
    got = sorted((round(d["x0"]), round(d["y0"]), round(d["x1"]), round(d["y1"]), d["cls"]) for d in dets)
    assert got == sorted(OBJECTS)                           # sin la repetida (NMS) ni la poco segura


def test_pt_is_rejected():
    with pytest.raises(ValueError, match="pt"):
        models.add(str(MODELS / "weights.pt"))


def test_objects_with_boxes():
    from fenotit.core.analysis import objects
    m = _added("yolov8_test")
    res = objects.run(scene(), {"segmentation": _seg(m), "measure_size": True, "measure_shape": True,
                                "mm_per_pixel": 0.1})
    s = res.stats
    assert s["n_objects"] == 3 and s["n_seed"] == 2 and s["n_pod"] == 1
    rows = {r["object_id"]: r for r in res.extra["tables"]["objects"]}
    first = rows[1]
    assert first["class"] == "seed" and first["box_width_mm"] == 12.0 and first["box_height_mm"] == 8.0
    assert "object_shape" not in res.extra["tables"]        # la forma necesita máscaras
    assert set(res.extra["geometry"]) == {1, 2, 3}


def test_exclude_and_filter_boxes():
    from fenotit.core.analysis import objects
    m = _added("yolov8_test")
    res = objects.run(scene(), {"segmentation": _seg(m), "drop_points": [[160 / 800, 140 / 600]]})
    assert res.stats["n_objects"] == 2 and res.stats["n_excluded_manual"] == 1
    res = objects.run(scene(), {"segmentation": _seg(m), "size_filter": True, "length_max": 110})
    assert res.stats["n_objects"] == 1                      # las de 120 px de largo no pasan


def test_distances_by_class():
    from fenotit.core.analysis import distances
    m = _added("yolov8_test")
    res = distances.run(scene(), {"segmentation": _seg(m), "mm_per_pixel": 0.1})
    pairs = res.extra["tables"]["distances"]
    assert {(p["class_a"], p["class_b"]) for p in pairs} >= {("seed", "seed")}
    assert res.stats["mean_nearest_seed_mm"] == pytest.approx(49.04, abs=0.01)
    assert res.stats["mean_nearest_pod_mm"] is None


def test_boxes_to_masks():
    from fenotit.core.segmentation import sam
    if not sam.available():
        pytest.skip("MobileSAM no está instalado")
    from fenotit.core.analysis import objects
    m = _added("yolov8_test")
    res = objects.run(scene(), {"segmentation": _seg(m, to_masks=True), "measure_size": True,
                                "measure_shape": True})
    rows = res.extra["tables"]["objects"]
    assert len(rows) == 3 and res.stats["n_seed"] == 2
    ellipses = sorted(3.1416 * ((x1 - x0) / 2 - 4) * ((y1 - y0) / 2 - 4) for x0, y0, x1, y1, _ in OBJECTS)
    for area, ellipse in zip(sorted(r["area_px2"] for r in rows), ellipses):
        assert abs(area - ellipse) / ellipse < 0.05          # contorno de la elipse, no la caja
    assert res.extra["tables"].get("object_shape")
