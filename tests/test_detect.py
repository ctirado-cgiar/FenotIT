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


def test_pt_without_tools(monkeypatch):
    from fenotit.core.segmentation import convert
    monkeypatch.setattr(convert, "tools", lambda: [])
    with pytest.raises(ValueError, match="pt"):
        models.add(str(MODELS / "weights.pt"))


def test_pt_is_converted(monkeypatch, tmp_path):
    import shutil
    from fenotit.core.segmentation import convert
    monkeypatch.setattr(convert, "tools", lambda: ["ultralytics"])

    def fake(pt, out_dir, timeout=0):
        dst = Path(out_dir) / "model.onnx"
        shutil.copy(MODELS / "yolov5_test.onnx", dst)
        return {"onnx": str(dst), "classes": ["seed", "pod"], "input": 640, "tool": "ultralytics"}
    monkeypatch.setattr(convert, "to_onnx", fake)
    pt = tmp_path / "plagas.pt"
    pt.write_bytes(b"")
    m = models.add(str(pt))
    assert m["name"] == "plagas" and m["classes"] == ["seed", "pod"] and m["format"] == "yolo"


def test_manual_class_names(tmp_path):
    import shutil
    src = tmp_path / "sin_nombres.onnx"
    shutil.copy(MODELS / "yolov5_test.onnx", src)           # sin metadatos ni data.yaml al lado
    m = models.add(str(src))
    assert models.generic_names(m) and m["classes"] == ["class_1", "class_2"]
    m = models.update(m["id"], classes=["semilla", "vaina"])
    assert m["classes"] == ["semilla", "vaina"] and not models.generic_names(m)
    with pytest.raises(ValueError):
        models.update(m["id"], classes=["solo una"])
    m2 = models.add(str(src), classes=["a", "b"])
    assert m2["classes"] == ["a", "b"]


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


def test_model_card():
    from fenotit.core.export.exporter import model_rows
    m = _added("yolov8_test")
    assert (m["description"], m["owner"], m["license"]) == ("Test detector for seeds and pods", "FenotIT tests",
                                                            "CC0-1.0")       # leídos del ONNX
    assert m["added"] and m["file"].endswith("yolov8_test.onnx") and models.is_user(m)
    assert not models.missing_info(m)
    m = models.update(m["id"], citation="FenotIT 2026", license="CC BY 4.0")
    rows = dict(model_rows(m["id"]))
    assert rows["model_license"] == "CC BY 4.0" and rows["model_citation"] == "FenotIT 2026"
    assert rows["model_classes"] == "seed; pod"
    sam = models.get("mobilesam")
    if sam:
        assert sam["license"] == "Apache-2.0" and not models.is_user(sam) and not models.missing_info(sam)


def test_card_missing_without_metadata():
    m = _added("yolov5_test")
    assert set(models.missing_info(m)) == {"description", "owner", "license"}



def test_models_are_locked_to_fenotit(tmp_path):
    from fenotit.core.segmentation import vault
    m = _added("yolov8_test")
    stored = Path(m["dir"]) / m["files"][0]
    data = stored.read_bytes()
    assert m["files"] == ["model.fnt"] and vault.is_locked(data)
    assert not list(Path(m["dir"]).glob("*.onnx"))          # sin el .onnx descifrado en el disco
    import onnxruntime as ort
    with pytest.raises(Exception):
        ort.InferenceSession(str(stored))                   # otro programa no lo abre
    assert vault.unlock(data) == (MODELS / "yolov8_test.onnx").read_bytes()
    assert len(detect.run(scene(), m, 0.25)) == 3


def test_share_package(tmp_path):
    m = _added("yolov8_test")
    pkg = models.export_package(m["id"], str(tmp_path / "plagas"))
    assert pkg.endswith(".fenotitmodel")
    m2 = models.add(pkg)
    assert m2["id"] != m["id"] and m2["classes"] == m["classes"] and m2["license"] == m["license"]
    assert len(detect.run(scene(), m2, 0.25)) == 3
    bad = tmp_path / "otro.fenotitmodel"
    bad.write_bytes(b"no es un zip")
    with pytest.raises(ValueError, match="package"):
        models.add(str(bad))
