"""Lote en procesos y exportación única (tablas, metadatos, imágenes de las vistas)."""
import csv
import sys
import tempfile
import threading
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit.core import batch
from fenotit.core.corrections.pipeline import Corrections
from fenotit.core.export import exporter
from fenotit.core.project import Project

IMAGES = sorted(str(p) for p in (Path(__file__).parent / "images").glob("0[1-3].jpg"))
PARAMS = {"color_space": "YCrCb", "channel_idx": 1, "min_val": 122, "max_val": 255, "measure_color": True}


def _run(jobs, task=batch.process):
    out, done = [], threading.Event()
    batch.BatchRunner("Objetos", jobs, Corrections(), lambda i, n, o: out.append(o),
                      lambda c: done.set(), workers=2, task=task).start()
    done.wait(300)
    return {o.path: o for o in out}


def _read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_batch_and_export():
    from fenotit import i18n
    i18n.load("es")                      # los procesos usan el idioma de la app (nombres de vistas)
    outs = _run([batch.Job(p, PARAMS) for p in IMAGES[:2]])
    assert all(o.status == "ok" and not o.result.step_images for o in outs.values())
    assert "Morfometría" in outs[IMAGES[0]].result.extra["views"], outs[IMAGES[0]].result.extra["views"]
    store = {p: o.result for p, o in outs.items()}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "objects"
        tables = exporter.collect("Objetos", store, IMAGES, {p: {"mm_per_px": ""} for p in IMAGES})
        meta = exporter.metadata("Objetos", store, Project())
        exporter.write_tables(folder, tables, meta, excel=True, excel_name="objects.xlsx")
        image = _read(folder / "image.csv")
        assert [r["status"] for r in image] == ["ok", "ok", "not_analyzed"]
        assert [r["Image_ID"] for r in image] == ["1", "2", "3"]
        assert image[0]["n_objects"] == "21"
        objects = _read(folder / "objects.csv")
        assert {r["Image_ID"] for r in objects} == {"1", "2"} and len(objects) == 42
        keys = {r["key"]: r["value"] for r in _read(folder / "metadata.csv")}
        assert keys["analysis"] == "Objetos" and keys["param.min_val"] == "122" and keys["version"]
        assert (folder / "objects.xlsx").exists() and (folder / "object_colors.csv").exists()

        views = Path(tmp) / "views"
        jobs = [batch.Job(p, store[p].extra["params"]) for p in IMAGES[:2]]
        outs = _run(jobs, partial(batch.export_views, str(views), {"legend": True}))
        assert all(o.status == "ok" for o in outs.values())
        saved = sorted(p.relative_to(views).as_posix() for p in views.rglob("*.jpg"))
        assert "count/01.jpg" in saved and "morphometry/02.jpg" in saved, saved


def test_cancel():
    out, done = [], threading.Event()
    r = batch.BatchRunner("Objetos", [batch.Job(p, PARAMS) for p in IMAGES * 3], Corrections(),
                          lambda i, n, o: out.append(o), lambda c: (out.append(c), done.set()), workers=1).start()
    r.cancel()
    assert done.wait(60) and out[-1] is True and len(out) < 10


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
