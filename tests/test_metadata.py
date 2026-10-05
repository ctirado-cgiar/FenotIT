import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit.core import metadata, stats
from fenotit.core.project import Project

IMAGES = ["001.jpg", "002.jpg", "003.JPG", "004.png"]


def _csv(text, name="meta.csv"):
    d = Path(tempfile.mkdtemp())
    (d / name).write_text(text, encoding="utf-8")
    return d / name


def test_match_numbers_and_names():
    cols, rows = metadata.read_table(_csv("foto;genotipo;rep\n1;G1;1\n2;G1;2\n3;G2;1\n9;G9;1\n"))
    assert cols == ["foto", "genotipo", "rep"]
    assert metadata.guess_key_column(cols, rows, IMAGES) == "foto"
    m = metadata.match(IMAGES, rows, "foto")
    assert m.by_image["003.JPG"]["genotipo"] == "G2"
    assert m.images_without_row == ["004.png"] and m.rows_without_image == ["9"]


def test_group_by_user_column():
    _, rows = metadata.read_table(_csv("imagen,genotipo\n001.jpg,G1\n002.jpg,G1\n003.jpg,G2\n"))
    m = metadata.match(IMAGES[:3], rows, "imagen")
    results = [(n, {"objects": [{"object_id": 1, "area_px2": a}, {"object_id": 2, "area_px2": a + 2}]})
               for n, a in zip(IMAGES[:3], (10, 20, 30))]
    objs = stats.combine(results, meta=m.by_image, key_column="imagen")
    assert objs[0]["genotipo"] == "G1" and "imagen" not in objs[0]
    by_g = {g["group"]: g for g in stats.summarize(objs, by="genotipo", columns=["area_px2"])}
    assert by_g["G1"]["n"] == 4 and by_g["G1"]["area_px2_mean"] == 16 and by_g["G2"]["area_px2_mean"] == 31


def test_project_keeps_metadata():
    d = Path(tempfile.mkdtemp())
    (d / "meta.csv").write_text("foto,genotipo\n1,G1\n", encoding="utf-8")
    p = Project(name="t", metadata={"file": str(d / "meta.csv"), "key_column": "foto"})
    p.save(d)
    q = Project.load(p.file)
    assert Path(q.metadata["file"]) == d / "meta.csv" and q.metadata["key_column"] == "foto"
    assert "meta.csv" == Project.load(p.file).to_dict()["metadata"]["file"]


def test_export_adds_user_columns():
    from types import SimpleNamespace
    from fenotit.core.export import exporter
    res = {"a/001.jpg": SimpleNamespace(status="ok", error="", extra={"tables": {
        "image": [{"n_objects": 2}], "objects": [{"object_id": 1, "status": "ok"}]}})}
    meta = {"a/001.jpg": {"genotipo": "G1", "status": "control"}}
    tables = exporter.collect("Objetos", res, ["a/001.jpg", "a/002.jpg"], meta=meta)
    img = tables["image"][0]
    assert list(img)[:4] == ["Image_ID", "Image_name", "genotipo", "user_status"]
    assert img["status"] == "ok" and img["user_status"] == "control" and img["n_objects"] == 2
    obj = tables["objects"][0]
    assert obj["genotipo"] == "G1" and obj["status"] == "ok" and obj["user_status"] == "control"
    assert "genotipo" not in tables["image"][1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
