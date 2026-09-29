import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit.core.project import Project, Scale, Segmentation

IMAGES = Path(__file__).resolve().parent / "images"


def test_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        p = Project(name="Ensayo", mode="batch", analysis="morphometry")
        p.set_images(sorted(IMAGES.glob("*.jpg")), mode="batch")
        p.current_index = 3
        p.segmentation = Segmentation("YCrCb", 1, 122, 255)
        p.scale = Scale(0.042, "manual")
        p.params = {"morphometry": {"area_min": 1000}, "color_kmeans": {"n_colors": 4}}
        p.corrections.perspective.enabled, p.corrections.perspective.width_mm = True, 210.0
        p.corrections.color.enabled, p.corrections.color.mode = True, "fixed"
        p.corrections.color.mask = __import__("numpy").eye(20, dtype="uint8") * 10
        p.save(Path(tmp) / "Ensayo")

        q = Project.load(Path(tmp) / "Ensayo")
        assert q.name == "Ensayo" and q.mode == "batch" and q.analysis == "morphometry"
        assert [x.resolve() for x in q.images] == [x.resolve() for x in p.images]
        assert q.current_index == 3 and q.current_image.name == "04.jpg"
        assert q.segmentation == p.segmentation and q.scale == p.scale
        assert q.params == p.params
        assert q.corrections == p.corrections
        assert q.corrections.color.mask_file == "calibration/colorcard_mask.png"
        assert (q.corrections.color.mask == p.corrections.color.mask).all()
        assert not q.missing_images()
        assert (q.folder / "calibration").is_dir() and (q.folder / "results").is_dir()
        assert q.file.name == "Ensayo.fenotit" and q.file.exists()
        assert Project.load(q.file).name == "Ensayo"


def test_relative_images():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "P"
        (folder / "fotos").mkdir(parents=True)
        img = folder / "fotos" / "a.jpg"
        img.write_bytes(b"x")
        p = Project(folder=folder)
        p.set_images([img])
        assert p.to_dict()["images"] == ["fotos/a.jpg"]
        p.save()
        moved = Path(tmp) / "Movido"
        folder.rename(moved)
        assert Project.load(moved).images[0].exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
