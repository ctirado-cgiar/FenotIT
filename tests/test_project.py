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
        p.corrections.color.place = {"rows": 6, "cols": 4, "centers": [[[1.0, 2.0]] * 4] * 6, "chip": 90.0,
                                     "found": 22, "size": [4010, 3006]}
        p.save(Path(tmp) / "Ensayo")

        q = Project.load(Path(tmp) / "Ensayo")
        assert q.name == "Ensayo" and q.mode == "batch" and q.analysis == "morphometry"
        assert [x.resolve() for x in q.images] == [x.resolve() for x in p.images]
        assert q.current_index == 3 and q.current_image.name == "04.jpg"
        assert q.segmentation == p.segmentation and q.scale == p.scale
        assert q.params == p.params
        assert q.corrections == p.corrections
        assert q.corrections.color.place == p.corrections.color.place
        assert not q.missing_images()
        assert (q.folder / "calibration").is_dir() and (q.folder / "results").is_dir()
        assert q.file.name == "Ensayo.fenotit" and q.file.exists()
        assert Project.load(q.file).name == "Ensayo"


def test_per_image_roi_and_scale():
    """Cada foto puede tener sus áreas y su escala; si no, usa las de todas."""
    rect = {"kind": "include", "type": "rect", "points": [[0.1, 0.1], [0.5, 0.5]]}
    hole = {"kind": "exclude", "type": "polygon", "points": [[0.2, 0.2], [0.3, 0.2], [0.3, 0.3]]}
    with tempfile.TemporaryDirectory() as tmp:
        p = Project(name="E")
        p.set_images(sorted(IMAGES.glob("*.jpg"))[:3])
        a, b, c = p.images
        p.apply_roi_to_all([rect])
        p.set_roi(b, [rect, hole])
        p.set_scale(c, Scale(0.05, "two_points"))
        assert p.roi_for(a) == [rect] and p.roi_for(b) == [rect, hole]
        assert p.others_with_own_roi(a) == 1 and p.others_with_own_roi(b) == 0
        assert p.scale_for(c).mm_per_pixel == 0.05 and p.scale_for(a).source == "none"
        p.save(Path(tmp) / "E")
        q = Project.load(Path(tmp) / "E")
        qa, qb, qc = q.images
        assert q.roi_for(qa) == [rect] and q.roi_for(qb) == [rect, hole]
        assert q.scale_for(qc) == Scale(0.05, "two_points")
        q.apply_roi_to_all([hole])
        assert q.roi_for(qb) == [hole] and not q.image_roi


def test_legacy_roi():
    from fenotit.core.roi import from_legacy, masks
    old = {"inclusion": [{"type": "rect", "points": [[0, 0], [0.5, 0.5]]}],
           "exclusions": [{"type": "exclusion", "points": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]}]}
    shapes = from_legacy(old)
    assert [s["kind"] for s in shapes] == ["include", "exclude"]
    inc, exc = masks(shapes, 100, 100)
    assert inc[10, 10] == 255 and inc[80, 80] == 0 and len(exc) == 1


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
