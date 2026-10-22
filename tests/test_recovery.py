"""El autoguardado guarda una copia aparte y se recupera con las mismas fotos y ajustes."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["LOCALAPPDATA"] = tmp
        from fenotit.core import recovery
        from fenotit.core.project import Project
        img = Path(__file__).parent / "images" / "01.jpg"
        proj = Project(name="demo", folder=Path(tmp) / "demo")
        proj.save()
        proj.images = [img]
        proj.roi = [{"kind": "include", "type": "rect", "points": [[0.1, 0.1], [0.5, 0.5]]}]
        f = recovery.write(proj, "s1")
        assert proj.file.read_text(encoding="utf-8").count("01.jpg") == 0     # el proyecto no se tocó
        items = recovery.pending()
        assert len(items) == 1 and items[0]["n_images"] == 1
        back = recovery.load(items[0])
        assert back.folder == proj.folder and back.images == [img] and back.roi == proj.roi
        recovery.discard(f)
        assert recovery.pending() == []


if __name__ == "__main__":
    test_roundtrip(); print("ok  test_roundtrip")
