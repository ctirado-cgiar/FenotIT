"""Presets: guardar partes de un proyecto, aplicarlas a otro y saber si se cambió algo."""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit.core import preset
from fenotit.core.export import exporter
from fenotit.core.project import Project, Scale


def _source() -> Project:
    project = Project(name="montaje")
    project.corrections.perspective.enabled = True
    project.corrections.perspective.width_mm = 200.0
    project.corrections.perspective.height_mm = 250.0
    project.corrections.distortion.photos = ["tablero.jpg"]
    project.scale = Scale(0.05, "manual")
    project.unit = "cm"
    project.roi = [{"kind": "include", "type": "rect", "points": [[0.1, 0.1], [0.9, 0.9]], "name": "bandeja"}]
    project.segmentation.color_space = "LAB"
    project.segmentation.channel = 2
    project.segmentation.model = {"name": "mobilesam", "mode": "auto", "grid": 16}
    project.analysis = "objects"
    project.params["objects"] = {"measure_color": True, "n_colors": 4}
    return project


def test_only_the_chosen_parts_are_saved_and_applied():
    data = preset.from_project(_source(), "Bandeja azul", ("segmentation", "analysis"))
    assert preset.parts_of(data) == ("segmentation", "analysis")
    with tempfile.TemporaryDirectory() as folder:
        path = preset.save(data, Path(folder) / "bandeja")
        assert path.suffix == preset.PRESET_EXT
        loaded = preset.load(path)
    target = Project()
    target.scale = Scale(0.2, "manual")
    assert preset.apply(loaded, target) == ("segmentation", "analysis")
    assert target.segmentation.color_space == "LAB" and target.segmentation.channel == 2
    assert target.analysis == "objects" and target.params["objects"]["n_colors"] == 4
    assert target.scale.mm_per_pixel == 0.2 and not target.corrections.perspective.enabled
    assert target.segmentation.model["name"] == "mobilesam"


def test_all_parts_and_modified_flag():
    data = preset.from_project(_source(), "Completo", preset.PARTS)
    assert data["corrections"]["distortion"]["photos"] == []
    target = Project()
    preset.apply(data, target)
    assert target.corrections.perspective.width_mm == 200.0 and target.unit == "cm"
    assert target.scale.mm_per_pixel == 0.05 and target.roi[0]["name"] == "bandeja"
    assert not preset.modified(target)
    assert dict(exporter.preset_rows(target)) == {"preset": "Completo", "preset_modified": "no"}
    target.segmentation.min_val = 40
    assert preset.modified(target)
    assert dict(exporter.preset_rows(target))["preset_modified"] == "yes"
    preset.rebase(target)
    assert not preset.modified(target)


def test_automatic_threshold_range_is_not_a_change():
    source = _source()
    source.segmentation.auto = True
    target = Project()
    preset.apply(preset.from_project(source, "Otsu", ("segmentation",)), target)
    target.segmentation.min_val, target.segmentation.max_val = 130, 255     # rango que Otsu puso en una foto
    assert not preset.modified(target)
    target.segmentation.auto = False
    assert preset.modified(target)


def test_aruco_measures_give_the_scale_of_each_photo():
    data = preset.from_project(_source(), "Correcciones", ("corrections",))
    target = Project()
    preset.apply(data, target)
    assert target.scale.source == "aruco" and target.scale.mm_per_pixel is None


def test_project_keeps_its_preset_and_other_files_are_rejected():
    project = Project()
    preset.apply(preset.from_project(_source(), "Completo", preset.PARTS), project)
    with tempfile.TemporaryDirectory() as folder:
        project.save(Path(folder) / "proyecto")
        again = Project.load(Path(folder) / "proyecto")
        assert again.preset["name"] == "Completo" and not preset.modified(again)
        other = Path(folder) / "otro.yaml"
        other.write_text("images: []\n", encoding="utf-8")
        with pytest.raises(ValueError):
            preset.load(other)
    assert dict(exporter.preset_rows(Project())) == {"preset": "none"}
