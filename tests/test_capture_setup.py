"""El preset de ejemplo (examples/capture_setup) corrige lente, perspectiva y color en sus
tres fotos y mide lo mismo que en el tablero plano."""
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit.core import batch, preset
from fenotit.core.project import Project

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "capture_setup"
COUNT_TOLERANCE = 0.03       # parte de los objetos
SIZE_TOLERANCE = 0.04        # parte del largo y del ancho medios


def _truth() -> list[dict]:
    with open(EXAMPLE / "truth.csv", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _job(project: Project, path: Path) -> batch.Job:
    seg = project.segmentation
    params = {**project.params[project.analysis], "segmentation": asdict(seg), "color_space": seg.color_space,
              "channel_idx": seg.channel, "min_val": seg.min_val, "max_val": seg.max_val, "auto_threshold": seg.auto,
              "mm_per_pixel": None, "length_unit": project.unit, "roi_shapes": project.roi, "area_unit": "px"}
    return batch.Job(str(path), params, aruco_scale=True, max_mpx=project.max_mpx)


@pytest.mark.parametrize("expected", _truth(), ids=lambda row: row["photo"])
def test_preset_measures_each_photo_like_the_flat_board(expected):
    project = Project()
    parts = preset.apply(preset.load(EXAMPLE / ("capture_setup" + preset.PRESET_EXT)), project)
    assert parts == ("corrections", "areas", "segmentation", "analysis")
    assert project.scale.source == "aruco"
    outcome = batch.process(project.analysis, _job(project, EXAMPLE / expected["photo"]), project.corrections)
    assert outcome.status == "ok", outcome.message
    stats, rows = outcome.result.stats, [row for row in outcome.result.measurements if row.get("length_mm")]
    assert abs(stats["n_objects"] - int(expected["n_objects"])) <= COUNT_TOLERANCE * int(expected["n_objects"])
    for column in ("length_mm", "width_mm"):
        mean = sum(row[column] for row in rows) / len(rows)
        assert mean == pytest.approx(float(expected[column + "_mean"]), rel=SIZE_TOLERANCE), column
    for table in ("object_shape", "object_colors"):
        assert outcome.result.extra["tables"].get(table), table
