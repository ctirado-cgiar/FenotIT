"""Excluir objetos a mano: no se cuentan ni se miden y los demás conservan su número."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit import i18n
from fenotit.core.analysis.registry import ANALYSES
from fenotit.core.image_io import load_image

IMAGE = Path(__file__).parent / "images" / "01.jpg"
PARAMS = {"color_space": "YCrCb", "channel_idx": 1, "min_val": 122, "max_val": 255,
          "measure_shape": True, "measure_color": True}


def test_exclude_keeps_numbers():
    i18n.load("en")
    img = load_image(str(IMAGE))
    h, w = img.shape[:2]
    before = ANALYSES["Objetos"].func(img, PARAMS)
    objs = {r["object_id"]: r for r in before.extra["tables"]["objects"]}
    target = objs[5]
    params = dict(PARAMS, drop_points=[[target["centroid_x_px"] / w, target["centroid_y_px"] / h]])
    after = ANALYSES["Objetos"].func(img, params)
    assert after.extra["excluded"] == [5]
    assert after.stats["n_objects"] == before.stats["n_objects"] - 1
    assert after.stats["n_excluded_manual"] == 1
    assert 5 not in {r["object_id"] for r in after.measurements}
    assert 5 not in {r["object_id"] for r in after.extra["tables"]["object_colors"]}
    assert {r["object_id"]: r["status"] for r in after.extra["tables"]["objects"]}[5] == "excluded"
    for r in after.measurements:                       # el resto: mismo número, mismas medidas
        assert r["area_px2"] == objs[r["object_id"]]["area_px2"]
    dist = ANALYSES["Distancias"].func(img, params)
    assert 5 not in {r["object_a"] for r in dist.extra["tables"]["distances"]} | \
        {r["object_b"] for r in dist.extra["tables"]["distances"]}



def test_highlight_is_only_a_mark():
    i18n.load("en")
    img = load_image(str(IMAGE))
    h, w = img.shape[:2]
    before = ANALYSES["Objetos"].func(img, PARAMS)
    target = {r["object_id"]: r for r in before.extra["tables"]["objects"]}[3]
    after = ANALYSES["Objetos"].func(img, dict(PARAMS, mark_points=[[target["centroid_x_px"] / w,
                                                                     target["centroid_y_px"] / h]]))
    status = {r["object_id"]: r["status"] for r in after.extra["tables"]["objects"]}
    assert status[3] == "highlighted" and after.extra["highlighted"] == [3]
    assert after.stats["n_objects"] == before.stats["n_objects"] and after.stats["n_highlighted"] == 1
    assert len(after.measurements) == len(before.measurements)


if __name__ == "__main__":
    test_exclude_keeps_numbers()
    print("ok  test_exclude_keeps_numbers")
    test_highlight_is_only_a_mark()
    print("ok  test_highlight_is_only_a_mark")
