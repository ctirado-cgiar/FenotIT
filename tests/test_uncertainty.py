"""El modelo de incertidumbre coincide con la dispersión de discos digitalizados al azar."""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fenotit.core import uncertainty as U


def test_disc_matches_model():
    rng = np.random.default_rng(1)
    for r in (6, 15, 40):
        areas = []
        for _ in range(200):
            m = np.zeros((2 * r + 10,) * 2, np.uint8)
            c = r + 5 + rng.random(2)
            yy, xx = np.mgrid[:m.shape[0], :m.shape[1]]
            areas.append(((xx + .5 - c[0]) ** 2 + (yy + .5 - c[1]) ** 2 <= r * r).sum())
        a, p = math.pi * r * r, 2 * math.pi * r
        sd = np.std(areas) / np.mean(areas)
        pred = U.area_rel_sd(a, p)
        assert 0.6 < sd / pred < 1.3, (r, sd, pred)


def test_planning_inverse():
    a = U.min_area_px(0.01)
    assert abs(U.area_rel_sd(a, U.CIRCLE_K * math.sqrt(a)) - 0.01) < 1e-9
    assert abs(U.combined(0.03, 0.02) - 0.05) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)
