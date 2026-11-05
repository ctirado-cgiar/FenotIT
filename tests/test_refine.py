"""Afinar máscara: cada operación hace lo que dice sobre una máscara conocida."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fenotit.core.segmentation import refine, refine_op  # noqa: E402


def _mask():
    """Un disco grande con un hueco, un punto suelto y un disco pequeño."""
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (80, 100), 50, 255, -1)
    cv2.circle(m, (80, 100), 8, 0, -1)          # hueco
    m[20, 180] = 255                             # punto suelto
    cv2.circle(m, (170, 160), 6, 255, -1)       # disco pequeño
    return m


def test_erode_dilate():
    m = _mask()
    assert (refine_op(m, "erode", 5) > 0).sum() < (m > 0).sum() < (refine_op(m, "dilate", 5) > 0).sum()


def test_open_removes_dot():
    assert refine_op(_mask(), "open", 3)[20, 180] == 0


def test_open_by_reconstruction_keeps_edges():
    m = _mask()
    out = refine_op(m, "open_rec", 15)
    assert out[20, 180] == 0 and out[160, 170] == 0          # lo pequeño desaparece
    big = np.zeros_like(m)
    cv2.circle(big, (80, 100), 50, 255, -1)
    cv2.circle(big, (80, 100), 8, 0, -1)
    assert ((out > 0) == (big > 0)).all()                    # el disco grande queda intacto


def test_fill_holes():
    assert refine_op(_mask(), "fill_holes")[100, 80] == 255


def test_order_and_unknown_ops():
    m = _mask()
    out = refine(m, [{"op": "open", "size": 3}, {"op": "fill_holes"}, {"op": "nope"}])
    assert out[20, 180] == 0 and out[100, 80] == 255


if __name__ == "__main__":
    for fn in (test_erode_dilate, test_open_removes_dot, test_open_by_reconstruction_keeps_edges,
               test_fill_holes, test_order_and_unknown_ops):
        fn()
        print("ok ", fn.__name__)
