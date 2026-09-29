import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit.core.corrections.pipeline import Corrections, Perspective, apply, rectify

PX_PER_MM = 5.0
W_MM, H_MM = 200.0, 120.0


def _scene():
    """Plantilla 200x120 mm entre centros de ArUco, un círculo de r=10 mm, vista en perspectiva."""
    img = np.full((1000, 1400, 3), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    ox, oy = 200, 200
    centers = [(ox, oy), (ox + W_MM * PX_PER_MM, oy),
               (ox + W_MM * PX_PER_MM, oy + H_MM * PX_PER_MM), (ox, oy + H_MM * PX_PER_MM)]
    for i, (cx, cy) in enumerate(centers):
        m = cv2.aruco.generateImageMarker(d, i, 80)
        img[int(cy) - 40:int(cy) + 40, int(cx) - 40:int(cx) + 40] = m[..., None]
    cv2.circle(img, (ox + 500, oy + 300), int(10 * PX_PER_MM), (0, 0, 0), -1)
    src = np.float32([[0, 0], [1400, 0], [1400, 1000], [0, 1000]])
    dst = np.float32([[80, 40], [1330, 110], [1250, 990], [150, 900]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (1400, 1000),
                               borderValue=(255, 255, 255))


def _blob(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max((c for c in cnts if cv2.contourArea(c) > 2000), key=lambda c: cv2.contourArea(c) / (cv2.arcLength(c, True) ** 2))
    (_, _), (a, b), _ = cv2.fitEllipse(c)
    return cv2.contourArea(c), max(a, b) / min(a, b)


def test_rectify_with_real_size():
    out, mm_per_px, err = rectify(_scene(), Perspective(True, 20, W_MM, H_MM))
    assert out is not None, err
    area_px, ratio = _blob(out)
    area_mm2 = area_px * mm_per_px ** 2
    assert abs(area_mm2 - np.pi * 100) / (np.pi * 100) < 0.03, area_mm2
    assert ratio < 1.03, ratio


def test_rectify_without_size_keeps_shape():
    out, mm_per_px, err = rectify(_scene(), Perspective(True, 20))
    assert out is not None and mm_per_px is None, err
    assert _blob(out)[1] < 1.05


def test_pipeline_off_is_identity():
    img = _scene()
    out, info = apply(img, Corrections())
    assert out is img and not info.applied and not info.warnings


def test_missing_markers_warns():
    blank = np.full((300, 300, 3), 255, np.uint8)
    out, info = apply(blank, Corrections(perspective=Perspective(True)))
    assert out is blank and info.warnings


def test_roundtrip_dict():
    c = Corrections(perspective=Perspective(True, 10, 200, 120))
    c.distortion.enabled, c.distortion.mtx, c.distortion.dist = True, np.eye(3).tolist(), [[0, 0, 0, 0, 0]]
    assert Corrections.from_dict(c.to_dict()) == c


CC = [(115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177), (103, 189, 170),
      (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46),
      (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31), (187, 86, 149), (8, 133, 161),
      (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52)]


def _card(gain=(1.0, 1.0, 1.0)):
    img = np.full((900, 1200, 3), 200, np.uint8)
    cv2.rectangle(img, (300, 250), (850, 620), (20, 20, 20), -1)
    for i, (r, g, b) in enumerate(CC):
        y, x = divmod(i, 6)
        col = tuple(int(min(255, c * k)) for c, k in zip((b, g, r), gain))
        cv2.rectangle(img, (310 + x * 90, 260 + y * 90), (390 + x * 90, 340 + y * 90), col, -1)
    return img


def _chip_error(a, b):
    return np.mean([np.abs(a[300 + y * 90, 350 + x * 90].astype(float) - b[300 + y * 90, 350 + x * 90]).mean()
                    for y in range(4) for x in range(6)])


def test_color_card_auto_orientation():
    from fenotit.core.corrections.pipeline import ColorCard, plantcv_available
    if not plantcv_available():
        print("skip (PlantCV no instalado)")
        return
    ref, cast = _card(), _card((0.8, 1.0, 1.15))
    for k in (0, 2):
        img = np.ascontiguousarray(np.rot90(cast, k))
        out, info = apply(img, Corrections(color=ColorCard(True, "per_image")))
        assert info.applied == ["color"], info.warnings
        assert _chip_error(np.rot90(out, -k), ref) < 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
