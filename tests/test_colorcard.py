"""Tarjeta de color propia (sin PlantCV) con una foto real (24ColorCard de CameraTrax)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit.core.corrections import colorcard as CC
from fenotit.core.corrections.pipeline import ColorCard, Corrections, apply

IMG = cv2.imread(str(Path(__file__).parent / "images" / "card_cameratrax.jpg"))


def _rotate(img, ang):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), borderValue=(40, 40, 40))


def test_real_card_any_rotation():
    """Mismo resultado girada (validado contra PlantCV: diferencia máx. 1 nivel)."""
    _, ref = CC.correct(IMG)
    assert ref["found"] >= 20 and ref["mean_error_rgb"] < 9 and not ref["dropped"]
    for ang in (45, 60, 180):
        _, info = CC.correct(_rotate(IMG, ang))
        assert np.abs(info["matrix"] - ref["matrix"]).max() < 0.01, ang


def test_glare_chips_dropped():
    img = IMG.copy()
    img[100:300, 2400:2700] = 255
    _, ref = CC.correct(IMG)
    _, info = CC.correct(img)
    assert info["dropped"] and np.abs(info["matrix"] - ref["matrix"]).max() < 0.15


def test_reference_photo_and_white():
    warm = np.clip(IMG.astype(float) * [0.7, 0.9, 1.0], 0, 255).astype(np.uint8)
    cfg = ColorCard(True, reference="photo")
    cfg.target = CC.measure_reference(IMG, cfg)
    out, _ = apply(warm, Corrections(color=cfg))
    assert np.abs(out.astype(int) - IMG.astype(int)).mean() < 2
    card = CC.detect(IMG)
    x, y = card.centers[0, 0]                     # cuadrito blanco
    r, (h, w) = card.chip * 0.3, IMG.shape[:2]
    wb = ColorCard(True, card="white", region=[(x - r) / w, (y - r) / h, (x + r) / w, (y + r) / h])
    out, info = apply(warm, Corrections(color=wb))
    m = CC.region_mean(out, wb.region)
    assert info.applied == ["color"] and np.ptp(m) < 0.01


def test_custom_card_and_old_project():
    vals = (CC.COLORCHECKER24 * 255).reshape(-1, 3).round().tolist()
    cfg = ColorCard(True, card="custom", rows=4, cols=6, values=vals)
    out, info = apply(IMG, Corrections(color=cfg))
    assert info.applied == ["color"]
    old = Corrections.from_dict({"color": {"enabled": True, "mode": "per_image", "pos": "auto", "radius": 20,
                                           "mask_file": None}})
    assert old.color.card == "colorchecker24" and old.color.enabled


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
