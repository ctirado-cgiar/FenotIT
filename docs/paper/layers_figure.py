"""Ilustración de las capas de un análisis, apiladas como en un SIG: la foto abajo y, encima,
máscara, morfometría, color y distancias. Las capas usan los resultados reales del pipeline
(tests/images/01.jpg); cada una se dibuja sobre fondo transparente salvo la foto."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fenotit.core.analysis import distances, objects
from fenotit.core.pipeline import overlay

OUT = ROOT / "docs" / "img" / "layers.png"
SEG = {"method": "threshold", "color_space": "LAB", "channel": 2, "auto": True, "refine": [{"op": "fill_holes"}]}


def layers():
    img = cv2.imread(str(ROOT / "tests" / "images" / "01.jpg"))
    h, w = img.shape[:2]
    img = img[int(h * .18):int(h * .82), int(w * .1):int(w * .9)]
    p = {"segmentation": SEG, "measure_size": True, "measure_color": True, "n_colors": 3}
    res = objects.run(img, p)
    dres = distances.run(img, {**p, "neighbors": "knn", "k": 3})
    colors = overlay.resolve(None, auto=res.extra.get("contrast"))
    labels_mask = None
    out = [("Image", img, None)]
    mask = None
    for k, v in res.step_images.items():
        if res.extra["step_folders"][k] == "mask":
            mask = v[..., 0] > 0
    rgba = lambda bgr, a: np.dstack([bgr, (a * 255).astype(np.uint8)])
    m = np.zeros_like(img); m[mask] = (60, 200, 255)
    out.append(("Mask", m, mask.astype(float) * 0.75))
    for name, folder, ov_src in (("Morphometry", "morphometry", res), ("Color", "color", res),
                                 ("Distances", None, dres)):
        canvas = np.zeros_like(img)
        if folder == "color":
            view = next(v for k, v in res.step_images.items() if res.extra["step_folders"][k] == "color")
            canvas[mask] = view[mask]
            alpha = mask.astype(float)
        else:
            key = next(k for k in ov_src.extra["overlays"]
                       if (folder and ov_src.extra["step_folders"].get(k) == folder) or (not folder))
            if not folder:
                key = next(k for k, f in ov_src.extra["step_folders"].items() if f == "distances")
            ov = dict(ov_src.extra["overlays"][key]); ov["labels"] = []
            overlay.draw(canvas, ov, colors)
            alpha = (canvas.max(axis=2) > 0).astype(float)
            alpha = cv2.dilate(alpha, np.ones((3, 3)))
        out.append((name, canvas, alpha))
    return out


def stack(layer_list, width=900, gap=190):
    """Cada capa como un plano inclinado (perspectiva simple), de abajo hacia arriba."""
    ih, iw = layer_list[0][1].shape[:2]
    k = width / iw
    lw, lh = width, int(ih * k * 0.42)                 # el plano se ve achatado (inclinado)
    skew = int(lw * 0.28)
    n = len(layer_list)
    H = lh + gap * (n - 1) + 80
    W = lw + skew + 360
    canvas = np.full((H, W, 3), 255, np.uint8)
    for i, (name, bgr, alpha) in enumerate(layer_list):
        y0 = H - 40 - lh - gap * i
        src = np.float32([[0, 0], [iw, 0], [iw, ih], [0, ih]])
        dst = np.float32([[40 + skew, y0], [40 + skew + lw, y0], [40 + lw, y0 + lh], [40, y0 + lh]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_AREA)
        a = np.ones(bgr.shape[:2], np.float32) if alpha is None else alpha.astype(np.float32)
        aw = cv2.warpPerspective(a, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
        plane = cv2.warpPerspective(np.ones(bgr.shape[:2], np.float32), M, (W, H))[..., None]
        if alpha is not None:                          # plano de vidrio tenue
            canvas[:] = (canvas * (1 - 0.6 * plane) + np.array([250, 246, 240]) * 0.6 * plane).astype(np.uint8)
        canvas[:] = (canvas * (1 - aw) + warped * aw).astype(np.uint8)
        poly = dst.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], True, (172, 102, 33), 2, cv2.LINE_AA)
        ty = int(y0 + lh / 2)
        cv2.line(canvas, (40 + lw + skew // 2 + 10, ty), (W - 330, ty), (172, 102, 33), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{i + 1}. {name}", (W - 320, ty + 8), cv2.FONT_HERSHEY_DUPLEX,
                    0.9, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), stack(layers()))
    print(OUT)
