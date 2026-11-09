"""Genera las miniaturas que muestran qué da cada tipo de modelo de IA (máscaras o cajas),
a partir de un recorte de tests/images/01.jpg: fenotit/assets/models/masks.png y boxes.png.

    python docs/design/model_examples.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fenotit" / "assets" / "models"
SIZE = (84, 60)
COLORS = [(68, 158, 46), (5, 183, 242), (40, 39, 214), (173, 68, 142), (34, 126, 230)]


def main():
    img = cv2.imread(str(ROOT / "tests" / "images" / "01.jpg"))
    h, w = img.shape[:2]
    crop = img[int(h * 0.34):int(h * 0.58), int(w * 0.33):int(w * 0.67)]
    crop = cv2.resize(crop, (SIZE[0] * 4, SIZE[1] * 4), interpolation=cv2.INTER_AREA)
    blue = cv2.GaussianBlur(crop[:, :, 0], (5, 5), 0)
    _, mask = cv2.threshold(blue, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    keep = [i for i in range(1, n) if st[i, 4] > 400]
    base = (crop * 0.85).astype(np.uint8)
    masks, boxes = base.copy(), base.copy()
    for k, i in enumerate(keep):
        c = COLORS[k % len(COLORS)]
        m = lab == i
        masks[m] = (0.45 * crop[m] + 0.55 * np.array(c)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(masks, cnts, -1, (255, 255, 255), 3)
        x, y, bw, bh = st[i, :4]
        cv2.rectangle(boxes, (x - 4, y - 4), (x + bw + 4, y + bh + 4), c, 6)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, im in (("masks", masks), ("boxes", boxes)):
        cv2.imwrite(str(OUT / f"{name}.png"), cv2.resize(im, SIZE, interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    main()
