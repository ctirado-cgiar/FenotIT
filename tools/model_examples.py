"""Genera las miniaturas que muestran qué da cada tipo de modelo de IA: máscaras (el contorno
exacto de cada objeto) o cajas (un rectángulo por objeto), sobre formas irregulares en gris:
fenotit/assets/models/masks.png y boxes.png.

    python tools/model_examples.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fenotit" / "assets" / "models"
SIZE = (84, 60)                       # tamaño de la miniatura en la ventana (px)
SCALE = 8                             # se dibuja 8 veces más grande y se reduce (bordes suaves)
WIDTH, HEIGHT = SIZE[0] * SCALE, SIZE[1] * SCALE
ACCENT = np.array((172, 102, 33), float)        # azul de FenotIT (#2166AC) en BGR
BACKGROUND = np.array((246, 244, 242), float)
MASK_OPACITY = 0.45
LINE = int(1.2 * SCALE)
SEED = 11


def polygon_mask(points) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.fillPoly(mask, [(np.asarray(points) * SCALE).astype(np.int32)], 255, cv2.LINE_AA)
    return mask


def turned(center_x, center_y, xs, ys, angle):
    cosine, sine = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return np.c_[center_x + cosine * xs - sine * ys, center_y + sine * xs + cosine * ys]


def blob(rng, center_x, center_y, radius_x, radius_y, angle, wobble):
    """Forma irregular: una elipse con ondulaciones de baja frecuencia en el borde."""
    angles = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    radius = np.ones_like(angles)
    for harmonic in range(2, 6):
        radius += wobble * rng.uniform(-1, 1) / harmonic * np.cos(harmonic * angles + rng.uniform(0, 2 * np.pi))
    return polygon_mask(turned(center_x, center_y, radius_x * radius * np.cos(angles),
                               radius_y * radius * np.sin(angles), angle))


def leaf(center_x, center_y, length, width, angle):
    """Hoja: más ancha cerca de la base y en punta en el ápice."""
    steps = np.linspace(0, np.pi, 80)
    xs = -length / 2 * np.cos(steps)
    ys = width / 2 * np.sin(steps) ** 1.4 * (1 - 0.25 * np.cos(steps))
    return polygon_mask(turned(center_x, center_y, np.r_[xs, xs[::-1]], np.r_[ys, -ys[::-1]], angle))


def main():
    rng = np.random.default_rng(SEED)
    objects = [(leaf(17, 16, 22, 10, -25), 128), (blob(rng, 47, 14, 6, 6, 0, 0.12), 150),
               (blob(rng, 70, 19, 9, 6, 35, 0.45), 118), (blob(rng, 22, 44, 9, 4, 15, 0.25), 140),
               (leaf(53, 44, 18, 9, 70), 125), (blob(rng, 75, 47, 5, 5, 0, 0.5), 160)]
    height_share = np.linspace(0, 1, HEIGHT)[:, None, None]
    plain = (BACKGROUND * (1 - 0.05 * height_share)).repeat(WIDTH, 1)
    for mask, tone in objects:
        shadow = cv2.GaussianBlur(np.roll(np.roll(mask, SCALE, 0), SCALE // 2, 1), (0, 0), SCALE) / 255.0
        plain *= 1 - 0.12 * shadow[..., None]
        alpha = (mask / 255.0)[..., None]
        shade = np.repeat(tone + 25 * (1 - height_share), WIDTH, 1)      # un poco más claro arriba
        plain = plain * (1 - alpha) + shade * alpha
    masks, boxes = plain.copy(), plain.clip(0, 255).astype(np.uint8)
    for mask, _ in objects:
        alpha = (mask / 255.0)[..., None]
        masks = masks * (1 - MASK_OPACITY * alpha) + ACCENT * MASK_OPACITY * alpha
    masks = masks.clip(0, 255).astype(np.uint8)
    for mask, _ in objects:
        binary = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(masks, contours, -1, ACCENT.tolist(), LINE, cv2.LINE_AA)
        x, y, width, height = cv2.boundingRect(binary)
        cv2.rectangle(boxes, (x - SCALE, y - SCALE), (x + width + SCALE, y + height + SCALE), ACCENT.tolist(), LINE,
                      cv2.LINE_AA)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, image in (("masks", masks), ("boxes", boxes)):
        cv2.imwrite(str(OUT / f"{name}.png"), cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    main()
