"""Decoración común (carga e inicio): mosaico de cuadritos en tonos del azul de la app,
en escalera como la mitad "píxel" del logo."""
from __future__ import annotations

import random
import tkinter as tk

BLUE = "#2166AC"
TINTS = ("#2166AC", "#3B7DC0", "#6A9FD3", "#A9C8E8", "#D6E5F4")


def mosaic(canvas: tk.Canvas, x: int, y: int, cell: int, heights: list[int], mirror: bool = False,
           up: bool = False, seed: int = 7, gap: int = 2):
    """Columnas de cuadritos con alturas `heights` (en celdas), desde (x, y).
    `up`: crecen hacia arriba; `mirror`: las columnas van hacia la izquierda. Los colores
    se aclaran hacia la punta de cada columna (con algo de azar fijo)."""
    rnd = random.Random(seed)
    for c, h in enumerate(heights):
        for r in range(h):
            k = min(len(TINTS) - 1, max(0, int((r + c * 0.6) / max(h, 1) * 2.2) + rnd.choice((0, 0, 1))))
            cx = x - (c + 1) * cell if mirror else x + c * cell
            cy = y - (r + 1) * cell if up else y + r * cell
            canvas.create_rectangle(cx, cy, cx + cell - gap, cy + cell - gap, fill=TINTS[k], width=0)


EDGES = ("#5A93CC", "#7DA8D8", "#9DBEE3", "#B9D1EC", "#CFE0F2")


def pixel_dissolve(canvas: tk.Canvas, w: int, h: int, cell: int, density, seed: int = 5, gap: int = 3):
    """Cuadritos solo con borde, en tonos de azul, que se desvanecen. `density(x, y)`
    (x, y de 0 a 1) = probabilidad de que una celda aparezca; más densa = más oscura."""
    rnd = random.Random(seed)
    for gy in range(h // cell + 1):
        for gx in range(w // cell + 1):
            p = density(gx * cell / w, gy * cell / h)
            if p > 0 and rnd.random() < p:
                k = min(len(EDGES) - 1, max(0, int((1 - p) * 4.5) + rnd.choice((0, 0, 1))))
                x, y = gx * cell + gap, gy * cell + gap
                canvas.create_rectangle(x, y, x + cell - 2 * gap, y + cell - 2 * gap, outline=EDGES[k], width=1)


def dot_grid(canvas: tk.Canvas, w: int, h: int, step: int = 16, color: str = "#DCE6F1", fade_from: float = 0.0):
    """Retícula de puntos (como papel milimetrado o la platina de un escáner).
    `fade_from`: fracción del alto desde la que los puntos empiezan a aparecer."""
    for y in range(step // 2, h, step):
        if y < h * fade_from:
            continue
        for x in range(step // 2, w, step):
            canvas.create_rectangle(x, y, x + 1, y + 1, fill=color, outline=color)


