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
