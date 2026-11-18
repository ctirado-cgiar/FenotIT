"""Decoración común: mosaico en escalera (carga) y píxeles con borde que se desvanecen
(inicio, Acerca de), en tonos del azul de la app."""
from __future__ import annotations

import random
import tkinter as tk

# Valores del tema (fenotit/assets/themes): COLORS (colores); los tonos de borde del logo; los tonos
# de relleno del logo.
from fenotit.gui.style.theme import COLORS, LOGO_EDGES as EDGES, LOGO_TINTS as TINTS

BLUE = COLORS["accent"]


def mosaic(canvas: tk.Canvas, x: int, y: int, cell: int, heights: list[int], mirror: bool = False,
           upward: bool = False, seed: int = 7, gap: int = 2):
    """Columnas de cuadritos con alturas `heights` (en celdas), desde (x, y).
    `up`: crecen hacia arriba; `mirror`: las columnas van hacia la izquierda. Los colores
    se aclaran hacia la punta de cada columna (con algo de azar fijo)."""
    rnd = random.Random(seed)
    for column, n_cells in enumerate(heights):
        for row in range(n_cells):
            tint = min(len(TINTS) - 1, max(0, int((row + column * 0.6) / max(n_cells, 1) * 2.2) + rnd.choice((0, 0, 1))))
            cx = x - (column + 1) * cell if mirror else x + column * cell
            cy = y - (row + 1) * cell if upward else y + row * cell
            canvas.create_rectangle(cx, cy, cx + cell - gap, cy + cell - gap, fill=TINTS[tint], width=0)



def pixel_dissolve(canvas: tk.Canvas, width: int, height: int, cell: int, density, seed: int = 5, gap: int = 3):
    """Cuadritos solo con borde, en tonos de azul, que se desvanecen. `density(x, y)`
    (x, y de 0 a 1) = probabilidad de que una celda aparezca; más densa = más oscura."""
    rnd = random.Random(seed)
    for cell_y in range(height // cell + 1):
        for cell_x in range(width // cell + 1):
            chance = density(cell_x * cell / width, cell_y * cell / height)
            if chance > 0 and rnd.random() < chance:
                edge = min(len(EDGES) - 1, max(0, int((1 - chance) * 4.5) + rnd.choice((0, 0, 1))))
                x, y = cell_x * cell + gap, cell_y * cell + gap
                canvas.create_rectangle(x, y, x + cell - 2 * gap, y + cell - 2 * gap, outline=EDGES[edge], width=1)
