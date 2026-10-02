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


def dot_grid(canvas: tk.Canvas, w: int, h: int, step: int = 16, color: str = "#DCE6F1", fade_from: float = 0.0):
    """Retícula de puntos (como papel milimetrado o la platina de un escáner).
    `fade_from`: fracción del alto desde la que los puntos empiezan a aparecer."""
    for y in range(step // 2, h, step):
        if y < h * fade_from:
            continue
        for x in range(step // 2, w, step):
            canvas.create_rectangle(x, y, x + 1, y + 1, fill=color, outline=color)


def ruler(canvas: tk.Canvas, x0: int, y: int, length: int, color: str = BLUE, mm: int = 5, label: str = ""):
    """Regla con marcas (una cada `mm` px; larga cada 5 y 10), como la barra de escala."""
    canvas.create_line(x0, y, x0 + length, y, fill=color, width=1)
    for i, x in enumerate(range(x0, x0 + length + 1, mm)):
        tick = 9 if i % 10 == 0 else 6 if i % 5 == 0 else 3
        canvas.create_line(x, y, x, y - tick, fill=color)
    if label:
        canvas.create_text(x0 + length + 6, y, text=label, anchor="sw", fill=color, font=("Segoe UI", 7))


def seed_outlines(canvas: tk.Canvas, x0: int, y0: int, w: int, h: int, n: int = 7, seed: int = 3,
                  color: str = "#A9C8E8", numbers: bool = True, measure: int | None = 3):
    """Contornos de semillas numeradas, como el resultado del análisis de objetos; a una
    se le dibuja su largo (cota) en azul."""
    import math
    rnd = random.Random(seed)
    placed, tries = [], 0
    while len(placed) < n and tries < 500:
        tries += 1
        a = rnd.uniform(16, 23)
        cx, cy = rnd.uniform(x0 + a, x0 + w - a), rnd.uniform(y0 + a, y0 + h - a)
        if all(math.hypot(cx - px, cy - py) > a + pa + 10 for px, py, pa in placed):
            placed.append((cx, cy, a))
    placed.sort(key=lambda p: (round(p[1] / 45), p[0]))
    for i, (cx, cy, a) in enumerate(placed, 1):
        b = a * rnd.uniform(0.62, 0.72)
        rot = rnd.uniform(-0.6, 0.6) + (math.pi / 2 if rnd.random() < 0.3 else 0)
        pts = []
        for k in range(48):
            t = 2 * math.pi * k / 48
            r = 1 + 0.07 * math.cos(t)                       # un extremo algo más ancho (frijol)
            x, y = a * r * math.cos(t), b * (1 + 0.1 * math.cos(t)) * math.sin(t)
            pts += [cx + x * math.cos(rot) - y * math.sin(rot), cy + x * math.sin(rot) + y * math.cos(rot)]
        hit = i == measure
        canvas.create_polygon(*pts, outline=BLUE if hit else color, fill="", width=1.4 if hit else 1.1,
                              smooth=True)
        if hit:                                              # largo: eje mayor, como en Morfometría
            ux, uy = math.cos(rot), math.sin(rot)
            p1 = (cx - a * 1.07 * ux, cy - a * 1.07 * uy)
            p2 = (cx + a * 0.93 * ux, cy + a * 0.93 * uy)
            canvas.create_line(*p1, *p2, fill=BLUE, width=1)
            for px, py in (p1, p2):
                canvas.create_oval(px - 2, py - 2, px + 2, py + 2, fill=BLUE, outline=BLUE)
            nx, ny = (-uy, ux) if -uy >= 0 else (uy, -ux)
            canvas.create_text(cx + nx * (b + 6), cy + ny * (b + 6), text="8.4 mm", anchor="w",
                               fill=BLUE, font=("Segoe UI", 7))
        elif numbers:
            canvas.create_text(cx, cy, text=str(i), fill=color, font=("Segoe UI", 7))
