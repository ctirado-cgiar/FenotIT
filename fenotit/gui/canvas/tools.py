"""Pincel de la segmentación por ejemplos sobre la Entrada: arrastrar con el botón
izquierdo pinta la clase elegida; con el derecho, borra los trazos que toca. Los trazos se
guardan en coordenadas 0-1 de la foto y el ancho como fracción de su ancho, así siguen en
su lugar con cualquier zoom."""
from __future__ import annotations

import tkinter as tk
from typing import Callable
# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras).
from fenotit.gui.style.theme import COLORS, FONTS


class StrokeEditor:

    def __init__(self, canvas: tk.Canvas, on_change: Callable[[list], None]):
        self.canvas = canvas
        self.on_change = on_change
        self.active = False                 # ZoomController no mueve la imagen con el izquierdo
        self.strokes: list[dict] = []
        self.colors: list[str] = []
        self.current = 1
        self.brush = 12                     # px en pantalla
        self.visible = True
        self._view = (0, 0, 1, 1)
        self._drawing: dict | None = None
        self._erasing = False
        for seq, func in (("<ButtonPress-1>", self._press), ("<B1-Motion>", self._drag),
                        ("<ButtonRelease-1>", self._release), ("<ButtonPress-3>", self._erase_start),
                        ("<B3-Motion>", self._erase), ("<ButtonRelease-3>", self._erase_end),
                        ("<Motion>", self._cursor), ("<Leave>", lambda event: canvas.delete("brush"))):
            canvas.bind(seq, func, add="+")

    # ── estado ────────────────────────────────────────────────────────────────

    def set_strokes(self, strokes: list[dict] | None):
        self.strokes = [dict(stroke) for stroke in strokes or []]
        self.redraw()

    def set_image_offset(self, x, y, width, height):
        self._view = (x, y, max(width, 1), max(height, 1))
        self.redraw()

    def _norm(self, cx, cy):
        x, y, width, height = self._view
        return (min(1.0, max(0.0, (cx - x) / width)), min(1.0, max(0.0, (cy - y) / height)))

    def _canvas(self, norm_x, norm_y):
        x, y, width, height = self._view
        return x + norm_x * width, y + norm_y * height

    # ── dibujo ────────────────────────────────────────────────────────────────

    def redraw(self):
        canvas = self.canvas
        canvas.delete("stroke")
        if not self.visible:
            return
        for stroke in self.strokes + ([self._drawing] if self._drawing else []):
            self._draw(stroke)
        canvas.tag_raise("stroke")

    def _draw(self, stroke: dict):
        pts = [coord for point in stroke["pts"] for coord in self._canvas(*point)]
        width = max(1.0, stroke["w"] * self._view[2])
        color = self.colors[stroke["c"]] if stroke["c"] < len(self.colors) else COLORS["error"]
        if len(pts) == 2:
            x, y = pts
            radius = width / 2
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="", tags="stroke")
        else:
            self.canvas.create_line(*pts, fill=color, width=width, capstyle="round", joinstyle="round",
                                    smooth=False, tags="stroke")

    def _cursor(self, event):
        canvas = self.canvas
        canvas.delete("brush")
        if not self.active:
            return
        radius = self.brush / 2
        color = self.colors[self.current] if self.current < len(self.colors) else COLORS["white"]
        canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, outline=color, width=2, tags="brush")
        canvas.create_oval(event.x - radius - 1, event.y - radius - 1, event.x + radius + 1, event.y + radius + 1, outline=COLORS["white"], width=1, tags="brush")

    # ── pintar ────────────────────────────────────────────────────────────────

    def _press(self, event):
        if not self.active:
            return
        self._drawing = {"c": int(self.current), "w": round(self.brush / self._view[2], 5),
                         "pts": [list(map(lambda coord: round(coord, 5), self._norm(event.x, event.y)))]}
        self.redraw()

    def _drag(self, event):
        if not self.active or self._drawing is None:
            return
        point = [round(coord, 5) for coord in self._norm(event.x, event.y)]
        last = self._drawing["pts"][-1]
        x0, y0 = self._canvas(*last)
        if (event.x - x0) ** 2 + (event.y - y0) ** 2 >= 9:           # un punto cada 3 px de pantalla
            self._drawing["pts"].append(point)
            self._cursor(event)
            self.canvas.delete("stroke_live")
            self.redraw()

    def _release(self, event):
        if self._drawing is None:
            return
        self.strokes.append(self._drawing)
        self._drawing = None
        self.redraw()
        self.on_change([dict(stroke) for stroke in self.strokes])

    # ── borrar con el botón derecho ───────────────────────────────────────────

    def _erase_start(self, event):
        if self.active:
            self._erasing = False
            self._erase(event)

    def _erase(self, event):
        if not self.active:
            return
        self._cursor(event)
        x, y, width, height = self._view
        radius = self.brush / 2
        keep = []
        for stroke in self.strokes:
            screen_width = stroke["w"] * width / 2
            hit = any((cx - event.x) ** 2 + (cy - event.y) ** 2 <= (radius + screen_width) ** 2
                      for cx, cy in (self._canvas(*point) for point in stroke["pts"]))
            if hit:
                self._erasing = True
            else:
                keep.append(stroke)
        if len(keep) != len(self.strokes):
            self.strokes = keep
            self.redraw()

    def _erase_end(self, event):
        if self._erasing:
            self._erasing = False
            self.on_change([dict(stroke) for stroke in self.strokes])


class PointEditor:
    """Clics para el modelo de IA: izquierdo = objeto (verde), derecho = fondo (rojo);
    clic sobre un punto existente lo quita. Coordenadas 0-1 de la foto."""

    RADIUS = 6

    def __init__(self, canvas: tk.Canvas, on_change: Callable[[list], None]):
        self.canvas = canvas
        self.on_change = on_change
        self.active = False
        self.visible = True
        self.points: list[list] = []
        self._view = (0, 0, 1, 1)
        canvas.bind("<ButtonRelease-1>", lambda event: self._click(event, 1), add="+")
        canvas.bind("<ButtonRelease-3>", lambda event: self._click(event, 0), add="+")
        canvas.bind("<ButtonPress-1>", self._press, add="+")
        self._down = None

    def set_points(self, points):
        self.points = [list(point) for point in points or []]
        self.redraw()

    def set_image_offset(self, x, y, width, height):
        self._view = (x, y, max(width, 1), max(height, 1))
        self.redraw()

    def _canvas(self, norm_x, norm_y):
        x, y, width, height = self._view
        return x + norm_x * width, y + norm_y * height

    def redraw(self):
        canvas = self.canvas
        canvas.delete("prompt")
        if not (self.visible and self.active):
            return
        for norm_x, norm_y, pos in self.points:
            x, y = self._canvas(norm_x, norm_y)
            radius = self.RADIUS
            canvas.create_oval(x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1, fill=COLORS["white"], outline="", tags="prompt")
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=COLORS["point_positive"] if pos else COLORS["point_negative"], outline="", tags="prompt")
            canvas.create_text(x, y, text="+" if pos else "−", fill=COLORS["white"], font=FONTS["small_bold"], tags="prompt")

    def _press(self, event):
        self._down = (event.x, event.y)

    def _click(self, event, pos):
        if not self.active:
            return
        if pos and self._down and (event.x - self._down[0]) ** 2 + (event.y - self._down[1]) ** 2 > 16:
            return                                              # fue un arrastre, no un clic
        for i, (norm_x, norm_y, _) in enumerate(self.points):            # clic sobre un punto: se quita
            x, y = self._canvas(norm_x, norm_y)
            if (x - event.x) ** 2 + (y - event.y) ** 2 <= (self.RADIUS + 2) ** 2:
                del self.points[i]
                self.redraw()
                self.on_change([list(point) for point in self.points])
                return
        x, y, width, height = self._view
        norm_x, norm_y = (event.x - x) / width, (event.y - y) / height
        if 0 <= norm_x <= 1 and 0 <= norm_y <= 1:
            self.points.append([round(norm_x, 5), round(norm_y, 5), int(pos)])
            self.redraw()
            self.on_change([list(point) for point in self.points])


class RegionPicker:
    """Clic izquierdo (sin arrastrar) sobre una región del modelo: on_click(x, y) en 0-1."""

    def __init__(self, canvas: tk.Canvas, on_click: Callable[[float, float], None]):
        self.canvas = canvas
        self.on_click = on_click
        self.active = False
        self._view = (0, 0, 1, 1)
        self._down = None
        canvas.bind("<ButtonPress-1>", lambda event: setattr(self, "_down", (event.x, event.y)), add="+")
        canvas.bind("<ButtonRelease-1>", self._release, add="+")

    def set_image_offset(self, x, y, width, height):
        self._view = (x, y, max(width, 1), max(height, 1))

    def redraw(self):
        pass

    def _release(self, event):
        if not self.active or not self._down:
            return
        if (event.x - self._down[0]) ** 2 + (event.y - self._down[1]) ** 2 > 16:
            return                                              # fue un arrastre
        x, y, width, height = self._view
        norm_x, norm_y = (event.x - x) / width, (event.y - y) / height
        if 0 <= norm_x <= 1 and 0 <= norm_y <= 1:
            self.on_click(norm_x, norm_y)


class CanvasTools:
    """Las herramientas que dibujan sobre la Entrada (áreas y pincel): el zoom no mueve la
    imagen con el botón izquierdo mientras una está activa, y todas siguen el zoom."""

    def __init__(self, *tools):
        self.tools = tools

    @property
    def active(self) -> bool:
        return any(t.active for t in self.tools)

    def set_image_offset(self, *offset):
        for t in self.tools:
            t.set_image_offset(*offset)
