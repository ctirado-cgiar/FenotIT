"""Editor de áreas sobre la imagen de entrada: varias áreas de análisis y zonas
excluidas por foto (rectángulo o polígono), × en cada una para borrarla y una
herramienta de selección (arrastrar un recuadro o clic; Supr borra lo seleccionado).
Las formas se guardan en coordenadas 0-1 (core.roi) y se dibujan con el zoom actual."""
from __future__ import annotations

import tkinter as tk
from typing import Callable

import cv2
import numpy as np

from fenotit.core.roi import polygon

INCLUDE, EXCLUDE, SELECTED = "#0B2E59", "#E67E00", "#FFD400"
MODES = ("include_rect", "include_poly", "exclude_rect", "exclude_poly", "select")


class AreaEditor:

    def __init__(self, canvas: tk.Canvas, on_change: Callable[[list], None] | None = None):
        self.canvas = canvas
        self.on_change = on_change
        self.on_done: Callable | None = None
        self.mode: str | None = None
        self.shapes: list[dict] = []
        self.selected: set[int] = set()
        self._view = (0, 0, 1, 1)          # x, y, ancho, alto de la imagen en el canvas
        self._start = None
        self._points: list[tuple[float, float]] = []
        for seq, fn in (("<ButtonPress-1>", self._press), ("<B1-Motion>", self._drag),
                        ("<ButtonRelease-1>", self._release), ("<Double-Button-1>", self._double),
                        ("<Button-3>", self._undo_point), ("<Motion>", self._hover)):
            canvas.bind(seq, fn, add="+")

    # ── estado ────────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:                 # ZoomController no mueve la imagen si está activo
        return self.mode is not None

    def set_mode(self, mode: str):
        assert mode in MODES
        self._cancel_temp()
        self.mode = mode
        if mode != "select":
            self.selected.clear()
        self.redraw()

    def stop(self):
        self._cancel_temp()
        self.mode = None
        self.selected.clear()
        self.redraw()
        if self.on_done:
            self.on_done()

    def set_shapes(self, shapes: list[dict]):
        self.shapes = [dict(s) for s in shapes or []]
        self.selected.clear()
        self.redraw()

    def delete_selected(self) -> bool:
        if not self.selected:
            return False
        self.shapes = [s for i, s in enumerate(self.shapes) if i not in self.selected]
        self.selected.clear()
        self._changed()
        return True

    def clear(self):
        self.shapes, self.selected = [], set()
        self._changed()

    def _changed(self):
        self.redraw()
        if self.on_change:
            self.on_change([dict(s) for s in self.shapes])

    # ── coordenadas ───────────────────────────────────────────────────────────

    def set_image_offset(self, x, y, w, h):
        self._view = (x, y, max(w, 1), max(h, 1))
        self.redraw()

    def _to_norm(self, cx, cy):
        x, y, w, h = self._view
        return (min(1.0, max(0.0, (cx - x) / w)), min(1.0, max(0.0, (cy - y) / h)))

    def _to_canvas(self, nx, ny):
        x, y, w, h = self._view
        return x + nx * w, y + ny * h

    # ── dibujo ────────────────────────────────────────────────────────────────

    def redraw(self):
        c = self.canvas
        c.delete("area")
        for i, s in enumerate(self.shapes):
            pts = [self._to_canvas(*p) for p in polygon(s)]
            flat = [v for p in pts for v in p]
            sel = i in self.selected
            color = SELECTED if sel else (EXCLUDE if s["kind"] == "exclude" else INCLUDE)
            c.create_polygon(*flat, outline="#FFFFFF", fill="", width=4, tags="area")
            c.create_polygon(*flat, outline=color, width=3 if sel else 2, tags="area",
                             fill=EXCLUDE if s["kind"] == "exclude" else "",
                             stipple="gray25" if s["kind"] == "exclude" else "",
                             dash=(6, 3) if s["kind"] == "exclude" else ())
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            bx, by = max(xs), min(ys)                     # × en la esquina superior derecha
            c.create_oval(bx - 8, by - 8, bx + 8, by + 8, fill="#FFFFFF", outline=color, width=2,
                          tags=("area", "area_del", f"area_del_{i}"))
            c.create_text(bx, by, text="×", fill=color, font=("Segoe UI", 10, "bold"),
                          tags=("area", "area_del", f"area_del_{i}"))
        self._draw_temp()

    def _draw_temp(self, cursor=None):
        c = self.canvas
        c.delete("area_tmp")
        if self._points:
            color = EXCLUDE if self.mode and self.mode.startswith("exclude") else INCLUDE
            pts = [self._to_canvas(*p) for p in self._points] + ([cursor] if cursor else [])
            for x, y in pts[:len(self._points)]:
                c.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="#FFFFFF", tags="area_tmp")
            if len(pts) >= 2:
                c.create_line(*[v for p in pts for v in p], fill=color, width=2, dash=(4, 3), tags="area_tmp")

    def _cancel_temp(self):
        self._start, self._points = None, []
        self.canvas.delete("area_tmp")

    # ── eventos ───────────────────────────────────────────────────────────────

    def _hit_delete(self, event) -> int | None:
        for item in self.canvas.find_overlapping(event.x - 1, event.y - 1, event.x + 1, event.y + 1):
            for tag in self.canvas.gettags(item):
                if tag.startswith("area_del_"):
                    return int(tag.rsplit("_", 1)[1])
        return None

    def _press(self, event):
        hit = self._hit_delete(event)
        if hit is not None and hit < len(self.shapes):
            del self.shapes[hit]
            self.selected.clear()
            self._changed()
            return "break"
        if not self.mode:
            return
        if self.mode.endswith("_poly"):
            p = self._to_norm(event.x, event.y)
            if len(self._points) >= 3:                    # clic sobre el primer punto: cerrar
                fx, fy = self._to_canvas(*self._points[0])
                if abs(fx - event.x) < 8 and abs(fy - event.y) < 8:
                    self._finish_polygon()
                    return
            self._points.append(p)
            self._draw_temp()
        else:
            self._start = (event.x, event.y)

    def _square(self, event, x0, y0):
        x, y = event.x, event.y
        if event.state & 0x0001 and self.mode != "select":          # Shift = cuadrado
            d = min(abs(x - x0), abs(y - y0))
            x, y = x0 + (d if x >= x0 else -d), y0 + (d if y >= y0 else -d)
        return x, y

    def _drag(self, event):
        if not self.mode or self._start is None:
            return
        x0, y0 = self._start
        x, y = self._square(event, x0, y0)
        self.canvas.delete("area_tmp")
        color = SELECTED if self.mode == "select" else (EXCLUDE if self.mode.startswith("exclude") else INCLUDE)
        self.canvas.create_rectangle(x0, y0, x, y, outline=color, dash=(5, 3), width=2, tags="area_tmp")

    def _release(self, event):
        if not self.mode or self._start is None:
            return
        x0, y0 = self._start
        x, y = self._square(event, x0, y0)
        self._start = None
        self.canvas.delete("area_tmp")
        if self.mode == "select":
            self._select(x0, y0, x, y, add=bool(event.state & 0x0004))
            return
        if abs(x - x0) < 4 or abs(y - y0) < 4:
            return
        kind = "exclude" if self.mode.startswith("exclude") else "include"
        a, b = self._to_norm(x0, y0), self._to_norm(x, y)
        if abs(a[0] - b[0]) < 0.003 or abs(a[1] - b[1]) < 0.003:      # dibujado fuera de la foto
            return
        self.shapes.append({"kind": kind, "type": "rect", "points": [list(map(_r, a)), list(map(_r, b))]})
        self._changed()
        self.stop()

    def _double(self, event):
        if self.mode and self.mode.endswith("_poly") and len(self._points) >= 3:
            self._finish_polygon()

    def _undo_point(self, event):
        if self._points:
            self._points.pop()
            self._draw_temp()

    def _hover(self, event):
        if self._points:
            self._draw_temp((event.x, event.y))

    def _finish_polygon(self):
        kind = "exclude" if self.mode.startswith("exclude") else "include"
        pts = [[_r(x), _r(y)] for x, y in self._points]
        self._points = []
        self.shapes.append({"kind": kind, "type": "polygon", "points": pts})
        self._changed()
        self.stop()

    def _select(self, x0, y0, x1, y1, add: bool):
        """Recuadro: las áreas que toca. Clic: el área que contiene el punto (Ctrl suma)."""
        if not add:
            self.selected.clear()
        click = abs(x1 - x0) < 4 and abs(y1 - y0) < 4
        bx0, bx1 = sorted((x0, x1))
        by0, by1 = sorted((y0, y1))
        for i, s in enumerate(self.shapes):
            pts = np.array([self._to_canvas(*p) for p in polygon(s)], np.float32)
            if click:
                if cv2.pointPolygonTest(pts.reshape(-1, 1, 2), (float(x1), float(y1)), False) >= 0:
                    self.selected ^= {i}
            elif not (pts[:, 0].max() < bx0 or pts[:, 0].min() > bx1 or
                      pts[:, 1].max() < by0 or pts[:, 1].min() > by1):
                self.selected.add(i)
        self.redraw()


def _r(v: float) -> float:
    return round(float(v), 5)
