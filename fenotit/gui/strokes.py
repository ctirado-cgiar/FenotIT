"""Pincel de la segmentación por ejemplos sobre la Entrada: arrastrar con el botón
izquierdo pinta la clase elegida; con el derecho, borra los trazos que toca. Los trazos se
guardan en coordenadas 0-1 de la foto y el ancho como fracción de su ancho, así siguen en
su lugar con cualquier zoom."""
from __future__ import annotations

import tkinter as tk
from typing import Callable


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
        for seq, fn in (("<ButtonPress-1>", self._press), ("<B1-Motion>", self._drag),
                        ("<ButtonRelease-1>", self._release), ("<ButtonPress-3>", self._erase_start),
                        ("<B3-Motion>", self._erase), ("<ButtonRelease-3>", self._erase_end),
                        ("<Motion>", self._cursor), ("<Leave>", lambda e: canvas.delete("brush"))):
            canvas.bind(seq, fn, add="+")

    # ── estado ────────────────────────────────────────────────────────────────

    def set_strokes(self, strokes: list[dict] | None):
        self.strokes = [dict(s) for s in strokes or []]
        self.redraw()

    def set_image_offset(self, x, y, w, h):
        self._view = (x, y, max(w, 1), max(h, 1))
        self.redraw()

    def _norm(self, cx, cy):
        x, y, w, h = self._view
        return (min(1.0, max(0.0, (cx - x) / w)), min(1.0, max(0.0, (cy - y) / h)))

    def _canvas(self, nx, ny):
        x, y, w, h = self._view
        return x + nx * w, y + ny * h

    # ── dibujo ────────────────────────────────────────────────────────────────

    def redraw(self):
        c = self.canvas
        c.delete("stroke")
        if not self.visible:
            return
        for s in self.strokes + ([self._drawing] if self._drawing else []):
            self._draw(s)
        c.tag_raise("stroke")

    def _draw(self, s: dict):
        pts = [v for p in s["pts"] for v in self._canvas(*p)]
        width = max(1.0, s["w"] * self._view[2])
        color = self.colors[s["c"]] if s["c"] < len(self.colors) else "#FF00FF"
        if len(pts) == 2:
            x, y = pts
            r = width / 2
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="", tags="stroke")
        else:
            self.canvas.create_line(*pts, fill=color, width=width, capstyle="round", joinstyle="round",
                                    smooth=False, tags="stroke")

    def _cursor(self, e):
        c = self.canvas
        c.delete("brush")
        if not self.active:
            return
        r = self.brush / 2
        color = self.colors[self.current] if self.current < len(self.colors) else "#FFFFFF"
        c.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, outline=color, width=2, tags="brush")
        c.create_oval(e.x - r - 1, e.y - r - 1, e.x + r + 1, e.y + r + 1, outline="#FFFFFF", width=1, tags="brush")

    # ── pintar ────────────────────────────────────────────────────────────────

    def _press(self, e):
        if not self.active:
            return
        self._drawing = {"c": int(self.current), "w": round(self.brush / self._view[2], 5),
                         "pts": [list(map(lambda v: round(v, 5), self._norm(e.x, e.y)))]}
        self.redraw()

    def _drag(self, e):
        if not self.active or self._drawing is None:
            return
        p = [round(v, 5) for v in self._norm(e.x, e.y)]
        last = self._drawing["pts"][-1]
        x0, y0 = self._canvas(*last)
        if (e.x - x0) ** 2 + (e.y - y0) ** 2 >= 9:           # un punto cada 3 px de pantalla
            self._drawing["pts"].append(p)
            self._cursor(e)
            self.canvas.delete("stroke_live")
            self.redraw()

    def _release(self, e):
        if self._drawing is None:
            return
        self.strokes.append(self._drawing)
        self._drawing = None
        self.redraw()
        self.on_change([dict(s) for s in self.strokes])

    # ── borrar con el botón derecho ───────────────────────────────────────────

    def _erase_start(self, e):
        if self.active:
            self._erasing = False
            self._erase(e)

    def _erase(self, e):
        if not self.active:
            return
        self._cursor(e)
        x, y, w, h = self._view
        r = self.brush / 2
        keep = []
        for s in self.strokes:
            sw = s["w"] * w / 2
            hit = any((cx - e.x) ** 2 + (cy - e.y) ** 2 <= (r + sw) ** 2
                      for cx, cy in (self._canvas(*p) for p in s["pts"]))
            if hit:
                self._erasing = True
            else:
                keep.append(s)
        if len(keep) != len(self.strokes):
            self.strokes = keep
            self.redraw()

    def _erase_end(self, e):
        if self._erasing:
            self._erasing = False
            self.on_change([dict(s) for s in self.strokes])
