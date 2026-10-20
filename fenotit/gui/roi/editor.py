"""Editor de áreas sobre la imagen de entrada: varias áreas de análisis y zonas
excluidas por foto (rectángulo o polígono), × en cada una para borrarla y una
herramienta de selección (arrastrar un recuadro o clic; Supr borra lo seleccionado).
Las áreas seleccionadas se editan: arrastrar un vértice (cuadrito) lo mueve; arrastrar el
punto medio de un lado (circulito) agrega un vértice en un polígono o mueve ese lado en un
rectángulo; arrastrar por dentro mueve el área; clic derecho en un vértice lo quita.
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
        self._edit = None                   # arrastre en curso: (tipo, área, índice, datos)
        for seq, fn in (("<ButtonPress-1>", self._press), ("<B1-Motion>", self._drag),
                        ("<ButtonRelease-1>", self._release), ("<Double-Button-1>", self._double),
                        ("<Button-3>", self._undo_point), ("<Motion>", self._hover)):
            canvas.bind(seq, fn, add="+")
        canvas.tag_bind("area_del", "<Enter>", lambda e: self._hover_delete(True))
        canvas.tag_bind("area_del", "<Leave>", lambda e: self._hover_delete(False))

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
            # × sobre el vértice de arriba a la derecha (siempre sobre la forma)
            bx, by = max(pts, key=lambda q: q[0] - q[1])
            c.create_oval(bx - 6, by - 6, bx + 6, by + 6, fill="#FFFFFF", outline="#777777", width=1,
                          tags=("area", "area_del", "area_del_bg", f"area_del_{i}"))
            c.create_text(bx, by - 1, text="×", fill="#555555", font=("Segoe UI", 8, "bold"),
                          tags=("area", "area_del", "area_del_x", f"area_del_{i}"))
        if self.mode == "select":
            for i in sorted(self.selected):
                if i >= len(self.shapes):
                    continue
                for kind, k, (x, y) in self._handles(i):
                    if kind == "v":
                        c.create_rectangle(x - 4, y - 4, x + 4, y + 4, fill="#FFFFFF", outline="#333333", tags="area")
                    else:
                        c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=SELECTED, outline="#333333", tags="area")
        self._draw_temp()

    def _handles(self, i: int):
        """Vértices ("v", k) y puntos medios de los lados ("m", k) del área i, en el canvas."""
        pts = [self._to_canvas(*p) for p in polygon(self.shapes[i])]
        out = [("v", k, p) for k, p in enumerate(pts)]
        for k, (a, b) in enumerate(zip(pts, pts[1:] + pts[:1])):
            out.append(("m", k, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)))
        return out

    def _hit_handle(self, x: float, y: float):
        for i in sorted(self.selected):
            if i < len(self.shapes):
                for kind, k, (hx, hy) in self._handles(i):
                    if abs(hx - x) <= 6 and abs(hy - y) <= 6:
                        return kind, i, k
        return None

    def _inside(self, i: int, x: float, y: float) -> bool:
        pts = np.array([self._to_canvas(*p) for p in polygon(self.shapes[i])], np.float32)
        return cv2.pointPolygonTest(pts.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0

    def _start_edit(self, event) -> bool:
        """En "seleccionar", sobre un área elegida: empieza a mover un vértice, un lado o todo."""
        if self.mode != "select" or not self.selected:
            return False
        hit = self._hit_handle(event.x, event.y)
        if hit:
            kind, i, k = hit
            s = self.shapes[i]
            if kind == "m" and s["type"] == "polygon":          # nuevo vértice en el medio del lado
                s["points"].insert(k + 1, list(self._to_norm(event.x, event.y)))
                kind, k = "v", k + 1
            self._edit = (kind, i, k, None)
            self._before = [list(p) for p in self.shapes[i]["points"]] if kind != "v" or s["type"] != "polygon" \
                else None                   # un vértice nuevo ya es un cambio
            return True
        for i in sorted(self.selected):
            if i < len(self.shapes) and self._inside(i, event.x, event.y):
                self._edit = ("move", i, 0, (self._to_norm(event.x, event.y),
                                             [list(p) for p in self.shapes[i]["points"]]))
                self._before = [list(p) for p in self.shapes[i]["points"]]
                return True
        return False

    def _do_edit(self, event):
        kind, i, k, data = self._edit
        s = self.shapes[i]
        nx, ny = self._to_norm(event.x, event.y)
        if kind == "move":
            (sx, sy), orig = data
            dx, dy = nx - sx, ny - sy
            s["points"] = [[_r(min(1, max(0, x + dx))), _r(min(1, max(0, y + dy)))] for x, y in orig]
        elif s["type"] == "polygon":
            s["points"][k] = [_r(nx), _r(ny)]
        else:                                               # rectángulo: sigue siendo rectángulo
            (x0, y0), _, (x1, y1), _ = polygon(s)
            if kind == "v":
                opposite = polygon(s)[(k + 2) % 4]
                s["points"] = [[_r(nx), _r(ny)], [_r(opposite[0]), _r(opposite[1])]]
            else:
                if k == 0:
                    y0 = ny
                elif k == 1:
                    x1 = nx
                elif k == 2:
                    y1 = ny
                else:
                    x0 = nx
                s["points"] = [[_r(x0), _r(y0)], [_r(x1), _r(y1)]]
        self.redraw()

    def _remove_vertex(self, event) -> bool:
        hit = self._hit_handle(event.x, event.y) if self.mode == "select" else None
        if not hit or hit[0] != "v":
            return False
        _, i, k = hit
        s = self.shapes[i]
        if s["type"] == "polygon" and len(s["points"]) > 3:
            del s["points"][k]
            self._changed()
        return True

    def _hover_delete(self, on: bool):
        """La × se pone roja bajo el mouse (clic = borrar esa área)."""
        c = self.canvas
        tags = [tg for tg in c.gettags("current") if tg.startswith("area_del_") and tg[9:].isdigit()]
        for tg in tags:
            for item in c.find_withtag(tg):
                kind = c.gettags(item)
                if "area_del_bg" in kind:
                    c.itemconfig(item, fill="#D32F2F" if on else "#FFFFFF", outline="#D32F2F" if on else "#777777")
                elif "area_del_x" in kind:
                    c.itemconfig(item, fill="#FFFFFF" if on else "#555555")
        c.config(cursor="hand2" if on else "")

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
                if tag.startswith("area_del_") and tag[9:].isdigit():
                    return int(tag[9:])
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
        if self._start_edit(event):
            return "break"
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
        if self._edit:
            self._do_edit(event)
            return
        if not self.mode or self._start is None:
            return
        x0, y0 = self._start
        x, y = self._square(event, x0, y0)
        self.canvas.delete("area_tmp")
        color = SELECTED if self.mode == "select" else (EXCLUDE if self.mode.startswith("exclude") else INCLUDE)
        self.canvas.create_rectangle(x0, y0, x, y, outline=color, dash=(5, 3), width=2, tags="area_tmp")

    def _release(self, event):
        if self._edit:
            i = self._edit[1]
            self._edit = None
            if getattr(self, "_before", None) != self.shapes[i]["points"]:
                self._changed()             # queda como áreas de esta foto ("Aplicar a todas" la copia)
            else:
                self.redraw()
            return
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
        if self._remove_vertex(event):
            return
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
