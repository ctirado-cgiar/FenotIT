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

# roi da los nombres y números de las áreas, que se escriben sobre cada una.
from fenotit.core import roi
# polygon da los vértices de un área (rectángulo o polígono) para dibujarla.
from fenotit.core.roi import polygon
# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras); SPACE (márgenes).
from fenotit.gui.style.theme import COLORS, FONTS, SPACE

INCLUDE, EXCLUDE, SELECTED = COLORS["area_include"], COLORS["area_exclude"], COLORS["selection"]
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
        for seq, func in (("<ButtonPress-1>", self._press), ("<B1-Motion>", self._drag),
                        ("<ButtonRelease-1>", self._release), ("<Double-Button-1>", self._double),
                        ("<Button-3>", self._undo_point), ("<Motion>", self._hover)):
            canvas.bind(seq, func, add="+")
        canvas.tag_bind("area_del", "<Enter>", lambda event: self._hover_delete(True))
        canvas.tag_bind("area_del", "<Leave>", lambda event: self._hover_delete(False))

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
        self.shapes = [dict(shape) for shape in shapes or []]
        self.selected.clear()
        self.redraw()

    def delete_selected(self) -> bool:
        if not self.selected:
            return False
        self.shapes = [shape for i, shape in enumerate(self.shapes) if i not in self.selected]
        self.selected.clear()
        self._changed()
        return True

    def clear(self):
        self.shapes, self.selected = [], set()
        self._changed()

    def _changed(self):
        self.redraw()
        if self.on_change:
            self.on_change([dict(shape) for shape in self.shapes])

    # ── coordenadas ───────────────────────────────────────────────────────────

    def set_image_offset(self, x, y, width, height):
        self._view = (x, y, max(width, 1), max(height, 1))
        self.redraw()

    def _to_norm(self, cx, cy):
        x, y, width, height = self._view
        return (min(1.0, max(0.0, (cx - x) / width)), min(1.0, max(0.0, (cy - y) / height)))

    def _to_canvas(self, norm_x, norm_y):
        x, y, width, height = self._view
        return x + norm_x * width, y + norm_y * height

    # ── dibujo ────────────────────────────────────────────────────────────────

    def redraw(self):
        canvas = self.canvas
        canvas.delete("area")
        area_names = iter(roi.area_names(self.shapes))
        several_areas = len(roi.analysis_areas(self.shapes)) > 1
        for i, shape in enumerate(self.shapes):
            pts = [self._to_canvas(*point) for point in polygon(shape)]
            if shape["kind"] != "exclude":
                area_name = next(area_names)
                if several_areas or shape.get("name"):                     # nombre del área junto a su vértice de arriba a la izquierda
                    label_x, label_y = min(pts, key=lambda corner: corner[0] + corner[1])
                    label = canvas.create_text(label_x + SPACE["md"], label_y + SPACE["sm"], text=area_name, anchor="nw",
                                          fill=COLORS["on_accent"], font=FONTS["small_bold"],
                                          tags=("area", f"area_name_{i}"))
                    box = canvas.bbox(label)
                    canvas.tag_lower(canvas.create_rectangle(box[0] - SPACE["sm"], box[1] - 1, box[2] + SPACE["sm"], box[3] + 1,
                                                   fill=INCLUDE, outline="", tags=("area", f"area_name_{i}")), label)
            flat = [coord for point in pts for coord in point]
            sel = i in self.selected
            color = SELECTED if sel else (EXCLUDE if shape["kind"] == "exclude" else INCLUDE)
            canvas.create_polygon(*flat, outline=COLORS["white"], fill="", width=4, tags="area")
            canvas.create_polygon(*flat, outline=color, width=3 if sel else 2, tags="area",
                             fill=EXCLUDE if shape["kind"] == "exclude" else "",
                             stipple="gray25" if shape["kind"] == "exclude" else "",
                             dash=(6, 3) if shape["kind"] == "exclude" else ())
            # × sobre el vértice de arriba a la derecha (siempre sobre la forma)
            corner_x, corner_y = max(pts, key=lambda corner: corner[0] - corner[1])
            canvas.create_oval(corner_x - 6, corner_y - 6, corner_x + 6, corner_y + 6, fill=COLORS["white"], outline=COLORS["text_faint"], width=1,
                          tags=("area", "area_del", "area_del_bg", f"area_del_{i}"))
            canvas.create_text(corner_x, corner_y - 1, text="×", fill=COLORS["text_muted"], font=FONTS["small_bold"],
                          tags=("area", "area_del", "area_del_x", f"area_del_{i}"))
        if self.mode == "select":
            for i in sorted(self.selected):
                if i >= len(self.shapes):
                    continue
                for kind, handle_index, (x, y) in self._handles(i):
                    if kind == "v":
                        canvas.create_rectangle(x - 4, y - 4, x + 4, y + 4, fill=COLORS["white"], outline=COLORS["text"], tags="area")
                    else:
                        canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=SELECTED, outline=COLORS["text"], tags="area")
        self._draw_temp()

    def _handles(self, i: int):
        """Vértices ("v", k) y puntos medios de los lados ("m", k) del área i, en el canvas."""
        pts = [self._to_canvas(*point) for point in polygon(self.shapes[i])]
        out = [("v", handle_index, point) for handle_index, point in enumerate(pts)]
        for handle_index, (start, end) in enumerate(zip(pts, pts[1:] + pts[:1])):
            out.append(("m", handle_index, ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)))
        return out

    def _hit_handle(self, x: float, y: float):
        for i in sorted(self.selected):
            if i < len(self.shapes):
                for kind, handle_index, (handle_x, handle_y) in self._handles(i):
                    if abs(handle_x - x) <= 6 and abs(handle_y - y) <= 6:
                        return kind, i, handle_index
        return None

    def _inside(self, i: int, x: float, y: float) -> bool:
        pts = np.array([self._to_canvas(*point) for point in polygon(self.shapes[i])], np.float32)
        return cv2.pointPolygonTest(pts.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0

    def _start_edit(self, event) -> bool:
        """En "seleccionar", sobre un área elegida: empieza a mover un vértice, un lado o todo."""
        if self.mode != "select" or not self.selected:
            return False
        hit = self._hit_handle(event.x, event.y)
        if hit:
            kind, i, handle_index = hit
            shape = self.shapes[i]
            if kind == "m" and shape["type"] == "polygon":          # nuevo vértice en el medio del lado
                shape["points"].insert(handle_index + 1, list(self._to_norm(event.x, event.y)))
                kind, handle_index = "v", handle_index + 1
            self._edit = (kind, i, handle_index, None)
            self._before = [list(point) for point in self.shapes[i]["points"]] if kind != "v" or shape["type"] != "polygon" \
                else None                   # un vértice nuevo ya es un cambio
            return True
        for i in sorted(self.selected):
            if i < len(self.shapes) and self._inside(i, event.x, event.y):
                self._edit = ("move", i, 0, (self._to_norm(event.x, event.y),
                                             [list(point) for point in self.shapes[i]["points"]]))
                self._before = [list(point) for point in self.shapes[i]["points"]]
                return True
        return False

    def _do_edit(self, event):
        kind, i, handle_index, data = self._edit
        shape = self.shapes[i]
        norm_x, norm_y = self._to_norm(event.x, event.y)
        if kind == "move":
            (start_x, start_y), orig = data
            dx, dy = norm_x - start_x, norm_y - start_y
            shape["points"] = [[_r(min(1, max(0, x + dx))), _r(min(1, max(0, y + dy)))] for x, y in orig]
        elif shape["type"] == "polygon":
            shape["points"][handle_index] = [_r(norm_x), _r(norm_y)]
        else:                                               # rectángulo: sigue siendo rectángulo
            (x0, y0), _, (x1, y1), _ = polygon(shape)
            if kind == "v":
                opposite = polygon(shape)[(handle_index + 2) % 4]
                shape["points"] = [[_r(norm_x), _r(norm_y)], [_r(opposite[0]), _r(opposite[1])]]
            else:
                if handle_index == 0:
                    y0 = norm_y
                elif handle_index == 1:
                    x1 = norm_x
                elif handle_index == 2:
                    y1 = norm_y
                else:
                    x0 = norm_x
                shape["points"] = [[_r(x0), _r(y0)], [_r(x1), _r(y1)]]
        self.redraw()

    def _remove_vertex(self, event) -> bool:
        hit = self._hit_handle(event.x, event.y) if self.mode == "select" else None
        if not hit or hit[0] != "v":
            return False
        _, i, handle_index = hit
        shape = self.shapes[i]
        if shape["type"] == "polygon" and len(shape["points"]) > 3:
            del shape["points"][handle_index]
            self._changed()
        return True

    def _hover_delete(self, active: bool):
        """La × se pone roja bajo el mouse (clic = borrar esa área)."""
        canvas = self.canvas
        tags = [tag for tag in canvas.gettags("current") if tag.startswith("area_del_") and tag[9:].isdigit()]
        for tag in tags:
            for item in canvas.find_withtag(tag):
                kind = canvas.gettags(item)
                if "area_del_bg" in kind:
                    canvas.itemconfig(item, fill=COLORS["delete"] if active else COLORS["white"], outline=COLORS["delete"] if active else COLORS["text_faint"])
                elif "area_del_x" in kind:
                    canvas.itemconfig(item, fill=COLORS["white"] if active else COLORS["text_muted"])
        canvas.config(cursor="hand2" if active else "")

    def _draw_temp(self, cursor=None):
        canvas = self.canvas
        canvas.delete("area_tmp")
        if self._points:
            color = EXCLUDE if self.mode and self.mode.startswith("exclude") else INCLUDE
            pts = [self._to_canvas(*point) for point in self._points] + ([cursor] if cursor else [])
            for x, y in pts[:len(self._points)]:
                canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline=COLORS["white"], tags="area_tmp")
            if len(pts) >= 2:
                canvas.create_line(*[coord for point in pts for coord in point], fill=color, width=2, dash=(4, 3), tags="area_tmp")

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
            point = self._to_norm(event.x, event.y)
            if len(self._points) >= 3:                    # clic sobre el primer punto: cerrar
                first_x, first_y = self._to_canvas(*self._points[0])
                if abs(first_x - event.x) < 8 and abs(first_y - event.y) < 8:
                    self._finish_polygon()
                    return
            self._points.append(point)
            self._draw_temp()
        else:
            self._start = (event.x, event.y)

    def _square(self, event, x0, y0):
        x, y = event.x, event.y
        if event.state & 0x0001 and self.mode != "select":          # Shift = cuadrado
            side = min(abs(x - x0), abs(y - y0))
            x, y = x0 + (side if x >= x0 else -side), y0 + (side if y >= y0 else -side)
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
        corner_a, corner_b = self._to_norm(x0, y0), self._to_norm(x, y)
        if abs(corner_a[0] - corner_b[0]) < 0.003 or abs(corner_a[1] - corner_b[1]) < 0.003:      # dibujado fuera de la foto
            return
        self.shapes.append({"kind": kind, "type": "rect", "points": [list(map(_r, corner_a)), list(map(_r, corner_b))]})
        self._changed()
        self.stop()

    def _hit_name(self, event) -> int | None:
        for item in self.canvas.find_overlapping(event.x - 1, event.y - 1, event.x + 1, event.y + 1):
            for tag in self.canvas.gettags(item):
                if tag.startswith("area_name_") and tag[10:].isdigit():
                    return int(tag[10:])
        return None

    def _edit_name(self, index: int):
        """Campo de texto sobre el nombre del área: Enter o salir del campo lo guarda, Esc lo deja igual."""
        left, top, _right, _bottom = self.canvas.bbox(f"area_name_{index}")
        shape = self.shapes[index]
        current = roi.area_names(self.shapes)[roi.analysis_areas(self.shapes).index(shape)]
        entry = tk.Entry(self.canvas, font=FONTS["small_bold"], width=max(8, len(current) + 2),
                         relief="flat", highlightthickness=1, highlightbackground=INCLUDE, highlightcolor=INCLUDE)
        entry.insert(0, current)
        entry.select_range(0, tk.END)
        window = self.canvas.create_window(left, top, window=entry, anchor="nw", tags="area")

        def close(save: bool):
            if not entry.winfo_exists():
                return
            new_name = entry.get().strip()
            self.canvas.delete(window)
            entry.destroy()
            if save and new_name != current:
                if new_name:
                    shape["name"] = new_name
                else:
                    shape.pop("name", None)
                self._changed()

        entry.bind("<Return>", lambda event: close(True))
        entry.bind("<FocusOut>", lambda event: close(True))
        entry.bind("<Escape>", lambda event: close(False))
        entry.focus_set()

    def _double(self, event):
        named = self._hit_name(event)
        if named is not None and named < len(self.shapes) and not self._points:
            self._edit_name(named)
            return "break"
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
        for i, shape in enumerate(self.shapes):
            pts = np.array([self._to_canvas(*point) for point in polygon(shape)], np.float32)
            if click:
                if cv2.pointPolygonTest(pts.reshape(-1, 1, 2), (float(x1), float(y1)), False) >= 0:
                    self.selected ^= {i}
            elif not (pts[:, 0].max() < bx0 or pts[:, 0].min() > bx1 or
                      pts[:, 1].max() < by0 or pts[:, 1].min() > by1):
                self.selected.add(i)
        self.redraw()


def _r(value: float) -> float:
    return round(float(value), 5)
