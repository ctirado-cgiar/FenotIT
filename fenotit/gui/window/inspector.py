"""Inspector: objeto elegido en la foto o en la tabla, excluir y destacar."""
from __future__ import annotations

import tkinter as tk

import cv2
import numpy as np

from fenotit import log
from fenotit.core import units
from fenotit.gui.style.theme import COLORS, FONTS
from fenotit.gui.window.common import _on_base
from fenotit.i18n import t


_log = log.get("gui.window.inspector")


class InspectorMixin:
    def _geometry(self) -> dict:
        return (self.last_result.extra.get("geometry") or {}) if self.last_result else {}

    def _pick_object(self, x: float, y: float):
        """Clic con Inspeccionar: el objeto que contiene el punto (o nada)."""
        for oid, contour in self._geometry().items():
            x0, y0 = contour.min(axis=0)
            x1, y1 = contour.max(axis=0)
            if x0 - 2 <= x <= x1 + 2 and y0 - 2 <= y <= y1 + 2 \
                    and cv2.pointPolygonTest(contour.reshape(-1, 1, 2).astype(np.float32), (float(x), float(y)), True) >= -2:
                self._sel_point = (x, y)
                self._select_object(oid)
                return
        self._clear_selection()

    def _object_at(self, x: float, y: float) -> int | None:
        for oid, contour in self._geometry().items():
            x0, y0 = contour.min(axis=0)
            x1, y1 = contour.max(axis=0)
            if x0 - 2 <= x <= x1 + 2 and y0 - 2 <= y <= y1 + 2 \
                    and cv2.pointPolygonTest(contour.reshape(-1, 1, 2).astype(np.float32), (float(x), float(y)), True) >= -2:
                return oid
        return None

    def _hover_object(self, canvas, point):
        """Inspeccionar en la capa Distancias: al pasar el mouse sobre un objeto se ven las
        líneas a sus vecinos con la distancia (en los dos lados)."""
        result = self.last_result
        view = self.step_names[self.step_idx] if result and self.step_names and 0 <= self.step_idx < len(self.step_names) else None
        kind = (result.extra.get("step_folders") or {}).get(view) if result else None
        oid = self._object_at(*point) if point is not None and kind == "distances" else None
        if oid == getattr(self, "_hover_oid", None):
            return
        self._hover_oid = oid
        for canvas_widget in (self.canvas_left, self.canvas_right):
            canvas_widget.delete("hover")
        if oid is None:
            return
        tables = result.extra.get("tables") or {}
        centers = {obj["object_id"]: (obj.get("centroid_x_px"), obj.get("centroid_y_px")) for obj in tables.get("objects", [])}
        pairs = [pair for pair in tables.get("distances", []) if oid in (pair.get("object_a"), pair.get("object_b"))]
        distance_column = next((column for column in (pairs[0] if pairs else {}) if column.startswith("distance_")), None)
        from fenotit.gui.widgets import inspector as inspector_draw
        for canvas_widget in (self.canvas_left, self.canvas_right):
            if not canvas_widget.winfo_ismapped() or centers.get(oid, (None,))[0] is None:
                continue
            start = self.zoom_ctrl.to_canvas(canvas_widget, *centers[oid])
            for pair in pairs:
                other = pair["object_b"] if pair["object_a"] == oid else pair["object_a"]
                if centers.get(other, (None,))[0] is None or start is None:
                    continue
                end = self.zoom_ctrl.to_canvas(canvas_widget, *centers[other])
                canvas_widget.create_line(*start, *end, fill=COLORS["white"], width=4, tags="hover")
                canvas_widget.create_line(*start, *end, fill=COLORS["accent"], width=2, tags="hover")
                distance = pair.get(distance_column)
                if isinstance(distance, (int, float)):
                    text_x, text_y = (start[0] + 2 * end[0]) / 3, (start[1] + 2 * end[1]) / 3     # cerca del vecino
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        canvas_widget.create_text(text_x + dx, text_y + dy, text=inspector_draw.fmt(float(distance)), fill=COLORS["white"],
                                      font=FONTS["body_bold"], tags="hover")
                    canvas_widget.create_text(text_x, text_y, text=inspector_draw.fmt(float(distance)), fill=COLORS["area_include"], font=FONTS["body_bold"],
                                  tags="hover")
            canvas_widget.create_oval(start[0] - 4, start[1] - 4, start[0] + 4, start[1] + 4, fill=COLORS["highlight"], outline=COLORS["white"], tags="hover")

    def _select_object(self, oid: int, from_table: bool = False):
        geom = self._geometry()
        if oid not in geom:
            self._clear_selection()
            return
        if oid != self._sel and from_table:
            self._sel_point = None
        self._sel = oid
        self.zoom_ctrl.highlight = geom[oid]
        self.zoom_ctrl._redraw()
        if not from_table:
            self._sync_table_selection()
        self._show_inspector()

    def _forget_selection(self):
        """Ningún objeto elegido (solo el estado, sin redibujar)."""
        self._sel, self._sel_point = None, None

    def _clear_selection(self):
        self._forget_selection()
        if self.zoom_ctrl:
            self.zoom_ctrl.highlight = None
            self.zoom_ctrl._redraw()
        if self.inspector.winfo_ismapped():
            self.inspector.pack_forget()
            self.image_list.refit()
        self._sync_table_selection()

    def _refresh_selection(self):
        """Nuevo resultado o capa: el mismo objeto si sigue existiendo (la numeración se
        conserva al excluir); si no, se quita la selección."""
        pick = getattr(self, "_pending_pick", None)
        if pick and pick[0] == self.current_image_path and pick[1] in self._geometry():
            self._pending_pick, self._sel, self._sel_point = None, pick[1], None
        if self._sel is None:
            return
        if self._sel in self._geometry():
            self._select_object(self._sel, from_table=False)
        else:
            self._clear_selection()

    def _show_inspector(self):
        """Solo lo de la capa que se está viendo (ver gui/widgets/inspector.py)."""
        from fenotit.core.pipeline import overlay
        from fenotit.core.pipeline.views import _color_text
        from fenotit.gui.widgets import inspector as inspector_draw
        result, oid = self.last_result, self._sel
        if result is None or oid is None:
            return
        geometry = self._geometry()
        contour = geometry[oid]
        view = self.step_names[self.step_idx] if self.step_names and 0 <= self.step_idx < len(self.step_names) else None
        kind = (result.extra.get("step_folders") or {}).get(view, "")
        overlays = result.extra.get("overlays") or {}
        base = result.extra.get("base_image")
        if base is None and self.scaler_left.has_image:
            base = self.scaler_left.original
        image = base if _on_base(result, view) else (result.step_images.get(view) if result.step_images.get(view) is not None else base)
        tables = result.extra.get("tables") or {}
        obj = next((record for record in tables.get("objects", []) if record.get("object_id") == oid), {})
        excluded, starred = oid in (result.extra.get("excluded") or []), oid in (result.extra.get("highlighted") or [])
        unit = self.project.unit
        size = (max(140, min(260, self._panel_of("inspector").content.winfo_width() - 24)), 140)    # cabe en el panel
        units_per_px = inspector_draw.per_unit(result.stats.get("mm_per_px") or None, unit)        # unidad por px de la foto

        def col(prefix):
            return next((column for column in obj if column.startswith(prefix)), None)

        def value(key):
            number = obj.get(key) if key else None
            return f"{inspector_draw.fmt(float(number))} {units.display(key.rsplit('_', 1)[-1])}" if isinstance(number, (int, float)) else "—"

        rows, crop = [], None
        if kind == "morphometry":
            crop, origin, zoom = inspector_draw.crop_around(image, contour, size, margin=0.5) if image is not None else (None, (0, 0), 1)
            if crop is not None:
                overlay.highlight(crop, contour, origin, zoom)
                (length_start, length_end), (width_start, width_end) = inspector_draw.morph_axes(contour)
                length_column, width_column = col("length_"), col("width_")
                mask, taken = inspector_draw.object_mask(crop.shape, contour, origin, zoom), []
                for (start, end), key in (((length_start, length_end), length_column), ((width_start, width_end), width_column)):
                    measured = obj.get(key)
                    label = inspector_draw.fmt(float(measured)) if isinstance(measured, (int, float)) else ""
                    inspector_draw.cota(crop, (np.asarray(start) - origin) * zoom, (np.asarray(end) - origin) * zoom, label, mask, taken)
                inspector_draw.scale_bar(crop, units_per_px / zoom if units_per_px else None, unit, mask, taken)
            for name, prefix in (("insp.area", "area_"), ("insp.length", "length_"), ("insp.width", "width_"),
                                 ("insp.perimeter", "perimeter_")):
                rows.append((t(name), value(col(prefix))))
        elif kind == "shape":
            shapes = tables.get("object_shape", [])
            ids = [record["object_id"] for record in shapes]
            if oid in ids:
                from fenotit.core.efd import from_row
                from fenotit.core.stats.shape import align
                coeffs = align([from_row(record) for record in shapes])
                mean = np.mean(coeffs, axis=0)
                differences = [float(np.sqrt(((coefficients - mean) ** 2).sum())) for coefficients in coeffs]
                own = differences[ids.index(oid)]
                crop = inspector_draw.shape_view(coeffs[ids.index(oid)], mean, size)
                rows += [(t("insp.shape_diff"), f"{own:.3f}"), (t("insp.shape_typical"), f"{np.median(differences):.3f}")]
            else:
                crop, origin, zoom = inspector_draw.crop_around(image, contour, size) if image is not None else (None, (0, 0), 1)
                if crop is not None:
                    overlay.highlight(crop, contour, origin, zoom)
                rows.append((t("insp.no_shape"), ""))
        elif kind == "color":
            crop, origin, zoom = inspector_draw.crop_around(image, contour, size) if image is not None else (None, (0, 0), 1)
            if crop is not None:
                overlay.highlight(crop, contour, origin, zoom)
            colors = [color for color in tables.get("object_colors", []) if color.get("object_id") == oid and color.get("hex")]
            color_format = self.project.display.get("color_format", "RGB")
            if colors:
                rows.append(("bar", [(color["hex"], float(color.get("pct", 0))) for color in colors]))
                rows += [("color", color["hex"], _color_text(color, color_format), float(color.get("pct", 0))) for color in colors]
        elif kind == "distances":
            pairs = [record for record in tables.get("distances", []) if oid in (record.get("object_a"), record.get("object_b"))]
            distance_column = next((column for column in (pairs[0] if pairs else {}) if column.startswith("distance_")), None)

            def centre(points):
                moments = cv2.moments(points.reshape(-1, 1, 2).astype(np.float32))
                return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]) if moments["m00"] else points.mean(axis=0)
            own = centre(contour)
            neighbors = [(record["object_b"] if record["object_a"] == oid else record["object_a"], record.get(distance_column)) for record in pairs]
            neighbors = [(neighbor, measured) for neighbor, measured in neighbors if neighbor in geometry]
            extra = np.array([centre(geometry[neighbor]) for neighbor, _ in neighbors]) if neighbors else None
            crop, origin, zoom = inspector_draw.crop_around(image, contour, size, extra=extra) if image is not None else (None, (0, 0), 1)
            if crop is not None:                          # foto tenue: resaltan las líneas y los valores
                crop[:] = (crop.astype(np.float32) * 0.35 + 255 * 0.65).astype(np.uint8)
                for neighbor, _ in neighbors:
                    pts = np.round((geometry[neighbor] - origin) * zoom).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(crop, [pts], True, (150, 150, 150), 1, cv2.LINE_AA)
                pts = np.round((contour - origin) * zoom).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(crop, [pts], True, (60, 60, 60), 1, cv2.LINE_AA)
                mask, taken = inspector_draw.object_mask(crop.shape, contour, origin, zoom), []
                start = (own - origin) * zoom
                for neighbor, _ in neighbors:
                    end = (centre(geometry[neighbor]) - origin) * zoom
                    cv2.line(crop, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)),
                             (172, 102, 33), 1, cv2.LINE_AA)
                    cv2.circle(crop, tuple(np.round(end).astype(int)), 2, (172, 102, 33), -1, cv2.LINE_AA)
                for neighbor, measured in neighbors:                         # el valor en la punta, por fuera
                    if isinstance(measured, (int, float)):
                        end = (centre(geometry[neighbor]) - origin) * zoom
                        direction = (end - start) / (np.hypot(*(end - start)) or 1)
                        normal = np.array([-direction[1], direction[0]])
                        inspector_draw.label_at(crop, inspector_draw.fmt(float(measured)), [end + direction * 13, end + direction * 13 + normal * 10,
                                                               end + direction * 13 - normal * 10, end + normal * 11, end - normal * 11],
                                     mask, taken, 0.34, dark=True)
            for name, prefix in (("insp.nearest", "nearest_"), ("insp.mean_neighbor", "mean_neighbor_")):
                rows.append((t(name), value(col(prefix))))
            rows.append((t("insp.n_neighbors"), str(obj.get("n_neighbors", len(neighbors)))))
        else:                                              # conteo, máscara
            crop, origin, zoom = inspector_draw.crop_around(image, contour, size) if image is not None else (None, (0, 0), 1)
            if crop is not None:
                if view in overlays:
                    overlay.draw(crop, overlays[view], self._mark_colors(result), origin, zoom, screen=True)
                overlay.highlight(crop, contour, origin, zoom)
            status = "excluded" if excluded else "highlighted" if starred else "ok"
            rows += [(t("insp.touching"), t("insp.yes") if obj.get("touching") else t("insp.no")),
                     (t("insp.status"), t(f"insp.status.{status}"))]
        self.inspector.show(oid, crop, rows, excluded, starred)
        if not self.inspector.winfo_ismapped():
            same = self._layers.master is self.inspector.master        # con CAPAS: debajo de ella
            self.inspector.pack(side=tk.BOTTOM, fill=tk.X, **({"before": self._layers} if same else {}))
        self.root.update_idletasks()
        self.image_list.refit()
        self.root.after(30, self._keep_layers_visible)

    def _toggle_exclude(self):
        self._toggle_mark("excluded")

    def _toggle_highlight(self):
        self._toggle_mark("highlighted")

    def _toggle_mark(self, kind: str):
        """Excluir / destacar (o quitarlo): se guarda un punto del objeto (0-1) en el proyecto
        y la foto se recalcula al momento. El número del objeto no cambia."""
        result, oid, path = self.last_result, self._sel, self.current_image_path
        if result is None or oid is None or not path:
            return
        contour = self._geometry()[oid].reshape(-1, 1, 2).astype(np.float32)
        base = result.extra.get("base_image")
        width = result.stats.get("image_width_px") or (base.shape[1] if base is not None else 0)
        height = result.stats.get("image_height_px") or (base.shape[0] if base is not None else 0)
        if not width or not height:
            return

        def outside(points):
            return [point for point in points if cv2.pointPolygonTest(contour, (point[0] * width, point[1] * height), True) < -15]
        self._remember(f"undo.{kind}")
        if oid in (result.extra.get(kind) or []):
            self.project.set_marks(kind, path, outside(self.project.marks_for(kind, path)))
        else:
            point = self._sel_point
            if point is None or cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) < 0:
                moments = cv2.moments(contour)
                point = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]) if moments["m00"] else tuple(contour[0, 0])
                if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) < 0:
                    point = tuple(contour[0, 0])
            self.project.set_marks(kind, path, self.project.marks_for(kind, path) + [[point[0] / width, point[1] / height]])
            other = "highlighted" if kind == "excluded" else "excluded"      # una sola marca por objeto
            self.project.set_marks(other, path, outside(self.project.marks_for(other, path)))
        self._run_analysis(all_images=False)

    def _goto_object(self, path: str, oid: int | None = None):
        """Desde un gráfico: abre esa foto y, si es un objeto, lo elige (inspector)."""
        if path not in self.batch_paths:
            return
        if path == self.current_image_path:
            if oid is not None:
                self._select_object(oid, from_table=False)
            return
        self._pending_pick = (path, oid) if oid is not None else None
        self._select_image(self.batch_paths.index(path))

    def _object_thumb(self, path: str, oid: int | None):
        """Imagen para el punto elegido: el objeto recortado si la foto está abierta; si
        no, la foto en miniatura."""
        from fenotit.core.pipeline import overlay
        from fenotit.gui.widgets import inspector as inspector_draw
        result = self.last_result
        if path == self.current_image_path and result is not None:
            base = result.extra.get("base_image")
            if base is None and self.scaler_left.has_image:
                base = self.scaler_left.original
            geometry = self._geometry()
            if base is not None:
                if oid in geometry:
                    crop, origin, zoom = inspector_draw.crop_around(base, geometry[oid], (206, 170))
                    if crop is not None:
                        overlay.highlight(crop, geometry[oid], origin, zoom)
                    return crop
                return cv2.resize(base, None, fx=206 / base.shape[1], fy=206 / base.shape[1],
                                  interpolation=cv2.INTER_AREA)
        try:
            from PIL import Image, ImageOps
            with Image.open(path) as picture:
                picture.draft("RGB", (412, 412))
                picture = ImageOps.exif_transpose(picture).convert("RGB")
                picture.thumbnail((206, 206))
                return np.asarray(picture)[:, :, ::-1].copy()
        except Exception:
            _log.debug("miniatura %s", path, exc_info=True)
            return None
