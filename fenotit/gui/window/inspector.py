"""Inspector: objeto elegido en la foto o en la tabla, excluir y destacar."""
from __future__ import annotations

import tkinter as tk

import cv2
import numpy as np

from fenotit import log
from fenotit.core import units
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
