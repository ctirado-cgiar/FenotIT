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
        for oid, c in self._geometry().items():
            x0, y0 = c.min(axis=0)
            x1, y1 = c.max(axis=0)
            if x0 - 2 <= x <= x1 + 2 and y0 - 2 <= y <= y1 + 2 \
                    and cv2.pointPolygonTest(c.reshape(-1, 1, 2).astype(np.float32), (float(x), float(y)), True) >= -2:
                self._sel_point = (x, y)
                self._select_object(oid)
                return
        self._clear_selection()

    def _object_at(self, x: float, y: float) -> int | None:
        for oid, c in self._geometry().items():
            x0, y0 = c.min(axis=0)
            x1, y1 = c.max(axis=0)
            if x0 - 2 <= x <= x1 + 2 and y0 - 2 <= y <= y1 + 2 \
                    and cv2.pointPolygonTest(c.reshape(-1, 1, 2).astype(np.float32), (float(x), float(y)), True) >= -2:
                return oid
        return None

    def _hover_object(self, canvas, p):
        """Inspeccionar en la capa Distancias: al pasar el mouse sobre un objeto se ven las
        líneas a sus vecinos con la distancia (en los dos lados)."""
        r = self.last_result
        view = self.step_names[self.step_idx] if r and self.step_names and 0 <= self.step_idx < len(self.step_names) else None
        kind = (r.extra.get("step_folders") or {}).get(view) if r else None
        oid = self._object_at(*p) if p is not None and kind == "distances" else None
        if oid == getattr(self, "_hover_oid", None):
            return
        self._hover_oid = oid
        for c in (self.canvas_left, self.canvas_right):
            c.delete("hover")
        if oid is None:
            return
        tables = r.extra.get("tables") or {}
        cents = {o["object_id"]: (o.get("centroid_x_px"), o.get("centroid_y_px")) for o in tables.get("objects", [])}
        pairs = [x for x in tables.get("distances", []) if oid in (x.get("object_a"), x.get("object_b"))]
        dcol = next((k for k in (pairs[0] if pairs else {}) if k.startswith("distance_")), None)
        from fenotit.gui.widgets import inspector as ins
        for c in (self.canvas_left, self.canvas_right):
            if not c.winfo_ismapped() or cents.get(oid, (None,))[0] is None:
                continue
            a = self.zoom_ctrl.to_canvas(c, *cents[oid])
            for x in pairs:
                other = x["object_b"] if x["object_a"] == oid else x["object_a"]
                if cents.get(other, (None,))[0] is None or a is None:
                    continue
                b = self.zoom_ctrl.to_canvas(c, *cents[other])
                c.create_line(*a, *b, fill=COLORS["white"], width=4, tags="hover")
                c.create_line(*a, *b, fill=COLORS["accent"], width=2, tags="hover")
                v = x.get(dcol)
                if isinstance(v, (int, float)):
                    tx, ty = (a[0] + 2 * b[0]) / 3, (a[1] + 2 * b[1]) / 3     # cerca del vecino
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        c.create_text(tx + dx, ty + dy, text=ins.fmt(float(v)), fill=COLORS["white"],
                                      font=FONTS["body_bold"], tags="hover")
                    c.create_text(tx, ty, text=ins.fmt(float(v)), fill=COLORS["area_include"], font=FONTS["body_bold"],
                                  tags="hover")
            c.create_oval(a[0] - 4, a[1] - 4, a[0] + 4, a[1] + 4, fill=COLORS["highlight"], outline=COLORS["white"], tags="hover")

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

    def _clear_selection(self):
        self._sel, self._sel_point = None, None
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
        from fenotit.gui.widgets import inspector as ins
        r, oid = self.last_result, self._sel
        if r is None or oid is None:
            return
        geom = self._geometry()
        contour = geom[oid]
        view = self.step_names[self.step_idx] if self.step_names and 0 <= self.step_idx < len(self.step_names) else None
        kind = (r.extra.get("step_folders") or {}).get(view, "")
        ovs = r.extra.get("overlays") or {}
        base = r.extra.get("base_image")
        if base is None and self.scaler_left.has_image:
            base = self.scaler_left.original
        img = base if _on_base(r, view) else (r.step_images.get(view) if r.step_images.get(view) is not None else base)
        tables = r.extra.get("tables") or {}
        obj = next((x for x in tables.get("objects", []) if x.get("object_id") == oid), {})
        excluded, starred = oid in (r.extra.get("excluded") or []), oid in (r.extra.get("highlighted") or [])
        unit = self.project.unit
        size = (max(140, min(260, self._panel_of("inspector").content.winfo_width() - 24)), 140)    # cabe en el panel
        upx = ins.per_unit(r.stats.get("mm_per_px") or None, unit)        # unidad por px de la foto

        def col(prefix):
            return next((k for k in obj if k.startswith(prefix)), None)

        def value(key):
            v = obj.get(key) if key else None
            return f"{ins.fmt(float(v))} {units.display(key.rsplit('_', 1)[-1])}" if isinstance(v, (int, float)) else "—"

        rows, crop = [], None
        if kind == "morphometry":
            crop, origin, k = ins.crop_around(img, contour, size, margin=0.5) if img is not None else (None, (0, 0), 1)
            if crop is not None:
                overlay.highlight(crop, contour, origin, k)
                (l1, l2), (w1, w2) = ins.morph_axes(contour)
                lc, wc = col("length_"), col("width_")
                mask, taken = ins.object_mask(crop.shape, contour, origin, k), []
                for (p1, p2), key in (((l1, l2), lc), ((w1, w2), wc)):
                    v = obj.get(key)
                    label = ins.fmt(float(v)) if isinstance(v, (int, float)) else ""
                    ins.cota(crop, (np.asarray(p1) - origin) * k, (np.asarray(p2) - origin) * k, label, mask, taken)
                ins.scale_bar(crop, upx / k if upx else None, unit, mask, taken)
            for name, prefix in (("insp.area", "area_"), ("insp.length", "length_"), ("insp.width", "width_"),
                                 ("insp.perimeter", "perimeter_")):
                rows.append((t(name), value(col(prefix))))
        elif kind == "shape":
            shapes = tables.get("object_shape", [])
            ids = [x["object_id"] for x in shapes]
            if oid in ids:
                from fenotit.core.efd import from_row
                from fenotit.core.stats.shape import align
                coeffs = align([from_row(x) for x in shapes])
                mean = np.mean(coeffs, axis=0)
                d = [float(np.sqrt(((c - mean) ** 2).sum())) for c in coeffs]
                me = d[ids.index(oid)]
                crop = ins.shape_view(coeffs[ids.index(oid)], mean, size)
                rows += [(t("insp.shape_diff"), f"{me:.3f}"), (t("insp.shape_typical"), f"{np.median(d):.3f}")]
            else:
                crop, origin, k = ins.crop_around(img, contour, size) if img is not None else (None, (0, 0), 1)
                if crop is not None:
                    overlay.highlight(crop, contour, origin, k)
                rows.append((t("insp.no_shape"), ""))
        elif kind == "color":
            crop, origin, k = ins.crop_around(img, contour, size) if img is not None else (None, (0, 0), 1)
            if crop is not None:
                overlay.highlight(crop, contour, origin, k)
            cs = [c for c in tables.get("object_colors", []) if c.get("object_id") == oid and c.get("hex")]
            fmt_ = self.project.display.get("color_format", "RGB")
            if cs:
                rows.append(("bar", [(c["hex"], float(c.get("pct", 0))) for c in cs]))
                rows += [("color", c["hex"], _color_text(c, fmt_), float(c.get("pct", 0))) for c in cs]
        elif kind == "distances":
            pairs = [x for x in tables.get("distances", []) if oid in (x.get("object_a"), x.get("object_b"))]
            dcol = next((k for k in (pairs[0] if pairs else {}) if k.startswith("distance_")), None)

            def centre(c):
                m = cv2.moments(c.reshape(-1, 1, 2).astype(np.float32))
                return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) if m["m00"] else c.mean(axis=0)
            me = centre(contour)
            neigh = [(x["object_b"] if x["object_a"] == oid else x["object_a"], x.get(dcol)) for x in pairs]
            neigh = [(n, v) for n, v in neigh if n in geom]
            extra = np.array([centre(geom[n]) for n, _ in neigh]) if neigh else None
            crop, origin, k = ins.crop_around(img, contour, size, extra=extra) if img is not None else (None, (0, 0), 1)
            if crop is not None:                          # foto tenue: resaltan las líneas y los valores
                crop[:] = (crop.astype(np.float32) * 0.35 + 255 * 0.65).astype(np.uint8)
                for n, _ in neigh:
                    pts = np.round((geom[n] - origin) * k).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(crop, [pts], True, (150, 150, 150), 1, cv2.LINE_AA)
                pts = np.round((contour - origin) * k).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(crop, [pts], True, (60, 60, 60), 1, cv2.LINE_AA)
                mask, taken = ins.object_mask(crop.shape, contour, origin, k), []
                p1 = (me - origin) * k
                for n, _ in neigh:
                    p2 = (centre(geom[n]) - origin) * k
                    cv2.line(crop, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)),
                             (172, 102, 33), 1, cv2.LINE_AA)
                    cv2.circle(crop, tuple(np.round(p2).astype(int)), 2, (172, 102, 33), -1, cv2.LINE_AA)
                for n, v in neigh:                         # el valor en la punta, por fuera
                    if isinstance(v, (int, float)):
                        p2 = (centre(geom[n]) - origin) * k
                        u = (p2 - p1) / (np.hypot(*(p2 - p1)) or 1)
                        nn = np.array([-u[1], u[0]])
                        ins.label_at(crop, ins.fmt(float(v)), [p2 + u * 13, p2 + u * 13 + nn * 10,
                                                               p2 + u * 13 - nn * 10, p2 + nn * 11, p2 - nn * 11],
                                     mask, taken, 0.34, dark=True)
            for name, prefix in (("insp.nearest", "nearest_"), ("insp.mean_neighbor", "mean_neighbor_")):
                rows.append((t(name), value(col(prefix))))
            rows.append((t("insp.n_neighbors"), str(obj.get("n_neighbors", len(neigh)))))
        else:                                              # conteo, máscara
            crop, origin, k = ins.crop_around(img, contour, size) if img is not None else (None, (0, 0), 1)
            if crop is not None:
                if view in ovs:
                    overlay.draw(crop, ovs[view], self._mark_colors(r), origin, k, screen=True)
                overlay.highlight(crop, contour, origin, k)
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
        r, oid, path = self.last_result, self._sel, self.current_image_path
        if r is None or oid is None or not path:
            return
        contour = self._geometry()[oid].reshape(-1, 1, 2).astype(np.float32)
        base = r.extra.get("base_image")
        w = r.stats.get("image_width_px") or (base.shape[1] if base is not None else 0)
        h = r.stats.get("image_height_px") or (base.shape[0] if base is not None else 0)
        if not w or not h:
            return

        def outside(points):
            return [p for p in points if cv2.pointPolygonTest(contour, (p[0] * w, p[1] * h), True) < -15]
        self._remember(f"undo.{kind}")
        if oid in (r.extra.get(kind) or []):
            self.project.set_marks(kind, path, outside(self.project.marks_for(kind, path)))
        else:
            pt = self._sel_point
            if pt is None or cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) < 0:
                m = cv2.moments(contour)
                pt = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else tuple(contour[0, 0])
                if cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) < 0:
                    pt = tuple(contour[0, 0])
            self.project.set_marks(kind, path, self.project.marks_for(kind, path) + [[pt[0] / w, pt[1] / h]])
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
        from fenotit.gui.widgets import inspector as ins
        r = self.last_result
        if path == self.current_image_path and r is not None:
            base = r.extra.get("base_image")
            if base is None and self.scaler_left.has_image:
                base = self.scaler_left.original
            geom = self._geometry()
            if base is not None:
                if oid in geom:
                    crop, origin, k = ins.crop_around(base, geom[oid], (206, 170))
                    if crop is not None:
                        overlay.highlight(crop, geom[oid], origin, k)
                    return crop
                return cv2.resize(base, None, fx=206 / base.shape[1], fy=206 / base.shape[1],
                                  interpolation=cv2.INTER_AREA)
        try:
            from PIL import Image as _Im, ImageOps
            with _Im.open(path) as im:
                im.draft("RGB", (412, 412))
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((206, 206))
                return np.asarray(im)[:, :, ::-1].copy()
        except Exception:
            _log.debug("miniatura %s", path, exc_info=True)
            return None
