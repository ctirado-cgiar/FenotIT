"""
roi/selectors.py  v1.2
ROI selector con modo inclusión y exclusión.
- Inclusión: rectángulo, cuadrado, polígono — define área de análisis
- Exclusión: zona pintada con color personalizado — excluida del análisis
Ambos modos coexisten y se combinan.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import colorchooser
from typing import Callable

from fenotit import log

_log = log.get("gui.roi")


# Colores de dibujo
ROI_COLOR_RECT    = "#2166AC"
ROI_COLOR_POLY    = "#006837"
ROI_COLOR_EXCLUDE = "#E67E00"


class ROISelector:
    MODES = ["rectángulo", "cuadrado", "polígono", "hueco", "exclusión"]

    def __init__(self, canvas: tk.Canvas,
                 on_roi_change: Callable | None = None):
        self.canvas        = canvas
        self.on_roi_change = on_roi_change

        self.mode          = "rectángulo"
        self.drawing       = False
        self.start_pt      = None
        self.points        = []
        self.temp_shapes   = []
        self.final_shapes  = []

        # Máscara de inclusión (255 = analizar)
        self._inclusion_mask: np.ndarray | None = None
        # Lista de zonas de exclusión: [(pts_array, color_bgr), ...]
        self._exclusion_zones: list[tuple] = []
        # Color de exclusión actual (BGR)
        self._exclude_color = (255, 255, 255)   # blanco por defecto
        self._exclude_color_hex = "#FFFFFF"
        # Formas en coordenadas normalizadas (0-1) para guardar/restaurar
        self._inclusion_ops: list[dict] = []
        self._exclusion_ops: list[dict] = []
        self._pending: dict | None = None

        self._canvas_w = 400
        self._canvas_h = 300
        self._img_x = 0
        self._img_y = 0
        self._img_w = 400
        self._img_h = 300

        self._double_click_pending = False
        self._bind_events()

    # ── Configuración ─────────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        assert mode in self.MODES, f"Modo inválido: {mode}"
        self.mode = mode
        # No limpiar al cambiar de modo — permitir combinar

    def set_canvas_size(self, w: int, h: int):
        self._canvas_w = w
        self._canvas_h = h

    def set_image_offset(self, img_x, img_y, img_w, img_h):
        self._img_x = img_x
        self._img_y = img_y
        self._img_w = img_w
        self._img_h = img_h
        if self._pending is not None:
            data, self._pending = self._pending, None
            self.load_dict(data)

    def pick_exclusion_color(self, parent=None):
        """Abre selector de color para el parche de exclusión."""
        result = colorchooser.askcolor(
            color=self._exclude_color_hex,
            title="Color de zona de exclusión",
            parent=parent)
        if result and result[1]:
            self._exclude_color_hex = result[1]
            # Convertir hex → BGR
            h = result[1].lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            self._exclude_color = (b, g, r)

    def clear(self):
        """Limpia todo — inclusión y exclusión."""
        for sid in self.temp_shapes + self.final_shapes:
            try:
                self.canvas.delete(sid)
            except Exception:
                _log.debug("ignorado", exc_info=True)
        self.temp_shapes      = []
        self.final_shapes     = []
        self.points           = []
        self.drawing          = False
        self.start_pt         = None
        self._inclusion_mask  = None
        self._exclusion_zones = []
        self._inclusion_ops   = []
        self._exclusion_ops   = []
        self._double_click_pending = False
        if self.on_roi_change:
            self.on_roi_change(None)

    def clear_exclusions(self):
        """Solo limpia zonas de exclusión."""
        self._exclusion_zones = []
        self._exclusion_ops = []
        # Redibujar solo shapes de inclusión
        # (simplificado: limpiar todo y redibujar)
        self._redraw_all()

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def mask(self) -> np.ndarray | None:
        """Máscara de inclusión en coords de imagen display."""
        return self._inclusion_mask

    @property
    def has_exclusions(self) -> bool:
        return len(self._exclusion_zones) > 0

    def _op(self, kind: str, points, **extra) -> dict:
        w, h = max(self._img_w, 1), max(self._img_h, 1)
        return {"type": kind,
                "points": [[round(x / w, 5), round(y / h, 5)] for x, y in points],
                **extra}

    def restore_when_ready(self, data: dict | None):
        """Guarda formas para dibujarlas cuando la imagen tenga tamaño en pantalla."""
        self.clear()
        self._pending = data or None

    def to_dict(self) -> dict:
        if self._pending is not None:
            return self._pending
        if not self._inclusion_ops and not self._exclusion_ops:
            return {}
        return {"inclusion": self._inclusion_ops, "exclusions": self._exclusion_ops}

    def load_dict(self, data: dict | None):
        """Reconstruye máscaras y formas a partir de to_dict()."""
        self.clear()
        if not data:
            return
        for op in data.get("inclusion", []):
            pts = self._to_px(op["points"])
            if op["type"] == "rect":
                (x0, y0), (x1, y1) = pts
                self._build_rect_mask(x0, y0, x1, y1)
            else:
                self._build_polygon_mask(pts, hollow=op["type"] == "hole")
        for op in data.get("exclusions", []):
            color = tuple(op.get("color", (255, 255, 255)))
            self._exclusion_zones.append((np.array(self._to_px(op["points"]), dtype=np.int32), color))
        self._inclusion_ops = list(data.get("inclusion", []))
        self._exclusion_ops = list(data.get("exclusions", []))
        self.redraw_shapes()
        if self.on_roi_change:
            self.on_roi_change(self._inclusion_mask)

    def _to_px(self, pts):
        w, h = self._img_w, self._img_h
        return [(int(round(x * w)), int(round(y * h))) for x, y in pts]

    def redraw_shapes(self):
        """Vuelve a dibujar las formas guardadas con el zoom y desplazamiento actuales."""
        for sid in self.final_shapes:
            self.canvas.delete(sid)
        self.final_shapes = []
        for op in self._inclusion_ops:
            flat = [c for x, y in self._to_px(op["points"]) for c in self._img_to_canvas(x, y)]
            if op["type"] == "rect":
                sid = self.canvas.create_rectangle(*flat, outline=ROI_COLOR_RECT, width=2, tags="roi_final")
            elif op["type"] == "polygon":
                sid = self.canvas.create_polygon(*flat, outline=ROI_COLOR_POLY, fill=ROI_COLOR_POLY,
                                                 stipple="gray25", width=2, tags="roi_final")
            else:
                sid = self.canvas.create_polygon(*flat, outline=ROI_COLOR_EXCLUDE, fill="",
                                                 width=2, tags="roi_final")
            self.final_shapes.append(sid)
        for op in self._exclusion_ops:
            color = op.get("color", (255, 255, 255))
            hexc = "#%02x%02x%02x" % (color[2], color[1], color[0])
            flat = [c for x, y in self._to_px(op["points"]) for c in self._img_to_canvas(x, y)]
            self.final_shapes.append(self.canvas.create_polygon(
                *flat, outline=hexc, fill=hexc, stipple="gray50", width=2, tags="roi_final"))

    def exclusions_normalized(self) -> list[tuple]:
        """Zonas de exclusión con coordenadas 0-1 (sirven para cualquier tamaño de imagen)."""
        size = np.array([max(self._img_w, 1), max(self._img_h, 1)], float)
        return [(np.asarray(pts, float) / size, color) for pts, color in self._exclusion_zones]

    def get_combined_params(self) -> dict:
        """
        Retorna dict con inclusión y exclusiones para pasar a análisis.
        Las coordenadas están en imagen de display — main_window escala
        a original antes de pasar a los módulos.
        """
        return {
            "inclusion_mask":   self._inclusion_mask,
            "exclusion_zones":  self._exclusion_zones,
            "exclude_color":    self._exclude_color,
        }

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica inclusión + exclusión sobre una imagen BGR.
        - Fuera del ROI de inclusión → blanco
        - Zonas de exclusión → color del parche
        Retorna imagen modificada (no modifica original).
        """
        result = image.copy()
        ih, iw = result.shape[:2]

        # Aplicar inclusión: todo fuera del ROI → blanco
        if self._inclusion_mask is not None:
            # Escalar máscara a tamaño original
            mask_full = cv2.resize(
                self._inclusion_mask, (iw, ih),
                interpolation=cv2.INTER_NEAREST)
            result[mask_full == 0] = (255, 255, 255)

        # Aplicar exclusiones
        for pts_display, color_bgr in self._exclusion_zones:
            # Escalar puntos de display a original
            scale_x = iw / max(self._img_w, 1)
            scale_y = ih / max(self._img_h, 1)
            pts_orig = (pts_display * np.array([scale_x, scale_y])
                       ).astype(np.int32)
            cv2.fillPoly(result, [pts_orig], color_bgr)

        return result

    # ── Eventos ───────────────────────────────────────────────────────────────

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>",  self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Button-3>",        self._on_right_click)

    def _canvas_to_img(self, cx, cy):
        ix = max(0, min(cx - self._img_x, self._img_w - 1))
        iy = max(0, min(cy - self._img_y, self._img_h - 1))
        return ix, iy

    def _img_to_canvas(self, ix, iy):
        return ix + self._img_x, iy + self._img_y

    def _on_press(self, event):
        if self._double_click_pending:
            self._double_click_pending = False
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        if self.mode in ("rectángulo", "cuadrado"):
            self.drawing  = True
            self.start_pt = (ix, iy)
        elif self.mode in ("polígono", "hueco", "exclusión"):
            self.points.append((ix, iy))
            self._draw_polygon_temp()

    def _on_drag(self, event):
        if not self.drawing:
            return
        self._clear_temp()
        x0, y0 = self.start_pt
        ix, iy  = self._canvas_to_img(event.x, event.y)
        if self.mode == "cuadrado":
            size = min(abs(ix - x0), abs(iy - y0))
            ix   = x0 + (size if ix >= x0 else -size)
            iy   = y0 + (size if iy >= y0 else -size)
        cx0, cy0 = self._img_to_canvas(x0, y0)
        cx1, cy1 = self._img_to_canvas(ix, iy)
        color = ROI_COLOR_RECT
        sid = self.canvas.create_rectangle(
            cx0, cy0, cx1, cy1,
            outline=color, dash=(6, 3), width=2, tags="roi_temp")
        self.temp_shapes.append(sid)

    def _on_release(self, event):
        if not self.drawing:
            return
        self.drawing = False
        x0, y0 = self.start_pt
        ix, iy  = self._canvas_to_img(event.x, event.y)
        if self.mode == "cuadrado":
            size = min(abs(ix - x0), abs(iy - y0))
            ix   = x0 + (size if ix >= x0 else -size)
            iy   = y0 + (size if iy >= y0 else -size)
        self._clear_temp()
        x_min, x_max = sorted([x0, ix])
        y_min, y_max = sorted([y0, iy])
        cx0, cy0 = self._img_to_canvas(x_min, y_min)
        cx1, cy1 = self._img_to_canvas(x_max, y_max)
        sid = self.canvas.create_rectangle(
            cx0, cy0, cx1, cy1,
            outline=ROI_COLOR_RECT, width=2, tags="roi_final")
        self.final_shapes.append(sid)
        self._inclusion_ops = [self._op("rect", [(x_min, y_min), (x_max, y_max)])]
        self._build_rect_mask(x_min, y_min, x_max, y_max)

    def _on_double_click(self, event):
        self._double_click_pending = True
        if self.mode in ("polígono", "hueco", "exclusión") \
                and len(self.points) >= 3:
            self._finalize_polygon()
        else:
            self._double_click_pending = False

    def _on_right_click(self, event):
        if self.mode in ("polígono", "hueco", "exclusión") \
                and self.points:
            self.points.pop()
            self._draw_polygon_temp()

    # ── Dibujo ────────────────────────────────────────────────────────────────

    def _clear_temp(self):
        for sid in self.temp_shapes:
            try:
                self.canvas.delete(sid)
            except Exception:
                _log.debug("ignorado", exc_info=True)
        self.temp_shapes = []

    def _get_color(self):
        if self.mode == "exclusión":
            return self._exclude_color_hex
        if self.mode == "hueco":
            return ROI_COLOR_EXCLUDE
        return ROI_COLOR_POLY

    def _draw_polygon_temp(self):
        self._clear_temp()
        if not self.points:
            return
        color = self._get_color()
        canvas_pts = [self._img_to_canvas(px, py)
                      for px, py in self.points]
        for cx, cy in canvas_pts:
            sid = self.canvas.create_oval(
                cx-4, cy-4, cx+4, cy+4,
                fill=color, outline="#FFFFFF",
                width=1, tags="roi_temp")
            self.temp_shapes.append(sid)
        if len(canvas_pts) >= 2:
            flat = [c for pt in canvas_pts for c in pt]
            sid  = self.canvas.create_line(
                *flat, fill=color, width=2,
                dash=(4,3), tags="roi_temp")
            self.temp_shapes.append(sid)
            # Línea de cierre tentativa
            sx, sy = canvas_pts[0]
            ex, ey = canvas_pts[-1]
            self.temp_shapes.append(
                self.canvas.create_line(
                    ex, ey, sx, sy,
                    fill=color, width=1,
                    dash=(2,4), tags="roi_temp"))

    def _finalize_polygon(self):
        self._clear_temp()
        color = self._get_color()
        canvas_pts = [self._img_to_canvas(px, py)
                      for px, py in self.points]
        flat = [c for pt in canvas_pts for c in pt]

        if self.mode == "exclusión":
            # Dibujo con color de exclusión
            sid = self.canvas.create_polygon(
                *flat, outline=color, fill=color,
                stipple="gray50", width=2, tags="roi_final")
            self.final_shapes.append(sid)
            # Guardar zona de exclusión
            pts_arr = np.array(self.points, dtype=np.int32)
            self._exclusion_zones.append(
                (pts_arr, self._exclude_color))
            self._exclusion_ops.append(
                self._op("exclusion", self.points, color=list(self._exclude_color)))
        elif self.mode == "hueco":
            sid = self.canvas.create_polygon(
                *flat, outline=ROI_COLOR_EXCLUDE, fill="",
                width=2, tags="roi_final")
            self.final_shapes.append(sid)
            self._inclusion_ops.append(self._op("hole", self.points))
            self._build_polygon_mask(self.points, hollow=True)
        else:
            sid = self.canvas.create_polygon(
                *flat, outline=color, fill=color,
                stipple="gray25", width=2, tags="roi_final")
            self.final_shapes.append(sid)
            self._inclusion_ops = [self._op("polygon", self.points)]
            self._build_polygon_mask(self.points, hollow=False)

        self.points = []
        self._double_click_pending = False
        if self.on_roi_change:
            self.on_roi_change(self._inclusion_mask)

    def _redraw_all(self):
        """Redibuja todo desde cero (usado al limpiar exclusiones)."""
        for sid in self.final_shapes:
            try:
                self.canvas.delete(sid)
            except Exception:
                _log.debug("ignorado", exc_info=True)
        self.final_shapes = []
        if self.on_roi_change:
            self.on_roi_change(self._inclusion_mask)

    # ── Máscaras ──────────────────────────────────────────────────────────────

    def _build_rect_mask(self, x0, y0, x1, y1):
        mask = np.zeros((self._img_h, self._img_w), dtype=np.uint8)
        cv2.rectangle(mask,
                      (max(0,x0), max(0,y0)),
                      (min(self._img_w,x1), min(self._img_h,y1)),
                      255, -1)
        self._inclusion_mask = mask
        if self.on_roi_change:
            self.on_roi_change(mask)

    def _build_polygon_mask(self, points, hollow=False):
        pts = np.array(points, dtype=np.int32)
        if hollow and self._inclusion_mask is not None:
            base = self._inclusion_mask
            if base.shape[:2] != (self._img_h, self._img_w):
                base = cv2.resize(base, (self._img_w, self._img_h), interpolation=cv2.INTER_NEAREST)
            base = base.copy()
            cv2.fillPoly(base, [pts], 0)
        else:
            base = np.zeros((self._img_h, self._img_w), dtype=np.uint8)
            cv2.fillPoly(base, [pts], 255)
        self._inclusion_mask = base
        if self.on_roi_change:
            self.on_roi_change(base)