"""
ui/zoom_controller.py  v1.2
Zoom simplificado — solo botones + / - y flechas del teclado.
Sin rueda del mouse para no interferir con ROI ni otros eventos.
Pan con flechas cuando hay zoom activo.
"""

import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

from fenotit import log

_log = log.get("gui.zoom")


class ZoomState:
    ZOOM_MIN:  float = 0.05
    ZOOM_MAX:  float = 40.0
    ZOOM_STEP: float = 1.30

    def __init__(self):
        self.zoom:  float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

    def fit(self, canvas_w: int, canvas_h: int,
            img_w: int, img_h: int):
        """Ajusta zoom para que la imagen quepa completa."""
        if img_w <= 0 or img_h <= 0:
            return
        self.zoom  = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        self.pan_x = 0.0
        self.pan_y = 0.0

    def zoom_at_center(self, canvas_w: int, canvas_h: int,
                       img_w: int, img_h: int, factor: float):
        """Zoom centrado en el centro del canvas."""
        new_zoom = max(self.ZOOM_MIN,
                       min(self.ZOOM_MAX, self.zoom * factor))
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        # Mantener el centro fijo
        cx, cy = canvas_w / 2, canvas_h / 2
        disp_w = img_w * self.zoom
        disp_h = img_h * self.zoom
        off_x  = (canvas_w - disp_w) / 2 + self.pan_x
        off_y  = (canvas_h - disp_h) / 2 + self.pan_y
        img_x  = (cx - off_x) / self.zoom
        img_y  = (cy - off_y) / self.zoom
        self.pan_x += img_x * (self.zoom - new_zoom)
        self.pan_y += img_y * (self.zoom - new_zoom)
        self.zoom   = new_zoom

    def pan(self, dx: float, dy: float,
            canvas_w: int, canvas_h: int,
            img_w: int, img_h: int):
        """Desplaza la imagen."""
        self.pan_x += dx
        self.pan_y += dy
        self.clamp_pan(canvas_w, canvas_h, img_w, img_h)

    def clamp_pan(self, canvas_w: int, canvas_h: int,
                  img_w: int, img_h: int):
        disp_w   = img_w * self.zoom
        disp_h   = img_h * self.zoom
        margin_x = max(canvas_w * 0.5, disp_w * 0.5)
        margin_y = max(canvas_h * 0.5, disp_h * 0.5)
        self.pan_x = max(-margin_x, min(margin_x, self.pan_x))
        self.pan_y = max(-margin_y, min(margin_y, self.pan_y))

    def image_offset(self, canvas_w: int, canvas_h: int,
                     img_w: int, img_h: int) -> tuple[int, int]:
        disp_w = img_w * self.zoom
        disp_h = img_h * self.zoom
        x = (canvas_w - disp_w) / 2 + self.pan_x
        y = (canvas_h - disp_h) / 2 + self.pan_y
        return int(x), int(y)

    def display_size(self, img_w: int,
                     img_h: int) -> tuple[int, int]:
        return max(1, int(img_w * self.zoom)), \
               max(1, int(img_h * self.zoom))

    @property
    def zoom_pct(self) -> int:
        return int(self.zoom * 100)

    @property
    def is_zoomed(self) -> bool:
        return self.zoom > 1.01 or \
               abs(self.pan_x) > 1 or abs(self.pan_y) > 1


class ZoomController:
    """
    Zoom y pan para par de canvas.
    Zoom: solo botones externos (zoom_in / zoom_out / fit).
    Pan:  flechas del teclado cuando hay zoom activo.
    No intercepta clics del mouse — el ROI funciona normalmente.
    """

    PAN_STEP = 20   # píxeles por tecla de flecha

    def __init__(self,
                 canvas_left: tk.Canvas,
                 canvas_right: tk.Canvas,
                 on_redraw=None):
        self.cl        = canvas_left
        self.cr        = canvas_right
        self.on_redraw = on_redraw
        self.state     = ZoomState()

        self._img_left:  np.ndarray | None = None
        self._img_right: np.ndarray | None = None
        self._photo_left  = None
        self._photo_right = None
        self._zoom_var: tk.StringVar | None = None

        # Solo flechas — no interferimos con clics
        canvas_left.bind("<Left>",  self._on_arrow)
        canvas_left.bind("<Right>", self._on_arrow)
        canvas_left.bind("<Up>",    self._on_arrow)
        canvas_left.bind("<Down>",  self._on_arrow)
        canvas_left.config(takefocus=True)
        # Clic izquierdo da foco al canvas (necesario para flechas)
        # IMPORTANTE: add="+" para no reemplazar el bind del ROI
        canvas_left.bind("<ButtonPress-1>",
                         lambda e: e.widget.focus_set(), add="+")

    # ── API pública ───────────────────────────────────────────────────────────

    def set_zoom_var(self, var: tk.StringVar):
        self._zoom_var = var

    def set_left(self, img_bgr: np.ndarray | None):
        self._img_left = img_bgr
        if img_bgr is not None:
            self._fit_to_canvas()
        else:
            self._redraw()

    def set_right(self, img_bgr: np.ndarray | None):
        self._img_right = img_bgr
        self._redraw()

    def set_images(self, left: np.ndarray | None,
                   right: np.ndarray | None):
        self._img_left  = left
        self._img_right = right
        if left is not None:
            self._fit_to_canvas()
        else:
            self._redraw()

    def zoom_in(self):
        self.cl.update_idletasks()
        cw = max(self.cl.winfo_width(),  1)
        ch = max(self.cl.winfo_height(), 1)
        img = self._img_left if self._img_left is not None else self._img_right
        if img is None:
            return
        ih, iw = img.shape[:2]
        self.state.zoom_at_center(cw, ch, iw, ih,
                                  ZoomState.ZOOM_STEP)
        self.state.clamp_pan(cw, ch, iw, ih)
        self._redraw()

    def zoom_out(self):
        self.cl.update_idletasks()
        cw = max(self.cl.winfo_width(),  1)
        ch = max(self.cl.winfo_height(), 1)
        img = self._img_left if self._img_left is not None else self._img_right
        if img is None:
            return
        ih, iw = img.shape[:2]
        self.state.zoom_at_center(cw, ch, iw, ih,
                                  1.0 / ZoomState.ZOOM_STEP)
        self.state.clamp_pan(cw, ch, iw, ih)
        self._redraw()

    def fit(self):
        self._fit_to_canvas()

    def reset(self):
        self._fit_to_canvas()

    def current_image_offset(self) -> tuple[int, int, int, int]:
        if self._img_left is None:
            return 0, 0, max(self.cl.winfo_width(), 1), \
                         max(self.cl.winfo_height(), 1)
        self.cl.update_idletasks()
        cw = max(self.cl.winfo_width(),  1)
        ch = max(self.cl.winfo_height(), 1)
        ih, iw = self._img_left.shape[:2]
        dw, dh = self.state.display_size(iw, ih)
        ox, oy = self.state.image_offset(cw, ch, iw, ih)
        return ox, oy, dw, dh

    # ── Flechas ───────────────────────────────────────────────────────────────

    def _on_arrow(self, event):
        if not self.state.is_zoomed:
            return
        img = self._img_left if self._img_left is not None else self._img_right
        if img is None:
            return
        self.cl.update_idletasks()
        cw = max(self.cl.winfo_width(),  1)
        ch = max(self.cl.winfo_height(), 1)
        ih, iw = img.shape[:2]
        step = self.PAN_STEP
        dx, dy = 0, 0
        if event.keysym == "Left":  dx =  step
        if event.keysym == "Right": dx = -step
        if event.keysym == "Up":    dy =  step
        if event.keysym == "Down":  dy = -step
        self.state.pan(dx, dy, cw, ch, iw, ih)
        self._redraw()

    # ── Fit interno ───────────────────────────────────────────────────────────

    def _fit_to_canvas(self):
        if self._img_left is None:
            self._redraw()
            return
        self.cl.update_idletasks()
        cw = max(self.cl.winfo_width(),  100)
        ch = max(self.cl.winfo_height(), 100)
        ih, iw = self._img_left.shape[:2]
        self.state.fit(cw, ch, iw, ih)
        self._redraw()

    # ── Renderizado ───────────────────────────────────────────────────────────

    def _redraw(self):
        self._draw_canvas(self.cl,  self._img_left,
                          "_photo_left",  left=True)
        self._draw_canvas(self.cr,  self._img_right,
                          "_photo_right", left=False)
        if self._zoom_var is not None:
            self._zoom_var.set(f"{self.state.zoom_pct}%")
        if self.on_redraw:
            self.on_redraw()

    def _draw_canvas(self, canvas: tk.Canvas,
                     img_bgr: np.ndarray | None,
                     attr: str, left: bool):
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(),  1)
        ch = max(canvas.winfo_height(), 1)
        canvas.delete("all")
        if img_bgr is None:
            return
        ih, iw = img_bgr.shape[:2]
        dw, dh = self.state.display_size(iw, ih)
        ox, oy = self.state.image_offset(cw, ch, iw, ih)
        interp = cv2.INTER_AREA if self.state.zoom < 1.0 \
                 else cv2.INTER_LINEAR
        try:
            resized = cv2.resize(img_bgr, (dw, dh),
                                 interpolation=interp)
            rgb   = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            setattr(self, attr, photo)
            canvas.create_image(ox, oy, anchor="nw", image=photo)
        except Exception:
            _log.debug("Render falló", exc_info=True)
            return
        if left and hasattr(canvas, '_roi_selector_ref'):
            canvas._roi_selector_ref.set_image_offset(ox, oy, dw, dh)

    def redraw_with_overlay(self,
                            overlay_left: np.ndarray | None = None):
        img_l = overlay_left \
                if overlay_left is not None else self._img_left
        self._draw_canvas(self.cl, img_l,
                          "_photo_left",  left=True)
        self._draw_canvas(self.cr, self._img_right,
                          "_photo_right", left=False)
        if self._zoom_var is not None:
            self._zoom_var.set(f"{self.state.zoom_pct}%")
        if self.on_redraw:
            self.on_redraw()


# ── ZoomableCanvas para ventanas de calibración ───────────────────────────────

class ZoomableCanvas(tk.Canvas):
    """
    Canvas con zoom/pan para ventanas de calibración.
    Zoom: botones externos o métodos fit/zoom_in/zoom_out.
    Pan: flechas del teclado.
    """

    PAN_STEP = 20

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._img_bgr: np.ndarray | None = None
        self._photo   = None
        self._state   = ZoomState()
        self._fitted  = False

        self.config(takefocus=True)
        self.bind("<Left>",  self._on_arrow)
        self.bind("<Right>", self._on_arrow)
        self.bind("<Up>",    self._on_arrow)
        self.bind("<Down>",  self._on_arrow)
        self.bind("<ButtonPress-1>",
                  lambda e: e.widget.focus_set(), add="+")
        self.bind("<Configure>", lambda e: self._fit_if_needed())

    def set_image(self, img_bgr: np.ndarray | None):
        self._img_bgr = img_bgr
        self._fitted  = False
        self._fit_and_draw()

    def zoom_in(self):
        self._zoom(ZoomState.ZOOM_STEP)

    def zoom_out(self):
        self._zoom(1.0 / ZoomState.ZOOM_STEP)

    def fit(self):
        self._fit_and_draw()

    def _zoom(self, factor: float):
        if self._img_bgr is None:
            return
        self.update_idletasks()
        cw = max(self.winfo_width(),  1)
        ch = max(self.winfo_height(), 1)
        ih, iw = self._img_bgr.shape[:2]
        self._state.zoom_at_center(cw, ch, iw, ih, factor)
        self._state.clamp_pan(cw, ch, iw, ih)
        self._redraw()

    def _fit_if_needed(self):
        if not self._fitted:
            self._fit_and_draw()
        else:
            self._redraw()

    def _fit_and_draw(self):
        if self._img_bgr is None:
            return
        self.update_idletasks()
        cw = max(self.winfo_width(),  10)
        ch = max(self.winfo_height(), 10)
        ih, iw = self._img_bgr.shape[:2]
        self._state.fit(cw, ch, iw, ih)
        self._fitted = True
        self._redraw()

    def _on_arrow(self, event):
        if not self._state.is_zoomed or self._img_bgr is None:
            return
        self.update_idletasks()
        cw = max(self.winfo_width(),  1)
        ch = max(self.winfo_height(), 1)
        ih, iw = self._img_bgr.shape[:2]
        step = self.PAN_STEP
        dx, dy = 0, 0
        if event.keysym == "Left":  dx =  step
        if event.keysym == "Right": dx = -step
        if event.keysym == "Up":    dy =  step
        if event.keysym == "Down":  dy = -step
        self._state.pan(dx, dy, cw, ch, iw, ih)
        self._redraw()

    def _redraw(self):
        self.update_idletasks()
        cw = max(self.winfo_width(),  1)
        ch = max(self.winfo_height(), 1)
        self.delete("all")
        if self._img_bgr is None:
            return
        ih, iw = self._img_bgr.shape[:2]
        dw, dh = self._state.display_size(iw, ih)
        ox, oy = self._state.image_offset(cw, ch, iw, ih)
        interp = cv2.INTER_AREA if self._state.zoom < 1.0 \
                 else cv2.INTER_LINEAR
        try:
            resized = cv2.resize(self._img_bgr, (dw, dh),
                                 interpolation=interp)
            rgb   = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._photo = photo
            self.create_image(ox, oy, anchor="nw", image=photo)
        except Exception:
            _log.debug("ignorado", exc_info=True)
