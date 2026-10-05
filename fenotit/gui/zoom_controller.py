"""Zoom y desplazamiento de las imágenes (ventana principal y diálogos de calibración)."""

import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

from fenotit import log

_log = log.get("gui.zoom")


class ZoomState:
    """Zoom = píxeles de pantalla por píxel de imagen. Los pasos son multiplicativos, así
    que valen igual para una foto de 50 MP que para una de 300 px; los límites se miden
    desde "ajustar" (no se aleja a menos de 1/4 de eso) y hasta 32 px de pantalla por píxel."""
    ZOOM_MIN:  float = 0.01
    ZOOM_MAX:  float = 32.0
    ZOOM_STEP: float = 1.25          # botones y teclado
    WHEEL_STEP: float = 1.20         # una muesca de rueda (120); el trackpad manda fracciones
    FIT_MAX: float = 4.0             # una imagen chica se agranda hasta 4× al ajustar

    def __init__(self):
        self.zoom:  float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.fit_zoom: float = 1.0

    @staticmethod
    def fit_value(canvas_w, canvas_h, img_w, img_h) -> float:
        return min(canvas_w / img_w, canvas_h / img_h, ZoomState.FIT_MAX)

    def _limits(self):
        return max(self.ZOOM_MIN, self.fit_zoom / 4), max(self.ZOOM_MAX, self.fit_zoom)

    def fit(self, canvas_w: int, canvas_h: int,
            img_w: int, img_h: int):
        """Ajusta zoom para que la imagen quepa completa."""
        if img_w <= 0 or img_h <= 0:
            return
        self.zoom = self.fit_zoom = self.fit_value(canvas_w, canvas_h, img_w, img_h)
        self.pan_x = 0.0
        self.pan_y = 0.0

    def zoom_to_rect(self, x0, y0, x1, y1, canvas_w, canvas_h, img_w, img_h):
        """Lleva el rectángulo (en coordenadas del canvas) a llenar la vista."""
        off_x = (canvas_w - img_w * self.zoom) / 2 + self.pan_x
        off_y = (canvas_h - img_h * self.zoom) / 2 + self.pan_y
        ix0, ix1 = sorted(((x0 - off_x) / self.zoom, (x1 - off_x) / self.zoom))
        iy0, iy1 = sorted(((y0 - off_y) / self.zoom, (y1 - off_y) / self.zoom))
        w, h = max(ix1 - ix0, 1.0), max(iy1 - iy0, 1.0)
        lo, hi = self._limits()
        self.zoom = max(lo, min(hi, min(canvas_w / w, canvas_h / h)))
        cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
        self.pan_x = canvas_w / 2 - cx * self.zoom - (canvas_w - img_w * self.zoom) / 2
        self.pan_y = canvas_h / 2 - cy * self.zoom - (canvas_h - img_h * self.zoom) / 2

    def zoom_at_center(self, canvas_w: int, canvas_h: int,
                       img_w: int, img_h: int, factor: float):
        """Zoom centrado en el centro del canvas."""
        lo, hi = self._limits()
        new_zoom = max(lo, min(hi, self.zoom * factor))
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
        self.pan_x = cx - img_x * new_zoom - (canvas_w - img_w * new_zoom) / 2
        self.pan_y = cy - img_y * new_zoom - (canvas_h - img_h * new_zoom) / 2
        self.zoom   = new_zoom

    def zoom_at(self, px: float, py: float, canvas_w: int, canvas_h: int,
                img_w: int, img_h: int, factor: float):
        """Zoom manteniendo fijo el punto (px, py) del canvas (p. ej. el cursor)."""
        lo, hi = self._limits()
        new_zoom = max(lo, min(hi, self.zoom * factor))
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        off_x = (canvas_w - img_w * self.zoom) / 2 + self.pan_x
        off_y = (canvas_h - img_h * self.zoom) / 2 + self.pan_y
        img_x, img_y = (px - off_x) / self.zoom, (py - off_y) / self.zoom
        self.pan_x = px - img_x * new_zoom - (canvas_w - img_w * new_zoom) / 2
        self.pan_y = py - img_y * new_zoom - (canvas_h - img_h * new_zoom) / 2
        self.zoom = new_zoom

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
        return self.zoom > self.fit_zoom * 1.01 or \
               abs(self.pan_x) > 1 or abs(self.pan_y) > 1


def wheel_factor(event) -> float:
    """Factor de zoom de un evento de rueda. En Windows `delta` es 120 por muesca y el
    trackpad (pellizco = Ctrl+rueda) manda valores chicos: el zoom sigue al gesto."""
    if getattr(event, "num", 0) in (4, 5):
        notches = 1 if event.num == 4 else -1
    else:
        notches = max(-3.0, min(3.0, getattr(event, "delta", 0) / 120))
    return ZoomState.WHEEL_STEP ** notches


def render(img_bgr: np.ndarray, state: ZoomState, cw: int, ch: int, post=None, size=None):
    """Solo la parte visible, ya escalada: (rgb, x, y) o None. Así un zoom de 3200 % en una
    foto de 8 MP no intenta crear una imagen gigante. `size` = (ancho, alto) de la foto
    cuando `img_bgr` es una copia reducida (vista previa de fotos muy grandes)."""
    iw, ih = size or (img_bgr.shape[1], img_bgr.shape[0])
    z = state.zoom
    ox, oy = state.image_offset(cw, ch, iw, ih)
    x0, y0 = max(0, int(np.floor(-ox / z))), max(0, int(np.floor(-oy / z)))
    x1, y1 = min(iw, int(np.ceil((cw - ox) / z)) + 1), min(ih, int(np.ceil((ch - oy) / z)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    k = img_bgr.shape[1] / iw
    if k != 1:
        crop = img_bgr[int(y0 * k):max(int(y0 * k) + 1, int(np.ceil(y1 * k))),
                       int(x0 * k):max(int(x0 * k) + 1, int(np.ceil(x1 * k)))]
    else:
        crop = img_bgr[y0:y1, x0:x1]
    dw, dh = max(1, int(round((x1 - x0) * z))), max(1, int(round((y1 - y0) * z)))
    interp = cv2.INTER_AREA if z < 1 else (cv2.INTER_NEAREST if z >= 4 else cv2.INTER_LINEAR)
    out = cv2.resize(crop, (dw, dh), interpolation=interp)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if post is not None:                      # marcas a tamaño de pantalla (overlay.draw)
        out = np.ascontiguousarray(out) if out is not crop else out.copy()
        post(out, (x0, y0), z)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), ox + int(round(x0 * z)), oy + int(round(y0 * z))


class ZoomController:
    """Zoom y desplazamiento sincronizados para Entrada y Resultado.

    Herramientas (botón izquierdo, si no se está dibujando una ROI):
      "pan"       arrastrar mueve la imagen (por defecto)
      "zoom_area" arrastrar un rectángulo y acercarse a él; luego vuelve a "pan"
    Siempre: rueda = zoom en el cursor (el trackpad da pasos finos), Shift+rueda =
    mover a los lados, botón del medio = mover.
    """

    PAN_STEP = 40   # píxeles por tecla de flecha

    def __init__(self,
                 canvas_left: tk.Canvas,
                 canvas_right: tk.Canvas,
                 on_redraw=None):
        self.cl        = canvas_left
        self.cr        = canvas_right
        self.on_redraw = on_redraw
        self.on_tool_change = None
        self.state     = ZoomState()
        self.tool      = "pan"

        self._img_left:  np.ndarray | None = None
        self._img_right: np.ndarray | None = None
        self._overlay_left: np.ndarray | None = None
        self._marks_right = None              # (overlay, colores): se dibujan a tamaño de pantalla
        self._photo_left  = None
        self._photo_right = None
        self._zoom_var: tk.StringVar | None = None
        self._drag = None
        self._rect_start = None
        self._click = None
        self.on_pick = None                   # (x, y) en px de imagen: clic con "inspeccionar"
        self.highlight = None                 # contorno (px de imagen) del objeto elegido

        for c in (canvas_left, canvas_right):
            c.config(takefocus=True)
            c.bind("<MouseWheel>", self._on_wheel, add="+")
            c.bind("<Shift-MouseWheel>", self._on_hwheel, add="+")
            c.bind("<Button-4>", self._on_wheel, add="+")
            c.bind("<Button-5>", self._on_wheel, add="+")
            for btn in ("1", "2"):
                c.bind(f"<ButtonPress-{btn}>", lambda e, b=btn: self._press(e, b), add="+")
                c.bind(f"<B{btn}-Motion>", lambda e, b=btn: self._motion(e, b), add="+")
                c.bind(f"<ButtonRelease-{btn}>", lambda e, b=btn: self._release(e, b), add="+")
            c.bind("<Enter>", lambda e: self._update_cursor(e.widget), add="+")

    # ── Herramientas ──────────────────────────────────────────────────────────

    def set_tool(self, tool: str):
        self.tool = tool
        for c in (self.cl, self.cr):
            self._update_cursor(c)
        if self.on_tool_change:
            self.on_tool_change(tool)

    def _roi_drawing(self, canvas) -> bool:
        roi = getattr(self.cl, "_roi_selector_ref", None)
        return canvas is self.cl and roi is not None and roi.active

    def _update_cursor(self, canvas):
        busy = self._roi_drawing(canvas) or self.tool == "zoom_area"
        canvas.config(cursor="crosshair" if busy else ("hand2" if self.tool == "inspect" else "fleur"))

    def _ref(self) -> tk.Canvas:
        """El canvas que se ve (si se ocultó la entrada, el del resultado)."""
        return self.cr if self.cr.winfo_ismapped() and not self.cl.winfo_ismapped() else self.cl

    def _ref_size(self):
        img = self._img_left if self._img_left is not None else self._img_right
        if img is None:
            return None
        ref = self._ref()
        ref.update_idletasks()
        return (max(ref.winfo_width(), 1), max(ref.winfo_height(), 1),
                img.shape[1], img.shape[0])

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _on_wheel(self, event):
        size = self._ref_size()
        if size is None:
            return
        self.state.zoom_at(event.x, event.y, *size, wheel_factor(event))
        self.state.clamp_pan(*size)
        self._redraw()

    def _on_hwheel(self, event):
        size = self._ref_size()
        if size is None:
            return
        self.state.pan(getattr(event, "delta", 0) / 120 * self.PAN_STEP, 0, *size)
        self._redraw()
        return "break"

    def _press(self, event, button):
        event.widget.focus_set()
        if button == "1" and self._in_minimap(event):
            self._mini_drag = True
            self._minimap_goto(event)
            return "break"
        if button == "1" and self._roi_drawing(event.widget):
            return
        if button == "1" and self.tool == "zoom_area":
            self._rect_start = (event.widget, event.x, event.y)
            return
        self._drag = (event.x, event.y)
        self._click = (event.widget, event.x, event.y) if button == "1" and self.tool == "inspect" else None

    def _to_image(self, canvas, x, y):
        img = self._img_left if self._img_left is not None else self._img_right
        if img is None:
            return None
        cw, ch = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        ox, oy = self.state.image_offset(cw, ch, img.shape[1], img.shape[0])
        return (x - ox) / self.state.zoom, (y - oy) / self.state.zoom

    def _motion(self, event, button):
        if getattr(self, "_mini_drag", False):
            self._minimap_goto(event)
            return "break"
        if button == "1" and self._roi_drawing(event.widget):
            return
        if self._rect_start and button == "1":
            _, x0, y0 = self._rect_start
            for c in (self.cl, self.cr):
                c.delete("zoomrect")
                c.create_rectangle(x0, y0, event.x, event.y, outline="#FFFFFF", width=1, tags="zoomrect")
                c.create_rectangle(x0, y0, event.x, event.y, outline="#1F5FA8", dash=(4, 3), tags="zoomrect")
            return
        if self._drag is None:
            return
        size = self._ref_size()
        if size is None:
            return
        dx, dy = event.x - self._drag[0], event.y - self._drag[1]
        self._drag = (event.x, event.y)
        self.state.pan(dx, dy, *size)
        self._redraw()

    def _release(self, event, button):
        self._drag = None
        click, self._click = self._click, None
        if click and button == "1" and click[0] is event.widget \
                and abs(event.x - click[1]) < 4 and abs(event.y - click[2]) < 4 and self.on_pick:
            pt = self._to_image(event.widget, event.x, event.y)
            if pt is not None:
                self.on_pick(*pt)
            return
        if getattr(self, "_mini_drag", False):
            self._mini_drag = False
            return "break"
        if not (self._rect_start and button == "1"):
            return
        _, x0, y0 = self._rect_start
        self._rect_start = None
        for c in (self.cl, self.cr):
            c.delete("zoomrect")
        size = self._ref_size()
        if size is None:
            return
        if abs(event.x - x0) < 6 or abs(event.y - y0) < 6:      # clic: acercar ahí
            self.state.zoom_at(event.x, event.y, *size, 2.0)
        else:
            self.state.zoom_to_rect(x0, y0, event.x, event.y, *size)
        self.state.clamp_pan(*size)
        self._redraw()
        self.set_tool("pan")

    # ── API pública ───────────────────────────────────────────────────────────

    def set_zoom_var(self, var: tk.StringVar):
        self._zoom_var = var

    def set_left(self, img_bgr: np.ndarray | None):
        self._img_left = img_bgr
        self._overlay_left = None
        if img_bgr is not None:
            self._fit_to_canvas()
        else:
            self._redraw()

    def set_right(self, img_bgr: np.ndarray | None, marks=None):
        self._img_right = img_bgr
        self._marks_right = marks
        self._redraw()

    def set_images(self, left: np.ndarray | None,
                   right: np.ndarray | None):
        self._img_left  = left
        self._img_right = right
        if left is not None:
            self._fit_to_canvas()
        else:
            self._redraw()

    def _zoom_center(self, factor: float):
        size = self._ref_size()
        if size is None:
            return
        self.state.zoom_at_center(*size, factor)
        self.state.clamp_pan(*size)
        self._redraw()

    def zoom_in(self):
        self._zoom_center(ZoomState.ZOOM_STEP)

    def zoom_out(self):
        self._zoom_center(1.0 / ZoomState.ZOOM_STEP)

    def actual_size(self):
        """100 %: un píxel de la imagen = un píxel de pantalla."""
        self._zoom_center(1.0 / self.state.zoom)

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

    def pan_by(self, dx: int, dy: int):
        size = self._ref_size()
        if size is None:
            return
        self.state.pan(dx, dy, *size)
        self._redraw()

    # ── Fit interno ───────────────────────────────────────────────────────────

    def _fit_to_canvas(self):
        if self._img_left is None:
            self._redraw()
            return
        ref = self._ref()
        ref.update_idletasks()
        cw = max(ref.winfo_width(),  100)
        ch = max(ref.winfo_height(), 100)
        ih, iw = self._img_left.shape[:2]
        self.state.fit(cw, ch, iw, ih)
        self._redraw()

    # ── Renderizado ───────────────────────────────────────────────────────────

    def _redraw(self):
        left = self._overlay_left if self._overlay_left is not None else self._img_left
        size = None
        if left is not None and self._img_left is not None and left is not self._img_left:
            size = (self._img_left.shape[1], self._img_left.shape[0])
        self._draw_canvas(self.cl,  left,
                          "_photo_left",  left=True, size=size, marks=None)
        self._draw_canvas(self.cr,  self._right_aligned(),
                          "_photo_right", left=False, marks=self._marks_right)
        self._draw_minimap()
        if self._zoom_var is not None:
            self._zoom_var.set(f"{self.state.zoom_pct}%")
        if self.on_redraw:
            self.on_redraw()

    def _draw_canvas(self, canvas: tk.Canvas,
                     img_bgr: np.ndarray | None,
                     attr: str, left: bool, marks=None, size=None):
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(),  1)
        ch = max(canvas.winfo_height(), 1)
        canvas.delete("all")
        if img_bgr is None:
            return
        iw, ih = size or (img_bgr.shape[1], img_bgr.shape[0])
        try:
            post = None
            hl = self.highlight
            if marks is not None or hl is not None:
                from fenotit.core.pipeline import overlay

                def post(out, origin, z):
                    if marks is not None:
                        ov, colors, keep = marks
                        overlay.draw(out, ov, colors, origin, z, screen=True, keep=keep)
                    if hl is not None:
                        overlay.highlight(out, hl, origin, z)
            view = render(img_bgr, self.state, cw, ch, post, size)
            if view is not None:
                rgb, x, y = view
                photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                setattr(self, attr, photo)
                canvas.create_image(x, y, anchor="nw", image=photo)
        except Exception:
            _log.debug("Render falló", exc_info=True)
            return
        if left and hasattr(canvas, '_roi_selector_ref'):
            dw, dh = self.state.display_size(iw, ih)
            ox, oy = self.state.image_offset(cw, ch, iw, ih)
            canvas._roi_selector_ref.set_image_offset(ox, oy, dw, dh)

    # ── Minimapa ──────────────────────────────────────────────────────────────

    MINI = 110            # lado mayor del minimapa (px)

    def _draw_minimap(self):
        """Con zoom: proporción de la foto y rectángulo de lo que se ve, abajo a la derecha de
        la Entrada (o del Resultado si la Entrada está oculta). Sin la foto, semitransparente.
        Clic o arrastre en él = ir a ese lugar."""
        self._mini = None
        img = self._img_left if self._img_left is not None else self._img_right
        canvas = self.cl if self.cl.winfo_ismapped() else self.cr     # en la Entrada; si está oculta, en el Resultado
        for c in (self.cl, self.cr):
            c.delete("minimap")
        if img is None or not canvas.winfo_ismapped():
            return
        cw, ch = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        ih, iw = img.shape[:2]
        z = self.state.zoom
        if iw * z <= cw + 1 and ih * z <= ch + 1:
            return
        k = self.MINI / max(iw, ih)
        mw, mh = max(8, int(iw * k)), max(8, int(ih * k))
        x, y = cw - mw - 12, ch - mh - 12
        ox, oy = self.state.image_offset(cw, ch, iw, ih)
        vx0, vy0 = max(0.0, -ox / z), max(0.0, -oy / z)
        vx1, vy1 = min(iw, (cw - ox) / z), min(ih, (ch - oy) / z)
        tag = "minimap"
        canvas.create_rectangle(x - 1, y - 1, x + mw + 1, y + mh + 1, outline="#000000", width=1, tags=tag)
        canvas.create_rectangle(x, y, x + mw, y + mh, outline="#FFFFFF", fill="#FFFFFF", stipple="gray50",
                                width=1, tags=tag)
        rx0, ry0, rx1, ry1 = x + vx0 * k, y + vy0 * k, x + vx1 * k, y + vy1 * k
        canvas.create_rectangle(rx0, ry0, rx1, ry1, outline="#1F5FA8", fill="#1F5FA8", stipple="gray25",
                                width=2, tags=tag)
        self._mini = (canvas, x, y, mw, mh, k)

    def _in_minimap(self, event) -> bool:
        m = getattr(self, "_mini", None)
        if not m or event.widget is not m[0]:
            return False
        _, x, y, mw, mh, _ = m
        return x - 2 <= event.x <= x + mw + 2 and y - 2 <= event.y <= y + mh + 2

    def _minimap_goto(self, event):
        m, size = getattr(self, "_mini", None), self._ref_size()
        if not m or size is None:
            return
        canvas, x, y, mw, mh, k = m
        cw, ch, iw, ih = size
        px = min(max((event.x - x) / k, 0), iw)
        py = min(max((event.y - y) / k, 0), ih)
        z = self.state.zoom
        self.state.pan_x = cw / 2 - px * z - (cw - iw * z) / 2
        self.state.pan_y = ch / 2 - py * z - (ch - ih * z) / 2
        self.state.clamp_pan(*size)
        self._redraw()

    def _right_aligned(self) -> np.ndarray | None:
        """Imagen derecha llevada al tamaño de la izquierda para que el zoom coincida."""
        r, l = self._img_right, self._img_left
        if r is None or l is None or r.shape[:2] == l.shape[:2]:
            return r
        interp = cv2.INTER_NEAREST if r.shape[0] < l.shape[0] else cv2.INTER_AREA
        return cv2.resize(r, (l.shape[1], l.shape[0]), interpolation=interp)

    def redraw_with_overlay(self,
                            overlay_left: np.ndarray | None = None):
        """Muestra la izquierda con una superposición que se conserva al hacer zoom."""
        self._overlay_left = overlay_left
        self._redraw()


# ── ZoomableCanvas para ventanas de calibración ───────────────────────────────

class ZoomableCanvas(tk.Canvas):
    """Canvas con el mismo zoom que la ventana principal: rueda en el cursor, arrastrar
    para mover (izquierdo si se mueve más de unos píxeles; medio siempre), clic =
    `on_click(x, y)` en coordenadas de la imagen. `on_overlay(canvas)` dibuja marcas a
    tamaño de pantalla después de cada redibujo (usar `to_screen`)."""

    PAN_STEP = 40
    DRAG_PX = 4

    def __init__(self, parent, on_click=None, on_overlay=None, on_zoom=None, **kw):
        super().__init__(parent, **kw)
        self._img_bgr: np.ndarray | None = None
        self._photo   = None
        self._state   = ZoomState()
        self._fitted  = False
        self.on_click, self.on_overlay, self.on_zoom = on_click, on_overlay, on_zoom
        self._press = None
        self._moved = False

        self.config(takefocus=True)
        for key in ("<Left>", "<Right>", "<Up>", "<Down>"):
            self.bind(key, self._on_arrow)
        self.bind("<Configure>", lambda e=None: self._fit_if_needed())
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Shift-MouseWheel>", self._on_hwheel)
        self.bind("<Button-4>", self._on_wheel)
        self.bind("<Button-5>", self._on_wheel)
        for b in ("1", "2"):
            self.bind(f"<ButtonPress-{b}>", lambda e, b=b: self._down(e, b))
            self.bind(f"<B{b}-Motion>", self._drag)
            self.bind(f"<ButtonRelease-{b}>", lambda e, b=b: self._up(e, b))

    # coordenadas
    def _size(self):
        self.update_idletasks()
        ih, iw = self._img_bgr.shape[:2]
        return max(self.winfo_width(), 1), max(self.winfo_height(), 1), iw, ih

    def to_image(self, x, y):
        cw, ch, iw, ih = self._size()
        ox, oy = self._state.image_offset(cw, ch, iw, ih)
        return ((x - ox) / self._state.zoom, (y - oy) / self._state.zoom)

    def to_screen(self, x, y):
        cw, ch, iw, ih = self._size()
        ox, oy = self._state.image_offset(cw, ch, iw, ih)
        return (ox + x * self._state.zoom, oy + y * self._state.zoom)

    # mouse
    def _down(self, e, b):
        self.focus_set()
        self._press, self._moved = (e.x, e.y, b), False

    def _drag(self, e):
        if self._press is None or self._img_bgr is None:
            return
        x0, y0, _ = self._press
        if not self._moved and abs(e.x - x0) + abs(e.y - y0) < self.DRAG_PX:
            return
        self._moved = True
        self._state.pan(e.x - x0, e.y - y0, *self._size())
        self._press = (e.x, e.y, self._press[2])
        self._redraw()

    def _up(self, e, b):
        click = self._press is not None and not self._moved and b == "1"
        self._press = None
        if click and self._img_bgr is not None and self.on_click:
            x, y = self.to_image(e.x, e.y)
            ih, iw = self._img_bgr.shape[:2]
            if 0 <= x < iw and 0 <= y < ih:
                self.on_click(int(round(x)), int(round(y)))

    def _on_wheel(self, event):
        if self._img_bgr is None:
            return
        self._state.zoom_at(event.x, event.y, *self._size(), wheel_factor(event))
        self._state.clamp_pan(*self._size())
        self._redraw()

    def _on_hwheel(self, event):
        if self._img_bgr is None:
            return "break"
        self._state.pan(getattr(event, "delta", 0) / 120 * self.PAN_STEP, 0, *self._size())
        self._redraw()
        return "break"

    # API
    def set_image(self, img_bgr: np.ndarray | None, keep_view: bool = False):
        self._img_bgr = img_bgr
        if keep_view and self._fitted:
            self._redraw()
            return
        self._fitted = False
        self._fit_and_draw()

    def zoom_in(self):
        self._zoom(ZoomState.ZOOM_STEP)

    def zoom_out(self):
        self._zoom(1.0 / ZoomState.ZOOM_STEP)

    def actual_size(self):
        self._zoom(1.0 / self._state.zoom)

    def fit(self):
        self._fit_and_draw()

    def _zoom(self, factor: float):
        if self._img_bgr is None:
            return
        self._state.zoom_at_center(*self._size(), factor)
        self._state.clamp_pan(*self._size())
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
        step = self.PAN_STEP
        dx = {"Left": step, "Right": -step}.get(event.keysym, 0)
        dy = {"Up": step, "Down": -step}.get(event.keysym, 0)
        self._state.pan(dx, dy, *self._size())
        self._redraw()

    def _redraw(self):
        self.update_idletasks()
        cw = max(self.winfo_width(),  1)
        ch = max(self.winfo_height(), 1)
        self.delete("all")
        if self._img_bgr is None:
            return
        try:
            view = render(self._img_bgr, self._state, cw, ch)
            if view is not None:
                rgb, x, y = view
                self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.create_image(x, y, anchor="nw", image=self._photo)
        except Exception:
            _log.debug("ignorado", exc_info=True)
        if self.on_overlay:
            self.on_overlay(self)
        if self.on_zoom:
            self.on_zoom(self._state.zoom_pct)
