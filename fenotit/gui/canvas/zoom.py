"""Zoom y desplazamiento de las imágenes (ventana principal y diálogos de calibración)."""

import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

# log escribe los mensajes y errores en el archivo de registro.
from fenotit import log
# Valores del tema (fenotit/assets/themes): COLORS (colores).
from fenotit.gui.style.theme import COLORS

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
        width, height = max(ix1 - ix0, 1.0), max(iy1 - iy0, 1.0)
        low, high = self._limits()
        self.zoom = max(low, min(high, min(canvas_w / width, canvas_h / height)))
        cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
        self.pan_x = canvas_w / 2 - cx * self.zoom - (canvas_w - img_w * self.zoom) / 2
        self.pan_y = canvas_h / 2 - cy * self.zoom - (canvas_h - img_h * self.zoom) / 2

    def zoom_at_center(self, canvas_w: int, canvas_h: int,
                       img_w: int, img_h: int, factor: float):
        """Zoom centrado en el centro del canvas."""
        low, high = self._limits()
        new_zoom = max(low, min(high, self.zoom * factor))
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

    def zoom_at(self, pixel_x: float, pixel_y: float, canvas_w: int, canvas_h: int,
                img_w: int, img_h: int, factor: float):
        """Zoom manteniendo fijo el punto (px, py) del canvas (p. ej. el cursor)."""
        low, high = self._limits()
        new_zoom = max(low, min(high, self.zoom * factor))
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        off_x = (canvas_w - img_w * self.zoom) / 2 + self.pan_x
        off_y = (canvas_h - img_h * self.zoom) / 2 + self.pan_y
        img_x, img_y = (pixel_x - off_x) / self.zoom, (pixel_y - off_y) / self.zoom
        self.pan_x = pixel_x - img_x * new_zoom - (canvas_w - img_w * new_zoom) / 2
        self.pan_y = pixel_y - img_y * new_zoom - (canvas_h - img_h * new_zoom) / 2
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


def render(img_bgr: np.ndarray, state: ZoomState, canvas_width: int, canvas_height: int, post=None, size=None):
    """Solo la parte visible, ya escalada: (rgb, x, y) o None. `size` = (ancho, alto) de la foto
    cuando `img_bgr` es una copia reducida (vista previa de fotos muy grandes)."""
    image_width, image_height = size or (img_bgr.shape[1], img_bgr.shape[0])
    z = state.zoom
    offset_x, offset_y = state.image_offset(canvas_width, canvas_height, image_width, image_height)
    x0, y0 = max(0, int(np.floor(-offset_x / z))), max(0, int(np.floor(-offset_y / z)))
    x1, y1 = min(image_width, int(np.ceil((canvas_width - offset_x) / z)) + 1), min(image_height, int(np.ceil((canvas_height - offset_y) / z)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    factor = img_bgr.shape[1] / image_width
    if factor != 1:
        crop = img_bgr[int(y0 * factor):max(int(y0 * factor) + 1, int(np.ceil(y1 * factor))),
                       int(x0 * factor):max(int(x0 * factor) + 1, int(np.ceil(x1 * factor)))]
    else:
        crop = img_bgr[y0:y1, x0:x1]
    display_width, display_height = max(1, int(round((x1 - x0) * z))), max(1, int(round((y1 - y0) * z)))
    interp = cv2.INTER_AREA if z < 1 else (cv2.INTER_NEAREST if z >= 4 else cv2.INTER_LINEAR)
    out = cv2.resize(crop, (display_width, display_height), interpolation=interp)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if post is not None:                      # marcas a tamaño de pantalla (overlay.draw)
        out = np.ascontiguousarray(out) if out is not crop else out.copy()
        post(out, (x0, y0), z)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), offset_x + int(round(x0 * z)), offset_y + int(round(y0 * z))


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

        for canvas_widget in (canvas_left, canvas_right):
            canvas_widget.config(takefocus=True)
            canvas_widget.bind("<MouseWheel>", self._on_wheel, add="+")
            canvas_widget.bind("<Shift-MouseWheel>", self._on_hwheel, add="+")
            canvas_widget.bind("<Button-4>", self._on_wheel, add="+")
            canvas_widget.bind("<Button-5>", self._on_wheel, add="+")
            for btn in ("1", "2"):
                canvas_widget.bind(f"<ButtonPress-{btn}>", lambda event, button_number=btn: self._press(event, button_number), add="+")
                canvas_widget.bind(f"<B{btn}-Motion>", lambda event, button_number=btn: self._motion(event, button_number), add="+")
                canvas_widget.bind(f"<ButtonRelease-{btn}>", lambda event, button_number=btn: self._release(event, button_number), add="+")
            canvas_widget.bind("<Enter>", lambda event: self._update_cursor(event.widget), add="+")

    # ── Herramientas ──────────────────────────────────────────────────────────

    def set_tool(self, tool: str):
        self.tool = tool
        for canvas_widget in (self.cl, self.cr):
            self._update_cursor(canvas_widget)
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
        canvas_width, canvas_height = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        offset_x, offset_y = self.state.image_offset(canvas_width, canvas_height, img.shape[1], img.shape[0])
        return (x - offset_x) / self.state.zoom, (y - offset_y) / self.state.zoom

    def _motion(self, event, button):
        if getattr(self, "_mini_drag", False):
            self._minimap_goto(event)
            return "break"
        if button == "1" and self._roi_drawing(event.widget):
            return
        if self._rect_start and button == "1":
            _, x0, y0 = self._rect_start
            for canvas_widget in (self.cl, self.cr):
                canvas_widget.delete("zoomrect")
                canvas_widget.create_rectangle(x0, y0, event.x, event.y, outline=COLORS["white"], width=1, tags="zoomrect")
                canvas_widget.create_rectangle(x0, y0, event.x, event.y, outline=COLORS["accent_dark"], dash=(4, 3), tags="zoomrect")
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
            point = self._to_image(event.widget, event.x, event.y)
            if point is not None:
                self.on_pick(*point)
            return
        if getattr(self, "_mini_drag", False):
            self._mini_drag = False
            return "break"
        if not (self._rect_start and button == "1"):
            return
        _, x0, y0 = self._rect_start
        self._rect_start = None
        for canvas_widget in (self.cl, self.cr):
            canvas_widget.delete("zoomrect")
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
        canvas_width = max(self.cl.winfo_width(),  1)
        canvas_height = max(self.cl.winfo_height(), 1)
        image_height, image_width = self._img_left.shape[:2]
        display_width, display_height = self.state.display_size(image_width, image_height)
        offset_x, offset_y = self.state.image_offset(canvas_width, canvas_height, image_width, image_height)
        return offset_x, offset_y, display_width, display_height

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
        canvas_width = max(ref.winfo_width(),  100)
        canvas_height = max(ref.winfo_height(), 100)
        image_height, image_width = self._img_left.shape[:2]
        self.state.fit(canvas_width, canvas_height, image_width, image_height)
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
        canvas_width = max(canvas.winfo_width(),  1)
        canvas_height = max(canvas.winfo_height(), 1)
        canvas.delete("all")
        if img_bgr is None:
            return
        image_width, image_height = size or (img_bgr.shape[1], img_bgr.shape[0])
        try:
            post = None
            highlighted = self.highlight
            if marks is not None or highlighted is not None:
                from fenotit.core.pipeline import overlay

                def post(out, origin, z):
                    if marks is not None:
                        drawn, colors, keep = marks
                        overlay.draw(out, drawn, colors, origin, z, screen=True, keep=keep)
                    if highlighted is not None:
                        overlay.highlight(out, highlighted, origin, z)
            view = render(img_bgr, self.state, canvas_width, canvas_height, post, size)
            if view is not None:
                rgb, x, y = view
                photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                setattr(self, attr, photo)
                canvas.create_image(x, y, anchor="nw", image=photo)
        except Exception:
            _log.debug("Render falló", exc_info=True)
            return
        if left and hasattr(canvas, '_roi_selector_ref'):
            display_width, display_height = self.state.display_size(image_width, image_height)
            offset_x, offset_y = self.state.image_offset(canvas_width, canvas_height, image_width, image_height)
            canvas._roi_selector_ref.set_image_offset(offset_x, offset_y, display_width, display_height)

    # ── Minimapa ──────────────────────────────────────────────────────────────

    MINI = 110            # lado mayor del minimapa (px)

    def _draw_minimap(self):
        """Con zoom: proporción de la foto y rectángulo de lo que se ve, abajo a la derecha de
        la Entrada (o del Resultado si la Entrada está oculta). Sin la foto, semitransparente.
        Clic o arrastre en él = ir a ese lugar."""
        self._mini = None
        img = self._img_left if self._img_left is not None else self._img_right
        canvas = self.cl if self.cl.winfo_ismapped() else self.cr     # en la Entrada; si está oculta, en el Resultado
        for canvas_widget in (self.cl, self.cr):
            canvas_widget.delete("minimap")
        if img is None or not canvas.winfo_ismapped():
            return
        canvas_width, canvas_height = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        image_height, image_width = img.shape[:2]
        z = self.state.zoom
        if image_width * z <= canvas_width + 1 and image_height * z <= canvas_height + 1:
            return
        factor = self.MINI / max(image_width, image_height)
        mini_width, mini_height = max(8, int(image_width * factor)), max(8, int(image_height * factor))
        x, y = canvas_width - mini_width - 12, canvas_height - mini_height - 12
        offset_x, offset_y = self.state.image_offset(canvas_width, canvas_height, image_width, image_height)
        vx0, vy0 = max(0.0, -offset_x / z), max(0.0, -offset_y / z)
        vx1, vy1 = min(image_width, (canvas_width - offset_x) / z), min(image_height, (canvas_height - offset_y) / z)
        tag = "minimap"
        canvas.create_rectangle(x - 1, y - 1, x + mini_width + 1, y + mini_height + 1, outline=COLORS["black"], width=1, tags=tag)
        canvas.create_rectangle(x, y, x + mini_width, y + mini_height, outline=COLORS["white"], fill=COLORS["white"], stipple="gray50",
                                width=1, tags=tag)
        rx0, ry0, rx1, ry1 = x + vx0 * factor, y + vy0 * factor, x + vx1 * factor, y + vy1 * factor
        canvas.create_rectangle(rx0, ry0, rx1, ry1, outline=COLORS["accent_dark"], fill=COLORS["accent_dark"], stipple="gray25",
                                width=2, tags=tag)
        self._mini = (canvas, x, y, mini_width, mini_height, factor)

    def _in_minimap(self, event) -> bool:
        minimap = getattr(self, "_mini", None)
        if not minimap or event.widget is not minimap[0]:
            return False
        _, x, y, mini_width, mini_height, _ = minimap
        return x - 2 <= event.x <= x + mini_width + 2 and y - 2 <= event.y <= y + mini_height + 2

    def _minimap_goto(self, event):
        minimap, size = getattr(self, "_mini", None), self._ref_size()
        if not minimap or size is None:
            return
        canvas, x, y, mini_width, mini_height, factor = minimap
        canvas_width, canvas_height, image_width, image_height = size
        pixel_x = min(max((event.x - x) / factor, 0), image_width)
        pixel_y = min(max((event.y - y) / factor, 0), image_height)
        z = self.state.zoom
        self.state.pan_x = canvas_width / 2 - pixel_x * z - (canvas_width - image_width * z) / 2
        self.state.pan_y = canvas_height / 2 - pixel_y * z - (canvas_height - image_height * z) / 2
        self.state.clamp_pan(*size)
        self._redraw()

    def _right_aligned(self) -> np.ndarray | None:
        """Imagen derecha llevada al tamaño de la izquierda para que el zoom coincida."""
        right_image, left_image = self._img_right, self._img_left
        if right_image is None or left_image is None or right_image.shape[:2] == left_image.shape[:2]:
            return right_image
        interp = cv2.INTER_NEAREST if right_image.shape[0] < left_image.shape[0] else cv2.INTER_AREA
        return cv2.resize(right_image, (left_image.shape[1], left_image.shape[0]), interpolation=interp)

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

    def __init__(self, parent, on_click=None, on_overlay=None, on_zoom=None, **kwargs):
        super().__init__(parent, **kwargs)
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
        self.bind("<Configure>", lambda event=None: self._fit_if_needed())
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Shift-MouseWheel>", self._on_hwheel)
        self.bind("<Button-4>", self._on_wheel)
        self.bind("<Button-5>", self._on_wheel)
        for button_number in ("1", "2"):
            self.bind(f"<ButtonPress-{button_number}>", lambda event, button_number=button_number: self._down(event, button_number))
            self.bind(f"<B{button_number}-Motion>", self._drag)
            self.bind(f"<ButtonRelease-{button_number}>", lambda event, button_number=button_number: self._up(event, button_number))

    @property
    def image(self) -> np.ndarray | None:
        """La imagen que se muestra (BGR)."""
        return self._img_bgr

    @property
    def zoom(self) -> float:
        """Píxeles de pantalla por píxel de la imagen."""
        return self._state.zoom

    # coordenadas
    def _size(self):
        self.update_idletasks()
        image_height, image_width = self._img_bgr.shape[:2]
        return max(self.winfo_width(), 1), max(self.winfo_height(), 1), image_width, image_height

    def to_image(self, x, y):
        canvas_width, canvas_height, image_width, image_height = self._size()
        offset_x, offset_y = self._state.image_offset(canvas_width, canvas_height, image_width, image_height)
        return ((x - offset_x) / self._state.zoom, (y - offset_y) / self._state.zoom)

    def to_screen(self, x, y):
        canvas_width, canvas_height, image_width, image_height = self._size()
        offset_x, offset_y = self._state.image_offset(canvas_width, canvas_height, image_width, image_height)
        return (offset_x + x * self._state.zoom, offset_y + y * self._state.zoom)

    # mouse
    def _down(self, event, button_number):
        self.focus_set()
        self._press, self._moved = (event.x, event.y, button_number), False

    def _drag(self, event):
        if self._press is None or self._img_bgr is None:
            return
        x0, y0, _ = self._press
        if not self._moved and abs(event.x - x0) + abs(event.y - y0) < self.DRAG_PX:
            return
        self._moved = True
        self._state.pan(event.x - x0, event.y - y0, *self._size())
        self._press = (event.x, event.y, self._press[2])
        self._redraw()

    def _up(self, event, button_number):
        click = self._press is not None and not self._moved and button_number == "1"
        self._press = None
        if click and self._img_bgr is not None and self.on_click:
            x, y = self.to_image(event.x, event.y)
            image_height, image_width = self._img_bgr.shape[:2]
            if 0 <= x < image_width and 0 <= y < image_height:
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
        canvas_width = max(self.winfo_width(),  10)
        canvas_height = max(self.winfo_height(), 10)
        image_height, image_width = self._img_bgr.shape[:2]
        self._state.fit(canvas_width, canvas_height, image_width, image_height)
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
        canvas_width = max(self.winfo_width(),  1)
        canvas_height = max(self.winfo_height(), 1)
        self.delete("all")
        if self._img_bgr is None:
            return
        try:
            view = render(self._img_bgr, self._state, canvas_width, canvas_height)
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
