"""Diálogo base y calibración de escala por dos puntos."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk

from fenotit.gui.theme import COLORS, FONTS
from fenotit.gui.zoom_controller import ZoomController, ZoomState

from fenotit import log
from fenotit.i18n import t

_log = log.get("gui.calibration")


def _assets() -> Path:
    return Path(__file__).parent.parent / "assets"


# ── Base dialog ───────────────────────────────────────────────────────────────


class ZoomableCanvas(tk.Canvas):
    """
    Canvas con zoom/pan propio para ventanas de calibración.
    Responsive: se adapta al tamaño de la ventana.
    """
    ZOOM_FACTOR = 1.25

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._img_bgr:  np.ndarray | None = None
        self._photo     = None
        self._state     = ZoomState()
        self._drag_sx   = 0
        self._drag_sy   = 0
        self._drag_px   = 0.0
        self._drag_py   = 0.0

        self.bind("<MouseWheel>",      self._on_wheel)
        self.bind("<Button-4>",        self._on_wheel)
        self.bind("<Button-5>",        self._on_wheel)
        self.bind("<ButtonPress-2>",   self._on_drag_start)
        self.bind("<B2-Motion>",       self._on_drag_move)
        self.bind("<ButtonRelease-2>", self._on_drag_end)
        self.bind("<Control-ButtonPress-1>",   self._on_drag_start)
        self.bind("<Control-B1-Motion>",       self._on_drag_move)
        self.bind("<Control-ButtonRelease-1>", self._on_drag_end)
        self.bind("<Double-Button-2>", lambda e: self.reset())
        self.bind("<Configure>",       lambda e: self._redraw())

    def set_image(self, img_bgr: np.ndarray | None):
        self._img_bgr = img_bgr
        self._state.reset()
        self._redraw()

    def reset(self):
        self._state.reset()
        self._redraw()

    def _on_wheel(self, event):
        if self._img_bgr is None:
            return
        factor = self.ZOOM_FACTOR if (event.num == 4 or event.delta > 0) \
                 else 1.0 / self.ZOOM_FACTOR
        cw = max(self.winfo_width(),  1)
        ch = max(self.winfo_height(), 1)
        ih, iw = self._img_bgr.shape[:2]
        self._state.zoom_at(event.x, event.y, cw, ch, iw, ih, factor)
        self._state.clamp_pan(cw, ch, iw, ih)
        self._redraw()

    def _on_drag_start(self, event):
        self._drag_sx = event.x
        self._drag_sy = event.y
        self._drag_px = self._state.pan_x
        self._drag_py = self._state.pan_y
        self.config(cursor="fleur")

    def _on_drag_move(self, event):
        self._state.pan_x = self._drag_px + (event.x - self._drag_sx)
        self._state.pan_y = self._drag_py + (event.y - self._drag_sy)
        if self._img_bgr is not None:
            cw = max(self.winfo_width(),  1)
            ch = max(self.winfo_height(), 1)
            ih, iw = self._img_bgr.shape[:2]
            self._state.clamp_pan(cw, ch, iw, ih)
        self._redraw()

    def _on_drag_end(self, event):
        self.config(cursor="")

    def _redraw(self):
        self.update_idletasks()
        cw = max(self.winfo_width(),  1)
        ch = max(self.winfo_height(), 1)
        self.delete("all")
        if self._img_bgr is None:
            return
        ih, iw = self._img_bgr.shape[:2]
        dw, dh = self._state.display_size(iw, ih)
        dw, dh = max(1, dw), max(1, dh)
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


class BaseDialog(tk.Toplevel):
    W, H = 560, 480

    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=COLORS["bg_card"])
        self.resizable(True, True)
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(
            f"{self.W}x{self.H}+{(sw-self.W)//2}+{(sh-self.H)//2}")
        try:
            self.iconbitmap(str(_assets() / "logo.ico"))
        except Exception:
            _log.debug("ignorado", exc_info=True)
        self.grab_set()

        # Banda azul superior
        tk.Frame(self, bg=COLORS["accent"], height=4).pack(fill=tk.X)

        # Header
        hdr = tk.Frame(self, bg=COLORS["bg_card"])
        hdr.pack(fill=tk.X, padx=20, pady=(12, 0))
        tk.Label(hdr, text=title,
                 bg=COLORS["bg_card"], fg=COLORS["accent"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")

        tk.Frame(self, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, pady=8)

        # Cuerpo (lo rellena cada subclase)
        self.body = tk.Frame(self, bg=COLORS["bg_card"])
        self.body.pack(fill=tk.BOTH, expand=True, padx=20, pady=0)

        # Barra de progreso + log
        self._build_progress()

        # Botones de acción
        btn_row = tk.Frame(self, bg=COLORS["bg_card"])
        btn_row.pack(fill=tk.X, padx=20, pady=(8, 12))
        tk.Frame(self, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, before=btn_row)
        self._build_buttons(btn_row)

    def _build_progress(self):
        self._prog_frame = tk.Frame(self, bg=COLORS["bg_card"])
        self._prog_frame.pack(fill=tk.X, padx=20)

        self._prog_var  = tk.DoubleVar(value=0)
        self._prog_bar  = ttk.Progressbar(
            self._prog_frame, variable=self._prog_var,
            maximum=100, length=300)
        self._prog_bar.pack(fill=tk.X)
        self._prog_bar.pack_forget()   # oculta hasta que se use

        self._log_var = tk.StringVar(value="")
        tk.Label(self._prog_frame, textvariable=self._log_var,
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["small"], anchor="w").pack(fill=tk.X)

    def _build_buttons(self, parent):
        """Sobreescribir para agregar botones."""
        tk.Button(parent, text=t("common.close"), command=self.destroy,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=12).pack(side=tk.RIGHT)

    def _show_progress(self, current: int, total: int, msg: str = ""):
        self._prog_bar.pack(fill=tk.X)
        pct = int(current / total * 100) if total > 0 else 0
        self._prog_var.set(pct)
        self._log_var.set(msg)
        self.update_idletasks()

    def _hide_progress(self):
        self._prog_bar.pack_forget()

    def _lbl(self, parent, text, **kw):
        return tk.Label(parent, text=text,
                        bg=COLORS["bg_card"], fg=COLORS["text"],
                        font=FONTS["body"], **kw)

    def _entry_row(self, parent, label, var, width=8):
        f = tk.Frame(parent, bg=COLORS["bg_card"])
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label, bg=COLORS["bg_card"],
                 fg=COLORS["text"], font=FONTS["body"],
                 width=22, anchor="w").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var, width=width,
                 bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1,
                 font=FONTS["mono"]).pack(side=tk.LEFT, padx=4)
        return f


# ── Calibración de escala ─────────────────────────────────────────────────────

class ScaleDialog(BaseDialog):
    W, H = 760, 640
    RESIZABLE = True

    def __init__(self, parent,
                 current_image: np.ndarray | None = None,
                 current_path: str | None = None,
                 on_scale_set=None, loader=None):
        self._loader = loader
        super().__init__(parent, t("menu.cal_scale"))
        self._image      = current_image
        self._path       = current_path
        self._on_scale   = on_scale_set
        self._points     = []
        self._photo      = None
        self._scale_result = None
        self._display_scale = 1.0
        self._build_body()
        if current_image is not None:
            self._load_image(current_image)

    def _build_body(self):
        b = self.body

        self._lbl(b,
            t("scale.help"),
            justify="left", wraplength=500).pack(anchor="w", pady=(0, 6))

        # Fuente de imagen + botones de zoom
        src_f = tk.Frame(b, bg=COLORS["bg_card"])
        src_f.pack(fill=tk.X, pady=2)
        tk.Label(src_f, text=t("scale.image"),
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"], width=8, anchor="w").pack(side=tk.LEFT)
        self._img_path_var = tk.StringVar(value=self._path or "")
        tk.Entry(src_f, textvariable=self._img_path_var, width=22,
                 bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1,
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=4)
        tk.Button(src_f, text="…", command=self._pick_image,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", cursor="hand2",
                  font=FONTS["body"]).pack(side=tk.LEFT)
        if self._image is not None:
            tk.Button(src_f, text=t("scale.current_image"),
                      command=lambda: self._load_image(self._image),
                      bg=COLORS["btn_bg"], fg=COLORS["accent2"],
                      relief="flat", cursor="hand2",
                      font=FONTS["small"]).pack(side=tk.LEFT, padx=4)

        # Botones zoom para el canvas
        zoom_f = tk.Frame(b, bg=COLORS["bg_card"])
        zoom_f.pack(fill=tk.X, pady=2)
        def _zi(): self._canvas.zoom_in();  self._canvas.focus_set()
        def _zo(): self._canvas.zoom_out(); self._canvas.focus_set()
        def _zf(): self._canvas.fit();      self._canvas.focus_set()

        for txt, cmd, w in [
            ("+",  _zi, 2),
            ("−",  _zo, 2),
            ("⊡",  _zf, 2),
            (f"✕ {t('scale.clear')}", self._clear_points, 10),
        ]:
            tk.Button(zoom_f, text=txt, command=cmd,
                      bg=COLORS["btn_bg"], fg=COLORS["accent"],
                      relief="flat",
                      font=("Segoe UI", 9, "bold") if w==2 else FONTS["small"],
                      width=w, cursor="hand2").pack(side=tk.LEFT, padx=1)

        # Canvas interactivo — altura generosa, fill=BOTH
        self._canvas = ZoomableCanvas(b, bg=COLORS["bg_card"],
                                      highlightthickness=1,
                                      highlightbackground=COLORS["border"],
                                      cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True, pady=4)
        self._canvas.bind("<Button-1>", self._on_click, add="+")

        self._pts_label = tk.Label(
            b, text=t("scale.step2"),
            bg=COLORS["bg_card"], fg=COLORS["text_muted"],
            font=FONTS["small"], anchor="w")
        self._pts_label.pack(fill=tk.X)

        # Paso 3: distancia real + unidad (DESPUÉS de marcar puntos)
        step3 = tk.Frame(b, bg=COLORS["bg_card"])
        step3.pack(fill=tk.X, pady=(6,2))
        tk.Label(step3,
                 text=t("scale.step3"),
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"]).pack(side=tk.LEFT)
        self._dist_var = tk.DoubleVar(value=10.0)
        tk.Entry(step3, textvariable=self._dist_var, width=8,
                 bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1,
                 font=FONTS["mono"]).pack(side=tk.LEFT, padx=4)

        tk.Label(step3, text=t("scale.unit"),
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"]).pack(side=tk.LEFT, padx=(12, 2))
        self._unit_var = tk.StringVar(value="mm")
        from fenotit.core.corrections.scale import UNITS
        ttk.Combobox(step3, textvariable=self._unit_var,
                     values=UNITS, state="readonly", width=4,
                     font=FONTS["body"]).pack(side=tk.LEFT)

        tk.Button(step3, text=t("scale.compute"),
                  command=self._compute,
                  bg=COLORS["accent"], fg="#FFFFFF",
                  relief="flat", font=FONTS["small"],
                  cursor="hand2", padx=8).pack(side=tk.LEFT, padx=8)

        # Resultado
        self._result_var = tk.StringVar(value="—")
        tk.Label(b, textvariable=self._result_var,
                 bg=COLORS["bg_card"], fg=COLORS["accent"],
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(fill=tk.X)

    def _build_buttons(self, parent):
        tk.Button(parent, text=t("common.close"), command=self.destroy,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)
        tk.Button(parent, text=t("scale.clear_points"),
                  command=self._clear_points,
                  bg=COLORS["btn_bg"], fg=COLORS["text_muted"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)
        tk.Button(parent, text=t("scale.apply"),
                  command=self._apply,
                  bg=COLORS["accent"], fg="#FFFFFF",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title=t("scale.ref_image"),
            filetypes=[(t("common.images"),"*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       (t("common.all_files"),"*.*")],
            parent=self)
        if not path:
            return
        self._img_path_var.set(path)
        try:
            if self._loader:
                img = self._loader(path)
            else:
                img = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self._image = img
                self._load_image(img)
            else:
                messagebox.showerror(
                    t("common.error"), t("msg.image_unreadable", name=Path(path).name),
                    parent=self)
        except Exception as e:
            _log.exception("Error en calibración")
            messagebox.showerror(t("common.error"), str(e), parent=self)

    def _load_image(self, img: np.ndarray):
        self._points   = []
        self._img_full = img.copy()
        self._scale_result = None
        self._result_var.set("—")
        self._pts_label.config(text=t("scale.step2"))
        # set_image hace fit-to-canvas automáticamente
        self._canvas.set_image(img)

    def _redraw(self):
        if self._img_full is None:
            return
        img = self._img_full.copy()

        if len(self._points) >= 1:
            # Punto 1 — círculo azul visible
            cv2.circle(img, self._points[0], 10, (220, 80, 0), -1)
            cv2.circle(img, self._points[0], 12, (255, 255, 255), 2)
            cv2.putText(img, "1", 
                       (self._points[0][0]+14, self._points[0][1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (220, 80, 0), 2, cv2.LINE_AA)

        if len(self._points) == 2:
            # Punto 2
            cv2.circle(img, self._points[1], 10, (0, 160, 220), -1)
            cv2.circle(img, self._points[1], 12, (255, 255, 255), 2)
            cv2.putText(img, "2",
                       (self._points[1][0]+14, self._points[1][1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 160, 220), 2, cv2.LINE_AA)
            # Línea entre puntos
            cv2.line(img, self._points[0], self._points[1],
                     (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(img, self._points[0], self._points[1],
                     (0, 120, 220), 2, cv2.LINE_AA)
            # Etiqueta si hay resultado
            if self._scale_result:
                mid_x = (self._points[0][0] + self._points[1][0]) // 2
                mid_y = (self._points[0][1] + self._points[1][1]) // 2
                label = (f"{self._dist_var.get():.1f} "
                         f"{self._unit_var.get()}")
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img,
                    (mid_x - tw//2 - 4, mid_y - th - 8),
                    (mid_x + tw//2 + 4, mid_y + 4),
                    (255, 255, 255), -1)
                cv2.putText(img, label,
                    (mid_x - tw//2, mid_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 120, 220), 2, cv2.LINE_AA)
        # ZoomableCanvas preserva zoom/pan al actualizar imagen
        self._canvas._img_bgr = img
        self._canvas._redraw()

    def _on_click(self, event):
        if self._img_full is None:
            return
        if len(self._points) >= 2:
            self._points = []
            self._scale_result = None
            self._result_var.set("—")

        # Convertir coords canvas → imagen original via ZoomableCanvas state
        state = self._canvas._state
        ih, iw = self._img_full.shape[:2]
        cw = max(self._canvas.winfo_width(), 1)
        ch = max(self._canvas.winfo_height(), 1)
        ox, oy = state.image_offset(cw, ch, iw, ih)
        # Coord en imagen original
        x = int((event.x - ox) / state.zoom)
        y = int((event.y - oy) / state.zoom)
        x = max(0, min(x, iw - 1))
        y = max(0, min(y, ih - 1))
        self._points.append((x, y))
        # Dar foco al canvas para que funcionen las flechas
        self._canvas.focus_set()

        if len(self._points) == 2:
            p1, p2 = self._points
            dist_px = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
            self._pts_label.config(
                text=t("scale.two_points", p1=p1, p2=p2, d=f"{dist_px:.1f}"))
        else:
            self._pts_label.config(text=t("scale.one_point", p1=self._points[0]))

        self._redraw()

    def _compute(self):
        if len(self._points) != 2:
            return
        from fenotit.core.corrections.scale import compute_scale
        try:
            # Puntos ya están en coordenadas de imagen original
            result = compute_scale(
                self._points[0], self._points[1],
                self._dist_var.get(),
                self._unit_var.get())
            self._scale_result = result
            self._result_var.set(result.format())
        except Exception as e:
            _log.exception("Error calculando escala")
            self._result_var.set(f"{t('common.error')}: {e}")

    def _clear_points(self):
        self._points = []
        self._scale_result = None
        self._result_var.set("—")
        self._pts_label.config(text=t("scale.step2"))
        self._redraw()

    def _apply(self):
        # Calcular primero si no se ha hecho
        if len(self._points) == 2:
            self._compute()
        if not self._scale_result:
            messagebox.showwarning(t("scale.none_title"),
                                   t("scale.need_points"),
                                   parent=self)
            return
        if self._on_scale:
            self._on_scale(self._scale_result)
        self._log_var.set(f"✓ {t('status.scale', scale=self._scale_result.format())}")
        messagebox.showinfo(
            t("scale.set_title"),
            t("status.scale", scale=self._scale_result.format()),
            parent=self)