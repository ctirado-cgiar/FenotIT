"""Diálogo base y calibración de escala por dos puntos."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk

from fenotit.gui.theme import COLORS, FONTS
from fenotit.gui.zoom_controller import ZoomableCanvas

from fenotit import log
from fenotit.i18n import t

_log = log.get("gui.calibration")


def _assets() -> Path:
    return Path(__file__).parent.parent / "assets"


# ── Base dialog ───────────────────────────────────────────────────────────────



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
                 on_scale_set=None, loader=None, paths=None, unit: str = "mm", current=None):
        self._loader = loader
        self._current = current          # foto -> texto con la escala que ya tiene (o None)
        self._default_unit = unit
        self._paths = list(paths or ([current_path] if current_path else []))
        super().__init__(parent, t("menu.cal_scale").rstrip("…"))
        self._image      = current_image
        self._path       = current_path
        self._on_scale   = on_scale_set
        self._points     = []
        self._photo      = None
        self._scale_result = None
        self._display_scale = 1.0
        self._img_full = None
        self._build_body()
        self._show_current()
        if current_image is not None:
            self._load_image(current_image)

    def _build_body(self):
        b = self.body

        self._lbl(b,
            t("scale.help"),
            justify="left", wraplength=500).pack(anchor="w", pady=(0, 6))

        self._cur_var = tk.StringVar()
        self._cur_lbl = tk.Label(b, textvariable=self._cur_var, bg=COLORS["accent_light"], fg=COLORS["accent"],
                                 font=("Segoe UI", 9, "bold"), anchor="w", padx=8, pady=4)
        from fenotit.gui.widgets import ImagePicker
        self._picker = ImagePicker(b, self._paths, self._path, self._pick_image)
        self._picker.pack(anchor="w", pady=2)

        bar = tk.Frame(b, bg=COLORS["bg_card"])
        bar.pack(fill=tk.X, pady=2)
        self._canvas = ZoomableCanvas(b, bg=COLORS["bg_card"], highlightthickness=1,
                                      highlightbackground=COLORS["border"], cursor="crosshair",
                                      on_click=self._on_click, on_overlay=self._draw_points)
        from fenotit.gui.toolbar import zoom_bar
        zoom_bar(bar, self._canvas, COLORS)
        tk.Button(bar, text=f"✕ {t('scale.clear')}", command=self._clear_points,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"], relief="flat", font=FONTS["small"],
                  cursor="hand2").pack(side=tk.LEFT, padx=4)
        self._canvas.pack(fill=tk.BOTH, expand=True, pady=4)

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
        self._unit_var = tk.StringVar(value=self._default_unit)
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
        for key, scope, bold in (("scale.apply_all", "all", False), ("scale.apply_image", "image", True)):
            tk.Button(parent, text=t(key), command=lambda sc=scope: self._apply(sc),
                      bg=COLORS["accent"] if bold else COLORS["btn_bg"],
                      fg="#FFFFFF" if bold else COLORS["accent"], relief="flat",
                      font=("Segoe UI", 9, "bold") if bold else FONTS["body"],
                      cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)

    def _pick_image(self, path: str):
        try:
            if self._loader:
                img = self._loader(path)
            else:
                img = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            _log.exception("Error en calibración")
            messagebox.showerror(t("common.error"), str(e), parent=self)
            return
        if img is None:
            messagebox.showerror(t("common.error"), t("msg.image_unreadable", name=Path(path).name),
                                 parent=self)
            return
        self._image, self._path = img, path
        self._load_image(img)

    def _show_current(self):
        text = self._current(self._path) if self._current else None
        self._cur_var.set(text or "")
        if text:
            self._cur_lbl.pack(fill=tk.X, pady=(0, 6), before=self._picker)
        else:
            self._cur_lbl.pack_forget()

    def _load_image(self, img: np.ndarray):
        self._show_current()
        self._points   = []
        self._img_full = img.copy()
        self._scale_result = None
        self._result_var.set("—")
        self._pts_label.config(text=t("scale.step2"))
        # set_image hace fit-to-canvas automáticamente
        self._canvas.set_image(img)

    def _redraw(self):
        self._canvas._redraw()

    def _draw_points(self, c):
        """Puntos y línea a tamaño fijo de pantalla, con halo (se ven con cualquier zoom)."""
        pts = [c.to_screen(*p) for p in self._points]
        if len(pts) == 2:
            c.create_line(*pts[0], *pts[1], fill="#FFFFFF", width=5)
            c.create_line(*pts[0], *pts[1], fill="#0078DC", width=2)
        for i, (x, y) in enumerate(pts):
            fill = ("#0050DC", "#DCA000")[i]
            c.create_oval(x - 7, y - 7, x + 7, y + 7, fill=fill, outline="#FFFFFF", width=2)
            c.create_text(x + 16, y - 10, text=str(i + 1), fill="#000000", font=("Segoe UI", 11, "bold"))
            c.create_text(x + 15, y - 11, text=str(i + 1), fill="#FFFFFF", font=("Segoe UI", 11, "bold"))
        if len(pts) == 2 and self._scale_result:
            mx, my = (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
            label = f"{self._dist_var.get():.1f} {self._unit_var.get()}"
            tid = c.create_text(mx, my - 14, text=label, fill="#0050A0", font=("Segoe UI", 10, "bold"))
            x0, y0, x1, y1 = c.bbox(tid)
            c.tag_lower(c.create_rectangle(x0 - 4, y0 - 2, x1 + 4, y1 + 2, fill="#FFFFFF", outline=""), tid)

    def _on_click(self, x, y):
        if self._img_full is None:
            return
        if len(self._points) >= 2:
            self._points = []
            self._scale_result = None
            self._result_var.set("—")
        self._points.append((x, y))
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
            self._redraw()
        except Exception as e:
            _log.exception("Error calculando escala")
            self._result_var.set(f"{t('common.error')}: {e}")

    def _clear_points(self):
        self._points = []
        self._scale_result = None
        self._result_var.set("—")
        self._pts_label.config(text=t("scale.step2"))
        self._redraw()

    def _apply(self, scope: str = "image"):
        """scope = "image": solo la foto de la ventana; "all": todas las fotos."""
        if len(self._points) == 2:
            self._compute()
        if not self._scale_result:
            messagebox.showwarning(t("scale.none_title"),
                                   t("scale.need_points"),
                                   parent=self)
            return
        if self._on_scale and not self._on_scale(self._scale_result, scope, self._path):
            return
        self._log_var.set(f"✓ {t('status.scale', scale=self._scale_result.format())}")
        self._show_current()