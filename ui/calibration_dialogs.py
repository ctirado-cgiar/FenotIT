"""
ui/calibration_dialogs.py
Ventanas de calibración para FenotIT.
  - DistortionDialog      (calibración óptica — ajedrez)
  - ColorCardDialog       (corrección de color — color card)
  - ArucoDialog           (corrección de perspectiva — ArUcos)
  - ScaleDialog           (calibración de escala — puntos en imagen)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk

from ui.theme import COLORS, FONTS
from ui.zoom_controller import ZoomController, ZoomState


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
            pass


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
            pass
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
        tk.Button(parent, text="Cerrar", command=self.destroy,
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


# ── Calibración óptica ────────────────────────────────────────────────────────

class DistortionDialog(BaseDialog):
    W, H = 580, 500

    def __init__(self, parent):
        super().__init__(parent, "Calibración óptica — Tablero de ajedrez")
        self._params_path = None
        self._build_body()

    def _build_body(self):
        b = self.body

        self._lbl(b,
            "1. Selecciona la carpeta con las fotos del tablero de ajedrez.\n"
            "2. Configura las esquinas interiores del tablero.\n"
            "3. Calibra y luego aplica la corrección al lote de imágenes.",
            justify="left", wraplength=500).pack(anchor="w", pady=(0, 10))

        # Carpeta de tablero
        row1 = tk.Frame(b, bg=COLORS["bg_card"])
        row1.pack(fill=tk.X, pady=2)
        tk.Label(row1, text="Carpeta tablero:",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"], width=18, anchor="w").pack(side=tk.LEFT)
        self._chess_folder = tk.StringVar()
        tk.Entry(row1, textvariable=self._chess_folder, width=30,
                 bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1,
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=4)
        tk.Button(row1, text="…", command=self._pick_chess,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", cursor="hand2",
                  font=FONTS["body"]).pack(side=tk.LEFT)

        # Configuración del tablero
        cfg = tk.Frame(b, bg=COLORS["bg_card"])
        cfg.pack(fill=tk.X, pady=6)
        tk.Label(cfg, text="Esquinas interiores:",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"]).pack(side=tk.LEFT)

        self._cols = tk.IntVar(value=7)
        self._rows = tk.IntVar(value=6)

        for label, var in [("Columnas", self._cols), ("Filas", self._rows)]:
            tk.Label(cfg, text=f"  {label}:",
                     bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                     font=FONTS["small"]).pack(side=tk.LEFT)
            tk.Spinbox(cfg, textvariable=var,
                       from_=3, to=20, width=4,
                       bg=COLORS["bg_panel"], fg=COLORS["text"],
                       relief="solid", bd=1,
                       font=FONTS["mono"]).pack(side=tk.LEFT, padx=2)

        # Carpeta de imágenes a corregir
        row2 = tk.Frame(b, bg=COLORS["bg_card"])
        row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text="Imágenes a corregir:",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"], width=18, anchor="w").pack(side=tk.LEFT)
        self._input_folder = tk.StringVar()
        tk.Entry(row2, textvariable=self._input_folder, width=30,
                 bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1,
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=4)
        tk.Button(row2, text="…", command=self._pick_input,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", cursor="hand2",
                  font=FONTS["body"]).pack(side=tk.LEFT)

        # Info salida
        self._out_label = tk.Label(
            b, text="Salida: se creará carpeta noDistorcion/ automáticamente",
            bg=COLORS["bg_card"], fg=COLORS["text_muted"],
            font=FONTS["small"])
        self._out_label.pack(anchor="w", pady=4)

    def _build_buttons(self, parent):
        tk.Button(parent, text="Cerrar", command=self.destroy,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=12).pack(side=tk.RIGHT, padx=4)
        tk.Button(parent, text="2. Aplicar corrección",
                  command=self._apply,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=12).pack(side=tk.RIGHT, padx=4)
        tk.Button(parent, text="1. Calibrar",
                  command=self._calibrate,
                  bg=COLORS["accent"], fg="#FFFFFF",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  cursor="hand2", padx=12).pack(side=tk.RIGHT, padx=4)

    def _pick_chess(self):
        d = filedialog.askdirectory(title="Carpeta con fotos del tablero")
        if d:
            self._chess_folder.set(d)

    def _pick_input(self):
        d = filedialog.askdirectory(title="Carpeta con imágenes a corregir")
        if d:
            self._input_folder.set(d)

    def _calibrate(self):
        folder = self._chess_folder.get()
        if not folder:
            messagebox.showwarning("Sin carpeta",
                                   "Selecciona la carpeta del tablero.",
                                   parent=self)
            return

        from corrections.distortion import calibrate_from_folder
        self._log_var.set("Calibrando…")
        self.update_idletasks()

        def worker():
            result = calibrate_from_folder(
                folder,
                inner_cols=self._cols.get(),
                inner_rows=self._rows.get(),
                progress_cb=lambda c, t, n:
                    self.after(0, lambda: self._show_progress(c, t, n)),
            )
            self.after(0, lambda: self._on_calib_done(result))

        threading.Thread(target=worker, daemon=True).start()

    def _on_calib_done(self, result):
        self._hide_progress()
        if result.success:
            self._params_path = str(
                Path(self._chess_folder.get()).parent
                / "calibracionCamara" / "parametrosCorreccion"
                / "calibracion_params.npz")
            self._log_var.set(
                f"✓ {result.message}  |  "
                f"{result.n_images_used}/{result.n_images_total} imágenes usadas")
        else:
            self._log_var.set(f"✗ {result.message}")
            messagebox.showerror("Error de calibración",
                                 result.message, parent=self)

    def _apply(self):
        if not self._params_path:
            messagebox.showwarning(
                "Sin calibración",
                "Primero calibra con el paso 1.", parent=self)
            return
        folder = self._input_folder.get()
        if not folder:
            messagebox.showwarning(
                "Sin carpeta",
                "Selecciona la carpeta con imágenes a corregir.", parent=self)
            return

        from corrections.distortion import apply_correction

        def worker():
            results = apply_correction(
                folder, self._params_path,
                progress_cb=lambda c, t, n:
                    self.after(0, lambda: self._show_progress(c, t, n)),
            )
            self.after(0, lambda: self._on_apply_done(results))

        threading.Thread(target=worker, daemon=True).start()

    def _on_apply_done(self, results):
        self._hide_progress()
        msg = (f"✓ {results['ok']} imágenes corregidas\n"
               f"Carpeta: {results['output_folder']}")
        if results["error"]:
            msg += f"\n⚠ {results['error']} errores"
        self._log_var.set(msg.replace("\n", "   "))
        messagebox.showinfo("Corrección completada", msg, parent=self)


# ── Calibración color card ────────────────────────────────────────────────────

class ColorCardDialog(BaseDialog):
    W, H = 580, 520

    def __init__(self, parent, current_image_path: str | None = None):
        super().__init__(parent, "Calibración de color — Color Card")
        self._mask      = None
        self._cur_path  = current_image_path
        self._build_body()

    def _build_body(self):
        b = self.body

        self._lbl(b,
            "1. Selecciona la imagen con la tarjeta de colores o usa la imagen actual.\n"
            "2. Detecta la tarjeta y verifica que la posición es correcta.\n"
            "3. Aplica la corrección a la imagen actual o a todo el lote.",
            justify="left", wraplength=500).pack(anchor="w", pady=(0, 8))

        # Fuente de imagen
        src_f = tk.Frame(b, bg=COLORS["bg_card"])
        src_f.pack(fill=tk.X, pady=2)
        tk.Label(src_f, text="Imagen con tarjeta:",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"], width=20, anchor="w").pack(side=tk.LEFT)
        self._card_path = tk.StringVar(
            value=self._cur_path or "")
        tk.Entry(src_f, textvariable=self._card_path, width=26,
                 bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1,
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=4)
        tk.Button(src_f, text="…", command=self._pick_card,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", cursor="hand2",
                  font=FONTS["body"]).pack(side=tk.LEFT)
        if self._cur_path:
            tk.Button(src_f, text="Usar imagen actual",
                      command=lambda: self._card_path.set(
                          self._cur_path or ""),
                      bg=COLORS["btn_bg"], fg=COLORS["accent2"],
                      relief="flat", cursor="hand2",
                      font=FONTS["small"]).pack(side=tk.LEFT, padx=4)

        # Radio del detector y posición
        cfg = tk.Frame(b, bg=COLORS["bg_card"])
        cfg.pack(fill=tk.X, pady=4)

        tk.Label(cfg, text="Radio detección:",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["small"]).pack(side=tk.LEFT)
        self._radius = tk.IntVar(value=6)
        tk.Spinbox(cfg, textvariable=self._radius,
                   from_=3, to=20, width=4,
                   bg=COLORS["bg_panel"], fg=COLORS["text"],
                   relief="solid", bd=1,
                   font=FONTS["mono"]).pack(side=tk.LEFT, padx=(2, 16))

        tk.Label(cfg, text="Posición tarjeta (pos):",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["small"]).pack(side=tk.LEFT)
        self._pos = tk.IntVar(value=3)
        tk.Spinbox(cfg, textvariable=self._pos,
                   from_=1, to=4, width=3,
                   bg=COLORS["bg_panel"], fg=COLORS["text"],
                   relief="solid", bd=1,
                   font=FONTS["mono"]).pack(side=tk.LEFT, padx=2)
        tk.Label(cfg,
                 text="  (1-4, cambia si la corrección falla)",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT)

        # Carpeta del lote
        row_lote = tk.Frame(b, bg=COLORS["bg_card"])
        row_lote.pack(fill=tk.X, pady=2)
        tk.Label(row_lote, text="Carpeta lote (opcional):",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"], width=20, anchor="w").pack(side=tk.LEFT)
        self._lote_folder = tk.StringVar()
        if self._cur_path:
            self._lote_folder.set(str(Path(self._cur_path).parent))
        tk.Entry(row_lote, textvariable=self._lote_folder, width=26,
                 bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1,
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=4)
        tk.Button(row_lote, text="…", command=self._pick_lote,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", cursor="hand2",
                  font=FONTS["body"]).pack(side=tk.LEFT)

        tk.Label(b,
                 text="Salida: colorCorrejidas/ (no sobreescribe originales)",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(anchor="w", pady=2)

        # Advertencia lote
        self._warn = tk.Label(
            b,
            text="⚠  El procesamiento por lote puede tardar varios minutos.",
            bg=COLORS["bg_card"], fg=COLORS["warning"],
            font=FONTS["small"])
        self._warn.pack(anchor="w")

    def _build_buttons(self, parent):
        tk.Button(parent, text="Cerrar", command=self.destroy,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)
        tk.Button(parent, text="3b. Aplicar a lote",
                  command=self._apply_batch,
                  bg=COLORS["btn_bg"], fg=COLORS["accent2"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)
        tk.Button(parent, text="3a. Aplicar a imagen actual",
                  command=self._apply_single,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)
        tk.Button(parent, text="2. Detectar tarjeta",
                  command=self._detect,
                  bg=COLORS["accent"], fg="#FFFFFF",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)

    def _pick_card(self):
        p = filedialog.askopenfilename(
            title="Imagen con tarjeta de colores",
            filetypes=[("Imágenes","*.jpg *.jpeg *.png *.bmp"),
                       ("Todos","*.*")],
            parent=self)
        if p:
            self._card_path.set(p)

    def _pick_lote(self):
        d = filedialog.askdirectory(
            title="Carpeta con imágenes a corregir", parent=self)
        if d:
            self._lote_folder.set(d)

    def _detect(self):
        path = self._card_path.get()
        if not path:
            messagebox.showwarning("Sin imagen",
                                   "Selecciona una imagen con la tarjeta.",
                                   parent=self)
            return
        self._log_var.set("Detectando tarjeta…")
        from corrections.color_card import detect_and_save_mask
        ok, msg, mask = detect_and_save_mask(
            path, radius=self._radius.get())
        if ok:
            self._mask = mask
            self._log_var.set(f"✓ {msg}")
        else:
            self._log_var.set(f"✗ {msg}")
            messagebox.showerror("Error", msg, parent=self)

    def _apply_single(self):
        if self._mask is None:
            messagebox.showwarning("Sin máscara",
                                   "Primero detecta la tarjeta (paso 2).",
                                   parent=self)
            return
        path = self._cur_path or self._card_path.get()
        if not path:
            messagebox.showwarning("Sin imagen", "No hay imagen activa.",
                                   parent=self)
            return
        out_folder = str(Path(path).parent / "colorCorrejidas")
        from corrections.color_card import correct_single
        ok, msg = correct_single(path, self._mask, out_folder,
                                 pos=self._pos.get())
        self._log_var.set(f"{'✓' if ok else '✗'}  {msg}")
        if ok:
            messagebox.showinfo("Listo", msg, parent=self)
        else:
            messagebox.showerror("Error", msg, parent=self)

    def _apply_batch(self):
        if self._mask is None:
            messagebox.showwarning("Sin máscara",
                                   "Primero detecta la tarjeta (paso 2).",
                                   parent=self)
            return
        folder = self._lote_folder.get()
        if not folder:
            messagebox.showwarning("Sin carpeta",
                                   "Selecciona la carpeta con imágenes.",
                                   parent=self)
            return
        if not messagebox.askyesno(
                "Confirmar lote",
                "¿Aplicar corrección de color a todas las imágenes "
                "de la carpeta?\nEsto puede tardar varios minutos.",
                parent=self):
            return

        from corrections.color_card import correct_batch

        def worker():
            results = correct_batch(
                folder, self._mask,
                pos=self._pos.get(),
                progress_cb=lambda c, t, n:
                    self.after(0, lambda: self._show_progress(c, t, n)),
            )
            self.after(0, lambda: self._on_batch_done(results))

        threading.Thread(target=worker, daemon=True).start()

    def _on_batch_done(self, results):
        self._hide_progress()
        msg = (f"✓ {results['ok']} imágenes corregidas\n"
               f"Carpeta: {results['output_folder']}")
        if results["error"]:
            msg += f"\n⚠ {results['error']} errores"
        self._log_var.set(msg.replace("\n", "   "))
        messagebox.showinfo("Lote completado", msg, parent=self)


# ── Calibración ArUco ─────────────────────────────────────────────────────────

class ArucoDialog(BaseDialog):
    W, H = 620, 520

    def __init__(self, parent, current_image: np.ndarray | None = None,
                 current_path: str | None = None,
                 batch_paths: list[str] | None = None):
        super().__init__(parent, "Calibración de perspectiva — ArUcos")
        self._current_image  = current_image
        self._current_path   = current_path
        self._batch_paths    = batch_paths or []
        self._preview_photo  = None
        self._build_body()
        if current_image is not None:
            self._show_preview(current_image)

    def _build_body(self):
        b = self.body

        self._lbl(b,
            "Detecta los 4 marcadores ArUco (DICT_4X4_50, IDs 0-3) en cada "
            "imagen y corrige la perspectiva.\n"
            "La imagen de salida mantiene el tamaño original.",
            justify="left", wraplength=500).pack(anchor="w", pady=(0, 8))

        # Margen
        cfg = tk.Frame(b, bg=COLORS["bg_card"])
        cfg.pack(fill=tk.X, pady=4)
        tk.Label(cfg, text="Margen interior (px):",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"]).pack(side=tk.LEFT)
        self._margin = tk.IntVar(value=20)
        tk.Spinbox(cfg, textvariable=self._margin,
                   from_=0, to=200, width=5,
                   bg=COLORS["bg_panel"], fg=COLORS["text"],
                   relief="solid", bd=1,
                   font=FONTS["mono"]).pack(side=tk.LEFT, padx=4)
        tk.Label(cfg,
                 text="  píxeles desde los centros de los marcadores",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT)

        # Canvas preview
        self._canvas = ZoomableCanvas(b, bg=COLORS["bg_card"],
                                      highlightthickness=1,
                                      highlightbackground=COLORS["border"],
                                      height=220)
        self._canvas.pack(fill=tk.X, pady=6)

        # Info detección
        self._detect_var = tk.StringVar(
            value="—  carga una imagen para verificar la detección")
        tk.Label(b, textvariable=self._detect_var,
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["small"], anchor="w").pack(fill=tk.X)

        # Salida
        row = tk.Frame(b, bg=COLORS["bg_card"])
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="Carpeta lote:",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"], width=14, anchor="w").pack(side=tk.LEFT)
        self._lote_var = tk.StringVar(
            value=str(Path(self._current_path).parent)
            if self._current_path else "")
        tk.Entry(row, textvariable=self._lote_var, width=28,
                 bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1,
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=4)
        tk.Button(row, text="…",
                  command=lambda: self._lote_var.set(
                      filedialog.askdirectory(parent=self) or
                      self._lote_var.get()),
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", cursor="hand2",
                  font=FONTS["body"]).pack(side=tk.LEFT)

        tk.Label(b,
                 text="Salida: perspectivaCorregida/ (no sobreescribe originales)",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(anchor="w", pady=2)

    def _build_buttons(self, parent):
        tk.Button(parent, text="Cerrar", command=self.destroy,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)
        tk.Button(parent, text="Aplicar a lote",
                  command=self._apply_batch,
                  bg=COLORS["btn_bg"], fg=COLORS["accent2"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)
        tk.Button(parent, text="Aplicar a imagen actual",
                  command=self._apply_single,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)
        tk.Button(parent, text="Verificar detección",
                  command=self._verify,
                  bg=COLORS["accent"], fg="#FFFFFF",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)

    def _show_preview(self, img: np.ndarray):
        from corrections.aruco import preview_detection
        prev = preview_detection(img)
        self._canvas.set_image(prev)   # ZoomableCanvas maneja el escalado

        from corrections.aruco import detect_aruco_corners
        found = detect_aruco_corners(img)
        n = len(found)
        if n == 4:
            self._detect_var.set("✓ 4 ArUcos detectados — listo para corregir")
        else:
            missing = [i for i in range(4) if i not in found]
            self._detect_var.set(
                f"⚠  {n}/4 ArUcos detectados — "
                f"IDs faltantes: {missing}")

    def _verify(self):
        if self._current_image is None:
            messagebox.showwarning("Sin imagen",
                                   "Carga una imagen primero.",
                                   parent=self)
            return
        self._show_preview(self._current_image)

    def _apply_single(self):
        if self._current_image is None:
            messagebox.showwarning("Sin imagen",
                                   "Carga una imagen primero.",
                                   parent=self)
            return
        from corrections.aruco import correct_perspective
        corrected, msg = correct_perspective(
            self._current_image, margin=self._margin.get())
        if corrected is None:
            messagebox.showerror("Error", msg, parent=self)
            return
        if self._current_path:
            out_folder = str(Path(self._current_path).parent
                             / "perspectivaCorregida")
            Path(out_folder).mkdir(parents=True, exist_ok=True)
            out_path = Path(out_folder) / Path(self._current_path).name
            ext = Path(self._current_path).suffix.lower()
            ok, buf = cv2.imencode(ext, corrected)
            if ok:
                out_path.write_bytes(buf.tobytes())
                self._log_var.set(f"✓ Guardado: {out_path}")
                messagebox.showinfo("Listo",
                                    f"Imagen corregida guardada en:\n{out_path}",
                                    parent=self)

    def _apply_batch(self):
        folder = self._lote_var.get()
        if not folder:
            messagebox.showwarning("Sin carpeta",
                                   "Selecciona la carpeta del lote.",
                                   parent=self)
            return
        from corrections.aruco import correct_batch

        def worker():
            results = correct_batch(
                folder,
                margin=self._margin.get(),
                progress_cb=lambda c, t, n:
                    self.after(0, lambda: self._show_progress(c, t, n)),
            )
            self.after(0, lambda: self._on_batch_done(results))

        threading.Thread(target=worker, daemon=True).start()

    def _on_batch_done(self, results):
        self._hide_progress()
        msg = (f"✓ {results['ok']} imágenes corregidas\n"
               f"⚠ Sin marcadores: {results['no_markers']}\n"
               f"Carpeta: {results['output_folder']}")
        self._log_var.set(msg.replace("\n", "   "))
        messagebox.showinfo("Lote completado", msg, parent=self)


# ── Calibración de escala ─────────────────────────────────────────────────────

class ScaleDialog(BaseDialog):
    W, H = 760, 640
    RESIZABLE = True

    def __init__(self, parent,
                 current_image: np.ndarray | None = None,
                 current_path: str | None = None,
                 on_scale_set=None):
        super().__init__(parent, "Calibración de escala")
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
            "1. Carga una imagen de referencia.\n"
            "2. Haz clic en DOS puntos sobre un objeto de tamaño conocido.\n"
            "3. Ingresa la distancia real y la unidad.\n"
            "Usa + / - para hacer zoom y las flechas del teclado para navegar.",
            justify="left", wraplength=500).pack(anchor="w", pady=(0, 6))

        # Fuente de imagen + botones de zoom
        src_f = tk.Frame(b, bg=COLORS["bg_card"])
        src_f.pack(fill=tk.X, pady=2)
        tk.Label(src_f, text="Imagen:",
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
            tk.Button(src_f, text="Imagen actual",
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
            ("✕ Limpiar", self._clear_points, 10),
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
            b, text="Paso 2: haz clic en DOS puntos sobre la imagen",
            bg=COLORS["bg_card"], fg=COLORS["text_muted"],
            font=FONTS["small"], anchor="w")
        self._pts_label.pack(fill=tk.X)

        # Paso 3: distancia real + unidad (DESPUÉS de marcar puntos)
        step3 = tk.Frame(b, bg=COLORS["bg_card"])
        step3.pack(fill=tk.X, pady=(6,2))
        tk.Label(step3,
                 text="Paso 3 — Distancia real:",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"]).pack(side=tk.LEFT)
        self._dist_var = tk.DoubleVar(value=10.0)
        tk.Entry(step3, textvariable=self._dist_var, width=8,
                 bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1,
                 font=FONTS["mono"]).pack(side=tk.LEFT, padx=4)

        tk.Label(step3, text="Unidad:",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"]).pack(side=tk.LEFT, padx=(12, 2))
        self._unit_var = tk.StringVar(value="mm")
        from corrections.scale import UNITS
        ttk.Combobox(step3, textvariable=self._unit_var,
                     values=UNITS, state="readonly", width=4,
                     font=FONTS["body"]).pack(side=tk.LEFT)

        tk.Button(step3, text="Calcular",
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
        tk.Button(parent, text="Cerrar", command=self.destroy,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)
        tk.Button(parent, text="Limpiar puntos",
                  command=self._clear_points,
                  bg=COLORS["btn_bg"], fg=COLORS["text_muted"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)
        tk.Button(parent, text="Aplicar escala al lote",
                  command=self._apply,
                  bg=COLORS["accent"], fg="#FFFFFF",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=2)

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Imagen de referencia",
            filetypes=[("Imágenes","*.jpg *.jpeg *.png *.bmp *.tif"),
                       ("Todos","*.*")],
            parent=self)
        if not path:
            return
        self._img_path_var.set(path)
        try:
            raw = Path(path).read_bytes()
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                self._image = img
                self._load_image(img)
            else:
                messagebox.showerror(
                    "Error", "No se pudo cargar la imagen.",
                    parent=self)
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)

    def _load_image(self, img: np.ndarray):
        self._points   = []
        self._img_full = img.copy()
        self._scale_result = None
        self._result_var.set("—")
        self._pts_label.config(
            text="Paso 2: haz clic en DOS puntos sobre la imagen")
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
                text=f"Punto 1: {p1}  →  Punto 2: {p2}  |  "
                     f"Distancia: {dist_px:.1f} px  —  "
                     "Ahora ingresa la distancia real y pulsa Calcular")
        else:
            self._pts_label.config(
                text=f"Punto 1 marcado en {self._points[0]}  —  "
                     "haz clic en el segundo punto")

        self._redraw()

    def _compute(self):
        if len(self._points) != 2:
            return
        from corrections.scale import compute_scale
        try:
            # Puntos ya están en coordenadas de imagen original
            result = compute_scale(
                self._points[0], self._points[1],
                self._dist_var.get(),
                self._unit_var.get())
            self._scale_result = result
            self._result_var.set(result.format())
        except Exception as e:
            self._result_var.set(f"Error: {e}")

    def _clear_points(self):
        self._points = []
        self._scale_result = None
        self._result_var.set("—")
        self._pts_label.config(
            text="Haz clic en DOS puntos sobre la imagen")
        self._redraw()

    def _apply(self):
        # Calcular primero si no se ha hecho
        if len(self._points) == 2:
            self._compute()
        if not self._scale_result:
            messagebox.showwarning("Sin escala",
                                   "Marca dos puntos primero.",
                                   parent=self)
            return
        if self._on_scale:
            self._on_scale(self._scale_result)
        self._log_var.set(
            f"✓ Escala aplicada: {self._scale_result.format()}")
        messagebox.showinfo(
            "Escala configurada",
            f"Factor aplicado al lote:\n{self._scale_result.format()}",
            parent=self)