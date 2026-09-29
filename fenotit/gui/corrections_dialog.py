"""Diálogo único para configurar las correcciones del proyecto."""
import copy
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fenotit import log
from fenotit.core.corrections import pipeline as P
from fenotit.core.corrections.aruco import detect_aruco_corners
from fenotit.core.corrections.distortion import calibrate_from_folder
from fenotit.gui.calibration_dialogs import BaseDialog
from fenotit.gui.theme import COLORS, FONTS
from fenotit.i18n import t

_log = log.get("gui.corrections")


def _float_or_none(s: str) -> float | None:
    try:
        v = float(s.replace(",", "."))
        return v if v > 0 else None
    except ValueError:
        return None


class CorrectionsDialog(BaseDialog):
    W, H = 620, 560

    def __init__(self, parent, corrections: P.Corrections, raw_image=None,
                 aruco_scale: bool = False, on_apply=None):
        self.c = copy.deepcopy(corrections)
        self._raw = raw_image
        self._on_apply = on_apply
        self._aruco_scale = aruco_scale
        super().__init__(parent, t("corr.title"))
        self._build_body()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_buttons(self, parent):
        tk.Button(parent, text=t("common.close"), command=self.destroy,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"], relief="flat",
                  font=FONTS["body"], cursor="hand2", padx=12).pack(side=tk.RIGHT, padx=2)
        tk.Button(parent, text=t("corr.apply"), command=self._apply,
                  bg=COLORS["accent"], fg="#FFFFFF", relief="flat",
                  font=("Segoe UI", 9, "bold"), cursor="hand2", padx=12).pack(side=tk.RIGHT, padx=2)

    def _build_body(self):
        self._lbl(self.body, t("corr.intro"), justify="left", wraplength=560).pack(anchor="w", pady=(0, 8))
        nb = ttk.Notebook(self.body)
        nb.pack(fill=tk.BOTH, expand=True)
        for build, key in ((self._tab_distortion, "corr.tab.distortion"),
                           (self._tab_perspective, "corr.tab.perspective"),
                           (self._tab_color, "corr.tab.color")):
            f = tk.Frame(nb, bg=COLORS["bg_card"], padx=12, pady=10)
            nb.add(f, text=f"  {t(key)}  ")
            build(f)

    def _entry_row(self, parent, label, var, width=8):
        f = tk.Frame(parent, bg=COLORS["bg_card"])
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label, bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"], width=34, anchor="w").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var, width=width, bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1, font=FONTS["mono"]).pack(side=tk.LEFT, padx=4)
        return f

    def _muted(self, parent, text):
        return tk.Label(parent, text=text, bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                        font=FONTS["small"], justify="left", wraplength=540)

    def _check(self, parent, var, key):
        return tk.Checkbutton(parent, text=t(key), variable=var,
                              bg=COLORS["bg_card"], fg=COLORS["text"],
                              activebackground=COLORS["bg_card"],
                              selectcolor=COLORS["bg_panel"], font=FONTS["body"])

    def _button(self, parent, key, cmd):
        return tk.Button(parent, text=t(key), command=cmd, bg=COLORS["btn_bg"],
                         fg=COLORS["accent"], relief="flat", font=FONTS["body"],
                         cursor="hand2", padx=10)

    def _status(self, parent):
        var = tk.StringVar()
        tk.Label(parent, textvariable=var, bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["small"], justify="left", wraplength=540).pack(anchor="w", pady=(8, 0))
        return var

    # Distorsión
    def _tab_distortion(self, f):
        d = self.c.distortion
        self._d_on = tk.BooleanVar(value=d.enabled)
        self._check(f, self._d_on, "corr.enable").pack(anchor="w")
        self._muted(f, t("corr.distortion.help")).pack(anchor="w", pady=(4, 8))
        self._d_cols, self._d_rows = tk.StringVar(value="7"), tk.StringVar(value="6")
        self._entry_row(f, t("corr.distortion.cols"), self._d_cols, 5)
        self._entry_row(f, t("corr.distortion.rows"), self._d_rows, 5)
        row = tk.Frame(f, bg=COLORS["bg_card"])
        row.pack(anchor="w", pady=6)
        self._button(row, "corr.distortion.calibrate", self._calibrate).pack(side=tk.LEFT, padx=(0, 6))
        self._button(row, "corr.distortion.load_npz", self._load_npz).pack(side=tk.LEFT)
        self._d_status = self._status(f)
        self._refresh_distortion()

    def _refresh_distortion(self):
        d = self.c.distortion
        if not d.mtx:
            self._d_status.set(t("corr.distortion.none"))
        else:
            rms = f" · RMS {d.rms:.3f} px" if d.rms else ""
            self._d_status.set(t("corr.distortion.ready", source=d.source) + rms)

    def _calibrate(self):
        folder = filedialog.askdirectory(title=t("corr.distortion.folder"), parent=self)
        if not folder:
            return
        try:
            cols, rows = int(self._d_cols.get()), int(self._d_rows.get())
        except ValueError:
            messagebox.showwarning(t("common.error"), t("corr.distortion.bad_grid"), parent=self)
            return
        self._d_status.set(t("corr.distortion.running"))

        def work():
            res = calibrate_from_folder(folder, cols, rows,
                                        progress_cb=lambda i, n, name: self.after(
                                            0, lambda: self._show_progress(i, n, name)))
            self.after(0, lambda: self._calib_done(res, folder))
        threading.Thread(target=work, daemon=True).start()

    def _calib_done(self, res, folder):
        self._hide_progress()
        self._log_var.set("")
        if not res.success:
            self._d_status.set(t("corr.distortion.failed", used=res.n_images_used, total=res.n_images_total))
            return
        from pathlib import Path
        self.c.distortion = P.Distortion(True, res.mtx.tolist(), res.dist.tolist(), res.rms_error,
                                         Path(folder).name)
        self._d_on.set(True)
        self._refresh_distortion()
        self._d_status.set(self._d_status.get() + "\n" + t("corr.distortion.used", used=res.n_images_used,
                                                           total=res.n_images_total))

    def _load_npz(self):
        path = filedialog.askopenfilename(title=t("corr.distortion.load_npz"),
                                          filetypes=[("NumPy", "*.npz")], parent=self)
        if not path:
            return
        try:
            self.c.distortion = P.load_npz(path)
        except Exception as e:
            _log.exception("No se pudo leer %s", path)
            messagebox.showerror(t("common.error"), str(e), parent=self)
            return
        self._d_on.set(True)
        self._refresh_distortion()

    # Perspectiva
    def _tab_perspective(self, f):
        p = self.c.perspective
        self._p_on = tk.BooleanVar(value=p.enabled)
        self._check(f, self._p_on, "corr.enable").pack(anchor="w")
        self._muted(f, t("corr.perspective.help")).pack(anchor="w", pady=(4, 8))
        self._p_w = tk.StringVar(value="" if p.width_mm is None else f"{p.width_mm:g}")
        self._p_h = tk.StringVar(value="" if p.height_mm is None else f"{p.height_mm:g}")
        self._p_m = tk.StringVar(value=str(p.margin_px))
        self._entry_row(f, t("corr.perspective.width"), self._p_w)
        self._entry_row(f, t("corr.perspective.height"), self._p_h)
        self._entry_row(f, t("corr.perspective.margin"), self._p_m)
        self._p_scale = tk.BooleanVar(value=self._aruco_scale)
        self._check(f, self._p_scale, "corr.perspective.use_scale").pack(anchor="w", pady=(6, 0))
        self._button(f, "corr.verify", self._verify_aruco).pack(anchor="w", pady=6)
        self._p_status = self._status(f)

    def _perspective_cfg(self) -> P.Perspective:
        try:
            margin = max(0, int(self._p_m.get()))
        except ValueError:
            margin = 20
        return P.Perspective(self._p_on.get(), margin,
                             _float_or_none(self._p_w.get()), _float_or_none(self._p_h.get()))

    def _base_image(self, upto: str):
        """Imagen actual con las correcciones anteriores a `upto` aplicadas."""
        if self._raw is None:
            return None
        c = P.Corrections(distortion=copy.deepcopy(self.c.distortion))
        c.distortion.enabled = self._d_on.get()
        if upto == "color":
            c.perspective = self._perspective_cfg()
        return P.apply(self._raw, c)[0]

    def _verify_aruco(self):
        img = self._base_image("perspective")
        if img is None:
            self._p_status.set(t("corr.no_image"))
            return
        found = detect_aruco_corners(img)
        missing = [i for i in range(4) if i not in found]
        if missing:
            self._p_status.set(t("corr.perspective.missing", n=len(found), ids=missing))
            return
        cfg = self._perspective_cfg()
        _, mm_per_px, _ = P.rectify(img, cfg)
        msg = t("corr.perspective.ok")
        if mm_per_px:
            msg += "\n" + t("status.scale", scale=f"{mm_per_px:.6f} mm/px")
        self._p_status.set(msg)

    # Color
    def _tab_color(self, f):
        col = self.c.color
        ok = P.plantcv_available()
        self._c_on = tk.BooleanVar(value=col.enabled and ok)
        cb = self._check(f, self._c_on, "corr.enable")
        cb.pack(anchor="w")
        self._muted(f, t("corr.color.help")).pack(anchor="w", pady=(4, 8))
        if not ok:
            cb.config(state="disabled")
            tk.Label(f, text=t("corr.color.no_plantcv"), bg=COLORS["bg_card"],
                     fg=COLORS["warning"], font=FONTS["body"], justify="left").pack(anchor="w")
        self._c_mode = tk.StringVar(value=col.mode)
        for mode in ("per_image", "fixed"):
            tk.Radiobutton(f, text=t(f"corr.color.mode.{mode}"), variable=self._c_mode, value=mode,
                           bg=COLORS["bg_card"], fg=COLORS["text"], selectcolor=COLORS["bg_panel"],
                           activebackground=COLORS["bg_card"], font=FONTS["body"],
                           state="normal" if ok else "disabled").pack(anchor="w")
        row = tk.Frame(f, bg=COLORS["bg_card"])
        row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(row, text=t("corr.color.pos"), bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"], width=34, anchor="w").pack(side=tk.LEFT)
        self._c_pos = tk.StringVar(value=str(col.pos))
        ttk.Combobox(row, textvariable=self._c_pos, values=["auto", "0", "1", "2", "3"],
                     state="readonly", width=6).pack(side=tk.LEFT, padx=4)
        self._c_radius = tk.StringVar(value=str(col.radius))
        self._entry_row(f, t("corr.color.radius"), self._c_radius, 5)
        b = self._button(f, "corr.color.detect", self._detect_card)
        b.pack(anchor="w", pady=6)
        if not ok:
            b.config(state="disabled")
        self._c_status = self._status(f)
        if col.mask is not None:
            self._c_status.set(t("corr.color.mask_saved"))

    def _detect_card(self):
        img = self._base_image("color")
        if img is None:
            self._c_status.set(t("corr.no_image"))
            return
        self._c_status.set(t("corr.color.detecting"))
        self.update_idletasks()
        try:
            mask = P.detect_card(img, self._radius())
        except Exception as e:
            _log.exception("Detección de tarjeta falló")
            mask, err = None, str(e)
        else:
            err = ""
        if mask is None:
            self._c_status.set(t("corr.color.not_found") + (f"\n{err}" if err else ""))
            return
        import numpy as np
        n = len(np.unique(mask)) - 1
        self.c.color.mask = mask.astype("uint8")
        self._c_status.set(t("corr.color.found", n=n))

    def _radius(self) -> int:
        try:
            return max(1, int(self._c_radius.get()))
        except ValueError:
            return 20

    # ── Aplicar ───────────────────────────────────────────────────────────────

    def _apply(self):
        self.c.distortion.enabled = self._d_on.get() and bool(self.c.distortion.mtx)
        self.c.perspective = self._perspective_cfg()
        pos = self._c_pos.get()
        self.c.color.enabled = self._c_on.get()
        self.c.color.mode = self._c_mode.get()
        self.c.color.pos = pos if pos == "auto" else int(pos)
        self.c.color.radius = self._radius()
        if self.c.color.enabled and self.c.color.mode == "fixed" and self.c.color.mask is None:
            messagebox.showwarning(t("corr.tab.color"), t("corr.color.need_mask"), parent=self)
            return
        use_scale = self._p_scale.get() and self.c.perspective.enabled \
            and bool(self.c.perspective.width_mm and self.c.perspective.height_mm)
        if self._on_apply:
            self._on_apply(self.c, use_scale)
        self.destroy()
