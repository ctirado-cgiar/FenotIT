"""Diálogo único para configurar las correcciones del proyecto."""
import copy
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fenotit import log
from fenotit.core.corrections import colorcard as cc
from fenotit.core.corrections import pipeline as P
from fenotit.gui.colorcard_dialogs import edit_values, pick_region, show_detection
from fenotit.gui.help import HelpIcon
from fenotit.core.corrections.aruco import detect_aruco_corners
from fenotit.core.image_io import load_image
from fenotit.gui.widgets import ImagePicker
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
    W, H = 760, 660

    def __init__(self, parent, corrections: P.Corrections, paths=None, current_path=None,
                 aruco_scale: bool = False, on_apply=None):
        self.c = copy.deepcopy(corrections)
        self._paths = list(paths or [])
        self._raw_path = current_path
        self._raw = load_image(current_path) if current_path else None
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
        self._muted(self.body, t("corr.intro")).pack(anchor="w", pady=(0, 6))
        ImagePicker(self.body, self._paths, self._raw_path, self._set_raw).pack(anchor="w", pady=(0, 8))
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

    def _set_raw(self, path: str):
        self._raw_path, self._raw = path, load_image(path)
        for var in (self._p_status, self._c_status):
            var.set("")

    def _muted(self, parent, text):
        return tk.Label(parent, text=text, bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                        font=FONTS["small"], justify="left", wraplength=680)

    def _check(self, parent, var, key):
        return tk.Checkbutton(parent, text=t(key), variable=var,
                              bg=COLORS["bg_card"], fg=COLORS["text"],
                              activebackground=COLORS["bg_card"],
                              selectcolor=COLORS["bg_panel"], font=FONTS["body"])

    def _small(self, parent, key, cmd):
        return tk.Button(parent, text=t(key), command=cmd, bg=COLORS["btn_bg"], fg=COLORS["accent"],
                         relief="flat", font=FONTS["small"], cursor="hand2", padx=6, pady=0)

    def _button(self, parent, key, cmd):
        return tk.Button(parent, text=t(key), command=cmd, bg=COLORS["btn_bg"],
                         fg=COLORS["accent"], relief="flat", font=FONTS["body"],
                         cursor="hand2", padx=10)

    def _enable_row(self, parent, var, kind):
        """Casilla "Aplicar" y el "?" con la explicación (estándar de ayuda: gui/help.py)."""
        row = tk.Frame(parent, bg=COLORS["bg_card"])
        row.pack(fill=tk.X)
        self._check(row, var, "corr.enable").pack(side=tk.LEFT)
        HelpIcon(row, t(f"corr.tab.{kind}"), t(f"corr.{kind}.why"), t(f"corr.{kind}.short")).pack(
            side=tk.LEFT, padx=(4, 0))
        return row

    def _status(self, parent):
        var = tk.StringVar()
        tk.Label(parent, textvariable=var, bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["small"], justify="left", wraplength=680).pack(anchor="w", pady=(8, 0))
        return var

    # Distorsión
    def _tab_distortion(self, f):
        d = self.c.distortion
        self._d_on = tk.BooleanVar(value=d.enabled)
        self._enable_row(f, self._d_on, "distortion")
        self._muted(f, t("corr.distortion.help")).pack(anchor="w", pady=(4, 8))
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
        from fenotit import settings
        from fenotit.gui.chessboard_dialog import ChessboardDialog
        cols, rows = settings.get("board_grid", [7, 6])

        def use(res, board_photos=()):
            self.c.distortion = P.Distortion(True, res.mtx.tolist(), res.dist.tolist(), res.rms_error,
                                             t("corr.distortion.n_photos", n=res.n_images_used),
                                             list(res.size) if res.size else None, list(board_photos))
            settings.set("board_grid", [int(dlg.cols.get()), int(dlg.rows.get())])
            self._d_on.set(True)
            self._refresh_distortion()
        dlg = ChessboardDialog(self, cols, rows, on_use=use, project_paths=self._paths)

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
        self._enable_row(f, self._p_on, "perspective")
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
        bg = COLORS["bg_card"]
        self._c_on = tk.BooleanVar(value=col.enabled)
        self._enable_row(f, self._c_on, "color")
        self._muted(f, t("corr.color.help")).pack(anchor="w", pady=(4, 6))

        def radios(title, var, options, cmd=None):
            box = tk.Frame(f, bg=bg)
            box.pack(fill=tk.X, pady=(4, 0))
            tk.Label(box, text=title, bg=bg, fg=COLORS["text"], font=("Segoe UI", 9, "bold"),
                     width=14, anchor="nw").pack(side=tk.LEFT, anchor="n")
            col_f = tk.Frame(box, bg=bg)
            col_f.pack(side=tk.LEFT, fill=tk.X)
            rows = {}
            for value, label in options:
                r = tk.Frame(col_f, bg=bg)
                r.pack(fill=tk.X)
                tk.Radiobutton(r, text=label, variable=var, value=value, command=cmd, bg=bg, fg=COLORS["text"],
                               selectcolor=COLORS["bg_panel"], activebackground=bg,
                               font=FONTS["body"]).pack(side=tk.LEFT)
                rows[value] = r
            return rows

        self._c_card = tk.StringVar(value=col.card)
        cards = radios(t("corr.color.card"), self._c_card, [
            ("colorchecker24", t("corr.color.card24")), ("custom", t("corr.color.custom")),
            ("white", t("corr.color.white"))], self._color_changed)
        self._c_rows, self._c_cols = tk.StringVar(value=str(col.rows)), tk.StringVar(value=str(col.cols))
        cr = cards["custom"]
        for i, var in enumerate((self._c_rows, self._c_cols)):
            if i:
                tk.Label(cr, text="×", bg=bg, fg=COLORS["text_muted"]).pack(side=tk.LEFT)
            tk.Entry(cr, textvariable=var, width=3, justify="center", bg=COLORS["bg_panel"], relief="solid",
                     bd=1, font=FONTS["mono"]).pack(side=tk.LEFT, padx=2)
        self._small(cr, "corr.color.values", self._edit_values).pack(side=tk.LEFT, padx=6)
        self._small(cards["white"], "corr.color.mark_area", self._mark_area).pack(side=tk.LEFT, padx=6)

        self._c_ref = tk.StringVar(value=col.reference)
        refs = radios(t("corr.color.towards"), self._c_ref, [
            ("values", t("corr.color.ref_values")), ("photo", t("corr.color.ref_photo"))], self._color_changed)
        self._small(refs["photo"], "corr.color.use_current", self._use_reference_photo).pack(side=tk.LEFT, padx=6)

        self._c_mode = tk.StringVar(value=col.mode)
        self._mode_rows = radios(t("corr.color.where"), self._c_mode, [
            ("per_image", t("corr.color.mode.per_image")), ("fixed", t("corr.color.mode.fixed"))],
            self._color_changed)

        self._button(f, "corr.color.detect", self._detect_card).pack(anchor="w", pady=(10, 0))
        self._c_status = self._status(f)
        self._color_changed()

    def _color_cfg(self) -> P.ColorCard:
        c = copy.deepcopy(self.c.color)
        c.enabled = self._c_on.get()
        c.card, c.reference, c.mode = self._c_card.get(), self._c_ref.get(), self._c_mode.get()
        try:
            c.rows, c.cols = max(2, int(self._c_rows.get())), max(2, int(self._c_cols.get()))
        except ValueError:
            pass
        if c.card == "white":
            c.mode = "fixed"
        return c

    def _color_changed(self):
        c = self._color_cfg()
        white = c.card == "white"
        for row in self._mode_rows.values():
            for w in row.winfo_children():
                w.config(state="disabled" if white else "normal")
        notes = []
        if c.card == "custom" and c.reference == "values":
            ok = c.values and len(c.values) == c.rows * c.cols
            notes.append(t("corr.color.values_ok", n=len(c.values)) if ok else t("corr.color.values_missing"))
        if white:
            notes.append(t("corr.color.area_ok") if c.region else t("corr.color.area_missing"))
        if c.reference == "photo":
            notes.append(t("corr.color.ref_ok", name=c.target_photo) if c.target else t("corr.color.ref_missing"))
        if c.mode == "fixed" and not white:
            notes.append(t("corr.color.place_ok") if c.place else t("corr.color.place_missing"))
        self._c_status.set("\n".join(notes))

    def _edit_values(self):
        c = self._color_cfg()
        old = c.values if c.values and len(c.values) == c.rows * c.cols else None
        vals = edit_values(self, c.rows, c.cols, old)
        if vals:
            self.c.color.values, self.c.color.rows, self.c.color.cols = vals, c.rows, c.cols
            self._c_card.set("custom")
            self._color_changed()

    def _mark_area(self):
        img = self._base_image("color")
        if img is None:
            self._c_status.set(t("corr.no_image"))
            return
        r = pick_region(self, img, self.c.color.region)
        if r:
            self.c.color.region = r
            if self.c.color.card != "white":
                self.c.color.target = None             # la referencia medida era de otra tarjeta
            self._c_card.set("white")
            self._color_changed()

    def _use_reference_photo(self):
        img = self._base_image("color")
        if img is None:
            self._c_status.set(t("corr.no_image"))
            return
        c = self._color_cfg()
        try:
            self.c.color.target = cc.measure_reference(img, c)
        except Exception as e:
            self._c_status.set(t("corr.color.not_found") + f"\n{e}")
            return
        self.c.color.target_photo = Path(self._raw_path).name if self._raw_path else ""
        self._c_ref.set("photo")
        self._color_changed()

    def _detect_card(self):
        img = self._base_image("color")
        if img is None:
            self._c_status.set(t("corr.no_image"))
            return
        c = self._color_cfg()
        self._c_status.set(t("corr.color.detecting"))
        self.update_idletasks()
        if c.card == "white":
            if not c.region:
                self._c_status.set(t("corr.color.area_missing"))
                return
            fixed, info = cc.white_balance(img, c)
            show_detection(self, None, img, fixed, t("corr.color.white_done", gains=", ".join(
                f"{g:.2f}" for g in info.get("gains", []))))
            self._color_changed()
            return
        rows, cols = c.grid()
        card = cc.detect(img, rows, cols)
        if card is None:
            self._c_status.set(t("corr.color.not_found"))
            return
        self.c.color.place = {**card.to_dict(), "size": [img.shape[1], img.shape[0]]}
        try:
            fixed, info = cc.correct(img, card, cc.reference_of(c))
        except Exception as e:
            show_detection(self, cc.preview(img, card), img, None,
                           t("corr.color.found_only", found=card.found, n=rows * cols) + f"\n{e}")
            self._color_changed()
            return
        text = t("corr.color.found", found=card.found, n=rows * cols, err=f"{info['mean_error_rgb']:.1f}")
        if info["dropped"]:
            text += "\n" + t("corr.color.dropped", chips=", ".join(str(i + 1) for i in info["dropped"]))
        show_detection(self, cc.preview(img, card, info["dropped"], info["turns"]), img, fixed, text)
        self._color_changed()

    # ── Aplicar ───────────────────────────────────────────────────────────────

    def _apply(self):
        self.c.distortion.enabled = self._d_on.get() and bool(self.c.distortion.mtx)
        self.c.perspective = self._perspective_cfg()
        self.c.color = self._color_cfg()
        col = self.c.color
        if col.enabled:
            missing = None
            if col.card == "white" and not col.region:
                missing = "corr.color.area_missing"
            elif col.reference == "photo" and not col.target:
                missing = "corr.color.ref_missing"
            elif col.card == "custom" and col.reference == "values" and                     not (col.values and len(col.values) == col.rows * col.cols):
                missing = "corr.color.values_missing"
            elif col.mode == "fixed" and col.card != "white" and not col.place:
                missing = "corr.color.place_missing"
            if missing:
                messagebox.showwarning(t("corr.tab.color"), t(missing), parent=self)
                return
        use_scale = self._p_scale.get() and self.c.perspective.enabled \
            and bool(self.c.perspective.width_mm and self.c.perspective.height_mm)
        if self._on_apply:
            self._on_apply(self.c, use_scale)
        self.destroy()
