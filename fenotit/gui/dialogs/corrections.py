"""Diálogo único para configurar las correcciones del proyecto."""
import copy
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from fenotit import log
from fenotit.core.corrections import colorcard
from fenotit.core.corrections import pipeline as corrections_pipeline
from fenotit.core.corrections.distortion import MAX_PHOTOS as D_MAX
from fenotit.gui.dialogs.colorcard import edit_values, pick_region, show_detection
from fenotit.gui.widgets.help import HelpIcon
from fenotit.core.corrections.aruco import detect_aruco_corners
from fenotit.core.image_io import load_image
from fenotit.gui.widgets.basic import ImagePicker
from fenotit.gui.dialogs.scale import BaseDialog
from fenotit.gui.style.components import button
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.i18n import t

_log = log.get("gui.corrections")


def _float_or_none(text: str) -> float | None:
    try:
        value = float(text.replace(",", "."))
        return value if value > 0 else None
    except ValueError:
        return None


class CorrectionsDialog(BaseDialog):
    """Ventana de una corrección: distorsión del lente, perspectiva o tarjeta de color."""
    SIZES = {"distortion": (760, 640), "perspective": (620, 460), "color": (660, 620)}

    def __init__(self, parent, corrections: corrections_pipeline.Corrections, paths=None, current_path=None,
                 aruco_scale: bool = False, on_apply=None, kind: str = "distortion"):
        self._kind = kind
        self.WIDTH, self.HEIGHT = self.SIZES[kind]
        self.c = copy.deepcopy(corrections)
        self._paths = list(paths or [])
        self._raw_path = current_path
        self._raw = load_image(current_path) if current_path else None
        self._on_apply = on_apply
        self._aruco_scale = aruco_scale
        super().__init__(parent, t(f"corr.tab.{kind}"))
        self._build_body()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_buttons(self, parent):
        button(parent, text=t("common.close"), command=self.destroy, kind="secondary").pack(side=tk.RIGHT, padx=SPACE["xs"])
        button(parent, text=t("corr.apply"), command=self._apply, kind="primary").pack(side=tk.RIGHT, padx=SPACE["xs"])

    def _build_body(self):
        ImagePicker(self.body, self._paths, self._raw_path, self._set_raw).pack(anchor="w", pady=(0, SPACE["md"]))
        frame = tk.Frame(self.body, bg=COLORS["bg_card"])
        frame.pack(fill=tk.BOTH, expand=True)
        {"distortion": self._tab_distortion, "perspective": self._tab_perspective, "color": self._tab_color}[self._kind](frame)
        if self._kind == "color":
            self._color_changed()

    def _entry_row(self, parent, label, var, width=8):
        frame = tk.Frame(parent, bg=COLORS["bg_card"])
        frame.pack(fill=tk.X, pady=SPACE["xs"])
        tk.Label(frame, text=label, bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"], width=34, anchor="w").pack(side=tk.LEFT)
        tk.Entry(frame, textvariable=var, width=width, bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1, font=FONTS["mono"]).pack(side=tk.LEFT, padx=SPACE["sm"])
        return frame

    def _set_raw(self, path: str):
        self._raw_path, self._raw = path, load_image(path)
        for var in (self._p_status, self._c_status):
            var.set("")

    def _muted(self, parent, text):
        return tk.Label(parent, text=text, bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                        font=FONTS["small"], justify="left", wraplength=self.WIDTH - 70)

    def _check(self, parent, var, key):
        return tk.Checkbutton(parent, text=t(key), variable=var,
                              bg=COLORS["bg_card"], fg=COLORS["text"],
                              activebackground=COLORS["bg_card"],
                              selectcolor=COLORS["bg_panel"], font=FONTS["body"])

    def _small(self, parent, key, cmd):
        return button(parent, text=t(key), command=cmd, kind="secondary", size="small")

    def _button(self, parent, key, cmd):
        return button(parent, text=t(key), command=cmd, kind="secondary")

    def _enable_row(self, parent, var, kind):
        """Casilla "Aplicar" y el "?" con la explicación (estándar de ayuda: gui/widgets/help.py)."""
        row = tk.Frame(parent, bg=COLORS["bg_card"])
        row.pack(fill=tk.X)
        self._check(row, var, "corr.enable").pack(side=tk.LEFT)
        HelpIcon(row, t(f"corr.tab.{kind}"), t(f"corr.{kind}.why"), t(f"corr.{kind}.short")).pack(
            side=tk.LEFT, padx=(SPACE["sm"], 0))
        return row

    def _status(self, parent):
        var = tk.StringVar()
        tk.Label(parent, textvariable=var, bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["small"], justify="left", wraplength=self.WIDTH - 70).pack(anchor="w", pady=(SPACE["md"], 0))
        return var

    # Distorsión
    def _tab_distortion(self, frame):
        from fenotit import settings
        from fenotit.gui.dialogs.chessboard import ChessboardPanel
        distortion_cfg = self.c.distortion
        self._d_on = tk.BooleanVar(value=distortion_cfg.enabled)
        self._enable_row(frame, self._d_on, "distortion")
        self._muted(frame, t("board.help", n=D_MAX)).pack(anchor="w", pady=(SPACE["xs"], SPACE["md"]))
        cols, rows = settings.get("board_grid", [7, 6])

        def use(res, board_photos=()):
            self.c.distortion = corrections_pipeline.Distortion(True, res.mtx.tolist(), res.dist.tolist(), res.rms_error,
                                             t("corr.distortion.n_photos", n=res.n_images_used),
                                             list(res.size) if res.size else None, list(board_photos))
            settings.set("board_grid", [int(self._board.cols.get()), int(self._board.rows.get())])
            self._d_on.set(True)
        self._board = ChessboardPanel(frame, cols, rows, on_result=use, project_paths=self._paths,
                                      extra=(t("corr.distortion.load_npz"), self._load_npz),
                                      status=self._distortion_text())
        self._board.pack(fill=tk.BOTH, expand=True)

    def _distortion_text(self) -> str:
        distortion_cfg = self.c.distortion
        if not distortion_cfg.mtx:
            return t("corr.distortion.none")
        rms = f", RMS {distortion_cfg.rms:.3f} px" if distortion_cfg.rms else ""
        return t("corr.distortion.ready", source=distortion_cfg.source) + rms

    def _refresh_distortion(self):
        self._board.status.set(self._distortion_text())

    def _load_npz(self):
        path = filedialog.askopenfilename(title=t("corr.distortion.load_npz"),
                                          filetypes=[("NumPy", "*.npz")], parent=self)
        if not path:
            return
        try:
            self.c.distortion = corrections_pipeline.load_npz(path)
        except Exception as e:
            _log.exception("No se pudo leer %s", path)
            messagebox.showerror(t("common.error"), str(e), parent=self)
            return
        self._d_on.set(True)
        self._refresh_distortion()

    # Perspectiva
    def _tab_perspective(self, frame):
        perspective_cfg = self.c.perspective
        self._p_on = tk.BooleanVar(value=perspective_cfg.enabled)
        self._enable_row(frame, self._p_on, "perspective")
        self._muted(frame, t("corr.perspective.help")).pack(anchor="w", pady=(SPACE["sm"], SPACE["md"]))
        self._p_w = tk.StringVar(value="" if perspective_cfg.width_mm is None else f"{perspective_cfg.width_mm:g}")
        self._p_h = tk.StringVar(value="" if perspective_cfg.height_mm is None else f"{perspective_cfg.height_mm:g}")
        self._p_m = tk.StringVar(value=str(perspective_cfg.margin_px))
        self._entry_row(frame, t("corr.perspective.width"), self._p_w)
        self._entry_row(frame, t("corr.perspective.height"), self._p_h)
        self._entry_row(frame, t("corr.perspective.margin"), self._p_m)
        self._p_scale = tk.BooleanVar(value=self._aruco_scale)
        self._check(frame, self._p_scale, "corr.perspective.use_scale").pack(anchor="w", pady=(SPACE["md"], 0))
        self._button(frame, "corr.verify", self._verify_aruco).pack(anchor="w", pady=SPACE["md"])
        self._p_status = self._status(frame)

    def _perspective_cfg(self) -> corrections_pipeline.Perspective:
        try:
            margin = max(0, int(self._p_m.get()))
        except ValueError:
            margin = 20
        return corrections_pipeline.Perspective(self._p_on.get(), margin,
                             _float_or_none(self._p_w.get()), _float_or_none(self._p_h.get()))

    def _base_image(self, upto: str):
        """Imagen actual con las correcciones anteriores a `upto` aplicadas: las de esta
        ventana con lo que muestra; las de otras ventanas, como están guardadas."""
        if self._raw is None:
            return None
        corrections = corrections_pipeline.Corrections(distortion=copy.deepcopy(self.c.distortion))
        if self._kind == "distortion":
            corrections.distortion.enabled = self._d_on.get()
        if upto == "color":
            corrections.perspective = self._perspective_cfg() if self._kind == "perspective" else copy.deepcopy(self.c.perspective)
        return corrections_pipeline.apply(self._raw, corrections)[0]

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
        _, mm_per_px, _ = corrections_pipeline.rectify(img, cfg)
        msg = t("corr.perspective.ok")
        if mm_per_px:
            msg += "\n" + t("status.scale", scale=f"{mm_per_px:.6f} mm/px")
        self._p_status.set(msg)
        self._p_on.set(True)                     # probarla y que funcione = usarla

    # Color
    def _tab_color(self, frame):
        col = self.c.color
        bg = COLORS["bg_card"]
        self._c_on = tk.BooleanVar(value=col.enabled)
        self._enable_row(frame, self._c_on, "color")
        self._muted(frame, t("corr.color.help")).pack(anchor="w", pady=(SPACE["sm"], SPACE["md"]))

        def radios(title, var, options, cmd=None):
            box = tk.Frame(frame, bg=bg)
            box.pack(fill=tk.X, pady=(SPACE["sm"], 0))
            tk.Label(box, text=title, bg=bg, fg=COLORS["text"], font=FONTS["body_bold"],
                     width=14, anchor="nw").pack(side=tk.LEFT, anchor="n")
            col_f = tk.Frame(box, bg=bg)
            col_f.pack(side=tk.LEFT, fill=tk.X)
            rows = {}
            for value, label in options:
                option_row = tk.Frame(col_f, bg=bg)
                option_row.pack(fill=tk.X)
                tk.Radiobutton(option_row, text=label, variable=var, value=value, command=cmd, bg=bg, fg=COLORS["text"],
                               selectcolor=COLORS["bg_panel"], activebackground=bg,
                               font=FONTS["body"]).pack(side=tk.LEFT)
                rows[value] = option_row
            return rows

        self._c_card = tk.StringVar(value=col.card)
        cards = radios(t("corr.color.card"), self._c_card, [
            ("colorchecker24", t("corr.color.card24")), ("custom", t("corr.color.custom")),
            ("white", t("corr.color.white"))], self._color_changed)
        self._c_rows, self._c_cols = tk.StringVar(value=str(col.rows)), tk.StringVar(value=str(col.cols))
        custom_card = cards["custom"]
        for i, var in enumerate((self._c_rows, self._c_cols)):
            if i:
                tk.Label(custom_card, text="×", bg=bg, fg=COLORS["text_muted"]).pack(side=tk.LEFT)
            tk.Entry(custom_card, textvariable=var, width=3, justify="center", bg=COLORS["bg_panel"], relief="solid",
                     bd=1, font=FONTS["mono"]).pack(side=tk.LEFT, padx=SPACE["xs"])
        self._small(custom_card, "corr.color.values", self._edit_values).pack(side=tk.LEFT, padx=SPACE["md"])
        self._small(cards["white"], "corr.color.mark_area", self._mark_area).pack(side=tk.LEFT, padx=SPACE["md"])

        self._c_ref = tk.StringVar(value=col.reference)
        refs = radios(t("corr.color.towards"), self._c_ref, [
            ("values", t("corr.color.ref_values")), ("photo", t("corr.color.ref_photo"))], self._color_changed)
        self._small(refs["photo"], "corr.color.use_current", self._use_reference_photo).pack(side=tk.LEFT, padx=SPACE["md"])

        self._c_mode = tk.StringVar(value=col.mode)
        self._mode_rows = radios(t("corr.color.where"), self._c_mode, [
            ("per_image", t("corr.color.mode.per_image")), ("fixed", t("corr.color.mode.fixed"))],
            self._color_changed)

        self._button(frame, "corr.color.detect", self._detect_card).pack(anchor="w", pady=(SPACE["lg"], 0))
        self._c_status = self._status(frame)
        self._color_changed()

    def _color_cfg(self) -> corrections_pipeline.ColorCard:
        color_cfg = copy.deepcopy(self.c.color)
        color_cfg.enabled = self._c_on.get()
        color_cfg.card, color_cfg.reference, color_cfg.mode = self._c_card.get(), self._c_ref.get(), self._c_mode.get()
        try:
            color_cfg.rows, color_cfg.cols = max(2, int(self._c_rows.get())), max(2, int(self._c_cols.get()))
        except ValueError:
            pass
        if color_cfg.card == "white":
            color_cfg.mode = "fixed"
        return color_cfg

    def _color_changed(self):
        color_cfg = self._color_cfg()
        white = color_cfg.card == "white"
        for row in self._mode_rows.values():
            for widget in row.winfo_children():
                widget.config(state="disabled" if white else "normal")
        notes = []
        if color_cfg.card == "custom" and color_cfg.reference == "values":
            ok = color_cfg.values and len(color_cfg.values) == color_cfg.rows * color_cfg.cols
            notes.append(t("corr.color.values_ok", n=len(color_cfg.values)) if ok else t("corr.color.values_missing"))
        if white:
            notes.append(t("corr.color.area_ok") if color_cfg.region else t("corr.color.area_missing"))
        if color_cfg.reference == "photo":
            notes.append(t("corr.color.ref_ok", name=color_cfg.target_photo) if color_cfg.target else t("corr.color.ref_missing"))
        if color_cfg.mode == "fixed" and not white:
            notes.append(t("corr.color.place_ok") if color_cfg.place else t("corr.color.place_missing"))
        self._c_status.set("\n".join(notes))

    def _edit_values(self):
        color_cfg = self._color_cfg()
        old = color_cfg.values if color_cfg.values and len(color_cfg.values) == color_cfg.rows * color_cfg.cols else None
        vals = edit_values(self, color_cfg.rows, color_cfg.cols, old)
        if vals:
            self.c.color.values, self.c.color.rows, self.c.color.cols = vals, color_cfg.rows, color_cfg.cols
            self._c_card.set("custom")
            self._c_on.set(True)
            self._color_changed()

    def _mark_area(self):
        img = self._base_image("color")
        if img is None:
            self._c_status.set(t("corr.no_image"))
            return
        region = pick_region(self, img, self.c.color.region)
        if region:
            self.c.color.region = region
            if self.c.color.card != "white":
                self.c.color.target = None             # la referencia medida era de otra tarjeta
            self._c_card.set("white")
            self._c_on.set(True)
            self._color_changed()

    def _use_reference_photo(self):
        img = self._base_image("color")
        if img is None:
            self._c_status.set(t("corr.no_image"))
            return
        color_cfg = self._color_cfg()
        try:
            self.c.color.target = colorcard.measure_reference(img, color_cfg)
        except ValueError as e:
            self._c_status.set(t("corr.color.incomplete_short") if str(e) == "card incomplete"
                               else t("corr.color.not_found"))
            return
        self.c.color.target_photo = Path(self._raw_path).name if self._raw_path else ""
        self._c_ref.set("photo")
        self._c_on.set(True)
        self._color_changed()

    def _detect_card(self):
        img = self._base_image("color")
        if img is None:
            self._c_status.set(t("corr.no_image"))
            return
        color_cfg = self._color_cfg()
        self._c_status.set(t("corr.color.detecting"))
        self.update_idletasks()
        if color_cfg.card == "white":
            if not color_cfg.region:
                self._c_status.set(t("corr.color.area_missing"))
                return
            fixed, info = colorcard.white_balance(img, color_cfg)
            show_detection(self, None, img, fixed, t("corr.color.white_done", gains=", ".join(
                f"{gain:.2f}" for gain in info.get("gains", []))))
            self._color_changed()
            return
        rows, cols = color_cfg.grid()
        card = colorcard.detect(img, rows, cols)
        if card is None:
            self._c_status.set(t("corr.color.not_found"))
            return
        self.c.color.place = {**card.to_dict(), "size": [img.shape[1], img.shape[0]]}
        self._c_on.set(True)
        try:
            fixed, info = colorcard.correct(img, card, colorcard.reference_of(color_cfg))
        except Exception as e:
            _log.exception("Tarjeta de color")
            fixed, info = None, {"error": str(e)}
        if fixed is None:
            reason = t("corr.color.incomplete", n=info.get("visible", 0), total=rows * cols) \
                if info.get("error") == "card incomplete" else t("corr.color.no_fit")
            show_detection(self, colorcard.preview(img, card), img, None,
                           t("corr.color.found_only", found=card.found, n=rows * cols) + "\n" + reason)
            self._color_changed()
            return
        text = t("corr.color.found", found=card.found, n=rows * cols, err=f"{info['mean_error_rgb']:.1f}")
        if info["dropped"]:
            text += "\n" + t("corr.color.dropped", chips=", ".join(str(i + 1) for i in info["dropped"]))
        show_detection(self, colorcard.preview(img, card, info["dropped"], info["turns"]), img, fixed, text)
        self._color_changed()

    # ── Aplicar ───────────────────────────────────────────────────────────────

    def _apply(self):
        """Guarda solo la corrección de esta ventana; las demás quedan como estaban."""
        use_scale = self._aruco_scale and self.c.perspective.enabled
        if self._kind == "distortion":
            self.c.distortion.enabled = self._d_on.get() and bool(self.c.distortion.mtx)
        elif self._kind == "perspective":
            self.c.perspective = self._perspective_cfg()
            use_scale = self._p_scale.get() and self.c.perspective.enabled \
                and bool(self.c.perspective.width_mm and self.c.perspective.height_mm)
        else:
            self.c.color = self._color_cfg()
        col = self.c.color
        if self._kind == "color" and col.enabled:
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
        if self._on_apply:
            self._on_apply(self.c, use_scale)
        self.destroy()
