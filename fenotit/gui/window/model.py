"""Segmentación con un modelo de IA: lista de modelos (máscaras o cajas, con una miniatura de
ejemplo), agregar un detector .onnx, MobileSAM (modos, regiones, categorías por clic y clics en
objetos) y detectores (confianza, clases, pasar las cajas a máscaras)."""
from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from fenotit import log, settings
from fenotit.gui.style.components import button
from fenotit.gui.style.theme import CLASS_COLORS, COLORS, FONTS, SPACE
from fenotit.gui.widgets.help import HelpIcon
from fenotit.i18n import t


_log = log.get("gui.window.model")


class ModelMixin:
    def _grid(self) -> int:
        return next((k for k, v in self._grid_names.items() if v == self.grid_var.get()), 16)

    def _build_model_box(self, outer):
        bg, small = COLORS["bg_panel"], FONTS["small"]
        from fenotit.core.segmentation import models
        self._models = models.available()
        self._model_id = next((m["id"] for m in self._models if m["id"] == "mobilesam"),
                              self._models[0]["id"] if self._models else "")
        head = tk.Frame(outer, bg=bg)
        head.pack(fill=tk.X)
        tk.Label(head, text=t("model.title"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(head, t("model.title"), t("model.help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self._model_rows = tk.Frame(outer, bg=bg)
        self._model_rows.pack(fill=tk.X, pady=(SPACE["xs"], 0))
        add = tk.Label(outer, text="+ " + t("model.add"), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2",
                       disabledforeground=COLORS["text_faint"])
        add.pack(anchor="w", pady=(SPACE["xs"], SPACE["md"]))
        add.bind("<Button-1>", lambda e: str(add.cget("state")) != "disabled" and self._add_model())
        self._model_add_link = add
        self._thumbs = {}
        box = self._sam_box = tk.Frame(outer, bg=bg)
        self._det_box = tk.Frame(outer, bg=bg)
        self._build_detector_box(self._det_box)
        head = tk.Frame(box, bg=bg)
        head.pack(fill=tk.X)
        tk.Label(head, text="MobileSAM", bg=bg, fg=COLORS["text"], font=FONTS["body_bold"]).pack(side=tk.LEFT)
        HelpIcon(head, "MobileSAM", t("sam.help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self.model_mode = tk.StringVar(value="auto")
        for key in ("auto", "points"):
            tk.Radiobutton(box, text=t(f"sam.mode.{key}"), variable=self.model_mode, value=key, bg=bg,
                           fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg, font=small,
                           command=self._on_model_change).pack(anchor="w")
        grow = tk.Frame(box, bg=bg)
        tk.Label(grow, text=t("sam.grid"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        self._grid_names = {8: t("sam.grid.8"), 16: t("sam.grid.16"), 24: t("sam.grid.24"), 32: t("sam.grid.32")}
        self.grid_var = tk.StringVar(value=self._grid_names[16])
        cb = ttk.Combobox(grow, textvariable=self.grid_var, values=list(self._grid_names.values()),
                          state="readonly", width=16, font=small)
        cb.pack(side=tk.LEFT, padx=SPACE["sm"])
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())
        HelpIcon(grow, t("sam.grid"), t("sam.grid_help"), bg=bg).pack(side=tk.LEFT)
        self._grid_row = grow
        arow = tk.Frame(box, bg=bg)
        self._sam_actions = arow
        self._sam_try = button(arow, text=t("sam.try"), command=self._sam_try_now, kind="secondary", size="small")
        self._sam_clear = tk.Label(arow, text=t("sam.clear"), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
        self._sam_clear.bind("<Button-1>", lambda e: self._clear_prompts())
        # categorías de las regiones (solo Automático)
        cat = tk.Frame(box, bg=bg)
        chead = tk.Frame(cat, bg=bg)
        chead.pack(fill=tk.X, pady=(SPACE["md"], SPACE["xs"]))
        tk.Label(chead, text=t("cat.title"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(chead, t("cat.title"), t("cat.help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        from fenotit.gui.widgets.segment import CategoryList
        self.category_list = CategoryList(cat, self._on_categories_change, lambda i: None)
        self.category_list.set(self._default_categories())
        self.category_list.pack(fill=tk.X)
        links = tk.Frame(cat, bg=bg)
        links.pack(fill=tk.X, pady=(SPACE["sm"], 0))
        tk.Label(links, text=t("sam.regions_photo"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        for key, fn in (("sam.all", lambda: self._set_all_regions(True)),
                        ("sam.none", lambda: self._set_all_regions(False))):
            lb = tk.Label(links, text=t(key), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
            lb.pack(side=tk.LEFT, padx=(SPACE["md"], 0))
            lb.bind("<Button-1>", lambda e, f=fn: f())
        self._cat_box = cat
        self._sam_status_lbl = tk.Label(box, textvariable=self.sam_status, bg=bg, fg=COLORS["text_muted"],
                                        font=small, justify="left", wraplength=240, anchor="w")
        self._render_models()
        self._on_model_change(quiet=True)

    # ── Lista de modelos ────────────────────────────────────────────────────

    def _model(self) -> dict | None:
        return next((m for m in self._models if m["id"] == self._model_id), None)

    def _model_output(self) -> str:
        """masks o boxes según el modelo elegido."""
        m = self._model()
        return m["output"] if m else "masks"

    def _sam_mode(self) -> str | None:
        """Modo de MobileSAM (auto o points) si el modelo elegido da máscaras; None si da cajas."""
        return self.model_mode.get() if self._model_output() == "masks" else None

    def _boxes_only(self) -> bool:
        if not hasattr(self, "to_masks_var"):
            return False
        return self._seg_method == "model" and self._model_output() == "boxes" and not self.to_masks_var.get()

    def _thumb(self, output: str):
        if output not in self._thumbs:
            from PIL import Image, ImageTk
            from fenotit.gui.window.common import _assets
            self._thumbs[output] = ImageTk.PhotoImage(Image.open(_assets() / "models" / f"{output}.png"))
        return self._thumbs[output]

    def _render_models(self):
        """Una fila por modelo: miniatura de lo que da (máscaras o cajas), nombre y descripción."""
        small = FONTS["small"]
        for w in self._model_rows.winfo_children():
            w.destroy()
        if not self._models:
            tk.Label(self._model_rows, text=t("model.none"), bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                     font=small, justify="left", wraplength=240, anchor="w").pack(fill=tk.X)
        from fenotit.core.segmentation import models
        for m in self._models:
            on = m["id"] == self._model_id
            bg = COLORS["accent_light"] if on else COLORS["bg_card"]
            row = tk.Frame(self._model_rows, bg=bg, highlightthickness=1, cursor="hand2",
                           highlightbackground=COLORS["accent_faint"] if on else COLORS["divider"])
            row.pack(fill=tk.X, pady=(0, SPACE["sm"]))
            pic = tk.Label(row, image=self._thumb(m["output"]), bg=bg, bd=0)
            pic.pack(side=tk.LEFT, padx=SPACE["sm"], pady=SPACE["sm"])
            text = tk.Frame(row, bg=bg)
            text.pack(side=tk.LEFT, fill=tk.X, expand=True)
            name = tk.Label(text, text=m["name"], bg=bg, fg=COLORS["text"], font=FONTS["small_bold"], anchor="w")
            name.pack(fill=tk.X)
            desc = t("model.out.masks") if m["output"] == "masks" else t("model.out.boxes", n=len(m["classes"]))
            sub = tk.Label(text, text=desc, bg=bg, fg=COLORS["text_muted"], font=small, anchor="w", justify="left",
                           wraplength=150)
            sub.pack(fill=tk.X)
            missing = models.missing_info(m)
            card = tk.Label(text, text=t("model.card_missing") if missing else t("model.card"), bg=bg,
                            fg=COLORS["warning_text"] if missing else COLORS["accent"], font=small, anchor="w",
                            cursor="hand2")
            card.pack(fill=tk.X)
            card.bind("<Button-1>", lambda e, i=m["id"]: self._open_model_info(i))
            if m.get("description"):
                from fenotit.gui.widgets.toolbar import Tooltip
                for w in (pic, name, sub):
                    Tooltip(w, m["description"])
            widgets = (row, pic, text, name, sub)
            if models.is_user(m):
                x = tk.Label(row, text="✕", bg=bg, fg=COLORS["text_muted"], font=small, cursor="hand2",
                             padx=SPACE["sm"])
                x.pack(side=tk.RIGHT, anchor="n", pady=SPACE["xs"])
                x.bind("<Button-1>", lambda e, i=m["id"]: self._remove_model(i))
            for w in widgets:
                w.bind("<Button-1>", lambda e, i=m["id"]: self._choose_model(i))

    def _set_model_state(self, m: dict):
        """Modelo elegido y opciones del detector guardadas en el proyecto (o deshechas)."""
        self.conf_var.set(float(m.get("conf", 0.25)))
        self.to_masks_var.set(bool(m.get("to_masks", False)))
        ids = [x["id"] for x in self._models]
        self._model_id = m.get("name") if m.get("name") in ids else (
            "mobilesam" if "mobilesam" in ids else (ids[0] if ids else ""))
        if self._model_output() == "boxes":
            self._set_classes(self._model()["classes"], m.get("use") if m.get("name") == self._model_id else None)
        self._render_models()

    def _choose_model(self, model_id: str, quiet: bool = False):
        if model_id == self._model_id and not quiet:
            return
        if not quiet:
            self._remember("undo.segmentation")
        self._model_id = model_id
        m = self._model()
        if m and m["output"] == "boxes":
            self._set_classes(m["classes"], None)
        self._render_models()
        self._on_model_change(quiet=quiet)
        if not quiet and self.config_panel is not None:
            self._update_unavailable()

    def _add_model(self):
        from tkinter import filedialog, messagebox
        from fenotit.core.segmentation import models
        path = filedialog.askopenfilename(parent=self.root, title=t("model.add_title"),
                                          filetypes=[(t("model.filetype"), "*.onnx *.pt *.pth"), ("*", "*")])
        if not path:
            return
        from fenotit.core.segmentation import convert
        if path.lower().endswith((".pt", ".pth")):
            if not convert.tools():
                messagebox.showwarning(t("model.add_title"), t("model.err_pt"), parent=self.root)
                return
            self.sam_status.set(t("model.converting"))
            self._model_add_link.config(state=tk.DISABLED)

            def work():
                try:
                    m = models.add(path)
                    self.root.after(0, lambda: self._on_model_added(m))
                except Exception as e:
                    _log.exception("Convertir modelo")
                    msg = t("model.err_format") if str(e) == "format" else t("model.err_convert", error=str(e))
                    self.root.after(0, lambda: self._on_model_failed(msg))
            threading.Thread(target=work, daemon=True).start()
            return
        try:
            m = models.add(path)
        except ValueError as e:
            key = "model.err_pt" if str(e) == "pt" else "model.err_format"
            messagebox.showwarning(t("model.add_title"), t(key), parent=self.root)
            return
        except Exception as e:
            _log.exception("Agregar modelo")
            messagebox.showerror(t("model.add_title"), t("model.err_read", error=str(e)), parent=self.root)
            return
        self._on_model_added(m)

    def _on_model_failed(self, msg: str):
        from tkinter import messagebox
        self._model_add_link.config(state=tk.NORMAL)
        self.sam_status.set("")
        messagebox.showerror(t("model.add_title"), msg, parent=self.root)

    def _on_model_added(self, m: dict):
        from fenotit.core.segmentation import models
        self._model_add_link.config(state=tk.NORMAL)
        self.sam_status.set("")
        self._models = models.available()
        self._choose_model(m["id"])
        if self._seg_method != "model":
            self._set_seg_method("model")
        self._open_model_info(m["id"])                    # ficha: para qué sirve, propietario, licencia

    def _open_model_info(self, model_id: str):
        """Ficha del modelo: editable si lo agregó el usuario; los de la aplicación, solo lectura."""
        from fenotit.core.segmentation import models
        from fenotit.gui.dialogs.model_info import ModelInfoDialog
        m = next((x for x in self._models if x["id"] == model_id), None)
        if m is None:
            return None

        def save(values):
            try:
                new = models.update(model_id, **values)
            except (OSError, ValueError) as e:
                return str(e)
            self._models = models.available()
            if model_id == self._model_id and new["output"] == "boxes":
                use = [i for i, v in enumerate(self._det_use) if v.get()]
                self._set_classes(new["classes"], use)
                self._det_cache = None
                self._on_detector_change()
            self._render_models()
            return None
        return self._single("model_info", lambda: ModelInfoDialog(self.root, m, save if models.is_user(m) else None))

    def _remove_model(self, model_id: str):
        from tkinter import messagebox
        from fenotit.core.segmentation import models
        m = next((x for x in self._models if x["id"] == model_id), None)
        if not m or not messagebox.askyesno(t("model.remove"), t("model.remove_q", name=m["name"]), parent=self.root):
            return
        models.remove(model_id)
        self._models = models.available()
        if model_id == self._model_id:
            self._model_id = self._models[0]["id"] if self._models else ""
            self._on_model_change()
        self._render_models()

    # ── Detector (cajas) ────────────────────────────────────────────────────

    def _build_detector_box(self, box):
        bg, small = COLORS["bg_panel"], FONTS["small"]
        self.sam_status = tk.StringVar(value="")
        row = tk.Frame(box, bg=bg)
        row.pack(fill=tk.X)
        tk.Label(row, text=t("det.conf"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(row, t("det.conf"), t("det.conf_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self.conf_var = tk.DoubleVar(value=0.25)
        val = tk.Label(row, bg=bg, fg=COLORS["text"], font=small, width=4, anchor="e")
        val.pack(side=tk.RIGHT)
        self.conf_var.trace_add("write", lambda *_: val.config(text=f"{self.conf_var.get():.2f}"))
        val.config(text="0.25")
        from fenotit.gui.widgets.slider import Slider
        Slider(box, self.conf_var, 0.05, 0.95, command=lambda v: self._on_detector_change(), bg=bg).pack(
            fill=tk.X, pady=(SPACE["xs"], SPACE["md"]))
        chead = tk.Frame(box, bg=bg)
        chead.pack(fill=tk.X)
        tk.Label(chead, text=t("det.classes"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(chead, t("det.classes"), t("det.classes_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        for key, on in (("sam.none", False), ("sam.all", True)):
            lb = tk.Label(chead, text=t(key), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
            lb.pack(side=tk.RIGHT, padx=(SPACE["md"], 0))
            lb.bind("<Button-1>", lambda e, v=on: self._use_all_classes(v))
        self._det_classes = tk.Frame(box, bg=bg)
        self._det_classes.pack(fill=tk.X, pady=(SPACE["xs"], SPACE["md"]))
        self._det_use: list[tk.BooleanVar] = []
        self._det_names: list[str] = []
        mrow = tk.Frame(box, bg=bg)
        mrow.pack(fill=tk.X)
        self.to_masks_var = tk.BooleanVar(value=False)
        self._to_masks_cb = tk.Checkbutton(mrow, text=t("det.to_masks"), variable=self.to_masks_var, bg=bg,
                                           fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg,
                                           disabledforeground=COLORS["text_faint"], font=small, padx=0,
                                           command=self._on_to_masks)
        self._to_masks_cb.pack(side=tk.LEFT)
        HelpIcon(mrow, t("det.to_masks"), t("det.to_masks_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        tk.Label(box, textvariable=self.sam_status, bg=bg, fg=COLORS["text_muted"], font=small, justify="left",
                 wraplength=240, anchor="w").pack(fill=tk.X, pady=(SPACE["md"], 0))

    def _set_classes(self, names: list[str], use: list[int] | None):
        """Una casilla por clase del detector, con su color; `use` = las marcadas (None = todas)."""
        bg, small = COLORS["bg_panel"], FONTS["small"]
        for w in self._det_classes.winfo_children():
            w.destroy()
        self._det_names = list(names)
        self._det_use = []
        for i, name in enumerate(names):
            var = tk.BooleanVar(value=use is None or i in use)
            row = tk.Frame(self._det_classes, bg=bg)
            row.pack(fill=tk.X)
            tk.Label(row, bg=self._class_color(i), width=2).pack(side=tk.LEFT, padx=(0, SPACE["sm"]), pady=1)
            tk.Checkbutton(row, text=name, variable=var, bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"],
                           activebackground=bg, font=small, padx=0,
                           command=self._on_detector_change).pack(side=tk.LEFT)
            self._det_use.append(var)

    def _use_all_classes(self, on: bool):
        for v in self._det_use:
            v.set(on)
        self._on_detector_change()

    @staticmethod
    def _class_color(i: int) -> str:
        pal = CLASS_COLORS[1:]
        return pal[i % len(pal)]

    def _detector_state(self) -> dict:
        return {"conf": round(float(self.conf_var.get()), 3),
                "use": [i for i, v in enumerate(self._det_use) if v.get()],
                "to_masks": bool(self.to_masks_var.get()),
                "classes": list(self._det_names),
                "colors": [self._class_color(i) for i in range(len(self._det_names))]}

    def _on_detector_change(self):
        self._settings_changed()
        self._on_slider_change()

    def _on_to_masks(self):
        self._remember("undo.segmentation")
        self._update_unavailable()
        self._update_seg_tab()
        self._on_detector_change()

    def _update_unavailable(self):
        """Mediciones que necesitan el contorno de cada objeto: en gris si el modelo da cajas."""
        if self.config_panel is None:
            return
        from fenotit.core.analysis.objects import NEEDS_MASKS
        why = t("det.needs_masks")
        self.config_panel.set_unavailable({k: why for k in NEEDS_MASKS} if self._boxes_only() else {})

    @staticmethod
    def _default_categories() -> list[dict]:
        return [{"name": t("cat.default"), "color": COLORS["accent"]}]

    def _on_model_change(self, quiet: bool = False):
        boxes = self._model_output() == "boxes"
        self._sam_box.pack_forget()
        self._det_box.pack_forget()
        if self._models:
            (self._det_box if boxes else self._sam_box).pack(fill=tk.X)
        from fenotit.core.segmentation import sam
        self._to_masks_cb.config(state=tk.NORMAL if sam.available() else tk.DISABLED)
        if boxes:
            self.sam_status.set("")
            if not quiet:
                self._settings_changed()
                self._update_stroke_tool()
                self._update_seg_tab()
                self._on_slider_change()
            return
        auto = self.model_mode.get() == "auto"
        for w in (self._grid_row, self._sam_actions, self._sam_status_lbl, self._sam_try, self._sam_clear,
                  self._cat_box):
            w.pack_forget()
        if auto:
            self._grid_row.pack(fill=tk.X, pady=(SPACE["sm"], 0))
        self._sam_actions.pack(fill=tk.X, pady=(SPACE["md"], 0))
        (self._sam_try if auto else self._sam_clear).pack(side=tk.LEFT)
        self._sam_status_lbl.pack(fill=tk.X, pady=(SPACE["sm"], 0))
        if auto:
            self._cat_box.pack(fill=tk.X)
        self.sam_status.set(t("sam.hint_auto") if auto else t("sam.hint_points"))
        if not quiet:
            self._settings_changed()
            self._update_stroke_tool()
            self._update_seg_tab()
            self._on_slider_change()

    def _regions_for(self, path) -> dict:
        return self.project.regions.get(self.project.key(path)) or {"default": 1, "picks": []}

    def _sam_labels(self):
        """Las regiones detectadas en la foto actual (Automático), o None."""
        cached = getattr(self, "_sam_cache", None)
        full = self.scaler_left.original
        if cached is None or full is None or cached[0] != (id(full), "auto", self._grid()):
            return None
        return cached[1]

    def _on_region_click(self, nx: float, ny: float):
        from fenotit.core.segmentation import sam
        labels = self._sam_labels()
        path = self.current_image_path
        if labels is None or not path:
            self.sam_status.set(t("sam.pick_first"))
            return
        h, w = labels.shape
        r = int(labels[min(h - 1, int(ny * h)), min(w - 1, int(nx * w))])
        if not r:
            return
        reg = self._regions_for(path)
        if self.project.key(path) not in self.project.regions:
            reg = self._learned_entry(labels) or reg         # se parte de lo que se ve
        cats = sam.categorize(labels, reg["default"], reg["picks"])
        cur = self.category_list.current + 1
        new = 0 if cats[r] == cur else cur                 # otro clic con la misma categoría: sin usar
        picks = [p for p in reg["picks"]
                 if labels[min(h - 1, int(p[1] * h)), min(w - 1, int(p[0] * w))] != r]
        picks.append([round(nx, 5), round(ny, 5), new])
        self._remember("undo.regions")
        self.project.regions[self.project.key(path)] = {"default": reg["default"], "picks": picks}
        self._store_region_examples(path)
        self._on_slider_change()

    def _learned_entry(self, labels):
        """La clasificación aprendida de esta foto como clics: la categoría más común es la
        de partida y cada región distinta lleva un clic en un punto interior."""
        model = self._region_model()
        if model is None:
            return None
        from fenotit.core.segmentation import regions
        from fenotit.core.pipeline.views import inside_point
        from scipy.ndimage import find_objects
        cats = regions.predict(model, self.scaler_left.original, labels)
        default = int(np.bincount(cats[1:]).argmax()) if len(cats) > 1 else 1
        h, w = labels.shape
        picks = []
        for i, sl in enumerate(find_objects(labels), 1):
            if sl is None or cats[i] == default:
                continue
            x, y = inside_point((labels[sl] == i).astype(np.uint8))
            picks.append([round((x + sl[1].start + 0.5) / w, 5), round((y + sl[0].start + 0.5) / h, 5), int(cats[i])])
        return {"default": default, "picks": picks}

    def _set_all_regions(self, on: bool):
        path = self.current_image_path
        if not path:
            return
        self._remember("undo.regions")
        self.project.regions[self.project.key(path)] = {
            "default": self.category_list.current + 1 if on else 0, "picks": []}
        self._store_region_examples(path)
        self._on_slider_change()

    def _region_file(self, key: str):
        import hashlib
        return settings.data_dir() / "regions" / (hashlib.blake2b(key.encode(), digest_size=10).hexdigest() + ".npz")

    def _store_region_examples(self, path):
        """Rasgos y categoría de cada región de esta foto: ejemplos para clasificar las
        regiones de las fotos donde no se hizo clic."""
        from fenotit.core.segmentation import regions, sam
        labels, full = self._sam_labels(), self.scaler_left.original
        if labels is None:
            return
        key = self.project.key(path)
        reg = self._regions_for(path)
        X, y = regions.examples(full, labels, sam.categorize(labels, reg["default"], reg["picks"]))
        if not hasattr(self, "_region_ex"):
            self._region_ex = {}
        self._region_ex[key] = (X, y)
        try:
            f = self._region_file(key)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, X=X, y=y)
        except OSError:
            _log.exception("Ejemplos de regiones")

    def _region_examples(self):
        """Todos los ejemplos de las fotos con regiones clasificadas, o None."""
        mem = getattr(self, "_region_ex", {})
        Xs, ys = [], []
        for p in self.project.images:
            key = self.project.key(p)
            if key not in self.project.regions:
                continue
            ex = mem.get(key)
            if ex is None and self._region_file(key).exists():
                with np.load(self._region_file(key)) as d:
                    ex = (d["X"], d["y"])
                mem[key] = ex
            if ex is not None and len(ex[1]):
                Xs.append(ex[0]); ys.append(ex[1])
        if not ys:
            return None
        return np.concatenate(Xs), np.concatenate(ys)

    def _region_model(self):
        from fenotit.core.segmentation import classifier as C, regions
        ex = self._region_examples()
        return None if ex is None else regions.train(*ex, C.fingerprint(*ex))

    def _on_categories_change(self, removed: int | None = None):
        if removed is not None:                          # las regiones de esa categoría quedan sin usar
            k = removed + 1
            for reg in self.project.regions.values():
                fix = lambda c: 0 if c == k else (c - 1 if c > k else c)
                reg["default"] = fix(reg.get("default", 1))
                reg["picks"] = [[x, y, fix(c)] for x, y, c in reg.get("picks", [])]
        self._settings_changed()
        self._on_slider_change()

    def _on_prompts_change(self, points: list):
        path = self.current_image_path
        if not path:
            return
        self._remember("undo.prompts")
        key = self.project.key(path)
        if points:
            self.project.prompts[key] = points
        else:
            self.project.prompts.pop(key, None)
        self._on_slider_change()

    def _clear_prompts(self):
        path = self.current_image_path
        if path and self.project.prompts.get(self.project.key(path)):
            self._remember("undo.prompts")
            self.project.prompts.pop(self.project.key(path), None)
            self.prompts.set_points([])
            self._on_slider_change()

    def _sam_try_now(self):
        self._sam_try_requested = True
        self._show_preview()

    def _model_preview(self, img, inc, exc):
        """Con clics: al instante (el codificador corre una vez por foto). Automático: solo
        al pedirlo con «Probar en esta foto» (tarda decenas de segundos)."""
        from fenotit.core.segmentation import sam
        if self._model_output() == "boxes":
            self._detect_preview(img, inc, exc)
            return
        full = self.scaler_left.original
        auto = self.model_mode.get() == "auto"
        pts = self.project.prompts.get(self.project.key(self.current_image_path), [])
        key = (id(full), "auto", self._grid()) if auto else (id(full), "points", json.dumps(pts))
        cached = getattr(self, "_sam_cache", None)
        if cached is not None and cached[0] == key:
            self._render_instances(img, cached[1], inc, exc)
            return
        self.zoom_ctrl.redraw_with_overlay(overlay_left=img)
        self.preview_var.set("")
        if auto and not getattr(self, "_sam_try_requested", False):
            return
        if not auto and not any(p[2] for p in pts):
            self.sam_status.set(t("sam.hint_points"))
            return
        self._sam_try_requested = False
        self._sam_gen = getattr(self, "_sam_gen", 0) + 1
        gen, grid = self._sam_gen, self._grid()
        self.sam_status.set(t("sam.working"))
        self._sam_try.config(state=tk.DISABLED)

        def work():
            t0 = time.perf_counter()
            try:
                labels = sam.automatic(full, grid=grid) if auto else sam.from_points(full, pts)
                secs = time.perf_counter() - t0
                self.root.after(0, lambda: self._on_sam_done(gen, key, labels, secs))
            except Exception as e:
                _log.exception("MobileSAM")
                msg = str(e)
                self.root.after(0, lambda: (self.sam_status.set(t("sam.failed", error=msg)),
                                            self._sam_try.config(state=tk.NORMAL)))
        threading.Thread(target=work, daemon=True).start()

    def _on_sam_done(self, gen, key, labels, secs):
        self._sam_try.config(state=tk.NORMAL)
        if gen != getattr(self, "_sam_gen", 0):
            return
        self._sam_cache = (key, labels)
        self.sam_status.set(t("sam.done", n=int(labels.max()), s=f"{secs:.1f}"))
        self._show_preview()

    def _render_instances(self, img, labels, inc, exc):
        """Con clics: cada objeto con su color. Automático: cada región con el color de su
        categoría y el borde de todas (las sin usar, solo el borde)."""
        lab = cv2.resize(labels.astype(np.int32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.model_mode.get() != "auto":
            mask = self._finish_mask(((lab > 0) * 255).astype(np.uint8), img, inc, exc)
            rng = np.random.default_rng(7)
            pal = rng.integers(60, 255, (int(lab.max()) + 1, 3)).astype(np.uint8)
            colored = img.copy()
            inside = mask > 0
            colored[inside] = (0.45 * img[inside] + 0.55 * pal[lab[inside]]).astype(np.uint8)
            self._show_mask_overlay(img, mask, colored)
            return
        from fenotit.core.segmentation import sam
        key = self.project.key(self.current_image_path)
        model = None if key in self.project.regions else self._region_model()
        if model is not None:                              # sin clics aquí: lo aprendido de las otras fotos
            from fenotit.core.segmentation import regions
            cats = regions.predict(model, self.scaler_left.original, labels)
            self.sam_status.set(t("sam.learned"))
        else:
            reg = self._regions_for(self.current_image_path)
            cats = sam.categorize(lab, reg["default"], reg["picks"])
        cat_img = cats[lab]
        mask = self._finish_mask(((cat_img > 0) * 255).astype(np.uint8), img, inc, exc)
        pal = np.zeros((len(self.category_list.cats) + 1, 3), np.uint8)
        for i, c in enumerate(self.category_list.cats, 1):
            h = c["color"].lstrip("#")
            pal[i] = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
        colored = (img * 0.75).astype(np.uint8)            # sin usar: un poco más oscuras
        inside = mask > 0
        colored[inside] = (0.5 * img[inside] + 0.5 * pal[np.clip(cat_img[inside], 0, len(pal) - 1)]).astype(np.uint8)
        edges = cv2.morphologyEx(lab.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        colored[edges] = (255, 255, 255)
        self._show_mask_overlay(img, mask, colored)

    def _detect_preview(self, img, inc, exc):
        """Cajas del detector sobre la foto, del color de su clase (o el contorno de cada
        objeto si se pasan a máscaras); se calculan en otro hilo y se guardan por foto."""
        from fenotit.core.segmentation import detect, sam
        full, m = self.scaler_left.original, self._model()
        st = self._detector_state()
        key = (id(full), self._model_id, st["conf"], tuple(st["use"]), st["to_masks"])
        cached = getattr(self, "_det_cache", None)
        if cached is not None and cached[0] == key:
            self._render_detections(img, cached[1], cached[2], inc, exc)
            return
        self.zoom_ctrl.redraw_with_overlay(overlay_left=img)
        self.preview_var.set("")
        if m is None:
            return
        self._sam_gen = getattr(self, "_sam_gen", 0) + 1
        gen = self._sam_gen
        self.sam_status.set(t("det.working"))

        def work():
            t0 = time.perf_counter()
            try:
                dets = [d for d in detect.run(full, m, st["conf"]) if d["cls"] in st["use"]]
                labels = sam.from_boxes(full, [(d["x0"], d["y0"], d["x1"], d["y1"]) for d in dets]) \
                    if st["to_masks"] else None
                secs = time.perf_counter() - t0
                self.root.after(0, lambda: self._on_detect_done(gen, key, dets, labels, secs))
            except Exception as e:
                _log.exception("Detector")
                msg = str(e)
                self.root.after(0, lambda: self.sam_status.set(t("sam.failed", error=msg)))
        threading.Thread(target=work, daemon=True).start()

    def _on_detect_done(self, gen, key, dets, labels, secs):
        if gen != getattr(self, "_sam_gen", 0):
            return
        self._det_cache = (key, dets, labels)
        self.sam_status.set(t("det.done", n=len(dets), s=f"{secs:.1f}"))
        self._show_preview()

    def _render_detections(self, img, dets, labels, inc, exc):
        k = img.shape[1] / self.scaler_left.original.shape[1]
        h, w = img.shape[:2]
        colors = [self._class_color(i) for i in range(len(self._det_names))]
        bgr = [(int(c[5:7], 16), int(c[3:5], 16), int(c[1:3], 16)) for c in colors] or [(255, 255, 255)]
        region = np.full((h, w), 255, np.uint8) if inc is None else inc.copy()
        if exc:
            cv2.fillPoly(region, [pts for pts, _ in exc], 0)
        view = (img * 0.8).astype(np.uint8)
        mask = np.zeros((h, w), np.uint8)
        if labels is not None:
            lab = cv2.resize(labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
            lab[region == 0] = 0
            cls = np.zeros(len(dets) + 1, np.int32)
            cls[1:] = [d["cls"] for d in dets]
            pal = np.array(bgr, np.uint8)[np.clip(cls, 0, len(bgr) - 1)]
            inside = lab > 0
            view[inside] = (0.45 * img[inside] + 0.55 * pal[lab[inside]]).astype(np.uint8)
            mask[inside] = 255
        else:
            thick = max(1, int(round(max(h, w) / 350)))
            for d in dets:
                x0, y0, x1, y1 = (int(d["x0"] * k), int(d["y0"] * k), int(d["x1"] * k), int(d["y1"] * k))
                cx, cy = min(w - 1, (x0 + x1) // 2), min(h - 1, (y0 + y1) // 2)
                if not region[cy, cx]:
                    continue
                cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
                cv2.rectangle(view, (x0, y0), (x1, y1), bgr[d["cls"] % len(bgr)], thick)
        self._show_mask_overlay(img, mask, view)
