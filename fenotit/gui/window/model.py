"""Segmentación con un modelo de IA: lista de modelos de máscaras (con una miniatura de
ejemplo), agregar modelos (.onnx, .pt o paquete; los de cajas van al paso Detectar), ficha de
cada modelo y MobileSAM (modos, regiones, categorías por clic y clics en objetos)."""
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
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.gui.widgets.help import HelpIcon
from fenotit.i18n import t


_log = log.get("gui.window.model")


class ModelMixin:
    def _grid(self) -> int:
        return next((grid for grid, name in self._grid_names.items() if name == self.grid_var.get()), 16)

    def _build_model_box(self, outer):
        bg, small = COLORS["bg_panel"], FONTS["small"]
        from fenotit.core.segmentation import models
        self._models = models.available()
        masks = self._mask_models()
        self._model_id = next((model["id"] for model in masks if model["id"] == "mobilesam"), masks[0]["id"] if masks else "")
        self.sam_status = tk.StringVar(value="")
        head = tk.Frame(outer, bg=bg)
        head.pack(fill=tk.X)
        tk.Label(head, text=t("model.title"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(head, t("model.title"), t("model.help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self._model_rows = tk.Frame(outer, bg=bg)
        self._model_rows.pack(fill=tk.X, pady=(SPACE["xs"], 0))
        links = tk.Frame(outer, bg=bg)
        links.pack(fill=tk.X, pady=(SPACE["xs"], SPACE["md"]))
        add = tk.Label(links, text="+ " + t("model.add"), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2",
                       disabledforeground=COLORS["text_faint"])
        add.pack(side=tk.LEFT)
        add.bind("<Button-1>", lambda event: str(add.cget("state")) != "disabled" and self._add_model())
        self._add_links = [add]
        converter_link = tk.Label(links, text=t("conv.link"), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
        converter_link.pack(side=tk.RIGHT)
        converter_link.bind("<Button-1>", lambda event: self._open_converter())
        self._thumbs = {}
        box = self._sam_box = tk.Frame(outer, bg=bg)
        head = tk.Frame(box, bg=bg)
        head.pack(fill=tk.X)
        tk.Label(head, text="MobileSAM", bg=bg, fg=COLORS["text"], font=FONTS["body_bold"]).pack(side=tk.LEFT)
        HelpIcon(head, "MobileSAM", t("sam.help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self.model_mode = tk.StringVar(value="auto")
        for key in ("auto", "points"):
            tk.Radiobutton(box, text=t(f"sam.mode.{key}"), variable=self.model_mode, value=key, bg=bg,
                           fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg, font=small,
                           command=self._on_model_change).pack(anchor="w")
        grid_row = tk.Frame(box, bg=bg)
        tk.Label(grid_row, text=t("sam.grid"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        self._grid_names = {8: t("sam.grid.8"), 16: t("sam.grid.16"), 24: t("sam.grid.24"), 32: t("sam.grid.32")}
        self.grid_var = tk.StringVar(value=self._grid_names[16])
        grid_combo = ttk.Combobox(grid_row, textvariable=self.grid_var, values=list(self._grid_names.values()),
                          state="readonly", width=16, font=small)
        grid_combo.pack(side=tk.LEFT, padx=SPACE["sm"])
        grid_combo.bind("<<ComboboxSelected>>", lambda event: self._on_model_change())
        HelpIcon(grid_row, t("sam.grid"), t("sam.grid_help"), bg=bg).pack(side=tk.LEFT)
        self._grid_row = grid_row
        action_row = tk.Frame(box, bg=bg)
        self._sam_actions = action_row
        self._sam_try = button(action_row, text=t("sam.try"), command=self._sam_try_now, kind="secondary", size="small")
        self._sam_clear = tk.Label(action_row, text=t("sam.clear"), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
        self._sam_clear.bind("<Button-1>", lambda event: self._clear_prompts())
        # categorías de las regiones (solo Automático)
        category_box = tk.Frame(box, bg=bg)
        category_head = tk.Frame(category_box, bg=bg)
        category_head.pack(fill=tk.X, pady=(SPACE["md"], SPACE["xs"]))
        tk.Label(category_head, text=t("cat.title"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(category_head, t("cat.title"), t("cat.help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        from fenotit.gui.widgets.segment import CategoryList
        self.category_list = CategoryList(category_box, self._on_categories_change, lambda index: None)
        self.category_list.set(self._default_categories())
        self.category_list.pack(fill=tk.X)
        links = tk.Frame(category_box, bg=bg)
        links.pack(fill=tk.X, pady=(SPACE["sm"], 0))
        tk.Label(links, text=t("sam.regions_photo"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        for key, action in (("sam.all", lambda: self._set_all_regions(True)),
                        ("sam.none", lambda: self._set_all_regions(False))):
            link = tk.Label(links, text=t(key), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
            link.pack(side=tk.LEFT, padx=(SPACE["md"], 0))
            link.bind("<Button-1>", lambda event, action=action: action())
        self._cat_box = category_box
        self._sam_status_lbl = tk.Label(box, textvariable=self.sam_status, bg=bg, fg=COLORS["text_muted"],
                                        font=small, justify="left", wraplength=240, anchor="w")
        self._render_models()
        self._on_model_change(quiet=True)

    # ── Lista de modelos ────────────────────────────────────────────────────

    def _mask_models(self) -> list[dict]:
        return [model for model in self._models if model["output"] == "masks"]

    def _model(self) -> dict | None:
        return next((model for model in self._mask_models() if model["id"] == self._model_id), None)

    def _sam_mode(self) -> str | None:
        """Modo de MobileSAM (auto o points); None si no hay modelo de máscaras."""
        return self.model_mode.get() if self._model() else None

    def _model_status(self, text: str):
        """Mensaje de los modelos (p. ej. «Convirtiendo…») en Segmentar y en Detectar."""
        self.sam_status.set(text)
        if hasattr(self, "det_status"):
            self.det_status.set(text)

    def _thumb(self, output: str):
        if output not in self._thumbs:
            from PIL import Image, ImageTk
            from fenotit.gui.window.common import _assets
            self._thumbs[output] = ImageTk.PhotoImage(Image.open(_assets() / "models" / f"{output}.png"))
        return self._thumbs[output]

    def _render_models(self):
        """Las dos listas de modelos: máscaras (Segmentar) y cajas (Detectar)."""
        self._render_model_rows(self._model_rows, self._mask_models(), self._model_id, self._choose_model,
                                t("model.none"))
        if hasattr(self, "_det_rows"):
            self._render_detectors()

    def _render_model_rows(self, container, items: list[dict], selected: str, on_choose, empty_text: str):
        """Una fila por modelo: miniatura de lo que da (máscaras o cajas), nombre y descripción."""
        small = FONTS["small"]
        for widget in container.winfo_children():
            widget.destroy()
        if not items:
            tk.Label(container, text=empty_text, bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                     font=small, justify="left", wraplength=240, anchor="w").pack(fill=tk.X)
        from fenotit.core.segmentation import models
        for model in items:
            is_selected = model["id"] == selected
            bg = COLORS["accent_light"] if is_selected else COLORS["bg_card"]
            row = tk.Frame(container, bg=bg, highlightthickness=1, cursor="hand2",
                           highlightbackground=COLORS["accent_faint"] if is_selected else COLORS["divider"])
            row.pack(fill=tk.X, pady=(0, SPACE["sm"]))
            picture = tk.Label(row, image=self._thumb(model["output"]), bg=bg, bd=0)
            picture.pack(side=tk.LEFT, padx=SPACE["sm"], pady=SPACE["sm"])
            text = tk.Frame(row, bg=bg)
            text.pack(side=tk.LEFT, fill=tk.X, expand=True)
            name = tk.Label(text, text=model["name"], bg=bg, fg=COLORS["text"], font=FONTS["small_bold"], anchor="w")
            name.pack(fill=tk.X)
            description = t("model.out.masks") if model["output"] == "masks" else t("model.out.boxes", n=len(model["classes"]))
            subtitle = tk.Label(text, text=description, bg=bg, fg=COLORS["text_muted"], font=small, anchor="w", justify="left",
                           wraplength=150)
            subtitle.pack(fill=tk.X)
            missing = models.missing_info(model)
            card = tk.Label(text, text=t("model.card_missing") if missing else t("model.card"), bg=bg,
                            fg=COLORS["warning_text"] if missing else COLORS["accent"], font=small, anchor="w",
                            cursor="hand2")
            card.pack(fill=tk.X)
            card.bind("<Button-1>", lambda event, i=model["id"]: self._open_model_info(i))
            if model.get("description"):
                from fenotit.gui.widgets.toolbar import Tooltip
                for widget in (picture, name, subtitle):
                    Tooltip(widget, model["description"])
            widgets = (row, picture, text, name, subtitle)
            if models.is_user(model):
                remove_link = tk.Label(row, text="✕", bg=bg, fg=COLORS["text_muted"], font=small, cursor="hand2",
                             padx=SPACE["sm"])
                remove_link.pack(side=tk.RIGHT, anchor="n", pady=SPACE["xs"])
                remove_link.bind("<Button-1>", lambda event, i=model["id"]: self._remove_model(i))
            for widget in widgets:
                widget.bind("<Button-1>", lambda event, i=model["id"]: on_choose(i))

    def _set_model_state(self, state: dict):
        """Modelo de máscaras guardado en el proyecto (o deshecho)."""
        ids = [model["id"] for model in self._mask_models()]
        self._model_id = state.get("name") if state.get("name") in ids else (
            "mobilesam" if "mobilesam" in ids else (ids[0] if ids else ""))
        self._render_models()

    def _choose_model(self, model_id: str, quiet: bool = False):
        if model_id == self._model_id and not quiet:
            return
        if not quiet:
            self._remember("undo.segmentation")
        self._model_id = model_id
        self._render_models()
        self._on_model_change(quiet=quiet)

    def _add_model(self):
        from tkinter import filedialog, messagebox
        from fenotit.core.segmentation import models
        path = filedialog.askopenfilename(parent=self.root, title=t("model.add_title"),
                                          filetypes=[(t("model.filetype"), "*.onnx *.pt *.pth *.fntm *.fenotitmodel"),
                                                     ("*", "*")])
        if not path:
            return
        from fenotit.core.segmentation import convert
        if path.lower().endswith((".pt", ".pth")):
            if not convert.tools():
                if messagebox.askyesno(t("model.add_title"), t("conv.ask"), parent=self.root):
                    self._open_converter(lambda: self._convert_pt(path))
                return
            self._convert_pt(path)
            return
        try:
            model = models.add(path)
        except ValueError as error:
            key = {"pt": "model.err_pt", "package": "model.err_package"}.get(str(error), "model.err_format")
            messagebox.showwarning(t("model.add_title"), t(key), parent=self.root)
            return
        except Exception as error:
            _log.exception("Agregar modelo")
            messagebox.showerror(t("model.add_title"), t("model.err_read", error=str(error)), parent=self.root)
            return
        self._on_model_added(model)

    def _open_converter(self, on_ready=None):
        from fenotit.gui.dialogs.converter import ConverterDialog
        return self._single("converter", lambda: ConverterDialog(self.root, on_ready))

    def _convert_pt(self, path: str):
        """Convierte un .pt a ONNX en otro hilo y lo agrega."""
        from fenotit.core.segmentation import models
        self._model_status(t("model.converting"))
        for link in self._add_links:
            link.config(state=tk.DISABLED)

        def work():
            try:
                model = models.add(path)
                self.root.after(0, lambda: self._on_model_added(model))
            except Exception as error:
                _log.exception("Convertir modelo")
                msg = t("model.err_format") if str(error) == "format" else t("model.err_convert", error=str(error))
                self.root.after(0, lambda: self._on_model_failed(msg))
        threading.Thread(target=work, daemon=True).start()

    def _on_model_failed(self, msg: str):
        from tkinter import messagebox
        for link in self._add_links:
            link.config(state=tk.NORMAL)
        self._model_status("")
        messagebox.showerror(t("model.add_title"), msg, parent=self.root)

    def _on_model_added(self, model: dict):
        from fenotit.core.segmentation import models
        for link in self._add_links:
            link.config(state=tk.NORMAL)
        self._model_status("")
        self._models = models.available()
        if model["output"] == "boxes":                        # detector: va al paso Detectar
            self._choose_detector(model["id"])
            if self._right_tab != "det":
                from tkinter import messagebox
                messagebox.showinfo(t("model.add_title"), t("det.added_here", name=model["name"]), parent=self.root)
        else:
            self._choose_model(model["id"])
            if self._seg_method != "model":
                self._set_seg_method("model")
        self._open_model_info(model["id"])                    # ficha: para qué sirve, propietario, licencia

    def _open_model_info(self, model_id: str):
        """Ficha del modelo: editable si lo agregó el usuario; los de la aplicación, solo lectura."""
        from fenotit.core.segmentation import models
        from fenotit.gui.dialogs.model_info import ModelInfoDialog
        model = next((candidate for candidate in self._models if candidate["id"] == model_id), None)
        if model is None:
            return None

        def save(values):
            try:
                new = models.update(model_id, **values)
            except (OSError, ValueError) as error:
                return str(error)
            self._models = models.available()
            if new["output"] == "boxes":
                self._on_detector_info_changed(new)
            self._render_models()
            return None
        def share(win):
            from tkinter import filedialog, messagebox
            dest = filedialog.asksaveasfilename(parent=win, title=t("minfo.share"), initialfile=model["name"],
                                                defaultextension=models.PACKAGE,
                                                filetypes=[(t("minfo.package"), "*" + models.PACKAGE)])
            if dest:
                out = models.export_package(model_id, dest)
                messagebox.showinfo(t("minfo.share"), t("minfo.shared", path=out), parent=win)
        user = models.is_user(model)
        return self._single("model_info", lambda: ModelInfoDialog(self.root, model, save if user else None,
                                                                  share if user else None))

    def _remove_model(self, model_id: str):
        from tkinter import messagebox
        from fenotit.core.segmentation import models
        model = next((candidate for candidate in self._models if candidate["id"] == model_id), None)
        if not model or not messagebox.askyesno(t("model.remove"), t("model.remove_q", name=model["name"]), parent=self.root):
            return
        models.remove(model_id)
        self._models = models.available()
        if model_id == self._model_id:
            masks = self._mask_models()
            self._model_id = masks[0]["id"] if masks else ""
            self._on_model_change()
        self._forget_detector(model_id)
        self._render_models()

    @staticmethod
    def _default_categories() -> list[dict]:
        return [{"name": t("cat.default"), "color": COLORS["accent"]}]

    def _on_model_change(self, quiet: bool = False):
        self._sam_box.pack_forget()
        if self._mask_models():
            self._sam_box.pack(fill=tk.X)
        auto = self.model_mode.get() == "auto"
        for widget in (self._grid_row, self._sam_actions, self._sam_status_lbl, self._sam_try, self._sam_clear,
                  self._cat_box):
            widget.pack_forget()
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

    def _on_region_click(self, norm_x: float, norm_y: float):
        from fenotit.core.segmentation import sam
        labels = self._sam_labels()
        path = self.current_image_path
        if labels is None or not path:
            self.sam_status.set(t("sam.pick_first"))
            return
        height, width = labels.shape
        region = int(labels[min(height - 1, int(norm_y * height)), min(width - 1, int(norm_x * width))])
        if not region:
            return
        entry = self._regions_for(path)
        if self.project.key(path) not in self.project.regions:
            entry = self._learned_entry(labels) or entry         # se parte de lo que se ve
        categories = sam.categorize(labels, entry["default"], entry["picks"])
        current = self.category_list.current + 1
        new = 0 if categories[region] == current else current                 # otro clic con la misma categoría: sin usar
        picks = [pick for pick in entry["picks"]
                 if labels[min(height - 1, int(pick[1] * height)), min(width - 1, int(pick[0] * width))] != region]
        picks.append([round(norm_x, 5), round(norm_y, 5), new])
        self._remember("undo.regions")
        self.project.regions[self.project.key(path)] = {"default": entry["default"], "picks": picks}
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
        categories = regions.predict(model, self.scaler_left.original, labels)
        default = int(np.bincount(categories[1:]).argmax()) if len(categories) > 1 else 1
        height, width = labels.shape
        picks = []
        for i, bounds in enumerate(find_objects(labels), 1):
            if bounds is None or categories[i] == default:
                continue
            x, y = inside_point((labels[bounds] == i).astype(np.uint8))
            picks.append([round((x + bounds[1].start + 0.5) / width, 5), round((y + bounds[0].start + 0.5) / height, 5), int(categories[i])])
        return {"default": default, "picks": picks}

    def _set_all_regions(self, active: bool):
        path = self.current_image_path
        if not path:
            return
        self._remember("undo.regions")
        self.project.regions[self.project.key(path)] = {
            "default": self.category_list.current + 1 if active else 0, "picks": []}
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
        entry = self._regions_for(path)
        features, categories = regions.examples(full, labels, sam.categorize(labels, entry["default"], entry["picks"]))
        if not hasattr(self, "_region_ex"):
            self._region_ex = {}
        self._region_ex[key] = (features, categories)
        try:
            file = self._region_file(key)
            file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(file, X=features, y=categories)
        except OSError:
            _log.exception("Ejemplos de regiones")

    def _region_examples(self):
        """Todos los ejemplos de las fotos con regiones clasificadas, o None."""
        memory = getattr(self, "_region_ex", {})
        feature_sets, category_sets = [], []
        for image in self.project.images:
            key = self.project.key(image)
            if key not in self.project.regions:
                continue
            examples = memory.get(key)
            if examples is None and self._region_file(key).exists():
                with np.load(self._region_file(key)) as data:
                    examples = (data["X"], data["y"])
                memory[key] = examples
            if examples is not None and len(examples[1]):
                feature_sets.append(examples[0]); category_sets.append(examples[1])
        if not category_sets:
            return None
        return np.concatenate(feature_sets), np.concatenate(category_sets)

    def _region_model(self):
        from fenotit.core.segmentation import classifier, regions
        examples = self._region_examples()
        return None if examples is None else regions.train(*examples, classifier.fingerprint(*examples))

    def _on_categories_change(self, removed: int | None = None):
        if removed is not None:                          # las regiones de esa categoría quedan sin usar
            removed_category = removed + 1
            for entry in self.project.regions.values():
                fix = lambda category: 0 if category == removed_category else (category - 1 if category > removed_category else category)
                entry["default"] = fix(entry.get("default", 1))
                entry["picks"] = [[x, y, fix(category)] for x, y, category in entry.get("picks", [])]
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

    def _model_preview(self, image, include, exclude):
        """Con clics: al instante (el codificador corre una vez por foto). Automático: solo
        al pedirlo con «Probar en esta foto» (tarda decenas de segundos)."""
        from fenotit.core.segmentation import sam
        full = self.scaler_left.original
        auto = self.model_mode.get() == "auto"
        points = self.project.prompts.get(self.project.key(self.current_image_path), [])
        key = (id(full), "auto", self._grid()) if auto else (id(full), "points", json.dumps(points))
        cached = getattr(self, "_sam_cache", None)
        if cached is not None and cached[0] == key:
            self._render_instances(image, cached[1], include, exclude)
            return
        self.zoom_ctrl.redraw_with_overlay(overlay_left=image)
        self.preview_var.set("")
        if auto and not getattr(self, "_sam_try_requested", False):
            return
        if not auto and not any(point[2] for point in points):
            self.sam_status.set(t("sam.hint_points"))
            return
        self._sam_try_requested = False
        self._sam_gen = getattr(self, "_sam_gen", 0) + 1
        gen, grid = self._sam_gen, self._grid()
        self.sam_status.set(t("sam.working"))
        self._sam_try.config(state=tk.DISABLED)

        def work():
            started = time.perf_counter()
            try:
                labels = sam.automatic(full, grid=grid) if auto else sam.from_points(full, points)
                seconds = time.perf_counter() - started
                self.root.after(0, lambda: self._on_sam_done(gen, key, labels, seconds))
            except Exception as error:
                _log.exception("MobileSAM")
                msg = str(error)
                self.root.after(0, lambda: (self.sam_status.set(t("sam.failed", error=msg)),
                                            self._sam_try.config(state=tk.NORMAL)))
        threading.Thread(target=work, daemon=True).start()

    def _on_sam_done(self, gen, key, labels, seconds):
        self._sam_try.config(state=tk.NORMAL)
        if gen != getattr(self, "_sam_gen", 0):
            return
        self._sam_cache = (key, labels)
        self.sam_status.set(t("sam.done", n=int(labels.max()), s=f"{seconds:.1f}"))
        self._show_preview()

    def _render_instances(self, image, labels, include, exclude):
        """Con clics: cada objeto con su color. Automático: cada región con el color de su
        categoría y el borde de todas (las sin usar, solo el borde)."""
        label_image = cv2.resize(labels.astype(np.int32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.model_mode.get() != "auto":
            mask = self._finish_mask(((label_image > 0) * 255).astype(np.uint8), image, include, exclude)
            generator = np.random.default_rng(7)
            palette = generator.integers(60, 255, (int(label_image.max()) + 1, 3)).astype(np.uint8)
            colored = image.copy()
            inside = mask > 0
            colored[inside] = (0.45 * image[inside] + 0.55 * palette[label_image[inside]]).astype(np.uint8)
            self._show_mask_overlay(image, mask, colored)
            return
        from fenotit.core.segmentation import sam
        key = self.project.key(self.current_image_path)
        model = None if key in self.project.regions else self._region_model()
        if model is not None:                              # sin clics aquí: lo aprendido de las otras fotos
            from fenotit.core.segmentation import regions
            categories = regions.predict(model, self.scaler_left.original, labels)
            self.sam_status.set(t("sam.learned"))
        else:
            entry = self._regions_for(self.current_image_path)
            categories = sam.categorize(label_image, entry["default"], entry["picks"])
        cat_img = categories[label_image]
        mask = self._finish_mask(((cat_img > 0) * 255).astype(np.uint8), image, include, exclude)
        palette = np.zeros((len(self.category_list.cats) + 1, 3), np.uint8)
        for i, category in enumerate(self.category_list.cats, 1):
            hex_color = category["color"].lstrip("#")
            palette[i] = (int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16))
        colored = (image * 0.75).astype(np.uint8)            # sin usar: un poco más oscuras
        inside = mask > 0
        colored[inside] = (0.5 * image[inside] + 0.5 * palette[np.clip(cat_img[inside], 0, len(palette) - 1)]).astype(np.uint8)
        edges = cv2.morphologyEx(label_image.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        colored[edges] = (255, 255, 255)
        self._show_mask_overlay(image, mask, colored)
