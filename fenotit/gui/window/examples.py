"""Segmentación por ejemplos: clases, pincel, aprender y clasificar."""
from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

# log escribe los mensajes y errores en el archivo de registro; settings guarda las preferencias del
# usuario entre sesiones.
from fenotit import log, settings
# button crea un botón con el estilo del tema.
from fenotit.gui.style.components import button
# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras); SPACE (márgenes).
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
# Slider es el deslizador del tamaño del pincel.
from fenotit.gui.widgets.slider import Slider
# HelpIcon es el «?» que explica una opción al pasar el mouse o al hacer clic.
from fenotit.gui.widgets.help import HelpIcon
# t devuelve cada texto de la ventana en el idioma elegido (fenotit/lang).
from fenotit.i18n import t


_log = log.get("gui.window.examples")


class ExamplesMixin:
    def _update_stroke_tool(self):
        stroke_editor = getattr(self, "strokes", None)
        if stroke_editor is None or not hasattr(self, "class_list"):
            return
        seg_tab = self._right_tab == "seg" and self._panel_of("steps").winfo_ismapped()
        active = self._seg_method == "classifier" and seg_tab and bool(self.current_image_path) \
            and self._analysis_segments() and not (self.areas and self.areas.mode)
        stroke_editor.active = active
        stroke_editor.visible = active and bool(self.show_strokes_var.get())
        point_editor = self.prompts
        point_editor.active = self._seg_method == "model" and self._sam_mode() == "points" and seg_tab \
            and bool(self.current_image_path) and self._analysis_segments() and not (self.areas and self.areas.mode)
        point_editor.redraw()
        self.region_picker.active = self._seg_method == "model" and self._sam_mode() == "auto" \
            and seg_tab and bool(self.current_image_path) and self._analysis_segments() \
            and not (self.areas and self.areas.mode)
        stroke_editor.colors = [item["color"] for item in self.class_list.classes]
        stroke_editor.current = self.class_list.current
        stroke_editor.brush = int(self.brush_var.get())
        stroke_editor.redraw()
        if self.zoom_ctrl:
            self.zoom_ctrl._update_cursor(self.canvas_left)
        if self._seg_method == "classifier":
            self._update_cls_status()

    def _on_strokes_change(self, strokes: list):
        path = self.current_image_path
        if not path:
            return
        self._remember("undo.strokes")
        key = self.project.key(path)
        if strokes:
            self.project.strokes[key] = strokes
        else:
            self.project.strokes.pop(key, None)
        self._on_slider_change()

    def _clear_strokes(self):
        path = self.current_image_path
        if not path or not self.project.strokes.get(self.project.key(path)):
            return
        self._remember("undo.strokes")
        self.project.strokes.pop(self.project.key(path), None)
        self.strokes.set_strokes([])
        self._on_slider_change()

    def _on_classes_change(self, removed: int | None = None):
        if removed is not None:                   # los trazos de esa clase se van; las siguientes bajan
            self._remember("undo.classes")
            for key, strokes in list(self.project.strokes.items()):
                kept = [dict(stroke, c=stroke["c"] - (stroke["c"] > removed)) for stroke in strokes if stroke["c"] != removed]
                if kept:
                    self.project.strokes[key] = kept
                else:
                    self.project.strokes.pop(key)
            if self.current_image_path:
                self.strokes.set_strokes(self.project.strokes.get(self.project.key(self.current_image_path)))
        else:
            self._settings_changed()
        self._update_stroke_tool()
        self._on_slider_change()

    def _on_class_pick(self, index: int):
        self.strokes.current = index
        self._update_stroke_tool()

    def _on_brush(self):
        self.strokes.brush = int(self.brush_var.get())
        if getattr(self, "_brush_job", None):
            self.root.after_cancel(self._brush_job)
        self._brush_job = self.root.after(600, lambda: settings.set("brush", int(self.brush_var.get())))

    def _on_show_strokes(self):
        self._update_stroke_tool()

    def _update_cls_status(self, extra: str = ""):
        n_strokes = sum(len(photo_strokes) for photo_strokes in self.project.strokes.values())
        n_photos = sum(1 for photo_strokes in self.project.strokes.values() if photo_strokes)
        painted = {stroke["c"] for photo_strokes in self.project.strokes.values() for stroke in photo_strokes}
        text = t("cls.status", n=n_strokes, photos=n_photos) if n_strokes else t("cls.start")
        if n_strokes and len(painted) < 2:
            text += ". " + t("cls.need_two")
        self.cls_status.set(text + (f". {extra}" if extra else ""))

    def _classifier_preview(self, image, include, exclude):
        """Aprende de los trazos y clasifica la foto actual en un hilo; mientras, se ve la foto
        con los trazos. Si los trazos cambian antes de terminar, el resultado viejo se tira.
        Con «Ver predicción» apagada solo se ve la foto; con la predicción automática apagada
        se conserva la última predicción de la foto hasta que se pide actualizar."""
        from fenotit.core.segmentation import classifier
        full = self.scaler_left.original
        detail = self._detail()
        strokes = {photo_key: list(photo_strokes) for photo_key, photo_strokes in self.project.strokes.items()
                   if photo_key in {self.project.key(image_path) for image_path in self.project.images}}
        key = (id(full), detail, json.dumps(strokes, sort_keys=True), self._corr_signature())
        if not self.show_prediction_var.get():
            self.zoom_ctrl.redraw_with_overlay(overlay_left=image)
            self._update_cls_status()
            self.preview_var.set("")
            return
        cached = getattr(self, "_cls_cache", None)
        if cached is not None and cached[0] == key:
            self._render_classes(image, cached[1], include, exclude)
            return
        painted = {stroke["c"] for photo_strokes in strokes.values() for stroke in photo_strokes}
        paused = not self.auto_predict_var.get() and not self._predict_requested
        same_photo = cached is not None and cached[0][:2] == key[:2]
        if paused and same_photo and len(painted) >= 2:
            self._render_classes(image, cached[1], include, exclude)
            self._update_cls_status(t("cls.paused"))
            return
        self.zoom_ctrl.redraw_with_overlay(overlay_left=image)
        if len(painted) < 2 or paused:
            self._update_cls_status(t("cls.paused") if paused and len(painted) >= 2 else "")
            self.preview_var.set("")
            return
        self._predict_requested = False
        self._cls_gen = getattr(self, "_cls_gen", 0) + 1
        gen = self._cls_gen
        self._update_cls_status(t("cls.learning"))
        current = self.project.key(self.current_image_path)
        corrections, max_mpx, signature = self.project.corrections, self.project.max_mpx, self._corr_signature()
        feat_cache = getattr(self, "_feat_cache", None)

        def work():
            started = time.perf_counter()
            try:
                if feat_cache is not None and feat_cache[0] == (id(full), detail):
                    features = feat_cache[1]
                else:
                    features = classifier.features(classifier.work_copy(full, detail))
                    self._feat_cache = ((id(full), detail), features)
                if gen != self._cls_gen:
                    return
                values, labels = self._gather_examples(strokes, detail, current, features, corrections, max_mpx, signature)
                if values is None or gen != self._cls_gen:
                    return
                model = classifier.train(values, labels)
                class_map = classifier.predict(model, features)
                seconds = time.perf_counter() - started
                self.root.after(0, lambda: self._on_classified(gen, key, class_map, seconds))
            except Exception:
                _log.exception("Segmentación por ejemplos")
                self.root.after(0, lambda: self._update_cls_status(t("cls.failed")))
        threading.Thread(target=work, daemon=True).start()

    def _on_classified(self, gen, key, class_map, seconds):
        if gen != getattr(self, "_cls_gen", 0):
            return
        self._cls_cache = (key, class_map)
        self._update_cls_status(t("cls.learned", s=f"{seconds:.1f}"))
        self._show_preview()

    def _on_prediction_options(self):
        """Ver predicción o predicción automática cambiaron: el botón Actualizar solo sirve en pausa."""
        waiting = self.show_prediction_var.get() and not self.auto_predict_var.get()
        self._predict_button.config(state=tk.NORMAL if waiting else tk.DISABLED)
        self._show_preview()

    def _predict_now(self):
        self._predict_requested = True
        self._show_preview()

    def _render_classes(self, image, class_map, include, exclude):
        """Clases sobre la foto: las de objeto pintadas con su color dentro de la máscara."""
        from fenotit.core.segmentation import classifier
        class_image = cv2.resize(class_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        classes = self.class_list.classes
        objects = [index for index, item in enumerate(classes) if index and item.get("object", True)]
        mask = self._finish_mask((np.isin(class_image, objects) * 255).astype(np.uint8), image, include, exclude)
        colored = image.copy()
        inside = mask > 0
        lut = classifier.colorize(np.arange(len(classes), dtype=np.uint8), [item["color"] for item in classes])
        paint = lut[np.minimum(class_image, len(classes) - 1)]
        colored[inside] = (0.45 * image[inside] + 0.55 * paint[inside]).astype(np.uint8)
        self._show_mask_overlay(image, mask, colored)

    def _build_classifier_box(self, box):
        """Por ejemplos: clases (la elegida es la del pincel), pincel, detalle y estado."""
        from fenotit.gui.widgets.segment import ClassList
        bg, small = COLORS["bg_panel"], FONTS["small"]
        head = tk.Frame(box, bg=bg)
        head.pack(fill=tk.X)
        tk.Label(head, text=t("cls.title"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(head, t("cls.title"), t("cls.help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self.class_list = ClassList(box, self._on_classes_change, self._on_class_pick)
        self.class_list.pack(fill=tk.X, pady=(SPACE["xs"], SPACE["sm"]))
        self.class_list.show_objects(self._analysis_objects())
        brush_row = tk.Frame(box, bg=bg)
        brush_row.pack(fill=tk.X, pady=(SPACE["xs"], 0))
        tk.Label(brush_row, text=t("cls.brush"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        self.brush_var = tk.IntVar(value=int(settings.get("brush", 12)))
        Slider(brush_row, self.brush_var, 3, 60, command=lambda value: self._on_brush(), bg=bg).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=SPACE["md"])
        detail_row = tk.Frame(box, bg=bg)
        detail_row.pack(fill=tk.X, pady=(SPACE["sm"], 0))
        tk.Label(detail_row, text=t("cls.detail"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        self._detail_names = {0.5: t("cls.detail.low"), 1.0: t("cls.detail.mid"), 2.0: t("cls.detail.high")}
        self.detail_var = tk.StringVar(value=self._detail_names[1.0])
        detail_combo = ttk.Combobox(detail_row, textvariable=self.detail_var, values=list(self._detail_names.values()),
                          state="readonly", width=12, font=small)
        detail_combo.pack(side=tk.LEFT, padx=(SPACE["sm"], SPACE["sm"]))
        detail_combo.bind("<<ComboboxSelected>>", lambda event: self._on_classes_change())
        HelpIcon(detail_row, t("cls.detail"), t("cls.detail_help"), bg=bg).pack(side=tk.LEFT)
        action_row = tk.Frame(box, bg=bg)
        action_row.pack(fill=tk.X, pady=(SPACE["md"], 0))
        clear = tk.Label(action_row, text=t("cls.clear_photo"), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
        clear.pack(side=tk.LEFT)
        clear.bind("<Button-1>", lambda event: self._clear_strokes())
        self.show_strokes_var = tk.BooleanVar(value=True)
        tk.Checkbutton(action_row, text=t("cls.show_strokes"), variable=self.show_strokes_var, bg=bg,
                       activebackground=bg, selectcolor=COLORS["bg_card"], font=small,
                       command=self._on_show_strokes).pack(side=tk.RIGHT)
        prediction_row = tk.Frame(box, bg=bg)
        prediction_row.pack(fill=tk.X, pady=(SPACE["sm"], 0))
        self.show_prediction_var = tk.BooleanVar(value=True)
        self.auto_predict_var = tk.BooleanVar(value=True)
        self._predict_requested = False
        for text, variable in ((t("cls.show_prediction"), self.show_prediction_var),
                               (t("cls.auto_predict"), self.auto_predict_var)):
            tk.Checkbutton(prediction_row, text=text, variable=variable, bg=bg, activebackground=bg,
                           selectcolor=COLORS["bg_card"], font=small,
                           command=self._on_prediction_options).pack(anchor="w")
        update_row = tk.Frame(box, bg=bg)
        update_row.pack(fill=tk.X, pady=(SPACE["xs"], 0))
        self._predict_button = button(update_row, t("cls.predict_now"), self._predict_now, size="small", state=tk.DISABLED)
        self._predict_button.pack(side=tk.LEFT)
        HelpIcon(update_row, t("cls.auto_predict"), t("cls.auto_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self.cls_status = tk.StringVar(value="")
        tk.Label(box, textvariable=self.cls_status, bg=bg, fg=COLORS["text_muted"], font=small,
                 justify="left", wraplength=240, anchor="w").pack(fill=tk.X, pady=(SPACE["sm"], 0))
