"""Segmentación por ejemplos: clases, pincel, aprender y clasificar."""
from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from fenotit import log, settings
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.gui.widgets.slider import Slider
from fenotit.gui.widgets.help import HelpIcon
from fenotit.i18n import t


_log = log.get("gui.window.examples")


class ExamplesMixin:
    def _update_stroke_tool(self):
        st = getattr(self, "strokes", None)
        if st is None or not hasattr(self, "class_list"):
            return
        seg_tab = self._right_tab == "seg" and self._panel_of("steps").winfo_ismapped()
        on = self._seg_method == "classifier" and seg_tab and bool(self.current_image_path) \
            and self._analysis_segments() and not (self.areas and self.areas.mode)
        st.active = on
        st.visible = on and bool(self.show_strokes_var.get())
        pt = self.prompts
        pt.active = self._seg_method == "model" and self.model_mode.get() == "points" and seg_tab \
            and bool(self.current_image_path) and self._analysis_segments() and not (self.areas and self.areas.mode)
        pt.redraw()
        self.region_picker.active = self._seg_method == "model" and self.model_mode.get() == "auto" \
            and seg_tab and bool(self.current_image_path) and self._analysis_segments() \
            and not (self.areas and self.areas.mode)
        st.colors = [c["color"] for c in self.class_list.classes]
        st.current = self.class_list.current
        st.brush = int(self.brush_var.get())
        st.redraw()
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
                kept = [dict(s, c=s["c"] - (s["c"] > removed)) for s in strokes if s["c"] != removed]
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

    def _on_class_pick(self, i: int):
        self.strokes.current = i
        self._update_stroke_tool()

    def _on_brush(self):
        self.strokes.brush = int(self.brush_var.get())
        if getattr(self, "_brush_job", None):
            self.root.after_cancel(self._brush_job)
        self._brush_job = self.root.after(600, lambda: settings.set("brush", int(self.brush_var.get())))

    def _on_show_strokes(self):
        self._update_stroke_tool()

    def _update_cls_status(self, extra: str = ""):
        n_strokes = sum(len(v) for v in self.project.strokes.values())
        n_photos = sum(1 for v in self.project.strokes.values() if v)
        painted = {s["c"] for v in self.project.strokes.values() for s in v}
        text = t("cls.status", n=n_strokes, photos=n_photos) if n_strokes else t("cls.start")
        if n_strokes and len(painted) < 2:
            text += ". " + t("cls.need_two")
        self.cls_status.set(text + (f". {extra}" if extra else ""))

    def _classifier_preview(self, img, inc, exc):
        """Aprende de los trazos y clasifica la foto actual en un hilo; mientras, se ve la foto
        con los trazos. Si los trazos cambian antes de terminar, el resultado viejo se tira."""
        from fenotit.core.segmentation import classifier as C
        full = self.scaler_left.original
        detail = self._detail()
        strokes = {k: list(v) for k, v in self.project.strokes.items()
                   if k in {self.project.key(p) for p in self.project.images}}
        key = (id(full), detail, json.dumps(strokes, sort_keys=True), self._corr_signature())
        cached = getattr(self, "_cls_cache", None)
        if cached is not None and cached[0] == key:
            self._render_classes(img, cached[1], inc, exc)
            return
        self.zoom_ctrl.redraw_with_overlay(overlay_left=img)
        painted = {s["c"] for v in strokes.values() for s in v}
        if len(painted) < 2:
            self._update_cls_status()
            self.preview_var.set("")
            return
        self._cls_gen = getattr(self, "_cls_gen", 0) + 1
        gen = self._cls_gen
        self._update_cls_status(t("cls.learning"))
        cur = self.project.key(self.current_image_path)
        corrections, max_mpx, sig = self.project.corrections, self.project.max_mpx, self._corr_signature()
        feat_cache = getattr(self, "_feat_cache", None)

        def work():
            t0 = time.perf_counter()
            try:
                if feat_cache is not None and feat_cache[0] == (id(full), detail):
                    F = feat_cache[1]
                else:
                    F = C.features(C.work_copy(full, detail))
                    self._feat_cache = ((id(full), detail), F)
                if gen != self._cls_gen:
                    return
                X, y = self._gather_examples(strokes, detail, cur, F, corrections, max_mpx, sig)
                if X is None or gen != self._cls_gen:
                    return
                model = C.train(X, y)
                cm = C.predict(model, F)
                secs = time.perf_counter() - t0
                self.root.after(0, lambda: self._on_classified(gen, key, cm, secs))
            except Exception:
                _log.exception("Segmentación por ejemplos")
                self.root.after(0, lambda: self._update_cls_status(t("cls.failed")))
        threading.Thread(target=work, daemon=True).start()

    def _on_classified(self, gen, key, cm, secs):
        if gen != getattr(self, "_cls_gen", 0):
            return
        self._cls_cache = (key, cm)
        self._update_cls_status(t("cls.learned", s=f"{secs:.1f}"))
        self._show_preview()

    def _render_classes(self, img, cm, inc, exc):
        """Clases sobre la foto: las de objeto pintadas con su color dentro de la máscara."""
        from fenotit.core.segmentation import classifier as C
        cmp = cv2.resize(cm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        classes = self.class_list.classes
        objects = [i for i, c in enumerate(classes) if i and c.get("object", True)]
        mask = self._finish_mask((np.isin(cmp, objects) * 255).astype(np.uint8), img, inc, exc)
        colored = img.copy()
        inside = mask > 0
        lut = C.colorize(np.arange(len(classes), dtype=np.uint8), [c["color"] for c in classes])
        paint = lut[np.minimum(cmp, len(classes) - 1)]
        colored[inside] = (0.45 * img[inside] + 0.55 * paint[inside]).astype(np.uint8)
        self._show_mask_overlay(img, mask, colored)

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
        brow = tk.Frame(box, bg=bg)
        brow.pack(fill=tk.X, pady=(SPACE["xs"], 0))
        tk.Label(brow, text=t("cls.brush"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        self.brush_var = tk.IntVar(value=int(settings.get("brush", 12)))
        Slider(brow, self.brush_var, 3, 60, command=lambda v: self._on_brush(), bg=bg).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=SPACE["md"])
        drow = tk.Frame(box, bg=bg)
        drow.pack(fill=tk.X, pady=(SPACE["sm"], 0))
        tk.Label(drow, text=t("cls.detail"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        self._detail_names = {0.5: t("cls.detail.low"), 1.0: t("cls.detail.mid"), 2.0: t("cls.detail.high")}
        self.detail_var = tk.StringVar(value=self._detail_names[1.0])
        cb = ttk.Combobox(drow, textvariable=self.detail_var, values=list(self._detail_names.values()),
                          state="readonly", width=12, font=small)
        cb.pack(side=tk.LEFT, padx=(SPACE["sm"], SPACE["sm"]))
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_classes_change())
        HelpIcon(drow, t("cls.detail"), t("cls.detail_help"), bg=bg).pack(side=tk.LEFT)
        arow = tk.Frame(box, bg=bg)
        arow.pack(fill=tk.X, pady=(SPACE["md"], 0))
        clear = tk.Label(arow, text=t("cls.clear_photo"), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
        clear.pack(side=tk.LEFT)
        clear.bind("<Button-1>", lambda e: self._clear_strokes())
        self.show_strokes_var = tk.BooleanVar(value=True)
        tk.Checkbutton(arow, text=t("cls.show_strokes"), variable=self.show_strokes_var, bg=bg,
                       activebackground=bg, selectcolor=COLORS["bg_card"], font=small,
                       command=self._on_show_strokes).pack(side=tk.RIGHT)
        self.cls_status = tk.StringVar(value="")
        tk.Label(box, textvariable=self.cls_status, bg=bg, fg=COLORS["text_muted"], font=small,
                 justify="left", wraplength=240, anchor="w").pack(fill=tk.X, pady=(SPACE["sm"], 0))
