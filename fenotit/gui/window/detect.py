"""Paso Detectar: lista de modelos de detección (cajas), confianza mínima, clases que se
conservan, pasar las cajas a máscaras y vista previa de las cajas sobre la foto. Lo usan el
análisis Detección y la medición «Detecciones por objeto» del análisis Objetos."""
from __future__ import annotations

import threading
import time
import tkinter as tk

import cv2
import numpy as np

from fenotit import log
from fenotit.core.analysis.registry import ANALYSES
from fenotit.gui.style.components import button, section
from fenotit.gui.style.theme import CLASS_COLORS, COLORS, FONTS, SPACE
from fenotit.gui.widgets.help import HelpIcon
from fenotit.i18n import t

_log = log.get("gui.window.detect")


class DetectMixin:
    def _build_det_tab(self, tab):
        """MODELO (lista de detectores) → confianza → clases → cajas a máscaras → siguiente paso."""
        bg, small = COLORS["bg_panel"], FONTS["small"]
        head = section(tab, t("det.model_title"), bg=bg, line="below")
        head.pack(fill=tk.X)
        HelpIcon(head, t("det.model_title").capitalize(), t("det.model_help"), bg=bg).pack(side=tk.LEFT)
        body = tk.Frame(tab, bg=bg)
        body.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["xs"], SPACE["sm"]))
        self._det_model_id = ""
        self._det_rows = tk.Frame(body, bg=bg)
        self._det_rows.pack(fill=tk.X, pady=(SPACE["xs"], 0))
        links = tk.Frame(body, bg=bg)
        links.pack(fill=tk.X, pady=(SPACE["xs"], SPACE["md"]))
        add = tk.Label(links, text="+ " + t("model.add"), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2",
                       disabledforeground=COLORS["text_faint"])
        add.pack(side=tk.LEFT)
        add.bind("<Button-1>", lambda event: str(add.cget("state")) != "disabled" and self._add_model())
        self._add_links.append(add)
        conv = tk.Label(links, text=t("conv.link"), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
        conv.pack(side=tk.RIGHT)
        conv.bind("<Button-1>", lambda event: self._open_converter())
        self._det_box = tk.Frame(body, bg=bg)
        self._build_detector_box(self._det_box)
        self.det_status = tk.StringVar(value="")
        tk.Label(body, textvariable=self.det_status, bg=bg, fg=COLORS["text_muted"], font=small, justify="left",
                 wraplength=240, anchor="w").pack(fill=tk.X, side=tk.BOTTOM, pady=(SPACE["md"], 0))
        self._det_next_btn = button(tab, text="", command=lambda: self._select_right_tab("analysis"), kind="primary")
        self._det_cache = None
        self._render_detectors()

    def _build_detector_box(self, box):
        bg, small = COLORS["bg_panel"], FONTS["small"]
        row = tk.Frame(box, bg=bg)
        row.pack(fill=tk.X)
        tk.Label(row, text=t("det.conf"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(row, t("det.conf"), t("det.conf_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self.conf_var = tk.DoubleVar(value=0.25)
        value_label = tk.Label(row, bg=bg, fg=COLORS["text"], font=small, width=4, anchor="e")
        value_label.pack(side=tk.RIGHT)
        self.conf_var.trace_add("write", lambda *_: value_label.config(text=f"{self.conf_var.get():.2f}"))
        value_label.config(text="0.25")
        from fenotit.gui.widgets.slider import Slider
        Slider(box, self.conf_var, 0.05, 0.95, command=lambda value: self._on_detector_change(), bg=bg).pack(
            fill=tk.X, pady=(SPACE["xs"], SPACE["md"]))
        classes_head = tk.Frame(box, bg=bg)
        classes_head.pack(fill=tk.X)
        tk.Label(classes_head, text=t("det.classes"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(classes_head, t("det.classes"), t("det.classes_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        for key, active in (("sam.none", False), ("sam.all", True)):
            link = tk.Label(classes_head, text=t(key), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
            link.pack(side=tk.RIGHT, padx=(SPACE["md"], 0))
            link.bind("<Button-1>", lambda event, value=active: self._use_all_classes(value))
        self._det_classes = tk.Frame(box, bg=bg)
        self._det_classes.pack(fill=tk.X, pady=(SPACE["xs"], SPACE["md"]))
        self._det_use: list[tk.BooleanVar] = []
        self._det_names: list[str] = []
        masks_row = tk.Frame(box, bg=bg)
        masks_row.pack(fill=tk.X)
        self._to_masks_row = masks_row
        self.to_masks_var = tk.BooleanVar(value=False)
        self._to_masks_cb = tk.Checkbutton(masks_row, text=t("det.to_masks"), variable=self.to_masks_var, bg=bg,
                                           fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg,
                                           disabledforeground=COLORS["text_faint"], font=small, padx=0,
                                           command=self._on_to_masks)
        self._to_masks_cb.pack(side=tk.LEFT)
        HelpIcon(masks_row, t("det.to_masks"), t("det.to_masks_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))

    # ── Modelos de detección ────────────────────────────────────────────────

    def _detectors(self) -> list[dict]:
        return [model for model in self._models if model["output"] == "boxes"]

    def _detector(self) -> dict | None:
        return next((model for model in self._detectors() if model["id"] == self._det_model_id), None)

    def _render_detectors(self):
        detectors = self._detectors()
        if self._det_model_id not in [model["id"] for model in detectors]:
            self._det_model_id = detectors[0]["id"] if detectors else ""
            if detectors:
                self._set_classes(detectors[0]["classes"], None)
        self._render_model_rows(self._det_rows, detectors, self._det_model_id, self._choose_detector,
                                t("det.none"))
        self._det_box.pack_forget()
        if detectors:
            self._det_box.pack(fill=tk.X)
        self._to_masks_row.pack_forget()
        if self._detection_analysis():                 # cajas a máscaras: solo en el análisis Detección
            self._to_masks_row.pack(fill=tk.X)
        from fenotit.core.segmentation import sam
        self._to_masks_cb.config(state=tk.NORMAL if sam.available() else tk.DISABLED)

    def _choose_detector(self, model_id: str, quiet: bool = False):
        if model_id == self._det_model_id and not quiet:
            return
        if not quiet:
            self._remember("undo.detector")
        self._det_model_id = model_id
        model = self._detector()
        if model:
            self._set_classes(model["classes"], None)
        self._render_detectors()
        if not quiet:
            self._on_detector_change()
            self._update_unavailable()

    def _forget_detector(self, model_id: str):
        """El modelo `model_id` se quitó: se elige otro detector si era el elegido."""
        if model_id == self._det_model_id:
            self._det_model_id = ""
            self._det_cache = None
        self._render_detectors()

    def _on_detector_info_changed(self, model: dict):
        """La ficha del detector elegido cambió (p. ej. los nombres de las clases)."""
        if model["id"] != self._det_model_id:
            return
        use = [index for index, var in enumerate(self._det_use) if var.get()]
        self._set_classes(model["classes"], use)
        self._det_cache = None
        self._on_detector_change()

    def _set_classes(self, names: list[str], use: list[int] | None):
        """Una casilla por clase del detector, con su color; `use` = las marcadas (None = todas)."""
        bg, small = COLORS["bg_panel"], FONTS["small"]
        for widget in self._det_classes.winfo_children():
            widget.destroy()
        self._det_names = list(names)
        self._det_use = []
        for index, name in enumerate(names):
            var = tk.BooleanVar(value=use is None or index in use)
            row = tk.Frame(self._det_classes, bg=bg)
            row.pack(fill=tk.X)
            tk.Label(row, bg=self._class_color(index), width=2).pack(side=tk.LEFT, padx=(0, SPACE["sm"]), pady=1)
            tk.Checkbutton(row, text=name, variable=var, bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"],
                           activebackground=bg, font=small, padx=0,
                           command=self._on_detector_change).pack(side=tk.LEFT)
            self._det_use.append(var)

    def _use_all_classes(self, active: bool):
        for var in self._det_use:
            var.set(active)
        self._on_detector_change()

    @staticmethod
    def _class_color(index: int) -> str:
        palette = CLASS_COLORS[1:]
        return palette[index % len(palette)]

    def _detector_state(self) -> dict:
        """El detector elegido y sus opciones (lo guarda el proyecto y lo usa el análisis)."""
        return {"name": self._det_model_id,
                "conf": round(float(self.conf_var.get()), 3),
                "use": [index for index, var in enumerate(self._det_use) if var.get()],
                "to_masks": bool(self.to_masks_var.get()),
                "classes": list(self._det_names),
                "colors": [self._class_color(index) for index in range(len(self._det_names))]}

    def _set_detector_state(self, state: dict):
        self.conf_var.set(float(state.get("conf", 0.25)))
        self.to_masks_var.set(bool(state.get("to_masks", False)))
        ids = [model["id"] for model in self._detectors()]
        self._det_model_id = state.get("name") if state.get("name") in ids else (ids[0] if ids else "")
        model = self._detector()
        if model:
            self._set_classes(model["classes"], state.get("use") if state.get("name") == self._det_model_id else None)
        self._render_detectors()

    def _on_detector_change(self):
        self._settings_changed()
        self._on_slider_change()

    def _on_to_masks(self):
        self._remember("undo.detector")
        self._update_unavailable()
        self._on_detector_change()

    # ── Qué análisis detecta ────────────────────────────────────────────────

    def _detection_analysis(self) -> bool:
        """Los objetos del análisis elegido salen del detector, sin segmentar (análisis Detección)."""
        name = self.active_analysis
        if not getattr(self, "_analysis_chosen", False) or name not in ANALYSES:
            return False
        steps = self._analysis_steps(name)
        return "detect" in steps and "segment" not in steps

    def _boxes_only(self) -> bool:
        """Los objetos son las cajas del detector (sin pasarlas a máscaras)."""
        return self._detection_analysis() and not self.to_masks_var.get()

    def _update_unavailable(self):
        """Mediciones que necesitan el contorno de cada objeto: en gris si los objetos son cajas."""
        if self.config_panel is None:
            return
        from fenotit.core.analysis.objects import NEEDS_MASKS
        why = t("det.needs_masks")
        self.config_panel.set_unavailable({key: why for key in NEEDS_MASKS} if self._boxes_only() else {})

    def _det_preview_on(self) -> bool:
        """Las cajas se ven en la pestaña Detectar y, en el análisis Detección, en sus demás pasos."""
        if not getattr(self, "_analysis_chosen", False) or not self._panel_of("steps").winfo_ismapped():
            return False
        return self._right_tab == "det" or (self._detection_analysis() and self._right_tab != "choose")

    def _set_det_next_step(self, label: str | None):
        """Botón al pie de Detectar con el nombre del paso siguiente (None = no hay)."""
        self._det_next_btn.pack_forget()
        if label:
            self._det_next_btn.config(text=t("seg.next_step", step=label))
            self._det_next_btn.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["md"], SPACE["md"]))

    # ── Vista previa ────────────────────────────────────────────────────────

    def _detect_preview(self, img, include, exclude):
        """Cajas del detector sobre la foto, del color de su clase (o el contorno de cada
        objeto si se pasan a máscaras); se calculan en otro hilo y se guardan por foto."""
        from fenotit.core.segmentation import detect, sam
        full, model = self.scaler_left.original, self._detector()
        state = self._detector_state()
        key = (id(full), self._det_model_id, state["conf"], tuple(state["use"]), state["to_masks"])
        cached = self._det_cache
        if cached is not None and cached[0] == key:
            self._render_detections(img, cached[1], cached[2], include, exclude)
            return
        self.zoom_ctrl.redraw_with_overlay(overlay_left=img)
        self.preview_var.set("")
        if model is None:
            self.det_status.set("")
            return
        self._det_gen = getattr(self, "_det_gen", 0) + 1
        generation = self._det_gen
        self.det_status.set(t("det.working"))

        def work():
            started = time.perf_counter()
            try:
                found = [box for box in detect.run(full, model, state["conf"]) if box["cls"] in state["use"]]
                labels = sam.from_boxes(full, [(box["x0"], box["y0"], box["x1"], box["y1"]) for box in found]) \
                    if state["to_masks"] and self._detection_analysis() else None
                seconds = time.perf_counter() - started
                self.root.after(0, lambda: self._on_detect_done(generation, key, found, labels, seconds))
            except Exception as error:
                _log.exception("Detector")
                message = str(error)
                self.root.after(0, lambda: self.det_status.set(t("sam.failed", error=message)))
        threading.Thread(target=work, daemon=True).start()

    def _on_detect_done(self, generation, key, found, labels, seconds):
        if generation != getattr(self, "_det_gen", 0):
            return
        self._det_cache = (key, found, labels)
        self.det_status.set(t("det.done", n=len(found), s=f"{seconds:.1f}"))
        self._show_preview()

    def _render_detections(self, img, found, labels, include, exclude):
        scale = img.shape[1] / self.scaler_left.original.shape[1]
        height, width = img.shape[:2]
        colors = [self._class_color(index) for index in range(len(self._det_names))]
        bgr = [(int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16)) for color in colors] or [(255, 255, 255)]
        region = np.full((height, width), 255, np.uint8) if include is None else include.copy()
        if exclude:
            cv2.fillPoly(region, [points for points, _ in exclude], 0)
        view = (img * 0.8).astype(np.uint8)
        mask = np.zeros((height, width), np.uint8)
        if labels is not None:
            resized = cv2.resize(labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
            resized[region == 0] = 0
            classes = np.zeros(len(found) + 1, np.int32)
            classes[1:] = [box["cls"] for box in found]
            palette = np.array(bgr, np.uint8)[np.clip(classes, 0, len(bgr) - 1)]
            inside = resized > 0
            view[inside] = (0.45 * img[inside] + 0.55 * palette[resized[inside]]).astype(np.uint8)
            mask[inside] = 255
        else:
            thickness = max(1, int(round(max(height, width) / 350)))
            for box in found:
                x0, y0 = int(box["x0"] * scale), int(box["y0"] * scale)
                x1, y1 = int(box["x1"] * scale), int(box["y1"] * scale)
                center_x, center_y = min(width - 1, (x0 + x1) // 2), min(height - 1, (y0 + y1) // 2)
                if not region[center_y, center_x]:
                    continue
                cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
                cv2.rectangle(view, (x0, y0), (x1, y1), bgr[box["cls"] % len(bgr)], thickness)
        self._show_mask_overlay(img, mask, view)
