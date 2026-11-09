"""Pestaña «1 Segmentar»: método, estado de la segmentación, vista previa y Afinar máscara."""
from __future__ import annotations

import tkinter as tk

import cv2
import numpy as np

from fenotit import log
from fenotit.core import pipeline
from fenotit.core.analysis.registry import ANALYSES
from fenotit.gui.style.components import button, divider, section
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.gui.widgets.help import HelpIcon
from fenotit.i18n import t


_log = log.get("gui.window.segment")


class SegmentMixin:
    def _detail(self) -> float:
        return next((k for k, v in self._detail_names.items() if v == self.detail_var.get()), 1.0)

    def _seg_state(self) -> dict:
        """Todo lo del paso 1 (lo guarda el proyecto y lo devuelve Ctrl+Z)."""
        return {"method": self._seg_method, "color_space": self.cs_var.get(), "channel": int(self.ch_var.get()),
                "min_val": int(self.min_slider.get()), "max_val": int(self.max_slider.get()),
                "auto": bool(self.auto_var.get()), "refine": self.refine_list.get(),
                "classes": self.class_list.get(), "detail": self._detail(),
                "model": {"name": "mobilesam", "mode": self.model_mode.get(), "grid": self._grid(),
                          "categories": self.category_list.get()}}

    def _set_seg_state(self, d: dict):
        self.cs_var.set(d.get("color_space", "BGR"))
        self._update_channel_names()
        self.ch_var.set(int(d.get("channel", 0)))
        self._sync_sliders(int(d.get("min_val", 0)), int(d.get("max_val", 255)))
        self.auto_var.set(bool(d.get("auto", False)))
        self.refine_list.set(d.get("refine") or [])
        self.class_list.set(d.get("classes") or self._default_classes())
        self.detail_var.set(self._detail_names.get(float(d.get("detail", 1.0)), self._detail_names[1.0]))
        m = d.get("model") or {}
        self.model_mode.set(m.get("mode", "auto"))
        self.grid_var.set(self._grid_names.get(int(m.get("grid", 16)), self._grid_names[16]))
        self.category_list.set(m.get("categories") or self._default_categories())
        self._on_model_change(quiet=True)
        if d.get("method") == "model" and "model" not in self._method_btns:
            d = dict(d, method="threshold")               # sin el modelo instalado
        self._set_seg_method(d.get("method", "threshold"), quiet=True)
        self.strokes.colors = [c["color"] for c in self.class_list.classes]
        self._update_seg_tab()

    @staticmethod
    def _default_classes() -> list[dict]:
        return [{"name": t("cls.background_name"), "color": COLORS["text_faint"], "object": False},
                {"name": t("cls.object_name"), "color": COLORS["accent"], "object": True}]

    def _model_for(self, model: dict, path) -> dict:
        """La configuración del modelo con lo propio de la foto: clics y categorías de regiones."""
        if not path:
            return dict(model, points=[])
        reg = self._regions_for(path)
        return dict(model, points=self.project.prompts.get(self.project.key(path), []),
                    default=reg["default"], picks=reg["picks"],
                    own=self.project.key(path) in self.project.regions)

    def _seg_params(self) -> dict:
        """La segmentación para el análisis (core.segmentation). Los ejemplos de la
        segmentación por ejemplos se agregan al correr (_with_examples)."""
        d = self._seg_state()
        seg = {k: d[k] for k in ("method", "color_space", "channel", "min_val", "max_val", "auto", "refine")}
        if d["method"] == "model":
            seg["model"] = self._model_for(d["model"], self.current_image_path)
        if d["method"] == "classifier":
            seg["classifier"] = {"names": [c["name"] for c in d["classes"]],
                                 "colors": [c["color"] for c in d["classes"]],
                                 "objects": [i for i, c in enumerate(d["classes"]) if i and c.get("object", True)],
                                 "detail": d["detail"]}
        return seg

    def _on_slider_change(self):
        """
        Llamado cada vez que cambia un slider, canal o espacio de color.
        Debounce de 60ms para no saturar mientras se arrastra.
        """
        if self._after_preview_id:
            self.root.after_cancel(self._after_preview_id)
        self._after_preview_id = self.root.after(60, self._show_preview)
        self._settings_changed()

    PREVIEW_PX = 16_000_000

    def _preview_source(self) -> np.ndarray:
        """La foto para la vista previa de la segmentación. Las muy grandes (p. ej. 200 MP de
        celular) se reducen a ~16 MP: una vista previa a tamaño completo pide varios GB. El
        análisis siempre usa la foto completa."""
        img = self.scaler_left.original
        n = img.shape[0] * img.shape[1]
        if n <= self.PREVIEW_PX:
            return img
        cached = getattr(self, "_preview_cache", None)
        if cached is not None and cached[0] is img:
            return cached[1]
        k = (self.PREVIEW_PX / n) ** 0.5
        small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        self._preview_cache = (img, small)
        return small

    def _show_preview(self):
        """La segmentación (paso 1) sobre la Entrada, con el zoom actual: umbral al instante;
        por ejemplos, en segundo plano (aprender y clasificar tarda un poco)."""
        if not self.scaler_left.has_image or self.zoom_ctrl is None:
            return
        if not self._seg_preview_on():
            self.zoom_ctrl.redraw_with_overlay(overlay_left=None)
            self.preview_var.set("")
            return
        img = self._preview_source()
        try:
            from fenotit.core import roi as areas
            inc, exc = areas.masks(self.project.roi_for(self.current_image_path), img.shape[1], img.shape[0])
            if self._seg_method == "classifier":
                self._classifier_preview(img, inc, exc)
                return
            if self._seg_method == "model":
                self._model_preview(img, inc, exc)
                return
            if self.auto_var.get():                   # Otsu solo con el área de análisis
                ctx = pipeline.run(img, [{"step": "otsu", "params": {
                    "color_space": self.cs_var.get(), "channel": int(self.ch_var.get())}}], roi=inc, exclusions=exc)
                mask = ctx.mask
                self._otsu_value = int(ctx.extra.get("otsu_threshold", 0))
                self._sync_sliders(*ctx.extra.get("otsu_range", (0, 255)))   # muestran el rango de Otsu
            else:
                from fenotit.core import colorspaces
                ch = colorspaces.channel(img, self.cs_var.get(), int(self.ch_var.get()))
                mask = ((ch >= int(self.min_slider.get())) & (ch <= int(self.max_slider.get()))).astype(np.uint8) * 255
            self._update_histogram(img, inc, exc)
            self._show_mask_overlay(img, self._finish_mask(mask, img, inc, exc))
        except Exception:
            _log.debug("Preview falló", exc_info=True)
            if self.zoom_ctrl:
                self.zoom_ctrl.redraw_with_overlay(overlay_left=img)
            self.preview_var.set("")

    def _finish_mask(self, mask, img, inc, exc):
        """Áreas de análisis, zonas excluidas y "Afinar máscara" (a la escala de la vista previa)."""
        from fenotit.core import segmentation as S
        if inc is not None:
            mask = cv2.bitwise_and(mask, inc)
        if exc:
            mask = mask.copy()
            cv2.fillPoly(mask, [pts for pts, _ in exc], 0)
        k = img.shape[1] / self.scaler_left.original.shape[1]
        return S.refine(mask, self.refine_list.get(), scale=k)

    def _show_mask_overlay(self, img, mask, colored=None):
        from fenotit.core.pipeline import overlay as marks
        if colored is None:
            colors = marks.resolve(self.project.display.get("style"), img, mask)
            view = marks.tint(img, mask, colors)
        else:
            view = colored
        self.zoom_ctrl.redraw_with_overlay(overlay_left=view)
        full = self.scaler_left.original
        k = full.shape[0] * full.shape[1] / mask.size        # >1 si la vista previa es reducida
        n_px = int(np.count_nonzero(mask) * k)
        pct = round(n_px / (mask.size * k) * 100, 1)
        text = t("status.preview", px=f"{n_px:,}", pct=pct)
        if self._seg_method == "threshold" and self.auto_var.get():
            text += ", Otsu = " + str(getattr(self, "_otsu_value", ""))
        self.preview_var.set(text)

    def _seg_preview_on(self) -> bool:
        """La segmentación se ve en la pestaña Segmentar y, después, si el análisis elegido
        segmenta."""
        if not self._analysis_segments():
            return False
        if getattr(self, "_analysis_chosen", False):
            return True
        return self._right_tab == "seg" and self._panel_of("steps").winfo_ismapped()

    def _build_seg_tab(self, rf):
        """MÉTODO (Umbral / Por ejemplos) → sus ajustes → AFINAR MÁSCARA → Analizar."""
        self._after_preview_id = None
        bg, small = COLORS["bg_panel"], FONTS["small"]
        self._seg_note = tk.Label(rf, text=t("seg.not_used"), bg=bg, fg=COLORS["text_muted"], font=small,
                                  justify="left", wraplength=240)
        head = self._section_lbl(rf, t("seg.method"))
        HelpIcon(head, t("seg.method").capitalize(), t("seg.method_help"), bg=bg).pack(side=tk.LEFT)
        self._seg_head = head
        bar = tk.Frame(rf, bg=bg)
        bar.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["xs"], SPACE["sm"]))
        self._seg_method = "threshold"
        self._method_btns = {}
        from fenotit.core.segmentation import sam
        for key in ("threshold", "classifier") + (("model",) if sam.available() else ()):
            b = tk.Label(bar, text=t(f"seg.method.{key}"), font=small, cursor="hand2", padx=SPACE["md"], pady=SPACE["sm"],
                         highlightthickness=1)
            b.pack(side=tk.LEFT, padx=(0, SPACE["sm"]))
            b.bind("<Button-1>", lambda e, k=key: self._set_seg_method(k))
            self._method_btns[key] = b
        self._seg_bar = bar
        self._thr_box = tk.Frame(rf, bg=bg)
        self._build_threshold_box(self._thr_box)
        self._cls_box = tk.Frame(rf, bg=bg)
        self._build_classifier_box(self._cls_box)
        self._model_box = tk.Frame(rf, bg=bg)
        self._build_model_box(self._model_box)
        self._refine_head = self._section_lbl(rf, t("refine.title"))
        HelpIcon(self._refine_head, t("refine.title").capitalize(), t("refine.help"), bg=bg).pack(side=tk.LEFT)
        from fenotit.gui.widgets.segment import RefineList
        self.refine_list = RefineList(rf, self._on_refine_change)
        self.refine_list.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["xs"], SPACE["md"]))
        self._next_btn = button(rf, text="", command=lambda: self._select_right_tab("analysis"), kind="primary")
        self._next_label = None
        self._shown_methods = None
        self._paint_method_btns()

    def _set_next_step(self, label: str | None):
        """Botón al pie de Segmentar con el nombre del paso siguiente (None = no hay)."""
        self._next_label = label
        if label:
            self._next_btn.config(text=t("seg.next_step", step=label))
        self._update_seg_tab()

    def _show_method_buttons(self, allowed: tuple[str, ...] | None):
        """Solo los métodos de segmentación que admite el análisis; con uno solo, sin botones."""
        self._shown_methods = [k for k in self._method_btns if not allowed or k in allowed]
        for k, b in self._method_btns.items():
            b.pack_forget()
            if k in self._shown_methods:
                b.pack(side=tk.LEFT, padx=(0, SPACE["sm"]))
        self._update_seg_tab()

    def _update_seg_tab(self):
        """Si el análisis elegido no segmenta (Color de imagen), la pestaña lo dice."""
        uses = self._analysis_segments()
        widgets = (self._seg_head, self._seg_bar, self._thr_box, self._cls_box, self._model_box,
                   self._refine_head, self.refine_list, self._next_btn)
        if uses:
            self._seg_note.pack_forget()
            for w in widgets:
                w.pack_forget()
            if len(self._shown_methods or self._method_btns) > 1:
                self._seg_head.pack(fill=tk.X)
                self._seg_bar.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["xs"], SPACE["sm"]))
            box = {"classifier": self._cls_box, "model": self._model_box}.get(self._seg_method, self._thr_box)
            box.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["xs"], SPACE["sm"]))
            if self._analysis_refines():
                self._refine_head.pack(fill=tk.X)
                self.refine_list.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["xs"], SPACE["md"]))
            if self._next_label:
                self._next_btn.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["md"], SPACE["md"]))
        else:
            for w in widgets:
                w.pack_forget()
            self._seg_note.pack(fill=tk.X, padx=SPACE["lg"], pady=SPACE["lg"])

    def _analysis_segments(self) -> bool:
        """El análisis elegido usa la segmentación (o aún no hay análisis elegido)."""
        name = self._selected_analysis()
        if not getattr(self, "_analysis_chosen", False) or name not in ANALYSES:
            return True
        return ANALYSES[name].segmentation is not None

    def _analysis_refines(self) -> bool:
        """El análisis elegido usa «Afinar máscara» (o aún no hay análisis elegido)."""
        name = self._selected_analysis()
        if not getattr(self, "_analysis_chosen", False) or name not in ANALYSES:
            return True
        return ANALYSES[name].refine

    def _paint_method_btns(self):
        for k, b in self._method_btns.items():
            on = k == self._seg_method
            b.config(bg=COLORS["accent_light"] if on else COLORS["bg_card"], fg=COLORS["accent"] if on else COLORS["text"],
                     highlightbackground=COLORS["accent_faint"] if on else COLORS["divider"])

    def _set_seg_method(self, key: str, quiet: bool = False):
        if key == self._seg_method:
            return
        if not quiet:
            self._remember("undo.segmentation")
        self._seg_method = key
        self._paint_method_btns()
        self._update_seg_tab()
        self._update_stroke_tool()
        if self.config_panel is not None and not quiet:
            self._on_analysis_selected(None, choose=getattr(self, "_analysis_chosen", False))
        self._on_slider_change()

    def _on_refine_change(self):
        self._settings_changed()
        if self._seg_method == "classifier":
            self._show_preview()
        else:
            self._on_slider_change()

    def _section_lbl(self, parent, text):
        f = section(parent, text, bg=COLORS["bg_panel"], line="below")
        f.pack(fill=tk.X)
        return f

    def _divider(self, parent):
        divider(parent).pack(fill=tk.X, pady=SPACE["sm"])
