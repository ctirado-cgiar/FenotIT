"""Resultados de cada foto: mostrar, guardar en memoria y recalcular la vista de los del lote."""
from __future__ import annotations

import dataclasses
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from fenotit import log
from fenotit.core import batch
from fenotit.core.analysis.registry import ANALYSES, AnalysisResult
from fenotit.gui.charts.panels import IntraImageChartPanel
from fenotit.gui.window.common import _on_base, _analysis_label, _duration, _summary
from fenotit.i18n import t


_log = log.get("gui.window.results")


class ResultsMixin:
    def _init_results(self):
        self.last_result: AnalysisResult | None = None
        self.step_names: list[str] = []
        self._results: dict[str, dict[str, AnalysisResult]] = {}   # análisis → foto → resultado
        self._step_names: dict[str, dict[str, list]] = {}
        self._full_paths: list[str] = []          # fotos con imágenes en memoria (las demás, solo datos)
        self._rehydrating = None
        self._rehydrate_job = None

    @property
    def results_cache(self) -> dict[str, AnalysisResult]:
        """Resultados del análisis elegido (foto -> resultado)."""
        return self._results.setdefault(getattr(self, "active_analysis", None) or "", {})

    @property
    def step_names_cache(self) -> dict[str, list]:
        return self._step_names.setdefault(getattr(self, "active_analysis", None) or "", {})

    def _clear_all_results(self, path: str | None = None):
        for store in (self._results, self._step_names):
            for per in store.values():
                if path is None:
                    per.clear()
                else:
                    per.pop(path, None)

    def _clear_result(self):
        """La foto actual queda sin resultado a la vista."""
        self.last_result, self.step_names = None, []
        self._set_step(0)

    def _set_base_image(self, image):
        """Foto sobre la que se dibujan las marcas del resultado a la vista."""
        self.last_result = dataclasses.replace(self.last_result,
                                               extra={**self.last_result.extra, "base_image": image})

    def _show_cached_result(self):
        """Al cambiar de análisis: el resultado que ya tenía esta foto con ese análisis."""
        path = getattr(self, "current_image_path", None)
        if not path or not hasattr(self, "_bottom_wrap"):
            return
        if path in self.results_cache:
            self._display_result(self.results_cache[path], self.step_names_cache.get(path, []))
        else:
            self._clear_result()
            self._clear_selection()
            self._show_results_ui(False)
            self.canvas_right.delete("all")
        self._refresh_history()
        self._update_batch_list()
        self._sync_listbox()

    def _refresh_display(self):
        self._show_preview()
        # El canvas derecho se actualiza via zoom_ctrl en _show_preview
        # Si no hay imagen derecha, aseguramos que zoom_ctrl la tenga
        if self.zoom_ctrl and self.scaler_right.has_image:
            if self.zoom_ctrl._img_right is None:
                self.zoom_ctrl.set_right(self.scaler_right.original)

    def _show_on(self, canvas: tk.Canvas,
                 img_bgr: np.ndarray, attr: str):
        rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        setattr(self, attr, photo)
        canvas.delete("all")
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        x  = max(0, (cw - img_bgr.shape[1]) // 2)
        y  = max(0, (ch - img_bgr.shape[0]) // 2)
        canvas.create_image(x, y, anchor="nw", image=photo)
        # Informar al ROI selector el offset real de la imagen
        if canvas is self.canvas_left and self.areas:
            self.areas.set_image_offset(
                x, y,
                img_bgr.shape[1],
                img_bgr.shape[0])

    def _schedule_rehydrate(self, delay: int = 600):
        """Recalcula la foto actual un momento después, solo si el usuario se quedó en ella
        (pasar rápido de foto en foto no lanza un cálculo por cada una)."""
        if self._rehydrate_job:
            self.root.after_cancel(self._rehydrate_job)
        key = (self.active_analysis, self.current_image_path)

        def go():
            self._rehydrate_job = None
            if (self.active_analysis, self.current_image_path) != key or not self.last_result:
                return
            if not self.last_result.step_images:
                self._rehydrate(*key, self.last_result)
        self._rehydrate_job = self.root.after(delay, go)

    def _rehydrate(self, name: str, path: str, light: AnalysisResult):
        """Recalcula la foto con los mismos parámetros para tener sus imágenes."""
        if self._rehydrating == (name, path) or not self.scaler_left.has_image:
            return
        self._rehydrating = (name, path)
        image, params = self.scaler_left.original, light.extra.get("params") or self._build_params()
        self._set_status(t("status.preparing_view"))

        def worker():
            try:
                full = ANALYSES[name].func(image, params)
            except Exception as e:
                _log.exception("%s falló en %s", name, path)
                full = AnalysisResult(status="error", error=str(e))
            full.extra["params"] = params
            self.root.after(0, lambda: self._on_rehydrated(name, path, full))
        threading.Thread(target=worker, daemon=True).start()

    def _on_rehydrated(self, name: str, path: str, full: AnalysisResult):
        self._rehydrating = None
        if full.status != "ok" or self._results.get(name, {}).get(path) is None:
            return
        self._results[name][path] = full
        self._trim_memory(path)
        if path == self.current_image_path and name == self.active_analysis:
            self._display_result(full, list(full.step_images))
            self._refresh_history()
            if self.status_var.get() == t("status.preparing_view"):
                self._set_status(t("status.ready"))

    def _trim_memory(self, path: str):
        """Solo las 2 últimas fotos vistas guardan imágenes (cada una ~100 MB en 8 MP)."""
        if path in self._full_paths:
            self._full_paths.remove(path)
        self._full_paths.append(path)
        del self._full_paths[:-2]
        for name, per in self._results.items():
            for p, r in per.items():
                if p not in self._full_paths and r.step_images:
                    per[p] = batch.light(r)

    def _on_result(self, name: str, result: AnalysisResult, path: str):
        self.root.config(cursor="")
        self.last_result = result
        if result.status == "error":
            messagebox.showerror(t("common.error"), result.error)
            self._set_status(f"{t('common.error')}: {result.error}")
            return
        names = list(result.step_images.keys())
        # Guardar en cache para restaurar al navegar
        self.results_cache[path]    = result
        self.step_names_cache[path] = names
        self._display_result(result, names, fresh=True)
        self._mark_ran(name)
        self._write_preview()
        self._trim_memory(path)
        self._update_batch_list()
        self._set_status(t("status.done", name=_analysis_label(name),
                           detail=_summary(result.stats, aliases=self._class_aliases(result))))
        if result.extra.get("processing_s") is not None:
            self._set_status(self.status_var.get() + f" ({_duration(result.extra['processing_s'])})")
        reused = result.extra.get("reused")
        if reused:
            items = ", ".join(t(f"step.{k}", k) for k in reused)
            self._set_status(self.status_var.get() + ". " + t("status.reused", items=items))
        self._refresh_history()
        self._log_process(name)

    def _display_result(self, result: AnalysisResult, step_names: list[str], fresh: bool = False):
        """Muestra un resultado. Tras correr: la medición que se recalculó (si solo cambió
        esa); si no, se conserva la vista que se estaba viendo; si no existe, la primera."""
        prev = self.step_names[self.step_idx] if self.step_names and 0 <= self.step_idx < len(self.step_names) else None
        self.last_result = result
        self.step_names = list(step_names)
        self._show_results_ui(True)
        if self._chart_panel:
            self._chart_panel.load(result.measurements)
            full = (getattr(self, "_open_windows", {}) or {}).get("chart_full")
            if full is not None and full.winfo_exists():        # la ventana completa sigue a la foto
                for w in full.winfo_children():
                    if isinstance(w, IntraImageChartPanel):
                        w.load(result.measurements)
        if not self.step_names:
            self._update_table(result)
            return
        changed = result.extra.get("default_view") if fresh else None
        for name in (changed, prev, result.extra.get("first_view")):
            if name in self.step_names:
                self._jump_to_step(self.step_names.index(name))
                return
        self._jump_to_step(len(self.step_names) - 1)

    def _current_view_image(self):
        r = self.last_result
        if not r or not self.step_names or not 0 <= self.step_idx < len(self.step_names):
            return None
        name = self.step_names[self.step_idx]
        img = r.step_images.get(name)
        if img is None and _on_base(r, name):
            img = r.extra.get("base_image")
        if img is None:
            return None
        stem = Path(self.current_image_path).stem if self.current_image_path else None
        return name, self._decorate_fn(r, stem)(name, img)
