"""Lista CAPAS: capas de cada análisis, cambiar de capa y leyenda de la capa."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np

# ANALYSES tiene los análisis registrados (Objetos, Detección…) por clave; AnalysisResult es el
# resultado de un análisis (tablas, vistas, leyendas).
from fenotit.core.analysis.registry import ANALYSES, AnalysisResult
# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras); SPACE (márgenes).
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
# Funciones comunes de la ventana: _on_base dice si una vista se dibuja como foto con marcas;
# _analysis_label da el nombre del análisis para mostrar.
from fenotit.gui.window.common import _on_base, _analysis_label


class LayersMixin:
    def _set_step(self, idx: int):
        """Capa a la vista (posición en step_names)."""
        self.step_idx = idx

    def _layers_need(self) -> int:
        """Alto que piden CAPAS (título + filas) y el inspector, si está abierto."""
        rows = self.history_frame.winfo_reqheight() if self.history_frame.winfo_children() else 0
        insp = self.inspector.winfo_reqheight() if self.inspector.winfo_ismapped() else 0
        return self._layers_title.winfo_reqheight() + max(rows, 22) + 12 + insp

    def _layers_scroll(self):
        canvas = self._layers_canvas
        height = self.history_frame.winfo_reqheight()
        canvas.config(scrollregion=(0, 0, canvas.winfo_width(), height))
        need = height > canvas.winfo_height() + 1
        if need != self._layers_sb.winfo_ismapped():
            if need:
                self._layers_sb.pack(side=tk.RIGHT, fill=tk.Y, before=canvas)
            else:
                self._layers_sb.pack_forget()
                canvas.yview_moveto(0)

    def _layers_wheel(self, event, delta):
        widget = event.widget
        while widget is not None and widget is not self._layers_canvas:
            widget = getattr(widget, "master", None)
        if widget is not None and self._layers_sb.winfo_ismapped():
            self._layers_canvas.yview_scroll(delta, "units")

    def _keep_layers_visible(self):
        """CAPAS nunca desaparece: si con el inspector no cabe, el recorte se achica."""
        if not self.inspector.winfo_ismapped():
            return
        self.root.update_idletasks()
        rows = self.history_frame.winfo_reqheight() if self.history_frame.winfo_children() else 22
        need = self._layers_title.winfo_reqheight() + min(rows, 4 * 22) + 12      # al menos 4 capas
        short = need - self._layers.winfo_height()
        if short > 0 and self.inspector.shrink(short):
            self.root.after(30, self._keep_layers_visible)

    def _prev_step(self):
        if self.last_result and self.step_names:
            self._jump_to_step(max(0, self.step_idx - 1))

    def _next_step(self):
        if self.last_result and self.step_names:
            self._jump_to_step(min(len(self.step_names) - 1, self.step_idx + 1))

    def _mark_colors(self, result: AnalysisResult) -> dict:
        from fenotit.core.pipeline import overlay
        return overlay.resolve(self.project.display.get("style"), auto=result.extra.get("contrast"))

    def _decorate_fn(self, result: AnalysisResult, image_name: str | None = None):
        """Marcas con el estilo elegido y leyenda si está activada (core.pipeline.views)."""
        from fenotit.core.pipeline.views import decorate
        return decorate(result, self.project.display, image_name)

    def _on_display_change(self, _=None):
        self.project.display = {**self.project.display, "legend": bool(self.legend_var.get()),
                                "color_format": self.color_fmt_var.get()}
        if self.last_result and self.step_names:
            self._jump_to_step(self.step_idx)

    def _legend_size(self, factor: float):
        """A− / A+: tamaño de la leyenda (se guarda en el proyecto)."""
        scale = self.project.display.get("legend_scale", 1.0) * factor
        self.project.display = {**self.project.display, "legend_scale": round(min(4.0, max(0.4, scale)), 2)}
        self._on_display_change()

    def _show_step_img(self, name: str, img: np.ndarray):
        marks = None
        if self.last_result is not None:
            ovs = self.last_result.extra.get("overlays") or {}
            keep = []
            shown = Path(self.current_image_path).stem if self.current_image_path else None
            img = self._decorate_fn(self.last_result, shown)(name, img, with_marks=False, box=keep)
            if name in ovs and self.last_result.extra.get("base_image") is not None:
                marks = (ovs[name], self._mark_colors(self.last_result), keep)
        self.scaler_right.set_image(img)
        if self.zoom_ctrl:
            self.zoom_ctrl.set_right(img, marks)
        else:
            self.canvas_right.update_idletasks()
            width = max(self.canvas_right.winfo_width(),  100)
            height = max(self.canvas_right.winfo_height(), 100)
            self.scaler_right.set_canvas_size(width, height)
            self._show_on(self.canvas_right,
                          self.scaler_right.display, "_photo_right")

    def _refresh_history(self):
        """Vistas de la foto actual, agrupadas por análisis (solo los que ya tienen
        resultado). Tocar una vista de otro análisis lo activa y la muestra al instante."""
        for widget in self.history_frame.winfo_children():
            widget.destroy()
        path = self.current_image_path
        if not path:
            return
        done = [name for name in ANALYSES if path in self._results.get(name, {})]
        for name in done:
            current = name == getattr(self, "active_analysis", None)
            if len(done) > 1:
                tk.Label(self.history_frame, text=_analysis_label(name), bg=COLORS["bg_panel"],
                         fg=COLORS["accent"] if current else COLORS["text_muted"],
                         font=FONTS["small_italic"], anchor="w", padx=SPACE["md"]).pack(fill=tk.X, pady=(SPACE["sm"], 0))
            for i, view in enumerate(self._step_names.get(name, {}).get(path, [])):
                sel = current and i == self.step_idx
                bg = COLORS["accent_light"] if sel else COLORS["bg_panel"]
                row = tk.Frame(self.history_frame, bg=bg, cursor="hand2")
                row.pack(fill=tk.X)
                tk.Frame(row, bg=COLORS["accent"] if sel else bg, width=3).pack(side=tk.LEFT, fill=tk.Y)
                lbl = tk.Label(row, text=view, bg=bg, fg=COLORS["text"] if current else COLORS["text_muted"],
                               font=FONTS["small"], anchor="w", padx=SPACE["md"], pady=SPACE["xs"], cursor="hand2")
                lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
                for wdg in (row, lbl):
                    wdg.bind("<Button-1>", lambda event, analysis=name, view=view: self._open_view(analysis, view))
                    wdg.bind("<Enter>", lambda event, row_frame=row, label_widget=lbl: (row_frame.config(bg=COLORS["accent_light"]),
                                                                 label_widget.config(bg=COLORS["accent_light"])))
                    wdg.bind("<Leave>", lambda event, row_frame=row, label_widget=lbl, color=bg: (row_frame.config(bg=color), label_widget.config(bg=color)))
        self.image_list.refit()                # más capas = la lista de fotos cede espacio

    def _step_view(self, delta: int):
        """↑ ↓: capa anterior / siguiente de la foto, pasando de un análisis al otro."""
        path = self.current_image_path
        if not path:
            return
        flat = [(name, view) for name in ANALYSES if path in self._results.get(name, {})
                for view in self._step_names.get(name, {}).get(path, [])]
        if not flat:
            return
        cur = (self.active_analysis, self.step_names[self.step_idx]) \
            if self.step_names and 0 <= self.step_idx < len(self.step_names) else None
        position = flat.index(cur) + delta if cur in flat else 0
        if 0 <= position < len(flat):
            self._open_view(*flat[position])

    def _open_view(self, analysis: str, view: str):
        if analysis != self.active_analysis:
            self.analysis_var.set(_analysis_label(analysis))
            self._on_analysis_selected(None)
        if view in self.step_names:
            self._jump_to_step(self.step_names.index(view))

    def _jump_to_step(self, idx: int):
        if not self.last_result or idx >= len(self.step_names):
            return
        self._set_step(idx)
        name = self.step_names[idx]
        img = self.last_result.step_images.get(name)
        if img is None:                    # resultado del lote: datos listos, imágenes en camino
            if self.scaler_left.has_image:
                # las marcas (contornos, puntos, números) ya vienen en los datos: se dibujan al
                # instante sobre la entrada; las demás vistas se preparan después, en segundo plano
                if self.last_result.extra.get("base_image") is None:
                    self._set_base_image(self.scaler_left.original)
                self._show_step_img(name, self.scaler_left.original)
            self._schedule_rehydrate(600 if _on_base(self.last_result, name) else 30)
        else:
            self._show_step_img(name, img)
        self._update_view_controls(name)
        self.step_label_var.set(
            f"{name}  ({idx+1}/{len(self.step_names)})")
        self._update_table(self.last_result, name)
        self._refresh_history()
        self._refresh_selection()
