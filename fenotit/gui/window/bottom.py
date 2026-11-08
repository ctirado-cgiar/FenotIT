"""Panel de abajo: pestañas Tabla y Gráficos, su alto y el gráfico llevado arriba."""
from __future__ import annotations

import tkinter as tk

from fenotit import settings
from fenotit.gui.style.theme import COLORS


class BottomMixin:
    def _toggle_results_panel(self):
        """Oculta o muestra la tabla y los gráficos (las pestañas quedan a la vista)."""
        self._results_visible = not self._results_visible
        self._layout_results()
        self.root.after(50, self._refit_images)

    def _select_bottom_tab(self, key: str):
        if key == "charts" and self._chart_dock:          # el gráfico está arriba: vuelve aquí
            self._undock_chart()
            return
        if self._results_visible and self._bottom_tab == key:
            return
        reopen = not self._results_visible
        self._bottom_tab, self._results_visible = key, True
        self._layout_results()
        if reopen:
            self.root.after(50, self._refit_images)

    RESULTS_MIN_H = 110

    def _results_drag_start(self, e):
        h0 = int(self._bottom_body.cget("height"))
        top = h0 + self._canvas_row.winfo_height() - 180      # las imágenes conservan al menos 180 px
        self._drag = (e.y_root, h0, max(top, self.RESULTS_MIN_H))

    def _results_drag(self, e):
        y0, h0, top = self._drag
        h = int(min(max(h0 - (e.y_root - y0), self.RESULTS_MIN_H), top))
        if h != int(self._bottom_body.cget("height")):
            self._bottom_body.configure(height=h)

    def _results_drag_end(self, _e):
        settings.set("results_height", int(self._bottom_body.cget("height")))
        self.root.update_idletasks()
        self.root.after(80, self._refit_images)

    def _paint_tabs(self):
        for key, lbl in self._tab_labels.items():
            on = self._results_visible and key == self._bottom_tab and not (key == "charts" and self._chart_dock)
            lbl.config(bg=COLORS["bg_card"] if on else COLORS["bg_panel"],
                       fg=COLORS["accent"] if on else COLORS["text_muted"])

    def _layout_results(self):
        """Pestañas Tabla/Gráficos solo si la foto tiene resultados; abiertas o solo pestañas."""
        wrap = self._bottom_wrap
        for w in (self._bottom_grip, self._bottom_bar, self._bottom_body):
            w.pack_forget()
        if not self._results_shown:
            wrap.pack_forget()
            return
        wrap.pack(fill=tk.X, expand=False, padx=6, pady=(0, 2), after=self._canvas_row)
        if self._results_visible:
            self._bottom_grip.pack(fill=tk.X)
            self._bottom_bar.pack(fill=tk.X)
            self._bottom_body.pack(fill=tk.BOTH)
            for key, f in self._tab_frames.items():
                if key == self._bottom_tab:
                    f.pack(fill=tk.BOTH, expand=True)
                else:
                    f.pack_forget()
            if self._bottom_tab == "charts":
                self._chart_panel.show()
        else:
            self._bottom_bar.pack(fill=tk.X, pady=(2, 0))
        self._paint_tabs()

    def _show_results_ui(self, on: bool):
        if on == self._results_shown:
            return
        self._results_shown = on
        if not on:
            self._view_ctrl.place_forget()
        self._layout_results()
        self._layout_sides()
        self.root.after(50, self._refit_images)

    def _dock_chart(self):
        """El gráfico de la foto pasa al lugar de la imagen oculta (o de la Entrada) y el
        panel de abajo se cierra; el ojo del gráfico lo devuelve como estaba."""
        if self._chart_dock or not self._results_shown:
            return
        hidden = [s for s, h in self._hidden.items() if h]
        slot = hidden[0] if len(hidden) == 1 else "input"
        self._dock_saved = (dict(self._hidden), self._results_visible)
        self._hidden[slot] = True
        self._chart_dock = slot
        edge = tk.LEFT if slot == "input" else tk.RIGHT
        self._chart_title.pack_forget()
        self._chart_eye.pack_forget()
        self._chart_title.pack(side=edge, padx=2)
        self._chart_eye.pack(side=edge)
        self._chart_panel.pack_forget()
        self._chart_panel.lift()
        self._fit_docked_chart()
        self._dock_btn.pack_forget()
        self._results_visible = False
        if self._bottom_tab == "charts":
            self._bottom_tab = "table"
        self._layout_sides()
        self._layout_results()
        self.root.after(80, self._after_dock)

    def _fit_docked_chart(self):
        """Arriba, el gráfico mide lo mismo que la foto ajustada a su recuadro (no todo el
        recuadro), centrado como ella."""
        if not self._chart_dock:
            return
        host = self._chart_host
        hw, hh = host.winfo_width(), host.winfo_height()
        if hw < 50 or hh < 50:
            return
        w, h = hw, hh
        if self.scaler_left.has_image:
            ih, iw = self.scaler_left.original.shape[:2]
            k = min(hw / iw, hh / ih)
            w, h = max(300, min(hw, round(iw * k))), max(220, min(hh, round(ih * k)))
        self._chart_panel.place(in_=host, relx=0.5, rely=0.5, anchor="center", width=w, height=h)
        self._chart_panel.lift()

    def _undock_chart(self):
        if not self._chart_dock:
            return
        self._chart_panel.place_forget()
        hidden, _visible = self._dock_saved
        self._hidden, self._chart_dock = hidden, None
        self._chart_panel.pack_forget()
        self._chart_panel.pack(in_=self._tab_frames["charts"], fill=tk.BOTH, expand=True)
        self._chart_panel.lift()
        self._dock_btn.pack(side=tk.RIGHT, padx=(2, 6), before=self._save_btn)
        self._results_visible, self._bottom_tab = True, "charts"
        self._layout_sides()
        self._layout_results()
        self.root.after(80, self._after_dock)

    def _after_dock(self):
        self._refit_images()
        self._chart_panel.redraw()

    def _after_panel_toggle(self):
        self.root.after(50, self._refit_images)
        self.root.after(60, self._update_stroke_tool)
