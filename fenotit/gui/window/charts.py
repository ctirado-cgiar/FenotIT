"""Gráficos: comparar fotos, colores 3D, pantalla completa y guardar."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from fenotit.gui.charts.panels import IntraImageChartPanel, BatchChartWindow
from fenotit.gui.style.components import button
from fenotit.gui.style.theme import COLORS, SPACE
from fenotit.i18n import t


class ChartsMixin:
    def _show_batch_chart(self):
        """Ventana de gráficos del lote (análisis elegido)."""
        results = {p: r for p, r in self.results_cache.items() if r is not None and r.status == "ok"}
        if not results:
            messagebox.showinfo(t("msg.no_data_title"), t("msg.no_data_batch"), parent=self.root)
            return
        self._replace_window("batch_chart", lambda: BatchChartWindow(
            self.root, results, meta={p: self._meta_row(p) for p in results},
            groups=self._meta_columns(), on_pick=self._goto_object))

    def _color_images(self, paths) -> list:
        """[(ruta, Image_ID, nombre, tablas)] con resultado del análisis elegido."""
        ids = {p: i for i, p in enumerate(self.batch_paths, 1)}
        out = []
        for p in paths:
            r = self.results_cache.get(p)
            if r is not None and r.status == "ok" and p in ids:
                out.append((p, ids[p], Path(p).name, r.extra.get("tables") or {}))
        return out

    def _show_color3d(self, scope: str = "image"):
        from fenotit.gui.charts.color3d import Color3DWindow
        current = self._color_images([self.current_image_path] if self.current_image_path else [])
        batch = self._color_images(self.batch_paths)
        self._replace_window("colors", lambda: Color3DWindow(
            self.root, current, batch, scope=scope, on_goto=self._goto_object, thumb=self._object_thumb))

    def _replace_window(self, key: str, build):
        """Una sola ventana de cada tipo: la abierta se cierra y la nueva (con los datos de
        ahora) toma su lugar y tamaño."""
        wins = self.__dict__.setdefault("_open_windows", {})
        old = wins.get(key)
        geom = None
        if old is not None and old.winfo_exists():
            geom = old.geometry()
            old.destroy()
        win = build()
        if win is not None:
            if geom:
                win.geometry(geom)
            wins[key] = win

    def _chart_fullscreen(self):
        """El gráfico de la foto en una ventana maximizada (una sola), con las mismas opciones."""
        src = self._chart_panel
        if not getattr(src, "_rows", None):
            return

        def build():
            win = tk.Toplevel(self.root)
            win.title(t("chart.fullscreen"))
            win.configure(bg=COLORS["bg_card"])
            win.geometry(f"{win.winfo_screenwidth()}x{win.winfo_screenheight() - 60}+0+0")
            try:
                win.state("zoomed")                  # Windows
            except tk.TclError:
                win.attributes("-zoomed", True)      # Linux
            panel = IntraImageChartPanel(win, colors=COLORS,
                                         on_pick=lambda oid: self._goto_object(self.current_image_path, oid))
            if hasattr(panel, "bar"):
                button(panel.bar, text=t("chart.save"), command=lambda: panel.save_figure(), kind="secondary", size="small").pack(side=tk.RIGHT, padx=SPACE["md"], pady=SPACE["sm"])
                panel.chooser.chosen = {k: list(v) for k, v in src.chooser.chosen.items()}
                panel.chooser.count = dict(src.chooser.count)
                panel.chooser.kind.set(src.chooser.kind.get())
            panel.pack(fill=tk.BOTH, expand=True)
            win.bind("<Escape>", lambda e: win.destroy())
            win.update_idletasks()
            panel.load(src._rows)
            win.after(100, panel.redraw)
            return win
        self._replace_window("chart_full", build)

    def _save_intra_chart(self, with_image: bool = False):
        """Guarda el gráfico de la foto; opcional, también la imagen de la vista actual."""
        if self._chart_panel:
            self._chart_panel.save_figure(self._current_view_image() if with_image else None)
