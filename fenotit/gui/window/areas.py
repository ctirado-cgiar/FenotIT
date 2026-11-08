"""Áreas de análisis y zonas excluidas."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

from fenotit.gui.style.theme import COLORS, FONTS
from fenotit.i18n import t


class AreasMixin:
    def _set_roi_mode(self, mode: str):
        """Herramienta de áreas; al terminar la forma se vuelve a mover la imagen
        (la de seleccionar queda activa hasta Esc)."""
        if not self.areas or not self.current_image_path:
            return
        self.areas.set_mode(mode)
        if self.zoom_ctrl:
            if self.zoom_ctrl.tool != "pan":
                self.zoom_ctrl.set_tool("pan")
            self.zoom_ctrl._update_cursor(self.canvas_left)
        self._paint_area_tools()
        hint = {"include_poly": "roi.hint_polygon", "exclude_poly": "roi.hint_polygon",
                "select": "roi.hint_select"}.get(mode, "roi.hint_rect")
        self._set_status(t(hint))

    def _on_roi_tool_done(self):
        if self.zoom_ctrl:
            self.zoom_ctrl._update_cursor(self.canvas_left)
        self._paint_area_tools()
        self._set_status(t("status.ready"))

    def _clear_roi(self):
        """Borra las áreas de esta foto (las demás no cambian)."""
        if self.areas and self.current_image_path:
            self.areas.stop()
            self.areas.clear()

    def _on_areas_change(self, shapes: list):
        """Lo que se dibuja o borra queda como áreas propias de esta foto."""
        self._remember("undo.areas")
        if self.current_image_path:
            self.project.set_roi(self.current_image_path, shapes)
        self._update_area_scope()
        self._show_preview()

    def _apply_areas_to_all(self):
        path = self.current_image_path
        if not path:
            return
        n = self.project.others_with_own_roi(path)
        if n and not messagebox.askyesno(t("roi.apply_all"), t("roi.apply_all_warn", n=n), parent=self.root):
            return
        self._remember("undo.apply_all")
        self.project.apply_roi_to_all(self.project.roi_for(path))
        self._update_area_scope()
        self._set_status(t("roi.applied_all", n=len(self.project.images)))

    def _build_area_tools(self):
        """Abajo a la izquierda de la entrada: de qué fotos son las áreas (esta foto /
        todas) con "Aplicar a todas" y "Borrar". Las herramientas están en el menú Áreas."""
        bg = COLORS["bg_card"]
        self._area_btns = {}
        scope = tk.Frame(self.canvas_left, bg=bg, highlightthickness=1, highlightbackground=COLORS["border"])
        self._area_scope = scope
        self._area_scope_var = tk.StringVar()
        tk.Label(scope, textvariable=self._area_scope_var, bg=bg, fg=COLORS["text"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(6, 4))
        self._apply_all_btn = tk.Button(scope, text=t("roi.apply_all_short"), command=self._apply_areas_to_all,
                                        bg=COLORS["accent"], fg=COLORS["on_accent"], activebackground=COLORS["accent_dark"],
                                        activeforeground=COLORS["on_accent"], relief="flat", bd=0, font=FONTS["small"],
                                        cursor="hand2", padx=8, pady=1)      # botón lleno: se ve que es una acción
        self._clear_area_btn = tk.Button(scope, text=t("roi.clear_short"), command=self._clear_roi, bg=bg,
                                         fg=COLORS["text_muted"], relief="flat", bd=0, font=FONTS["small"],
                                         cursor="hand2", padx=4)
        self._clear_area_btn.pack(side=tk.RIGHT, padx=(0, 4))
        tk.Button(scope, text=t("roi.edit_short"), command=self._edit_areas, bg=bg, fg=COLORS["accent"],
                  relief="flat", bd=0, font=FONTS["small"], cursor="hand2", padx=4).pack(side=tk.RIGHT)

    def _edit_areas(self):
        """Seleccionar con todas las áreas de la foto elegidas: se ven sus vértices para editarlas."""
        if not self.areas:
            return
        self._set_roi_mode("select")
        self.areas.selected = set(range(len(self.areas.shapes)))
        self.areas.redraw()
        self._set_status(t("roi.hint_edit"))

    def _toggle_area_tool(self, mode: str):
        if self.areas and self.areas.mode == mode:
            self.areas.stop()
        else:
            self._set_roi_mode(mode)

    def _paint_area_tools(self):
        for mode, b in getattr(self, "_area_btns", {}).items():
            b.select(bool(self.areas) and self.areas.mode == mode)

    def _update_area_scope(self):
        """Indicador abajo a la izquierda, solo si esta foto tiene áreas: si son suyas o
        de todas las fotos, y "Aplicar a todas" cuando son suyas."""
        path = self.current_image_path
        shapes = self.project.roi_for(path) if path else []
        if not shapes:
            self._area_scope.place_forget()
            return
        own = self.project.has_own_roi(path)
        self._area_scope_var.set(t("roi.scope_image" if own else "roi.scope_all", n=len(shapes)))
        if own and len(self.project.images) > 1:
            self._apply_all_btn.pack(side=tk.LEFT, before=self._clear_area_btn, padx=(2, 4), pady=3)
        else:
            self._apply_all_btn.pack_forget()
        self._area_scope.place(x=6, rely=1.0, y=-6, anchor="sw")
