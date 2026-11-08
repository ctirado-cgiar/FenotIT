"""Atajos de teclado."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class KeysMixin:
    def _bind_shortcuts(self):
        def typing() -> bool:
            w = self.root.focus_get()
            return isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Spinbox, tk.Text))

        def key(handler):
            return lambda e=None: None if typing() else (handler(), "break")[1]
        r = self.root
        r.bind_all("<Control-Return>", lambda e=None: self._run_analysis(False))
        r.bind_all("<Control-Shift-Return>", lambda e=None: self._run_analysis(True))
        r.bind_all("<Control-o>", lambda e=None: self._open_image())
        r.bind_all("<Control-s>", lambda e=None: self._save_project())
        r.bind_all("<Control-e>", lambda e=None: self._export_results())
        r.bind_all("<Control-comma>", lambda e=None: self._open_preferences())
        r.bind_all("<Left>", key(self._prev_image))
        r.bind_all("<Right>", key(self._next_image))

        def key_lists(handler):        # en listas y tablas ↑ ↓ siguen siendo de la lista
            return lambda e=None: None if typing() or isinstance(self.root.focus_get(), (tk.Listbox, ttk.Treeview)) \
                else (handler(), "break")[1]
        r.bind_all("<Up>", key_lists(lambda: self._step_view(-1)))
        r.bind_all("<Down>", key_lists(lambda: self._step_view(1)))
        # Zoom: Ctrl + / − / 0 / 1 (también sin Ctrl y con el teclado numérico)
        for seq, fn in (("plus", self.do_zoom_in), ("equal", self.do_zoom_in), ("KP_Add", self.do_zoom_in),
                        ("minus", self.do_zoom_out), ("KP_Subtract", self.do_zoom_out),
                        ("Key-0", self.do_zoom_fit), ("KP_0", self.do_zoom_fit),
                        ("Key-1", self.do_zoom_100), ("KP_1", self.do_zoom_100)):
            r.bind_all(f"<Control-{seq}>", lambda e=None, f=fn: (f(), "break")[1])
            r.bind_all(f"<{seq}>", key(fn))
        r.bind_all("<Key-z>", key(lambda: self._set_zoom_tool("zoom_area")))
        r.bind_all("<Key-h>", key(lambda: self._set_zoom_tool("pan")))
        r.bind_all("<Key-i>", key(lambda: self._set_zoom_tool("inspect")))
        r.bind_all("<Key-s>", key(lambda: self._toggle_area_tool("select")))
        r.bind_all("<Key-r>", key(lambda: self._toggle_area_tool("include_rect")))
        r.bind_all("<Key-p>", key(lambda: self._toggle_area_tool("include_poly")))
        r.bind_all("<Key-x>", key(lambda: self._toggle_area_tool("exclude_rect")))
        r.bind_all("<Key-X>", key(lambda: self._toggle_area_tool("exclude_poly")))
        r.bind_all("<Key-l>", key(self._toggle_legend))
        r.bind_all("<Key-e>", key(lambda: self._toggle_mark("excluded")))
        r.bind_all("<Key-d>", key(lambda: self._toggle_mark("highlighted")))
        for seq, fn in (("n", self._new_project), ("O", self._open_project), ("S", self._save_project_as),
                        ("d", self._open_metadata), ("q", self._on_close), ("t", self._toggle_results_panel),
                        ("z", self._undo), ("y", self._redo), ("Z", self._redo)):
            r.bind_all(f"<Control-{seq}>", key(fn))         # en un campo de texto, Ctrl+Z es del campo
        r.bind_all("<F1>", lambda e=None: self._show_shortcuts())
        r.bind_all("<Delete>", key(self._on_delete_key))
        r.bind_all("<Escape>", lambda e=None: self._on_escape())

    def _on_delete_key(self):
        """Supr: borra las áreas seleccionadas; si no hay, en la lista quita la foto."""
        if self.areas and self.areas.delete_selected():
            return "break"

    def _on_escape(self):
        if self.areas and self.areas.active:
            self.areas.stop()
        elif self.zoom_ctrl and self.zoom_ctrl.tool != "pan":
            self.zoom_ctrl.set_tool("pan")
