"""Ventanas de la aplicación: preferencias, idioma, atajos, registro y Acerca de."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from fenotit import __version__, log, settings
from fenotit import i18n
from fenotit.gui.style.theme import COLORS, FONTS
from fenotit.gui.window.common import _assets
from fenotit.i18n import t


_log = log.get("gui.window.dialogs")


class DialogsMixin:
    def _open_preferences(self):
        from fenotit.gui.dialogs import preferences as preferences_dialog
        self._single("preferences", lambda: preferences_dialog.show(self.root, self.project.max_mpx,
                                                                     self._apply_preferences))

    def _apply_preferences(self, v: dict):
        for key in ("workers", "show_table_on_run", "autosave_min"):
            settings.set(key, v[key])
        settings.set("default_max_mpx", v["max_mpx"] or 0)
        self._set_working_res(v["max_mpx"])
        restart = v["language"] != i18n.current() or v["theme"] != (settings.get("theme") or "classic")
        settings.set("theme", v["theme"])
        if v["language"] != i18n.current():
            i18n.set_language(v["language"])
        if restart:
            self._restart()

    def _set_working_res(self, value: float | None):
        """Resolución de trabajo: las correcciones usan la foto completa; el análisis, a lo
        más esta resolución. Se guarda en el proyecto."""
        if value == self.project.max_mpx:
            return
        self._remember("undo.working_res")
        self.project.max_mpx = value
        self._invalidate_results()
        if self.current_image_path:
            self._load_single(self.current_image_path)
        self._update_corr_indicator()

    @staticmethod
    def _default_max_mpx() -> float | None:
        v = settings.get("default_max_mpx", 50.0)
        return float(v) if v else None

    def _open_log_folder(self):
        folder = Path(log.log_file()).parent
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)            # noqa: S606 (solo Windows)
            else:
                subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", str(folder)])
        except Exception:
            _log.exception("No se pudo abrir %s", folder)
            messagebox.showinfo(t("help.log"), str(folder), parent=self.root)

    def _show_shortcuts(self):
        groups = [
            (t("menu.project"), [("Ctrl+N", t("menu.new_project")), ("Ctrl+Shift+O", t("menu.open_project")),
                              ("Ctrl+S", t("menu.save_project")), ("Ctrl+Shift+S", t("menu.save_project_as")),
                              ("Ctrl+E", t("menu.export")), ("Ctrl+Q", t("menu.exit"))]),
            (t("menu.images"), [("Ctrl+O", t("menu.add_images")), ("Ctrl+D", t("menu.metadata"))]),
            (t("menu.edit"), [("Ctrl+Z", t("edit.undo")), ("Ctrl+Y", t("edit.redo"))]),
            (t("menu.analysis"), [("Ctrl+Enter", t("run.current")), ("Ctrl+Shift+Enter", t("run.all"))]),
            (t("menu.areas"), [("R", t("roi.area_rect")), ("P", t("roi.area_polygon")), ("X", t("roi.exclude_rect")),
                               ("Shift+X", t("roi.exclude_polygon")), ("S", t("roi.select")),
                               ("Supr", t("help.key_delete"))]),
            (t("help.inspector"), [("I", t("view.inspect")), ("E", t("insp.exclude") + " / " + t("insp.include")),
                                   ("D", t("insp.highlight") + " / " + t("insp.unhighlight"))]),
            (t("menu.view"), [("← →", t("help.key_images")), ("↑ ↓", t("help.key_layers")),
                              ("Ctrl+ +  −", t("help.key_zoom")), ("Ctrl+0", t("view.zoom_fit")),
                              ("Ctrl+1", t("view.zoom_100")), ("Z", t("view.zoom_area")), ("H", t("view.pan")),
                              ("L", t("view.toggle_legend")), ("Ctrl+T", t("view.results_panel")),
                              (t("help.mouse_wheel"), t("help.key_zoom")),
                              ("Shift + " + t("help.mouse_wheel"), t("help.key_pan")), ("Esc", t("help.key_esc"))]),
        ]

        import re

        def build():
            win = tk.Toplevel(self.root)
            win.title(t("help.shortcuts"))
            win.configure(bg=COLORS["bg_card"])
            win.transient(self.root)
            win.resizable(False, False)
            tk.Frame(win, bg=COLORS["accent"], height=4).pack(fill=tk.X)
            body = tk.Frame(win, bg=COLORS["bg_card"])
            body.pack(padx=20, pady=(10, 14))
            col = None
            for i, (title, rows) in enumerate(groups):
                if i in (0, 3):                       # dos columnas
                    col = tk.Frame(body, bg=COLORS["bg_card"])
                    col.pack(side=tk.LEFT, anchor="n", padx=(0, 28))
                tk.Label(col, text=title.upper(), bg=COLORS["bg_card"], fg=COLORS["accent"],
                         font=FONTS["small_bold"]).pack(anchor="w", pady=(8, 2))
                g = tk.Frame(col, bg=COLORS["bg_card"])
                g.pack(anchor="w", fill=tk.X)
                g.columnconfigure(0, minsize=150 if i < 3 else 170)
                for r, (k, v) in enumerate(rows):
                    v = re.sub(r"^[^\w(]+", "", v).rstrip("…").strip()          # sin íconos ni "…"
                    v = v[:1].upper() + v[1:]
                    tk.Label(g, text=k, bg=COLORS["bg_panel"], fg=COLORS["text"], font=FONTS["small"],
                             padx=5).grid(row=r, column=0, sticky="w", pady=1)
                    tk.Label(g, text=v, bg=COLORS["bg_card"], fg=COLORS["text"], font=FONTS["small"]).grid(
                        row=r, column=1, sticky="w", padx=(8, 0))
            win.bind("<Escape>", lambda e: win.destroy())
            return win
        self._single("shortcuts", build)

    def _single(self, key: str, build):
        """Una sola ventana de cada tipo: si ya está abierta, se trae al frente."""
        wins = self.__dict__.setdefault("_open_windows", {})
        win = wins.get(key)
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            win.focus_force()
            return win
        win = build()
        if win is not None:
            wins[key] = win
        return win

    def _choose_language(self):
        self._single("language", self._language_window)

    def _language_window(self):
        langs = i18n.available()
        win = tk.Toplevel(self.root)
        win.title(t("lang.title"))
        win.configure(bg=COLORS["bg_card"])
        win.resizable(False, False)
        win.transient(self.root)
        tk.Frame(win, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        body = tk.Frame(win, bg=COLORS["bg_card"])
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        var = tk.StringVar(value=i18n.current())
        for code, name in langs.items():
            tk.Radiobutton(body, text=name, variable=var, value=code,
                           bg=COLORS["bg_card"], fg=COLORS["text"],
                           selectcolor=COLORS["bg_panel"],
                           activebackground=COLORS["bg_card"],
                           font=FONTS["body"]).pack(anchor="w")

        def apply():
            code = var.get()
            win.destroy()
            if code == i18n.current():
                return
            i18n.set_language(code)
            self._restart()

        tk.Button(body, text="OK", command=apply,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=12).pack(anchor="e", pady=(12, 0))
        win.update_idletasks()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = max(win.winfo_reqwidth(), 260), win.winfo_reqheight()
        win.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        return win

    def _system_info(self) -> str:
        try:
            import ctypes
            scale = ctypes.windll.shcore.GetScaleFactorForDevice(0)
        except Exception:
            scale = round(self.root.winfo_fpixels("1i") / 96 * 100)
        return t("about.system", version=__version__, python=platform.python_version(),
                 os=f"{platform.system()} {platform.release()}",
                 screen=f"{self.root.winfo_screenwidth()}×{self.root.winfo_screenheight()}",
                 scale=scale, log=log.log_file())

    def _about(self):
        from fenotit.gui.dialogs import about as about_dialog
        self._single("about", lambda: about_dialog.show(self.root, _assets() / "logo.ico", self._system_info()))
