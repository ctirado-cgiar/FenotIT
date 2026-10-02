"""Pantalla de inicio (cuando no hay fotos): nuevo proyecto con fotos o carpeta, abrir
un proyecto y los recientes. Se pone encima de Entrada/Resultado."""
from __future__ import annotations

import datetime as dt
import tkinter as tk
from pathlib import Path

from PIL import ImageTk

from fenotit import APP_NAME, __version__, settings
from fenotit.gui.toolbar import icon
from fenotit.i18n import t


class StartScreen(tk.Frame):

    def __init__(self, parent, colors: dict, fonts: dict, on_photos, on_folder, on_open, on_recent):
        super().__init__(parent, bg=colors["bg"])
        self.c, self.f = colors, fonts
        self._cb = (on_photos, on_folder, on_open, on_recent)
        self._icons = {n: ImageTk.PhotoImage(icon(n, 36, colors["accent"])) for n in ("photo", "folder", "project")}
        self._body = tk.Frame(self, bg=colors["bg"])
        self._body.place(relx=0.5, rely=0.42, anchor="center")
        self.refresh()

    def refresh(self):
        """Vuelve a armar la lista de recientes (al mostrarse)."""
        c, b = self.c, self._body
        for w in b.winfo_children():
            w.destroy()
        on_photos, on_folder, on_open, on_recent = self._cb
        tk.Label(b, text=APP_NAME, bg=c["bg"], fg=c["accent"], font=("Segoe UI", 20, "bold")).pack(anchor="w")
        tk.Label(b, text=t("start.subtitle"), bg=c["bg"], fg=c["text_muted"],
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 14))

        tk.Label(b, text=t("start.new"), bg=c["bg"], fg=c["text"], font=("Segoe UI", 9, "bold")).pack(anchor="w")
        row = tk.Frame(b, bg=c["bg"])
        row.pack(anchor="w", pady=(4, 16))
        self._tile(row, "photo", t("start.photos"), t("start.photos_hint"), on_photos)
        self._tile(row, "folder", t("start.folder"), t("start.folder_hint"), on_folder)
        self._tile(row, "project", t("start.open"), t("start.open_hint"), on_open)

        tk.Label(b, text=t("start.recent"), bg=c["bg"], fg=c["text"], font=("Segoe UI", 9, "bold")).pack(anchor="w")
        box = tk.Frame(b, bg=c["bg_card"], highlightthickness=1, highlightbackground=c["border"])
        box.pack(fill=tk.X, pady=(4, 0))
        recent = settings.recent_projects()
        if not recent:
            tk.Label(box, text=t("start.no_recent"), bg=c["bg_card"], fg=c["text_muted"],
                     font=("Segoe UI", 8), padx=10, pady=8).pack(anchor="w")
        for path in recent:
            self._recent_row(box, Path(path), on_recent)

    def _tile(self, parent, ico, title, hint, cmd):
        c = self.c
        tile = tk.Frame(parent, bg=c["bg_card"], highlightthickness=1, highlightbackground=c["border"],
                        cursor="hand2", width=170, height=112)
        tile.pack(side=tk.LEFT, padx=(0, 10))
        tile.pack_propagate(False)
        parts = [tk.Label(tile, image=self._icons[ico], bg=c["bg_card"]),
                 tk.Label(tile, text=title, bg=c["bg_card"], fg=c["text"], font=("Segoe UI", 9, "bold")),
                 tk.Label(tile, text=hint, bg=c["bg_card"], fg=c["text_muted"], font=("Segoe UI", 7),
                          wraplength=150)]
        parts[0].pack(pady=(12, 2))
        parts[1].pack()
        parts[2].pack()

        def paint(on):
            bg = c["accent_light"] if on else c["bg_card"]
            tile.config(bg=bg, highlightbackground=c["accent"] if on else c["border"])
            for p in parts:
                p.config(bg=bg)
        for w in [tile] + parts:
            w.bind("<Enter>", lambda e: paint(True))
            w.bind("<Leave>", lambda e: paint(False))
            w.bind("<Button-1>", lambda e: cmd())

    def _recent_row(self, parent, path: Path, cmd):
        c = self.c
        row = tk.Frame(parent, bg=c["bg_card"], cursor="hand2")
        row.pack(fill=tk.X)
        try:
            when = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            when = ""
        name = tk.Label(row, text=path.stem, bg=c["bg_card"], fg=c["accent"], font=("Segoe UI", 9, "bold"),
                        anchor="w", padx=10)
        name.grid(row=0, column=0, sticky="w", pady=(5, 0))
        date = tk.Label(row, text=when, bg=c["bg_card"], fg=c["text_muted"], font=("Segoe UI", 7), padx=10)
        date.grid(row=0, column=1, sticky="e", pady=(5, 0))
        where = tk.Label(row, text=str(path.parent), bg=c["bg_card"], fg=c["text_muted"], font=("Segoe UI", 7),
                         anchor="w", padx=10)
        where.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row.columnconfigure(0, weight=1)
        parts = (row, name, date, where)

        def paint(on):
            for w in parts:
                w.config(bg=c["accent_light"] if on else c["bg_card"])
        for w in parts:
            w.bind("<Enter>", lambda e: paint(True))
            w.bind("<Leave>", lambda e: paint(False))
            w.bind("<Button-1>", lambda e: cmd(str(path)))
