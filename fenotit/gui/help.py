"""Ayuda estándar de la app: un "?" pequeño en un círculo junto a lo que explica.
Pasar el mouse = una línea (si hay); clic = la explicación completa en una ventana.
Las etiquetas nunca llevan aclaraciones: van aquí."""
from __future__ import annotations

import tkinter as tk

from fenotit.gui.theme import COLORS, FONTS
from fenotit.gui.toolbar import Tooltip
from fenotit.i18n import t

SIZE = 15


class HelpIcon(tk.Canvas):

    def __init__(self, parent, title: str, text: str, short: str | None = None, bg: str | None = None):
        bg = bg or parent.cget("bg")
        super().__init__(parent, width=SIZE, height=SIZE, bg=bg, highlightthickness=0, cursor="hand2")
        self.title, self.text = title, text
        self._paint(False)
        self.bind("<Enter>", lambda e: self._paint(True))
        self.bind("<Leave>", lambda e: self._paint(False))
        self.bind("<Button-1>", lambda e: help_box(self.winfo_toplevel(), self.title, self.text))
        Tooltip(self, short or t("help.more"))

    def _paint(self, hover: bool):
        self.delete("all")
        c = COLORS["accent"]
        self.create_oval(1, 1, SIZE - 2, SIZE - 2, outline=c, fill=c if hover else self.cget("bg"), width=1)
        self.create_text(SIZE / 2 - 0.5, SIZE / 2, text="?", fill="#FFFFFF" if hover else c,
                         font=("Segoe UI", 7, "bold"))


def help_box(parent, title: str, text: str):
    win = tk.Toplevel(parent)
    win.title(title)
    win.configure(bg=COLORS["bg_card"])
    win.transient(parent)
    tk.Frame(win, bg=COLORS["accent"], height=4).pack(fill=tk.X)
    tk.Label(win, text=title, bg=COLORS["bg_card"], fg=COLORS["accent"],
             font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=18, pady=(12, 4))
    tk.Label(win, text=text, bg=COLORS["bg_card"], fg=COLORS["text"], font=FONTS["body"], justify="left",
             wraplength=470, anchor="w").pack(fill=tk.X, padx=18, pady=(0, 6))
    row = tk.Frame(win, bg=COLORS["bg_card"])
    row.pack(fill=tk.X, padx=16, pady=10)
    tk.Button(row, text=t("common.close"), command=win.destroy, bg=COLORS["accent"], fg="#FFFFFF", relief="flat",
              font=("Segoe UI", 9, "bold"), cursor="hand2", padx=14).pack(side=tk.RIGHT)
    win.bind("<Escape>", lambda e: win.destroy())
    win.update_idletasks()
    w, h = win.winfo_reqwidth(), win.winfo_reqheight()
    win.geometry(f"+{parent.winfo_rootx() + max(0, (parent.winfo_width() - w) // 2)}"
                 f"+{parent.winfo_rooty() + max(0, (parent.winfo_height() - h) // 4)}")
    win.grab_set()
