"""Ayuda estándar de la app: un "?" pequeño en un círculo junto a lo que explica.
Pasar el mouse = una línea breve; clic = la explicación en un recuadro pequeño junto al
"?" (sin ventana). Las etiquetas nunca llevan aclaraciones: van aquí."""
from __future__ import annotations

import tkinter as tk

from fenotit.gui.theme import COLORS, FONTS
from fenotit.gui.toolbar import Tooltip
from fenotit.i18n import t

SIZE = 15


class HelpIcon(tk.Canvas):

    def __init__(self, parent, title: str, text: str, short: str | None = None, bg: str | None = None):
        short = short or first_sentence(text)
        bg = bg or parent.cget("bg")
        super().__init__(parent, width=SIZE, height=SIZE, bg=bg, highlightthickness=0, cursor="hand2")
        self.title, self.text = title, text
        self._paint(False)
        self.bind("<Enter>", lambda e: self._paint(True))
        self.bind("<Leave>", lambda e: self._paint(False))
        self.bind("<Button-1>", lambda e: help_popup(self, self.title, self.text))
        Tooltip(self, short)

    def _paint(self, hover: bool):
        self.delete("all")
        c = COLORS["accent"]
        self.create_oval(1, 1, SIZE - 2, SIZE - 2, outline=c, fill=c if hover else self.cget("bg"), width=1)
        self.create_text(SIZE / 2 - 0.5, SIZE / 2, text="?", fill="#FFFFFF" if hover else c,
                         font=("Segoe UI", 7, "bold"))


def first_sentence(text: str, limit: int = 90) -> str:
    s = text.strip().split("\n")[0]
    s = s.split(". ")[0].rstrip(".") + "."
    return s if len(s) <= limit else s[:limit - 1].rsplit(" ", 1)[0] + "…"


_open: list = []


def help_popup(widget, title: str, text: str):
    """Recuadro pequeño junto al "?": título, texto y Cerrar (Esc o clic fuera también)."""
    for w in _open:
        if w.winfo_exists():
            w.destroy()
    _open.clear()
    tw = tk.Toplevel(widget)
    tw.wm_overrideredirect(True)
    tw.configure(bg=COLORS["border"])
    tw.attributes("-topmost", True)
    inner = tk.Frame(tw, bg=COLORS["bg_card"], padx=10, pady=8)
    inner.pack(padx=1, pady=1)
    tk.Label(inner, text=title, bg=COLORS["bg_card"], fg=COLORS["accent"],
             font=("Segoe UI", 9, "bold")).pack(anchor="w")
    tk.Label(inner, text=text, bg=COLORS["bg_card"], fg=COLORS["text"], font=("Segoe UI", 8),
             justify="left", wraplength=300).pack(anchor="w", pady=(3, 0))
    tk.Button(inner, text=t("common.close"), bg=COLORS["btn_bg"], fg=COLORS["accent"], font=("Segoe UI", 7),
              relief="flat", padx=6, cursor="hand2", command=tw.destroy).pack(anchor="e", pady=(6, 0))
    tw.update_idletasks()
    w, h = tw.winfo_reqwidth(), tw.winfo_reqheight()
    sw, sh = tw.winfo_screenwidth(), tw.winfo_screenheight()
    x, y = widget.winfo_rootx() + 22, widget.winfo_rooty() + 2
    if x + w > sw - 20:
        x = widget.winfo_rootx() - w - 6
    y = min(y, sh - h - 40)
    tw.wm_geometry(f"+{max(10, x)}+{max(10, y)}")
    tw.bind("<Escape>", lambda e: tw.destroy())

    def lost(_e):
        def check():
            if tw.winfo_exists():
                f = tw.focus_get()
                if f is None or not str(f).startswith(str(tw)):
                    tw.destroy()
        tw.after(150, check)
    tw.bind("<FocusOut>", lost)
    tw.focus_force()
    _open.append(tw)
    return tw
