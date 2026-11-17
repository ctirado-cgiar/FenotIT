"""Ayuda estándar de la app: un "?" pequeño en un círculo junto a lo que explica.
Pasar el mouse = una línea breve; clic = la explicación en un recuadro pequeño junto al
"?" (sin ventana). Las etiquetas nunca llevan aclaraciones: van aquí."""
from __future__ import annotations

import tkinter as tk

from fenotit.gui.style.components import button
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.gui.widgets.toolbar import Tooltip
from fenotit.i18n import t

SIZE = 15


class HelpIcon(tk.Canvas):

    def __init__(self, parent, title: str, text: str, short: str | None = None, bg: str | None = None):
        short = short or first_sentence(text)
        bg = bg or parent.cget("bg")
        super().__init__(parent, width=SIZE, height=SIZE, bg=bg, highlightthickness=0, cursor="hand2")
        self.title, self.text = title, text
        self._paint(False)
        self.bind("<Enter>", lambda event: self._paint(True))
        self.bind("<Leave>", lambda event: self._paint(False))
        self.bind("<Button-1>", lambda event: help_popup(self, self.title, self.text))
        Tooltip(self, short)

    def _paint(self, hover: bool):
        self.delete("all")
        color = COLORS["accent"] if hover else COLORS["help_icon"]
        self.create_oval(1, 1, SIZE - 2, SIZE - 2, outline=color, fill=color if hover else self.cget("bg"), width=1)
        self.create_text(SIZE / 2 - 0.5, SIZE / 2, text="?", fill=COLORS["white"] if hover else color,
                         font=FONTS["tiny_bold"])


def first_sentence(text: str, limit: int = 90) -> str:
    line = text.strip().split("\n")[0]
    line = line.split(". ")[0].rstrip(".") + "."
    return line if len(line) <= limit else line[:limit - 1].rsplit(" ", 1)[0] + "…"


_open: list = []


def help_popup(widget, title: str, text: str):
    """Recuadro pequeño junto al "?": título, texto y Cerrar (Esc o clic fuera también)."""
    for popup in _open:
        if popup.winfo_exists():
            popup.destroy()
    _open.clear()
    window = tk.Toplevel(widget)
    window.wm_overrideredirect(True)
    window.configure(bg=COLORS["border"])
    window.attributes("-topmost", True)
    inner = tk.Frame(window, bg=COLORS["bg_card"], padx=SPACE["lg"], pady=SPACE["md"])
    inner.pack(padx=1, pady=1)
    tk.Label(inner, text=title, bg=COLORS["bg_card"], fg=COLORS["accent"],
             font=FONTS["body_bold"]).pack(anchor="w")
    tk.Label(inner, text=text, bg=COLORS["bg_card"], fg=COLORS["text"], font=FONTS["small"],
             justify="left", wraplength=300).pack(anchor="w", pady=(SPACE["sm"], 0))
    button(inner, text=t("common.close"), command=window.destroy, kind="secondary", size="small").pack(anchor="e", pady=(SPACE["md"], 0))
    window.update_idletasks()
    width, height = window.winfo_reqwidth(), window.winfo_reqheight()
    screen_width, screen_height = window.winfo_screenwidth(), window.winfo_screenheight()
    x, y = widget.winfo_rootx() + 22, widget.winfo_rooty() + 2
    if x + width > screen_width - 20:
        x = widget.winfo_rootx() - width - 6
    y = min(y, screen_height - height - 40)
    window.wm_geometry(f"+{max(10, x)}+{max(10, y)}")
    window.bind("<Escape>", lambda event: window.destroy())

    def lost(_event):
        def check():
            if window.winfo_exists():
                focused = window.focus_get()
                if focused is None or not str(focused).startswith(str(window)):
                    window.destroy()
        window.after(150, check)
    window.bind("<FocusOut>", lost)
    window.focus_force()
    _open.append(window)
    return window
