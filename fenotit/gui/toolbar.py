"""Botones de herramienta con ayuda emergente y barra de zoom."""
from __future__ import annotations

import tkinter as tk

from PIL import ImageTk

from fenotit.gui.icons import icon
from fenotit.gui.theme import COLORS, FONTS


class Tooltip:
    """Texto corto al pasar el mouse."""

    def __init__(self, widget: tk.Widget, text: str, delay: int = 450):
        self.widget, self.text, self.delay = widget, text, delay
        self._after = self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._after = self.widget.after(self.delay, self._show)

    def _show(self):
        if self._tip:
            return
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tk.Label(tw, text=self.text, bg=COLORS["bg_tooltip"], fg=COLORS["text"], relief="solid", bd=1,
                 font=FONTS["small"], padx=6, pady=2).pack()
        tw.update_idletasks()
        x = min(x, tw.winfo_screenwidth() - tw.winfo_reqwidth() - 4)
        tw.wm_geometry(f"+{x}+{y}")

    def _hide(self, _e=None):
        if self._after:
            self.widget.after_cancel(self._after)
            self._after = None
        if self._tip:
            self._tip.destroy()
            self._tip = None


def zoom_bar(parent, canvas, colors: dict, extra=()):
    """Barra de zoom para ventanas con imagen (mismos íconos y atajos que la principal).
    `canvas` es un ZoomableCanvas. Devuelve la etiqueta del %."""
    top = parent.winfo_toplevel()
    size = max(16, int(round(16 * top.winfo_fpixels("1i") / 96)))
    kw = dict(bg=colors["bg_card"], hover=colors["btn_hover"], active=colors["accent_light"],
              size=size, color=colors["accent"])
    from fenotit.i18n import t

    def act(fn):
        return lambda: (fn(), canvas.focus_set())
    for name, fn, tip in (("zoom_in", canvas.zoom_in, f"{t('view.zoom_in')}  (Ctrl++)"),
                          ("zoom_out", canvas.zoom_out, f"{t('view.zoom_out')}  (Ctrl+−)"),
                          ("fit", canvas.fit, f"{t('view.zoom_fit')}  (Ctrl+0)")):
        IconButton(parent, name, act(fn), tip, **kw).pack(side=tk.LEFT, padx=1)
    pct = tk.Label(parent, text="", bg=colors["bg_card"], fg=colors["text_muted"], font=FONTS["small"],
                   width=6, cursor="hand2")
    pct.pack(side=tk.LEFT, padx=(4, 8))
    pct.bind("<Button-1>", lambda e: act(canvas.actual_size)())
    Tooltip(pct, f"{t('view.zoom_100')}  (Ctrl+1)")
    canvas.on_zoom = lambda p: pct.config(text=f"{p}%")
    for seq, fn in (("plus", canvas.zoom_in), ("equal", canvas.zoom_in), ("KP_Add", canvas.zoom_in),
                    ("minus", canvas.zoom_out), ("KP_Subtract", canvas.zoom_out),
                    ("Key-0", canvas.fit), ("KP_0", canvas.fit), ("Key-1", canvas.actual_size)):
        top.bind(f"<Control-{seq}>", lambda e, f=fn: (f(), "break")[1])
    return pct


class IconButton(tk.Label):
    """Botón de ícono para la barra superior; `selected` lo marca como herramienta activa."""

    def __init__(self, parent, name: str, command, tip: str, bg: str, hover: str, active: str,
                 size: int = 18, color: str = COLORS["on_accent"]):
        super().__init__(parent, bg=bg, cursor="hand2", padx=4, pady=3)
        self._colors = (bg, hover, active)
        self._photo = ImageTk.PhotoImage(icon(name, size, color))
        self.config(image=self._photo)
        self.selected = False
        self.bind("<Button-1>", lambda e: command())
        self.bind("<Enter>", lambda e: self.config(bg=self._colors[2] if self.selected else self._colors[1]))
        self.bind("<Leave>", lambda e: self._paint())
        Tooltip(self, tip)

    def select(self, on: bool):
        self.selected = on
        self._paint()

    def _paint(self):
        self.config(bg=self._colors[2] if self.selected else self._colors[0])
