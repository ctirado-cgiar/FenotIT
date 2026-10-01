"""Íconos dibujados (nítidos en cualquier DPI, sin archivos ni fuentes de íconos)
y botones de herramienta con ayuda emergente."""
from __future__ import annotations

import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

_SS = 4          # sobre-muestreo para bordes suaves


def _magnifier(d: ImageDraw.ImageDraw, s: int, w: int, color, sign: str | None):
    r = s * 0.30
    cx = cy = s * 0.40
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=w)
    a = r * 0.7071
    d.line((cx + a, cy + a, s * 0.90, s * 0.90), fill=color, width=int(w * 1.6))
    k = r * 0.5
    if sign:
        d.line((cx - k, cy, cx + k, cy), fill=color, width=w)
    if sign == "+":
        d.line((cx, cy - k, cx, cy + k), fill=color, width=w)


def _hand(d: ImageDraw.ImageDraw, s: int, w: int, color):
    """Mano abierta rellena (se lee mejor que un contorno en tamaño chico)."""
    fw, gap = s * 0.13, s * 0.03
    for i, top in enumerate((0.20, 0.08, 0.11, 0.22)):
        x = s * 0.27 + i * (fw + gap)
        d.rounded_rectangle((x, s * top, x + fw, s * 0.60), radius=fw / 2, fill=color)
    d.rounded_rectangle((s * 0.27, s * 0.42, s * 0.27 + 4 * fw + 3 * gap, s * 0.94),
                        radius=s * 0.17, fill=color)
    d.line((s * 0.30, s * 0.74, s * 0.10, s * 0.44), fill=color, width=int(fw))
    d.ellipse((s * 0.10 - fw / 2, s * 0.44 - fw / 2, s * 0.10 + fw / 2, s * 0.44 + fw / 2), fill=color)


def _fit(d: ImageDraw.ImageDraw, s: int, w: int, color):
    m, k = s * 0.12, s * 0.30
    for x, y, dx, dy in ((m, m, 1, 1), (s - m, m, -1, 1), (m, s - m, 1, -1), (s - m, s - m, -1, -1)):
        d.line((x, y, x + dx * k, y), fill=color, width=w)
        d.line((x, y, x, y + dy * k), fill=color, width=w)


def _zoom_area(d: ImageDraw.ImageDraw, s: int, w: int, color):
    x0, y0, x1, y1 = s * 0.08, s * 0.08, s * 0.70, s * 0.62
    dash = s * 0.10
    for a, b, horiz, fixed in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        p = a
        while p < b:
            q = min(p + dash, b)
            d.line((p, fixed, q, fixed) if horiz else (fixed, p, fixed, q), fill=color, width=w)
            p += dash * 1.8
    r = s * 0.17
    cx, cy = s * 0.62, s * 0.60
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=w, fill=None)
    d.line((cx + r * 0.7, cy + r * 0.7, s * 0.94, s * 0.94), fill=color, width=int(w * 1.6))
    k = r * 0.55
    d.line((cx - k, cy, cx + k, cy), fill=color, width=max(1, int(w * 0.8)))
    d.line((cx, cy - k, cx, cy + k), fill=color, width=max(1, int(w * 0.8)))


_DRAW = {"zoom_in": lambda d, s, w, c: _magnifier(d, s, w, c, "+"),
         "zoom_out": lambda d, s, w, c: _magnifier(d, s, w, c, "-"),
         "pan": _hand, "fit": _fit, "zoom_area": _zoom_area}


def icon(name: str, size: int, color: str = "#FFFFFF") -> Image.Image:
    s = size * _SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    _DRAW[name](ImageDraw.Draw(img), s, max(_SS, int(s * 0.085)), color)
    return img.resize((size, size), Image.LANCZOS)


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
        tk.Label(tw, text=self.text, bg="#FFFFE6", fg="#222222", relief="solid", bd=1,
                 font=("Segoe UI", 8), padx=6, pady=2).pack()
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


class IconButton(tk.Label):
    """Botón de ícono para la barra superior; `selected` lo marca como herramienta activa."""

    def __init__(self, parent, name: str, command, tip: str, bg: str, hover: str, active: str,
                 size: int = 18):
        super().__init__(parent, bg=bg, cursor="hand2", padx=4, pady=3)
        self._colors = (bg, hover, active)
        self._photo = ImageTk.PhotoImage(icon(name, size))
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
