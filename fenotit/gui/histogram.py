"""Histograma del canal elegido para segmentar, con el rango Mín–Máx marcado. Las dos
líneas se mueven con los deslizadores o arrastrándolas; lo que queda dentro (objeto) se
pinta en azul y lo de fuera en gris."""
from __future__ import annotations

import tkinter as tk

import numpy as np

from fenotit.gui.theme import COLORS, FONTS

HEIGHT = 64
PAD = 6


class ChannelHistogram(tk.Canvas):

    def __init__(self, parent, on_drag=None, bg: str | None = None):
        """on_drag(lo, hi): el usuario movió una línea."""
        bg = bg or parent.cget("bg")
        super().__init__(parent, height=HEIGHT + 14, bg=bg, highlightthickness=0, cursor="sb_h_double_arrow")
        self.on_drag = on_drag
        self.hist: np.ndarray | None = None
        self.lo, self.hi = 0, 255
        self._grab = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", lambda e: setattr(self, "_grab", None))

    # ── datos ─────────────────────────────────────────────────────────────────

    def set_data(self, values: np.ndarray | None):
        """values: píxeles del canal (uint8) o None (sin foto)."""
        if values is None or values.size == 0:
            self.hist = None
        else:
            h = np.bincount(values.ravel(), minlength=256)[:256].astype(float)
            self.hist = np.log1p(h)                 # el fondo domina: escala logarítmica
        self.redraw()

    def set_range(self, lo: int, hi: int):
        if (int(lo), int(hi)) != (self.lo, self.hi):
            self.lo, self.hi = int(lo), int(hi)
            self.redraw()

    # ── dibujo ────────────────────────────────────────────────────────────────

    def _x(self, v: float) -> float:
        w = max(self.winfo_width(), 60)
        return PAD + v / 255 * (w - 2 * PAD)

    def _v(self, x: float) -> int:
        w = max(self.winfo_width(), 60)
        return int(round(np.clip((x - PAD) / (w - 2 * PAD) * 255, 0, 255)))

    def redraw(self):
        self.delete("all")
        w = max(self.winfo_width(), 60)
        base = HEIGHT
        self.create_line(PAD, base, w - PAD, base, fill=COLORS["border"])
        if self.hist is not None and self.hist.max() > 0:
            hs = self.hist / self.hist.max() * (HEIGHT - 4)
            step = (w - 2 * PAD) / 256
            for v in range(256):
                if hs[v] <= 0:
                    continue
                inside = self.lo <= v <= self.hi
                x0 = PAD + v * step
                self.create_rectangle(x0, base - hs[v], x0 + max(step, 1), base, width=0,
                                      fill=COLORS["accent"] if inside else "#B9C2CC")
        else:
            self.create_text(w / 2, HEIGHT / 2, text="—", fill=COLORS["text_muted"], font=FONTS["small"])
        for v in (self.lo, self.hi):                  # las dos líneas del rango
            x = self._x(v)
            self.create_line(x, 2, x, base + 3, fill="#D73027", width=2)
            anchor = "w" if x < 20 else "e" if x > w - 20 else "center"     # sin cortarse en los bordes
            self.create_text(x, base + 9, text=str(v), anchor=anchor, fill=COLORS["text_muted"],
                             font=("Segoe UI", 7))

    # ── arrastrar las líneas ──────────────────────────────────────────────────

    def _press(self, e):
        lo_x, hi_x = self._x(self.lo), self._x(self.hi)
        self._grab = "lo" if abs(e.x - lo_x) <= abs(e.x - hi_x) else "hi"
        self._move(e)

    def _move(self, e):
        if self._grab is None:
            return
        v = self._v(e.x)
        if self._grab == "lo":
            self.lo = min(v, self.hi)               # Mín nunca pasa a Máx
        else:
            self.hi = max(v, self.lo)
        self.redraw()
        if self.on_drag:
            self.on_drag(self.lo, self.hi)
