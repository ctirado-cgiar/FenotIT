"""Deslizador fino de un valor: pista delgada, tramo lleno hasta el valor y manija redonda."""
from __future__ import annotations

import tkinter as tk

from fenotit.gui.style.theme import COLORS


class Slider(tk.Canvas):
    """Valor entre `from_` y `to` en `variable` (IntVar o DoubleVar). Arrastrar o hacer clic
    mueve la manija; ← → la mueven un paso. `command(valor)` se llama cuando el usuario lo
    cambia."""

    R = 6                     # radio de la manija
    TRACK = 3                 # grosor de la pista

    def __init__(self, parent, variable, from_: float, to: float, command=None, length: int | None = None,
                 bg: str | None = None):
        bg = bg or parent.cget("bg")
        super().__init__(parent, height=2 * self.R + 6, bg=bg, highlightthickness=0, takefocus=1, cursor="hand2",
                         **({"width": length} if length else {}))
        self.var, self.lo, self.hi, self.command = variable, float(from_), float(to), command
        self._integer = isinstance(variable, tk.IntVar)
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<ButtonPress-1>", self._move)
        self.bind("<B1-Motion>", self._move)
        self.bind("<Left>", lambda e: self._step(-1))
        self.bind("<Right>", lambda e: self._step(1))
        self.var.trace_add("write", lambda *_: self.redraw())

    def _x(self, v: float) -> float:
        w = max(self.winfo_width(), 40)
        k = (v - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0
        return self.R + 1 + max(0.0, min(1.0, k)) * (w - 2 * self.R - 2)

    def _value(self, x: float) -> float:
        w = max(self.winfo_width(), 40)
        k = (x - self.R - 1) / max(w - 2 * self.R - 2, 1)
        v = self.lo + max(0.0, min(1.0, k)) * (self.hi - self.lo)
        return round(v) if self._integer else v

    def redraw(self):
        self.delete("all")
        w, y = max(self.winfo_width(), 40), self.R + 3
        try:
            x = self._x(float(self.var.get()))
        except (tk.TclError, ValueError):
            x = self._x(self.lo)
        self.create_line(self.R + 1, y, w - self.R - 1, y, fill=COLORS["slider_trough"], width=self.TRACK,
                         capstyle="round")
        self.create_line(self.R + 1, y, x, y, fill=COLORS["accent"], width=self.TRACK, capstyle="round")
        ring = COLORS["accent_darker"] if self.focus_get() is self else COLORS["accent"]
        self.create_oval(x - self.R, y - self.R, x + self.R, y + self.R, fill=COLORS["white"], outline=ring,
                         width=1.5)

    def _set(self, v: float):
        if v != self.var.get():
            self.var.set(v)
            if self.command:
                self.command(v)

    def _move(self, e):
        self.focus_set()
        self._set(self._value(e.x))

    def _step(self, d: int):
        step = 1 if self._integer else (self.hi - self.lo) / 100
        self._set(max(self.lo, min(self.hi, self.var.get() + d * step)))
