"""Deslizador de rango Mín–Máx (0-255) y, debajo, el histograma del canal elegido para
segmentar. Los dos usan el mismo margen, así cada manija queda sobre su línea. El canal
elegido se dibuja relleno en su color (dentro del rango, intenso; fuera, claro) y los otros
dos canales del espacio de fondo, como silueta tenue."""
from __future__ import annotations

import tkinter as tk

import numpy as np

from fenotit.gui.theme import COLORS, FONTS

HEIGHT = 64
PAD = 9                                         # = radio de la manija + 2: alinea deslizador e histograma


class ChannelHistogram(tk.Canvas):

    def __init__(self, parent, on_drag=None, bg: str | None = None):
        """on_drag(lo, hi): el usuario movió una línea."""
        bg = bg or parent.cget("bg")
        super().__init__(parent, height=HEIGHT + 4, bg=bg, highlightthickness=0, cursor="sb_h_double_arrow")
        self.on_drag = on_drag
        self.hist: np.ndarray | None = None
        self.others: list[tuple[np.ndarray, str]] = []      # los otros dos canales, de fondo
        self.color = COLORS["accent"]
        self.lo, self.hi = 0, 255
        self._grab = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", lambda e: setattr(self, "_grab", None))

    # ── datos ─────────────────────────────────────────────────────────────────

    def set_data(self, values: np.ndarray | None, color: str | None = None,
                 others: list[tuple[np.ndarray, str]] | None = None):
        """values: píxeles del canal elegido (uint8) o None (sin foto); color: el del canal;
        others: [(píxeles, color)] de los otros canales del espacio, que se dibujan de fondo
        para comparar de un vistazo cuál separa mejor objeto y fondo."""
        def hist(v):
            return np.log1p(np.bincount(v.ravel(), minlength=256)[:256].astype(float))   # escala log
        if values is None or values.size == 0:
            self.hist, self.others = None, []
        else:
            self.hist = hist(values)
            self.others = [(hist(v), c) for v, c in (others or []) if v is not None and v.size]
        self.color = color or COLORS["accent"]
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
            top = max([self.hist.max()] + [h.max() for h, _ in self.others])
            step = (w - 2 * PAD) / 256
            for h, c in self.others:                      # otros canales: solo la silueta, tenue
                hs = h / top * (HEIGHT - 4)
                pts = []
                for v in range(256):
                    pts += [PAD + (v + 0.5) * step, base - hs[v]]
                self.create_line(*pts, fill=_tint(c, 0.55), width=1)
            hs = self.hist / top * (HEIGHT - 4)
            light = _tint(self.color, 0.72)
            for v in range(256):
                if hs[v] <= 0:
                    continue
                x0 = PAD + v * step
                self.create_rectangle(x0, base - hs[v], x0 + max(step, 1), base, width=0,
                                      fill=self.color if self.lo <= v <= self.hi else light)
        else:
            self.create_text(w / 2, HEIGHT / 2, text="—", fill=COLORS["text_muted"], font=FONTS["small"])
        for v in (self.lo, self.hi):                  # las dos líneas del rango (el valor va en el deslizador)
            x = self._x(v)
            self.create_line(x, 0, x, base + 3, fill="#222222", width=1.5)

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


class RangeSlider(tk.Canvas):
    """Una barra 0-255 con dos manijas redondas (Mín y Máx), el tramo entre ellas en azul y
    el valor de cada una en una burbuja encima. Mín nunca pasa a Máx. Arrastrar mueve la
    manija más cercana; ← → (o la rueda) la mueven de a uno después de tocarla."""

    R = 7                    # radio de la manija
    BUBBLE = 16              # alto de la burbuja

    def __init__(self, parent, on_change=None, bg: str | None = None):
        """on_change(lo, hi): el usuario movió una manija (no se llama al poner valores por código)."""
        bg = bg or parent.cget("bg")
        super().__init__(parent, height=self.BUBBLE + 2 * self.R + 8, bg=bg, highlightthickness=0,
                         takefocus=1, cursor="hand2")
        self.on_change = on_change
        self.lo, self.hi = 0, 255
        self.track = COLORS["accent"]                     # color del tramo elegido (el del canal)
        self.active = "lo"
        self._grab = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", lambda e: setattr(self, "_grab", None))
        self.bind("<FocusIn>", lambda e: self.redraw())
        self.bind("<FocusOut>", lambda e: self.redraw())
        for k, d in (("<Left>", -1), ("<Right>", 1), ("<Down>", -1), ("<Up>", 1)):
            self.bind(k, lambda e, d=d: self._step(d))
        self.bind("<MouseWheel>", lambda e: self._wheel(e, 1 if e.delta > 0 else -1))
        self.bind("<Button-4>", lambda e: self._wheel(e, 1))
        self.bind("<Button-5>", lambda e: self._wheel(e, -1))

    # ── valores ───────────────────────────────────────────────────────────────

    def set_values(self, lo: int, hi: int):
        lo, hi = int(np.clip(lo, 0, 255)), int(np.clip(hi, 0, 255))
        if lo > hi:
            lo = hi
        if (lo, hi) != (self.lo, self.hi):
            self.lo, self.hi = lo, hi
            self.redraw()

    def set_track(self, color: str):
        if color != self.track:
            self.track = color
            self.redraw()

    def end(self, which: str) -> "_End":
        """Objeto con get()/set() para un extremo (compatible con tk.Scale)."""
        return _End(self, which)

    def _put(self, which: str, v: int):
        """Cambio del usuario: frena la manija en la otra y avisa."""
        v = int(np.clip(v, 0, 255))
        lo, hi = (min(v, self.hi), self.hi) if which == "lo" else (self.lo, max(v, self.lo))
        if (lo, hi) == (self.lo, self.hi):
            return
        self.lo, self.hi = lo, hi
        self.redraw()
        if self.on_change:
            self.on_change(lo, hi)

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
        r, acc = self.R, COLORS["accent"]
        y = self.BUBBLE + 4 + r
        self.create_line(PAD, y, w - PAD, y, fill=COLORS["slider_trough"], width=4, capstyle="round")
        xl, xh = self._x(self.lo), self._x(self.hi)
        self.create_line(xl, y, xh, y, fill=self.track, width=4, capstyle="round")
        xs = self._bubble_x(xl, xh, w)
        focus = self.focus_get() is self
        for name, x, bx in (("lo", xl, xs[0]), ("hi", xh, xs[1])):
            ring = acc if not (focus and name == self.active) else "#0B3C6E"
            self.create_oval(x - r, y - r, x + r, y + r, fill="white", outline=ring, width=2)
            text = str(self.lo if name == "lo" else self.hi)
            bw = 26
            self.create_rectangle(bx - bw / 2, 1, bx + bw / 2, self.BUBBLE, fill=acc, outline=acc)
            self.create_polygon(x - 3, self.BUBBLE, x + 3, self.BUBBLE, x, self.BUBBLE + 3, fill=acc, outline=acc)
            self.create_text(bx, 1 + self.BUBBLE / 2, text=text, fill="white", font=("Segoe UI", 8, "bold"))

    @staticmethod
    def _bubble_x(xl: float, xh: float, w: float) -> tuple[float, float]:
        """Centros de las burbujas: dentro del canvas y sin encimarse cuando las manijas se juntan."""
        half, gap = 13, 2
        bl, bh = xl, xh
        if bh - bl < 2 * half + gap:
            mid = (bl + bh) / 2
            bl, bh = mid - half - gap / 2, mid + half + gap / 2
        shift = max(0, half + 1 - bl) - max(0, bh - (w - half - 1))
        return bl + shift, bh + shift

    # ── interacción ───────────────────────────────────────────────────────────

    def _nearest(self, x: float) -> str:
        xl, xh = self._x(self.lo), self._x(self.hi)
        if abs(xl - xh) < 1:                      # juntas: la que deja moverse hacia el clic
            return "lo" if x < xl else "hi"
        return "lo" if abs(x - xl) <= abs(x - xh) else "hi"

    def _press(self, e):
        self.focus_set()
        self._grab = self.active = self._nearest(e.x)
        self._move(e)

    def _move(self, e):
        if self._grab:
            self._put(self._grab, self._v(e.x))

    def _step(self, d: int):
        self._put(self.active, (self.lo if self.active == "lo" else self.hi) + d)
        return "break"                            # ← → no cambian de foto aquí

    def _wheel(self, e, d: int):
        self.active = self._nearest(e.x)
        return self._step(d)


class _End:
    def __init__(self, slider: RangeSlider, which: str):
        self.slider, self.which = slider, which

    def get(self) -> int:
        return self.slider.lo if self.which == "lo" else self.slider.hi

    def set(self, v):
        s = self.slider
        if self.which == "lo":
            s.set_values(v, max(int(v), s.hi))
        else:
            s.set_values(min(int(v), s.lo), v)


def _tint(color: str, k: float) -> str:
    """Mezcla el color con blanco (k = 0 color puro, 1 blanco)."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(c + (255 - c) * k) for c in (r, g, b))
