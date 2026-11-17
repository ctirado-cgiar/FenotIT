"""Deslizador de rango Mín–Máx (0-255) y, debajo, el histograma del canal elegido para
segmentar. Los dos usan el mismo margen, así cada manija queda sobre su línea. El canal
elegido se dibuja relleno en su color (dentro del rango, intenso; fuera, claro) y los otros
dos canales del espacio de fondo, como silueta tenue."""
from __future__ import annotations

import tkinter as tk

import numpy as np

from fenotit.gui.style.theme import COLORS, FONTS

HEIGHT = 64
PAD = 8                                         # = radio de la manija + 2: alinea deslizador e histograma


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
        self.bind("<Configure>", lambda event: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", lambda event: setattr(self, "_grab", None))

    # ── datos ─────────────────────────────────────────────────────────────────

    def set_data(self, values: np.ndarray | None, color: str | None = None,
                 others: list[tuple[np.ndarray, str]] | None = None):
        """values: píxeles del canal elegido (uint8) o None (sin foto); color: el del canal;
        others: [(píxeles, color)] de los otros canales del espacio, que se dibujan de fondo
        para comparar de un vistazo cuál separa mejor objeto y fondo."""
        def hist(pixels):
            return np.log1p(np.bincount(pixels.ravel(), minlength=256)[:256].astype(float))   # escala log
        if values is None or values.size == 0:
            self.hist, self.others = None, []
        else:
            self.hist = hist(values)
            self.others = [(hist(channel), channel_color) for channel, channel_color in (others or [])
                           if channel is not None and channel.size]
        self.color = color or COLORS["accent"]
        self.redraw()

    def set_range(self, low: int, high: int):
        if (int(low), int(high)) != (self.lo, self.hi):
            self.lo, self.hi = int(low), int(high)
            self.redraw()

    # ── dibujo ────────────────────────────────────────────────────────────────

    def _x(self, value: float) -> float:
        width = max(self.winfo_width(), 60)
        return PAD + value / 255 * (width - 2 * PAD)

    def _v(self, x: float) -> int:
        width = max(self.winfo_width(), 60)
        return int(round(np.clip((x - PAD) / (width - 2 * PAD) * 255, 0, 255)))

    def redraw(self):
        self.delete("all")
        width = max(self.winfo_width(), 60)
        base = HEIGHT
        self.create_line(PAD, base, width - PAD, base, fill=COLORS["divider"])
        if self.hist is not None and self.hist.max() > 0:
            top = max([self.hist.max()] + [histogram.max() for histogram, _ in self.others])
            step = (width - 2 * PAD) / 256
            for histogram, color in self.others:                      # otros canales: solo la silueta, tenue
                heights = histogram / top * (HEIGHT - 4)
                pts = []
                for value in range(256):
                    pts += [PAD + (value + 0.5) * step, base - heights[value]]
                self.create_line(*pts, fill=_tint(color, 0.55), width=1)
            heights = self.hist / top * (HEIGHT - 4)
            inside, outside = _tint(self.color, 0.45), _tint(self.color, 0.85)
            pts = []
            for value in range(256):
                pts += [PAD + (value + 0.5) * step, base - heights[value]]
                if heights[value] <= 0:
                    continue
                x0 = PAD + value * step
                self.create_rectangle(x0, base - heights[value], x0 + max(step, 1), base, width=0,
                                      fill=inside if self.lo <= value <= self.hi else outside)
            self.create_line(*pts, fill=self.color, width=1)
        else:
            self.create_text(width / 2, HEIGHT / 2, text="—", fill=COLORS["text_muted"], font=FONTS["small"])
        for value in (self.lo, self.hi):                  # las dos líneas del rango (el valor va en el deslizador)
            x = self._x(value)
            self.create_line(x, 0, x, base + 2, fill=COLORS["accent_darker"], width=1)

    # ── arrastrar las líneas ──────────────────────────────────────────────────

    def _press(self, event):
        lo_x, hi_x = self._x(self.lo), self._x(self.hi)
        self._grab = "lo" if abs(event.x - lo_x) <= abs(event.x - hi_x) else "hi"
        self._move(event)

    def _move(self, event):
        if self._grab is None:
            return
        value = self._v(event.x)
        if self._grab == "lo":
            self.lo = min(value, self.hi)               # Mín nunca pasa a Máx
        else:
            self.hi = max(value, self.lo)
        self.redraw()
        if self.on_drag:
            self.on_drag(self.lo, self.hi)


class RangeSlider(tk.Canvas):
    """Una barra 0-255 con dos manijas redondas (Mín y Máx), el tramo entre ellas en azul y
    el valor de cada una en una burbuja encima. Mín nunca pasa a Máx. Arrastrar mueve la
    manija más cercana; ← → (o la rueda) la mueven de a uno después de tocarla."""

    RADIUS = 6                    # radio de la manija
    BUBBLE = 14              # alto de la burbuja

    def __init__(self, parent, on_change=None, bg: str | None = None):
        """on_change(lo, hi): el usuario movió una manija (no se llama al poner valores por código)."""
        bg = bg or parent.cget("bg")
        super().__init__(parent, height=self.BUBBLE + 2 * self.RADIUS + 8, bg=bg, highlightthickness=0,
                         takefocus=1, cursor="hand2")
        self.on_change = on_change
        self.lo, self.hi = 0, 255
        self.track = COLORS["accent"]                     # color del tramo elegido (el del canal)
        self.active = "lo"
        self._grab = None
        self.bind("<Configure>", lambda event: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", lambda event: setattr(self, "_grab", None))
        self.bind("<FocusIn>", lambda event: self.redraw())
        self.bind("<FocusOut>", lambda event: self.redraw())
        for sequence, delta in (("<Left>", -1), ("<Right>", 1), ("<Down>", -1), ("<Up>", 1)):
            self.bind(sequence, lambda event, delta=delta: self._step(delta))
        self.bind("<MouseWheel>", lambda event: self._wheel(event, 1 if event.delta > 0 else -1))
        self.bind("<Button-4>", lambda event: self._wheel(event, 1))
        self.bind("<Button-5>", lambda event: self._wheel(event, -1))

    # ── valores ───────────────────────────────────────────────────────────────

    def set_values(self, low: int, high: int):
        low, high = int(np.clip(low, 0, 255)), int(np.clip(high, 0, 255))
        if low > high:
            low = high
        if (low, high) != (self.lo, self.hi):
            self.lo, self.hi = low, high
            self.redraw()

    def set_track(self, color: str):
        if color != self.track:
            self.track = color
            self.redraw()

    def end(self, which: str) -> "_End":
        """Objeto con get()/set() para un extremo (compatible con tk.Scale)."""
        return _End(self, which)

    def _put(self, which: str, value: int):
        """Cambio del usuario: frena la manija en la otra y avisa."""
        value = int(np.clip(value, 0, 255))
        low, high = (min(value, self.hi), self.hi) if which == "lo" else (self.lo, max(value, self.lo))
        if (low, high) == (self.lo, self.hi):
            return
        self.lo, self.hi = low, high
        self.redraw()
        if self.on_change:
            self.on_change(low, high)

    # ── dibujo ────────────────────────────────────────────────────────────────

    def _x(self, value: float) -> float:
        width = max(self.winfo_width(), 60)
        return PAD + value / 255 * (width - 2 * PAD)

    def _v(self, x: float) -> int:
        width = max(self.winfo_width(), 60)
        return int(round(np.clip((x - PAD) / (width - 2 * PAD) * 255, 0, 255)))

    def redraw(self):
        self.delete("all")
        width = max(self.winfo_width(), 60)
        radius, acc = self.RADIUS, COLORS["accent"]
        y = self.BUBBLE + 4 + radius
        self.create_line(PAD, y, width - PAD, y, fill=COLORS["slider_trough"], width=3, capstyle="round")
        x_low, x_high = self._x(self.lo), self._x(self.hi)
        self.create_line(x_low, y, x_high, y, fill=self.track, width=3, capstyle="round")
        probe = self.create_text(0, 0, text="255", font=FONTS["tiny_bold"])
        x0, _, x1, _ = self.bbox(probe)
        self.delete(probe)
        bubble_width = x1 - x0 + 8
        xs_values = self._bubble_x(x_low, x_high, width, bubble_width / 2)
        focus = self.focus_get() is self
        for name, x, box_x in (("lo", x_low, xs_values[0]), ("hi", x_high, xs_values[1])):
            ring = acc if not (focus and name == self.active) else COLORS["accent_darker"]
            self.create_oval(x - radius, y - radius, x + radius, y + radius, fill=COLORS["white"], outline=ring, width=1.5)
            text = str(self.lo if name == "lo" else self.hi)
            self.create_rectangle(box_x - bubble_width / 2, 1, box_x + bubble_width / 2, self.BUBBLE, fill=acc, outline=acc)
            self.create_polygon(x - 3, self.BUBBLE, x + 3, self.BUBBLE, x, self.BUBBLE + 3, fill=acc, outline=acc)
            self.create_text(box_x, 1 + self.BUBBLE / 2, text=text, fill=COLORS["white"], font=FONTS["tiny_bold"])

    @staticmethod
    def _bubble_x(x_low: float, x_high: float, width: float, half: float) -> tuple[float, float]:
        """Centros de las burbujas: dentro del canvas y sin encimarse cuando las manijas se juntan."""
        gap, lo_x, hi_x = 2, half + 1, width - half - 1
        bubble_low, bubble_high = min(max(x_low, lo_x), hi_x), min(max(x_high, lo_x), hi_x)
        if bubble_high - bubble_low < 2 * half + gap:
            mid = min(max((bubble_low + bubble_high) / 2, lo_x + half + gap / 2), hi_x - half - gap / 2)
            bubble_low, bubble_high = mid - half - gap / 2, mid + half + gap / 2
        return bubble_low, bubble_high

    # ── interacción ───────────────────────────────────────────────────────────

    def _nearest(self, x: float) -> str:
        x_low, x_high = self._x(self.lo), self._x(self.hi)
        if abs(x_low - x_high) < 1:                      # juntas: la que deja moverse hacia el clic
            return "lo" if x < x_low else "hi"
        return "lo" if abs(x - x_low) <= abs(x - x_high) else "hi"

    def _press(self, event):
        self.focus_set()
        self._grab = self.active = self._nearest(event.x)
        self._move(event)

    def _move(self, event):
        if self._grab:
            self._put(self._grab, self._v(event.x))

    def _step(self, delta: int):
        self._put(self.active, (self.lo if self.active == "lo" else self.hi) + delta)
        return "break"                            # ← → no cambian de foto aquí

    def _wheel(self, event, delta: int):
        self.active = self._nearest(event.x)
        return self._step(delta)


class _End:
    def __init__(self, slider: RangeSlider, which: str):
        self.slider, self.which = slider, which

    def get(self) -> int:
        return self.slider.lo if self.which == "lo" else self.slider.hi

    def set(self, value):
        slider = self.slider
        if self.which == "lo":
            slider.set_values(value, max(int(value), slider.hi))
        else:
            slider.set_values(min(int(value), slider.lo), value)


def _tint(color: str, amount: float) -> str:
    """Mezcla el color con blanco (k = 0 color puro, 1 blanco)."""
    red, green, blue = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(channel + (255 - channel) * amount) for channel in (red, green, blue))
