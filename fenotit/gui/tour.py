"""Recorrido guiado (como driver.js): oscurece la ventana salvo el elemento del paso, lo
enmarca en azul y muestra al lado un globo con título, texto y Anterior / Siguiente /
Cerrar. Durante el recorrido solo responden Anterior / Siguiente / Terminar (el elemento se ve, no se toca).

El oscurecido son cuatro ventanas semitransparentes alrededor del hueco (Tk no recorta
ventanas), así que funciona igual en Windows y en Linux con compositor."""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import Callable

from fenotit.gui.theme import COLORS, FONTS
from fenotit.i18n import t

PAD = 6              # margen del hueco alrededor del elemento
RING = 2             # grosor del marco azul
DIM = 0.55           # opacidad del oscurecido
WIDTH = 320          # ancho del globo


@dataclass
class Step:
    title: str
    text: str
    target: Callable[[], tk.Widget | None] | None = None     # None = globo centrado, sin hueco
    when: Callable[[], bool] | None = None                   # si devuelve False, el paso se salta
    before: Callable[[], None] | None = None                  # preparar la pantalla (abrir un panel…)
    key: str = ""                                              # para empezar en un paso concreto


class Tour:

    def __init__(self, root: tk.Tk, steps: list[Step], on_close: Callable[[], None] | None = None):
        self.root, self.steps, self.on_close = root, steps, on_close
        self.i = -1
        self._shades: list[tk.Toplevel] = []
        self._ring: list[tk.Toplevel] = []
        self._pop: tk.Toplevel | None = None
        self._binds: list[tuple[str, str]] = []

    # ── navegación ────────────────────────────────────────────────────────────

    def start(self, at: str | None = None):
        """at = clave del paso donde empezar (el que corresponde a lo que el usuario ya hizo);
        Anterior sigue llevando a los pasos de antes."""
        self._build()
        keys = [s.key for s in self.steps]
        self.i = keys.index(at) - 1 if at in keys else -1
        self._go(+1)

    def _go(self, d: int):
        i = self.i + d
        while 0 <= i < len(self.steps):
            s = self.steps[i]
            if s.when is None or _safe(s.when, False):
                break
            i += d
        if not 0 <= i < len(self.steps):
            if d > 0:
                self.close()
            return
        self.i = i
        if self.steps[i].before:
            _safe(self.steps[i].before)
            self.root.update_idletasks()
        self._show()

    def close(self):
        for seq, fid in self._binds:
            try:
                self.root.unbind(seq, fid)
            except tk.TclError:
                pass
        self._binds = []
        for w in self._windows() if self._pop else []:
            try:
                w.destroy()
            except tk.TclError:
                pass
        self._shades, self._ring, self._pop = [], [], None
        if self.on_close:
            self.on_close()

    # ── ventanas ──────────────────────────────────────────────────────────────

    def _overlay(self, color: str, alpha: float) -> tk.Toplevel:
        w = tk.Toplevel(self.root)
        w.overrideredirect(True)
        w.configure(bg=color)
        w.transient(self.root)
        try:
            w.attributes("-alpha", alpha)
        except tk.TclError:
            pass
        w.bind("<Button-1>", lambda e: "break")              # fuera del hueco no se hace clic
        return w

    def _build(self):
        self._shades = [self._overlay("#000000", DIM) for _ in range(4)]
        self._ring = [self._overlay(COLORS["accent"], 1.0) for _ in range(4)]
        # durante el recorrido solo se usan Anterior / Siguiente / Terminar: el elemento
        # resaltado se ve pero no recibe clics (un menú abierto quedaría debajo del oscurecido)
        self._glass = self._overlay("#FFFFFF", 0.01)
        pop = tk.Toplevel(self.root)
        pop.overrideredirect(True)
        pop.transient(self.root)
        pop.configure(bg=COLORS["accent"])
        body = tk.Frame(pop, bg="#FFFFFF", padx=14, pady=12)
        body.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        top = tk.Frame(body, bg="#FFFFFF")
        top.pack(fill=tk.X)
        self._title = tk.Label(top, bg="#FFFFFF", fg=COLORS["accent"], font=FONTS["title"],
                               anchor="w", justify="left", wraplength=WIDTH - 60)
        self._title.pack(side=tk.LEFT, fill=tk.X, expand=True)
        x = tk.Label(top, text="✕", bg="#FFFFFF", fg=COLORS["text_muted"], cursor="hand2", font=FONTS["small"])
        x.pack(side=tk.RIGHT, anchor="n")
        x.bind("<Button-1>", lambda e: self.close())
        self._text = tk.Label(body, bg="#FFFFFF", fg=COLORS["text"], font=FONTS["body"], justify="left",
                              anchor="w", wraplength=WIDTH - 28)
        self._text.pack(fill=tk.X, pady=(6, 10))
        bar = tk.Frame(body, bg="#FFFFFF")
        bar.pack(fill=tk.X)
        self._count = tk.Label(bar, bg="#FFFFFF", fg=COLORS["text_muted"], font=FONTS["small"])
        self._count.pack(side=tk.LEFT)
        btn = dict(relief="flat", bd=0, cursor="hand2", font=FONTS["small"], padx=10, pady=3)
        self._next = tk.Button(bar, command=lambda: self._go(+1), bg=COLORS["accent"], fg="#FFFFFF",
                               activebackground=COLORS["accent"], activeforeground="#FFFFFF", **btn)
        self._next.pack(side=tk.RIGHT)
        self._prev = tk.Button(bar, text=t("tour.prev"), command=lambda: self._go(-1), bg=COLORS["btn_bg"],
                               fg=COLORS["accent"], activebackground=COLORS["btn_bg"], **btn)
        self._prev.pack(side=tk.RIGHT, padx=(0, 6))
        for k, f in (("<Escape>", lambda e: self.close()), ("<Right>", lambda e: self._go(+1)),
                     ("<Left>", lambda e: self._go(-1)), ("<Return>", lambda e: self._go(+1))):
            pop.bind(k, f)
        self._pop = pop
        # la ventana principal no debe tapar el recorrido: al hacer clic en ella (en el hueco o
        # fuera de la app y volver) se suben de nuevo; al minimizarla se esconden con ella
        for seq, fn in (("<Configure>", lambda e: e.widget is self.root and self._place()),
                        ("<FocusIn>", lambda e: self._raise()),
                        ("<ButtonRelease-1>", lambda e: self._raise()),
                        ("<Unmap>", lambda e: e.widget is self.root and self._hide(True)),
                        ("<Map>", lambda e: e.widget is self.root and self._hide(False))):
            self._binds.append((seq, self.root.bind(seq, fn, add="+")))

    def _windows(self):
        return self._shades + self._ring + [self._glass] + ([self._pop] if self._pop else [])

    def _raise(self):
        if self._pop is None or getattr(self, "_raising", False):
            return
        self._raising = True

        def go():
            self._raising = False
            for w in self._windows():
                if w.winfo_exists():
                    w.lift()
        self.root.after(10, go)

    def _hide(self, hidden: bool):
        if self._pop is None:
            return
        for w in self._windows():
            w.withdraw() if hidden else w.deiconify()
        if not hidden:
            self._raise()
            self.root.after(30, self._place)

    # ── dibujo ────────────────────────────────────────────────────────────────

    def _show(self):
        s = self.steps[self.i]
        shown = [k for k, st in enumerate(self.steps) if st.when is None or _safe(st.when, False)]
        pos = shown.index(self.i) + 1 if self.i in shown else 1
        self._title.config(text=s.title)
        self._text.config(text=s.text)
        self._count.config(text=t("tour.count", i=pos, n=len(shown)))
        last = pos == len(shown)
        self._next.config(text=t("tour.done") if last else t("tour.next"))
        self._prev.config(state=tk.NORMAL if pos > 1 else tk.DISABLED)
        for w in self._windows():
            w.lift()
        self._pop.update_idletasks()
        self._place()
        self._pop.focus_force()
        self.root.after(60, self._place)          # algunos sistemas ubican la ventana al mostrarla

    def _target_box(self):
        s = self.steps[self.i]
        w = _safe(s.target) if s.target else None
        if w is None:
            return None
        try:
            if not w.winfo_ismapped():
                return None
            x, y = w.winfo_rootx(), w.winfo_rooty()
            return x - PAD, y - PAD, x + w.winfo_width() + PAD, y + w.winfo_height() + PAD
        except tk.TclError:
            return None

    def _place(self):
        if not self._pop or self.i < 0:
            return
        r = self.root
        rx, ry, rw, rh = r.winfo_rootx(), r.winfo_rooty(), r.winfo_width(), r.winfo_height()
        box = self._target_box()
        if box is None:                                   # sin hueco: todo oscuro, globo al centro
            box = (rx + rw // 2, ry + rh // 2, rx + rw // 2, ry + rh // 2)
            hole = False
        else:
            hole = True
        x0, y0, x1, y1 = (max(rx, box[0]), max(ry, box[1]), min(rx + rw, box[2]), min(ry + rh, box[3]))
        rects = [(rx, ry, rw, y0 - ry), (rx, y1, rw, ry + rh - y1),
                 (rx, y0, x0 - rx, y1 - y0), (x1, y0, rx + rw - x1, y1 - y0)]
        for w, (x, y, ww, hh) in zip(self._shades, rects):
            _geom(w, x, y, ww, hh)
        ring = [(x0 - RING, y0 - RING, x1 - x0 + 2 * RING, RING), (x0 - RING, y1, x1 - x0 + 2 * RING, RING),
                (x0 - RING, y0, RING, y1 - y0), (x1, y0, RING, y1 - y0)] if hole else [(0, 0, 0, 0)] * 4
        for w, (x, y, ww, hh) in zip(self._ring, ring):
            _geom(w, x, y, ww, hh)
        _geom(self._glass, x0, y0, x1 - x0, y1 - y0) if hole else _geom(self._glass, 0, 0, 0, 0)
        pw, ph = self._pop.winfo_reqwidth(), self._pop.winfo_reqheight()
        if not hole:
            px, py = rx + (rw - pw) // 2, ry + (rh - ph) // 2
        else:                                              # primer lugar donde quepa: abajo, derecha, izquierda, arriba
            gap = 12
            options = [(x0, y1 + gap), (x1 + gap, y0), (x0 - pw - gap, y0), (x0, y0 - ph - gap)]
            px, py = options[0]
            for ox, oy in options:
                if rx <= ox and ox + pw <= rx + rw and ry <= oy and oy + ph <= ry + rh:
                    px, py = ox, oy
                    break
            px = min(max(px, rx + 8), rx + rw - pw - 8)
            py = min(max(py, ry + 8), ry + rh - ph - 8)
        self._pop.geometry(f"{pw}x{ph}+{int(px)}+{int(py)}")


def _geom(w: tk.Toplevel, x, y, ww, hh):
    """Sin ocultar ni volver a mostrar ventanas (en algunos sistemas eso mueve las demás):
    una franja vacía se manda fuera de la pantalla."""
    if ww <= 0 or hh <= 0:
        x, y, ww, hh = -50, -50, 1, 1
    w.geometry(f"{int(ww)}x{int(hh)}+{int(x)}+{int(y)}")


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default
