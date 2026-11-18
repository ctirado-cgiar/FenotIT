"""Recorrido guiado (como driver.js): oscurece la ventana salvo el elemento del paso, lo
enmarca en azul y muestra al lado un globo con título, texto y Anterior / Siguiente /
Cerrar. Durante el recorrido solo responden Anterior / Siguiente / Terminar (el elemento se ve, no se toca).

El oscurecido son cuatro ventanas semitransparentes alrededor del hueco."""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import Callable

# button crea un botón con el estilo del tema.
from fenotit.gui.style.components import button
# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras); SPACE (márgenes).
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
# t devuelve cada texto de la ventana en el idioma elegido (fenotit/lang).
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

    def start(self, start_key: str | None = None):
        """at = clave del paso donde empezar (el que corresponde a lo que el usuario ya hizo);
        Anterior sigue llevando a los pasos de antes."""
        self._build()
        keys = [step.key for step in self.steps]
        self.i = keys.index(start_key) - 1 if start_key in keys else -1
        self._go(+1)

    def _go(self, direction: int):
        i = self.i + direction
        while 0 <= i < len(self.steps):
            step = self.steps[i]
            if step.when is None or _safe(step.when, False):
                break
            i += direction
        if not 0 <= i < len(self.steps):
            if direction > 0:
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
        for widget in self._windows() if self._pop else []:
            try:
                widget.destroy()
            except tk.TclError:
                pass
        self._shades, self._ring, self._pop = [], [], None
        if self.on_close:
            self.on_close()

    # ── ventanas ──────────────────────────────────────────────────────────────

    def _overlay(self, color: str, alpha: float) -> tk.Toplevel:
        window = tk.Toplevel(self.root)
        window.overrideredirect(True)
        window.configure(bg=color)
        window.transient(self.root)
        try:
            window.attributes("-alpha", alpha)
        except tk.TclError:
            pass
        window.bind("<Button-1>", lambda event: "break")              # fuera del hueco no se hace clic
        return window

    def _build(self):
        self._shades = [self._overlay(COLORS["black"], DIM) for _ in range(4)]
        self._ring = [self._overlay(COLORS["accent"], 1.0) for _ in range(4)]
        # durante el recorrido solo se usan Anterior / Siguiente / Terminar: el elemento
        # resaltado se ve pero no recibe clics (un menú abierto quedaría debajo del oscurecido)
        self._glass = self._overlay(COLORS["white"], 0.01)
        pop = tk.Toplevel(self.root)
        pop.overrideredirect(True)
        pop.transient(self.root)
        pop.configure(bg=COLORS["accent"])
        body = tk.Frame(pop, bg=COLORS["bg_card"], padx=SPACE["lg"], pady=SPACE["lg"])
        body.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        top = tk.Frame(body, bg=COLORS["bg_card"])
        top.pack(fill=tk.X)
        self._title = tk.Label(top, bg=COLORS["bg_card"], fg=COLORS["accent"], font=FONTS["title"],
                               anchor="w", justify="left", wraplength=WIDTH - 60)
        self._title.pack(side=tk.LEFT, fill=tk.X, expand=True)
        x = tk.Label(top, text="✕", bg=COLORS["bg_card"], fg=COLORS["text_muted"], cursor="hand2", font=FONTS["small"])
        x.pack(side=tk.RIGHT, anchor="n")
        x.bind("<Button-1>", lambda event: self.close())
        self._text = tk.Label(body, bg=COLORS["bg_card"], fg=COLORS["text"], font=FONTS["body"], justify="left",
                              anchor="w", wraplength=WIDTH - 28)
        self._text.pack(fill=tk.X, pady=(SPACE["md"], SPACE["lg"]))
        bar = tk.Frame(body, bg=COLORS["bg_card"])
        bar.pack(fill=tk.X)
        self._count = tk.Label(bar, bg=COLORS["bg_card"], fg=COLORS["text_muted"], font=FONTS["small"])
        self._count.pack(side=tk.LEFT)
        self._next = button(bar, command=lambda: self._go(+1), kind="primary", size="small", pady=SPACE["sm"])
        self._next.pack(side=tk.RIGHT)
        self._prev = button(bar, text=t("tour.prev"), command=lambda: self._go(-1), kind="secondary", size="small", pady=SPACE["sm"])
        self._prev.pack(side=tk.RIGHT, padx=(0, SPACE["md"]))
        for sequence, handler in (("<Escape>", lambda event: self.close()), ("<Right>", lambda event: self._go(+1)),
                     ("<Left>", lambda event: self._go(-1)), ("<Return>", lambda event: self._go(+1))):
            pop.bind(sequence, handler)
        self._pop = pop
        # la ventana principal no debe tapar el recorrido: al hacer clic en ella (en el hueco o
        # fuera de la app y volver) se suben de nuevo; al minimizarla se esconden con ella
        for seq, func in (("<Configure>", lambda event: event.widget is self.root and self._place()),
                        ("<FocusIn>", lambda event: self._raise()),
                        ("<ButtonRelease-1>", lambda event: self._raise()),
                        ("<Unmap>", lambda event: event.widget is self.root and self._hide(True)),
                        ("<Map>", lambda event: event.widget is self.root and self._hide(False))):
            self._binds.append((seq, self.root.bind(seq, func, add="+")))

    def _windows(self):
        return self._shades + self._ring + [self._glass] + ([self._pop] if self._pop else [])

    def _raise(self):
        if self._pop is None or getattr(self, "_raising", False):
            return
        self._raising = True

        def go():
            self._raising = False
            for widget in self._windows():
                if widget.winfo_exists():
                    widget.lift()
        self.root.after(10, go)

    def _hide(self, hidden: bool):
        if self._pop is None:
            return
        for widget in self._windows():
            widget.withdraw() if hidden else widget.deiconify()
        if not hidden:
            self._raise()
            self.root.after(30, self._place)

    # ── dibujo ────────────────────────────────────────────────────────────────

    def _show(self):
        step = self.steps[self.i]
        shown = [index for index, state in enumerate(self.steps) if state.when is None or _safe(state.when, False)]
        pos = shown.index(self.i) + 1 if self.i in shown else 1
        self._title.config(text=step.title)
        self._text.config(text=step.text)
        self._count.config(text=t("tour.count", i=pos, n=len(shown)))
        last = pos == len(shown)
        self._next.config(text=t("tour.done") if last else t("tour.next"))
        self._prev.config(state=tk.NORMAL if pos > 1 else tk.DISABLED)
        for widget in self._windows():
            widget.lift()
        self._pop.update_idletasks()
        self._place()
        self._pop.focus_force()
        self.root.after(60, self._place)          # algunos sistemas ubican la ventana al mostrarla

    def _target_box(self):
        step = self.steps[self.i]
        widget = _safe(step.target) if step.target else None
        if widget is None:
            return None
        try:
            if not widget.winfo_ismapped():
                return None
            x, y = widget.winfo_rootx(), widget.winfo_rooty()
            return x - PAD, y - PAD, x + widget.winfo_width() + PAD, y + widget.winfo_height() + PAD
        except tk.TclError:
            return None

    def _place(self):
        if not self._pop or self.i < 0:
            return
        root = self.root
        root_x, root_y, root_width, root_height = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
        box = self._target_box()
        if box is None:                                   # sin hueco: todo oscuro, globo al centro
            box = (root_x + root_width // 2, root_y + root_height // 2, root_x + root_width // 2, root_y + root_height // 2)
            hole = False
        else:
            hole = True
        x0, y0, x1, y1 = (max(root_x, box[0]), max(root_y, box[1]), min(root_x + root_width, box[2]), min(root_y + root_height, box[3]))
        rects = [(root_x, root_y, root_width, y0 - root_y), (root_x, y1, root_width, root_y + root_height - y1),
                 (root_x, y0, x0 - root_x, y1 - y0), (x1, y0, root_x + root_width - x1, y1 - y0)]
        for widget, (x, y, width, height) in zip(self._shades, rects):
            _geom(widget, x, y, width, height)
        ring = [(x0 - RING, y0 - RING, x1 - x0 + 2 * RING, RING), (x0 - RING, y1, x1 - x0 + 2 * RING, RING),
                (x0 - RING, y0, RING, y1 - y0), (x1, y0, RING, y1 - y0)] if hole else [(0, 0, 0, 0)] * 4
        for widget, (x, y, width, height) in zip(self._ring, ring):
            _geom(widget, x, y, width, height)
        _geom(self._glass, x0, y0, x1 - x0, y1 - y0) if hole else _geom(self._glass, 0, 0, 0, 0)
        panel_width, panel_height = self._pop.winfo_reqwidth(), self._pop.winfo_reqheight()
        if not hole:
            pixel_x, pixel_y = root_x + (root_width - panel_width) // 2, root_y + (root_height - panel_height) // 2
        else:                                              # primer lugar donde quepa: abajo, derecha, izquierda, arriba
            gap = 12
            options = [(x0, y1 + gap), (x1 + gap, y0), (x0 - panel_width - gap, y0), (x0, y0 - panel_height - gap)]
            pixel_x, pixel_y = options[0]
            for offset_x, offset_y in options:
                if root_x <= offset_x and offset_x + panel_width <= root_x + root_width and root_y <= offset_y and offset_y + panel_height <= root_y + root_height:
                    pixel_x, pixel_y = offset_x, offset_y
                    break
            pixel_x = min(max(pixel_x, root_x + 8), root_x + root_width - panel_width - 8)
            pixel_y = min(max(pixel_y, root_y + 8), root_y + root_height - panel_height - 8)
        self._pop.geometry(f"{panel_width}x{panel_height}+{int(pixel_x)}+{int(pixel_y)}")


def _geom(window: tk.Toplevel, x, y, width, height):
    """Sin ocultar ni volver a mostrar ventanas (en algunos sistemas eso mueve las demás):
    una franja vacía se manda fuera de la pantalla."""
    if width <= 0 or height <= 0:
        x, y, width, height = -50, -50, 1, 1
    window.geometry(f"{int(width)}x{int(height)}+{int(x)}+{int(y)}")


def _safe(func, default=None):
    try:
        return func()
    except Exception:
        return default
