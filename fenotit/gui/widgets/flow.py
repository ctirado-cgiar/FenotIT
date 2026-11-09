"""Fila que pasa sus elementos a la línea siguiente cuando no caben a lo ancho."""
import tkinter as tk


class FlowRow(tk.Frame):
    """Contenedor de izquierda a derecha: los hijos agregados con `add(widget, gap)` se
    acomodan en líneas según el ancho disponible. `fill=True` reparte el ancho de cada línea
    entre sus elementos (botones de igual ancho)."""

    def __init__(self, parent, fill: bool = False, line_gap: int = 2, **kw):
        super().__init__(parent, **kw)
        self._items: list[tuple[tk.Widget, int]] = []
        self._fill, self._line_gap = fill, line_gap
        self._width = 0
        self.bind("<Configure>", self._on_resize)
        self.bind("<Map>", lambda e: self.reflow())

    def add(self, widget: tk.Widget, gap: int = 0):
        """Agrega `widget` (hijo de esta fila) con `gap` px de separación a su derecha."""
        self._items.append((widget, gap))
        self.after_idle(self.reflow)

    def visible(self, widget: tk.Widget, on: bool):
        """Muestra u oculta un elemento sin cambiar su lugar."""
        widget._flow_hidden = not on
        self.reflow()

    def _on_resize(self, e):
        if e.width != self._width:
            self._width = e.width
            self.reflow()

    def reflow(self):
        items = [(w, g) for w, g in self._items if not getattr(w, "_flow_hidden", False)]
        for w, _ in self._items:
            w.grid_forget()
        width = self.winfo_width() if self.winfo_width() > 1 else 10 ** 6
        if self._fill:                                 # todos en una línea de igual ancho, o uno por línea
            widest = max((w.winfo_reqwidth() for w, _ in items), default=0)
            one_line = widest * len(items) + sum(g for _, g in items[:-1]) <= width
            lines = [items] if one_line else [[it] for it in items]
        else:
            lines, line, used = [], [], 0
            for w, g in items:
                need = w.winfo_reqwidth() + g
                if line and used + need - g > width:
                    lines.append(line)
                    line, used = [], 0
                line.append((w, g))
                used += need
            if line:
                lines.append(line)
        cols = max((len(ln) for ln in lines), default=0)
        for c in range(len(self._items) + 1):
            self.columnconfigure(c, weight=0, uniform="")
        for r, ln in enumerate(lines):
            for c, (w, g) in enumerate(ln):
                last = c == len(ln) - 1
                w.grid(row=r, column=c, columnspan=cols - c if last else 1, sticky="ew" if self._fill else "w",
                       padx=(0, 0 if last and self._fill else g), pady=(self._line_gap if r else 0, 0))
        if self._fill:
            for c in range(cols):
                self.columnconfigure(c, weight=1, uniform="flow")
        else:
            self.columnconfigure(cols, weight=1)
