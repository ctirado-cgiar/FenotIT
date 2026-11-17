"""Fila que pasa sus elementos a la línea siguiente cuando no caben a lo ancho."""
import tkinter as tk


class FlowRow(tk.Frame):
    """Contenedor de izquierda a derecha: los hijos agregados con `add(widget, gap)` se
    acomodan en líneas según el ancho disponible. `fill=True` reparte el ancho de cada línea
    entre sus elementos (botones de igual ancho)."""

    def __init__(self, parent, fill: bool = False, line_gap: int = 2, **kwargs):
        super().__init__(parent, **kwargs)
        self._items: list[tuple[tk.Widget, int]] = []
        self._fill, self._line_gap = fill, line_gap
        self._width = 0
        self.bind("<Configure>", self._on_resize)
        self.bind("<Map>", lambda event: self.reflow())

    def add(self, widget: tk.Widget, gap: int = 0):
        """Agrega `widget` (hijo de esta fila) con `gap` px de separación a su derecha."""
        self._items.append((widget, gap))
        self.after_idle(self.reflow)

    def visible(self, widget: tk.Widget, active: bool):
        """Muestra u oculta un elemento sin cambiar su lugar."""
        widget._flow_hidden = not active
        self.reflow()

    def _on_resize(self, event):
        if event.width != self._width:
            self._width = event.width
            self.reflow()

    def reflow(self):
        items = [(widget, gap) for widget, gap in self._items if not getattr(widget, "_flow_hidden", False)]
        for widget, _ in self._items:
            widget.grid_forget()
        width = self.winfo_width() if self.winfo_width() > 1 else 10 ** 6
        if self._fill:                                 # todos en una línea de igual ancho, o uno por línea
            widest = max((widget.winfo_reqwidth() for widget, _ in items), default=0)
            one_line = widest * len(items) + sum(gap for _, gap in items[:-1]) <= width
            lines = [items] if one_line else [[item] for item in items]
        else:
            lines, line, used = [], [], 0
            for widget, gap in items:
                need = widget.winfo_reqwidth() + gap
                if line and used + need - gap > width:
                    lines.append(line)
                    line, used = [], 0
                line.append((widget, gap))
                used += need
            if line:
                lines.append(line)
        cols = max((len(line) for line in lines), default=0)
        for column in range(len(self._items) + 1):
            self.columnconfigure(column, weight=0, uniform="")
        for row, line in enumerate(lines):
            for column, (widget, gap) in enumerate(line):
                last = column == len(line) - 1
                widget.grid(row=row, column=column, columnspan=cols - column if last else 1, sticky="ew" if self._fill else "w",
                       padx=(0, 0 if last and self._fill else gap), pady=(self._line_gap if row else 0, 0))
        if self._fill:
            for column in range(cols):
                self.columnconfigure(column, weight=1, uniform="flow")
        else:
            self.columnconfigure(cols, weight=1)
