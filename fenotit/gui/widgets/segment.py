"""Piezas del panel Segmentar: la lista de operaciones de "Afinar máscara" y la lista de
clases de la segmentación por ejemplos."""
from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser
from tkinter import font as tkfont
from typing import Callable

from fenotit.core.segmentation import REFINE_OPS
from fenotit.gui.style.theme import CLASS_COLORS, COLORS, FONTS, SPACE
from fenotit.i18n import t



def _num(var, default, low=1, high=999):
    try:
        return min(high, max(low, int(var.get())))
    except (tk.TclError, ValueError):
        return default


class RefineList(tk.Frame):
    """Operaciones morfológicas en orden, con títulos arriba (Operación, Tamaño (px), Veces);
    ↑ ↓ ✕ aparecen al pasar el mouse. «+ Agregar operación» abre un menú."""

    def __init__(self, parent, on_change: Callable[[], None]):
        bg = COLORS["bg_panel"]
        super().__init__(parent, bg=bg)
        self.on_change = on_change
        self.ops: list[dict] = []
        self._names = {name: t(f"refine.{name}") for name in REFINE_OPS}
        self._rows = tk.Frame(self, bg=bg)
        self._rows.pack(fill=tk.X)
        self._add_btn = tk.Label(self, text="+ " + t("refine.add_short"), bg=bg, fg=COLORS["accent"],
                                 font=FONTS["small"], cursor="hand2")
        self._add_btn.pack(anchor="w", pady=(SPACE["sm"], 0))
        self._add_btn.bind("<Button-1>", lambda event: self._menu())
        self._muting = False
        self._width = 0
        self.bind("<Configure>", self._on_resize, add="+")

    def _on_resize(self, event):
        """Al cambiar el ancho, los nombres vuelven a repartirse en el espacio disponible."""
        if abs(event.width - self._width) > 8:
            self._width = event.width
            if self.ops:
                self.after_idle(self._render)

    def get(self) -> list[dict]:
        return [dict(option) for option in self.ops]

    def set(self, ops: list[dict] | None):
        self.ops = [dict(option) for option in ops or [] if option.get("op") in REFINE_OPS]
        self._render()

    def _menu(self):
        menu = tk.Menu(self, tearoff=0, font=FONTS["small"], bg=COLORS["bg_card"], fg=COLORS["text"],
                    activebackground=COLORS["accent_light"], activeforeground=COLORS["text"], bd=0)
        for key, name in self._names.items():
            menu.add_command(label=name, command=lambda option=key: self._add(option))
        button_widget = self._add_btn
        menu.tk_popup(button_widget.winfo_rootx(), button_widget.winfo_rooty() + button_widget.winfo_height())

    def _add(self, operation: str):
        self.ops.append({"op": operation} if not REFINE_OPS[operation]["size"] else {"op": operation, "size": 3, "iter": 1})
        self._render()
        self.on_change()

    def _move(self, i: int, delta: int):
        j = i + delta
        if 0 <= j < len(self.ops):
            self.ops[i], self.ops[j] = self.ops[j], self.ops[i]
            self._render()
            self.on_change()

    def _remove(self, i: int):
        del self.ops[i]
        self._render()
        self.on_change()

    def _widths(self):
        """Anchos de las columnas de tamaño y veces: lo que ocupe el título o el número."""
        font = tkfont.Font(font=FONTS["small"])
        return [max(font.measure(t(key)), font.measure("000") + 18) + 8 for key in ("refine.size_px", "refine.times")]

    def _grid(self, frame, widths):
        frame.grid_columnconfigure(0, weight=1)
        for column, width in enumerate(widths, 1):
            frame.grid_columnconfigure(column, minsize=width)

    def _render(self):
        card, small, muted = COLORS["bg_card"], FONTS["small"], COLORS["text_muted"]
        for widget in self._rows.winfo_children():
            widget.destroy()
        self._muting = True
        widths = self._widths()
        wrap = max(80, (self.winfo_width() if self.winfo_width() > 1 else 260) - sum(widths) - 16)        # ↑ ↓ ✕ tapan el final del nombre solo al pasar el mouse
        if self.ops:                                   # títulos una sola vez, arriba
            head = tk.Frame(self._rows, bg=COLORS["bg_panel"])
            head.pack(fill=tk.X, padx=1)
            self._grid(head, widths)
            for column, key in enumerate(("refine.op", "refine.size_px", "refine.times")):
                tk.Label(head, text=t(key), bg=COLORS["bg_panel"], fg=muted, font=small,
                         anchor="w").grid(row=0, column=column, sticky="w", padx=(SPACE["md"] if column == 0 else 0, 0))
        for i, operation in enumerate(self.ops):
            row = tk.Frame(self._rows, bg=card, highlightthickness=1, highlightbackground=COLORS["border"])
            row.pack(fill=tk.X, pady=1)
            self._grid(row, widths)
            cell = tk.Frame(row, bg=card)
            cell.grid(row=0, column=0, sticky="we")
            tk.Label(cell, text=self._names[operation["op"]], bg=card, fg=COLORS["text"], font=small, anchor="w",
                     justify="left", wraplength=wrap).pack(side=tk.LEFT, padx=(SPACE["md"], SPACE["xs"]), pady=SPACE["xs"])
            if REFINE_OPS[operation["op"]]["size"]:
                for column, key, high in ((1, "size", 99), (2, "iter", 20)):
                    var = tk.StringVar(value=str(operation.get(key, 3 if key == "size" else 1)))
                    tk.Spinbox(row, from_=1, to=high, textvariable=var, width=3, font=small, relief="flat",
                               bg=COLORS["bg_panel"], buttonbackground=card,
                               command=lambda operation=operation, op_key=key, variable=var, upper=high: self._set(operation, op_key, variable, upper)
                               ).grid(row=0, column=column, sticky="w")
                    var.trace_add("write", lambda *trace_args, operation=operation, op_key=key, variable=var, upper=high: self._set(operation, op_key, variable, upper))
            # ↑ ↓ ✕ aparecen sobre el renglón, a la derecha, solo al pasar el mouse
            tools = tk.Frame(row, bg=card)
            for sym, cmd in (("↑", lambda i=i: self._move(i, -1)), ("↓", lambda i=i: self._move(i, 1)),
                             ("✕", lambda i=i: self._remove(i))):
                label = tk.Label(tools, text=sym, bg=card, fg=muted, font=small, cursor="hand2", padx=SPACE["xs"])
                label.pack(side=tk.LEFT)
                label.bind("<Button-1>", lambda event, command=cmd: command())

            def show(active, tools_frame=tools, row_frame=row, cell_frame=cell):
                if active:
                    tools_frame.place(in_=cell_frame, relx=1.0, rely=0.5, anchor="e", x=-2)
                    tools_frame.lift()
                elif not (row_frame.winfo_containing(*row_frame.winfo_pointerxy()) or "").__str__().startswith(str(row_frame)):
                    tools_frame.place_forget()
            for widget in [row, cell, tools] + list(row.winfo_children()) + list(cell.winfo_children()) \
                    + list(tools.winfo_children()):
                widget.bind("<Enter>", lambda event, show_tools=show: show_tools(True), add="+")
                widget.bind("<Leave>", lambda event, show_tools=show: show_tools(False), add="+")
        self._muting = False

    def _set(self, operation: dict, key: str, var, high: int):
        if self._muting:
            return
        value = _num(var, operation.get(key, 1), 1, high)
        if operation.get(key) != value:
            operation[key] = value
            self.on_change()


class ClassList(tk.Frame):
    """Clases de la segmentación por ejemplos. La fila elegida es la del pincel; la primera
    es el fondo. "Objeto" = la clase forma parte de los objetos (p. ej. sano y lesión sí,
    fondo no). Con `show_objects(False)` la casilla no se muestra (análisis sin objetos)."""

    def __init__(self, parent, on_change: Callable[[], None], on_pick: Callable[[int], None]):
        bg = COLORS["bg_panel"]
        super().__init__(parent, bg=bg)
        self.on_change, self.on_pick = on_change, on_pick
        self.classes: list[dict] = []
        self.current = 1
        self._objects = True
        self._rows = tk.Frame(self, bg=bg)
        self._rows.pack(fill=tk.X)
        add = tk.Label(self, text="+ " + t("cls.add"), bg=bg, fg=COLORS["accent"], font=FONTS["small"],
                       cursor="hand2")
        add.pack(anchor="w", pady=(SPACE["xs"], 0))
        add.bind("<Button-1>", lambda event: self._add())

    def get(self) -> list[dict]:
        return [dict(entry) for entry in self.classes]

    def set(self, classes: list[dict]):
        self.classes = [dict(entry) for entry in classes]
        self.current = min(self.current, len(self.classes) - 1)
        self._render()

    def show_objects(self, active: bool):
        if active != self._objects:
            self._objects = active
            self._render()

    def pick(self, i: int):
        if 0 <= i < len(self.classes):
            self.current = i
            self._render()
            self.on_pick(i)

    def _add(self):
        used = {entry["color"] for entry in self.classes}
        color = next((entry for entry in CLASS_COLORS if entry not in used), COLORS["text_muted"])
        self.classes.append({"name": t("cls.new", n=len(self.classes)), "color": color, "object": True})
        self.current = len(self.classes) - 1
        self._render()
        self.on_change()
        self.on_pick(self.current)

    def _remove(self, i: int):
        if i < 2:
            return
        del self.classes[i]
        self.current = min(self.current, len(self.classes) - 1)
        self._render()
        self.on_change(removed=i)

    def _color(self, i: int):
        got = colorchooser.askcolor(self.classes[i]["color"], parent=self, title=self.classes[i]["name"])
        if got and got[1]:
            self.classes[i]["color"] = got[1].upper()
            self._render()
            self.on_change()

    def _render(self):
        small = FONTS["small"]
        for widget in self._rows.winfo_children():
            widget.destroy()
        for i, entry in enumerate(self.classes):
            active = i == self.current
            bg = COLORS["accent_light"] if active else COLORS["bg_card"]
            row = tk.Frame(self._rows, bg=bg, highlightthickness=1,
                           highlightbackground=COLORS["accent"] if active else COLORS["border"], cursor="hand2")
            row.pack(fill=tk.X, pady=1)
            label = tk.Label(row, bg=entry["color"], width=2, cursor="hand2")
            label.pack(side=tk.LEFT, padx=SPACE["sm"], pady=SPACE["sm"])
            label.bind("<Button-1>", lambda event, i=i: self._color(i))
            var = tk.StringVar(value=entry["name"])
            ent = tk.Entry(row, textvariable=var, width=12, font=small, relief="flat", bg=bg)
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ent.bind("<FocusIn>", lambda event, i=i: self.current != i and self.after_idle(lambda: self.pick(i)))
            var.trace_add("write", lambda *trace_args, i=i, variable=var: self._rename(i, variable.get()))
            if i >= 2:
                x = tk.Label(row, text="✕", bg=bg, fg=COLORS["text_muted"], font=small, cursor="hand2", padx=SPACE["sm"])
                x.pack(side=tk.RIGHT)
                x.bind("<Button-1>", lambda event, i=i: self._remove(i))
            if i and self._objects:
                marks = tk.BooleanVar(value=bool(entry.get("object", True)))
                tk.Checkbutton(row, text=t("cls.object"), variable=marks, bg=bg, activebackground=bg,
                               selectcolor=COLORS["bg_card"], font=small,
                               command=lambda i=i, variable=marks: self._object(i, variable.get())).pack(side=tk.RIGHT)
            elif not i:
                tk.Label(row, text=t("cls.background"), bg=bg, fg=COLORS["text_muted"],
                         font=small).pack(side=tk.RIGHT, padx=SPACE["sm"])
            for widget in (row,):
                widget.bind("<Button-1>", lambda event, i=i: self.pick(i))

    def _rename(self, i: int, name: str):
        if i < len(self.classes) and name.strip() and self.classes[i]["name"] != name:
            self.classes[i]["name"] = name
            self.on_change()

    def _object(self, i: int, active: bool):
        self.classes[i]["object"] = bool(active)
        self.on_change()


class CategoryList(tk.Frame):
    """Categorías de las regiones del modelo (p. ej. Planta, Maleza, Suelo). La fila elegida
    es la que pone un clic sobre una región; el cuadrito de color cambia el color."""

    def __init__(self, parent, on_change: Callable[[], None], on_pick: Callable[[int], None]):
        bg = COLORS["bg_panel"]
        super().__init__(parent, bg=bg)
        self.on_change, self.on_pick = on_change, on_pick
        self.cats: list[dict] = []
        self.current = 0
        self._rows = tk.Frame(self, bg=bg)
        self._rows.pack(fill=tk.X)
        add = tk.Label(self, text="+ " + t("cat.add"), bg=bg, fg=COLORS["accent"], font=FONTS["small"],
                       cursor="hand2")
        add.pack(anchor="w", pady=(SPACE["xs"], 0))
        add.bind("<Button-1>", lambda event: self._add())

    def get(self) -> list[dict]:
        return [dict(entry) for entry in self.cats]

    def set(self, cats: list[dict]):
        self.cats = [dict(entry) for entry in cats]
        self.current = min(self.current, len(self.cats) - 1)
        self._render()

    def pick(self, i: int):
        if 0 <= i < len(self.cats):
            self.current = i
            self._render()
            self.on_pick(i)

    def _add(self):
        used = {entry["color"] for entry in self.cats}
        color = next((entry for entry in CLASS_COLORS[1:] if entry not in used), COLORS["text_muted"])
        self.cats.append({"name": t("cat.new", n=len(self.cats) + 1), "color": color})
        self.current = len(self.cats) - 1
        self._render()
        self.on_change()
        self.on_pick(self.current)

    def _remove(self, i: int):
        if len(self.cats) < 2:
            return
        del self.cats[i]
        self.current = min(self.current, len(self.cats) - 1)
        self._render()
        self.on_change(removed=i)

    def _color(self, i: int):
        got = colorchooser.askcolor(self.cats[i]["color"], parent=self, title=self.cats[i]["name"])
        if got and got[1]:
            self.cats[i]["color"] = got[1].upper()
            self._render()
            self.on_change()

    def _render(self):
        small = FONTS["small"]
        for widget in self._rows.winfo_children():
            widget.destroy()
        for i, entry in enumerate(self.cats):
            active = i == self.current
            bg = COLORS["accent_light"] if active else COLORS["bg_card"]
            row = tk.Frame(self._rows, bg=bg, highlightthickness=1, cursor="hand2",
                           highlightbackground=COLORS["accent"] if active else COLORS["border"])
            row.pack(fill=tk.X, pady=1)
            label = tk.Label(row, bg=entry["color"], width=2, cursor="hand2")
            label.pack(side=tk.LEFT, padx=SPACE["sm"], pady=SPACE["sm"])
            label.bind("<Button-1>", lambda event, i=i: self._color(i))
            var = tk.StringVar(value=entry["name"])
            ent = tk.Entry(row, textvariable=var, width=12, font=small, relief="flat", bg=bg)
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ent.bind("<FocusIn>", lambda event, i=i: self.current != i and self.after_idle(lambda: self.pick(i)))
            var.trace_add("write", lambda *trace_args, i=i, variable=var: self._rename(i, variable.get()))
            if len(self.cats) > 1:
                x = tk.Label(row, text="✕", bg=bg, fg=COLORS["text_muted"], font=small, cursor="hand2", padx=SPACE["sm"])
                x.pack(side=tk.RIGHT)
                x.bind("<Button-1>", lambda event, i=i: self._remove(i))
            row.bind("<Button-1>", lambda event, i=i: self.pick(i))

    def _rename(self, i: int, name: str):
        if i < len(self.cats) and name.strip() and self.cats[i]["name"] != name:
            self.cats[i]["name"] = name
            self.on_change()
