"""Piezas del panel Segmentar: la lista de operaciones de "Afinar máscara" y la lista de
clases de la segmentación por ejemplos."""
from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser, ttk
from typing import Callable

from fenotit.core.segmentation import REFINE_OPS
from fenotit.gui.theme import COLORS, FONTS
from fenotit.i18n import t

# colores para clases nuevas (se reparten en orden)
CLASS_COLORS = ("#9E9E9E", "#2166AC", "#2E9E44", "#D62728", "#F2B705", "#8E44AD", "#E67E22", "#17BECF")


def _num(var, default, lo=1, hi=999):
    try:
        return min(hi, max(lo, int(var.get())))
    except (tk.TclError, ValueError):
        return default


class RefineList(tk.Frame):
    """Operaciones morfológicas en orden, una por renglón: nombre, tamaño (px) y veces;
    ↑ ↓ ✕ aparecen al pasar el mouse. «+ Agregar operación» abre un menú."""

    def __init__(self, parent, on_change: Callable[[], None]):
        bg = COLORS["bg_panel"]
        super().__init__(parent, bg=bg)
        self.on_change = on_change
        self.ops: list[dict] = []
        self._names = {k: t(f"refine.{k}") for k in REFINE_OPS}
        self._rows = tk.Frame(self, bg=bg)
        self._rows.pack(fill=tk.X)
        self._add_btn = tk.Label(self, text="+ " + t("refine.add_short"), bg=bg, fg=COLORS["accent"],
                                 font=FONTS["small"], cursor="hand2")
        self._add_btn.pack(anchor="w", pady=(3, 0))
        self._add_btn.bind("<Button-1>", lambda e: self._menu())
        self._muting = False

    def get(self) -> list[dict]:
        return [dict(o) for o in self.ops]

    def set(self, ops: list[dict] | None):
        self.ops = [dict(o) for o in ops or [] if o.get("op") in REFINE_OPS]
        self._render()

    def _menu(self):
        m = tk.Menu(self, tearoff=0, font=FONTS["small"], bg=COLORS["bg_card"], fg=COLORS["text"],
                    activebackground=COLORS["accent_light"], activeforeground=COLORS["text"], bd=0)
        for op, name in self._names.items():
            m.add_command(label=name, command=lambda o=op: self._add(o))
        b = self._add_btn
        m.tk_popup(b.winfo_rootx(), b.winfo_rooty() + b.winfo_height())

    def _add(self, op: str):
        self.ops.append({"op": op} if not REFINE_OPS[op]["size"] else {"op": op, "size": 3, "iter": 1})
        self._render()
        self.on_change()

    def _move(self, i: int, d: int):
        j = i + d
        if 0 <= j < len(self.ops):
            self.ops[i], self.ops[j] = self.ops[j], self.ops[i]
            self._render()
            self.on_change()

    def _remove(self, i: int):
        del self.ops[i]
        self._render()
        self.on_change()

    def _render(self):
        card, small, muted = COLORS["bg_card"], FONTS["small"], COLORS["text_muted"]
        for w in self._rows.winfo_children():
            w.destroy()
        self._muting = True
        for i, o in enumerate(self.ops):
            row = tk.Frame(self._rows, bg=card, highlightthickness=1, highlightbackground=COLORS["border"])
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=self._names[o["op"]], bg=card, fg=COLORS["text"], font=small,
                     anchor="w").pack(side=tk.LEFT, padx=(6, 2), pady=2)
            tools = tk.Frame(row, bg=card)                 # ↑ ↓ ✕ solo al pasar el mouse
            for sym, cmd in (("✕", lambda i=i: self._remove(i)), ("↓", lambda i=i: self._move(i, 1)),
                             ("↑", lambda i=i: self._move(i, -1))):
                lb = tk.Label(tools, text=sym, bg=card, fg=muted, font=small, cursor="hand2", padx=2)
                lb.pack(side=tk.RIGHT)
                lb.bind("<Button-1>", lambda e, c=cmd: c())
            tools.pack(side=tk.RIGHT, padx=(0, 2))
            if REFINE_OPS[o["op"]]["size"]:            # de derecha a izquierda: × veces, px tamaño
                for key, unit, hi in (("iter", "×", 20), ("size", "px", 99)):
                    var = tk.StringVar(value=str(o.get(key, 3 if key == "size" else 1)))
                    tk.Label(row, text=unit, bg=card, fg=muted, font=small).pack(side=tk.RIGHT, padx=(1, 4))
                    tk.Spinbox(row, from_=1, to=hi, textvariable=var, width=2, font=small, relief="flat",
                               bg=COLORS["bg_panel"], buttonbackground=card,
                               command=lambda o=o, k=key, v=var, h=hi: self._set(o, k, v, h)).pack(side=tk.RIGHT)
                    var.trace_add("write", lambda *a, o=o, k=key, v=var, h=hi: self._set(o, k, v, h))

            def show(on, f=tools):
                for c in f.winfo_children():
                    c.config(fg=muted if on else card)
            show(False)
            for w in [row] + list(row.winfo_children()):
                w.bind("<Enter>", lambda e, s=show: s(True), add="+")
                w.bind("<Leave>", lambda e, s=show: s(False), add="+")
        self._muting = False

    def _set(self, o: dict, key: str, var, hi: int):
        if self._muting:
            return
        v = _num(var, o.get(key, 1), 1, hi)
        if o.get(key) != v:
            o[key] = v
            self.on_change()


class ClassList(tk.Frame):
    """Clases de la segmentación por ejemplos. La fila elegida es la del pincel; la primera
    es el fondo. "Objeto" = la clase forma parte de los objetos (p. ej. sano y lesión sí,
    fondo no)."""

    def __init__(self, parent, on_change: Callable[[], None], on_pick: Callable[[int], None]):
        bg = COLORS["bg_panel"]
        super().__init__(parent, bg=bg)
        self.on_change, self.on_pick = on_change, on_pick
        self.classes: list[dict] = []
        self.current = 1
        self._rows = tk.Frame(self, bg=bg)
        self._rows.pack(fill=tk.X)
        add = tk.Label(self, text="+ " + t("cls.add"), bg=bg, fg=COLORS["accent"], font=FONTS["small"],
                       cursor="hand2")
        add.pack(anchor="w", pady=(2, 0))
        add.bind("<Button-1>", lambda e: self._add())

    def get(self) -> list[dict]:
        return [dict(c) for c in self.classes]

    def set(self, classes: list[dict]):
        self.classes = [dict(c) for c in classes]
        self.current = min(self.current, len(self.classes) - 1)
        self._render()

    def pick(self, i: int):
        if 0 <= i < len(self.classes):
            self.current = i
            self._render()
            self.on_pick(i)

    def _add(self):
        used = {c["color"] for c in self.classes}
        color = next((c for c in CLASS_COLORS if c not in used), "#444444")
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
        for w in self._rows.winfo_children():
            w.destroy()
        for i, c in enumerate(self.classes):
            on = i == self.current
            bg = COLORS["accent_light"] if on else COLORS["bg_card"]
            row = tk.Frame(self._rows, bg=bg, highlightthickness=1,
                           highlightbackground=COLORS["accent"] if on else COLORS["border"], cursor="hand2")
            row.pack(fill=tk.X, pady=1)
            sw = tk.Label(row, bg=c["color"], width=2, cursor="hand2")
            sw.pack(side=tk.LEFT, padx=4, pady=3)
            sw.bind("<Button-1>", lambda e, i=i: self._color(i))
            var = tk.StringVar(value=c["name"])
            ent = tk.Entry(row, textvariable=var, width=12, font=small, relief="flat", bg=bg)
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ent.bind("<FocusIn>", lambda e, i=i: self.current != i and self.after_idle(lambda: self.pick(i)))
            var.trace_add("write", lambda *a, i=i, v=var: self._rename(i, v.get()))
            if i >= 2:
                x = tk.Label(row, text="✕", bg=bg, fg=COLORS["text_muted"], font=small, cursor="hand2", padx=3)
                x.pack(side=tk.RIGHT)
                x.bind("<Button-1>", lambda e, i=i: self._remove(i))
            if i:
                ov = tk.BooleanVar(value=bool(c.get("object", True)))
                tk.Checkbutton(row, text=t("cls.object"), variable=ov, bg=bg, activebackground=bg,
                               selectcolor=COLORS["bg_card"], font=small,
                               command=lambda i=i, v=ov: self._object(i, v.get())).pack(side=tk.RIGHT)
            else:
                tk.Label(row, text=t("cls.background"), bg=bg, fg=COLORS["text_muted"],
                         font=small).pack(side=tk.RIGHT, padx=4)
            for w in (row,):
                w.bind("<Button-1>", lambda e, i=i: self.pick(i))

    def _rename(self, i: int, name: str):
        if i < len(self.classes) and name.strip() and self.classes[i]["name"] != name:
            self.classes[i]["name"] = name
            self.on_change()

    def _object(self, i: int, on: bool):
        self.classes[i]["object"] = bool(on)
        self.on_change()
