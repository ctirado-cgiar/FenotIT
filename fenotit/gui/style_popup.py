"""Colores de dibujo: máscara de segmentación, contornos y puntos (ventana pequeña
bajo el ícono de paleta). Los cambios se ven al momento."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from fenotit.core.pipeline import overlay
from fenotit.i18n import t

_SWATCHES = ("green", "magenta", "yellow", "cyan", "red", "orange", "blue", "white", "black")
_D = 14            # diámetro del punto de color


def _hex(bgr) -> str:
    return "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])


class StylePopup(tk.Toplevel):

    def __init__(self, parent, anchor: tk.Widget, style: dict, colors: dict, on_change):
        super().__init__(parent)
        self.wm_overrideredirect(True)
        self.configure(bg=colors["border"])
        self._c = colors
        self._style = {**overlay.DEFAULT_STYLE, **(style or {})}
        self._on_change = on_change
        bg = colors["bg_card"]
        box = tk.Frame(self, bg=bg, padx=10, pady=8)
        box.pack(padx=1, pady=1)
        self._dots: dict[str, tk.Canvas] = {}
        row = 0
        for role in overlay.ROLES:
            tk.Label(box, text=t(f"style.{role}"), bg=bg, fg=colors["text"], font=("Segoe UI", 8),
                     anchor="w").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            names = (() if role == "mask" else ("auto",)) + _SWATCHES
            c = tk.Canvas(box, bg=bg, highlightthickness=0, width=len(names) * (_D + 6), height=_D + 6,
                          cursor="hand2")
            c.grid(row=row, column=1, sticky="w")
            c.bind("<Button-1>", lambda e, r=role, n=names: self._click(e, r, n))
            self._dots[role] = (c, names)
            row += 1
            if role == "mask":
                row = self._slider(box, row, "style.opacity", 0.1, 0.9, "mask_alpha", "{:.0%}")
            if role == "outline":
                row = self._slider(box, row, "style.width", 0.5, 4.0, "width", "{:.1f} px")
        tk.Label(box, text=t("style.hint"), bg=bg, fg=colors["text_muted"], font=("Segoe UI", 7),
                 justify="left", wraplength=240).grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._paint()
        self.attributes("-topmost", True)
        self.bind("<Escape>", lambda e: self.destroy())
        self._place_under(anchor)
        # Se cierra con Esc, con el ícono o con un clic en la ventana principal (no con la
        # pérdida de foco: en Windows una ventana sin marco no siempre recibe el foco).

    def _place_under(self, anchor):
        self.update_idletasks()
        w = self.winfo_reqwidth()
        top = anchor.winfo_toplevel()
        left, right = top.winfo_rootx(), top.winfo_rootx() + top.winfo_width()
        x = anchor.winfo_rootx() + anchor.winfo_width() // 2 - w // 2
        x = max(left + 4, min(x, right - w - 4))
        self.wm_geometry(f"+{x}+{anchor.winfo_rooty() + anchor.winfo_height() + 2}")

    def _slider(self, box, row, key, lo, hi, field, fmt):
        bg = self._c["bg_card"]
        tk.Label(box, text=t(key), bg=bg, fg=self._c["text_muted"], font=("Segoe UI", 7),
                 anchor="w").grid(row=row, column=0, sticky="w", padx=(10, 8))
        f = tk.Frame(box, bg=bg)
        f.grid(row=row, column=1, sticky="we")
        value = tk.Label(f, text=fmt.format(self._style[field]), bg=bg, fg=self._c["text"],
                         font=("Segoe UI", 7), width=6, anchor="e")
        var = tk.DoubleVar(value=float(self._style[field]))

        def changed(_v=None):
            v = round(var.get(), 2)
            value.config(text=fmt.format(v))
            self._style[field] = v
            self._on_change(dict(self._style))
        ttk.Scale(f, from_=lo, to=hi, variable=var, command=changed, length=150).pack(side=tk.LEFT)
        value.pack(side=tk.LEFT)
        return row + 1

    def _click(self, e, role, names):
        i = int(e.x // (_D + 6))
        if 0 <= i < len(names):
            self._style[role] = names[i]
            self._paint()
            self._on_change(dict(self._style))

    def _paint(self):
        for role, (c, names) in self._dots.items():
            c.delete("all")
            for i, name in enumerate(names):
                x, y = i * (_D + 6) + 3, 3
                sel = self._style[role] == name
                if sel:
                    c.create_oval(x - 2, y - 2, x + _D + 2, y + _D + 2, outline=self._c["accent"], width=2)
                if name == "auto":
                    c.create_oval(x, y, x + _D, y + _D, fill="#FFFFFF", outline="#999999")
                    c.create_text(x + _D / 2, y + _D / 2, text="A", font=("Segoe UI", 7, "bold"), fill="#555555")
                else:
                    c.create_oval(x, y, x + _D, y + _D, fill=_hex(overlay.PALETTE[name]), outline="#999999")
