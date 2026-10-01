"""Colores de dibujo: máscara de segmentación, contornos y puntos (ventana pequeña
que se abre desde la barra superior). Los cambios se ven al momento."""
from __future__ import annotations

import tkinter as tk

from fenotit.core.pipeline import overlay
from fenotit.i18n import t

_SWATCHES = ("green", "magenta", "yellow", "cyan", "red", "orange", "blue", "white", "black")


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
        box = tk.Frame(self, bg=colors["bg_card"], padx=10, pady=8)
        box.pack(padx=1, pady=1)
        self._cells: dict[str, dict[str, tk.Widget]] = {}
        for role in overlay.ROLES:
            tk.Label(box, text=t(f"style.{role}"), bg=colors["bg_card"], fg=colors["text"],
                     font=("Segoe UI", 8, "bold"), anchor="w").pack(fill=tk.X, pady=(4, 1))
            row = tk.Frame(box, bg=colors["bg_card"])
            row.pack(fill=tk.X)
            cells = {}
            if role != "mask":
                b = tk.Label(row, text="Auto", bg=colors["bg_panel"], fg=colors["text"], font=("Segoe UI", 7),
                             width=4, cursor="hand2", highlightthickness=2)
                b.pack(side=tk.LEFT, padx=1)
                b.bind("<Button-1>", lambda e, r=role: self._pick(r, "auto"))
                cells["auto"] = b
            for name in _SWATCHES:
                sw = tk.Frame(row, bg=_hex(overlay.PALETTE[name]), width=18, height=18, cursor="hand2",
                              highlightthickness=2, bd=1, relief="solid")
                sw.pack(side=tk.LEFT, padx=1)
                sw.bind("<Button-1>", lambda e, r=role, n=name: self._pick(r, n))
                cells[name] = sw
            self._cells[role] = cells
            if role == "outline":
                self._width = tk.IntVar(value=int(self._style["width"]))
                tk.Scale(box, from_=1, to=6, resolution=1, orient=tk.HORIZONTAL, showvalue=True,
                         variable=self._width, command=lambda _v: self._set_width(), length=200,
                         bg=colors["bg_card"], highlightthickness=0, troughcolor=colors["bg_panel"],
                         label=t("style.width"), font=("Segoe UI", 7)).pack(fill=tk.X)
            if role == "mask":
                self._alpha = tk.DoubleVar(value=self._style["mask_alpha"])
                tk.Scale(box, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL, showvalue=False,
                         variable=self._alpha, command=lambda _v: self._set_alpha(), length=200,
                         bg=colors["bg_card"], highlightthickness=0, troughcolor=colors["bg_panel"],
                         label=t("style.opacity"), font=("Segoe UI", 7)).pack(fill=tk.X)
        tk.Label(box, text=t("style.hint"), bg=colors["bg_card"], fg=colors["text_muted"],
                 font=("Segoe UI", 7), justify="left", wraplength=220).pack(anchor="w", pady=(6, 0))
        self._paint()
        self.update_idletasks()
        x = anchor.winfo_rootx() + anchor.winfo_width() - self.winfo_reqwidth()
        y = anchor.winfo_rooty() + anchor.winfo_height() + 2
        self.wm_geometry(f"+{max(4, x)}+{y}")
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<FocusOut>", lambda e: self.after(150, self._close_if_outside))
        self.focus_force()

    def _close_if_outside(self):
        try:
            focus = self.focus_get()
        except (KeyError, tk.TclError):
            focus = None
        if focus is None or not str(focus).startswith(str(self)):
            self.destroy()

    def _paint(self):
        for role, cells in self._cells.items():
            for name, w in cells.items():
                sel = self._style[role] == name
                w.config(highlightbackground=self._c["accent"] if sel else self._c["bg_card"])

    def _pick(self, role: str, name: str):
        self._style[role] = name
        self._paint()
        self._on_change(dict(self._style))

    def _set_width(self):
        self._style["width"] = int(self._width.get())
        self._on_change(dict(self._style))

    def _set_alpha(self):
        self._style["mask_alpha"] = round(float(self._alpha.get()), 2)
        self._on_change(dict(self._style))
