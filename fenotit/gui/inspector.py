"""Inspector de objetos (panel izquierdo, bajo CAPAS): el objeto elegido acercado, los
datos de la capa que se está viendo, sus colores y Excluir / Incluir."""
from __future__ import annotations

import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from fenotit.gui.theme import COLORS, FONTS
from fenotit.i18n import t

CROP = (220, 150)
MAX_ROWS = 10


class Inspector(tk.Frame):

    def __init__(self, parent, on_close, on_toggle):
        bg = COLORS["bg_panel"]
        super().__init__(parent, bg=bg)
        self._on_close, self._on_toggle = on_close, on_toggle
        head = tk.Frame(self, bg=bg)
        head.pack(fill=tk.X)
        self._title = tk.Label(head, text="", bg=bg, fg=COLORS["accent"], font=("Segoe UI", 8, "bold"), pady=5)
        self._title.pack(side=tk.LEFT, padx=(10, 6))
        close = tk.Label(head, text="×", bg=bg, fg=COLORS["text_muted"], font=("Segoe UI", 11), cursor="hand2")
        close.pack(side=tk.RIGHT, padx=8)
        close.bind("<Button-1>", lambda e: self._on_close())
        tk.Frame(self, bg=COLORS["border"], height=1).pack(fill=tk.X)
        self._pic = tk.Label(self, bg=COLORS["bg_card"], bd=0, highlightthickness=1,
                             highlightbackground=COLORS["border"])
        self._pic.pack(padx=8, pady=(6, 4))
        self._state = tk.Label(self, text="", bg=bg, fg=COLORS["warning"], font=FONTS["small"],
                               justify="left", anchor="w", wraplength=200)
        self._rows = tk.Frame(self, bg=bg)
        self._rows.pack(fill=tk.X, padx=10)
        self._colors = tk.Frame(self, bg=bg)
        self._colors.pack(fill=tk.X, padx=10, pady=(4, 0))
        self._btn = tk.Button(self, text="", command=self._on_toggle, bg=COLORS["btn_bg"], fg=COLORS["accent"],
                              relief="flat", font=FONTS["small"], cursor="hand2", pady=2)
        self._btn.pack(fill=tk.X, padx=10, pady=(6, 8))
        self._photo = None

    def show(self, oid: int, crop: np.ndarray | None, metrics: list[tuple[str, str]],
             colors: list[tuple[str, float]], excluded: bool):
        self._title.config(text=t("insp.title", n=oid))
        if crop is not None:
            im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            self._photo = ImageTk.PhotoImage(im)
            self._pic.config(image=self._photo, width=im.width, height=im.height)
            self._pic.pack(padx=8, pady=(6, 4), before=self._rows)
        else:
            self._pic.pack_forget()
        if excluded:
            self._state.config(text=t("insp.excluded"))
            self._state.pack(fill=tk.X, padx=10, before=self._rows)
        else:
            self._state.pack_forget()
        for w in self._rows.winfo_children():
            w.destroy()
        for i, (k, v) in enumerate(metrics[:MAX_ROWS]):
            tk.Label(self._rows, text=k, bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONTS["small"],
                     anchor="w").grid(row=i, column=0, sticky="w")
            tk.Label(self._rows, text=v, bg=COLORS["bg_panel"], fg=COLORS["text"], font=FONTS["small"],
                     anchor="e").grid(row=i, column=1, sticky="e", padx=(8, 0))
        self._rows.columnconfigure(1, weight=1)
        for w in self._colors.winfo_children():
            w.destroy()
        for hexc, pct in colors[:8]:
            cell = tk.Frame(self._colors, bg=COLORS["bg_panel"])
            cell.pack(side=tk.LEFT, padx=(0, 6))
            tk.Frame(cell, bg=hexc, width=18, height=14, highlightthickness=1,
                     highlightbackground=COLORS["border"]).pack()
            tk.Label(cell, text=f"{pct:.0f}%", bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                     font=("Segoe UI", 7)).pack()
        self._btn.config(text=t("insp.include") if excluded else t("insp.exclude"))


def crop_around(img: np.ndarray, contour: np.ndarray, size=CROP):
    """Recorte del objeto con margen, escalado para caber en `size`: (recorte, origen, zoom)."""
    x, y, w, h = cv2.boundingRect(contour.reshape(-1, 1, 2))
    m = int(max(w, h) * 0.35) + 4
    H, W = img.shape[:2]
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(W, x + w + m), min(H, y + h + m)
    part = img[y0:y1, x0:x1]
    if part.size == 0:
        return None, (0, 0), 1.0
    k = min(size[0] / part.shape[1], size[1] / part.shape[0])
    out = cv2.resize(part, (max(1, round(part.shape[1] * k)), max(1, round(part.shape[0] * k))),
                     interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(out), (x0, y0), k
