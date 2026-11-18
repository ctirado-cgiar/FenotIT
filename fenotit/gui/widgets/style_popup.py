"""Colores de dibujo: máscara de segmentación, contornos y puntos (ventana pequeña
bajo el ícono de paleta). Los cambios se ven al momento."""
from __future__ import annotations

import tkinter as tk

# overlay tiene los colores y el dibujo de las marcas sobre la foto.
from fenotit.core.pipeline import overlay
# t devuelve cada texto de la ventana en el idioma elegido (fenotit/lang).
from fenotit.i18n import t
# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras); SPACE (márgenes).
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
# Slider es el deslizador fino de la app.
from fenotit.gui.widgets.slider import Slider

_SWATCHES = ("green", "magenta", "yellow", "cyan", "red", "orange", "blue", "white", "black")
_DOT = 14            # diámetro del punto de color


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
        box = tk.Frame(self, bg=bg, padx=SPACE["lg"], pady=SPACE["md"])
        box.pack(padx=1, pady=1)
        self._dots: dict[str, tk.Canvas] = {}
        row = 0
        for role in overlay.ROLES:
            tk.Label(box, text=t(f"style.{role}"), bg=bg, fg=colors["text"], font=FONTS["small"],
                     anchor="w").grid(row=row, column=0, sticky="w", padx=(0, SPACE["md"]), pady=SPACE["sm"])
            names = (() if role == "mask" else ("auto",)) + _SWATCHES
            canvas = tk.Canvas(box, bg=bg, highlightthickness=0, width=len(names) * (_DOT + 6), height=_DOT + 6,
                          cursor="hand2")
            canvas.grid(row=row, column=1, sticky="w")
            canvas.bind("<Button-1>", lambda event, role_key=role, role_names=names: self._click(event, role_key, role_names))
            self._dots[role] = (canvas, names)
            row += 1
            if role == "mask":
                row = self._slider(box, row, "style.opacity", 0.1, 0.9, "mask_alpha", "{:.0%}")
            if role == "outline":
                row = self._slider(box, row, "style.width", 0.5, 4.0, "width", "{:.1f} px")
        tk.Label(box, text=t("style.hint"), bg=bg, fg=colors["text_muted"], font=FONTS["tiny"],
                 justify="left", wraplength=240).grid(row=row, column=0, columnspan=2, sticky="w", pady=(SPACE["md"], 0))
        self._paint()
        self.attributes("-topmost", True)
        self.bind("<Escape>", lambda event: self.destroy())
        self._place_under(anchor)
        # Se cierra con Esc, con el ícono o con un clic en la ventana principal (no con la
        # pérdida de foco: en Windows una ventana sin marco no siempre recibe el foco).

    def _place_under(self, anchor):
        self.update_idletasks()
        width = self.winfo_reqwidth()
        top = anchor.winfo_toplevel()
        left, right = top.winfo_rootx(), top.winfo_rootx() + top.winfo_width()
        x = anchor.winfo_rootx() + anchor.winfo_width() // 2 - width // 2
        x = max(left + 4, min(x, right - width - 4))
        self.wm_geometry(f"+{x}+{anchor.winfo_rooty() + anchor.winfo_height() + 2}")

    def _slider(self, box, row, key, low, high, field, fmt):
        bg = self._c["bg_card"]
        tk.Label(box, text=t(key), bg=bg, fg=self._c["text_muted"], font=FONTS["tiny"],
                 anchor="w").grid(row=row, column=0, sticky="w", padx=(SPACE["lg"], SPACE["md"]))
        frame = tk.Frame(box, bg=bg)
        frame.grid(row=row, column=1, sticky="we")
        value = tk.Label(frame, text=fmt.format(self._style[field]), bg=bg, fg=self._c["text"],
                         font=FONTS["tiny"], width=6, anchor="e")
        var = tk.DoubleVar(value=float(self._style[field]))

        def changed(_value=None):
            rounded = round(var.get(), 2)
            value.config(text=fmt.format(rounded))
            self._style[field] = rounded
            self._on_change(dict(self._style))
        Slider(frame, var, low, high, command=changed, length=150, bg=bg).pack(side=tk.LEFT)
        value.pack(side=tk.LEFT)
        return row + 1

    def _click(self, event, role, names):
        i = int(event.x // (_DOT + 6))
        if 0 <= i < len(names):
            self._style[role] = names[i]
            self._paint()
            self._on_change(dict(self._style))

    def _paint(self):
        for role, (canvas, names) in self._dots.items():
            canvas.delete("all")
            for i, name in enumerate(names):
                x, y = i * (_DOT + 6) + 3, 3
                sel = self._style[role] == name
                if sel:
                    canvas.create_oval(x - 2, y - 2, x + _DOT + 2, y + _DOT + 2, outline=self._c["accent"], width=2)
                if name == "auto":
                    canvas.create_oval(x, y, x + _DOT, y + _DOT, fill=COLORS["white"], outline=COLORS["text_faint"])
                    canvas.create_text(x + _DOT / 2, y + _DOT / 2, text="A", font=FONTS["tiny_bold"], fill=COLORS["text_muted"])
                else:
                    canvas.create_oval(x, y, x + _DOT, y + _DOT, fill=_hex(overlay.PALETTE[name]), outline=COLORS["text_faint"])
