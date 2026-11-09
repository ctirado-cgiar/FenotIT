"""Controles comunes de la interfaz con el aspecto del tema: botones, títulos de sección y
separadores. Las pantallas los crean con estas funciones; su color, letra y márgenes salen de
las secciones [buttons], [colors], [fonts] y [sizes] del tema."""
from __future__ import annotations

import tkinter as tk

from fenotit.gui.style.theme import BUTTONS, COLORS, FONTS, SIZES

KINDS = tuple(BUTTONS)            # primary, secondary, quiet, link
SIZE_NAMES = ("normal", "small")


def button(parent: tk.Misc, text: str = "", command=None, kind: str = "secondary", size: str = "normal",
           bg: str | None = None, **kw) -> tk.Button:
    """Botón del tema. kind: primary (acción principal), secondary, quiet (cancelar, cerrar) o
    link (sin fondo, sobre el color `bg` de su contenedor). size: normal o small."""
    style = BUTTONS[kind]
    back = COLORS[style["bg"]] if style["bg"] else (bg or COLORS["bg_panel"])
    small = size == "small"
    options = dict(bg=back, fg=COLORS[style["fg"]], activebackground=COLORS[style["pressed"]],
                   activeforeground=COLORS[style.get("pressed_fg", style["fg"])],
                   font=FONTS[style["font_small" if small else "font"]], relief="flat", bd=0, cursor="hand2",
                   padx=SIZES["button_small_pad_x" if small else "button_pad_x"],
                   pady=SIZES["button_small_pad_y" if small else "button_pad_y"],
                   highlightthickness=0)
    options.update(kw)
    b = tk.Button(parent, text=text, command=command, **options)
    hover = style.get("hover")
    if hover:
        b.bind("<Enter>", lambda e: b.cget("state") != "disabled" and b.config(bg=COLORS[hover]), add="+")
        b.bind("<Leave>", lambda e: b.config(bg=back), add="+")
    return b


def section(parent: tk.Misc, text: str, bg: str | None = None, line: str = "after",
            help_title: str | None = None, help_text: str | None = None) -> tk.Frame:
    """Título de una sección. line="after": raya a continuación del título (diálogos);
    line="below": raya debajo, de lado a lado (paneles laterales). Devuelve el contenedor
    sin empacar."""
    back = bg or COLORS["bg_card"]
    row = tk.Frame(parent, bg=back)
    title = tk.Label(row, text=text, bg=back, fg=COLORS["accent"], font=FONTS["section"])
    if line == "below":
        title.pack(side=tk.LEFT, padx=(SIZES["panel_pad"], SIZES["panel_pad"]), pady=SIZES["section_pad_y"])
        divider(row).pack(fill=tk.X, side=tk.BOTTOM)
        return row
    title.pack(side=tk.LEFT)
    if help_text:
        from fenotit.gui.widgets.help import HelpIcon
        HelpIcon(row, help_title or text, help_text, bg=back).pack(side=tk.LEFT, padx=(6, 0))
    divider(row).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), pady=(2, 0))
    return row


def divider(parent: tk.Misc, color: str = "border", vertical: bool = False) -> tk.Frame:
    """Raya fina de separación (sin empacar)."""
    return tk.Frame(parent, bg=COLORS[color], **({"width": 1} if vertical else {"height": 1}))
