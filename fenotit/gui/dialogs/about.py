"""Ventana "Acerca de": nombre, lema, descripción corta, desarrollo, colaboración y
datos del sistema (con botón para copiarlos al reportar un error)."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

from fenotit import APP_NAME
from fenotit.gui.style.decor import BLUE, pixel_dissolve
from fenotit.i18n import t
from fenotit.gui.style.components import button, divider, section as heading
from fenotit.gui.style.theme import COLORS, FONTS, SPACE, font as theme_font

_W = 460
_TEXT, _MUTED, _SOFT = COLORS["text"], COLORS["text_muted"], COLORS["text_faint"]
AUTHOR = "Cristian Tirado-Murcia"
EMAIL = "c.tirado@cgiar.org"


def show(root: tk.Tk, logo: Path, system_info: str):
    win = tk.Toplevel(root)
    win.title(t("menu.about"))
    win.configure(bg=COLORS["bg_card"])
    win.resizable(False, False)
    win.transient(root)
    try:
        win.iconbitmap(str(logo))
    except tk.TclError:
        pass

    tk.Frame(win, bg=BLUE, height=4).pack(fill=tk.X)
    head = tk.Canvas(win, bg=COLORS["bg_card"], width=_W, height=118, highlightthickness=0)
    head.pack(fill=tk.X)
    pixel_dissolve(head, _W, 118, 16,
                   lambda x, y: max(0.0, (x - 0.62) / 0.38 * 1.1 - y * 0.9) if x > 0.62 else 0.0, seed=4, gap=2)
    try:
        img = Image.open(logo).convert("RGBA").resize((64, 64), Image.LANCZOS)
        win._logo = ImageTk.PhotoImage(img)
        head.create_image(28, 28, image=win._logo, anchor="nw")
    except OSError:
        pass
    head.create_text(108, 34, text=APP_NAME, anchor="nw", fill=BLUE, font=FONTS["display"])
    head.create_text(110, 74, text=t("start.subtitle"), anchor="nw", fill=COLORS["text_muted"], font=FONTS["motto"])

    body = tk.Frame(win, bg=COLORS["bg_card"])
    body.pack(fill=tk.BOTH, expand=True, padx=SPACE["xxl"], pady=(0, SPACE["xl"]))

    def label(text, size=9, color=_TEXT, style="", pady=(0, 0), wrap=_W - 56):
        font = theme_font(size, style)
        tk.Label(body, text=text, bg=COLORS["bg_card"], fg=color, font=font, justify="left", anchor="w",
                 wraplength=wrap).pack(fill=tk.X, pady=pady)

    def section(title):
        heading(body, title, bg=COLORS["bg_card"]).pack(fill=tk.X, pady=(SPACE["xl"], SPACE["sm"]))

    label(t("about.description"), 9, _MUTED, pady=(SPACE["xs"], 0))

    section(t("about.development"))
    label(AUTHOR, 10, _TEXT, "bold")
    label(t("about.role"), 9, _MUTED)
    label(EMAIL, 9, BLUE)

    section(t("about.collaboration"))
    label(t("about.team"), 9, _TEXT, "bold")
    label("Alliance of Bioversity International & CIAT, Palmira, Colombia", 9, _MUTED)
    label(t("about.team_note"), 8, _SOFT, "italic", pady=(SPACE["sm"], 0))

    divider(body, "divider").pack(fill=tk.X, pady=(SPACE["xl"], SPACE["md"]))
    label("© 2026 Alliance of Bioversity International & CIAT. " + t("about.license"), 8, _SOFT)
    label(system_info, 7, _SOFT, pady=(SPACE["sm"], 0))

    buttons = tk.Frame(body, bg=COLORS["bg_card"])
    buttons.pack(fill=tk.X, pady=(SPACE["lg"], 0))

    def copy(btn):
        win.clipboard_clear()
        win.clipboard_append(f"{APP_NAME}\n{system_info}")
        btn.config(text=t("about.copied"))
        win.after(1500, lambda: btn.winfo_exists() and btn.config(text=t("about.copy_info")))
    cp = button(buttons, text=t("about.copy_info"), kind="link", bg=COLORS["bg_card"], fg=_MUTED, padx=0)
    cp.config(command=lambda: copy(cp))
    cp.pack(side=tk.LEFT)
    button(buttons, text=t("common.close"), command=win.destroy, kind="primary", padx=SPACE["xl"], pady=SPACE["sm"]).pack(side=tk.RIGHT)
    win.bind("<Escape>", lambda e: win.destroy())

    win.update_idletasks()
    w, h = max(_W, win.winfo_reqwidth()), win.winfo_reqheight()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"{w}x{min(h, sh - 80)}+{(sw - w) // 2}+{(sh - h) // 2}")
    return win
