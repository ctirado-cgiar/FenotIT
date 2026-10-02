"""Ventana "Acerca de": nombre, lema, descripción corta, desarrollo, colaboración y
datos del sistema (con botón para copiarlos al reportar un error)."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

from fenotit import APP_NAME
from fenotit.gui.decor import BLUE, pixel_dissolve
from fenotit.i18n import t

_W = 460
_TEXT, _MUTED, _SOFT = "#1F2A30", "#5F6B70", "#9AA3A7"
AUTHOR = "Cristian Tirado-Murcia"
EMAIL = "c.tirado@cgiar.org"


def show(root: tk.Tk, logo: Path, system_info: str):
    win = tk.Toplevel(root)
    win.title(t("menu.about"))
    win.configure(bg="#FFFFFF")
    win.resizable(False, False)
    win.transient(root)
    try:
        win.iconbitmap(str(logo))
    except tk.TclError:
        pass

    tk.Frame(win, bg=BLUE, height=4).pack(fill=tk.X)
    head = tk.Canvas(win, bg="#FFFFFF", width=_W, height=118, highlightthickness=0)
    head.pack(fill=tk.X)
    pixel_dissolve(head, _W, 118, 16,
                   lambda x, y: max(0.0, (x - 0.62) / 0.38 * 1.1 - y * 0.9) if x > 0.62 else 0.0, seed=4, gap=2)
    try:
        img = Image.open(logo).convert("RGBA").resize((64, 64), Image.LANCZOS)
        win._logo = ImageTk.PhotoImage(img)
        head.create_image(28, 28, image=win._logo, anchor="nw")
    except OSError:
        pass
    head.create_text(108, 34, text=APP_NAME, anchor="nw", fill=BLUE, font=("Segoe UI", 22, "bold"))
    head.create_text(110, 74, text=t("start.subtitle"), anchor="nw", fill="#4A4A4A", font=("Georgia", 11, "italic"))

    body = tk.Frame(win, bg="#FFFFFF")
    body.pack(fill=tk.BOTH, expand=True, padx=28, pady=(0, 18))

    def label(text, size=9, color=_TEXT, style="", pady=(0, 0), wrap=_W - 56):
        font = ("Segoe UI", size, style) if style else ("Segoe UI", size)
        tk.Label(body, text=text, bg="#FFFFFF", fg=color, font=font, justify="left", anchor="w",
                 wraplength=wrap).pack(fill=tk.X, pady=pady)

    def section(title):
        row = tk.Frame(body, bg="#FFFFFF")
        row.pack(fill=tk.X, pady=(16, 4))
        tk.Label(row, text=title, bg="#FFFFFF", fg=BLUE, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
        tk.Frame(row, bg="#E3E8EE", height=1).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), pady=(2, 0))

    label(t("about.description"), 9, _MUTED, pady=(2, 0))

    section(t("about.development"))
    label(AUTHOR, 10, _TEXT, "bold")
    label(t("about.role"), 9, _MUTED)
    label(EMAIL, 9, BLUE)

    section(t("about.collaboration"))
    label(t("about.team"), 9, _TEXT, "bold")
    label("Alliance of Bioversity International & CIAT · Palmira, Colombia", 9, _MUTED)
    label(t("about.team_note"), 8, _SOFT, "italic", pady=(4, 0))

    tk.Frame(body, bg="#E3E8EE", height=1).pack(fill=tk.X, pady=(18, 8))
    label("© 2026 Alliance of Bioversity International & CIAT  ·  " + t("about.license"), 8, _SOFT)
    label(system_info, 7, _SOFT, pady=(4, 0))

    buttons = tk.Frame(body, bg="#FFFFFF")
    buttons.pack(fill=tk.X, pady=(14, 0))

    def copy(btn):
        win.clipboard_clear()
        win.clipboard_append(f"{APP_NAME}\n{system_info}")
        btn.config(text=t("about.copied"))
        win.after(1500, lambda: btn.winfo_exists() and btn.config(text=t("about.copy_info")))
    cp = tk.Button(buttons, text=t("about.copy_info"), bg="#FFFFFF", fg=_MUTED, activebackground="#EEF3F9",
                   relief="flat", bd=0, font=("Segoe UI", 9), cursor="hand2")
    cp.config(command=lambda: copy(cp))
    cp.pack(side=tk.LEFT)
    tk.Button(buttons, text=t("common.close"), command=win.destroy, bg=BLUE, fg="#FFFFFF",
              activebackground="#1A5390", activeforeground="#FFFFFF", relief="flat", bd=0,
              font=("Segoe UI", 9, "bold"), padx=16, pady=4, cursor="hand2").pack(side=tk.RIGHT)
    win.bind("<Escape>", lambda e: win.destroy())

    win.update_idletasks()
    w, h = max(_W, win.winfo_reqwidth()), win.winfo_reqheight()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"{w}x{min(h, sh - 80)}+{(sw - w) // 2}+{(sh - h) // 2}")
    return win
