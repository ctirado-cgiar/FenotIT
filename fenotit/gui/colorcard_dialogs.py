"""Ventanas pequeñas de la tarjeta de color: marcar el área blanca/gris, escribir los
colores de una tarjeta propia y ver la tarjeta detectada (antes y después)."""
from __future__ import annotations

import re
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from fenotit.gui.theme import COLORS, FONTS
from fenotit.gui.zoom_controller import ZoomableCanvas
from fenotit.i18n import t


def _window(parent, title: str) -> tk.Toplevel:
    w = tk.Toplevel(parent)
    w.title(title)
    w.configure(bg=COLORS["bg_card"])
    w.transient(parent)
    tk.Frame(w, bg=COLORS["accent"], height=4).pack(fill=tk.X)
    return w


def _buttons(win, ok, ok_text=None):
    row = tk.Frame(win, bg=COLORS["bg_card"])
    row.pack(fill=tk.X, padx=16, pady=10, side=tk.BOTTOM)
    tk.Button(row, text=t("common.cancel"), command=win.destroy, bg=COLORS["btn_bg"], fg=COLORS["accent"],
              relief="flat", font=FONTS["body"], cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=(6, 0))
    tk.Button(row, text=ok_text or t("common.ok"), command=ok, bg=COLORS["accent"], fg="#FFFFFF",
              relief="flat", font=("Segoe UI", 9, "bold"), cursor="hand2", padx=14).pack(side=tk.RIGHT)
    return row


def _center(win, parent):
    win.update_idletasks()
    w, h = win.winfo_reqwidth(), win.winfo_reqheight()
    win.geometry(f"+{parent.winfo_rootx() + max(0, (parent.winfo_width() - w) // 2)}"
                 f"+{parent.winfo_rooty() + max(0, (parent.winfo_height() - h) // 4)}")


def pick_region(parent, image: np.ndarray, region=None):
    """Dos clics = esquinas del área blanca o gris. Devuelve [x0, y0, x1, y1] (0-1) o None."""
    win = _window(parent, t("card.white_title"))
    h, w = image.shape[:2]
    pts = [] if not region else [(region[0] * w, region[1] * h), (region[2] * w, region[3] * h)]
    out = {}
    tk.Label(win, text=t("card.white_hint"), bg=COLORS["bg_card"], fg=COLORS["text_muted"],
             font=FONTS["small"], anchor="w").pack(fill=tk.X, padx=16, pady=(8, 4))

    def draw(c):
        if not pts:
            return
        a = c.to_screen(*pts[0])
        b = c.to_screen(*pts[-1])
        c.create_rectangle(*a, *b, outline="#FFFFFF", width=4)
        c.create_rectangle(*a, *b, outline=COLORS["accent"], width=2)

    def click(x, y):
        if len(pts) >= 2:
            pts.clear()
        pts.append((x, y))
        canvas._redraw()

    canvas = ZoomableCanvas(win, width=760, height=520, bg=COLORS["bg_panel"], highlightthickness=0,
                            cursor="crosshair", on_click=click, on_overlay=draw)

    def ok():
        if len(pts) == 2 and abs(pts[0][0] - pts[1][0]) > 2 and abs(pts[0][1] - pts[1][1]) > 2:
            (x0, y0), (x1, y1) = pts
            out["r"] = [min(x0, x1) / w, min(y0, y1) / h, max(x0, x1) / w, max(y0, y1) / h]
            win.destroy()
    _buttons(win, ok)
    canvas.pack(fill=tk.BOTH, expand=True, padx=16)
    canvas.set_image(image)
    _center(win, parent)
    win.grab_set()
    win.wait_window()
    return out.get("r")


_NUM = re.compile(r"#?[0-9a-fA-F]{6}|\d+(?:[.,]\d+)?")


def parse_values(text: str, n: int) -> list | None:
    """Una línea por cuadrito: «R G B», «R,G,B», «R;G;B» o «#RRGGBB»."""
    vals = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        hexm = re.search(r"#([0-9a-fA-F]{6})", line)
        if hexm:
            hx = hexm.group(1)
            vals.append([int(hx[i:i + 2], 16) for i in (0, 2, 4)])
            continue
        nums = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", line)]
        if len(nums) >= 3:
            vals.append(nums[-3:])
    if len(vals) != n or any(not 0 <= v <= 255 for c in vals for v in c):
        return None
    return vals


def edit_values(parent, rows: int, cols: int, values=None):
    """Colores de referencia de una tarjeta propia (de la marca), fila por fila."""
    n = rows * cols
    win = _window(parent, t("card.values_title"))
    out = {}
    tk.Label(win, text=t("card.values_hint", n=n, rows=rows, cols=cols), bg=COLORS["bg_card"],
             fg=COLORS["text_muted"], font=FONTS["small"], justify="left", wraplength=460,
             anchor="w").pack(fill=tk.X, padx=16, pady=(8, 4))
    body = tk.Frame(win, bg=COLORS["bg_card"])
    body.pack(fill=tk.BOTH, expand=True, padx=16)
    txt = tk.Text(body, width=22, height=min(max(n, 6), 24), font=FONTS["mono"], relief="solid", bd=1)
    txt.pack(side=tk.LEFT, fill=tk.Y)
    if values:
        txt.insert("1.0", "\n".join(" ".join(f"{v:g}" for v in c) for c in values))
    sw = tk.Canvas(body, width=cols * 34 + 4, height=rows * 34 + 4, bg=COLORS["bg_card"], highlightthickness=0)
    sw.pack(side=tk.LEFT, padx=(12, 0), anchor="n")
    msg = tk.StringVar()
    tk.Label(win, textvariable=msg, bg=COLORS["bg_card"], fg=COLORS["warning"], font=FONTS["small"],
             anchor="w").pack(fill=tk.X, padx=16, pady=(6, 0))

    def refresh(_=None):
        sw.delete("all")
        vals = parse_values(txt.get("1.0", "end"), n)
        lines = [ln for ln in txt.get("1.0", "end").splitlines() if ln.strip()]
        msg.set("" if vals else t("card.values_count", got=len(lines), n=n))
        for k, c in enumerate(vals or []):
            r, q = divmod(k, cols)
            sw.create_rectangle(4 + q * 34, 4 + r * 34, 34 + q * 34, 34 + r * 34, outline="#888888",
                                fill="#%02x%02x%02x" % tuple(int(round(v)) for v in c))
    txt.bind("<KeyRelease>", refresh)
    txt.bind("<<Paste>>", lambda e: win.after(10, refresh))
    refresh()

    def ok():
        vals = parse_values(txt.get("1.0", "end"), n)
        if vals:
            out["v"] = vals
            win.destroy()
    _buttons(win, ok)
    _center(win, parent)
    win.grab_set()
    win.wait_window()
    return out.get("v")


def show_detection(parent, crop: np.ndarray | None, before: np.ndarray, after: np.ndarray | None, text: str):
    """La tarjeta encontrada (cuadritos numerados) y la foto antes / después."""
    win = _window(parent, t("card.preview_title"))
    row = tk.Frame(win, bg=COLORS["bg_card"])
    row.pack(padx=16, pady=(10, 4))
    win._imgs = []

    def put(img, caption, side):
        f = tk.Frame(row, bg=COLORS["bg_card"])
        f.pack(side=tk.LEFT, padx=6, anchor="n")
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        im.thumbnail((side, side))
        ph = ImageTk.PhotoImage(im)
        win._imgs.append(ph)
        tk.Label(f, image=ph, bg=COLORS["bg_card"]).pack()
        tk.Label(f, text=caption, bg=COLORS["bg_card"], fg=COLORS["text_muted"], font=FONTS["small"]).pack()
    if crop is not None:
        put(crop, t("card.found_chips"), 300)
    put(before, t("card.before"), 300)
    if after is not None:
        put(after, t("card.after"), 300)
    tk.Label(win, text=text, bg=COLORS["bg_card"], fg=COLORS["text"], font=FONTS["body"], justify="left",
             wraplength=900, anchor="w").pack(fill=tk.X, padx=22, pady=(4, 0))
    r = tk.Frame(win, bg=COLORS["bg_card"])
    r.pack(fill=tk.X, padx=16, pady=10)
    tk.Button(r, text=t("common.close"), command=win.destroy, bg=COLORS["accent"], fg="#FFFFFF", relief="flat",
              font=("Segoe UI", 9, "bold"), cursor="hand2", padx=14).pack(side=tk.RIGHT)
    win.bind("<Escape>", lambda e: win.destroy())
    _center(win, parent)
