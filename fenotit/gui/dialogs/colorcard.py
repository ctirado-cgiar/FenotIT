"""Ventanas pequeñas de la tarjeta de color: marcar el área blanca/gris, escribir los
colores de una tarjeta propia y ver la tarjeta detectada (antes y después)."""
from __future__ import annotations

import re
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from fenotit.gui.style.components import button
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.gui.canvas.zoom import ZoomableCanvas
from fenotit.i18n import t


def _window(parent, title: str) -> tk.Toplevel:
    window = tk.Toplevel(parent)
    window.title(title)
    window.configure(bg=COLORS["bg_card"])
    window.transient(parent)
    tk.Frame(window, bg=COLORS["accent"], height=4).pack(fill=tk.X)
    return window


def _buttons(win, ok, ok_text=None):
    row = tk.Frame(win, bg=COLORS["bg_card"])
    row.pack(fill=tk.X, padx=SPACE["xl"], pady=SPACE["lg"], side=tk.BOTTOM)
    button(row, text=t("common.cancel"), command=win.destroy, kind="secondary").pack(side=tk.RIGHT, padx=(SPACE["md"], 0))
    button(row, text=ok_text or t("common.ok"), command=ok, kind="primary").pack(side=tk.RIGHT)
    return row


def _center(win, parent):
    win.update_idletasks()
    width, height = win.winfo_reqwidth(), win.winfo_reqheight()
    win.geometry(f"+{parent.winfo_rootx() + max(0, (parent.winfo_width() - width) // 2)}"
                 f"+{parent.winfo_rooty() + max(0, (parent.winfo_height() - height) // 4)}")


def pick_region(parent, image: np.ndarray, region=None):
    """Dos clics = esquinas del área blanca o gris. Devuelve [x0, y0, x1, y1] (0-1) o None."""
    win = _window(parent, t("card.white_title"))
    height, width = image.shape[:2]
    pts = [] if not region else [(region[0] * width, region[1] * height), (region[2] * width, region[3] * height)]
    out = {}
    tk.Label(win, text=t("card.white_hint"), bg=COLORS["bg_card"], fg=COLORS["text_muted"],
             font=FONTS["small"], anchor="w").pack(fill=tk.X, padx=SPACE["xl"], pady=(SPACE["md"], SPACE["sm"]))

    def draw(canvas):
        if not pts:
            return
        start = canvas.to_screen(*pts[0])
        end = canvas.to_screen(*pts[-1])
        canvas.create_rectangle(*start, *end, outline=COLORS["white"], width=4)
        canvas.create_rectangle(*start, *end, outline=COLORS["accent"], width=2)

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
            out["r"] = [min(x0, x1) / width, min(y0, y1) / height, max(x0, x1) / width, max(y0, y1) / height]
            win.destroy()
    _buttons(win, ok)
    canvas.pack(fill=tk.BOTH, expand=True, padx=SPACE["xl"])
    canvas.set_image(image)
    _center(win, parent)
    win.grab_set()
    win.wait_window()
    return out.get("r")


_NUM = re.compile(r"#?[0-9a-fA-F]{6}|\d+(?:[.,]\d+)?")


def parse_values(text: str, count: int) -> list | None:
    """Una línea por cuadrito: «R G B», «R,G,B», «R;G;B» o «#RRGGBB»."""
    vals = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        hexm = re.search(r"#([0-9a-fA-F]{6})", line)
        if hexm:
            hex_digits = hexm.group(1)
            vals.append([int(hex_digits[i:i + 2], 16) for i in (0, 2, 4)])
            continue
        nums = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", line)]
        if len(nums) >= 3:
            vals.append(nums[-3:])
    if len(vals) != count or any(not 0 <= value <= 255 for color in vals for value in color):
        return None
    return vals


def edit_values(parent, rows: int, cols: int, values=None):
    """Colores de referencia de una tarjeta propia (de la marca), fila por fila."""
    count = rows * cols
    win = _window(parent, t("card.values_title"))
    out = {}
    tk.Label(win, text=t("card.values_hint", n=count, rows=rows, cols=cols), bg=COLORS["bg_card"],
             fg=COLORS["text_muted"], font=FONTS["small"], justify="left", wraplength=460,
             anchor="w").pack(fill=tk.X, padx=SPACE["xl"], pady=(SPACE["md"], SPACE["sm"]))
    body = tk.Frame(win, bg=COLORS["bg_card"])
    body.pack(fill=tk.BOTH, expand=True, padx=SPACE["xl"])
    txt = tk.Text(body, width=22, height=min(max(count, 6), 24), font=FONTS["mono"], relief="solid", bd=1)
    txt.pack(side=tk.LEFT, fill=tk.Y)
    if values:
        txt.insert("1.0", "\n".join(" ".join(f"{value:g}" for value in color) for color in values))
    canvas = tk.Canvas(body, width=cols * 34 + 4, height=rows * 34 + 4, bg=COLORS["bg_card"], highlightthickness=0)
    canvas.pack(side=tk.LEFT, padx=(SPACE["lg"], 0), anchor="n")
    msg = tk.StringVar()
    tk.Label(win, textvariable=msg, bg=COLORS["bg_card"], fg=COLORS["warning"], font=FONTS["small"],
             anchor="w").pack(fill=tk.X, padx=SPACE["xl"], pady=(SPACE["md"], 0))

    def refresh(_=None):
        canvas.delete("all")
        vals = parse_values(txt.get("1.0", "end"), count)
        lines = [line for line in txt.get("1.0", "end").splitlines() if line.strip()]
        msg.set("" if vals else t("card.values_count", got=len(lines), n=count))
        for index, color in enumerate(vals or []):
            row, column = divmod(index, cols)
            canvas.create_rectangle(4 + column * 34, 4 + row * 34, 34 + column * 34, 34 + row * 34, outline=COLORS["text_faint"],
                                fill="#%02x%02x%02x" % tuple(int(round(value)) for value in color))
    txt.bind("<KeyRelease>", refresh)
    txt.bind("<<Paste>>", lambda event: win.after(10, refresh))
    refresh()

    def ok():
        vals = parse_values(txt.get("1.0", "end"), count)
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
    row.pack(padx=SPACE["xl"], pady=(SPACE["lg"], SPACE["sm"]))
    win._imgs = []

    def put(img, caption, side):
        frame = tk.Frame(row, bg=COLORS["bg_card"])
        frame.pack(side=tk.LEFT, padx=SPACE["md"], anchor="n")
        picture = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        picture.thumbnail((side, side))
        image = ImageTk.PhotoImage(picture)
        win._imgs.append(image)
        tk.Label(frame, image=image, bg=COLORS["bg_card"]).pack()
        tk.Label(frame, text=caption, bg=COLORS["bg_card"], fg=COLORS["text_muted"], font=FONTS["small"]).pack()
    if crop is not None:
        put(crop, t("card.found_chips"), 300)
    put(before, t("card.before"), 300)
    if after is not None:
        put(after, t("card.after"), 300)
    tk.Label(win, text=text, bg=COLORS["bg_card"], fg=COLORS["text"], font=FONTS["body"], justify="left",
             wraplength=900, anchor="w").pack(fill=tk.X, padx=SPACE["xxl"], pady=(SPACE["sm"], 0))
    frame = tk.Frame(win, bg=COLORS["bg_card"])
    frame.pack(fill=tk.X, padx=SPACE["xl"], pady=SPACE["lg"])
    button(frame, text=t("common.close"), command=win.destroy, kind="primary").pack(side=tk.RIGHT)
    win.bind("<Escape>", lambda event: win.destroy())
    _center(win, parent)
