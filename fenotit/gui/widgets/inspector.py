"""Inspector de objetos (panel izquierdo, bajo CAPAS): el objeto elegido acercado y solo
los datos de la capa que se está viendo, más Destacar ★ y Excluir.

Por capa:
  Conteo       número, si se tocaba con otro y su estado
  Morfometría  cotas de largo y ancho con su valor y barra de escala; área, largo, ancho, perímetro
  Forma        su silueta sobre la forma media de la foto y cuánto se aparta de ella
  Color        barra de proporciones y cada color en el formato elegido, con %
  Distancias   líneas a sus vecinos con la distancia; vecino más cercano, promedio, n
"""
from __future__ import annotations

import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from fenotit.core.units import TO_MM, display
from fenotit.gui.style.components import button, divider
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.i18n import t

CROP = (220, 150)
WHITE, DARK = (255, 255, 255), (25, 25, 25)


class Inspector(tk.Frame):

    def __init__(self, parent, on_close, on_exclude, on_highlight):
        bg = COLORS["bg_panel"]
        super().__init__(parent, bg=bg)
        head = tk.Frame(self, bg=bg)
        head.pack(fill=tk.X)
        self._title = tk.Label(head, text="", bg=bg, fg=COLORS["accent"], font=FONTS["small_bold"], pady=SPACE["sm"])
        self._title.pack(side=tk.LEFT, padx=(SPACE["lg"], SPACE["md"]))
        close = tk.Label(head, text="×", bg=bg, fg=COLORS["text_muted"], font=FONTS["subtitle"], cursor="hand2")
        close.pack(side=tk.RIGHT, padx=SPACE["md"])
        close.bind("<Button-1>", lambda event: on_close())
        divider(self).pack(fill=tk.X)
        self._pic = tk.Label(self, bg=COLORS["bg_card"], bd=0, highlightthickness=1,
                             highlightbackground=COLORS["border"])
        self._pic.pack(padx=SPACE["md"], pady=(SPACE["md"], SPACE["sm"]))
        self._state = tk.Label(self, text="", bg=bg, fg=COLORS["warning"], font=FONTS["small"],
                               justify="left", anchor="w", wraplength=200)
        self._rows = tk.Frame(self, bg=bg)
        self._keys: list[tk.Label] = []
        self._rows.bind("<Configure>", lambda event: self._wrap_keys())
        self._rows.pack(fill=tk.X, padx=SPACE["lg"])
        btns = tk.Frame(self, bg=bg)
        btns.pack(fill=tk.X, padx=SPACE["lg"], pady=(SPACE["md"], SPACE["md"]))
        kwargs = dict(kind="secondary", size="small", pady=SPACE["xs"])
        self._star = button(btns, command=on_highlight, **kwargs)
        self._star.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, SPACE["sm"]))
        self._excl = button(btns, command=on_exclude, **kwargs)
        self._excl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._photo = None
        self._crop = None

    def show(self, oid: int, crop: np.ndarray | None, rows: list, excluded: bool, highlighted: bool):
        """rows: (etiqueta, valor) o ("color", hex, texto, %) o ("bar", [(hex, %)])."""
        self._title.config(text=t("insp.title", n=oid) + ("  ★" if highlighted else ""))
        self._crop = crop
        self._set_pic(crop)
        if excluded:
            self._state.config(text=t("insp.excluded"))
            self._state.pack(fill=tk.X, padx=SPACE["lg"], before=self._rows)
        else:
            self._state.pack_forget()
        self._fill(rows)
        self._star.config(text=t("insp.unhighlight") if highlighted else t("insp.highlight"),
                          state="disabled" if excluded else "normal")
        self._excl.config(text=t("insp.include") if excluded else t("insp.exclude"))

    def shrink(self, pixel_x: int) -> bool:
        """Achica el recorte para dejar sitio a CAPAS. False si ya no se puede."""
        if self._crop is None or not self._pic.winfo_ismapped():
            return False
        height = self._pic.winfo_reqheight() - pixel_x - 4
        if height < 50:
            self._pic.pack_forget()
            return False
        factor = height / self._crop.shape[0]
        self._set_pic(cv2.resize(self._crop, (max(1, int(self._crop.shape[1] * factor)), max(1, int(height))),
                                 interpolation=cv2.INTER_AREA), keep=True)
        return True

    def _set_pic(self, crop, keep: bool = False):
        if crop is None:
            self._pic.pack_forget()
            return
        picture = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(picture)
        self._pic.config(image=self._photo, width=picture.width, height=picture.height)
        if not keep:
            self._pic.pack(padx=SPACE["md"], pady=(SPACE["md"], SPACE["sm"]), before=self._rows)

    def _fill(self, rows):
        bg = COLORS["bg_panel"]
        for widget in self._rows.winfo_children():
            widget.destroy()
        keys = []
        for row, item in enumerate(rows):
            if item[0] == "bar":
                bar = tk.Canvas(self._rows, width=200, height=12, bg=bg, highlightthickness=0)
                bar.grid(row=row, column=0, columnspan=3, sticky="w", pady=(SPACE["xs"], SPACE["sm"]))
                x = 0.0
                for hexc, pct in item[1]:
                    widget = 200 * pct / 100
                    bar.create_rectangle(x, 0, x + widget, 12, fill=hexc, outline="")
                    x += widget
                bar.create_rectangle(0, 0, 199, 11, outline=COLORS["border"])
            elif item[0] == "color":
                _, hexc, text, pct = item
                tk.Frame(self._rows, bg=hexc, width=12, height=12, highlightthickness=1,
                         highlightbackground=COLORS["border"]).grid(row=row, column=0, sticky="w")
                tk.Label(self._rows, text=text, bg=bg, fg=COLORS["text"], font=FONTS["small"],
                         anchor="w").grid(row=row, column=1, sticky="w", padx=(SPACE["md"], 0))
                tk.Label(self._rows, text=f"{pct:.0f} %", bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                         anchor="e").grid(row=row, column=2, sticky="e")
            else:
                name, value = item
                key = tk.Label(self._rows, text=name, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                               anchor="w", justify="left")
                key.grid(row=row, column=0, columnspan=2, sticky="w")
                keys.append(key)
                tk.Label(self._rows, text=value, bg=bg, fg=COLORS["text"], font=FONTS["small"],
                         anchor="e").grid(row=row, column=2, sticky="ne", padx=(SPACE["md"], 0))
        self._rows.columnconfigure(1, weight=1)
        self._rows.columnconfigure(2, weight=0)
        self._keys = keys
        self.after_idle(self._wrap_keys)

    def _wrap_keys(self):
        """Los nombres de los datos pasan a otra línea si no caben junto al valor."""
        if not self._rows.winfo_exists():
            return
        width = self._rows.winfo_width()
        if width <= 1:
            return
        values = [widget for widget in self._rows.grid_slaves(column=2) if isinstance(widget, tk.Label)]
        widest = max((widget.winfo_reqwidth() for widget in values), default=0)
        for key in self._keys:
            if key.winfo_exists():
                key.config(wraplength=max(60, width - widest - SPACE["md"] - 4))


# ── Dibujo del recorte por capa (sin tkinter) ────────────────────────────────

def crop_around(img: np.ndarray, contour: np.ndarray, size=CROP, extra=None, margin: float = 0.25):
    """Recorte del objeto (y de `extra`, puntos que también deben verse) con margen,
    escalado para caber en `size`: (recorte, origen, zoom)."""
    pts = contour if extra is None or not len(extra) else np.vstack([contour, np.asarray(extra, float)])
    x, y, width, height = cv2.boundingRect(np.asarray(pts, np.float32).reshape(-1, 1, 2))
    margin_px = int(max(width, height) * margin) + 6
    box_height, box_width = img.shape[:2]
    x0, y0 = max(0, x - margin_px), max(0, y - margin_px)
    x1, y1 = min(box_width, x + width + margin_px), min(box_height, y + height + margin_px)
    part = img[y0:y1, x0:x1]
    if part.size == 0:
        return None, (0, 0), 1.0
    factor = min(size[0] / part.shape[1], size[1] / part.shape[0])
    out = cv2.resize(part, (max(1, round(part.shape[1] * factor)), max(1, round(part.shape[0] * factor))),
                     interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(out), (x0, y0), factor


def text(img, label, x, y, scale=0.38, dark=False):
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(min(max(2, x - text_width / 2), img.shape[1] - text_width - 2))
    y = int(min(max(text_height + 2, y + text_height / 2), img.shape[0] - 3))
    halo, ink = (WHITE, DARK) if dark else (DARK, WHITE)      # dark = texto oscuro sobre fondo claro
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, halo, 3, cv2.LINE_AA)
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1, cv2.LINE_AA)


def fmt(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.1f}" if abs(value) >= 10 else f"{value:.2f}"


def _box(img, label, x, y, scale):
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    return int(x - text_width / 2) - 2, int(y - text_height / 2) - 3, int(x + text_width / 2) + 2, int(y + text_height / 2) + 3


def _free(img, box, mask, taken) -> bool:
    """¿El rótulo cabe en el recorte sin tapar el objeto ni otro rótulo?"""
    x0, y0, x1, y1 = box
    height, width = img.shape[:2]
    if x0 < 1 or y0 < 1 or x1 > width - 1 or y1 > height - 1:
        return False
    if mask is not None and mask[y0:y1, x0:x1].mean() > 8:
        return False
    return not any(x0 < other[2] and other[0] < x1 and y0 < other[3] and other[1] < y1 for other in taken)


def label_at(img, label, candidates, mask, taken, scale=0.38, dark=False):
    """Escribe `s` en el primer lugar libre de `candidates` (si ninguno lo está, en el primero)."""
    spots = [np.asarray(spot, float) for spot in candidates]
    spot = next((candidate for candidate in spots if _free(img, _box(img, label, candidate[0], candidate[1], scale), mask, taken)), spots[0])
    taken.append(_box(img, label, spot[0], spot[1], scale))
    text(img, label, spot[0], spot[1], scale, dark)


def cota(img, start, end, label, mask=None, taken=None):
    """Cota: línea con topes en los extremos y el valor fuera del objeto, donde no tape
    nada (después de un extremo o a un lado)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    direction = end - start
    along = direction / (np.hypot(*direction) or 1)
    normal = np.array([-along[1], along[0]])
    for col, line_width in ((DARK, 3), (WHITE, 1)):
        cv2.line(img, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), col, line_width, cv2.LINE_AA)
        for end_point in (start, end):
            cv2.line(img, tuple(np.round(end_point - normal * 4).astype(int)), tuple(np.round(end_point + normal * 4).astype(int)), col, line_width,
                     cv2.LINE_AA)
    if label:
        mid = (start + end) / 2
        label_at(img, label, [end + along * 16, start - along * 16, end + along * 16 + normal * 10, start - along * 16 - normal * 10,
                              mid + normal * 14, mid - normal * 14], mask, taken if taken is not None else [])


def scale_bar(img, unit_per_px: float | None, unit: str, mask=None, taken=None):
    """Barra de escala (largo redondo ≈ 25 % del recorte) en la esquina que no tape el
    objeto ni los rótulos. unit_per_px: unidades reales por píxel del recorte."""
    if not unit_per_px:
        return
    taken = taken if taken is not None else []
    height, width = img.shape[:2]
    target = width * 0.25 * unit_per_px
    nice = min((step * 10 ** event for event in range(-3, 4) for step in (1, 2, 5)), key=lambda step: abs(step - target))
    pixel_x = int(round(nice / unit_per_px))
    best = None
    for x0, y in ((8, height - 8), (width - 8 - pixel_x, height - 8), (8, 22), (width - 8 - pixel_x, 22)):
        box = (x0 - 2, y - 22, x0 + pixel_x + 2, y + 3)
        if _free(img, box, mask, taken):
            best = (x0, y, box)
            break
    x0, y, box = best or (8, height - 8, (6, height - 30, 10 + pixel_x, height - 5))
    taken.append(box)
    for col, line_width in ((DARK, 4), (WHITE, 2)):
        cv2.line(img, (x0, y), (x0 + pixel_x, y), col, line_width)
    text(img, f"{nice:g} {display(unit)}", x0 + pixel_x / 2, y - 10, 0.34)


def object_mask(shape, contour, origin, factor):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round((np.asarray(contour, float) - origin) * factor).astype(np.int32).reshape(-1, 1, 2)], 255)
    return mask


def morph_axes(contour):
    """Largo y ancho del rectángulo mínimo (como se miden) como dos segmentos."""
    (cx, cy), (side_a, side_b), ang = cv2.minAreaRect(contour.reshape(-1, 1, 2).astype(np.float32))
    angle_rad = np.deg2rad(ang)
    along, across = np.array([np.cos(angle_rad), np.sin(angle_rad)]), np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    if side_a < side_b:
        along, across = across, along
        side_a, side_b = side_b, side_a
    center = np.array([cx, cy])
    return (center - along * side_a / 2, center + along * side_a / 2), (center - across * side_b / 2, center + across * side_b / 2)


def shape_view(obj: np.ndarray, mean: np.ndarray, size=CROP) -> np.ndarray:
    """Silueta del objeto (rojo) sobre la forma media (gris), alineadas y normalizadas."""
    from fenotit.core.efd import contour_points
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    mean_points, object_points = contour_points(mean, 200), contour_points(obj, 200)
    allp = np.vstack([mean_points, object_points])
    low, high = allp.min(axis=0), allp.max(axis=0)
    factor = 0.85 * min(size[0] / (high[0] - low[0] or 1), size[1] / (high[1] - low[1] or 1))
    off = np.array(size) / 2 - (low + high) / 2 * factor

    def poly(points):
        return np.round(points * factor + off).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly(mean_points)], True, (190, 190, 190), 5, cv2.LINE_AA)
    cv2.polylines(img, [poly(object_points)], True, (40, 40, 220), 2, cv2.LINE_AA)
    return img


def per_unit(mm_per_px, unit) -> float | None:
    return mm_per_px / TO_MM.get(unit, 1.0) if mm_per_px else None
