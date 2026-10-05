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
from fenotit.gui.theme import COLORS, FONTS
from fenotit.i18n import t

CROP = (220, 150)
WHITE, DARK = (255, 255, 255), (25, 25, 25)


class Inspector(tk.Frame):

    def __init__(self, parent, on_close, on_exclude, on_highlight):
        bg = COLORS["bg_panel"]
        super().__init__(parent, bg=bg)
        head = tk.Frame(self, bg=bg)
        head.pack(fill=tk.X)
        self._title = tk.Label(head, text="", bg=bg, fg=COLORS["accent"], font=("Segoe UI", 8, "bold"), pady=5)
        self._title.pack(side=tk.LEFT, padx=(10, 6))
        close = tk.Label(head, text="×", bg=bg, fg=COLORS["text_muted"], font=("Segoe UI", 11), cursor="hand2")
        close.pack(side=tk.RIGHT, padx=8)
        close.bind("<Button-1>", lambda e: on_close())
        tk.Frame(self, bg=COLORS["border"], height=1).pack(fill=tk.X)
        self._pic = tk.Label(self, bg=COLORS["bg_card"], bd=0, highlightthickness=1,
                             highlightbackground=COLORS["border"])
        self._pic.pack(padx=8, pady=(6, 4))
        self._state = tk.Label(self, text="", bg=bg, fg=COLORS["warning"], font=FONTS["small"],
                               justify="left", anchor="w", wraplength=200)
        self._rows = tk.Frame(self, bg=bg)
        self._rows.pack(fill=tk.X, padx=10)
        btns = tk.Frame(self, bg=bg)
        btns.pack(fill=tk.X, padx=10, pady=(6, 8))
        kw = dict(bg=COLORS["btn_bg"], fg=COLORS["accent"], relief="flat", font=FONTS["small"], cursor="hand2", pady=2)
        self._star = tk.Button(btns, command=on_highlight, **kw)
        self._star.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self._excl = tk.Button(btns, command=on_exclude, **kw)
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
            self._state.pack(fill=tk.X, padx=10, before=self._rows)
        else:
            self._state.pack_forget()
        self._fill(rows)
        self._star.config(text=t("insp.unhighlight") if highlighted else t("insp.highlight"),
                          state="disabled" if excluded else "normal")
        self._excl.config(text=t("insp.include") if excluded else t("insp.exclude"))

    def shrink(self, px: int) -> bool:
        """Achica el recorte para dejar sitio a CAPAS. False si ya no se puede."""
        if self._crop is None or not self._pic.winfo_ismapped():
            return False
        h = self._pic.winfo_reqheight() - px - 4
        if h < 50:
            self._pic.pack_forget()
            return False
        k = h / self._crop.shape[0]
        self._set_pic(cv2.resize(self._crop, (max(1, int(self._crop.shape[1] * k)), max(1, int(h))),
                                 interpolation=cv2.INTER_AREA), keep=True)
        return True

    def _set_pic(self, crop, keep: bool = False):
        if crop is None:
            self._pic.pack_forget()
            return
        im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(im)
        self._pic.config(image=self._photo, width=im.width, height=im.height)
        if not keep:
            self._pic.pack(padx=8, pady=(6, 4), before=self._rows)

    def _fill(self, rows):
        bg = COLORS["bg_panel"]
        for w in self._rows.winfo_children():
            w.destroy()
        for r, item in enumerate(rows):
            if item[0] == "bar":
                bar = tk.Canvas(self._rows, width=200, height=12, bg=bg, highlightthickness=0)
                bar.grid(row=r, column=0, columnspan=3, sticky="w", pady=(2, 4))
                x = 0.0
                for hexc, pct in item[1]:
                    w = 200 * pct / 100
                    bar.create_rectangle(x, 0, x + w, 12, fill=hexc, outline="")
                    x += w
                bar.create_rectangle(0, 0, 199, 11, outline=COLORS["border"])
            elif item[0] == "color":
                _, hexc, text, pct = item
                tk.Frame(self._rows, bg=hexc, width=12, height=12, highlightthickness=1,
                         highlightbackground=COLORS["border"]).grid(row=r, column=0, sticky="w")
                tk.Label(self._rows, text=text, bg=bg, fg=COLORS["text"], font=FONTS["small"],
                         anchor="w").grid(row=r, column=1, sticky="w", padx=(6, 0))
                tk.Label(self._rows, text=f"{pct:.0f} %", bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                         anchor="e").grid(row=r, column=2, sticky="e")
            else:
                k, v = item
                tk.Label(self._rows, text=k, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                         anchor="w").grid(row=r, column=0, columnspan=2, sticky="w")
                tk.Label(self._rows, text=v, bg=bg, fg=COLORS["text"], font=FONTS["small"],
                         anchor="e").grid(row=r, column=2, sticky="e", padx=(8, 0))
        self._rows.columnconfigure(2, weight=1)


# ── Dibujo del recorte por capa (sin tkinter) ────────────────────────────────

def crop_around(img: np.ndarray, contour: np.ndarray, size=CROP, extra=None, margin: float = 0.25):
    """Recorte del objeto (y de `extra`, puntos que también deben verse) con margen,
    escalado para caber en `size`: (recorte, origen, zoom)."""
    pts = contour if extra is None or not len(extra) else np.vstack([contour, np.asarray(extra, float)])
    x, y, w, h = cv2.boundingRect(np.asarray(pts, np.float32).reshape(-1, 1, 2))
    m = int(max(w, h) * margin) + 6
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


def text(img, s, x, y, scale=0.38, dark=False):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(min(max(2, x - tw / 2), img.shape[1] - tw - 2))
    y = int(min(max(th + 2, y + th / 2), img.shape[0] - 3))
    halo, ink = (WHITE, DARK) if dark else (DARK, WHITE)      # dark = texto oscuro sobre fondo claro
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, halo, 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1, cv2.LINE_AA)


def fmt(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def _box(img, s, x, y, scale):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    return int(x - tw / 2) - 2, int(y - th / 2) - 3, int(x + tw / 2) + 2, int(y + th / 2) + 3


def _free(img, box, mask, taken) -> bool:
    """¿El rótulo cabe en el recorte sin tapar el objeto ni otro rótulo?"""
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    if x0 < 1 or y0 < 1 or x1 > w - 1 or y1 > h - 1:
        return False
    if mask is not None and mask[y0:y1, x0:x1].mean() > 8:
        return False
    return not any(x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1 for b in taken)


def label_at(img, s, candidates, mask, taken, scale=0.38, dark=False):
    """Escribe `s` en el primer lugar libre de `candidates` (si ninguno lo está, en el primero)."""
    spots = [np.asarray(c, float) for c in candidates]
    c = next((p for p in spots if _free(img, _box(img, s, p[0], p[1], scale), mask, taken)), spots[0])
    taken.append(_box(img, s, c[0], c[1], scale))
    text(img, s, c[0], c[1], scale, dark)


def cota(img, p1, p2, label, mask=None, taken=None):
    """Cota: línea con topes en los extremos y el valor fuera del objeto, donde no tape
    nada (después de un extremo o a un lado)."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    u = d / (np.hypot(*d) or 1)
    n = np.array([-u[1], u[0]])
    for col, wd in ((DARK, 3), (WHITE, 1)):
        cv2.line(img, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), col, wd, cv2.LINE_AA)
        for p in (p1, p2):
            cv2.line(img, tuple(np.round(p - n * 4).astype(int)), tuple(np.round(p + n * 4).astype(int)), col, wd,
                     cv2.LINE_AA)
    if label:
        mid = (p1 + p2) / 2
        label_at(img, label, [p2 + u * 16, p1 - u * 16, p2 + u * 16 + n * 10, p1 - u * 16 - n * 10,
                              mid + n * 14, mid - n * 14], mask, taken if taken is not None else [])


def scale_bar(img, unit_per_px: float | None, unit: str, mask=None, taken=None):
    """Barra de escala (largo redondo ≈ 25 % del recorte) en la esquina que no tape el
    objeto ni los rótulos. unit_per_px: unidades reales por píxel del recorte."""
    if not unit_per_px:
        return
    taken = taken if taken is not None else []
    h, w = img.shape[:2]
    target = w * 0.25 * unit_per_px
    nice = min((v * 10 ** e for e in range(-3, 4) for v in (1, 2, 5)), key=lambda v: abs(v - target))
    px = int(round(nice / unit_per_px))
    best = None
    for x0, y in ((8, h - 8), (w - 8 - px, h - 8), (8, 22), (w - 8 - px, 22)):
        box = (x0 - 2, y - 22, x0 + px + 2, y + 3)
        if _free(img, box, mask, taken):
            best = (x0, y, box)
            break
    x0, y, box = best or (8, h - 8, (6, h - 30, 10 + px, h - 5))
    taken.append(box)
    for col, wd in ((DARK, 4), (WHITE, 2)):
        cv2.line(img, (x0, y), (x0 + px, y), col, wd)
    text(img, f"{nice:g} {display(unit)}", x0 + px / 2, y - 10, 0.34)


def object_mask(shape, contour, origin, k):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round((np.asarray(contour, float) - origin) * k).astype(np.int32).reshape(-1, 1, 2)], 255)
    return m


def morph_axes(contour):
    """Largo y ancho del rectángulo mínimo (como se miden) como dos segmentos."""
    (cx, cy), (a, b), ang = cv2.minAreaRect(contour.reshape(-1, 1, 2).astype(np.float32))
    tt = np.deg2rad(ang)
    u, v = np.array([np.cos(tt), np.sin(tt)]), np.array([-np.sin(tt), np.cos(tt)])
    if a < b:
        u, v = v, u
        a, b = b, a
    c = np.array([cx, cy])
    return (c - u * a / 2, c + u * a / 2), (c - v * b / 2, c + v * b / 2)


def shape_view(obj: np.ndarray, mean: np.ndarray, size=CROP) -> np.ndarray:
    """Silueta del objeto (rojo) sobre la forma media (gris), alineadas y normalizadas."""
    from fenotit.core.efd import contour_points
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    a, b = contour_points(mean, 200), contour_points(obj, 200)
    allp = np.vstack([a, b])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    k = 0.85 * min(size[0] / (hi[0] - lo[0] or 1), size[1] / (hi[1] - lo[1] or 1))
    off = np.array(size) / 2 - (lo + hi) / 2 * k

    def poly(p):
        return np.round(p * k + off).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly(a)], True, (190, 190, 190), 5, cv2.LINE_AA)
    cv2.polylines(img, [poly(b)], True, (40, 40, 220), 2, cv2.LINE_AA)
    return img


def per_unit(mm_per_px, unit) -> float | None:
    return mm_per_px / TO_MM.get(unit, 1.0) if mm_per_px else None
