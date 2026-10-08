"""Gráficos de la foto actual (panel bajo la tabla) y del lote (ventana aparte).

Cada tipo de gráfico muestra solo las opciones que usa: dispersión = X e Y;
barras, cajas e histograma = hasta 3 variables (un gráfico cada una); ± DE solo en
barras del lote (desviación entre los objetos de cada foto)."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    MPL_OK = True
except ImportError:
    MPL_OK = False

from fenotit import log
from fenotit.core.image_io import natural_key
from fenotit.gui.help import HelpIcon
from fenotit.gui.theme import COLORS, FONTS
from fenotit.i18n import t

_log = log.get("gui.charts")

PALETTE = ["#2166AC", "#D73027", "#1A9850", "#8073AC", "#E08214", "#01665E",
           "#762A83", "#4DAC26", "#F4A582", "#92C5DE", "#A6D96A", "#FDAE61"]

MPL_STYLE = {
    "axes.facecolor": "#FAFAFA", "figure.facecolor": "#FFFFFF", "axes.edgecolor": "#CCCCCC",
    "axes.labelcolor": "#333333", "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#EEEEEE",
    "grid.linewidth": 0.6, "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 96,
}

NONE = "—"
SKIP = {"index", "_total", "object_id", "touching", "Image_ID", "centroid_x", "centroid_y",
        "centroid_x_px", "centroid_y_px", "image_width_px", "image_height_px"}
PREFERRED = ("area", "length", "width")

# tipo → (n.º de variables, nombres de los ejes)
KINDS = {
    "bars":    (3, None),
    "box":     (3, None),
    "hist":    (3, None),
    "scatter": (2, ("X", "Y")),
}


def numeric_cols(rows: list[dict]) -> list[str]:
    cols = []
    for row in rows:
        for k, v in row.items():
            if k not in cols and k not in SKIP and not k.endswith("_per_px") \
                    and isinstance(v, (int, float)) and not isinstance(v, bool):
                cols.append(k)
    return cols


def _defaults(cols: list[str], n: int) -> list[str]:
    out = [c for p in PREFERRED for c in cols if c.lower().startswith(p)][:n]
    out = list(dict.fromkeys(out))
    out += [c for c in cols if c not in out][:n - len(out)]
    return out


def _values(rows: list[dict], col: str) -> list[float]:
    return [float(r[col]) for r in rows if isinstance(r.get(col), (int, float)) and r[col] == r[col]]


def _style_ax(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(0.5)


def _num(v: float) -> str:
    """Sin notación científica: enteros desde 1000, 4 cifras significativas por debajo."""
    return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def _label(col: str) -> str:
    return col.replace("_", " ")


class _Chooser(tk.Frame):
    """Tipo de gráfico + las variables que ese tipo usa (+ opciones propias)."""

    def __init__(self, parent, kinds: tuple, on_change, bg: str, owner=None):
        super().__init__(parent, bg=bg)
        self.bg, self.on_change = bg, on_change
        self.kind = tk.StringVar(value=kinds[0])
        self.cols: list[str] = []
        self.chosen: dict[str, list[str]] = {}
        for k in kinds:
            tk.Radiobutton(self, text=t(f"chart.{k}"), variable=self.kind, value=k, command=self._kind_changed,
                           bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg,
                           font=FONTS["small"]).pack(side=tk.LEFT, padx=2)
        tk.Frame(self, bg=COLORS["border"], width=1).pack(side=tk.LEFT, fill=tk.Y, pady=4, padx=6)
        self.vars_frame = tk.Frame(owner or self, bg=bg)      # con owner puede bajar a otra fila
        self.vars_frame.pack(in_=self, side=tk.LEFT)
        self.extra = tk.Frame(self, bg=bg)          # opciones solo de algunos tipos
        self.extra.pack(side=tk.LEFT, padx=(6, 0))
        self._svars: list[tk.StringVar] = []

    def set_columns(self, cols: list[str]):
        if cols != self.cols:
            self.cols = cols
            self.chosen = {k: [c for c in v if c in cols] for k, v in self.chosen.items()}
        self._build_vars()

    def selected(self) -> list[str]:
        return [v.get() for v in self._svars if v.get() in self.cols]

    def _kind_changed(self):
        self._build_vars()
        self.on_change()

    def _build_vars(self):
        for w in self.vars_frame.winfo_children():
            w.destroy()
        kind = self.kind.get()
        n, names = KINDS[kind]
        current = self.chosen.get(kind) or _defaults(self.cols, n)
        self._svars = []
        for i in range(n):
            name = names[i] if names else f"V{i + 1}"
            tk.Label(self.vars_frame, text=f"{name}:", bg=self.bg, fg=COLORS["text_muted"],
                     font=FONTS["small"]).pack(side=tk.LEFT, padx=(4, 1))
            optional = names is None and i > 0
            sv = tk.StringVar(value=current[i] if i < len(current) else NONE)
            cb = ttk.Combobox(self.vars_frame, textvariable=sv, state="readonly", width=14, font=FONTS["small"],
                              values=([NONE] if optional else []) + self.cols)
            cb.pack(side=tk.LEFT, padx=2, pady=3)
            cb.bind("<<ComboboxSelected>>", lambda e: self._vars_changed())
            self._svars.append(sv)

    def _vars_changed(self):
        self.chosen[self.kind.get()] = [v.get() for v in self._svars]
        self.on_change()


def _embed(frame, fig, toolbar=False):
    for w in frame.winfo_children():
        w.destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    if toolbar:
        bar = tk.Frame(frame, bg=COLORS["bg_card"])
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        NavigationToolbar2Tk(canvas, bar).update()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return canvas


def _message(frame, text):
    for w in frame.winfo_children():
        w.destroy()
    tk.Label(frame, text=text, bg=COLORS["bg_card"], fg=COLORS["text_muted"], font=FONTS["small"],
             justify="center").pack(pady=20)


class _Picker:
    """Clic (sin arrastrar) en un punto del gráfico → on_pick(dato de ese punto) y un
    anillo rojo lo marca."""

    def __init__(self):
        self.sets: list = []
        self._press = self._ring = None

    def add(self, ax, xs, ys, payloads):
        if len(payloads):
            self.sets.append((ax, np.c_[np.asarray(xs, float), np.asarray(ys, float)], list(payloads)))

    def connect(self, canvas, on_pick):
        if not self.sets or on_pick is None:
            return
        canvas.mpl_connect("button_press_event", lambda e: setattr(self, "_press", (e.x, e.y)))
        canvas.mpl_connect("button_release_event", lambda e: self._release(e, canvas, on_pick))

    def _release(self, e, canvas, on_pick):
        p = self._press
        if e.button != 1 or e.x is None or p is None or abs(e.x - p[0]) + abs(e.y - p[1]) > 4:
            return
        if getattr(getattr(canvas, "toolbar", None), "mode", ""):       # zoom/mover de la barra
            return
        best = None
        for ax, xy, pl in self.sets:
            if e.inaxes is not ax:
                continue
            d = np.hypot(*(ax.transData.transform(xy) - (e.x, e.y)).T)
            i = int(np.argmin(d))
            if d[i] <= 7 and (best is None or d[i] < best[0]):
                best = (d[i], ax, xy[i], pl[i])
        if best is None:
            return
        if self._ring is not None:
            try:
                self._ring.remove()
            except Exception:
                pass
        _, ax, (x, y), payload = best
        self._ring = ax.plot([x], [y], "o", mfc="none", mec="#D73027", mew=1.6, ms=11, zorder=5)[0]
        canvas.draw_idle()
        on_pick(payload)


def _save(fig, parent):
    if fig is None:
        return
    path = filedialog.asksaveasfilename(parent=parent, title=t("chart.save"), defaultextension=".png",
                                        filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")])
    if path:
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
            return path
        except Exception as e:
            _log.exception("Guardar gráfico")
            messagebox.showerror(t("common.error"), str(e), parent=parent)
    return None


def _scatter(fig, xs, ys, x_col, y_col, labels=None):
    ax = fig.add_subplot(1, 1, 1)
    fig._scatter_ax = ax
    ax.scatter(xs, ys, color=PALETTE[0], alpha=0.75, s=28 if labels else 16, edgecolors="white",
               linewidths=0.5, zorder=3)
    if labels and len(xs) <= 30:
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6, color="#555555")
    ax.set_xlabel(_label(x_col))
    ax.set_ylabel(_label(y_col))
    r = np.corrcoef(xs, ys)[0, 1] if len(xs) > 2 and np.std(xs) > 0 and np.std(ys) > 0 else None
    ax.set_title(f"n = {len(xs)}" + (f"   r = {r:.2f}" if r is not None else ""))
    _style_ax(ax)


# ── Foto actual ───────────────────────────────────────────────────────────────

class IntraImageChartPanel(tk.Frame):
    """Gráficos de los objetos de la foto actual. Se dibuja solo cuando está a la vista
    (al abrir la pestaña Gráficos o al cambiar de foto con la pestaña abierta)."""

    def __init__(self, parent, colors: dict, on_pick=None, **kw):
        super().__init__(parent, bg=colors["bg_card"], **kw)
        self.on_pick = on_pick                          # on_pick(object_id): clic en un punto
        self._rows: list[dict] = []
        self._dirty = False
        self._fig = None
        if not MPL_OK:
            tk.Label(self, text=t("chart.no_mpl"), bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                     font=FONTS["small"]).pack(pady=8)
            return
        bg = COLORS["bg_panel"]
        self.bar = tk.Frame(self, bg=bg)
        self.bar.pack(fill=tk.X)
        self.chooser = _Chooser(self.bar, ("box", "hist", "scatter"), self._refresh, bg, owner=self)
        self.chooser.pack(side=tk.LEFT, padx=4)
        self.bar2 = tk.Frame(self, bg=bg)               # las variables bajan aquí si no caben
        for w in (self.bar, self.chooser.vars_frame):
            w.bind("<Configure>", lambda e: self.after_idle(self._fit_bar), add="+")
        self._frame = tk.Frame(self, bg=COLORS["bg_card"])
        self._frame.pack(fill=tk.BOTH, expand=True)
        self._drawn = (0, 0)
        self._resize_job = None
        self._frame.bind("<Configure>", self._on_resize)

    def _fit_bar(self):
        vf = self.chooser.vars_frame
        below = vf.winfo_manager() == "pack" and str(vf.pack_info().get("in")) == str(self.bar2)
        width = self.bar.winfo_width()
        need = self.bar.winfo_reqwidth() + (vf.winfo_reqwidth() + 10 if below else 0)
        if not below and need > width > 1:
            vf.pack_forget()
            vf.pack(in_=self.bar2, side=tk.LEFT, padx=8, pady=(0, 3))
            self.bar2.pack(fill=tk.X, after=self.bar)
            vf.lift()
        elif below and need <= width:
            vf.pack_forget()
            vf.pack(in_=self.chooser, side=tk.LEFT, before=self.chooser.extra)
            self.bar2.pack_forget()
            vf.lift()

    def _on_resize(self, e):
        """Al cambiar de tamaño (panel de abajo ↔ lugar de una imagen) se vuelve a dibujar."""
        if abs(e.width - self._drawn[0]) < 30 and abs(e.height - self._drawn[1]) < 30:
            return
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(250, self.redraw)

    def redraw(self):
        self._resize_job = None
        if MPL_OK and self.winfo_ismapped() and self._rows:
            self._refresh()

    def load(self, rows: list[dict] | None):
        """Datos de la foto que se ve; se dibuja ahora si el panel está a la vista."""
        rows = list(rows or [])
        if rows == self._rows and not self._dirty and self._fig is not None:
            return
        self._rows = rows
        self._dirty = True
        if MPL_OK:
            self.chooser.set_columns(numeric_cols(self._rows))
            if self.winfo_ismapped():
                self._refresh()

    def show(self):
        if self._dirty and MPL_OK:
            self._refresh()

    def _refresh(self):
        self._dirty = False
        self._fig = None
        cols = self.chooser.selected()
        if not self._rows:
            _message(self._frame, t("chart.no_data_image"))
            return
        if not cols:
            _message(self._frame, t("chart.pick_var"))
            return
        plt.rcParams.update(MPL_STYLE)
        kind = self.chooser.kind.get()
        fw, fh = self._frame.winfo_width(), self._frame.winfo_height()
        self._drawn = (fw, fh)
        w, h = max(fw, 300) / 96, max(fh, 160) / 96
        fig = Figure(figsize=(w, h), dpi=96)
        picker = _Picker()
        stacked = w / h < 1.3                       # alto y angosto: uno debajo del otro
        n_axes = max(1, len(cols))
        narrow = (fw if stacked else fw / n_axes) < 260  # poco ancho por gráfico: título más corto
        if kind == "scatter":
            if len(cols) < 2:
                _message(self._frame, t("chart.pick_xy"))
                return
            pairs = [(float(r[cols[0]]), float(r[cols[1]]), r.get("object_id")) for r in self._rows
                     if isinstance(r.get(cols[0]), (int, float)) and isinstance(r.get(cols[1]), (int, float))]
            if not pairs:
                _message(self._frame, t("chart.no_data_image"))
                return
            xs, ys, ids = zip(*pairs)
            _scatter(fig, xs, ys, cols[0], cols[1])
            picker.add(fig._scatter_ax, xs, ys, ids)
        else:
            for i, col in enumerate(cols):
                ax = fig.add_subplot(len(cols), 1, i + 1) if stacked else fig.add_subplot(1, len(cols), i + 1)
                vals = _values(self._rows, col)
                color = PALETTE[i % len(PALETTE)]
                if not vals:
                    continue
                if kind == "box":
                    bp = ax.boxplot(vals, patch_artist=True, widths=0.5,
                                    medianprops=dict(color="#333333", linewidth=1.5))
                    bp["boxes"][0].set_facecolor(color + "66")
                    bp["boxes"][0].set_edgecolor(color)
                    rng = np.random.default_rng(0)
                    jx = 1 + rng.uniform(-0.15, 0.15, len(vals))
                    ax.scatter(jx, vals, alpha=0.4, s=10, color=color, zorder=3)
                    ids = [r.get("object_id") for r in self._rows
                           if isinstance(r.get(col), (int, float)) and r[col] == r[col]]
                    picker.add(ax, jx, vals, ids)
                    ax.set_xticks([])
                else:
                    ax.hist(vals, bins=min(20, max(5, len(vals) // 3)), color=color, edgecolor="white",
                            alpha=0.85, linewidth=0.5)
                    ax.set_ylabel(t("chart.frequency"))
                sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                med = f"{t('chart.mean')} = {_num(np.mean(vals))}"
                dev = f"{t('chart.sd')} = {_num(sd)}"
                ax.set_title(f"{_label(col)}\nn = {len(vals)}  {med}\n{dev}" if narrow else
                             f"{_label(col)}\nn = {len(vals)}   {med}   {dev}", fontsize=7 if narrow else 8)
                if narrow:
                    ax.tick_params(labelsize=7)
                _style_ax(ax)
        try:
            fig.tight_layout(pad=0.8)
        except Exception:
            _log.debug("tight_layout", exc_info=True)
        self._fig = fig
        picker.connect(_embed(self._frame, fig), self.on_pick)

    def save_figure(self, image=None):
        """Guarda el gráfico; con `image` = (vista, imagen BGR), también la imagen de la vista."""
        path = _save(self._fig, self)
        if path and image is not None:
            import cv2
            view, img = image
            out = Path(path).with_name(f"{Path(path).stem}_{view}.png")
            ok, buf = cv2.imencode(".png", img)
            if ok:
                out.write_bytes(buf.tobytes())


# ── Lote ──────────────────────────────────────────────────────────────────────

def batch_table(results: dict) -> tuple[list[str], dict[str, dict], dict[str, dict]]:
    """Una fila por foto: valores de la foto (conteos…) y la media de cada medición de
    sus objetos (`<col>_mean`); `sds` = desviación entre los objetos de cada foto."""
    names, means, sds = [], {}, {}
    for path in sorted(results, key=lambda p: natural_key(Path(p).name)):
        r = results[path]
        if r is None or r.status != "ok":
            continue
        name = Path(path).stem
        row, sd = {}, {}
        for k, v in (r.stats or {}).items():
            if k not in SKIP and not k.endswith("_per_px") and isinstance(v, (int, float)) \
                    and not isinstance(v, bool):
                row[k] = float(v)
        rows = r.measurements or []
        for col in numeric_cols(rows):
            vals = _values(rows, col)
            if vals:
                row[f"{col}_mean"] = float(np.mean(vals))
                sd[f"{col}_mean"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        names.append(name)
        means[name], sds[name] = row, sd
    return names, means, sds


def batch_records(results: dict, meta: dict | None = None, unit: str = "images") -> tuple[list[dict], list[str]]:
    """Los datos de la comparación: una fila por foto (unit="images") o por objeto
    ("objects"). Cada fila: `_name` (foto), `_meta` (columnas de la tabla del usuario),
    `_sd` (DE entre los objetos de la foto, solo por foto) y los valores numéricos."""
    meta = meta or {}
    stems = {Path(p).stem: p for p in results}
    recs, cols = [], []
    if unit == "images":
        names, means, sds = batch_table(results)
        for n in names:
            recs.append({"_name": n, "_path": stems[n], "_oid": None, "_meta": meta.get(stems[n]) or {},
                         "_sd": sds[n], **means[n]})
            cols += [k for k in means[n] if k not in cols]
        return recs, cols
    for path in sorted(results, key=lambda p: natural_key(Path(p).name)):
        r = results[path]
        if r is None or r.status != "ok":
            continue
        rows = r.measurements or []
        num = numeric_cols(rows)
        cols += [c for c in num if c not in cols]
        n = (r.stats or {}).get("n_objects")            # conteo de su foto: ¿más objetos, más chicos?
        for x in rows:
            recs.append({"_name": Path(path).stem, "_path": path, "_oid": x.get("object_id"),
                         "_meta": meta.get(path) or {}, "_sd": {},
                         **({"n_objects": float(n)} if isinstance(n, (int, float)) else {}),
                         **{c: float(x[c]) for c in num if isinstance(x.get(c), (int, float)) and x[c] == x[c]}})
    if any("n_objects" in r for r in recs):
        cols.append("n_objects")
    return recs, cols


def _count_vs_size(cols: list[str]) -> list[str]:
    """Dispersión por defecto en el lote: conteo de la foto vs. tamaño medio de sus objetos."""
    size = next((c for p in PREFERRED for c in cols if c.lower().startswith(p)), None)
    return ["n_objects", size] if "n_objects" in cols and size else []


def _group_name(v) -> str:
    return t("chart.no_value") if v in (None, "") else str(v)


class BatchChartWindow(tk.Toplevel):
    """Comparación entre fotos: un dato por foto (sus valores o la media de sus objetos)
    o por objeto; agrupados por una columna de la tabla del usuario (genotipo, rep…)."""

    def __init__(self, parent, results: dict, meta: dict | None = None, groups: list[str] | None = None,
                 on_pick=None):
        """on_pick(ruta, objeto|None): clic en un punto."""
        super().__init__(parent)
        self.on_pick = on_pick
        self.title(t("chart.batch_title"))
        self.configure(bg=COLORS["bg_card"])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1100, int(sw * 0.85)), min(700, int(sh * 0.82))
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self._fig = None
        self.results, self.meta = results, {p: (meta or {}).get(p) or {} for p in results}
        self.recs, cols = batch_records(results, self.meta, "images")
        if not MPL_OK or not self.recs:
            _message(self, t("chart.no_mpl") if not MPL_OK else t("chart.no_data_batch"))
            return
        tk.Frame(self, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        bg = COLORS["bg_panel"]
        top = tk.Frame(self, bg=bg)
        top.pack(fill=tk.X)
        self.units = {"images": "images", "objects": "objects"}
        self.unit = tk.StringVar(value="images")
        tk.Label(top, text=t("chart.unit") + ":", bg=bg, fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(10, 2))
        for key in ("images", "objects"):
            tk.Radiobutton(top, text=t(f"chart.unit_{key}"), variable=self.unit, value=key,
                           command=self._unit_changed, bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"],
                           activebackground=bg, font=FONTS["small"]).pack(side=tk.LEFT, pady=4)
        HelpIcon(top, t("chart.unit"), t("chart.unit_help"), bg=bg).pack(side=tk.LEFT, padx=(4, 0))
        self.group = tk.StringVar(value=NONE)
        if groups:
            tk.Label(top, text=t("chart.group_by") + ":", bg=bg, fg=COLORS["text_muted"],
                     font=FONTS["small"]).pack(side=tk.LEFT, padx=(14, 2))
            gb = ttk.Combobox(top, textvariable=self.group, state="readonly", width=14, font=FONTS["small"],
                              values=[NONE] + list(groups))
            gb.pack(side=tk.LEFT, pady=4)
            gb.bind("<<ComboboxSelected>>", lambda e: self._refresh())
        tk.Button(top, text=t("chart.save"), command=lambda: _save(self._fig, self), bg=COLORS["btn_bg"],
                  fg=COLORS["accent"], relief="flat", font=FONTS["small"], cursor="hand2").pack(side=tk.RIGHT, padx=8)
        bar = tk.Frame(self, bg=bg)
        bar.pack(fill=tk.X)
        self.chooser = _Chooser(bar, ("bars", "box", "hist", "scatter"), self._refresh, bg)
        self.chooser.pack(side=tk.LEFT, padx=8, pady=(0, 4))
        self.show_sd = tk.BooleanVar(value=False)
        self._sd_check = tk.Checkbutton(self.chooser.extra, text=t("chart.show_sd"), variable=self.show_sd,
                                        command=self._refresh, bg=bg, fg=COLORS["text"],
                                        selectcolor=COLORS["bg_card"], activebackground=bg, font=FONTS["small"])
        self._frame = tk.Frame(self, bg=COLORS["bg_card"])
        self._frame.pack(fill=tk.BOTH, expand=True)
        self.chooser.set_columns(cols)
        self.chooser.chosen.setdefault("scatter", _count_vs_size(cols))
        self.after(50, self._refresh)

    def _unit_changed(self):
        self.recs, cols = batch_records(self.results, self.meta, self.units.get(self.unit.get(), "images"))
        self.chooser.set_columns(cols)
        if not [c for c in self.chooser.chosen.get("scatter", []) if c in cols][1:]:
            self.chooser.chosen["scatter"] = _count_vs_size(cols)
        if self.chooser.kind.get() == "scatter":
            self.chooser._build_vars()
        self._refresh()

    def _categories(self, kind: str, col: str) -> list[tuple[str, list[dict]]]:
        """(nombre, filas) de cada barra/caja/serie: el grupo elegido; sin grupo, cada foto
        (barras, y cajas por objeto) o todo junto."""
        g = self.group.get()
        objects = self.units.get(self.unit.get()) == "objects"
        if g != NONE:
            key = lambda r: _group_name(r["_meta"].get(g))
        elif kind == "bars" or (kind == "box" and objects):
            key = lambda r: r["_name"]
        else:
            key = lambda r: ""
        out: dict[str, list[dict]] = {}
        for r in self.recs:
            if col is None or col in r:
                out.setdefault(key(r), []).append(r)
        empty = t("chart.no_value")                 # las fotos sin dato, al final
        return sorted(out.items(), key=lambda kv: (kv[0] == empty, natural_key(kv[0])))

    def _refresh(self):
        kind = self.chooser.kind.get()
        cols = self.chooser.selected()
        grouped = self.group.get() != NONE
        per_image = self.units.get(self.unit.get()) == "images"
        within = kind == "bars" and per_image and not grouped        # DE entre los objetos de cada foto
        has_sd = kind == "bars" and any(
            (any(r["_sd"].get(c) for r in self.recs) if within
             else any(len(rows) > 1 for _, rows in self._categories(kind, c))) for c in cols)
        if has_sd:
            self._sd_check.configure(text=t("chart.show_sd") if within else t("chart.show_sd_group"))
            self._sd_check.pack(side=tk.LEFT)
        else:
            self._sd_check.pack_forget()
        if not cols:
            _message(self._frame, t("chart.pick_var"))
            return
        plt.rcParams.update(MPL_STYLE)
        w = max(self._frame.winfo_width(), 600) / 96
        h = max(self._frame.winfo_height(), 300) / 96 - 0.4
        fig = Figure(figsize=(w, h), dpi=96)
        self._picker = _Picker()
        unit_n = "chart.n_images" if per_image else "chart.n_objects"
        if kind == "scatter":
            if len(cols) < 2:
                _message(self._frame, t("chart.pick_xy"))
                return
            if not self._scatter(fig, cols[0], cols[1], per_image and not grouped):
                _message(self._frame, t("chart.no_data_batch"))
                return
        else:
            groups = [name for name, _ in self._categories(kind, None)] if grouped else []
            colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
            for i, col in enumerate(cols):
                ax = fig.add_subplot(1, len(cols), i + 1)
                cats = self._categories(kind, col)
                vals = [[r[col] for r in rows] for _, rows in cats]
                n = sum(len(v) for v in vals)
                if not n:
                    continue
                names = [c for c, _ in cats]
                base = PALETTE[i % len(PALETTE)]
                cc = [colors.get(c, base) for c in names]
                if kind == "bars":
                    x = np.arange(len(names))
                    ax.bar(x, [np.mean(v) for v in vals], color=cc, alpha=0.85, width=0.65,
                           edgecolor="white", linewidth=0.5)
                    if self.show_sd.get() and has_sd:
                        err = [r["_sd"].get(col, 0) for _, rows in cats for r in rows[:1]] if within else \
                            [np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals]
                        ax.errorbar(x, [np.mean(v) for v in vals], yerr=err, fmt="none",
                                    color="#333333", capsize=2, linewidth=0.8)
                    self._xticks(ax, x, names)
                elif kind == "box":
                    bp = ax.boxplot(vals, patch_artist=True, widths=0.5, showfliers=False,
                                    medianprops=dict(color="#333333", linewidth=1.5))
                    rng = np.random.default_rng(0)
                    for k, (box, v) in enumerate(zip(bp["boxes"], vals)):
                        box.set_facecolor(cc[k] + "55")
                        box.set_edgecolor(cc[k])
                        if len(v) <= 2000:
                            jx = k + 1 + rng.uniform(-0.12, 0.12, len(v))
                            ax.scatter(jx, v, color=cc[k], alpha=0.6, s=10 if len(v) > 100 else 14, zorder=3,
                                       linewidths=0)
                            self._picker.add(ax, jx, v, [(r["_path"], r["_oid"]) for r in cats[k][1]])
                    if len(names) > 1:
                        self._xticks(ax, np.arange(1, len(names) + 1), names)
                    else:
                        ax.set_xticks([])
                else:
                    allv = np.concatenate([np.asarray(v, float) for v in vals])
                    bins = np.histogram_bin_edges(allv, bins=min(20, max(5, len(allv) // 2)))
                    for k, (name, v) in enumerate(zip(names, vals)):
                        ax.hist(v, bins=bins, color=cc[k], alpha=0.55 if len(names) > 1 else 0.85,
                                edgecolor="white", linewidth=0.5, label=name if grouped else None)
                    ax.set_ylabel(t("chart.frequency"))
                    if grouped:
                        ax.legend(frameon=False, fontsize=7)
                ax.set_title(f"{_label(col)}\n{t(unit_n, n=n)}", fontsize=8)
                _style_ax(ax)
        try:
            fig.tight_layout(pad=1.0)
        except Exception:
            _log.debug("tight_layout", exc_info=True)
        self._fig = fig
        self._picker.connect(_embed(self._frame, fig, toolbar=True),
                             (lambda pl: self.on_pick(*pl)) if self.on_pick else None)

    @staticmethod
    def _xticks(ax, x, names):
        ax.set_xticks(x)
        many = len(names) > 8 or max((len(n) for n in names), default=0) > 10
        ax.set_xticklabels(names if len(names) <= 60 else [""] * len(names),
                           rotation=60 if many else 0, ha="right" if many else "center", fontsize=6 if many else 7)

    def _scatter(self, fig, xc: str, yc: str, labels: bool) -> bool:
        if self.group.get() == NONE:
            pts = [(r[xc], r[yc], r["_name"], (r["_path"], r["_oid"])) for r in self.recs if xc in r and yc in r]
            if not pts:
                return False
            xs, ys, names, pl = zip(*pts)
            _scatter(fig, xs, ys, xc, yc, list(names) if labels else None)
            self._picker.add(fig._scatter_ax, xs, ys, pl)
            return True
        cats = [(g, [r for r in rows if xc in r and yc in r]) for g, rows in self._categories("scatter", None)]
        cats = [(g, rows) for g, rows in cats if rows]
        if not cats:
            return False
        ax = fig.add_subplot(1, 1, 1)
        xs_all, ys_all = [], []
        for k, (g, rows) in enumerate(cats):
            xs, ys = [r[xc] for r in rows], [r[yc] for r in rows]
            xs_all += xs
            ys_all += ys
            ax.scatter(xs, ys, color=PALETTE[k % len(PALETTE)], alpha=0.75, s=22 if len(xs_all) < 300 else 10,
                       edgecolors="white", linewidths=0.4, zorder=3, label=g)
            self._picker.add(ax, xs, ys, [(r["_path"], r["_oid"]) for r in rows])
        ax.set_xlabel(_label(xc))
        ax.set_ylabel(_label(yc))
        ok = len(xs_all) > 2 and np.std(xs_all) > 0 and np.std(ys_all) > 0
        ax.set_title(f"n = {len(xs_all)}" + (f"   r = {np.corrcoef(xs_all, ys_all)[0, 1]:.2f}" if ok else ""))
        ax.legend(frameon=False, fontsize=7, title=self.group.get(), title_fontsize=7)
        _style_ax(ax)
        return True
