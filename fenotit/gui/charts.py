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


def _label(col: str) -> str:
    return col.replace("_", " ")


class _Chooser(tk.Frame):
    """Tipo de gráfico + las variables que ese tipo usa (+ opciones propias)."""

    def __init__(self, parent, kinds: tuple, on_change, bg: str):
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
        self.vars_frame = tk.Frame(self, bg=bg)
        self.vars_frame.pack(side=tk.LEFT)
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


def _message(frame, text):
    for w in frame.winfo_children():
        w.destroy()
    tk.Label(frame, text=text, bg=COLORS["bg_card"], fg=COLORS["text_muted"], font=FONTS["small"],
             justify="center").pack(pady=20)


def _save(fig, parent):
    if fig is None:
        return
    path = filedialog.asksaveasfilename(parent=parent, title=t("chart.save"), defaultextension=".png",
                                        filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")])
    if path:
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        except Exception as e:
            _log.exception("Guardar gráfico")
            messagebox.showerror(t("common.error"), str(e), parent=parent)


def _scatter(fig, xs, ys, x_col, y_col, labels=None):
    ax = fig.add_subplot(1, 1, 1)
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

    def __init__(self, parent, colors: dict, **kw):
        super().__init__(parent, bg=colors["bg_card"], **kw)
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
        self.chooser = _Chooser(self.bar, ("box", "hist", "scatter"), self._refresh, bg)
        self.chooser.pack(side=tk.LEFT, padx=4)
        self._frame = tk.Frame(self, bg=COLORS["bg_card"])
        self._frame.pack(fill=tk.BOTH, expand=True)

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
        w = max(self._frame.winfo_width(), 400) / 96
        h = max(self._frame.winfo_height(), 160) / 96
        fig = Figure(figsize=(w, h), dpi=96)
        if kind == "scatter":
            if len(cols) < 2:
                _message(self._frame, t("chart.pick_xy"))
                return
            pairs = [(float(r[cols[0]]), float(r[cols[1]])) for r in self._rows
                     if isinstance(r.get(cols[0]), (int, float)) and isinstance(r.get(cols[1]), (int, float))]
            if not pairs:
                _message(self._frame, t("chart.no_data_image"))
                return
            xs, ys = zip(*pairs)
            _scatter(fig, xs, ys, cols[0], cols[1])
        else:
            for i, col in enumerate(cols):
                ax = fig.add_subplot(1, len(cols), i + 1)
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
                    ax.scatter(1 + rng.uniform(-0.15, 0.15, len(vals)), vals, alpha=0.4, s=10, color=color, zorder=3)
                    ax.set_xticks([])
                else:
                    ax.hist(vals, bins=min(20, max(5, len(vals) // 3)), color=color, edgecolor="white",
                            alpha=0.85, linewidth=0.5)
                    ax.set_ylabel(t("chart.frequency"))
                ax.set_title(f"{_label(col)}\nn = {len(vals)}   {t('chart.median')} = {np.median(vals):.2f}"
                             f"   {t('chart.sd')} = {np.std(vals, ddof=1) if len(vals) > 1 else 0:.2f}",
                             fontsize=8)
                _style_ax(ax)
        try:
            fig.tight_layout(pad=0.8)
        except Exception:
            _log.debug("tight_layout", exc_info=True)
        self._fig = fig
        _embed(self._frame, fig)

    def save_figure(self):
        _save(self._fig, self)


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


class BatchChartWindow(tk.Toplevel):
    """Comparación entre fotos: un valor por foto (o la media de sus objetos)."""

    def __init__(self, parent, results: dict):
        super().__init__(parent)
        self.title(t("chart.batch_title"))
        self.configure(bg=COLORS["bg_card"])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1100, int(sw * 0.85)), min(700, int(sh * 0.82))
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self._fig = None
        self.names, self.means, self.sds = batch_table(results)
        cols = []
        for row in self.means.values():
            cols += [k for k in row if k not in cols]
        if not MPL_OK or not self.names:
            _message(self, t("chart.no_mpl") if not MPL_OK else t("chart.no_data_batch"))
            return
        tk.Frame(self, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        bg = COLORS["bg_panel"]
        bar = tk.Frame(self, bg=bg)
        bar.pack(fill=tk.X)
        self.chooser = _Chooser(bar, ("bars", "box", "hist", "scatter"), self._refresh, bg)
        self.chooser.pack(side=tk.LEFT, padx=8, pady=4)
        self.show_sd = tk.BooleanVar(value=False)
        self._sd_check = tk.Checkbutton(self.chooser.extra, text=t("chart.show_sd"), variable=self.show_sd,
                                        command=self._refresh, bg=bg, fg=COLORS["text"],
                                        selectcolor=COLORS["bg_card"], activebackground=bg, font=FONTS["small"])
        tk.Button(bar, text=t("chart.save"), command=lambda: _save(self._fig, self), bg=COLORS["btn_bg"],
                  fg=COLORS["accent"], relief="flat", font=FONTS["small"], cursor="hand2").pack(side=tk.RIGHT, padx=8)
        self._frame = tk.Frame(self, bg=COLORS["bg_card"])
        self._frame.pack(fill=tk.BOTH, expand=True)
        self.chooser.set_columns(cols)
        self.after(50, self._refresh)

    def _refresh(self):
        kind = self.chooser.kind.get()
        cols = self.chooser.selected()
        has_sd = kind == "bars" and any(self.sds[n].get(c) for n in self.names for c in cols)
        if has_sd:
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
        if kind == "scatter":
            if len(cols) < 2:
                _message(self._frame, t("chart.pick_xy"))
                return
            pts = [(self.means[n][cols[0]], self.means[n][cols[1]], n) for n in self.names
                   if cols[0] in self.means[n] and cols[1] in self.means[n]]
            if not pts:
                _message(self._frame, t("chart.no_data_batch"))
                return
            xs, ys, labels = zip(*pts)
            _scatter(fig, xs, ys, cols[0], cols[1], labels)
        else:
            for i, col in enumerate(cols):
                ax = fig.add_subplot(1, len(cols), i + 1)
                color = PALETTE[i % len(PALETTE)]
                names = [n for n in self.names if col in self.means[n]]
                vals = [self.means[n][col] for n in names]
                if not vals:
                    continue
                if kind == "bars":
                    x = np.arange(len(names))
                    ax.bar(x, vals, color=color, alpha=0.85, width=0.65, edgecolor="white", linewidth=0.5)
                    if self.show_sd.get() and has_sd:
                        ax.errorbar(x, vals, yerr=[self.sds[n].get(col, 0) for n in names], fmt="none",
                                    color="#333333", capsize=2, linewidth=0.8)
                    ax.set_xticks(x)
                    ax.set_xticklabels(names if len(names) <= 60 else [""] * len(names),
                                       rotation=60, ha="right", fontsize=6)
                elif kind == "box":
                    bp = ax.boxplot([vals], patch_artist=True, widths=0.5,
                                    medianprops=dict(color="#333333", linewidth=1.5))
                    bp["boxes"][0].set_facecolor(color + "55")
                    bp["boxes"][0].set_edgecolor(color)
                    rng = np.random.default_rng(0)
                    ax.scatter(1 + rng.uniform(-0.12, 0.12, len(vals)), vals, color=color, alpha=0.7, s=14, zorder=3)
                    ax.set_xticks([])
                else:
                    ax.hist(vals, bins=min(20, max(5, len(vals) // 2)), color=color, alpha=0.85,
                            edgecolor="white", linewidth=0.5)
                    ax.set_ylabel(t("chart.frequency"))
                ax.set_title(f"{_label(col)}\n{t('chart.n_images', n=len(vals))}", fontsize=8)
                _style_ax(ax)
        try:
            fig.tight_layout(pad=1.0)
        except Exception:
            _log.debug("tight_layout", exc_info=True)
        self._fig = fig
        _embed(self._frame, fig, toolbar=True)
