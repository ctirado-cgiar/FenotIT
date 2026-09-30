"""
ui/charts.py
Sistema de gráficos de FenotIT.

IntraImageChart  — gráficos de distribución de una sola imagen
BatchChart       — gráficos comparativos entre imágenes del lote

Dependencias: matplotlib (ya en el entorno del pipeline)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any
import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    from matplotlib.figure import Figure
    MPL_OK = True
except ImportError:
    MPL_OK = False

from fenotit.gui.theme import COLORS, FONTS

from fenotit import log

_log = log.get("gui.charts")


def _assets() -> Path:
    return Path(__file__).parent.parent / "assets"


# ── Paleta de colores para gráficos (científica, tipo R) ─────────────────────

PALETTE = [
    "#2166AC", "#D73027", "#1A9850", "#8073AC",
    "#E08214", "#01665E", "#762A83", "#4DAC26",
    "#F4A582", "#92C5DE", "#A6D96A", "#FDAE61",
]

MPL_STYLE = {
    "axes.facecolor":     "#FAFAFA",
    "figure.facecolor":   "#FFFFFF",
    "axes.edgecolor":     "#CCCCCC",
    "axes.labelcolor":    "#333333",
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.grid":          True,
    "grid.color":         "#EEEEEE",
    "grid.linestyle":     "-",
    "grid.linewidth":     0.6,
    "xtick.color":        "#555555",
    "ytick.color":        "#555555",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.85,
    "figure.dpi":         100,
}


def _apply_style():
    if MPL_OK:
        plt.rcParams.update(MPL_STYLE)


# ── Utilidades ────────────────────────────────────────────────────────────────

def _numeric_cols(measurements: list[dict]) -> list[str]:
    """Retorna columnas numéricas de una lista de mediciones."""
    if not measurements:
        return []
    seen = set()
    cols = []
    skip = {"index", "_total", "centroid_x", "centroid_y",
            "centroid_x (display)", "centroid_y (display)",
            "object_id", "touching", "centroid_x_px", "centroid_y_px", "Image_ID"}
    for row in measurements:
        for k, v in row.items():
            if k not in seen and k not in skip                     and isinstance(v, (int, float)):
                seen.add(k)
                cols.append(k)
    return cols


def _merge_all_analyses(results_by_analysis: dict) -> dict:
    """
    Combina resultados de múltiples análisis por imagen.
    results_by_analysis: {analysis_name: {stem: result}}
    Retorna: {stem: {col: value, ...}} con todas las columnas disponibles.
    
    Para morfometría (N filas por imagen) se usan medias.
    Para color y conteo (1 fila por imagen) se usan directamente.
    """
    merged: dict[str, dict] = {}

    for analysis_name, per_image in results_by_analysis.items():
        for stem, result in per_image.items():
            if stem not in merged:
                merged[stem] = {"imagen": stem}

            measurements = result.measurements if hasattr(result, "measurements")                            else result.get("measurements", [])
            stats        = result.stats if hasattr(result, "stats")                            else result.get("stats", {})

            if not measurements:
                # Solo stats (conteo)
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        merged[stem][k] = v
                continue

            # Si hay múltiples filas → calcular media
            if len(measurements) > 1:
                num_keys = _numeric_cols(measurements)
                for k in num_keys:
                    vals = [m[k] for m in measurements
                            if isinstance(m.get(k), (int, float))]
                    if vals:
                        clean = k.split(" (")[0]
                        merged[stem][f"{clean}_mean"] = round(
                            sum(vals)/len(vals), 4)
            else:
                # Una sola fila → copiar directamente
                for k, v in measurements[0].items():
                    if isinstance(v, (int, float)) and k not in                             ("index", "_total"):
                        merged[stem][k] = v

    return merged


def _clean_col_name(col: str) -> str:
    """Quita unidades del nombre de columna para el eje."""
    return col.split(" (")[0].replace("_", " ").strip()


# ── Panel de gráficos intra-imagen ────────────────────────────────────────────

class IntraImageChartPanel(tk.Frame):
    """
    Panel embebido debajo de la tabla de resultados.
    Muestra hasta 3 gráficos (boxplot/histograma) de la imagen actual.
    El usuario puede cambiar qué variable muestra cada gráfico.
    """

    DEFAULT_VARS = ["área", "largo", "ancho"]   # preferidas por defecto

    def __init__(self, parent, colors: dict, **kw):
        super().__init__(parent, bg=colors["bg_card"], **kw)
        self.colors = colors
        self._measurements: list[dict] = []
        self._available_cols: list[str] = []
        self._var_selectors: list[ttk.Combobox] = []
        self._chart_type_var = tk.StringVar(value="boxplot")
        self._fig = None
        self._canvas_widget = None

        if not MPL_OK:
            tk.Label(self, text="matplotlib no disponible",
                     bg=colors["bg_card"], fg=colors["text_muted"],
                     font=FONTS["small"]).pack(pady=8)
            return

        self._build()

    def _build(self):
        # Barra de controles
        ctrl = tk.Frame(self, bg=self.colors["bg_panel"])
        ctrl.pack(fill=tk.X)
        tk.Frame(ctrl, bg=self.colors["border"],
                 height=1).pack(fill=tk.X, side=tk.TOP)

        tk.Label(ctrl, text="Gráficos:",
                 bg=self.colors["bg_panel"],
                 fg=self.colors["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=8, pady=4)

        # Tipo de gráfico
        for val, lbl in [("boxplot","Boxplot"),
                         ("histograma","Histograma")]:
            tk.Radiobutton(ctrl, text=lbl,
                           variable=self._chart_type_var,
                           value=val,
                           command=self._refresh,
                           bg=self.colors["bg_panel"],
                           fg=self.colors["text"],
                           selectcolor=self.colors["bg_card"],
                           activebackground=self.colors["bg_panel"],
                           font=FONTS["small"]).pack(
                               side=tk.LEFT, padx=2)

        tk.Frame(ctrl, bg=self.colors["border"],
                 width=1).pack(side=tk.LEFT, fill=tk.Y,
                               pady=4, padx=6)

        # Selectores de variable (3)
        self._selected_vars = [tk.StringVar() for _ in range(3)]
        for i, sv in enumerate(self._selected_vars):
            tk.Label(ctrl, text=f"V{i+1}:",
                     bg=self.colors["bg_panel"],
                     fg=self.colors["text_muted"],
                     font=FONTS["small"]).pack(side=tk.LEFT, padx=(4,1))
            cb = ttk.Combobox(ctrl, textvariable=sv,
                              state="readonly", width=10,
                              font=FONTS["small"])
            cb.pack(side=tk.LEFT, padx=2, pady=3)
            cb.bind("<<ComboboxSelected>>",
                    lambda e=None: self._refresh())
            self._var_selectors.append(cb)

        # Área del gráfico
        self._chart_frame = tk.Frame(self, bg=self.colors["bg_card"])
        self._chart_frame.pack(fill=tk.BOTH, expand=True)

    def load(self, measurements: list[dict]):
        """Carga nuevas mediciones y actualiza los selectores."""
        self._measurements = measurements
        self._available_cols = _numeric_cols(measurements)

        if not self._available_cols or not MPL_OK:
            return

        # Actualizar combos
        for cb in self._var_selectors:
            cb["values"] = self._available_cols

        # Asignar defaults
        defaults = self._pick_defaults()
        for i, sv in enumerate(self._selected_vars):
            if i < len(defaults):
                sv.set(defaults[i])

        self._refresh()

    def _pick_defaults(self) -> list[str]:
        """Elige las 3 variables por defecto."""
        result = []
        # Buscar por nombre preferido
        for pref in self.DEFAULT_VARS:
            for col in self._available_cols:
                if pref in col.lower() and col not in result:
                    result.append(col)
                    break
        # Completar con lo que haya
        for col in self._available_cols:
            if col not in result:
                result.append(col)
            if len(result) == 3:
                break
        return result

    def _refresh(self):
        if not self._measurements or not MPL_OK:
            return

        # Limpiar frame anterior
        for w in self._chart_frame.winfo_children():
            w.destroy()

        # Solo columnas seleccionadas y no vacías
        cols = [sv.get() for sv in self._selected_vars
                if sv.get() and sv.get() in self._available_cols]
        if not cols:
            tk.Label(self._chart_frame,
                     text="Selecciona al menos una variable.",
                     bg=self.colors["bg_card"],
                     fg=self.colors["text_muted"],
                     font=FONTS["small"]).pack(pady=20)
            return

        _apply_style()
        n = len(cols)
        fig = Figure(figsize=(max(4*n, 8), 2.8), dpi=96)
        fig.patch.set_facecolor("#FFFFFF")

        chart_type = self._chart_type_var.get()

        for i, col in enumerate(cols):
            ax = fig.add_subplot(1, n, i + 1)
            ax.set_facecolor("#FAFAFA")
            vals = [m[col] for m in self._measurements
                    if isinstance(m.get(col), (int, float))]
            if not vals:
                continue

            color = PALETTE[i % len(PALETTE)]

            if chart_type == "boxplot":
                bp = ax.boxplot(vals, patch_artist=True,
                                widths=0.5,
                                medianprops=dict(color="#333333",
                                                 linewidth=1.5))
                bp["boxes"][0].set_facecolor(color + "66")
                bp["boxes"][0].set_edgecolor(color)
                ax.set_xticks([])
                # Agregar puntos individuales (jitter)
                jitter = np.random.uniform(-0.15, 0.15, len(vals))
                ax.scatter(1 + jitter, vals,
                           alpha=0.4, s=12,
                           color=color, zorder=3)
            else:
                ax.hist(vals, bins=min(20, max(5, len(vals)//3)),
                        color=color, edgecolor="white",
                        alpha=0.85, linewidth=0.5)
                ax.set_xlabel(_clean_col_name(col), fontsize=8)

            ax.set_title(_clean_col_name(col), fontsize=9,
                         fontweight="bold", pad=4)
            ax.grid(True, alpha=0.4, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")
                spine.set_linewidth(0.5)

            # Stats en el título
            if vals:
                med = np.median(vals)
                std = np.std(vals)
                ax.set_title(
                    f"{_clean_col_name(col)}\n"
                    f"n={len(vals)}  med={med:.2f}  σ={std:.2f}",
                    fontsize=8, fontweight="bold", pad=4)

        fig.tight_layout(pad=0.8)
        self._fig = fig

        canvas = FigureCanvasTkAgg(fig, master=self._chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas_widget = canvas

    def save_figure(self):
        """Guarda la figura actual."""
        if self._fig is None:
            return
        path = filedialog.asksaveasfilename(
            title="Guardar gráfico",
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("SVG","*.svg"),
                       ("PDF","*.pdf")])
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")


# ── Ventana de gráficos del lote ──────────────────────────────────────────────

class BatchChartWindow(tk.Toplevel):
    """
    Ventana flotante para gráficos comparativos entre imágenes del lote.
    """

    def __init__(self, parent,
                 results_by_analysis: dict,
                 active_analysis: str = ""):
        """
        results_by_analysis: {analysis_name: {path: AnalysisResult}}
        Combina datos de todos los análisis ejecutados por imagen.
        """
        super().__init__(parent)
        self.title("Gráficos del lote")
        self.configure(bg=COLORS["bg_card"])
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w, h = min(1100, int(sw*0.85)), min(700, int(sh*0.82))
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        try:
            self.iconbitmap(str(_assets()/"logo.ico"))
        except Exception:
            _log.debug("ignorado", exc_info=True)

        self._fig = None
        self._all_cols: list[str] = []

        # Construir {analysis: {stem: result}} para merge
        by_analysis_stem = {}
        for aname, cache in results_by_analysis.items():
            by_analysis_stem[aname] = {}
            for path, result in cache.items():
                if result.status != "ok":
                    continue
                stem = Path(path).stem
                if stem.lower().startswith("in_row="):
                    stem = stem[7:]
                by_analysis_stem[aname][stem] = result

        # Merge de todos los análisis por imagen
        merged = _merge_all_analyses(by_analysis_stem)
        self._merged = merged   # {stem: {col: val}}

        # Para gráficos que necesitan todos los objetos por imagen
        # usar el análisis con más mediciones (típicamente morfometría)
        self._per_image: dict[str, list[dict]] = {}
        for aname, stem_map in by_analysis_stem.items():
            for stem, result in stem_map.items():
                if result.measurements and                         len(result.measurements) >                         len(self._per_image.get(stem, [])):
                    self._per_image[stem] = result.measurements
                elif stem not in self._per_image and result.measurements:
                    self._per_image[stem] = result.measurements

        if not merged:
            tk.Label(self, text="Sin datos para graficar.",
                     bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                     font=FONTS["body"]).pack(pady=40)
            return

        # Columnas disponibles — todas de todos los análisis
        all_cols_set = set()
        for row in merged.values():
            for k, v in row.items():
                if isinstance(v, (int, float)) and k != "imagen":
                    all_cols_set.add(k)
        self._all_cols = sorted(all_cols_set)
        # Poner métricas de área/largo/ancho primero
        priority = ["area", "largo", "ancho", "ConteoTotal",
                    "total semillas"]
        front = [c for p in priority
                 for c in self._all_cols if p in c.lower()]
        rest  = [c for c in self._all_cols if c not in front]
        self._all_cols = front + rest

        if not MPL_OK:
            tk.Label(self,
                     text="matplotlib no disponible — "
                          "pip install matplotlib",
                     bg=COLORS["bg_card"], fg=COLORS["warning"],
                     font=FONTS["body"]).pack(pady=40)
            return

        self._build()

    def _build(self):
        tk.Frame(self, bg=COLORS["accent"],
                 height=4).pack(fill=tk.X)

        # Panel de controles
        ctrl_outer = tk.Frame(self, bg=COLORS["bg_panel"])
        ctrl_outer.pack(fill=tk.X)
        tk.Frame(ctrl_outer, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, side=tk.BOTTOM)

        ctrl = tk.Frame(ctrl_outer, bg=COLORS["bg_panel"])
        ctrl.pack(fill=tk.X, padx=12, pady=6)

        # Tipo de gráfico
        tk.Label(ctrl, text="Tipo:",
                 bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(0,4))

        self._chart_type = tk.StringVar(value="barras")
        for val, lbl in [("barras","Barras"),
                         ("boxplot","Boxplot"),
                         ("histograma","Histograma"),
                         ("dispersion","Dispersión X vs Y")]:
            tk.Radiobutton(ctrl, text=lbl,
                           variable=self._chart_type,
                           value=val,
                           command=self._refresh,
                           bg=COLORS["bg_panel"],
                           fg=COLORS["text"],
                           selectcolor=COLORS["bg_card"],
                           activebackground=COLORS["bg_panel"],
                           font=FONTS["small"]).pack(
                               side=tk.LEFT, padx=3)

        tk.Frame(ctrl, bg=COLORS["border"],
                 width=1).pack(side=tk.LEFT, fill=tk.Y,
                               pady=2, padx=8)

        # Variables
        self._vars = [tk.StringVar() for _ in range(3)]
        defaults = self._pick_defaults()
        for i, sv in enumerate(self._vars):
            lbl = "X:" if i == 0 else ("Y:" if i == 1 else "V3:")
            tk.Label(ctrl, text=lbl,
                     bg=COLORS["bg_panel"],
                     fg=COLORS["text_muted"],
                     font=FONTS["small"]).pack(side=tk.LEFT, padx=(6,1))
            cb = ttk.Combobox(ctrl, textvariable=sv,
                              values=self._all_cols,
                              state="readonly", width=12,
                              font=FONTS["small"])
            if i < len(defaults):
                sv.set(defaults[i])
            cb.pack(side=tk.LEFT, padx=2)
            cb.bind("<<ComboboxSelected>>",
                    lambda e=None: self._refresh())

        tk.Frame(ctrl, bg=COLORS["border"],
                 width=1).pack(side=tk.LEFT, fill=tk.Y,
                               pady=2, padx=8)

        # Desviación estándar
        self._show_std = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="± desv. estándar",
                       variable=self._show_std,
                       command=self._refresh,
                       bg=COLORS["bg_panel"],
                       fg=COLORS["text"],
                       selectcolor=COLORS["bg_card"],
                       activebackground=COLORS["bg_panel"],
                       font=FONTS["small"]).pack(side=tk.LEFT, padx=4)

        # Botón guardar
        tk.Button(ctrl, text="💾 Guardar gráfico",
                  command=self._save,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["small"],
                  cursor="hand2").pack(side=tk.RIGHT, padx=4)

        # Área del gráfico
        self._chart_frame = tk.Frame(self, bg=COLORS["bg_card"])
        self._chart_frame.pack(fill=tk.BOTH, expand=True)

        self._refresh()

    def _pick_defaults(self) -> list[str]:
        result = []
        for pref in ["área", "largo", "ancho"]:
            for col in self._all_cols:
                if pref in col.lower() and col not in result:
                    result.append(col)
                    break
        for col in self._all_cols:
            if col not in result:
                result.append(col)
            if len(result) == 3:
                break
        return result

    def _refresh(self):
        for w in self._chart_frame.winfo_children():
            w.destroy()

        chart_type = self._chart_type.get()
        # Filtrar variables vacías
        var_vals   = [sv.get() for sv in self._vars]
        show_std   = self._show_std.get()
        images     = sorted(self._merged.keys())

        if not images:
            return

        _apply_style()

        if chart_type == "dispersion":
            self._draw_scatter(var_vals, images)
        elif chart_type == "barras":
            self._draw_bars(var_vals, images, show_std)
        elif chart_type == "boxplot":
            self._draw_boxplot(var_vals, images)
        elif chart_type == "histograma":
            self._draw_histogram(var_vals, images)

    # ── Tipos de gráfico ──────────────────────────────────────────────────────

    def _make_fig(self, ncols=1, nrows=1):
        fig_w = max(8, len(self._per_image) * 0.4 + 2)
        fig = Figure(figsize=(min(fig_w*ncols, 16), 4.5*nrows),
                     dpi=96)
        fig.patch.set_facecolor("#FFFFFF")
        return fig

    def _embed(self, fig):
        self._fig = fig
        canvas = FigureCanvasTkAgg(
            fig, master=self._chart_frame)
        canvas.draw()
        # Toolbar de navegación de matplotlib
        toolbar_f = tk.Frame(self._chart_frame,
                             bg=COLORS["bg_card"])
        toolbar_f.pack(fill=tk.X, side=tk.BOTTOM)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_f)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _draw_bars(self, var_vals, images, show_std):
        active = [v for v in var_vals
                  if v and v in self._all_cols]
        if not active:
            return
        n_vars = len(active)
        fig    = self._make_fig(ncols=n_vars)

        for vi, col in enumerate(active):
            ax = fig.add_subplot(1, n_vars, vi + 1)
            means, stds = [], []
            for img in images:
                # Usar merged primero (datos de cualquier análisis)
                merged_val = self._merged.get(img, {}).get(col)
                if merged_val is not None:
                    means.append(float(merged_val))
                    stds.append(0)
                else:
                    # Fallback: calcular de mediciones individuales
                    vals = [m[col] for m in self._per_image.get(img,[])
                            if isinstance(m.get(col), (int, float))]
                    means.append(np.mean(vals) if vals else 0)
                    stds.append(np.std(vals)  if vals else 0)

            x     = np.arange(len(images))
            color = PALETTE[vi % len(PALETTE)]
            bars  = ax.bar(x, means, color=color,
                           alpha=0.8, width=0.6,
                           edgecolor="white", linewidth=0.5)
            if show_std:
                ax.errorbar(x, means, yerr=stds,
                            fmt="none", color="#333333",
                            capsize=3, linewidth=1)

            ax.set_xticks(x)
            ax.set_xticklabels(images, rotation=45,
                               ha="right", fontsize=7)
            ax.set_title(_clean_col_name(col),
                         fontsize=9, fontweight="bold")
            ax.set_ylabel(_clean_col_name(col), fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")
                spine.set_linewidth(0.5)

        fig.tight_layout(pad=1.0)
        self._embed(fig)

    def _draw_boxplot(self, var_vals, images):
        # Para boxplot entre imágenes: cada imagen es UN punto (su media)
        # Usamos _merged que tiene una fila por imagen con medias de todos los análisis
        active = [v for v in var_vals
                  if v and v in self._all_cols]
        if not active:
            return
        n_vars = len(active)
        fig    = self._make_fig(ncols=n_vars)

        for vi, col in enumerate(active):
            ax     = fig.add_subplot(1, n_vars, vi + 1)
            color  = PALETTE[vi % len(PALETTE)]
            # Un valor por imagen
            vals = [float(self._merged[img][col])
                    for img in images
                    if isinstance(self._merged.get(img,{}).get(col),
                                  (int,float))]
            valid_imgs = [img for img in images
                          if isinstance(self._merged.get(img,{}).get(col),
                                        (int,float))]
            if not vals:
                ax.set_title(f"{_clean_col_name(col)} (sin datos)", fontsize=9)
                continue

            bp = ax.boxplot([vals], patch_artist=True,
                            medianprops=dict(color="#333333",
                                             linewidth=1.5),
                            widths=0.5)
            bp["boxes"][0].set_facecolor(color + "55")
            bp["boxes"][0].set_edgecolor(color)

            # Puntos individuales (cada imagen)
            jitter = np.random.uniform(-0.12, 0.12, len(vals))
            ax.scatter(1 + jitter, vals, color=color,
                       alpha=0.7, s=20, zorder=3)
            ax.set_xticks([1])
            ax.set_xticklabels([_clean_col_name(col)], fontsize=8)
            ax.set_title(
                f"{_clean_col_name(col)}  n={len(vals)}",
                fontsize=9, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")
                spine.set_linewidth(0.5)

        fig.tight_layout(pad=1.0)
        self._embed(fig)

    def _draw_histogram(self, var_vals, images):
        # Histograma: distribución de valores por imagen
        # Cada imagen aporta un valor (de _merged) → histograma de valores entre imágenes
        active = [v for v in var_vals
                  if v and v in self._all_cols]
        if not active:
            return
        n_vars = len(active)
        fig    = self._make_fig(ncols=n_vars)

        for vi, col in enumerate(active):
            ax    = fig.add_subplot(1, n_vars, vi + 1)
            color = PALETTE[vi % len(PALETTE)]
            vals  = [float(self._merged[img][col])
                     for img in images
                     if isinstance(self._merged.get(img,{}).get(col),
                                   (int,float))]
            if not vals:
                ax.set_title(f"{_clean_col_name(col)} (sin datos)", fontsize=9)
                continue
            bins = min(20, max(5, len(vals)//2))
            ax.hist(vals, bins=bins, color=color,
                    alpha=0.8, edgecolor="white", linewidth=0.5)
            ax.set_title(
                f"{_clean_col_name(col)}  n={len(vals)}",
                fontsize=9, fontweight="bold")
            ax.set_xlabel(_clean_col_name(col), fontsize=8)
            ax.set_ylabel("Frecuencia", fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")
                spine.set_linewidth(0.5)

        fig.tight_layout(pad=1.0)
        self._embed(fig)

    def _draw_scatter(self, var_vals, images):
        """
        Dispersión X vs Y.
        Cada imagen es un punto (usa valores de _merged = medias).
        """
        x_col = var_vals[0] if len(var_vals) > 0 else ""
        y_col = var_vals[1] if len(var_vals) > 1 else ""

        if not x_col or not y_col or                 x_col not in self._all_cols or                 y_col not in self._all_cols:
            tk.Label(self._chart_frame,
                     text="Selecciona V1 (eje X) y V2 (eje Y)\n"
                          "para el gráfico de dispersión.",
                     bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                     font=FONTS["body"], justify="center").pack(pady=40)
            return

        fig = self._make_fig()
        ax  = fig.add_subplot(1, 1, 1)

        xs, ys, labels = [], [], []
        for img in images:
            row = self._merged.get(img, {})
            xv  = row.get(x_col)
            yv  = row.get(y_col)
            if isinstance(xv, (int,float)) and isinstance(yv, (int,float)):
                xs.append(float(xv))
                ys.append(float(yv))
                labels.append(img)

        if not xs:
            tk.Label(self._chart_frame,
                     text="Sin datos para las variables seleccionadas.",
                     bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                     font=FONTS["body"]).pack(pady=40)
            return

        colors = [PALETTE[i % len(PALETTE)] for i in range(len(xs))]
        ax.scatter(xs, ys, c=colors, alpha=0.8, s=60,
                   edgecolors="white", linewidths=0.5, zorder=3)

        # Etiquetas de imagen en cada punto
        if len(xs) <= 20:
            for xi, yi, lbl in zip(xs, ys, labels):
                ax.annotate(lbl, (xi, yi),
                            textcoords="offset points",
                            xytext=(4, 4),
                            fontsize=6, color="#555555")

        ax.set_xlabel(_clean_col_name(x_col), fontsize=9)
        ax.set_ylabel(_clean_col_name(y_col), fontsize=9)
        ax.set_title(
            f"{_clean_col_name(x_col)} vs {_clean_col_name(y_col)}\n"
            f"n={len(xs)} imágenes",
            fontsize=10, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")
            spine.set_linewidth(0.5)

        fig.tight_layout(pad=1.0)
        self._embed(fig)

    def _save(self):
        if self._fig is None:
            return
        path = filedialog.asksaveasfilename(
            title="Guardar gráfico",
            defaultextension=".png",
            filetypes=[("PNG","*.png"),
                       ("SVG","*.svg"),
                       ("PDF","*.pdf")],
            parent=self)
        if path:
            try:
                self._fig.savefig(path, dpi=150,
                                  bbox_inches="tight")
                messagebox.showinfo(
                    "Guardado", f"Gráfico guardado:\n{path}",
                    parent=self)
            except Exception as e:
                _log.exception("Error guardando gráfico")
                messagebox.showerror("Error", str(e),
                                     parent=self)