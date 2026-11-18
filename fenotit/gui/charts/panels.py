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

# log escribe los mensajes y errores en el archivo de registro.
from fenotit import log
# natural_key ordena los nombres de archivo como los ordena una persona (2 antes de 10).
from fenotit.core.image_io import natural_key
# HelpIcon es el «?» que explica una opción al pasar el mouse o al hacer clic.
from fenotit.gui.widgets.help import HelpIcon
# button crea un botón con el estilo del tema; divider dibuja una línea separadora.
from fenotit.gui.style.components import button, divider
# Valores del tema (fenotit/assets/themes): CHART_COLORS (colores de las series de los gráficos);
# COLORS (colores); FONTS (letras); SPACE (márgenes).
from fenotit.gui.style.theme import CHART_COLORS, COLORS, FONTS, SPACE
# t devuelve cada texto de la ventana en el idioma elegido (fenotit/lang).
from fenotit.i18n import t

_log = log.get("gui.charts")

PALETTE = list(CHART_COLORS)

MPL_STYLE = {
    "axes.facecolor": COLORS["bg_chart"], "figure.facecolor": COLORS["white"], "axes.edgecolor": COLORS["border"],
    "axes.labelcolor": COLORS["text"], "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.grid": True, "grid.color": COLORS["grid"],
    "grid.linewidth": 0.6, "xtick.color": COLORS["text_muted"], "ytick.color": COLORS["text_muted"],
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 96,
}

NONE = "—"
SKIP = {"index", "_total", "object_id", "touching", "Image_ID", "centroid_x", "centroid_y",
        "centroid_x_px", "centroid_y_px", "image_width_px", "image_height_px"}
PREFERRED = ("area", "length", "width")

# tipo → (variables al empezar, máximo, nombres de los ejes). Se agregan con "+"; en
# dispersión la tercera (Z) la vuelve 3D.
KINDS = {
    "bars":    (1, 3, None),
    "box":     (1, 3, None),
    "hist":    (1, 3, None),
    "scatter": (2, 3, ("X", "Y", "Z")),
}


def numeric_cols(rows: list[dict]) -> list[str]:
    cols = []
    for row in rows:
        for key, value in row.items():
            if key not in cols and key not in SKIP and not key.endswith("_per_px") \
                    and isinstance(value, (int, float)) and not isinstance(value, bool):
                cols.append(key)
    return cols


def _defaults(cols: list[str], count: int) -> list[str]:
    out = [column for prefix in PREFERRED for column in cols if column.lower().startswith(prefix)][:count]
    out = list(dict.fromkeys(out))
    out += [column for column in cols if column not in out][:count - len(out)]
    return out


def _values(rows: list[dict], col: str) -> list[float]:
    return [float(row[col]) for row in rows if isinstance(row.get(col), (int, float)) and row[col] == row[col]]


def _style_ax(axes):
    for spine in axes.spines.values():
        spine.set_edgecolor(COLORS["border"])
        spine.set_linewidth(0.5)


def _num(value: float) -> str:
    """Sin notación científica: enteros desde 1000, 4 cifras significativas por debajo."""
    return f"{value:.0f}" if abs(value) >= 1000 else f"{value:.4g}"


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
        self.count: dict[str, int] = {}             # cuántas variables muestra cada tipo
        for kind in kinds:
            tk.Radiobutton(self, text=t(f"chart.{kind}"), variable=self.kind, value=kind, command=self._kind_changed,
                           bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg,
                           font=FONTS["small"]).pack(side=tk.LEFT, padx=SPACE["xs"])
        divider(self, vertical=True).pack(side=tk.LEFT, fill=tk.Y, pady=SPACE["sm"], padx=SPACE["md"])
        self.vars_frame = tk.Frame(owner or self, bg=bg)      # con owner puede bajar a otra fila
        self.vars_frame.pack(in_=self, side=tk.LEFT)
        self.extra = tk.Frame(self, bg=bg)          # opciones solo de algunos tipos
        self.extra.pack(side=tk.LEFT, padx=(SPACE["md"], 0))
        self._svars: list[tk.StringVar] = []

    def set_columns(self, cols: list[str]):
        if cols != self.cols:
            self.cols = cols
            self.chosen = {key: [column for column in value if column in cols] for key, value in self.chosen.items()}
        self._build_vars()

    def selected(self) -> list[str]:
        return [variable.get() for variable in self._svars if variable.get() in self.cols]

    def _kind_changed(self):
        self._build_vars()
        self.on_change()

    def _build_vars(self):
        for widget in self.vars_frame.winfo_children():
            widget.destroy()
        kind = self.kind.get()
        base, most, names = KINDS[kind]
        count = min(max(self.count.get(kind, base), base), most, max(len(self.cols), 1))
        current = [column for column in (self.chosen.get(kind) or []) if column in self.cols][:count]
        current += [column for column in _defaults(self.cols, most + len(current)) if column not in current][:count - len(current)]
        self._svars = []
        for i in range(count):
            name = names[i] if names else f"V{i + 1}"
            tk.Label(self.vars_frame, text=f"{name}:", bg=self.bg, fg=COLORS["text_muted"],
                     font=FONTS["small"]).pack(side=tk.LEFT, padx=(SPACE["sm"], 1))
            string_var = tk.StringVar(value=current[i] if i < len(current) else NONE)
            combo = ttk.Combobox(self.vars_frame, textvariable=string_var, state="readonly", width=14, font=FONTS["small"],
                              values=self.cols)
            combo.pack(side=tk.LEFT, padx=SPACE["xs"], pady=SPACE["sm"])
            combo.bind("<<ComboboxSelected>>", lambda event: self._vars_changed())
            self._svars.append(string_var)
        from fenotit.gui.widgets.toolbar import Tooltip
        btn = dict(kind="secondary", size="small", font=FONTS["topbar"], padx=SPACE["md"], pady=0)
        if count > base:
            button_widget = button(self.vars_frame, text="−", command=lambda: self._resize(-1), **btn)
            button_widget.pack(side=tk.LEFT, padx=(SPACE["xs"], 0))
            Tooltip(button_widget, t("chart.remove_var"))
        if count < min(most, len(self.cols)):
            button_widget = button(self.vars_frame, text="+", command=lambda: self._resize(1), **btn)
            button_widget.pack(side=tk.LEFT, padx=(SPACE["xs"], 0))
            Tooltip(button_widget, t("chart.add_z") if kind == "scatter" else t("chart.add_var"))

    def _resize(self, delta: int):
        kind = self.kind.get()
        self.chosen[kind] = [variable.get() for variable in self._svars]
        self.count[kind] = len(self._svars) + delta
        self._build_vars()
        self.chosen[kind] = [variable.get() for variable in self._svars]
        self.on_change()

    def _vars_changed(self):
        self.chosen[self.kind.get()] = [variable.get() for variable in self._svars]
        self.on_change()


def _embed(frame, fig, toolbar=False):
    for widget in frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    if toolbar:
        bar = tk.Frame(frame, bg=COLORS["bg_card"])
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        NavigationToolbar2Tk(canvas, bar).update()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    if getattr(fig, "_rot3d", None) is not None:
        _rotatable(canvas, fig._rot3d)
    return canvas


def _message(frame, text):
    for widget in frame.winfo_children():
        widget.destroy()
    tk.Label(frame, text=text, bg=COLORS["bg_card"], fg=COLORS["text_muted"], font=FONTS["small"],
             justify="center").pack(pady=SPACE["xxl"])


class _Picker:
    """Clic (sin arrastrar) en un punto del gráfico → on_pick(dato de ese punto) y un
    anillo rojo lo marca."""

    def __init__(self):
        self.sets: list = []
        self._press = self._ring = None

    def add(self, axes, xs_values, ys_values, payloads):
        if len(payloads):
            self.sets.append((axes, np.c_[np.asarray(xs_values, float), np.asarray(ys_values, float)], list(payloads)))

    def connect(self, canvas, on_pick):
        if not self.sets or on_pick is None:
            return
        canvas.mpl_connect("button_press_event", lambda event: setattr(self, "_press", (event.x, event.y)))
        canvas.mpl_connect("button_release_event", lambda event: self._release(event, canvas, on_pick))

    def _release(self, event, canvas, on_pick):
        press = self._press
        if event.button != 1 or event.x is None or press is None or abs(event.x - press[0]) + abs(event.y - press[1]) > 4:
            return
        if getattr(getattr(canvas, "toolbar", None), "mode", ""):       # zoom/mover de la barra
            return
        best = None
        for axes, points, pick_data in self.sets:
            if event.inaxes is not axes:
                continue
            distance = np.hypot(*(axes.transData.transform(points) - (event.x, event.y)).T)
            i = int(np.argmin(distance))
            if distance[i] <= 7 and (best is None or distance[i] < best[0]):
                best = (distance[i], axes, points[i], pick_data[i])
        if best is None:
            return
        if self._ring is not None:
            try:
                self._ring.remove()
            except Exception:
                pass
        _, axes, (x, y), payload = best
        self._ring = axes.plot([x], [y], "o", mfc="none", mec=COLORS["highlight"], mew=1.6, ms=11, zorder=5)[0]
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


def _scatter(fig, xs_values, ys_values, x_col, y_col, labels=None):
    axes = fig.add_subplot(1, 1, 1)
    fig._scatter_ax = axes
    axes.scatter(xs_values, ys_values, color=PALETTE[0], alpha=0.75, s=28 if labels else 16, edgecolors=COLORS["white"],
               linewidths=0.5, zorder=3)
    if labels and len(xs_values) <= 30:
        for x, y, lbl in zip(xs_values, ys_values, labels):
            axes.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6, color=COLORS["text_muted"])
    axes.set_xlabel(_label(x_col))
    axes.set_ylabel(_label(y_col))
    correlation = np.corrcoef(xs_values, ys_values)[0, 1] if len(xs_values) > 2 and np.std(xs_values) > 0 and np.std(ys_values) > 0 else None
    axes.set_title(f"n = {len(xs_values)}" + (f"   r = {correlation:.2f}" if correlation is not None else ""))
    _style_ax(axes)


def _scatter3d(fig, series, cols, legend_title=None):
    """Dispersión 3D (X, Y, Z). series = [(nombre, xs, ys, zs, color)]. Se gira arrastrando."""
    axes = fig.add_subplot(1, 1, 1, projection="3d")
    count = 0
    for name, xs_values, ys_values, zs_values, color in series:
        axes.scatter(xs_values, ys_values, zs_values, color=color, s=18 if len(xs_values) < 300 else 8, alpha=0.8, depthshade=False,
                   edgecolors=COLORS["white"], linewidths=0.3, label=name)
        count += len(xs_values)
    axes.set_xlabel(_label(cols[0]))
    axes.set_ylabel(_label(cols[1]))
    axes.set_zlabel(_label(cols[2]))
    axes.set_title(f"n = {count}")
    if len(series) > 1:
        axes.legend(frameon=False, fontsize=7, title=legend_title, title_fontsize=7)
    axes.view_init(22, -60)
    axes.disable_mouse_rotation()
    fig._rot3d = axes
    return axes


def _rotatable(canvas, axes):
    """Girar un gráfico 3D arrastrando con el botón izquierdo (0.4° por píxel)."""
    state = {}

    def press(event):
        state["p"] = (event.x, event.y, axes.elev, axes.azim) if event.button == 1 else None

    def move(event):
        start = state.get("p")
        if start and event.x is not None:
            axes.view_init(float(np.clip(start[2] - (event.y - start[1]) * 0.4, -90, 90)), start[3] - (event.x - start[0]) * 0.4)
            canvas.draw_idle()
    canvas.mpl_connect("button_press_event", press)
    canvas.mpl_connect("motion_notify_event", move)
    canvas.mpl_connect("button_release_event", lambda event: state.update(p=None))


# ── Foto actual ───────────────────────────────────────────────────────────────

class IntraImageChartPanel(tk.Frame):
    """Gráficos de los objetos de la foto actual. Se dibuja solo cuando está a la vista
    (al abrir la pestaña Gráficos o al cambiar de foto con la pestaña abierta)."""

    def __init__(self, parent, colors: dict, on_pick=None, **kwargs):
        super().__init__(parent, bg=colors["bg_card"], **kwargs)
        self.on_pick = on_pick                          # on_pick(object_id): clic en un punto
        self._rows: list[dict] = []
        self._dirty = False
        self._fig = None
        if not MPL_OK:
            tk.Label(self, text=t("chart.no_mpl"), bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                     font=FONTS["small"]).pack(pady=SPACE["md"])
            return
        bg = COLORS["bg_panel"]
        self.bar = tk.Frame(self, bg=bg)
        self.bar.pack(fill=tk.X)
        self.chooser = _Chooser(self.bar, ("box", "hist", "scatter"), self._refresh, bg, owner=self)
        self.chooser.pack(side=tk.LEFT, padx=SPACE["sm"])
        self.bar2 = tk.Frame(self, bg=bg)               # las variables bajan aquí si no caben
        for widget in (self.bar, self.chooser.vars_frame):
            widget.bind("<Configure>", lambda event: self.after_idle(self._fit_bar), add="+")
        self._frame = tk.Frame(self, bg=COLORS["bg_card"])
        self._frame.pack(fill=tk.BOTH, expand=True)
        self._drawn = (0, 0)
        self._resize_job = None
        self._frame.bind("<Configure>", self._on_resize)

    def _fit_bar(self):
        vars_frame = self.chooser.vars_frame
        below = vars_frame.winfo_manager() == "pack" and str(vars_frame.pack_info().get("in")) == str(self.bar2)
        width = self.bar.winfo_width()
        need = self.bar.winfo_reqwidth() + (vars_frame.winfo_reqwidth() + 10 if below else 0)
        if not below and need > width > 1:
            vars_frame.pack_forget()
            vars_frame.pack(in_=self.bar2, side=tk.LEFT, padx=SPACE["md"], pady=(0, SPACE["sm"]))
            self.bar2.pack(fill=tk.X, after=self.bar)
            vars_frame.lift()
        elif below and need <= width:
            vars_frame.pack_forget()
            vars_frame.pack(in_=self.chooser, side=tk.LEFT, before=self.chooser.extra)
            self.bar2.pack_forget()
            vars_frame.lift()

    def _on_resize(self, event):
        """Al cambiar de tamaño (panel de abajo ↔ lugar de una imagen) se vuelve a dibujar."""
        if abs(event.width - self._drawn[0]) < 30 and abs(event.height - self._drawn[1]) < 30:
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
        frame_width, frame_height = self._frame.winfo_width(), self._frame.winfo_height()
        self._drawn = (frame_width, frame_height)
        fig_width, fig_height = max(frame_width, 300) / 96, max(frame_height, 160) / 96
        fig = Figure(figsize=(fig_width, fig_height), dpi=96)
        picker = _Picker()
        stacked = fig_width / fig_height < 1.3                       # alto y angosto: uno debajo del otro
        n_axes = max(1, len(cols))
        narrow = (frame_width if stacked else frame_width / n_axes) < 260  # poco ancho por gráfico: título más corto
        if kind == "scatter":
            if len(cols) < 2:
                _message(self._frame, t("chart.pick_xy"))
                return
            use = cols[:3]
            pts = [tuple(float(row[column]) for column in use) + (row.get("object_id"),) for row in self._rows
                   if all(isinstance(row.get(column), (int, float)) for column in use)]
            if not pts:
                _message(self._frame, t("chart.no_data_image"))
                return
            if len(use) == 3:
                xs_values, ys_values, zs_values, _ = zip(*pts)
                _scatter3d(fig, [("", xs_values, ys_values, zs_values, PALETTE[0])], use)
            else:
                xs_values, ys_values, ids = zip(*pts)
                _scatter(fig, xs_values, ys_values, cols[0], cols[1])
                picker.add(fig._scatter_ax, xs_values, ys_values, ids)
        else:
            for i, col in enumerate(cols):
                axes = fig.add_subplot(len(cols), 1, i + 1) if stacked else fig.add_subplot(1, len(cols), i + 1)
                vals = _values(self._rows, col)
                color = PALETTE[i % len(PALETTE)]
                if not vals:
                    continue
                if kind == "box":
                    boxes = axes.boxplot(vals, patch_artist=True, widths=0.5,
                                    medianprops=dict(color=COLORS["text"], linewidth=1.5))
                    boxes["boxes"][0].set_facecolor(color + "66")
                    boxes["boxes"][0].set_edgecolor(color)
                    rng = np.random.default_rng(0)
                    jitter_x = 1 + rng.uniform(-0.15, 0.15, len(vals))
                    axes.scatter(jitter_x, vals, alpha=0.4, s=10, color=color, zorder=3)
                    ids = [row.get("object_id") for row in self._rows
                           if isinstance(row.get(col), (int, float)) and row[col] == row[col]]
                    picker.add(axes, jitter_x, vals, ids)
                    axes.set_xticks([])
                else:
                    axes.hist(vals, bins=min(20, max(5, len(vals) // 3)), color=color, edgecolor=COLORS["white"],
                            alpha=0.85, linewidth=0.5)
                    axes.set_ylabel(t("chart.frequency"))
                spread = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                med = f"{t('chart.mean')} = {_num(np.mean(vals))}"
                dev = f"{t('chart.sd')} = {_num(spread)}"
                axes.set_title(f"{_label(col)}\nn = {len(vals)}  {med}\n{dev}" if narrow else
                             f"{_label(col)}\nn = {len(vals)}   {med}   {dev}", fontsize=7 if narrow else 8)
                if narrow:
                    axes.tick_params(labelsize=7)
                _style_ax(axes)
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
    for path in sorted(results, key=lambda path: natural_key(Path(path).name)):
        result = results[path]
        if result is None or result.status != "ok":
            continue
        name = Path(path).stem
        row, spread = {}, {}
        for key, value in (result.stats or {}).items():
            if key not in SKIP and not key.endswith("_per_px") and isinstance(value, (int, float)) \
                    and not isinstance(value, bool):
                row[key] = float(value)
        rows = result.measurements or []
        for col in numeric_cols(rows):
            vals = _values(rows, col)
            if vals:
                row[f"{col}_mean"] = float(np.mean(vals))
                spread[f"{col}_mean"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        names.append(name)
        means[name], sds[name] = row, spread
    return names, means, sds


def batch_records(results: dict, meta: dict | None = None, unit: str = "images") -> tuple[list[dict], list[str]]:
    """Los datos de la comparación: una fila por foto (unit="images") o por objeto
    ("objects"). Cada fila: `_name` (foto), `_meta` (columnas de la tabla del usuario),
    `_sd` (DE entre los objetos de la foto, solo por foto) y los valores numéricos."""
    meta = meta or {}
    stems = {Path(path).stem: path for path in results}
    recs, cols = [], []
    if unit == "images":
        names, means, sds = batch_table(results)
        for name in names:
            recs.append({"_name": name, "_path": stems[name], "_oid": None, "_meta": meta.get(stems[name]) or {},
                         "_sd": sds[name], **means[name]})
            cols += [key for key in means[name] if key not in cols]
        return recs, cols
    for path in sorted(results, key=lambda path: natural_key(Path(path).name)):
        result = results[path]
        if result is None or result.status != "ok":
            continue
        rows = result.measurements or []
        num = numeric_cols(rows)
        cols += [column for column in num if column not in cols]
        name = (result.stats or {}).get("n_objects")            # conteo de su foto: ¿más objetos, más chicos?
        for x in rows:
            recs.append({"_name": Path(path).stem, "_path": path, "_oid": x.get("object_id"),
                         "_meta": meta.get(path) or {}, "_sd": {},
                         **({"n_objects": float(name)} if isinstance(name, (int, float)) else {}),
                         **{column: float(x[column]) for column in num if isinstance(x.get(column), (int, float)) and x[column] == x[column]}})
    if any("n_objects" in result for result in recs):
        cols.append("n_objects")
    return recs, cols


def _count_vs_size(cols: list[str]) -> list[str]:
    """Dispersión por defecto en el lote: conteo de la foto vs. tamaño medio de sus objetos."""
    size = next((column for prefix in PREFERRED for column in cols if column.lower().startswith(prefix)), None)
    return ["n_objects", size] if "n_objects" in cols and size else []


def _group_name(value) -> str:
    return t("chart.no_value") if value in (None, "") else str(value)


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
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        width, height = min(1100, int(screen_width * 0.85)), min(700, int(screen_height * 0.82))
        self.geometry(f"{width}x{height}+{(screen_width - width) // 2}+{(screen_height - height) // 2}")
        self._fig = None
        self.results, self.meta = results, {path: (meta or {}).get(path) or {} for path in results}
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
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(SPACE["lg"], SPACE["xs"]))
        for key in ("images", "objects"):
            tk.Radiobutton(top, text=t(f"chart.unit_{key}"), variable=self.unit, value=key,
                           command=self._unit_changed, bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"],
                           activebackground=bg, font=FONTS["small"]).pack(side=tk.LEFT, pady=SPACE["sm"])
        HelpIcon(top, t("chart.unit"), t("chart.unit_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self.group = tk.StringVar(value=NONE)
        if groups:
            tk.Label(top, text=t("chart.group_by") + ":", bg=bg, fg=COLORS["text_muted"],
                     font=FONTS["small"]).pack(side=tk.LEFT, padx=(SPACE["lg"], SPACE["xs"]))
            combo = ttk.Combobox(top, textvariable=self.group, state="readonly", width=14, font=FONTS["small"],
                              values=[NONE] + list(groups))
            combo.pack(side=tk.LEFT, pady=SPACE["sm"])
            combo.bind("<<ComboboxSelected>>", lambda event: self._refresh())
        button(top, text=t("chart.save"), command=lambda: _save(self._fig, self), kind="secondary", size="small").pack(side=tk.RIGHT, padx=SPACE["md"])
        bar = tk.Frame(self, bg=bg)
        bar.pack(fill=tk.X)
        self.chooser = _Chooser(bar, ("bars", "box", "hist", "scatter"), self._refresh, bg)
        self.chooser.pack(side=tk.LEFT, padx=SPACE["md"], pady=(0, SPACE["sm"]))
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
        if not [column for column in self.chooser.chosen.get("scatter", []) if column in cols][1:]:
            self.chooser.chosen["scatter"] = _count_vs_size(cols)
        if self.chooser.kind.get() == "scatter":
            self.chooser._build_vars()
        self._refresh()

    def _categories(self, kind: str, col: str) -> list[tuple[str, list[dict]]]:
        """(nombre, filas) de cada barra/caja/serie: el grupo elegido; sin grupo, cada foto
        (barras, y cajas por objeto) o todo junto."""
        group = self.group.get()
        objects = self.units.get(self.unit.get()) == "objects"
        if group != NONE:
            key = lambda row: _group_name(row["_meta"].get(group))
        elif kind == "bars" or (kind == "box" and objects):
            key = lambda row: row["_name"]
        else:
            key = lambda row: ""
        out: dict[str, list[dict]] = {}
        for row in self.recs:
            if col is None or col in row:
                out.setdefault(key(row), []).append(row)
        empty = t("chart.no_value")                 # las fotos sin dato, al final
        return sorted(out.items(), key=lambda item: (item[0] == empty, natural_key(item[0])))

    def _refresh(self):
        kind = self.chooser.kind.get()
        cols = self.chooser.selected()
        grouped = self.group.get() != NONE
        per_image = self.units.get(self.unit.get()) == "images"
        within = kind == "bars" and per_image and not grouped        # DE entre los objetos de cada foto
        has_sd = kind == "bars" and any(
            (any(row["_sd"].get(column) for row in self.recs) if within
             else any(len(rows) > 1 for _, rows in self._categories(kind, column))) for column in cols)
        if has_sd:
            self._sd_check.configure(text=t("chart.show_sd") if within else t("chart.show_sd_group"))
            self._sd_check.pack(side=tk.LEFT)
        else:
            self._sd_check.pack_forget()
        if not cols:
            _message(self._frame, t("chart.pick_var"))
            return
        plt.rcParams.update(MPL_STYLE)
        width = max(self._frame.winfo_width(), 600) / 96
        height = max(self._frame.winfo_height(), 300) / 96 - 0.4
        fig = Figure(figsize=(width, height), dpi=96)
        self._picker = _Picker()
        unit_n = "chart.n_images" if per_image else "chart.n_objects"
        if kind == "scatter":
            if len(cols) < 2:
                _message(self._frame, t("chart.pick_xy"))
                return
            if not self._scatter(fig, cols[0], cols[1], per_image and not grouped, cols[2] if len(cols) > 2 else None):
                _message(self._frame, t("chart.no_data_batch"))
                return
        else:
            groups = [name for name, _ in self._categories(kind, None)] if grouped else []
            colors = {group: PALETTE[i % len(PALETTE)] for i, group in enumerate(groups)}
            for i, col in enumerate(cols):
                axes = fig.add_subplot(1, len(cols), i + 1)
                cats = self._categories(kind, col)
                vals = [[row[col] for row in rows] for _, rows in cats]
                count = sum(len(values) for values in vals)
                if not count:
                    continue
                names = [column for column, _ in cats]
                base = PALETTE[i % len(PALETTE)]
                bar_colors = [colors.get(column, base) for column in names]
                if kind == "bars":
                    x = np.arange(len(names))
                    axes.bar(x, [np.mean(values) for values in vals], color=bar_colors, alpha=0.85, width=0.65,
                           edgecolor=COLORS["white"], linewidth=0.5)
                    if self.show_sd.get() and has_sd:
                        err = [row["_sd"].get(col, 0) for _, rows in cats for row in rows[:1]] if within else \
                            [np.std(values, ddof=1) if len(values) > 1 else 0 for values in vals]
                        axes.errorbar(x, [np.mean(values) for values in vals], yerr=err, fmt="none",
                                    color=COLORS["text"], capsize=2, linewidth=0.8)
                    self._xticks(axes, x, names)
                elif kind == "box":
                    boxes = axes.boxplot(vals, patch_artist=True, widths=0.5, showfliers=False,
                                    medianprops=dict(color=COLORS["text"], linewidth=1.5))
                    rng = np.random.default_rng(0)
                    for index, (box, values) in enumerate(zip(boxes["boxes"], vals)):
                        box.set_facecolor(bar_colors[index] + "55")
                        box.set_edgecolor(bar_colors[index])
                        if len(values) <= 2000:
                            jitter_x = index + 1 + rng.uniform(-0.12, 0.12, len(values))
                            axes.scatter(jitter_x, values, color=bar_colors[index], alpha=0.6, s=10 if len(values) > 100 else 14, zorder=3,
                                       linewidths=0)
                            self._picker.add(axes, jitter_x, values, [(row["_path"], row["_oid"]) for row in cats[index][1]])
                    if len(names) > 1:
                        self._xticks(axes, np.arange(1, len(names) + 1), names)
                    else:
                        axes.set_xticks([])
                else:
                    allv = np.concatenate([np.asarray(values, float) for values in vals])
                    bins = np.histogram_bin_edges(allv, bins=min(20, max(5, len(allv) // 2)))
                    for index, (name, values) in enumerate(zip(names, vals)):
                        axes.hist(values, bins=bins, color=bar_colors[index], alpha=0.55 if len(names) > 1 else 0.85,
                                edgecolor=COLORS["white"], linewidth=0.5, label=name if grouped else None)
                    axes.set_ylabel(t("chart.frequency"))
                    if grouped:
                        axes.legend(frameon=False, fontsize=7)
                axes.set_title(f"{_label(col)}\n{t(unit_n, n=count)}", fontsize=8)
                _style_ax(axes)
        try:
            fig.tight_layout(pad=1.0)
        except Exception:
            _log.debug("tight_layout", exc_info=True)
        self._fig = fig
        self._picker.connect(_embed(self._frame, fig, toolbar=True),
                             (lambda pick_data: self.on_pick(*pick_data)) if self.on_pick else None)

    @staticmethod
    def _xticks(axes, x, names):
        axes.set_xticks(x)
        many = len(names) > 8 or max((len(name) for name in names), default=0) > 10
        axes.set_xticklabels(names if len(names) <= 60 else [""] * len(names),
                           rotation=60 if many else 0, ha="right" if many else "center", fontsize=6 if many else 7)

    def _scatter(self, fig, x_column: str, y_column: str, labels: bool, z_column: str | None = None) -> bool:
        if z_column:                                       # tercera variable: dispersión 3D
            cats = [(group, [row for row in rows if x_column in row and y_column in row and z_column in row])
                    for group, rows in self._categories("scatter", None)]
            cats = [(group, rows) for group, rows in cats if rows]
            if not cats:
                return False
            grouped = self.group.get() != NONE
            _scatter3d(fig, [(group if grouped else "", [row[x_column] for row in rows], [row[y_column] for row in rows],
                              [row[z_column] for row in rows], PALETTE[index % len(PALETTE)] if grouped else PALETTE[0])
                             for index, (group, rows) in enumerate(cats)], [x_column, y_column, z_column],
                       self.group.get() if grouped else None)
            return True
        if self.group.get() == NONE:
            pts = [(row[x_column], row[y_column], row["_name"], (row["_path"], row["_oid"])) for row in self.recs if x_column in row and y_column in row]
            if not pts:
                return False
            xs_values, ys_values, names, pick_data = zip(*pts)
            _scatter(fig, xs_values, ys_values, x_column, y_column, list(names) if labels else None)
            self._picker.add(fig._scatter_ax, xs_values, ys_values, pick_data)
            return True
        cats = [(group, [row for row in rows if x_column in row and y_column in row]) for group, rows in self._categories("scatter", None)]
        cats = [(group, rows) for group, rows in cats if rows]
        if not cats:
            return False
        axes = fig.add_subplot(1, 1, 1)
        xs_all, ys_all = [], []
        for index, (group, rows) in enumerate(cats):
            xs_values, ys_values = [row[x_column] for row in rows], [row[y_column] for row in rows]
            xs_all += xs_values
            ys_all += ys_values
            axes.scatter(xs_values, ys_values, color=PALETTE[index % len(PALETTE)], alpha=0.75, s=22 if len(xs_all) < 300 else 10,
                       edgecolors=COLORS["white"], linewidths=0.4, zorder=3, label=group)
            self._picker.add(axes, xs_values, ys_values, [(row["_path"], row["_oid"]) for row in rows])
        axes.set_xlabel(_label(x_column))
        axes.set_ylabel(_label(y_column))
        ok = len(xs_all) > 2 and np.std(xs_all) > 0 and np.std(ys_all) > 0
        axes.set_title(f"n = {len(xs_all)}" + (f"   r = {np.corrcoef(xs_all, ys_all)[0, 1]:.2f}" if ok else ""))
        axes.legend(frameon=False, fontsize=7, title=self.group.get(), title_fontsize=7)
        _style_ax(axes)
        return True
