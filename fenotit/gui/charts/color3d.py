"""Colores en 3D (CIELab, RGB o HSV): navegable (arrastrar = girar, rueda = zoom),
capas por nivel (colores de cada objeto, objetos como torta, colores de la foto, fotos
como torta), etiquetas ID/nombre, leyenda y clic en un punto = de qué foto/objeto es."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

# log escribe los mensajes y errores en el archivo de registro.
from fenotit import log
# Los puntos de color de cada objeto o foto, ya pasados al espacio elegido (Lab, RGB, HSV).
from fenotit.core.stats import colors as color_stats
# Lo común con los demás gráficos: si matplotlib está, su estilo, el aviso sin datos y Guardar.
from fenotit.gui.charts.panels import MPL_OK, MPL_STYLE, _message, _save
# HelpIcon es el «?» que explica una opción al pasar el mouse o al hacer clic.
from fenotit.gui.widgets.help import HelpIcon
# button crea un botón con el estilo del tema.
from fenotit.gui.style.components import button
# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras); SPACE (márgenes).
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
# t devuelve cada texto de la ventana en el idioma elegido (fenotit/lang).
from fenotit.i18n import t

if MPL_OK:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    from matplotlib import patheffects
    from matplotlib.path import Path as MPath
    from mpl_toolkits.mplot3d import proj3d

_log = log.get("gui.color3d")

STEPS = 36                     # sectores de las tortas en pasos de 10°
MANY = 4000                    # aviso de muchos puntos
EDGE = COLORS["text_muted"]
NONE = "—"


def _wedge(first_step: int, last_step: int):
    """Marcador de pantalla con el sector [a, b) de STEPS (empieza arriba, sentido horario)."""
    theta_start, theta_end = 90 - 360 * last_step / STEPS, 90 - 360 * first_step / STEPS
    wedge = MPath.wedge(theta_start, theta_end)
    # cuatro puntos invisibles fijan la escala: todos los sectores del mismo tamaño
    verts = np.vstack([[[1, 0], [0, 1], [-1, 0], [0, -1]], wedge.vertices])
    codes = np.concatenate([[MPath.MOVETO] * 4, wedge.codes])
    return MarkerStyle(MPath(verts, codes))


class Color3DWindow(tk.Toplevel):

    def __init__(self, parent, current: list, batch: list, scope: str = "image", on_goto=None, thumb=None):
        """current / batch = [(ruta, Image_ID, nombre, tablas)] de la foto actual / del lote.
        on_goto(ruta, objeto|None) = ver en la foto; thumb(ruta, objeto|None) = imagen BGR o None."""
        super().__init__(parent)
        self.title(t("c3d.title"))
        self.configure(bg=COLORS["bg_card"])
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        width, height = min(1150, int(screen_width * 0.85)), min(760, int(screen_height * 0.85))
        self.geometry(f"{width}x{height}+{(screen_width - width) // 2}+{(screen_height - height) // 2}")
        self.data = {"image": current, "batch": batch}
        self.on_goto, self.thumb = on_goto, thumb
        self._fig = self._ax = self._canvas = None
        self._pts: list[color_stats.ColorPoint] = []
        self._sizes = np.zeros(0)
        self._view = None
        self._picked = None
        if not MPL_OK:
            _message(self, t("chart.no_mpl"))
            return
        if not any(self._has(entry) for entry in self.data.values()):
            _message(self, t("c3d.no_data"))
            return
        if not self._has(self.data.get(scope)):
            scope = "batch" if self._has(batch) else "image"

        bg = COLORS["bg_panel"]
        tk.Frame(self, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        top = tk.Frame(self, bg=bg)
        top.pack(fill=tk.X)

        def label(parent, text, pad=(10, 2)):
            tk.Label(parent, text=text, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"]).pack(
                side=tk.LEFT, padx=pad)

        def combo(parent, var, values, width):
            box = ttk.Combobox(parent, textvariable=var, state="readonly", values=values, width=width,
                              font=FONTS["small"])
            box.pack(side=tk.LEFT, pady=SPACE["sm"])
            box.bind("<<ComboboxSelected>>", lambda event: self._redraw())
            return box

        self.scope = tk.StringVar(value=scope)
        if self._has(current) and self._has(batch) and len(batch) > 1:
            label(top, t("c3d.photos") + ":")
            for key, text in (("image", t("c3d.this_photo")), ("batch", t("c3d.all_photos", n=len(batch)))):
                tk.Radiobutton(top, text=text, variable=self.scope, value=key, command=self._scope_changed,
                               bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg,
                               font=FONTS["small"]).pack(side=tk.LEFT)
        label(top, t("c3d.space") + ":", (14, 2))
        self.space = tk.StringVar(value="Lab")
        combo(top, self.space, list(color_stats.SPACES), 6).bind("<<ComboboxSelected>>", lambda event: self._space_changed())
        self._dims_label = tk.Label(top, text=t("c3d.view") + ":", bg=bg, fg=COLORS["text_muted"], font=FONTS["small"])
        self.dims = tk.StringVar(value="3d")           # 3D o, en Lab, el plano a*·b* (para figuras)
        self.dims_bar = tk.Frame(top, bg=bg)
        self.dims_bar.pack(side=tk.LEFT)
        self._build_dims()
        label(top, t("c3d.labels") + ":", (14, 2))
        self.labels = {NONE: None, "ID": "id", t("c3d.name"): "name"}
        self.label_var = tk.StringVar(value=NONE)
        combo(top, self.label_var, list(self.labels), 8)
        self.legend = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text=t("c3d.legend"), variable=self.legend, command=self._redraw, bg=bg,
                       fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg,
                       font=FONTS["small"]).pack(side=tk.LEFT, padx=(SPACE["lg"], 0))
        HelpIcon(top, t("c3d.title"), t("c3d.help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["md"], 0))
        button(top, text=t("chart.save"), command=lambda: _save(self._fig, self), kind="secondary", size="small").pack(side=tk.RIGHT, padx=SPACE["md"])
        button(top, text=t("c3d.reset"), command=self._reset_view, kind="quiet", size="small").pack(side=tk.RIGHT, padx=SPACE["xs"])

        self.layer_bar = tk.Frame(self, bg=bg)
        self.layer_bar.pack(fill=tk.X)
        self.per = tk.StringVar(value="")
        self.warn = tk.Label(self.layer_bar, text="", bg=bg, fg=COLORS["warning_text"], font=FONTS["small"])

        body = tk.Frame(self, bg=COLORS["bg_card"])
        body.pack(fill=tk.BOTH, expand=True)
        self.info = tk.Frame(body, bg=COLORS["bg_card"], width=230, highlightthickness=1,
                             highlightbackground=COLORS["border"])
        self.info.pack_propagate(False)
        self._frame = tk.Frame(body, bg=COLORS["bg_card"])
        self._frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._scope_changed()

    def _build_dims(self):
        """Vista: 3D; en Lab también el plano a*·b* (diagrama de cromaticidad, útil en
        figuras impresas, donde un 3D no se puede girar)."""
        for widget in self.dims_bar.winfo_children():
            widget.destroy()
        lab = self.space.get() == "Lab"
        if not lab:
            self.dims.set("3d")
        bg = COLORS["bg_panel"]
        for key, text in (("3d", "3D"), ("1,2", "a*b*")) if lab else ():
            tk.Radiobutton(self.dims_bar, text=text, variable=self.dims, value=key, command=self._redraw, bg=bg,
                           fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg,
                           font=FONTS["small"]).pack(side=tk.LEFT)
        self._dims_label.pack(side=tk.LEFT, padx=(SPACE["lg"], SPACE["xs"]), before=self.dims_bar) if lab else self._dims_label.pack_forget()

    def _space_changed(self):
        self._build_dims()
        self._redraw()

    @property
    def is3d(self) -> bool:
        return self.dims.get() == "3d"

    def _plane(self) -> tuple[int, int]:
        i, j = (int(value) for value in self.dims.get().split(","))
        return i, j

    # ── datos ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _has(images) -> bool:
        return bool(images) and any(tables.get("object_colors") or tables.get("image_colors") for *_, tables in images)

    def _per_levels(self) -> dict[str, str]:
        """Un punto por… → nivel: color (de cada objeto, o de la foto si no hay objetos), objeto, foto."""
        imgs = self.data[self.scope.get()]
        have_obj = any(tables.get("object_colors") for *_, tables in imgs)
        out = {"color": "object_colors" if have_obj else "image_colors"}
        if have_obj:
            out["object"] = "objects"
        out["photo"] = "images"
        return out

    def _scope_changed(self):
        for widget in self.layer_bar.winfo_children():
            if widget is not self.warn:
                widget.destroy()
        bg = COLORS["bg_panel"]
        tk.Label(self.layer_bar, text=t("c3d.per") + ":", bg=bg, fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(SPACE["lg"], SPACE["xs"]))
        opts = self._per_levels()
        batch = self.scope.get() == "batch"
        self.per.set("photo" if batch else ("object" if "object" in opts else "color"))
        for key in opts:
            tk.Radiobutton(self.layer_bar, text=t(f"c3d.per.{key}"), variable=self.per, value=key,
                           command=self._redraw, bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"],
                           activebackground=bg, font=FONTS["small"]).pack(side=tk.LEFT)
        self.warn.pack(side=tk.LEFT, padx=SPACE["lg"])
        self._redraw()

    # ── dibujo ────────────────────────────────────────────────────────────────

    def _redraw(self):
        if self._ax is not None and getattr(self._ax, "name", "") == "3d":     # se conserva el giro
            self._view = (self._ax.elev, self._ax.azim, self._ax.get_xlim3d(), self._ax.get_ylim3d(),
                          self._ax.get_zlim3d(), self._drawn_space)
        levels = [self._per_levels().get(self.per.get(), "images")]
        space = self.space.get()
        pts = color_stats.points(self.data[self.scope.get()], levels, space)
        self._pts, self._picked = pts, None
        self._drawn_space = space
        self.info.pack_forget()
        self.warn.configure(text=t("c3d.many", n=len(pts)) if len(pts) > MANY else "")
        if not pts:
            self._fig = self._ax = None
            _message(self._frame, t("c3d.no_data"))
            return
        plt.rcParams.update(MPL_STYLE)
        if self._canvas is None or not self._canvas.get_tk_widget().winfo_exists():
            for widget in self._frame.winfo_children():
                widget.destroy()
            self._fig = Figure(figsize=(6, 5), dpi=96)
            self._canvas = FigureCanvasTkAgg(self._fig, master=self._frame)
            self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._canvas.mpl_connect("scroll_event", self._on_scroll)
            self._canvas.mpl_connect("button_press_event", self._on_press)
            self._canvas.mpl_connect("button_release_event", self._on_release)
            self._canvas.mpl_connect("motion_notify_event", self._on_drag)
        fig = self._fig
        fig.clear()
        three = self.is3d
        axes = self._ax = fig.add_subplot(111, projection="3d") if three else fig.add_subplot(111)
        if three:
            axes.disable_mouse_rotation()             # el giro lo hace _on_drag (más fiable en Windows)
        many = len(pts) > 1500
        sizes = []
        for point in pts:
            if point.level == "object_colors":
                sizes.append(6 + point.pct * (0.5 if many else 1.0))
            elif point.level == "image_colors":
                sizes.append(25 + point.pct * 4)
            elif point.level == "objects":
                sizes.append(45 if many else 90)
            else:
                sizes.append(240)
        self._sizes = np.array(sizes, float)
        xyz = np.array([point.xyz for point in pts])
        coords = xyz if three else xyz[:, list(self._plane())]
        kwargs = {"depthshade": False} if three else {}
        for level in color_stats.LEVELS:
            idx = [i for i, point in enumerate(pts) if point.level == level]
            if not idx:
                continue
            if level in ("objects", "images"):
                self._pies(axes, [pts[i] for i in idx], coords[idx], self._sizes[idx], kwargs)
            else:
                axes.scatter(*coords[idx].T, c=[pts[i].rgb for i in idx], s=self._sizes[idx],
                           edgecolors=EDGE, linewidths=0.3, alpha=0.9, **kwargs)
        mode = self.labels.get(self.label_var.get())
        if mode:
            halo = [patheffects.withStroke(linewidth=2.2, foreground=COLORS["white"])]
            top = [point for point in pts if point.level in ("images", "objects")] or pts
            size = {id(point): point_size for point, point_size in zip(pts, self._sizes)}
            where = {id(point): coord for point, coord in zip(pts, coords)}
            if three:
                axes.computed_zorder = False      # las etiquetas siempre encima de los puntos
            for point in top[:200]:
                text = point.label_id if mode == "id" else self._point_name(point)
                if three:
                    pad = " " * (int(np.sqrt(size[id(point)]) / 3.4) + 1)      # justo al lado del punto
                    axes.text(*point.xyz, pad + text, fontsize=6, color=COLORS["text"], zorder=10, path_effects=halo)
                else:
                    axes.annotate(text, where[id(point)], xytext=(np.sqrt(size[id(point)]) / 2 + 2, 0),
                                textcoords="offset points", va="center", fontsize=6, color=COLORS["text"],
                                zorder=10, path_effects=halo)
        names = color_stats.SPACES[space]
        if three:
            axes.set_xlabel(names[0])
            axes.set_ylabel(names[1])
            axes.set_zlabel(names[2])
            for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
                axis.pane.set_facecolor((0.97, 0.97, 0.97, 1))
        else:
            i, j = self._plane()
            axes.set_xlabel(names[i])
            axes.set_ylabel(names[j])
            if space == "Lab":                   # ejes neutros (a* = 0, b* = 0): grises
                for index, line in ((i, axes.axvline), (j, axes.axhline)):
                    if index > 0:
                        line(0, color=COLORS["text_faint"], lw=0.7, zorder=0)
                if (i, j) == (1, 2):
                    axes.set_aspect("equal", adjustable="datalim")   # distancias de color reales
            for spine in axes.spines.values():
                spine.set_edgecolor(COLORS["border"])
        n_img = len({point.path for point in pts})
        n_obj = len({(point.path, point.object_id) for point in pts if point.object_id is not None})
        title = f"{space}, " + t("chart.n_images", n=n_img) + (f", {t('chart.n_objects', n=n_obj)}" if n_obj else "")
        axes.set_title(title, fontsize=9)
        if self.legend.get():
            self._legend(fig, levels)
        if three:
            if self._view and self._view[5] == space:      # mismo giro; límites según los puntos de ahora
                axes.view_init(*self._view[:2])
            else:
                axes.view_init(22, -60)
            self._home = (axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d())
            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
        else:
            axes.margins(0.08)
            fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.93)
            self._home = None
        self._canvas.draw_idle()

    def _point_name(self, point) -> str:
        return point.name.rsplit(".", 1)[0] + (f"-{point.object_id}" if point.object_id is not None else "")

    @staticmethod
    def _pies(axes, pts, xyz, sizes, kwargs=None):
        """Cada punto es una torta con sus colores (sectores = % de cada color). Los
        sectores con la misma forma van juntos: pocas llamadas aunque haya muchos puntos."""
        groups: dict[tuple[int, int], list[tuple[int, tuple]]] = {}
        for i, point in enumerate(pts):
            total = sum(pct for _, pct in point.sectors) or 1
            acc = 0.0
            for rgb, pct in sorted(point.sectors, key=lambda sector: -sector[1]):
                start = round(acc / total * STEPS)
                acc += pct
                end = round(acc / total * STEPS)
                if end > start:
                    groups.setdefault((start, end), []).append((i, rgb))
        for (start, end), items in groups.items():
            indices = [i for i, _ in items]
            axes.scatter(*xyz[indices].T, c=[rgb for _, rgb in items], s=sizes[indices], marker=_wedge(start, end),
                       linewidths=0, **(kwargs or {}))
        axes.scatter(*xyz.T, s=sizes, facecolors="none", edgecolors=EDGE, linewidths=0.5, **(kwargs or {}))

    def _legend(self, fig, levels):
        batch = self.scope.get() == "batch"
        marks = {"object_colors": ("o", 4), "objects": ("o", 8), "image_colors": ("o", 6), "images": ("o", 11)}
        handles = []
        for level in levels:
            marker, marker_size = marks[level]
            count = sum(1 for point in self._pts if point.level == level)
            text = t(f"c3d.level.{level}" + ("_batch" if batch and level in ("image_colors", "images") else ""))
            handles.append(Line2D([], [], marker=marker, ls="", ms=marker_size, mfc=COLORS["text_faint"], mec=EDGE, mew=0.6,
                                  label=f"{text} ({count})"))
        if handles:
            fig.legend(handles=handles, loc="upper left", frameon=True, fontsize=8, framealpha=0.85,
                       title=t("c3d.legend_note"), title_fontsize=7, alignment="left")

    # ── navegación ────────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        if self._ax is None:
            return
        factor = 0.85 if event.button == "up" else 1 / 0.85
        if not self.is3d:                               # 2D: acerca hacia el cursor
            if event.xdata is None:
                return
            for (low, high), center, put in ((self._ax.get_xlim(), event.xdata, self._ax.set_xlim),
                                     (self._ax.get_ylim(), event.ydata, self._ax.set_ylim)):
                put(center - (center - low) * factor, center + (high - center) * factor)
            self._canvas.draw_idle()
            return
        for get, put in ((self._ax.get_xlim3d, self._ax.set_xlim3d), (self._ax.get_ylim3d, self._ax.set_ylim3d),
                         (self._ax.get_zlim3d, self._ax.set_zlim3d)):
            low, high = get()
            center, half_range = (low + high) / 2, (high - low) / 2 * factor
            put(center - half_range, center + half_range)
        self._canvas.draw_idle()

    def _reset_view(self):
        if self._ax is None:
            return
        if not self.is3d:
            self._redraw()
            return
        self._ax.view_init(22, -60)
        x_limits, y_limits, z_limits = self._home
        self._ax.set_xlim3d(x_limits)
        self._ax.set_ylim3d(y_limits)
        self._ax.set_zlim3d(z_limits)
        self._canvas.draw_idle()

    # ── identificar un punto ──────────────────────────────────────────────────

    def _on_press(self, event):
        self._press = (event.x, event.y)
        self._press_view = (self._ax.elev, self._ax.azim) if self._ax is not None and event.button == 1 \
            and self.is3d else None

    def _on_drag(self, event):
        """Arrastrar con el botón izquierdo = girar (0.4° por píxel)."""
        press_view, press = getattr(self, "_press_view", None), getattr(self, "_press", None)
        if not self.is3d or press_view is None or press is None or event.x is None or self._ax is None:
            return
        elev = float(np.clip(press_view[0] - (event.y - press[1]) * 0.4, -90, 90))
        self._ax.view_init(elev, press_view[1] - (event.x - press[0]) * 0.4)
        self._canvas.draw_idle()

    def _on_release(self, event):
        self._press_view = None
        press = getattr(self, "_press", None)
        if self._ax is None or press is None or event.button != 1 or event.x is None:
            return
        if abs(event.x - press[0]) + abs(event.y - press[1]) > 4:        # fue un giro, no un clic
            return
        i = self.nearest(event.x, event.y)
        if i is not None:
            self._show_point(i)

    def screen_xy(self) -> np.ndarray:
        xyz = np.array([point.xyz for point in self._pts])
        if not self.is3d:
            return self._ax.transData.transform(xyz[:, list(self._plane())])
        x, y, _ = proj3d.proj_transform(xyz[:, 0], xyz[:, 1], xyz[:, 2], self._ax.get_proj())
        return self._ax.transData.transform(np.c_[x, y])

    def nearest(self, x: float, y: float) -> int | None:
        if not self._pts:
            return None
        screen = self.screen_xy()
        distance = np.hypot(screen[:, 0] - x, screen[:, 1] - y)
        radius = np.sqrt(self._sizes) / 2 * self._fig.dpi / 72 + 3
        hit = np.where(distance <= radius)[0]
        if not len(hit):
            return None
        rank = {"object_colors": 0, "image_colors": 1, "objects": 2, "images": 3}   # el más chico encima
        return int(min(hit, key=lambda i: (rank[self._pts[i].level], distance[i])))

    def _show_point(self, i: int):
        point = self._pts[i]
        if self._picked is not None:
            try:
                self._picked.remove()
            except Exception:
                pass
        position = np.array([point.xyz]) if self.is3d else np.array([point.xyz])[:, list(self._plane())]
        self._picked = self._ax.scatter(*position.T, s=self._sizes[i] * 2.2 + 60, facecolors="none", edgecolors=COLORS["highlight"],
                                        linewidths=1.6, zorder=11, **({"depthshade": False} if self.is3d else {}))
        self._canvas.draw_idle()
        info = self.info
        for widget in info.winfo_children():
            widget.destroy()
        if not info.winfo_ismapped():
            info.pack(side=tk.LEFT, fill=tk.Y, before=self._frame)
        bg = COLORS["bg_card"]
        head = tk.Frame(info, bg=bg)
        head.pack(fill=tk.X, padx=SPACE["lg"], pady=(SPACE["md"], SPACE["sm"]))
        tk.Label(head, text=t("c3d.point"), bg=bg, fg=COLORS["accent"], font=FONTS["body_bold"]).pack(side=tk.LEFT)
        x = tk.Label(head, text="✕", bg=bg, fg=COLORS["text_muted"], cursor="hand2", font=FONTS["small"])
        x.pack(side=tk.RIGHT)
        x.bind("<Button-1>", lambda event: info.pack_forget())
        img = self.thumb(point.path, point.object_id) if self.thumb else None
        if img is not None:
            picture = Image.fromarray(img[:, :, ::-1])
            picture.thumbnail((206, 170))
            self._thumb_img = ImageTk.PhotoImage(picture)
            tk.Label(info, image=self._thumb_img, bg=bg).pack(padx=SPACE["lg"], pady=(0, SPACE["md"]))
        rows = [(t("c3d.photo"), f"{point.name} (ID {point.image_id})")]
        if point.object_id is not None:
            rows.append((t("c3d.object"), str(point.object_id)))
        if point.level in ("object_colors", "image_colors"):
            rows.append(("%", f"{point.pct:.1f}"))
        rows.append((self.space.get(), ", ".join(f"{value:.0f}" for value in point.xyz)))
        rgb = tuple(int(round(value * 255)) for value in point.rgb)
        rows.append(("HEX", "#%02x%02x%02x" % rgb))
        grid = tk.Frame(info, bg=bg)
        grid.pack(fill=tk.X, padx=SPACE["lg"])
        for row, (key, value) in enumerate(rows):
            tk.Label(grid, text=key, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"], anchor="w").grid(
                row=row, column=0, sticky="w")
            tk.Label(grid, text=value, bg=bg, fg=COLORS["text"], font=FONTS["small"], anchor="e",
                     wraplength=150, justify="right").grid(row=row, column=1, sticky="e", padx=(SPACE["md"], 0))
        grid.columnconfigure(1, weight=1)
        frame = tk.Frame(info, bg="#%02x%02x%02x" % rgb, height=14)
        frame.pack(fill=tk.X, padx=SPACE["lg"], pady=(SPACE["md"], 0))
        if point.sectors:
            bar = tk.Frame(info, bg=bg, height=10)
            bar.pack(fill=tk.X, padx=SPACE["lg"], pady=(SPACE["xs"], 0))
            total, acc = sum(percent for _, percent in point.sectors) or 1, 0.0
            for rgb_s, percent in sorted(point.sectors, key=lambda sector: -sector[1]):
                tk.Frame(bar, bg="#%02x%02x%02x" % tuple(int(round(value * 255)) for value in rgb_s)).place(
                    relx=acc / total, rely=0, relwidth=percent / total, relheight=1)
                acc += percent
        if self.on_goto:
            button(info, text=t("c3d.goto"), command=lambda: self.on_goto(point.path, point.object_id), kind="primary", size="small").pack(pady=SPACE["lg"])
        self._picked_point = point
