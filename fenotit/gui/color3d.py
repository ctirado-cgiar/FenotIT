"""Colores en 3D (CIELab, RGB o HSV): navegable (arrastrar = girar, rueda = zoom),
capas por nivel (colores de cada objeto, objetos como torta, colores de la foto, fotos
como torta), etiquetas ID/nombre, leyenda y clic en un punto = de qué foto/objeto es."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from fenotit import log
from fenotit.core.stats import colors as cs
from fenotit.gui.charts import MPL_OK, MPL_STYLE, _message, _save
from fenotit.gui.help import HelpIcon
from fenotit.gui.theme import COLORS, FONTS
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
EDGE = "#555555"
NONE = "—"


def _wedge(a: int, b: int):
    """Marcador de pantalla con el sector [a, b) de STEPS (empieza arriba, sentido horario)."""
    t1, t2 = 90 - 360 * b / STEPS, 90 - 360 * a / STEPS
    w = MPath.wedge(t1, t2)
    # cuatro puntos invisibles fijan la escala: todos los sectores del mismo tamaño
    verts = np.vstack([[[1, 0], [0, 1], [-1, 0], [0, -1]], w.vertices])
    codes = np.concatenate([[MPath.MOVETO] * 4, w.codes])
    return MarkerStyle(MPath(verts, codes))


class Color3DWindow(tk.Toplevel):

    def __init__(self, parent, current: list, batch: list, scope: str = "image", on_goto=None, thumb=None):
        """current / batch = [(ruta, Image_ID, nombre, tablas)] de la foto actual / del lote.
        on_goto(ruta, objeto|None) = ver en la foto; thumb(ruta, objeto|None) = imagen BGR o None."""
        super().__init__(parent)
        self.title(t("c3d.title"))
        self.configure(bg=COLORS["bg_card"])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1150, int(sw * 0.85)), min(760, int(sh * 0.85))
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self.data = {"image": current, "batch": batch}
        self.on_goto, self.thumb = on_goto, thumb
        self._fig = self._ax = self._canvas = None
        self._pts: list[cs.ColorPoint] = []
        self._sizes = np.zeros(0)
        self._view = None
        self._picked = None
        if not MPL_OK:
            _message(self, t("chart.no_mpl"))
            return
        if not any(self._has(d) for d in self.data.values()):
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
            cb = ttk.Combobox(parent, textvariable=var, state="readonly", values=values, width=width,
                              font=FONTS["small"])
            cb.pack(side=tk.LEFT, pady=4)
            cb.bind("<<ComboboxSelected>>", lambda e: self._redraw())
            return cb

        self.scope = tk.StringVar(value=scope)
        if self._has(current) and self._has(batch) and len(batch) > 1:
            label(top, t("c3d.photos") + ":")
            for key, text in (("image", t("c3d.this_photo")), ("batch", t("c3d.all_photos", n=len(batch)))):
                tk.Radiobutton(top, text=text, variable=self.scope, value=key, command=self._scope_changed,
                               bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg,
                               font=FONTS["small"]).pack(side=tk.LEFT)
        label(top, t("c3d.space") + ":", (14, 2))
        self.space = tk.StringVar(value="Lab")
        combo(top, self.space, list(cs.SPACES), 6).bind("<<ComboboxSelected>>", lambda e: self._space_changed())
        label(top, t("c3d.view") + ":", (14, 2))
        self.dims = tk.StringVar(value="1,2")          # 2D a*·b* (diagrama de cromaticidad) o "3d"
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
                       font=FONTS["small"]).pack(side=tk.LEFT, padx=(14, 0))
        HelpIcon(top, t("c3d.title"), t("c3d.help"), bg=bg).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(top, text=t("chart.save"), command=lambda: _save(self._fig, self), bg=COLORS["btn_bg"],
                  fg=COLORS["accent"], relief="flat", font=FONTS["small"], cursor="hand2").pack(side=tk.RIGHT, padx=8)
        tk.Button(top, text=t("c3d.reset"), command=self._reset_view, bg=COLORS["btn_bg"], fg=COLORS["text"],
                  relief="flat", font=FONTS["small"], cursor="hand2").pack(side=tk.RIGHT, padx=2)

        self.layer_bar = tk.Frame(self, bg=bg)
        self.layer_bar.pack(fill=tk.X)
        self.per = tk.StringVar(value="")
        self.warn = tk.Label(self.layer_bar, text="", bg=bg, fg="#B35806", font=FONTS["small"])

        body = tk.Frame(self, bg=COLORS["bg_card"])
        body.pack(fill=tk.BOTH, expand=True)
        self.info = tk.Frame(body, bg=COLORS["bg_card"], width=230, highlightthickness=1,
                             highlightbackground=COLORS["border"])
        self.info.pack_propagate(False)
        self._frame = tk.Frame(body, bg=COLORS["bg_card"])
        self._frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._scope_changed()

    def _build_dims(self):
        """Vista: tres planos 2D del espacio elegido (en Lab: a*·b*, L*·a*, L*·b*) o 3D."""
        for w in self.dims_bar.winfo_children():
            w.destroy()
        names = cs.SPACES[self.space.get()]
        bg = COLORS["bg_panel"]
        for key in ("1,2", "0,1", "0,2", "3d"):
            text = "3D" if key == "3d" else "·".join(names[int(i)] for i in key.split(","))
            tk.Radiobutton(self.dims_bar, text=text, variable=self.dims, value=key, command=self._redraw, bg=bg,
                           fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg,
                           font=FONTS["small"]).pack(side=tk.LEFT)

    def _space_changed(self):
        self._build_dims()
        self._redraw()

    @property
    def is3d(self) -> bool:
        return self.dims.get() == "3d"

    def _plane(self) -> tuple[int, int]:
        i, j = (int(v) for v in self.dims.get().split(","))
        return i, j

    # ── datos ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _has(images) -> bool:
        return bool(images) and any(tb.get("object_colors") or tb.get("image_colors") for *_, tb in images)

    def _per_levels(self) -> dict[str, str]:
        """Un punto por… → nivel: color (de cada objeto, o de la foto si no hay objetos), objeto, foto."""
        imgs = self.data[self.scope.get()]
        have_obj = any(tb.get("object_colors") for *_, tb in imgs)
        out = {"color": "object_colors" if have_obj else "image_colors"}
        if have_obj:
            out["object"] = "objects"
        out["photo"] = "images"
        return out

    def _scope_changed(self):
        for w in self.layer_bar.winfo_children():
            if w is not self.warn:
                w.destroy()
        bg = COLORS["bg_panel"]
        tk.Label(self.layer_bar, text=t("c3d.per") + ":", bg=bg, fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(10, 2))
        opts = self._per_levels()
        batch = self.scope.get() == "batch"
        self.per.set("photo" if batch else ("object" if "object" in opts else "color"))
        for k in opts:
            tk.Radiobutton(self.layer_bar, text=t(f"c3d.per.{k}"), variable=self.per, value=k,
                           command=self._redraw, bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"],
                           activebackground=bg, font=FONTS["small"]).pack(side=tk.LEFT)
        self.warn.pack(side=tk.LEFT, padx=12)
        self._redraw()

    # ── dibujo ────────────────────────────────────────────────────────────────

    def _redraw(self):
        if self._ax is not None and getattr(self._ax, "name", "") == "3d":     # se conserva el giro
            self._view = (self._ax.elev, self._ax.azim, self._ax.get_xlim3d(), self._ax.get_ylim3d(),
                          self._ax.get_zlim3d(), self._drawn_space)
        levels = [self._per_levels().get(self.per.get(), "images")]
        space = self.space.get()
        pts = cs.points(self.data[self.scope.get()], levels, space)
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
            for w in self._frame.winfo_children():
                w.destroy()
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
        ax = self._ax = fig.add_subplot(111, projection="3d") if three else fig.add_subplot(111)
        if three:
            ax.disable_mouse_rotation()             # el giro lo hace _on_drag (más fiable en Windows)
        many = len(pts) > 1500
        sizes = []
        for p in pts:
            if p.level == "object_colors":
                sizes.append(6 + p.pct * (0.5 if many else 1.0))
            elif p.level == "image_colors":
                sizes.append(25 + p.pct * 4)
            elif p.level == "objects":
                sizes.append(45 if many else 90)
            else:
                sizes.append(240)
        self._sizes = np.array(sizes, float)
        xyz = np.array([p.xyz for p in pts])
        coords = xyz if three else xyz[:, list(self._plane())]
        kw = {"depthshade": False} if three else {}
        for level in cs.LEVELS:
            idx = [i for i, p in enumerate(pts) if p.level == level]
            if not idx:
                continue
            if level in ("objects", "images"):
                self._pies(ax, [pts[i] for i in idx], coords[idx], self._sizes[idx], kw)
            else:
                ax.scatter(*coords[idx].T, c=[pts[i].rgb for i in idx], s=self._sizes[idx],
                           edgecolors=EDGE, linewidths=0.3, alpha=0.9, **kw)
        mode = self.labels.get(self.label_var.get())
        if mode:
            halo = [patheffects.withStroke(linewidth=2.2, foreground="white")]
            top = [p for p in pts if p.level in ("images", "objects")] or pts
            size = {id(p): sz for p, sz in zip(pts, self._sizes)}
            where = {id(p): c for p, c in zip(pts, coords)}
            if three:
                ax.computed_zorder = False      # las etiquetas siempre encima de los puntos
            for p in top[:200]:
                text = p.label_id if mode == "id" else self._point_name(p)
                if three:
                    pad = " " * (int(np.sqrt(size[id(p)]) / 3.4) + 1)      # justo al lado del punto
                    ax.text(*p.xyz, pad + text, fontsize=6, color="#222222", zorder=10, path_effects=halo)
                else:
                    ax.annotate(text, where[id(p)], xytext=(np.sqrt(size[id(p)]) / 2 + 2, 0),
                                textcoords="offset points", va="center", fontsize=6, color="#222222",
                                zorder=10, path_effects=halo)
        names = cs.SPACES[space]
        if three:
            ax.set_xlabel(names[0])
            ax.set_ylabel(names[1])
            ax.set_zlabel(names[2])
            for a in (ax.xaxis, ax.yaxis, ax.zaxis):
                a.pane.set_facecolor((0.97, 0.97, 0.97, 1))
        else:
            i, j = self._plane()
            ax.set_xlabel(names[i])
            ax.set_ylabel(names[j])
            if space == "Lab":                   # ejes neutros (a* = 0, b* = 0): grises
                for k, line in ((i, ax.axvline), (j, ax.axhline)):
                    if k > 0:
                        line(0, color="#999999", lw=0.7, zorder=0)
                if (i, j) == (1, 2):
                    ax.set_aspect("equal", adjustable="datalim")   # distancias de color reales
            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")
        n_img = len({p.path for p in pts})
        n_obj = len({(p.path, p.object_id) for p in pts if p.object_id is not None})
        title = f"{space}  ·  " + t("chart.n_images", n=n_img) + (f"  ·  {t('chart.n_objects', n=n_obj)}" if n_obj else "")
        ax.set_title(title, fontsize=9)
        if self.legend.get():
            self._legend(fig, levels)
        if three:
            if self._view and self._view[5] == space:      # mismo giro; límites según los puntos de ahora
                ax.view_init(*self._view[:2])
            else:
                ax.view_init(22, -60)
            self._home = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
        else:
            ax.margins(0.08)
            fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.93)
            self._home = None
        self._canvas.draw_idle()

    def _point_name(self, p) -> str:
        return p.name.rsplit(".", 1)[0] + (f"-{p.object_id}" if p.object_id is not None else "")

    @staticmethod
    def _pies(ax, pts, xyz, sizes, kw=None):
        """Cada punto es una torta con sus colores (sectores = % de cada color). Los
        sectores con la misma forma van juntos: pocas llamadas aunque haya muchos puntos."""
        groups: dict[tuple[int, int], list[tuple[int, tuple]]] = {}
        for i, p in enumerate(pts):
            total = sum(pct for _, pct in p.sectors) or 1
            acc = 0.0
            for rgb, pct in sorted(p.sectors, key=lambda s: -s[1]):
                a = round(acc / total * STEPS)
                acc += pct
                b = round(acc / total * STEPS)
                if b > a:
                    groups.setdefault((a, b), []).append((i, rgb))
        for (a, b), items in groups.items():
            ii = [i for i, _ in items]
            ax.scatter(*xyz[ii].T, c=[rgb for _, rgb in items], s=sizes[ii], marker=_wedge(a, b),
                       linewidths=0, **(kw or {}))
        ax.scatter(*xyz.T, s=sizes, facecolors="none", edgecolors=EDGE, linewidths=0.5, **(kw or {}))

    def _legend(self, fig, levels):
        batch = self.scope.get() == "batch"
        marks = {"object_colors": ("o", 4), "objects": ("o", 8), "image_colors": ("o", 6), "images": ("o", 11)}
        handles = []
        for k in levels:
            m, ms = marks[k]
            n = sum(1 for p in self._pts if p.level == k)
            text = t(f"c3d.level.{k}" + ("_batch" if batch and k in ("image_colors", "images") else ""))
            handles.append(Line2D([], [], marker=m, ls="", ms=ms, mfc="#BBBBBB", mec=EDGE, mew=0.6,
                                  label=f"{text} ({n})"))
        if handles:
            fig.legend(handles=handles, loc="upper left", frameon=True, fontsize=8, framealpha=0.85,
                       title=t("c3d.legend_note"), title_fontsize=7, alignment="left")

    # ── navegación ────────────────────────────────────────────────────────────

    def _on_scroll(self, e):
        if self._ax is None:
            return
        k = 0.85 if e.button == "up" else 1 / 0.85
        if not self.is3d:                               # 2D: acerca hacia el cursor
            if e.xdata is None:
                return
            for (lo, hi), c, put in ((self._ax.get_xlim(), e.xdata, self._ax.set_xlim),
                                     (self._ax.get_ylim(), e.ydata, self._ax.set_ylim)):
                put(c - (c - lo) * k, c + (hi - c) * k)
            self._canvas.draw_idle()
            return
        for get, put in ((self._ax.get_xlim3d, self._ax.set_xlim3d), (self._ax.get_ylim3d, self._ax.set_ylim3d),
                         (self._ax.get_zlim3d, self._ax.set_zlim3d)):
            lo, hi = get()
            c, r = (lo + hi) / 2, (hi - lo) / 2 * k
            put(c - r, c + r)
        self._canvas.draw_idle()

    def _reset_view(self):
        if self._ax is None:
            return
        if not self.is3d:
            self._redraw()
            return
        self._ax.view_init(22, -60)
        xl, yl, zl = self._home
        self._ax.set_xlim3d(xl)
        self._ax.set_ylim3d(yl)
        self._ax.set_zlim3d(zl)
        self._canvas.draw_idle()

    # ── identificar un punto ──────────────────────────────────────────────────

    def _on_press(self, e):
        self._press = (e.x, e.y)
        self._press_view = (self._ax.elev, self._ax.azim) if self._ax is not None and e.button == 1 \
            and self.is3d else None

    def _on_drag(self, e):
        """Arrastrar con el botón izquierdo = girar (0.4° por píxel)."""
        pv, p0 = getattr(self, "_press_view", None), getattr(self, "_press", None)
        if not self.is3d or pv is None or p0 is None or e.x is None or self._ax is None:
            return
        elev = float(np.clip(pv[0] - (e.y - p0[1]) * 0.4, -90, 90))
        self._ax.view_init(elev, pv[1] - (e.x - p0[0]) * 0.4)
        self._canvas.draw_idle()

    def _on_release(self, e):
        self._press_view = None
        p0 = getattr(self, "_press", None)
        if self._ax is None or p0 is None or e.button != 1 or e.x is None:
            return
        if abs(e.x - p0[0]) + abs(e.y - p0[1]) > 4:        # fue un giro, no un clic
            return
        i = self.nearest(e.x, e.y)
        if i is not None:
            self._show_point(i)

    def screen_xy(self) -> np.ndarray:
        xyz = np.array([p.xyz for p in self._pts])
        if not self.is3d:
            return self._ax.transData.transform(xyz[:, list(self._plane())])
        x, y, _ = proj3d.proj_transform(xyz[:, 0], xyz[:, 1], xyz[:, 2], self._ax.get_proj())
        return self._ax.transData.transform(np.c_[x, y])

    def nearest(self, x: float, y: float) -> int | None:
        if not self._pts:
            return None
        xy = self.screen_xy()
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        r = np.sqrt(self._sizes) / 2 * self._fig.dpi / 72 + 3
        hit = np.where(d <= r)[0]
        if not len(hit):
            return None
        rank = {"object_colors": 0, "image_colors": 1, "objects": 2, "images": 3}   # el más chico encima
        return int(min(hit, key=lambda i: (rank[self._pts[i].level], d[i])))

    def _show_point(self, i: int):
        p = self._pts[i]
        if self._picked is not None:
            try:
                self._picked.remove()
            except Exception:
                pass
        at = np.array([p.xyz]) if self.is3d else np.array([p.xyz])[:, list(self._plane())]
        self._picked = self._ax.scatter(*at.T, s=self._sizes[i] * 2.2 + 60, facecolors="none", edgecolors="#D73027",
                                        linewidths=1.6, zorder=11, **({"depthshade": False} if self.is3d else {}))
        self._canvas.draw_idle()
        info = self.info
        for w in info.winfo_children():
            w.destroy()
        if not info.winfo_ismapped():
            info.pack(side=tk.LEFT, fill=tk.Y, before=self._frame)
        bg = COLORS["bg_card"]
        head = tk.Frame(info, bg=bg)
        head.pack(fill=tk.X, padx=10, pady=(8, 4))
        tk.Label(head, text=t("c3d.point"), bg=bg, fg=COLORS["accent"], font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        x = tk.Label(head, text="✕", bg=bg, fg=COLORS["text_muted"], cursor="hand2", font=FONTS["small"])
        x.pack(side=tk.RIGHT)
        x.bind("<Button-1>", lambda e: info.pack_forget())
        img = self.thumb(p.path, p.object_id) if self.thumb else None
        if img is not None:
            im = Image.fromarray(img[:, :, ::-1])
            im.thumbnail((206, 170))
            self._thumb_img = ImageTk.PhotoImage(im)
            tk.Label(info, image=self._thumb_img, bg=bg).pack(padx=10, pady=(0, 6))
        rows = [(t("c3d.photo"), f"{p.image_id} · {p.name}")]
        if p.object_id is not None:
            rows.append((t("c3d.object"), str(p.object_id)))
        if p.level in ("object_colors", "image_colors"):
            rows.append(("%", f"{p.pct:.1f}"))
        rows.append((self.space.get(), ", ".join(f"{v:.0f}" for v in p.xyz)))
        rgb = tuple(int(round(v * 255)) for v in p.rgb)
        rows.append(("HEX", "#%02x%02x%02x" % rgb))
        grid = tk.Frame(info, bg=bg)
        grid.pack(fill=tk.X, padx=10)
        for r, (k, v) in enumerate(rows):
            tk.Label(grid, text=k, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"], anchor="w").grid(
                row=r, column=0, sticky="w")
            tk.Label(grid, text=v, bg=bg, fg=COLORS["text"], font=FONTS["small"], anchor="e",
                     wraplength=150, justify="right").grid(row=r, column=1, sticky="e", padx=(8, 0))
        grid.columnconfigure(1, weight=1)
        sw = tk.Frame(info, bg="#%02x%02x%02x" % rgb, height=14)
        sw.pack(fill=tk.X, padx=10, pady=(6, 0))
        if p.sectors:
            bar = tk.Frame(info, bg=bg, height=10)
            bar.pack(fill=tk.X, padx=10, pady=(2, 0))
            total, acc = sum(pc for _, pc in p.sectors) or 1, 0.0
            for rgb_s, pc in sorted(p.sectors, key=lambda s: -s[1]):
                tk.Frame(bar, bg="#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgb_s)).place(
                    relx=acc / total, rely=0, relwidth=pc / total, relheight=1)
                acc += pc
        if self.on_goto:
            tk.Button(info, text=t("c3d.goto"), command=lambda: self.on_goto(p.path, p.object_id),
                      bg=COLORS["accent"], fg="#FFFFFF", relief="flat", font=FONTS["small"], cursor="hand2",
                      padx=10).pack(pady=10)
        self._picked_point = p
