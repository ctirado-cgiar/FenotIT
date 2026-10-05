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
        combo(top, self.space, list(cs.SPACES), 6)
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
        self.layers = {k: tk.BooleanVar(value=False) for k in cs.LEVELS}
        self.warn = tk.Label(self.layer_bar, text="", bg=bg, fg="#B35806", font=FONTS["small"])

        body = tk.Frame(self, bg=COLORS["bg_card"])
        body.pack(fill=tk.BOTH, expand=True)
        self.info = tk.Frame(body, bg=COLORS["bg_card"], width=230, highlightthickness=1,
                             highlightbackground=COLORS["border"])
        self.info.pack_propagate(False)
        self._frame = tk.Frame(body, bg=COLORS["bg_card"])
        self._frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scope_changed()

    # ── datos ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _has(images) -> bool:
        return bool(images) and any(tb.get("object_colors") or tb.get("image_colors") for *_, tb in images)

    def _available(self) -> list[str]:
        imgs = self.data[self.scope.get()]
        have_obj = any(tb.get("object_colors") for *_, tb in imgs)
        return [k for k in cs.LEVELS if have_obj or k in ("image_colors", "images")]

    def _scope_changed(self):
        for w in self.layer_bar.winfo_children():
            if w is not self.warn:
                w.destroy()
        bg = COLORS["bg_panel"]
        tk.Label(self.layer_bar, text=t("c3d.layers") + ":", bg=bg, fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(10, 2))
        avail = self._available()
        batch = self.scope.get() == "batch"
        default = "images" if batch else ("objects" if "objects" in avail else "image_colors")
        for k in avail:
            self.layers[k].set(k == default)
            text = t(f"c3d.level.{k}" + ("_batch" if batch and k in ("image_colors", "images") else ""))
            tk.Checkbutton(self.layer_bar, text=text, variable=self.layers[k], command=self._redraw, bg=bg,
                           fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg,
                           font=FONTS["small"]).pack(side=tk.LEFT, padx=(4, 0))
        self.warn.pack(side=tk.LEFT, padx=12)
        self._redraw()

    # ── dibujo ────────────────────────────────────────────────────────────────

    def _redraw(self):
        if self._ax is not None:                         # se conserva el giro y el zoom
            self._view = (self._ax.elev, self._ax.azim, self._ax.get_xlim3d(), self._ax.get_ylim3d(),
                          self._ax.get_zlim3d(), self._drawn_space)
        levels = [k for k in self._available() if self.layers[k].get()]
        space = self.space.get()
        pts = cs.points(self.data[self.scope.get()], levels, space)
        self._pts, self._picked = pts, None
        self._drawn_space = space
        self.info.pack_forget()
        self.warn.configure(text=t("c3d.many", n=len(pts)) if len(pts) > MANY else "")
        if not pts:
            self._fig = self._ax = None
            _message(self._frame, t("c3d.no_points"))
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
        fig = self._fig
        fig.clear()
        ax = self._ax = fig.add_subplot(111, projection="3d")
        ax.mouse_init(rotate_btn=1, zoom_btn=3)
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
        for level in cs.LEVELS:
            idx = [i for i, p in enumerate(pts) if p.level == level]
            if not idx:
                continue
            if level in ("objects", "images"):
                self._pies(ax, [pts[i] for i in idx], xyz[idx], self._sizes[idx])
            else:
                ax.scatter(*xyz[idx].T, c=[pts[i].rgb for i in idx], s=self._sizes[idx], depthshade=False,
                           edgecolors=EDGE, linewidths=0.3, alpha=0.9)
        mode = self.labels.get(self.label_var.get())
        if mode:
            ax.computed_zorder = False          # las etiquetas siempre encima de los puntos
            halo = [patheffects.withStroke(linewidth=2.2, foreground="white")]
            top = [p for p in pts if p.level in ("images", "objects")] or pts
            size = {id(p): sz for p, sz in zip(pts, self._sizes)}
            for p in top[:200]:
                pad = " " * (int(np.sqrt(size[id(p)]) / 3.4) + 1)      # justo al lado del punto
                ax.text(*p.xyz, pad + (p.label_id if mode == "id" else self._point_name(p)), fontsize=6,
                        color="#222222", zorder=10, path_effects=halo)
        names = cs.SPACES[space]
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2])
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_facecolor((0.97, 0.97, 0.97, 1))
        n_img = len({p.path for p in pts})
        n_obj = len({(p.path, p.object_id) for p in pts if p.object_id is not None})
        title = f"{space}  ·  " + t("chart.n_images", n=n_img) + (f"  ·  {t('chart.n_objects', n=n_obj)}" if n_obj else "")
        ax.set_title(title, fontsize=9)
        if self.legend.get():
            self._legend(fig, levels)
        if self._view and self._view[5] == space:      # mismo giro; límites según los puntos de ahora
            ax.view_init(*self._view[:2])
        else:
            ax.view_init(22, -60)
        self._home = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
        self._canvas.draw_idle()

    def _point_name(self, p) -> str:
        return p.name.rsplit(".", 1)[0] + (f"-{p.object_id}" if p.object_id is not None else "")

    @staticmethod
    def _pies(ax, pts, xyz, sizes):
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
                       depthshade=False, linewidths=0)
        ax.scatter(*xyz.T, s=sizes, facecolors="none", edgecolors=EDGE, linewidths=0.5, depthshade=False)

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
        for get, put in ((self._ax.get_xlim3d, self._ax.set_xlim3d), (self._ax.get_ylim3d, self._ax.set_ylim3d),
                         (self._ax.get_zlim3d, self._ax.set_zlim3d)):
            lo, hi = get()
            c, r = (lo + hi) / 2, (hi - lo) / 2 * k
            put(c - r, c + r)
        self._canvas.draw_idle()

    def _reset_view(self):
        if self._ax is None:
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

    def _on_release(self, e):
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
        self._picked = self._ax.scatter(*np.array([p.xyz]).T, s=self._sizes[i] * 2.2 + 60, facecolors="none",
                                        edgecolors="#D73027", linewidths=1.6, depthshade=False)
        self._canvas.draw_idle()
        info = self.info
        for w in info.winfo_children():
            w.destroy()
        if not info.winfo_ismapped():
            info.pack(side=tk.RIGHT, fill=tk.Y, before=self._frame)
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
        batch = self.scope.get() == "batch"
        rows.append((t("c3d.layer"), t(f"c3d.level.{p.level}" + ("_batch" if batch and p.level in
                                                                   ("image_colors", "images") else ""))))
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
