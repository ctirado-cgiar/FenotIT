"""Distribución de la ventana: barra superior, paneles izquierdo, central y derecho,
barra de estado, anchos de los paneles y cambio de tamaño de la ventana.

PANELS dice qué secciones van en cada panel lateral, de arriba hacia abajo. Para mover
una sección a otro panel, se cambia de lista; cada sección se arma en el panel que le
toque con su función `_build_<sección>`.

  images     lista de fotos
  layers     lista CAPAS
  inspector  objeto elegido
  steps      pestañas «1 Segmentar» y «2 Analizar»
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from fenotit import settings
from fenotit.gui.style.components import button, divider
from fenotit.gui.style.theme import COLORS, FONTS, SIZES, SPACE
from fenotit.gui.widgets.collapsible import CollapsiblePanel
from fenotit.gui.widgets.image_list import ImageList
from fenotit.gui.widgets.toolbar import Tooltip
from fenotit.i18n import t


PANELS = {
    "left": ("images", "layers", "inspector"),
    "right": ("steps",),
}


class LayoutMixin:
    def _build_panels(self):
        """Arma cada sección de PANELS dentro de su panel lateral."""
        for side, sections in PANELS.items():
            parent = getattr(self, f"{side}_panel").content
            for name in sections:
                getattr(self, f"_build_{name}")(parent)

    def _panel_of(self, section: str) -> CollapsiblePanel:
        """El panel lateral donde está la sección."""
        side = next(s for s, names in PANELS.items() if section in names)
        return getattr(self, f"{side}_panel")

    def _build_layout(self):
        self.topbar = tk.Frame(self.root, bg=COLORS["bg_topbar"], height=40)
        self.topbar.pack(fill=tk.X, side=tk.TOP)
        self.topbar.pack_propagate(False)

        # PanedWindow horizontal para los 3 paneles
        self.paned = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bg=COLORS["sash"],
            sashwidth=5,
            sashrelief="flat",
            sashpad=0,
            showhandle=False,
            opaqueresize=True)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo colapsable
        self.left_panel = CollapsiblePanel(
            self.paned, side="left",
            title=t("panel.left"), colors=COLORS,
            default_width=SIZES["panel_left"], on_toggle=self._after_panel_toggle, max_width=420)
        self.paned.add(self.left_panel, minsize=CollapsiblePanel.COLLAPSED_W,
                       width=SIZES["panel_left"], stretch="never")

        # Centro
        self.center_frame = tk.Frame(self.paned, bg=COLORS["bg"])
        self.left_panel.tab_host = self.center_frame
        self.paned.add(self.center_frame, minsize=400, stretch="always")

        # Panel derecho colapsable
        self.right_panel = CollapsiblePanel(
            self.paned, side="right",
            title=t("panel.right"), colors=COLORS,
            default_width=SIZES["panel_right"], on_toggle=self._after_panel_toggle, max_width=380)
        self.right_panel.tab_host = self.center_frame
        for w in (self.paned, self.left_panel, self.right_panel):
            w.bind("<Configure>", self._limit_panels, add="+")
        self.paned.bind("<ButtonPress-1>", self._sash_press)
        self.paned.bind("<B1-Motion>", self._sash_motion)
        self.paned.bind("<ButtonRelease-1>", self._sash_release)
        self.paned.add(self.right_panel, minsize=CollapsiblePanel.COLLAPSED_W,
                       width=SIZES["panel_right"], stretch="never")      # si no, el último panel se queda con todo el espacio que sobra

        # Statusbar
        self.statusbar = tk.Frame(self.root, bg=COLORS["bg_panel"], height=24)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)
        self.statusbar.pack_propagate(False)
        divider(self.statusbar).pack(fill=tk.X)
        self.status_var = tk.StringVar(value=t("status.ready"))
        self.corr_var = tk.StringVar(value="")
        self._corr_label = tk.Label(self.statusbar, textvariable=self.corr_var,
                                    bg=COLORS["bg_panel"], fg=COLORS["accent"],
                                    font=FONTS["small"], padx=SPACE["lg"])
        self._corr_label.pack(side=tk.RIGHT)
        self._corr_tip = Tooltip(self._corr_label, t("corr.applied_tip"))
        self._progress_bar = ttk.Progressbar(self.statusbar, length=160, mode="determinate",
                                             style="Thin.Horizontal.TProgressbar")
        self._cancel_link = tk.Label(self.statusbar, text=t("common.cancel"), bg=COLORS["bg_panel"],
                                     fg=COLORS["accent"], font=FONTS["small"], cursor="hand2", padx=SPACE["sm"])
        self._cancel_link.bind("<Button-1>", lambda e: self._cancel_batch())
        self.preview_var = tk.StringVar(value="")
        tk.Label(self.statusbar, textvariable=self.preview_var, bg=COLORS["bg_panel"],
                 fg=COLORS["accent2"], font=FONTS["small"], padx=SPACE["md"]).pack(side=tk.RIGHT)
        self.image_info_var = tk.StringVar(value="")
        tk.Label(self.statusbar, textvariable=self.image_info_var, bg=COLORS["bg_panel"],
                 fg=COLORS["text"], font=FONTS["small"], padx=SPACE["md"]).pack(side=tk.RIGHT)
        self._status_label = tk.Label(self.statusbar, textvariable=self.status_var,
                                      bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                                      font=FONTS["small"], anchor="w", padx=SPACE["lg"])
        self._status_label.pack(fill=tk.X)

    def _build_images(self, lf):
        """Lista de fotos con su botón para agregar y el contador."""
        self._section_lbl(lf, t("left.images"))
        nav = tk.Frame(lf, bg=COLORS["bg_panel"])
        nav.pack(fill=tk.X, padx=SPACE["md"], pady=SPACE["xs"])
        button(nav, text="＋", command=self._open_image, width=2, kind="secondary", size="small").pack(side=tk.LEFT)
        self.batch_label = tk.Label(nav, text="—", bg=COLORS["bg_panel"],     # anterior/siguiente: ← → o clic
                                    fg=COLORS["text_muted"], font=FONTS["small"])
        self.batch_label.pack(side=tk.RIGHT, padx=SPACE["sm"])

        # Lista (buscador, lista o cuadrícula): alto según las fotos. Si CAPAS (y, después,
        # el inspector) necesita espacio, la lista cede, hasta dejar ver un par de fotos
        self.image_list = ImageList(lf, COLORS, FONTS, on_select=self._select_image,
                                    on_delete=self._remove_current_image,
                                    mode=settings.get("image_view", "list"),
                                    on_mode=lambda m: settings.set("image_view", m),
                                    max_height=self._image_list_room)
        self.image_list.pack(fill=tk.X)
        self.image_list.min_rows = lambda: 1 if self.inspector.winfo_ismapped() else 2   # con el inspector, cede más
        self.image_list.search_text = lambda p: " ".join(str(v) for v in self._meta_row(p).values())

    def _build_layers(self, lf):
        """Lista CAPAS: las capas de los análisis de la foto actual (con barra si no caben)."""
        views = self._layers = tk.Frame(lf, bg=COLORS["bg_panel"])
        views.pack(fill=tk.BOTH, expand=True)
        self._layers_title = self._section_lbl(views, t("left.history"))
        box = tk.Frame(views, bg=COLORS["bg_panel"])
        box.pack(fill=tk.BOTH, expand=True, padx=SPACE["md"], pady=(0, SPACE["md"]))
        self._layers_sb = ttk.Scrollbar(box, orient="vertical")
        cv = self._layers_canvas = tk.Canvas(box, bg=COLORS["bg_panel"], highlightthickness=0, bd=0,
                                             yscrollcommand=self._layers_sb.set)
        self._layers_sb.config(command=cv.yview)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.history_frame = tk.Frame(cv, bg=COLORS["bg_panel"])
        win = cv.create_window(0, 0, window=self.history_frame, anchor="nw")
        cv.bind("<Configure>", lambda e: (cv.itemconfig(win, width=e.width), self._layers_scroll()))
        self.history_frame.bind("<Configure>", lambda e: self._layers_scroll())
        for ev, d in (("<Button-4>", -1), ("<Button-5>", 1)):
            cv.bind_all(ev, lambda e, d=d: self._layers_wheel(e, d), add="+")
        cv.bind_all("<MouseWheel>", lambda e: self._layers_wheel(e, -1 if e.delta > 0 else 1), add="+")

    def _build_inspector(self, lf):
        """Inspector del objeto elegido; aparece al elegir un objeto."""
        from fenotit.gui.widgets.inspector import Inspector
        self.inspector = Inspector(lf, self._clear_selection, self._toggle_exclude, self._toggle_highlight)
        self._forget_selection()

    def _build_steps(self, rf):
        """Panel del análisis: sin análisis elegido, las tarjetas para elegirlo; con uno
        elegido, su nombre (con Cambiar) y las pestañas de sus pasos."""
        bg = COLORS["bg_panel"]
        head = self._an_head = tk.Frame(rf, bg=bg)
        self._an_title = tk.Label(head, bg=bg, fg=COLORS["text"], font=FONTS["title"], anchor="w")
        self._an_title.pack(side=tk.LEFT)
        button(head, text=t("rtab.change"), command=lambda: self._select_right_tab("choose"), kind="link",
               size="small", bg=bg).pack(side=tk.RIGHT)
        self._rtabs = tk.Frame(rf, bg=bg)
        self._rtab_labels = {}
        self._rtab_line = divider(rf)
        self._rtab_body = tk.Frame(rf, bg=bg)
        self._rtab_body.pack(fill=tk.BOTH, expand=True)
        self._choose_tab = tk.Frame(self._rtab_body, bg=bg)
        tk.Label(self._choose_tab, text=t("rtab.choose"), bg=bg, fg=COLORS["section_text"], font=FONTS["section"],
                 anchor="w").pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["md"], SPACE["xs"]))
        self._analysis_cards = tk.Frame(self._choose_tab, bg=bg)
        self._analysis_cards.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["xs"], SPACE["md"]))
        self._seg_tab = tk.Frame(self._rtab_body, bg=bg)
        self._an_tab = tk.Frame(self._rtab_body, bg=bg)
        self._build_seg_tab(self._seg_tab)
        self.config_container = tk.Frame(self._an_tab, bg=bg)
        self.config_container.pack(fill=tk.BOTH, expand=True, padx=SPACE["sm"])
        self._right_tab = "choose"
        self._select_right_tab("choose")

    def _render_step_tabs(self, steps: list[tuple[str, str]]):
        """Pestañas de los pasos del análisis elegido: (clave, nombre); con un solo paso, sin
        pestañas."""
        bg = COLORS["bg_panel"]
        for w in self._rtabs.winfo_children():
            w.destroy()
        self._rtab_labels = {}
        for key, text in steps if len(steps) > 1 else ():
            tab = tk.Frame(self._rtabs, bg=bg, cursor="hand2")
            tab.pack(side=tk.LEFT, padx=(0, SPACE["lg"]))
            lbl = tk.Label(tab, text=text, bg=bg, font=FONTS["body"], cursor="hand2", pady=SPACE["sm"])
            lbl.pack()
            line = tk.Frame(tab, height=2, bg=bg)
            line.pack(fill=tk.X)
            for w in (tab, lbl):
                w.bind("<Button-1>", lambda e, k=key: self._select_right_tab(k))
            self._rtab_labels[key] = (lbl, line)

    def _select_right_tab(self, key: str):
        """Muestra en el panel derecho las tarjetas («choose») o un paso («seg», «analysis»)."""
        if key != "choose" and not getattr(self, "_analysis_chosen", False):
            key = "choose"
        self._right_tab = key
        for w in (self._an_head, self._rtabs, self._rtab_line, self._choose_tab, self._seg_tab, self._an_tab):
            w.pack_forget()
        self._rtab_body.pack_forget()
        if key != "choose":
            self._an_head.pack(fill=tk.X, padx=(SPACE["xxl"], SPACE["md"]), pady=(SPACE["sm"], 0))
            if self._rtab_labels:
                self._rtabs.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["xs"], 0))
            self._rtab_line.pack(fill=tk.X)
        self._rtab_body.pack(fill=tk.BOTH, expand=True, pady=(SPACE["lg"] if key == "choose" else 0, 0))
        {"choose": self._choose_tab, "seg": self._seg_tab, "analysis": self._an_tab}[key].pack(fill=tk.BOTH, expand=True)
        for k, (lbl, line) in self._rtab_labels.items():
            on = k == key
            lbl.config(fg=COLORS["accent"] if on else COLORS["text_muted"])
            line.config(bg=COLORS["accent"] if on else COLORS["bg_panel"])
        self._update_seg_tab()
        self._update_stroke_tool()
        if hasattr(self, "zoom_ctrl"):
            self._on_slider_change()

    def _visible_panes(self):
        """Paneles visibles, como widgets (panes() devuelve nombres de Tcl)."""
        pw = self.paned
        names = {str(p) for p in pw.panes() if str(pw.panecget(p, "hide")) not in ("1", "true")}
        return [w for w in (self.left_panel, self.center_frame, self.right_panel) if str(w) in names]

    def _sash_press(self, event):
        """Arrastre propio de las barras divisorias: el panel lateral llega como máximo a su
        `max_width` y el centro nunca baja de 400 px (Tk solo, empujaba las otras barras)."""
        hit = self.paned.identify(event.x, event.y)
        if not hit or "sash" not in str(hit):
            self._sash_drag = None
            return
        self._sash_drag = int(hit[0])
        return "break"

    def _sash_motion(self, event):
        index = getattr(self, "_sash_drag", None)
        if index is None:
            return
        pw = self.paned
        visible = self._visible_panes()
        if index + 1 >= len(visible):
            return "break"
        sw = int(pw.cget("sashwidth"))
        total = pw.winfo_width()
        lo, hi = CollapsiblePanel.COLLAPSED_W, total - CollapsiblePanel.COLLAPSED_W - sw
        if visible[index] is self.left_panel:                     # barra izquierda
            hi = min(self.left_panel.max_width,
                     (pw.sash_coord(index + 1)[0] if index + 1 < len(visible) - 1 else total) - 400 - sw)
        elif visible[index + 1] is self.right_panel:              # barra derecha
            start = pw.sash_coord(index - 1)[0] + sw if index > 0 else 0
            lo = max(total - self.right_panel.max_width - sw, start + 400)
        x = int(min(max(event.x, lo), max(lo, hi)))
        pw.sash_place(index, x, 1)
        return "break"

    def _sash_release(self, _event=None):
        if getattr(self, "_sash_drag", None) is None:
            return
        self._sash_drag = None
        self._after_panel_toggle()
        return "break"

    def _limit_panels(self, _event=None):
        """Si al achicar la ventana un panel quedó aplastado, al agrandarla recupera su ancho."""
        pw = self.paned
        if pw.winfo_width() <= 1 or getattr(self, "_restoring_panel", False):
            return
        visible = self._visible_panes()
        room = self.center_frame.winfo_width() - 400
        for panel in (self.left_panel, self.right_panel):
            w_now = panel.winfo_width() if panel in visible else 0
            if (panel.winfo_ismapped() and panel._expanded and CollapsiblePanel.COLLAPSED_W <= w_now
                    < 0.6 * panel.default_width and room > 40):
                self._restoring_panel = True
                target = int(min(panel.default_width, w_now + room))

                def restore(panel=panel, target=target):
                    self._restoring_panel = False
                    vis = self._visible_panes()
                    if panel is self.left_panel:
                        pw.sash_place(0, target, 1)
                    else:
                        pw.sash_place(len(vis) - 2, pw.winfo_width() - target - int(pw.cget("sashwidth")), 1)
                self.root.after(60, restore)
                return

    def _bind_resize(self):
        self._after_resize_id = None
        self.root.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        if event.widget != self.root:
            return
        if self._after_resize_id:
            self.root.after_cancel(self._after_resize_id)
        self._after_resize_id = self.root.after(80, self._do_resize)

    def _do_resize(self):
        self.root.update_idletasks()
        self.canvas_left.update_idletasks()
        # Redibujar con zoom actual
        self._show_preview()
