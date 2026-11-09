"""Panel de abajo: pestañas Tabla y Gráficos, su alto y el gráfico llevado arriba."""
from __future__ import annotations

import tkinter as tk

from tkinter import ttk

from PIL import ImageTk

from fenotit import settings
from fenotit.gui.charts.panels import IntraImageChartPanel
from fenotit.gui.style.icons import icon
from fenotit.gui.style.components import button
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.gui.widgets.toolbar import IconButton, Tooltip
from fenotit.i18n import t


class BottomMixin:
    def _build_bottom_panel(self, cf):
        """Tabla y gráficos: pestañas propias; ocultar deja solo las pestañas (al tocarlas se
        vuelve a abrir). Arriba, una franja para cambiar la altura arrastrando."""
        self.step_label_var = tk.StringVar(value="—")
        wrap = self._bottom_wrap = tk.Frame(cf, bg=COLORS["bg"])
        grip = self._bottom_grip = tk.Frame(wrap, bg=COLORS["bg"], height=6, cursor="sb_v_double_arrow")
        grip.bind("<ButtonPress-1>", self._results_drag_start)
        grip.bind("<B1-Motion>", self._results_drag)
        grip.bind("<ButtonRelease-1>", self._results_drag_end)
        bar = self._bottom_bar = tk.Frame(wrap, bg=COLORS["bg"])
        self._tab_labels = {}
        for key, text in (("table", t("tab.table")), ("charts", t("tab.charts"))):
            lbl = tk.Label(bar, text=text, font=FONTS["small"], padx=SPACE["lg"], pady=SPACE["sm"], cursor="hand2", bd=0,
                           highlightthickness=1, highlightbackground=COLORS["border"])
            lbl.pack(side=tk.LEFT, padx=(0, SPACE["xs"]))
            lbl.bind("<Button-1>", lambda e, k=key: self._select_bottom_tab(k))
            self._tab_labels[key] = lbl
        self._panel_bottom_icon = ImageTk.PhotoImage(icon("panel_bottom", 12, COLORS["accent"]))
        corner = tk.Label(bar, image=self._panel_bottom_icon, bg=COLORS["bg"], cursor="hand2", padx=SPACE["sm"], pady=SPACE["xs"])
        corner.pack(side=tk.LEFT, padx=SPACE["md"])
        corner.bind("<Button-1>", lambda e: self._toggle_results_panel())
        Tooltip(corner, t("view.toggle_results"))
        self._meta_var = tk.StringVar(value="")               # valores de la tabla del usuario de esta foto
        self._meta_label = tk.Label(bar, textvariable=self._meta_var, bg=COLORS["bg"], fg=COLORS["text_muted"],
                                    font=FONTS["small"], anchor="e")
        self._meta_label.pack(side=tk.RIGHT, padx=(SPACE["md"], SPACE["sm"]))
        body = self._bottom_body = tk.Frame(wrap, bg=COLORS["bg_card"], highlightthickness=1,
                                            highlightbackground=COLORS["border"])
        body.pack_propagate(False)
        try:
            height = int(settings.get("results_height", 160))
        except (TypeError, ValueError):
            height = 160
        body.configure(height=max(self.RESULTS_MIN_H, height))
        # Preferencias → "Abrir la tabla al ejecutar": si no, solo quedan las pestañas a la vista
        self._results_visible = bool(settings.get("show_table_on_run", True))
        self._bottom_tab = "table"

        # ── Tabla ─────────────────────────────────────────────────────────
        tab_table = tk.Frame(body, bg=COLORS["bg_card"])
        tv_f = tk.Frame(tab_table, bg=COLORS["bg_card"])
        tv_f.pack(fill=tk.BOTH, expand=True)
        xsb = ttk.Scrollbar(tv_f, orient="horizontal")
        ysb = ttk.Scrollbar(tv_f, orient="vertical")
        self.table = ttk.Treeview(tv_f, show="headings",
                                   xscrollcommand=xsb.set,
                                   yscrollcommand=ysb.set, height=4)
        xsb.config(command=self.table.xview)
        ysb.config(command=self.table.yview)
        ysb.pack(side=tk.RIGHT,  fill=tk.Y)
        self._table_xsb = xsb                   # solo aparece si las columnas no caben
        self.table.pack(fill=tk.BOTH, expand=True)
        self.table.bind("<Configure>", self._fit_table_columns, add="+")
        self.table.bind("<<TreeviewSelect>>", self._on_table_select, add="+")

        # ── Gráficos (hijo de la zona central: puede pasar al lugar de una imagen) ──
        tab_charts = tk.Frame(body, bg=COLORS["bg_card"])
        self._tab_frames = {"table": tab_table, "charts": tab_charts}
        self._chart_panel = IntraImageChartPanel(
            cf, colors=COLORS, on_pick=lambda oid: self._goto_object(self.current_image_path, oid))
        self._chart_panel.pack(in_=tab_charts, fill=tk.BOTH, expand=True)
        if hasattr(self._chart_panel, "bar"):
            self._dock_icon = ImageTk.PhotoImage(icon("dock_up", 14, COLORS["accent"]))
            self._dock_btn = tk.Label(self._chart_panel.bar, image=self._dock_icon, bg=COLORS["bg_panel"],
                                      cursor="hand2", padx=SPACE["sm"])
            self._dock_btn.pack(side=tk.RIGHT, padx=(SPACE["xs"], SPACE["md"]))
            self._dock_btn.bind("<Button-1>", lambda e: self._dock_chart())
            Tooltip(self._dock_btn, t("chart.dock"))
            self._full_icon = ImageTk.PhotoImage(icon("expand", 14, COLORS["accent"]))
            full = tk.Label(self._chart_panel.bar, image=self._full_icon, bg=COLORS["bg_panel"], cursor="hand2", padx=SPACE["sm"])
            full.pack(side=tk.RIGHT, padx=(SPACE["xs"], 0))
            full.bind("<Button-1>", lambda e: self._chart_fullscreen())
            Tooltip(full, t("chart.fullscreen"))
            save = self._save_btn = tk.Menubutton(self._chart_panel.bar, text=t("chart.save") + " ▾", bg=COLORS["btn_bg"],
                                 fg=COLORS["text_muted"], relief="flat", font=FONTS["small"], cursor="hand2", pady=SPACE["xs"])
            menu = tk.Menu(save, tearoff=0)
            menu.add_command(label=t("chart.save_chart"), command=lambda: self._save_intra_chart(False))
            menu.add_command(label=t("chart.save_with_image"), command=lambda: self._save_intra_chart(True))
            save.configure(menu=menu)
            save.pack(side=tk.RIGHT, padx=SPACE["xs"], pady=SPACE["sm"])
            self._compare_btn = button(self._chart_panel.bar, text=t("chart.whole_batch"), command=self._show_batch_chart, kind="secondary", size="small")
            self._compare_btn.pack(side=tk.RIGHT, padx=SPACE["xs"], pady=SPACE["sm"])
            self._colors_btn = button(self._chart_panel.bar, text=t("c3d.open"), command=lambda: self._show_color3d("image"), kind="secondary", size="small")      # solo en la capa Color

    def _build_chart_dock(self, canvas_row):
        """Lugar del gráfico cuando se lleva arriba, en vez de una de las imágenes."""
        self._results_shown = False
        self._chart_dock, self._dock_saved = None, None
        cc = self._chart_col = tk.Frame(canvas_row, bg=COLORS["bg"])
        head = tk.Frame(cc, bg=COLORS["bg"])
        head.pack(fill=tk.X)
        self._chart_title = tk.Label(head, text=t("tab.charts"), bg=COLORS["bg"], fg=COLORS["text_muted"],
                                     font=FONTS["small"])
        self._chart_eye = IconButton(head, "eye", self._undock_chart, t("chart.undock"), bg=COLORS["bg"],
                                     hover=COLORS["btn_hover"], active=COLORS["accent_light"], size=14,
                                     color=COLORS["text_muted"])
        self._chart_host = tk.Frame(cc, bg=COLORS["bg_card"], highlightthickness=1, width=100, height=100,
                                    highlightbackground=COLORS["border"])
        self._chart_host.pack_propagate(False)        # el gráfico no agranda la ventana
        self._chart_host.pack(fill=tk.BOTH, expand=True)
        self._chart_host.bind("<Configure>", lambda e: self._fit_docked_chart(), add="+")

    def _toggle_results_panel(self):
        """Oculta o muestra la tabla y los gráficos (las pestañas quedan a la vista)."""
        self._results_visible = not self._results_visible
        self._layout_results()
        self.root.after(50, self._refit_images)

    def _select_bottom_tab(self, key: str):
        if key == "charts" and self._chart_dock:          # el gráfico está arriba: vuelve aquí
            self._undock_chart()
            return
        if self._results_visible and self._bottom_tab == key:
            return
        reopen = not self._results_visible
        self._bottom_tab, self._results_visible = key, True
        self._layout_results()
        if reopen:
            self.root.after(50, self._refit_images)

    RESULTS_MIN_H = 110

    def _results_drag_start(self, e):
        h0 = int(self._bottom_body.cget("height"))
        top = h0 + self._canvas_row.winfo_height() - 180      # las imágenes conservan al menos 180 px
        self._drag = (e.y_root, h0, max(top, self.RESULTS_MIN_H))

    def _results_drag(self, e):
        y0, h0, top = self._drag
        h = int(min(max(h0 - (e.y_root - y0), self.RESULTS_MIN_H), top))
        if h != int(self._bottom_body.cget("height")):
            self._bottom_body.configure(height=h)

    def _results_drag_end(self, _e):
        settings.set("results_height", int(self._bottom_body.cget("height")))
        self.root.update_idletasks()
        self.root.after(80, self._refit_images)

    def _paint_tabs(self):
        for key, lbl in self._tab_labels.items():
            on = self._results_visible and key == self._bottom_tab and not (key == "charts" and self._chart_dock)
            lbl.config(bg=COLORS["bg_card"] if on else COLORS["bg_panel"],
                       fg=COLORS["accent"] if on else COLORS["text_muted"])

    def _layout_results(self):
        """Pestañas Tabla/Gráficos solo si la foto tiene resultados; abiertas o solo pestañas."""
        wrap = self._bottom_wrap
        for w in (self._bottom_grip, self._bottom_bar, self._bottom_body):
            w.pack_forget()
        if not self._results_shown:
            wrap.pack_forget()
            return
        wrap.pack(fill=tk.X, expand=False, padx=SPACE["md"], pady=(0, SPACE["xs"]), after=self._canvas_row)
        if self._results_visible:
            self._bottom_grip.pack(fill=tk.X)
            self._bottom_bar.pack(fill=tk.X)
            self._bottom_body.pack(fill=tk.BOTH)
            for key, f in self._tab_frames.items():
                if key == self._bottom_tab:
                    f.pack(fill=tk.BOTH, expand=True)
                else:
                    f.pack_forget()
            if self._bottom_tab == "charts":
                self._chart_panel.show()
        else:
            self._bottom_bar.pack(fill=tk.X, pady=(SPACE["xs"], 0))
        self._paint_tabs()

    def _show_results_ui(self, on: bool):
        if on == self._results_shown:
            return
        self._results_shown = on
        if not on:
            self._view_ctrl.place_forget()
        self._layout_results()
        self._layout_sides()
        self.root.after(50, self._refit_images)

    def _dock_chart(self):
        """El gráfico de la foto pasa al lugar de la imagen oculta (o de la Entrada) y el
        panel de abajo se cierra; el ojo del gráfico lo devuelve como estaba."""
        if self._chart_dock or not self._results_shown:
            return
        hidden = [s for s, h in self._hidden.items() if h]
        slot = hidden[0] if len(hidden) == 1 else "input"
        self._dock_saved = (dict(self._hidden), self._results_visible)
        self._set_hidden({**self._hidden, slot: True})
        self._chart_dock = slot
        edge = tk.LEFT if slot == "input" else tk.RIGHT
        self._chart_title.pack_forget()
        self._chart_eye.pack_forget()
        self._chart_title.pack(side=edge, padx=SPACE["xs"])
        self._chart_eye.pack(side=edge)
        self._chart_panel.pack_forget()
        self._chart_panel.lift()
        self._fit_docked_chart()
        self._dock_btn.pack_forget()
        self._results_visible = False
        if self._bottom_tab == "charts":
            self._bottom_tab = "table"
        self._layout_sides()
        self._layout_results()
        self.root.after(80, self._after_dock)

    def _fit_docked_chart(self):
        """Arriba, el gráfico mide lo mismo que la foto ajustada a su recuadro (no todo el
        recuadro), centrado como ella."""
        if not self._chart_dock:
            return
        host = self._chart_host
        hw, hh = host.winfo_width(), host.winfo_height()
        if hw < 50 or hh < 50:
            return
        w, h = hw, hh
        if self.scaler_left.has_image:
            ih, iw = self.scaler_left.original.shape[:2]
            k = min(hw / iw, hh / ih)
            w, h = max(300, min(hw, round(iw * k))), max(220, min(hh, round(ih * k)))
        self._chart_panel.place(in_=host, relx=0.5, rely=0.5, anchor="center", width=w, height=h)
        self._chart_panel.lift()

    def _undock_chart(self):
        if not self._chart_dock:
            return
        self._chart_panel.place_forget()
        hidden, _visible = self._dock_saved
        self._chart_dock = None
        self._set_hidden(hidden)
        self._chart_panel.pack_forget()
        self._chart_panel.pack(in_=self._tab_frames["charts"], fill=tk.BOTH, expand=True)
        self._chart_panel.lift()
        self._dock_btn.pack(side=tk.RIGHT, padx=(SPACE["xs"], SPACE["md"]), before=self._save_btn)
        self._results_visible, self._bottom_tab = True, "charts"
        self._layout_sides()
        self._layout_results()
        self.root.after(80, self._after_dock)

    def _after_dock(self):
        self._refit_images()
        self._chart_panel.redraw()

    def _after_panel_toggle(self):
        self.root.after(50, self._refit_images)
        self.root.after(60, self._update_stroke_tool)
