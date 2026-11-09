"""Entrada y Resultado: disposición, controles de la vista, zoom y colores de dibujo."""
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk


from fenotit import settings
from fenotit.gui.canvas.areas import AreaEditor
from fenotit.gui.canvas.tools import CanvasTools
from fenotit.gui.canvas.zoom import ZoomController
from fenotit.gui.style.components import button
from fenotit.gui.style.theme import COLORS, FONTS
from fenotit.gui.widgets.toolbar import IconButton, Tooltip
from fenotit.i18n import t


class ViewerMixin:
    def _build_center_panel(self):
        cf = self.center_frame

        canvas_row = tk.Frame(cf, bg=COLORS["bg"])
        canvas_row.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 2))
        canvas_row.rowconfigure(0, weight=1)
        # Cada lado: título + ojo (ocultar) y su imagen. Oculto, queda una franja con el
        # ojo tachado para volver a mostrarlo.
        self._cols, self._strips, self._hstrips = {}, {}, {}
        self._hidden = {"input": False, "result": False}
        self._orient = "row"
        for side, title in (("input", t("center.input")), ("result", t("center.result"))):
            col = tk.Frame(canvas_row, bg=COLORS["bg"])
            head = tk.Frame(col, bg=COLORS["bg"])
            head.pack(fill=tk.X)
            # Entrada en el extremo izquierdo y Resultado en el derecho, cada uno junto a su ojo
            edge = tk.LEFT if side == "input" else tk.RIGHT
            tk.Label(head, text=title, bg=COLORS["bg"], fg=COLORS["text_muted"],
                     font=FONTS["small"]).pack(side=edge, padx=2)
            IconButton(head, "eye", lambda sd=side: self._toggle_side(sd), t("view.hide_side", name=title),
                       bg=COLORS["bg"], hover=COLORS["btn_hover"], active=COLORS["accent_light"],
                       size=14, color=COLORS["text_muted"]).pack(side=edge)
            canvas = tk.Canvas(col, bg=COLORS["bg_card"], highlightthickness=1, width=120, height=120,
                               highlightbackground=COLORS["border"])
            canvas.pack(fill=tk.BOTH, expand=True)
            # oculto: franja con el ojo tachado y el nombre en vertical
            strip = tk.Frame(canvas_row, bg=COLORS["bg_panel"], width=22, cursor="hand2")
            IconButton(strip, "eye_off", lambda sd=side: self._toggle_side(sd), t("view.show_side", name=title),
                       bg=COLORS["bg_panel"], hover=COLORS["btn_hover"], active=COLORS["accent_light"],
                       size=14, color=COLORS["accent"]).pack(side=tk.TOP, pady=4)
            font = tkfont.Font(font=FONTS["small"])
            vt = tk.Canvas(strip, width=20, height=font.measure(title) + 8, bg=COLORS["bg_panel"],
                           highlightthickness=0, cursor="hand2")
            vt.create_text(10, (font.measure(title) + 8) / 2, text=title, angle=90, fill=COLORS["text_muted"],
                           font=FONTS["small"])
            vt.pack(side=tk.TOP)
            for w in (strip, vt):
                w.bind("<Button-1>", lambda e, sd=side: self._toggle_side(sd))
            # la misma franja, acostada, cuando Entrada y Resultado van uno encima del otro
            hstrip = tk.Frame(canvas_row, bg=COLORS["bg_panel"], height=22, cursor="hand2")
            IconButton(hstrip, "eye_off", lambda sd=side: self._toggle_side(sd), t("view.show_side", name=title),
                       bg=COLORS["bg_panel"], hover=COLORS["btn_hover"], active=COLORS["accent_light"],
                       size=14, color=COLORS["accent"]).pack(side=tk.LEFT, padx=4)
            hl = tk.Label(hstrip, text=title, bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONTS["small"],
                          cursor="hand2")
            hl.pack(side=tk.LEFT)
            for w in (hstrip, hl):
                w.bind("<Button-1>", lambda e, sd=side: self._toggle_side(sd))
            self._hstrips[side] = hstrip
            self._cols[side], self._strips[side] = col, strip
            if side == "input":
                self.canvas_left = canvas
                canvas.config(cursor="crosshair")
            else:
                self.canvas_right = canvas
        self._build_chart_dock(canvas_row)
        self._canvas_row = canvas_row
        self._layout_sides()
        canvas_row.bind("<Configure>", self._on_canvas_row_resize, add="+")

        self.areas = AreaEditor(self.canvas_left, self._on_areas_change)
        self.areas.on_done = self._on_roi_tool_done
        from fenotit.gui.canvas.tools import PointEditor, RegionPicker, StrokeEditor
        self.strokes = StrokeEditor(self.canvas_left, self._on_strokes_change)
        self.prompts = PointEditor(self.canvas_left, self._on_prompts_change)
        self.region_picker = RegionPicker(self.canvas_left, self._on_region_click)

        # ZoomController sincroniza ambos canvas
        self.zoom_ctrl = ZoomController(
            self.canvas_left, self.canvas_right,
            on_redraw=self._on_zoom_redraw)
        self.zoom_ctrl.set_zoom_var(self._zoom_pct_var)
        self.zoom_ctrl.on_tool_change = self._on_zoom_tool_change
        self.zoom_ctrl.on_pick = self._pick_object
        self.zoom_ctrl.on_hover = self._hover_object
        # Referencia para que ZoomController informe al ROI
        self.canvas_left._roi_selector_ref = CanvasTools(self.areas, self.strokes, self.prompts,
                                                                self.region_picker)
        self._build_area_tools()

        self._canvas_row = canvas_row
        from fenotit.gui.screens.start import StartScreen
        self._start = StartScreen(cf, COLORS, FONTS, self._open_image, self._open_folder,
                                  self._open_project, self._open_recent,
                                  links=((t("menu.language"), self._choose_language), (t("menu.about"), self._about),
                                         (t("start.help"), self._start_tour)))
        self._build_view_controls()

        self._build_bottom_panel(cf)

    def _set_zoom_tool(self, tool: str):
        if self.zoom_ctrl is None:
            return
        if self.areas and self.areas.active:
            self.areas.stop()
        if tool != "pan" and self.zoom_ctrl.tool == tool:       # otro clic la apaga: vuelve la mano
            tool = "pan"
        self.zoom_ctrl.set_tool(tool)
        if tool == "zoom_area":
            self._set_status(t("view.zoom_area_hint"))
        elif tool == "inspect":
            self._set_status(t("view.inspect_hint"))

    def _on_zoom_tool_change(self, tool: str):
        for name, btn in getattr(self, "_zoom_tools", {}).items():
            btn.select(name == tool)
        if tool == "pan":
            self._set_status(t("status.ready"))

    def _on_style_change(self, style: dict):
        self.project.display = {**self.project.display, "style": style}
        if self.last_result and self.step_names:
            self._jump_to_step(self.step_idx)
        self._show_preview()

    def do_zoom_100(self):
        if self.zoom_ctrl:
            self.zoom_ctrl.actual_size()

    def do_zoom_in(self):
        if self.zoom_ctrl is None:
            return
        self.zoom_ctrl.zoom_in()

    def do_zoom_out(self):
        if self.zoom_ctrl is None:
            return
        self.zoom_ctrl.zoom_out()

    def do_zoom_fit(self):
        if self.zoom_ctrl is None:
            return
        self.zoom_ctrl.fit()

    def _on_zoom_redraw(self):
        """Callback del ZoomController — actualiza offset del ROI."""
        pass

    SIDES_MODES = ("auto", "row", "column")

    def _sides_orient(self) -> str:
        """row = lado a lado, column = uno encima del otro. Automático: lo que muestre más
        grande la foto en el espacio disponible (fotos anchas en pantallas anchas: arriba y abajo)."""
        mode = settings.get("sides_layout", "auto")
        if mode in ("row", "column"):
            return mode
        row = self._canvas_row
        W, H = row.winfo_width(), row.winfo_height()
        if not self.scaler_left.has_image or W < 50 or H < 50:
            return self._orient
        ih, iw = self.scaler_left.original.shape[:2]
        side = min(W / 2 / iw, H / ih)
        stacked = min(W / iw, H / 2 / ih)
        if stacked > side * 1.1:
            return "column"
        if side > stacked * 1.1:
            return "row"
        return self._orient                        # casi iguales: no cambiar por poco

    def _set_sides_mode(self, mode: str):
        settings.set("sides_layout", mode)
        self._layout_sides()
        self.root.after(50, self._refit_images)

    def _sides_label(self, mode: str):
        return lambda: ("✓ " if settings.get("sides_layout", "auto") == mode else "    ") + t(f"view.sides_{mode}")

    def _on_canvas_row_resize(self, _e=None):
        if self._results_shown and self._sides_orient() != self._orient:
            self._layout_sides()
            self.root.after(50, self._refit_images)

    def _layout_sides(self):
        """Entrada y Resultado lado a lado o uno encima del otro. Sin resultados solo se ve
        la entrada; con resultados, el ojo de cada lado lo oculta (queda una franja con su
        nombre) y el gráfico puede ocupar el lugar de uno de los dos."""
        has_result = self._results_shown
        dock = self._chart_dock if has_result else None
        row = self._canvas_row
        orient = self._orient = self._sides_orient() if has_result else self._orient
        stacked = orient == "column"
        for w in (list(self._cols.values()) + list(self._strips.values()) + list(self._hstrips.values())
                  + [self._chart_col]):
            w.grid_forget()
        for c in (0, 1):
            row.columnconfigure(c, weight=0, uniform="", minsize=0)
            row.rowconfigure(c, weight=0, uniform="", minsize=0)
        if stacked:
            row.columnconfigure(0, weight=1)
        else:
            row.rowconfigure(0, weight=1)
        for k, side in enumerate(("input", "result")):
            if dock == side:
                w, full = self._chart_col, True
            elif side == "result" and not has_result:
                continue
            elif has_result and self._hidden[side]:
                w, full = (self._hstrips if stacked else self._strips)[side], False
            else:
                w, full = self._cols[side], True
            r, c = (k, 0) if stacked else (0, k)
            if full:
                w.grid(row=r, column=c, sticky="nsew", padx=2, pady=(0, 2) if stacked else 0)
                (row.rowconfigure if stacked else row.columnconfigure)(k, weight=1, uniform="side")
            else:
                w.grid(row=r, column=c, sticky="ew" if stacked else "ns")

    def _set_hidden(self, hidden: dict):
        """Qué lados (Entrada, Resultado) están ocultos."""
        self._hidden = dict(hidden)

    def _toggle_side(self, side: str):
        other = "result" if side == "input" else "input"
        self._hidden[side] = not self._hidden[side]
        if self._hidden[side] and self._hidden[other] and not self._chart_dock:   # nunca las dos ocultas
            self._hidden[other] = False
        self._layout_sides()
        self.root.after(50, self._refit_images)

    def _build_view_controls(self):
        """Controles de la leyenda dentro del recuadro de resultado (esquina inferior derecha)."""
        from fenotit.core.pipeline.views import COLOR_FORMATS
        bg = COLORS["bg_card"]
        box = tk.Frame(self.canvas_right, bg=bg, highlightthickness=1,
                       highlightbackground=COLORS["border"], padx=2)
        self._view_ctrl = box
        self.legend_var = tk.BooleanVar(value=True)
        tk.Checkbutton(box, text=t("view.legend"), variable=self.legend_var,
                       command=self._on_display_change, bg=bg, fg=COLORS["text"],
                       selectcolor=bg, activebackground=bg, font=FONTS["small"]).pack(side=tk.LEFT)
        size_kw = dict(kind="link", size="small", bg=bg, padx=3, pady=0)
        a_minus = button(box, text="A−", command=lambda: self._legend_size(0.8), **size_kw)
        a_plus = button(box, text="A+", command=lambda: self._legend_size(1.25), **size_kw)
        a_minus.pack(side=tk.LEFT)
        a_plus.pack(side=tk.LEFT)
        # posición de la leyenda: cuadrícula 3×3 (el centro no se usa)
        self._pos_grid = tk.Canvas(box, width=25, height=19, bg=bg, highlightthickness=0, cursor="hand2")
        self._pos_grid.pack(side=tk.LEFT, padx=(4, 2), pady=1)
        self._pos_grid.bind("<Button-1>", self._pick_legend_pos)
        Tooltip(self._pos_grid, t("view.legend_pos"))
        self._paint_pos_grid()
        Tooltip(a_minus, t("view.legend_smaller"))
        Tooltip(a_plus, t("view.legend_bigger"))
        self.color_fmt_var = tk.StringVar(value="RGB")
        self._fmt_combo = ttk.Combobox(box, textvariable=self.color_fmt_var, values=list(COLOR_FORMATS),
                                       state="readonly", width=4, font=FONTS["small"])
        self._fmt_combo.bind("<<ComboboxSelected>>", self._on_display_change)

    _POS_GRID = (("tl", "tc", "tr"), ("ml", None, "mr"), ("bl", "bc", "br"))

    def _paint_pos_grid(self):
        c, cur = self._pos_grid, self.project.display.get("legend_pos", "tl") if hasattr(self, "project") else "tl"
        c.delete("all")
        for r, row in enumerate(self._POS_GRID):
            for k, pos in enumerate(row):
                if pos is None:
                    continue
                x, y = 2 + k * 8, 1 + r * 6
                c.create_rectangle(x, y, x + 6, y + 4, outline=COLORS["accent"],
                                   fill=COLORS["accent"] if pos == cur else "")

    def _pick_legend_pos(self, event):
        k, r = min(2, max(0, (event.x - 2) // 8)), min(2, max(0, (event.y - 1) // 6))
        pos = self._POS_GRID[r][k]
        if pos:
            self.project.display = {**self.project.display, "legend_pos": pos}
            self._paint_pos_grid()
            self._on_display_change()

    def _update_view_controls(self, name: str):
        """Solo si la vista tiene leyenda; el formato de color solo en la vista de color."""
        btn = getattr(self, "_colors_btn", None)
        if btn is not None:                  # "Colores…" solo en la capa Color
            kind = ((self.last_result.extra.get("step_folders") or {}).get(name) if self.last_result else None)
            if kind == "color" and not btn.winfo_ismapped():
                btn.pack(side=tk.RIGHT, padx=2, pady=3, after=self._compare_btn)
            elif kind != "color":
                btn.pack_forget()
        spec = (self.last_result.extra.get("legends") or {}).get(name) if self.last_result else None
        if not spec:
            self._view_ctrl.place_forget()
            return
        if spec.get("colors") is not None:
            self._fmt_combo.pack(side=tk.LEFT, padx=(2, 1), pady=1)
        else:
            self._fmt_combo.pack_forget()
        self._view_ctrl.place(relx=1.0, rely=1.0, anchor="se", x=-6, y=-6)

    def _refit_images(self):
        self._do_resize()
        if self.zoom_ctrl and self.scaler_left.has_image:
            self.do_zoom_fit()

    def _toggle_legend(self):
        self.legend_var.set(not self.legend_var.get())
        self._on_display_change()
