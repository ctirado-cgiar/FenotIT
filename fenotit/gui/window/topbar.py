"""Barra superior: menús, botón Ejecutar y herramientas de zoom."""
from __future__ import annotations

import tkinter as tk

from PIL import Image, ImageTk

from fenotit import APP_NAME, log
from fenotit.core.analysis.registry import ANALYSES
from fenotit.gui.style.theme import COLORS, FONTS
from fenotit.gui.widgets.dropmenu import DropMenu, Sub
from fenotit.gui.widgets.toolbar import IconButton, Tooltip
from fenotit.gui.window.common import _analysis_label, _assets
from fenotit.i18n import t


_log = log.get("gui.window.topbar")


class TopbarMixin:
    def _build_topbar(self):
        size = max(16, int(round(18 * self.root.winfo_fpixels("1i") / 96)))
        home = IconButton(self.topbar, "home", self._go_home, t("menu.home"), bg=COLORS["bg_topbar"],
                          hover=COLORS["topbar_hover"], active=COLORS["topbar_active"], size=size)
        home.pack(side=tk.LEFT, padx=(8, 0), pady=6)
        brand = []
        try:
            ico  = _assets() / "logo.ico"
            img  = Image.open(str(ico)).resize((26, 26), Image.LANCZOS)
            self._topbar_logo = ImageTk.PhotoImage(img)
            brand.append(tk.Label(self.topbar, image=self._topbar_logo, bg=COLORS["bg_topbar"], cursor="hand2"))
            brand[-1].pack(side=tk.LEFT, padx=(6, 2), pady=6)
        except Exception:
            _log.debug("ignorado", exc_info=True)

        brand.append(tk.Label(self.topbar, text=APP_NAME, bg=COLORS["bg_topbar"], fg=COLORS["text_topbar"],
                              font=FONTS["heading"], cursor="hand2"))
        brand[-1].pack(side=tk.LEFT, padx=(0, 8))
        for w in brand:                                    # como en la mayoría de apps: el logo lleva al inicio
            w.bind("<Button-1>", lambda e: self._go_home())
            Tooltip(w, t("menu.home"))
        self._vdiv()

        self._drop(self.topbar, t("menu.project"), [
            (t("menu.home"), self._go_home),
            None,
            (t("menu.new_project"), self._new_project, "Ctrl+N"),
            (t("menu.open_project"), self._open_project, "Ctrl+Shift+O"),
            (t("menu.recent"), Sub(self._recent_items)),
            None,
            (t("menu.save_project"), self._save_project, "Ctrl+S"),
            (t("menu.save_project_as"), self._save_project_as, "Ctrl+Shift+S"),
            None,
            (t("menu.export"), self._export_results, "Ctrl+E"),
            None,
            (t("menu.exit"), self._on_close, "Ctrl+Q"),
        ])
        self._drop(self.topbar, t("menu.edit"), [
            (lambda: self._undo_label("undo"), self._undo, "Ctrl+Z"),
            (lambda: self._undo_label("redo"), self._redo, "Ctrl+Y"),
            None,
            (t("prefs.title") + "…", self._open_preferences, "Ctrl+,"),
        ])
        self._drop(self.topbar, t("menu.images"), [
            (t("menu.add_images"), self._open_image, "Ctrl+O"),
            (t("menu.add_folder"), self._open_folder),
            None,
            (t("menu.remove_image"), self._remove_current_image),
            (t("menu.clear_images"), self._clear_images),
            (t("menu.find_missing"), self._find_missing),
            None,
            (t("menu.metadata"), self._open_metadata, "Ctrl+D"),
        ])
        self._drop(self.topbar, t("menu.calibration"), [
            (t("menu.lens"), lambda: self._open_corrections("distortion")),
            (t("menu.perspective"), lambda: self._open_corrections("perspective")),
            (t("menu.colorcard"), lambda: self._open_corrections("color")),
            None,
            (t("menu.scale"), [
                (t("menu.cal_scale"), self._open_scale_dialog),
                (t("menu.scale_manual"), self._calibrate_scale),
                None,
                (t("menu.scale_clear"), self._clear_scale),
            ]),
        ])
        self._drop(self.topbar, t("menu.areas"), [
            (t("roi.area_rect"), lambda: self._set_roi_mode("include_rect"), "R"),
            (t("roi.area_polygon"), lambda: self._set_roi_mode("include_poly"), "P"),
            (t("roi.exclude_rect"), lambda: self._set_roi_mode("exclude_rect"), "X"),
            (t("roi.exclude_polygon"), lambda: self._set_roi_mode("exclude_poly"), "Shift+X"),
            None,
            (t("roi.select"), lambda: self._set_roi_mode("select"), "S"),
            (t("roi.apply_all"), self._apply_areas_to_all),
            (t("roi.clear"), self._clear_roi),
        ])
        self._drop(self.topbar, t("menu.analysis"), [
            (_analysis_label(n), lambda n=n: self._open_analysis(n)) for n in ANALYSES])
        self._drop(self.topbar, t("menu.view"), [
            (t("view.panels"), [
                (t("view.left_panel"), lambda: self._toggle_panel(self.left_panel)),
                (t("view.analysis_panel"), lambda: self._toggle_panel(self._panel_of("steps"))),
                (t("view.results_panel"), self._toggle_results_panel, "Ctrl+T"),
            ]),
            (t("view.layout"), [
                (self._sides_label("auto"), lambda: self._set_sides_mode("auto")),
                (self._sides_label("row"), lambda: self._set_sides_mode("row")),
                (self._sides_label("column"), lambda: self._set_sides_mode("column")),
            ]),
            (t("view.zoom"), [
                (t("view.zoom_in"), self.do_zoom_in, "Ctrl++"),
                (t("view.zoom_out"), self.do_zoom_out, "Ctrl+−"),
                (t("view.zoom_fit"), self.do_zoom_fit, "Ctrl+0"),
                (t("view.zoom_100"), self.do_zoom_100, "Ctrl+1"),
            ]),
            (t("view.tool"), [
                (t("view.pan"), lambda: self._set_zoom_tool("pan"), "H"),
                (t("view.zoom_area"), lambda: self._set_zoom_tool("zoom_area"), "Z"),
                (t("view.inspect"), lambda: self._set_zoom_tool("inspect"), "I"),
            ]),
            None,
            (t("view.toggle_legend"), self._toggle_legend, "L"),
            (t("style.title") + "…", self._open_style),
        ])
        self._drop(self.topbar, t("menu.help"), [
            (t("tour.menu"), self._start_tour),
            (t("help.shortcuts"), self._show_shortcuts, "F1"),
            (t("help.log"), self._open_log_folder),
            None,
            (t("menu.about"), self._about),
        ])

        # Derecha: tipo de análisis y Ejecutar (con margen para no quedar junto a la X)
        tk.Frame(self.topbar, bg=COLORS["bg_topbar"], width=24).pack(side=tk.RIGHT)
        run = tk.Frame(self.topbar, bg=COLORS["bg_card"])
        run.pack(side=tk.RIGHT, padx=(6, 0), pady=8)
        self._run_frame = run
        tk.Button(run, text=t("topbar.run"), command=lambda: self._run_analysis(False),
                  bg=COLORS["bg_card"], fg=COLORS["accent"], font=FONTS["body_bold"],
                  relief="flat", bd=0, activebackground=COLORS["accent_light"],
                  activeforeground=COLORS["accent"], padx=10, pady=2,
                  cursor="hand2").pack(side=tk.LEFT)
        tk.Frame(run, bg=COLORS["border"], width=1).pack(side=tk.LEFT, fill=tk.Y, pady=3)
        DropMenu(run, "", [
            (t("run.current"), lambda: self._run_analysis(False), "Ctrl+Enter"),
            (t("run.all"), lambda: self._run_analysis(True), "Ctrl+Shift+Enter"),
        ], COLORS, arrow_only=True).pack(side=tk.LEFT)

        self.analysis_var = tk.StringVar()
        tk.Frame(self.topbar, bg=COLORS["topbar_divider"], width=1).pack(side=tk.RIGHT, fill=tk.Y, pady=8, padx=6)
        self._build_zoom_tools()

    def _build_zoom_tools(self):
        """Zoom en la barra superior: acercar, alejar, ajustar, zoom a un área, mano y el %."""
        self._zoom_pct_var = tk.StringVar(value="100%")
        bar = tk.Frame(self.topbar, bg=COLORS["bg_topbar"])
        bar.pack(side=tk.RIGHT)
        size = max(16, int(round(18 * self.root.winfo_fpixels("1i") / 96)))
        kw = dict(bg=COLORS["bg_topbar"], hover=COLORS["topbar_hover"], active=COLORS["topbar_active"], size=size)
        pct = tk.Label(bar, textvariable=self._zoom_pct_var, bg=COLORS["bg_topbar"], fg=COLORS["text_topbar"],
                       font=FONTS["small"], width=6, cursor="hand2")
        self._zoom_tools = {}
        for name, cmd, tip in (
                ("zoom_in", self.do_zoom_in, f"{t('view.zoom_in')}  (Ctrl++)"),
                ("zoom_out", self.do_zoom_out, f"{t('view.zoom_out')}  (Ctrl+−)"),
                ("fit", self.do_zoom_fit, f"{t('view.zoom_fit')}  (Ctrl+0)"),
                ("zoom_area", lambda: self._set_zoom_tool("zoom_area"), f"{t('view.zoom_area')}  (Z)"),
                ("pan", lambda: self._set_zoom_tool("pan"), f"{t('view.pan')}  (H)"),
                ("inspect", lambda: self._set_zoom_tool("inspect"), f"{t('view.inspect')}  (I)")):
            b = IconButton(bar, name, cmd, tip, **kw)
            b.pack(side=tk.LEFT, padx=1, pady=6)
            if name in ("zoom_area", "pan", "inspect"):
                self._zoom_tools[name] = b
        pct.pack(side=tk.LEFT, padx=(4, 0))
        tk.Frame(bar, bg=COLORS["topbar_divider"], width=1).pack(side=tk.LEFT, fill=tk.Y, pady=8, padx=(6, 4))
        self._style_btn = IconButton(bar, "palette", self._open_style, f"{t('style.title')}", **kw)
        self._style_btn.pack(side=tk.LEFT, padx=1, pady=6)
        self._style_popup = None
        self.root.bind("<ButtonPress-1>", self._close_style_popup, add="+")
        pct.bind("<Button-1>", lambda e: self.do_zoom_100())
        Tooltip(pct, f"{t('view.zoom_100')}  (Ctrl+1)")
        self._zoom_tools["pan"].select(True)

    def _vdiv(self):
        tk.Frame(self.topbar, bg=COLORS["topbar_divider"], width=1).pack(
            side=tk.LEFT, fill=tk.Y, pady=6, padx=4)

    def _drop(self, parent, text, items):
        m = DropMenu(parent, text, items, COLORS)
        m.pack(side=tk.LEFT)
        self._menus = getattr(self, "_menus", {})
        self._menus[text] = m
        return m

    def _populate_analysis_menu(self):
        names = list(ANALYSES.keys())
        if names:
            self.analysis_var.set(_analysis_label(names[0]))
            self._on_analysis_selected(None, choose=False)

    def _open_style(self):
        """Colores de la máscara, contornos y puntos (se guardan en el proyecto)."""
        from fenotit.gui.widgets.style_popup import StylePopup
        if self._style_popup is not None and self._style_popup.winfo_exists():
            self._style_popup.destroy()
            self._style_popup = None
            return
        self._style_popup = StylePopup(self.root, self._style_btn, self.project.display.get("style"),
                                       COLORS, self._on_style_change)

    def _close_style_popup(self, event=None):
        if event is not None and event.widget is self._style_btn:
            return
        if self._style_popup is not None and self._style_popup.winfo_exists():
            self._style_popup.destroy()
        self._style_popup = None
