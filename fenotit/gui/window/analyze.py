"""Análisis elegido: tarjetas para elegirlo, sus pasos, parámetros y unidades del panel."""
from __future__ import annotations

import tkinter as tk

from fenotit import log
from fenotit.core import units
from fenotit.core.analysis.registry import ANALYSES
from fenotit.gui.style.theme import COLORS, FONTS, SIZES, SPACE
from fenotit.gui.widgets.params import ConfigPanel
from fenotit.gui.window.common import _analysis_key, _analysis_label
from fenotit.i18n import t


_log = log.get("gui.window.analyze")


class AnalyzeMixin:
    def _init_analyze(self):
        self.active_analysis: str | None = None
        self._reset_analysis()

    def _reset_analysis(self):
        """Ningún análisis elegido y sin panel de parámetros."""
        self.config_panel: ConfigPanel | None = None
        self._analysis_chosen = False

    _UNIT_POWER = {"area": 2, "length": 1}

    def _unit_labels(self, unit: str) -> dict:
        """Etiquetas del panel: la unidad de la escala (µm, mm, cm, m) o px sin escala."""
        return {"area": f"{units.display(unit)}²", "length": units.display(unit)}

    def _area_unit(self) -> str:
        """La unidad de la escala si hay alguna (de todas las fotos o de alguna); si no, px."""
        p = self.project
        return p.unit if p.scale.source != "none" or p.image_scale else "px"

    def _has_area_units(self) -> bool:
        return any(i.get("unit") in self._UNIT_POWER
                   for i in ANALYSES[self.active_analysis].params_schema) if self.active_analysis in ANALYSES else False

    def _sync_area_units(self):
        """Si cambió la escala o su unidad, convierte las medidas del panel y la etiqueta."""
        panel = getattr(self, "config_panel", None)
        if panel is None or not self._has_area_units():
            return
        old, new = self._panel_area_unit, self._area_unit()
        if new != old:
            # mm de cada unidad; para pasar de/a px se usa la escala (al quitarla, la última usada)
            mpp = self.mm_per_pixel or getattr(self, "_panel_mpp", None)
            if old != "px" and new != "px":
                f = units.TO_MM[old] / units.TO_MM[new]
            elif mpp:
                f = mpp / units.TO_MM[new] if old == "px" else units.TO_MM[old] / mpp
            else:
                f = None
            if f:
                vals = panel.get_values(warn=False)
                changed = {}
                for i in ANALYSES[self.active_analysis].params_schema:
                    power = self._UNIT_POWER.get(i.get("unit"))
                    if power and i["key"] in vals:
                        changed[i["key"]] = round(vals[i["key"]] * f ** power, 6)
                panel.set_values(changed)
            else:
                _log.info("Sin escala para convertir; se conservan los números")
            self._panel_area_unit = new
        if new != "px" and self.mm_per_pixel:
            self._panel_mpp = self.mm_per_pixel
        panel.set_units(self._unit_labels(new))

    def _render_analysis_cards(self):
        """Los análisis como tarjetas: la elegida en azul con su descripción; clic = elegir."""
        box = self._analysis_cards
        for w in box.winfo_children():
            w.destroy()
        cur = self._selected_analysis() if getattr(self, "_analysis_chosen", False) else None
        for name in ANALYSES:
            on = name == cur
            bg = COLORS["accent_light"] if on else COLORS["bg_card"]
            card = tk.Frame(box, bg=bg, highlightthickness=1, cursor="hand2",
                            highlightbackground=COLORS["accent"] if on else COLORS["border"])
            card.pack(fill=tk.X, pady=1)
            title = tk.Label(card, text=_analysis_label(name), bg=bg, fg=COLORS["accent"] if on else COLORS["text"],
                             font=FONTS["body_bold"], anchor="w", cursor="hand2")
            title.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["sm"], 0))
            desc = t(f"analysis.{_analysis_key(name)}.desc", ANALYSES[name].description)
            d = tk.Label(card, text=desc, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"], anchor="w",
                         justify="left", wraplength=SIZES["panel_right"] - 50, cursor="hand2")
            d.pack(fill=tk.X, padx=SPACE["md"], pady=(0, SPACE["sm"]))
            parts = [card, title, d]
            for w in parts:
                w.bind("<Button-1>", lambda e, n=name: self._open_analysis(n))

    def _open_analysis(self, name: str):
        """Elige el análisis y abre su primer paso."""
        if name != self._selected_analysis() or not getattr(self, "_analysis_chosen", False):
            self.analysis_var.set(_analysis_label(name))
            self._on_analysis_selected(None)
        self._show_panel(self._panel_of("steps"), True)
        self._select_right_tab(self._step_tab(ANALYSES[name].steps[0]))

    def _choose_analysis(self):
        """Abre el panel derecho en las tarjetas de análisis."""
        self._show_panel(self._panel_of("steps"), True)
        self._select_right_tab("choose")

    @staticmethod
    def _step_tab(step: str) -> str:
        """Pestaña del panel derecho de cada paso declarado por el análisis."""
        return {"segment": "seg", "params": "analysis"}[step]

    def _step_label(self, step: str) -> str:
        if step == "segment":
            return t("rtab.segment")
        name = self.active_analysis
        return t(f"analysis.{_analysis_key(name)}.step", t("rtab.analyze")) if name else t("rtab.analyze")

    def _steps_after(self, step: str) -> list[str]:
        """Pasos del análisis elegido que siguen a `step`."""
        steps = list(ANALYSES[self.active_analysis].steps) if self.active_analysis in ANALYSES else []
        return steps[steps.index(step) + 1:] if step in steps else []

    def _render_analysis_steps(self):
        """Nombre del análisis elegido, pestañas de sus pasos y método de segmentación que admite."""
        name = self.active_analysis
        if name not in ANALYSES:
            return
        self._an_title.config(text=_analysis_label(name))
        self._render_step_tabs([(self._step_tab(s), self._step_label(s)) for s in ANALYSES[name].steps])
        allowed = ANALYSES[name].seg_methods
        if allowed and self._seg_method not in allowed:
            self._set_seg_method(allowed[0], quiet=True)
        self._show_method_buttons(allowed)
        after = self._steps_after("segment")
        self._set_next_step(self._step_label(after[0]) if after else None)

    def _show_panel(self, panel, on: bool):
        if on:
            panel.expand()
        else:
            panel.hide_all()
            self._after_panel_toggle()

    def _toggle_panel(self, panel):
        hidden = str(self.paned.panecget(panel, "hide")) in ("1", "true")
        panel.expand() if hidden else panel.collapse()

    def _selected_analysis(self) -> str | None:
        label = self.analysis_var.get()
        return next((n for n in ANALYSES if _analysis_label(n) == label), None)

    def _on_analysis_selected(self, _event, choose: bool = True):
        """choose=False: solo deja listo el panel (al iniciar); el análisis cuenta como
        elegido cuando el usuario lo abre en el menú Análisis o el proyecto ya tenía uno."""
        name = self._selected_analysis()
        if name not in ANALYSES:
            return
        self._store_panel_params()
        self.active_analysis = name
        if choose:
            self._analysis_chosen = True
            self.project.analysis = _analysis_key(name)
        self._render_analysis_cards()
        self._render_analysis_steps()
        self._update_seg_tab()
        for w in self.config_container.winfo_children():
            w.destroy()
        stored = self.project.params.get(self.project.analysis, {})
        self._panel_area_unit = stored.get("_area_unit", "px")
        self.config_panel = ConfigPanel(
            self.config_container,
            schema=[i for i in ANALYSES[name].params_schema
                    if i.get("segmentation") in (None, self._seg_method)],
            colors=COLORS, prefix=_analysis_key(name),
            units=self._unit_labels(self._panel_area_unit))
        self.config_panel.set_values({k: v for k, v in stored.items() if not k.startswith("_")})
        self.config_panel.pack(fill=tk.BOTH, expand=True)
        for var in self.config_panel._vars.values():          # cambios del panel: Ctrl+Z
            var.trace_add("write", lambda *_: self._settings_changed())
        self.root.after_idle(self._mark_settled)
        self._show_cached_result()
        self._sync_area_units()
