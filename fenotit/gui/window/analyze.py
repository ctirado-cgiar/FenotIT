"""Análisis elegido: tarjetas para elegirlo, sus pasos, parámetros y unidades del panel."""
from __future__ import annotations

import tkinter as tk

# log escribe los mensajes y errores en el archivo de registro.
from fenotit import log
# units convierte entre unidades de longitud y arma los textos con unidad.
from fenotit.core import units
# ANALYSES tiene los análisis registrados (Objetos, Detección…) por clave.
from fenotit.core.analysis.registry import ANALYSES
# divider dibuja una línea separadora.
from fenotit.gui.style.components import divider
# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras); SIZES (tamaños); SPACE
# (márgenes).
from fenotit.gui.style.theme import COLORS, FONTS, SIZES, SPACE
# ConfigPanel arma las casillas y opciones del paso Medir a partir del esquema del análisis.
from fenotit.gui.widgets.params import ConfigPanel
# Funciones comunes de la ventana: _analysis_label da el nombre del análisis para mostrar.
from fenotit.gui.window.common import _analysis_label
# t devuelve cada texto de la ventana en el idioma elegido (fenotit/lang).
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
        project = self.project
        return project.unit if project.scale.source != "none" or project.image_scale else "px"

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
                factor = units.TO_MM[old] / units.TO_MM[new]
            elif mpp:
                factor = mpp / units.TO_MM[new] if old == "px" else units.TO_MM[old] / mpp
            else:
                factor = None
            if factor:
                vals = panel.get_values(warn=False)
                changed = {}
                for i in ANALYSES[self.active_analysis].params_schema:
                    power = self._UNIT_POWER.get(i.get("unit"))
                    if power and i["key"] in vals:
                        changed[i["key"]] = round(vals[i["key"]] * factor ** power, 6)
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
        for widget in box.winfo_children():
            widget.destroy()
        cur = self._selected_analysis() if getattr(self, "_analysis_chosen", False) else None
        for i, name in enumerate(ANALYSES):
            active = name == cur
            bg = COLORS["accent_light"] if active else COLORS["bg_panel"]
            if i:
                divider(box, "divider").pack(fill=tk.X)
            card = tk.Frame(box, bg=bg, cursor="hand2")
            card.pack(fill=tk.X)
            title = tk.Label(card, text=_analysis_label(name), bg=bg, fg=COLORS["accent"] if active else COLORS["text"],
                             font=FONTS["body_bold"], anchor="w", cursor="hand2")
            title.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["sm"], 0))
            desc = t(f"analysis.{name}.desc", "")
            label = tk.Label(card, text=desc, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"], anchor="w",
                         justify="left", wraplength=SIZES["panel_right"] - 50, cursor="hand2")
            label.pack(fill=tk.X, padx=SPACE["md"], pady=(0, SPACE["sm"]))
            parts = [card, title, label]
            if not active:
                card.bind("<Enter>", lambda event, widgets=parts: [widget.config(bg=COLORS["bg_card"]) for widget in widgets])
                card.bind("<Leave>", lambda event, widgets=parts: [widget.config(bg=COLORS["bg_panel"]) for widget in widgets])
            for widget in parts:
                widget.bind("<Button-1>", lambda event, analysis_name=name: self._open_analysis(analysis_name))

    def _open_analysis(self, name: str):
        """Elige el análisis y abre su primer paso."""
        if name != self._selected_analysis() or not getattr(self, "_analysis_chosen", False):
            self.analysis_var.set(_analysis_label(name))
            self._on_analysis_selected(None)
        self._show_panel(self._panel_of("steps"), True)
        self._select_right_tab(self._step_tab(self._analysis_steps(name)[0]))

    def _choose_analysis(self):
        """Abre el panel derecho en las tarjetas de análisis."""
        self._show_panel(self._panel_of("steps"), True)
        self._select_right_tab("choose")

    @staticmethod
    def _step_tab(step: str) -> str:
        """Pestaña del panel derecho de cada paso declarado por el análisis."""
        return {"segment": "seg", "detect": "det", "params": "analysis"}[step]

    def _step_label(self, step: str) -> str:
        if step in ("segment", "detect"):
            return t(f"rtab.{step}")
        name = self.active_analysis
        return t(f"analysis.{name}.step", t("rtab.analyze")) if name else t("rtab.analyze")

    def _analysis_steps(self, name: str) -> list[str]:
        """Pasos del análisis: los fijos y los opcionales cuya casilla está marcada (antes de
        «params»)."""
        descriptor = ANALYSES[name]
        steps = list(descriptor.steps)
        if self.config_panel is not None and self.active_analysis == name:
            values = self.config_panel.get_values()
        else:
            values = self.project.params.get(name, {})
        for step, checkbox in descriptor.optional_steps.items():
            if values.get(checkbox) and step not in steps:
                steps.insert(steps.index("params") if "params" in steps else len(steps), step)
                replaced = descriptor.replaced_steps.get(step)
                if replaced in steps:
                    steps.remove(replaced)
        return steps

    def _steps_after(self, step: str) -> list[str]:
        """Pasos del análisis elegido que siguen a `step`."""
        steps = self._analysis_steps(self.active_analysis) if self.active_analysis in ANALYSES else []
        return steps[steps.index(step) + 1:] if step in steps else []

    def _render_analysis_steps(self):
        """Nombre del análisis elegido, pestañas de sus pasos y método de segmentación que admite."""
        name = self.active_analysis
        if name not in ANALYSES:
            return
        self._an_title.config(text=_analysis_label(name))
        steps = self._analysis_steps(name)
        self._render_step_tabs([(self._step_tab(step), self._step_label(step)) for step in steps])
        allowed = ANALYSES[name].seg_methods
        if allowed and self._seg_method not in allowed:
            self._set_seg_method(allowed[0], quiet=True)
        self._show_method_buttons(allowed)
        if hasattr(self, "class_list"):
            self.class_list.show_objects(self._analysis_objects())
        after = self._steps_after("segment")
        self._set_next_step(self._step_label(after[0]) if after else None)
        after = self._steps_after("detect")
        self._set_det_next_step(self._step_label(after[0]) if after else None)
        self._render_detectors()
        self._update_unavailable()
        if self._right_tab not in ("choose",) and self._right_tab not in [self._step_tab(step) for step in steps]:
            self._select_right_tab(self._step_tab(steps[-1]))
        elif self._right_tab in self._rtab_labels:
            self._select_right_tab(self._right_tab)

    def _show_panel(self, panel, active: bool):
        if active:
            panel.expand()
        else:
            panel.hide_all()
            self._after_panel_toggle()

    def _toggle_panel(self, panel):
        hidden = str(self.paned.panecget(panel, "hide")) in ("1", "true")
        panel.expand() if hidden else panel.collapse()

    def _selected_analysis(self) -> str | None:
        label = self.analysis_var.get()
        return next((name for name in ANALYSES if _analysis_label(name) == label), None)

    def _on_analysis_selected(self, _event, choose: bool = True):
        """choose=False: solo deja listo el panel (al iniciar); el análisis cuenta como
        elegido cuando el usuario lo abre en el menú Análisis o el proyecto ya tenía uno."""
        name = self._selected_analysis()
        if name not in ANALYSES:
            return
        self._store_panel_params()
        self.config_panel = None                              # el panel del análisis anterior
        self.active_analysis = name
        if choose:
            self._analysis_chosen = True
            self.project.analysis = name
        self._render_analysis_cards()
        self._render_analysis_steps()
        self._update_seg_tab()
        for widget in self.config_container.winfo_children():
            widget.destroy()
        stored = self.project.params.get(self.project.analysis, {})
        self._panel_area_unit = stored.get("_area_unit", "px")
        self.config_panel = ConfigPanel(
            self.config_container,
            schema=[i for i in ANALYSES[name].params_schema
                    if i.get("segmentation") in (None, self._seg_method)],
            colors=COLORS, prefix=ANALYSES[name].param_texts or name,
            units=self._unit_labels(self._panel_area_unit))
        self.config_panel.set_values({key: value for key, value in stored.items() if not key.startswith("_")})
        self.config_panel.pack(fill=tk.BOTH, expand=True)
        for var in self.config_panel._vars.values():          # cambios del panel: Ctrl+Z
            var.trace_add("write", lambda *_: self._settings_changed())
        for checkbox in ANALYSES[name].optional_steps.values():   # pasos que aparecen con una casilla
            if checkbox in self.config_panel._vars:
                self.config_panel._vars[checkbox].trace_add("write", lambda *_: self._render_analysis_steps())
        self._update_unavailable()
        self.root.after_idle(self._mark_settled)
        self._show_cached_result()
        self._sync_area_units()
