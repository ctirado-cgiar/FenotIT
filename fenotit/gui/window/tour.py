"""Pasos del recorrido guiado."""
from __future__ import annotations

from fenotit import settings
from fenotit.i18n import t


class TourMixin:
    def _start_tour(self):
        """Un solo recorrido guiado: en la pantalla de inicio explica esa pantalla; con un
        proyecto abierto empieza en lo que le toca hacer al usuario (Anterior lleva a lo de antes)."""
        from fenotit.gui.screens.tour import Step, Tour
        if getattr(self, "_tour", None):
            return

        def done():
            self._tour = None
        start = getattr(self, "_start", None)
        if start is not None and start.winfo_ismapped():
            settings.set("tour_seen", True)
            tg = start.targets
            steps = [Step(t("tour.home.title"), t("tour.home.text")),
                     Step(t("tour.new.title"), t("tour.new.text"), lambda: tg.get("new")),
                     Step(t("tour.open.title"), t("tour.open.text"), lambda: tg.get("open")),
                     Step(t("tour.recent.title"), t("tour.recent.text"), lambda: tg.get("recent"),
                          lambda: tg.get("recent") is not None),
                     Step(t("tour.lang.title"), t("tour.lang.text"), lambda: tg.get("language")),
                     Step(t("tour.help.title"), t("tour.help.text"), lambda: tg.get("help"))]
            self._tour = Tour(self.root, steps, on_close=done)
            self._tour.start()
            return
        settings.set("tour_work_seen", True)
        menus = getattr(self, "_menus", {})
        has_photo = lambda: bool(self.current_image_path)
        panel_on = lambda: self._panel_of("steps").winfo_ismapped() and getattr(self, "_analysis_chosen", False)
        results = lambda: bool(self.last_result)
        steps = [
            Step(t("tour.welcome.title"), t("tour.welcome.text"), key="welcome"),
            Step(t("tour.photos.title"), t("tour.photos.text"), lambda: self.image_list, has_photo, key="photos"),
            Step(t("tour.image.title"), t("tour.image.text"), lambda: self.canvas_left, has_photo, key="image"),
            Step(t("tour.areas.title"), t("tour.areas.text"), lambda: menus.get(t("menu.areas")), key="areas"),
            Step(t("tour.analysis.title"), t("tour.analysis.text"), lambda: self._analysis_cards, has_photo,
                 before=self._choose_analysis, key="analysis"),
            Step(t("tour.segment.title"), t("tour.segment.text"), lambda: self._rtab_body,
                 lambda: has_photo() and self._analysis_chosen and self._analysis_segments(),
                 before=lambda: (self._show_panel(self._panel_of("steps"), True), self._select_right_tab("seg")),
                 key="segment"),
            Step(t("tour.params.title"), t("tour.params.text"), lambda: self.config_container, panel_on,
                 before=lambda: self._select_right_tab("analysis"), key="params"),
            Step(t("tour.run.title"), t("tour.run.text"), lambda: getattr(self, "_run_frame", None), key="run"),
            Step(t("tour.layers.title"), t("tour.layers.text"), lambda: self._layers_title.master,
                 results, key="layers"),
            Step(t("tour.results.title"), t("tour.results.text"),
                 lambda: next(iter(self._tab_labels.values())).master, results, key="results"),
            Step(t("tour.export.title"), t("tour.export.text"), lambda: menus.get(t("menu.project")), key="export"),
            Step(t("tour.end.title"), t("tour.end.text"), key="end"),
        ]
        # empieza donde está el usuario: lo siguiente que le toca hacer
        if not has_photo():
            at = "welcome"
        elif results():
            at = "layers"
        elif panel_on():
            at = "params"
        elif self._panel_of("steps").winfo_ismapped():
            at = "segment" if self._right_tab == "seg" else "analysis"
        else:
            at = "areas"
        self._tour = Tour(self.root, steps, on_close=done)
        self._tour.start(at)
