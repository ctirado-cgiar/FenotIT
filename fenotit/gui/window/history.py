"""Deshacer y rehacer: estados del proyecto antes de cada cambio."""
from __future__ import annotations

from fenotit.i18n import t


class HistoryMixin:
    UNDO_MAX = 50

    _UNDO_FIELDS = ("roi", "image_roi", "image_scale", "excluded", "highlighted", "images",
                    "scale", "unit", "max_mpx", "corrections", "metadata", "strokes", "prompts", "regions")

    def _snapshot(self) -> dict:
        """Lo que Ctrl+Z puede devolver: áreas, marcas, fotos de la lista, segmentación
        (espacio, canal, Mín/Máx, Otsu) y parámetros del análisis elegido."""
        import copy
        snap = {f: copy.deepcopy(getattr(self.project, f)) for f in self._UNDO_FIELDS}
        snap["_seg"] = self._seg_state()
        cp = getattr(self, "config_panel", None)
        snap["_params"] = (self._selected_analysis(), cp.get_values(warn=False) if cp else {})
        return snap

    def _push_undo(self, what: str, path, snap: dict):
        stack = self.__dict__.setdefault("_undo_stack", [])
        stack.append((what, path, snap))
        del stack[:-self.UNDO_MAX]
        self._redo_stack = []

    def _remember(self, what: str):
        """Antes de cambiar áreas, marcas o la lista de fotos: guarda cómo estaba."""
        if getattr(self, "_settle_job", None):          # un cambio de umbral pendiente va antes
            self.root.after_cancel(self._settle_job)
            self._settle()
        self._push_undo(what, self.current_image_path, self._snapshot())
        self.root.after_idle(self._mark_settled)

    # Segmentación y parámetros cambian de a poco (arrastrar un deslizador): se guarda un
    # paso de deshacer cuando se quedan quietos 0.7 s.
    def _settings_changed(self):
        if getattr(self, "_restoring", False) or not hasattr(self, "min_slider"):
            return
        if getattr(self, "_settle_job", None):
            self.root.after_cancel(self._settle_job)
        self._settle_job = self.root.after(700, self._settle)

    def _settle(self):
        self._settle_job = None
        old, now = getattr(self, "_settled", None), self._snapshot()
        if old is not None:
            seg = old["_seg"] != now["_seg"]
            params = old["_params"][0] == now["_params"][0] and old["_params"][1] != now["_params"][1]
            if seg or params:
                self._push_undo("undo.segmentation" if seg else "undo.params", self.current_image_path, old)
        self._settled = now

    def _mark_settled(self):
        self._settled = self._snapshot()

    def _undo_label(self, which: str) -> str:
        stack = getattr(self, "_undo_stack" if which == "undo" else "_redo_stack", None) or []
        return t(f"edit.{which}") + (f": {t(stack[-1][0])}" if stack else "")

    def _undo(self):
        self._swap("_undo_stack", "_redo_stack", "edit.undone")

    def _redo(self):
        self._swap("_redo_stack", "_undo_stack", "edit.redone")

    def _swap(self, src: str, dst: str, msg: str):
        if getattr(self, "_settle_job", None):        # un cambio a medio guardar entra primero
            self.root.after_cancel(self._settle_job)
            self._settle()
        stack = getattr(self, src, None) or []
        if not stack:
            return
        what, path, state = stack.pop()
        self.__dict__.setdefault(dst, []).append((what, path, self._snapshot()))
        self._restoring = True
        try:
            self._restore(path, state)
        finally:
            self._restoring = False
        self._mark_settled()
        self._set_status(t(msg, what=t(what)))

    def _restore(self, path, state: dict):
        key = self.project.key(path) if path else None
        marks_changed = any(getattr(self.project, f).get(key) != state[f].get(key) for f in ("excluded", "highlighted"))
        # la lista de fotos solo vuelve atrás si se quitó una (agregar fotos no se deshace):
        # las que faltan vuelven a su lugar, las agregadas después se quedan
        now = [str(p) for p in self.project.images]
        missing = [(i, p) for i, p in enumerate(state["images"]) if str(p) not in now]
        images = list(self.project.images)
        for i, p in missing:
            images.insert(min(i, len(images)), p)
        images_changed = bool(missing)
        changed = {f for f in ("scale", "unit", "image_scale", "max_mpx", "corrections", "metadata")
                   if getattr(self.project, f) != state[f]}
        for f in self._UNDO_FIELDS:
            if f != "images":
                setattr(self.project, f, state[f])
        self.project.images = images
        if state["_seg"] != self._seg_state():
            method_changed = state["_seg"].get("method") != self._seg_method
            self._set_seg_state(state["_seg"])
            if method_changed and self.config_panel is not None:
                self._on_analysis_selected(None, choose=getattr(self, "_analysis_chosen", False))
        if self.current_image_path:
            self.strokes.set_strokes(self.project.strokes.get(self.project.key(self.current_image_path)))
        self._on_slider_change()
        name, values = state["_params"]
        if self.config_panel is not None and name == self._selected_analysis():
            self.config_panel.set_values(values)
        if images_changed:                            # foto quitada que vuelve (sin resultados)
            self._update_batch_list()
            self._update_start()
            self._show_panel(self.left_panel, True)
        if changed & {"scale", "unit", "image_scale"}:
            self._update_corr_indicator()
            self._sync_area_units()
        if changed & {"max_mpx", "corrections"}:      # cambian la foto que se analiza
            self._invalidate_results()
            if self.current_image_path:
                self._load_single(self.current_image_path)
            self._update_corr_indicator()
        if "metadata" in changed:
            self._meta_cache = None
            self.image_list._filter()
            self._show_meta()
        if images_changed and not self.current_image_path and self.batch_paths:
            self._select_image(0)
        if path and path in self.batch_paths and path != self.current_image_path:
            self._select_image(self.batch_paths.index(path))       # se ve lo que se deshizo
        elif self.areas and self.current_image_path:
            self.areas.set_shapes(self.project.roi_for(self.current_image_path))
            self._update_area_scope()
            self._show_preview()
        if marks_changed and path in self.results_cache:       # excluidos/destacados: se recalcula
            self._run_analysis(all_images=False)
