"""Exportar resultados."""
from __future__ import annotations

from pathlib import Path
from tkinter import messagebox

from fenotit import log, settings
from fenotit.core import batch
from fenotit.core import units
from fenotit.core.analysis.registry import ANALYSES, AnalysisResult
from fenotit.gui.dialogs.export import ExportDialog
from fenotit.gui.window.common import _analysis_key, _analysis_label
from fenotit.i18n import t


_log = log.get("gui.window.export")


class ExportMixin:
    def _export_results(self):
        """Una sola exportación: tablas + metadatos por análisis (y, si se pide, Excel e
        imágenes de las vistas, que se recalculan en procesos aparte)."""
        done = [(n, _analysis_label(n), sum(1 for p in self.batch_paths if p in self._results.get(n, {})
                                            and self._results[n][p].status == "ok"))
                for n in ANALYSES if self._results.get(n)]
        done = [d for d in done if d[2]]
        if not done:
            messagebox.showwarning(t("msg.no_results_title"), t("msg.no_results"), parent=self.root)
            return
        folder = settings.get("export_dir") or (str(self.project.folder) if self.project.folder else "") \
            or (str(Path(self.batch_paths[0]).parent) if self.batch_paths else "")
        n = len(self.batch_paths)
        scopes = [("all", t("export.scope_all", n=n))]
        if self.current_image_path and n > 1:
            scopes.append(("current", t("export.scope_current")))
        found = self._search_paths()
        if found is not None and 0 < len(found) < n:
            scopes.append(("search", t("export.scope_search", n=len(found))))
        ExportDialog(self.root, done, folder, run=self._run_export, cancel=self._cancel_export, scopes=scopes,
                     stamp=bool(settings.get("export_stamp", False)), existing=self._export_existing)

    def _search_paths(self) -> list[str] | None:
        """Fotos que deja ver el buscador de la lista (None si no hay búsqueda)."""
        if not self.image_list._query():
            return None
        return [self.batch_paths[i] for i in self.image_list.visible if i < len(self.batch_paths)]

    def _export_existing(self, folder: str, names: list[str]) -> str:
        """Subcarpetas de análisis que ya tienen archivos (se reemplazarían)."""
        hits = [f"{_analysis_key(n)}/" for n in names
                if (Path(folder) / _analysis_key(n)).is_dir() and any((Path(folder) / _analysis_key(n)).iterdir())]
        return ", ".join(hits)

    def _export_folder_name(self) -> str:
        import datetime as _dt
        import re
        stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M")
        base = self.project.name if self.project.folder else "export"
        base = re.sub(r'[<>:"/\\|?*]+', "_", base).strip() or "export"
        return f"{base}_{stamp}"

    def _image_info(self, path: str, result: AnalysisResult | None = None) -> dict:
        sc = self.project.scale_for(path)
        shapes = self.project.roi_for(path)
        own = self.project.key(path) in self.project.image_scale
        used = (result.extra.get("params") or {}) if result is not None else {}
        mpp = used.get("mm_per_pixel") if "mm_per_pixel" in used else self._scale_for(path)
        return {"file": path, "mm_per_px": mpp or "", "work_scale": used.get("work_scale", ""),
                "scale_source": sc.source + (" (image)" if own else ""),
                "length_unit": units.ascii_name(self.project.unit),
                "n_analysis_areas": sum(1 for x in shapes if x["kind"] == "include"),
                "n_excluded_areas": sum(1 for x in shapes if x["kind"] == "exclude")}

    def _run_export(self, folder: str, names: list[str], excel: bool, views: bool, dlg,
                    scope: str = "all", stamp: bool = False):
        from functools import partial
        from fenotit.core.export import exporter
        settings.set("export_dir", folder)
        settings.set("export_stamp", stamp)
        root = Path(folder) / self._export_folder_name() if stamp else Path(folder)
        ids = {p: i for i, p in enumerate(self.batch_paths, 1)}       # Image_ID = posición en el proyecto
        paths = {"current": [self.current_image_path] if self.current_image_path else [],
                 "search": self._search_paths() or []}.get(scope, self.batch_paths)
        written = 0
        try:
            for name in names:
                store = self._results.get(name, {})
                key = _analysis_key(name)
                info = {p: self._image_info(p, store.get(p)) for p in paths}
                part = {p: r for p, r in store.items() if p in info}
                tables = exporter.collect(name, part, paths, info, self._skipped,
                                          board=self._board_photos(), ids=ids,
                                          meta={p: self._meta_row(p) for p in paths})
                meta = exporter.metadata(key, part, self.project,
                                         {"view_images": "views/" if views else "no",
                                          "images_exported": f"{scope} ({len(paths)} of {len(self.batch_paths)})"})
                written += len(exporter.write_tables(root / key, tables, meta, excel, f"{key}.xlsx"))
        except Exception as e:
            _log.exception("Exportación")
            dlg.finished(t("export.error", error=e), ok=False)
            return
        if not views:
            dlg.finished(t("export.done", folder=root), folder=str(root))
            return
        queue = []
        for name in names:
            store = self._results.get(name, {})
            jobs = [batch.Job(p, store[p].extra.get("params") or self._build_params(),
                              max_mpx=self.project.max_mpx, full_res_scale=False)
                    for p in paths if p in store and store[p].status == "ok"]
            if jobs:
                queue.append((name, jobs))
        total = sum(len(j) for _, j in queue)
        state = {"i": 0, "errors": 0}

        def next_analysis(cancelled=False):
            if cancelled or not queue:
                self._export_runner = None
                text = t("export.cancelled") if cancelled else t("export.done", folder=root)
                if state["errors"]:
                    text += ". " + t("status.failed", n=state["errors"])
                dlg.finished(text, ok=not cancelled, folder=str(root))
                return
            name, jobs = queue.pop(0)
            task = partial(batch.export_views, str(root / _analysis_key(name) / "views"), dict(self.project.display))

            def outcome(i, n, out):
                state["i"] += 1
                state["errors"] += out.status != "ok"
                dlg.progress(state["i"], total, t("export.views_progress", i=state["i"], n=total))
            after = self.root.after
            self._export_runner = batch.BatchRunner(
                name, jobs, self.project.corrections, task=task,
                workers=int(settings.get("workers", 0) or 0) or None,
                on_outcome=lambda i, n, o: after(0, lambda: outcome(i, n, o)),
                on_done=lambda c: after(0, lambda: next_analysis(c))).start()
        dlg.progress(0, total, t("export.views_progress", i=0, n=total))
        next_analysis()

    def _cancel_export(self):
        runner = getattr(self, "_export_runner", None)
        if runner is not None:
            runner.cancel()

    def _export_batch(self):
        self._export_results()
