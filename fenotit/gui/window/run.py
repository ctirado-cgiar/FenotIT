"""Ejecutar el análisis en una foto o en el lote, y avance en la barra de estado."""
from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np

from fenotit import log, settings
from fenotit.core import batch
from fenotit.core.analysis.registry import ANALYSES, AnalysisResult
from fenotit.core.image_io import load_image
from fenotit.gui.window.common import CV2_CODES, _analysis_label, _duration
from fenotit.i18n import t
from fenotit.gui.style.theme import SPACE


_log = log.get("gui.window.run")


class RunMixin:
    def _init_run(self):
        self._skipped: set[str] = set()          # fotos omitidas (⚠ en la lista)
        self._pending: set[str] = set()          # en cola del lote (… en la lista)
        self._batch: batch.BatchRunner | None = None
        self._rerun_queue: list[str] = []
        self._forget_examples()

    def _forget_examples(self):
        """Los ejemplos de "Por ejemplos" se vuelven a leer de los trazos."""
        self._examples = {}

    def _queue_reruns(self, names: list[str]):
        """Corre los lotes de `names` uno tras otro."""
        self._rerun_queue = list(names)
        self._rerun_next()

    def _corr_signature(self) -> str:
        return json.dumps([self.project.corrections.to_dict(), self.project.max_mpx], sort_keys=True, default=str)

    def _gather_examples(self, strokes: dict, detail: float, cur_path, cur_feat, corrections, max_mpx, sig):
        """Rasgos de los píxeles pintados en todas las fotos con trazos (en un hilo). Las
        fotos que no están abiertas se leen y corrigen igual que en el análisis."""
        from fenotit.core.segmentation import classifier as C
        from fenotit.core.corrections import pipeline as corr
        Xs, ys = [], []
        for key, st in strokes.items():
            if not st:
                continue
            k = (json.dumps(st, sort_keys=True), detail, sig if key != cur_path else ("cur", id(cur_feat)))
            hit = self._examples.get(key)
            if hit and hit[0] == k:
                X, y = hit[1], hit[2]
            else:
                if key == cur_path and cur_feat is not None:
                    F = cur_feat
                else:
                    img = load_image(key)
                    if img is None:
                        continue
                    img, info = corr.apply(img, corrections)
                    img = corr.to_working(img, info, max_mpx)
                    F = C.features(C.work_copy(img, detail))
                X, y = C.samples(F, st)
                self._examples[key] = (k, X, y)
            Xs.append(X)
            ys.append(y)
        if not Xs:
            return None, None
        return C.balance(np.concatenate(Xs), np.concatenate(ys))

    def _with_examples(self, params: dict, then, on_fail=None):
        """Antes de correr con la segmentación por ejemplos: junta los ejemplos de todas
        las fotos, los guarda (.npz en la carpeta de datos de la app) y deja su ruta en los
        parámetros; luego `then(params)`."""
        seg = params.get("segmentation") or {}
        if seg.get("method") == "model" and (seg.get("model") or {}).get("mode") == "auto":
            ex = self._region_examples()
            if ex is not None:
                from fenotit.core.segmentation import classifier as C
                path, key = C.store(*ex, settings.data_dir() / "regions")
                seg["model"]["learn"] = {"samples": path, "hash": key}
            then(params)
            return
        if seg.get("method") != "classifier":
            then(params)
            return
        from fenotit.core.segmentation import classifier as C
        strokes = {k: list(v) for k, v in self.project.strokes.items()
                   if k in {self.project.key(p) for p in self.project.images}}
        if len({s["c"] for v in strokes.values() for s in v}) < 2:
            messagebox.showinfo(t("seg.method.classifier"), t("cls.need_two_msg"), parent=self.root)
            if on_fail:
                on_fail()
            return
        detail = seg["classifier"]["detail"]
        full = self.scaler_left.original if self.scaler_left.has_image else None
        cur = self.project.key(self.current_image_path) if self.current_image_path else None
        feat = getattr(self, "_feat_cache", None)
        F = feat[1] if feat is not None and full is not None and feat[0] == (id(full), detail) else None
        corrections, max_mpx, sig = self.project.corrections, self.project.max_mpx, self._corr_signature()
        self._set_status(t("cls.preparing"))

        def work():
            try:
                X, y = self._gather_examples(strokes, detail, cur if F is not None else None, F,
                                             corrections, max_mpx, sig)
                path, key = C.store(X, y, settings.data_dir() / "examples")
            except Exception as e:
                _log.exception("Ejemplos")
                msg = str(e)
                self.root.after(0, lambda: (messagebox.showerror(t("common.error"), msg, parent=self.root),
                                            on_fail and on_fail()))
                return
            seg["classifier"].update(samples=path, hash=key)
            self.root.after(0, lambda: then(params))
        threading.Thread(target=work, daemon=True).start()

    def _build_params(self) -> dict:
        p = {
            "color_space":      self.cs_var.get(),
            "color_space_code": CV2_CODES.get(self.cs_var.get()),
            "channel_idx":      self.ch_var.get(),
            "min_val":          self.min_slider.get(),
            "max_val":          self.max_slider.get(),
            "mm_per_pixel":     self.mm_per_pixel,
            "length_unit":      self.project.unit,
        }
        p["roi_shapes"] = self.project.roi_for(self.current_image_path)
        p["drop_points"] = self.project.excluded_for(self.current_image_path)
        p["mark_points"] = self.project.marks_for("highlighted", self.current_image_path)
        if self.config_panel:
            p.update(self.config_panel.get_values())
            p["area_unit"] = getattr(self, "_panel_area_unit", "px")
        p["auto_threshold"] = bool(self.auto_var.get())
        p["segmentation"] = self._seg_params()
        info = self._corr_info.get(self.current_image_path)
        p["work_scale"] = round(info.work_k, 6) if info else 1.0
        return p

    def _run_analysis(self, all_images: bool = False):
        """Ejecutar: la imagen actual; con all_images, todas las de la lista."""
        name = self._selected_analysis()
        if name not in ANALYSES or not getattr(self, "_analysis_chosen", False):
            messagebox.showinfo(t("msg.no_analysis_title"), t("msg.no_analysis"), parent=self.root)
            return
        if all_images:
            self._run_batch(name)
            return
        if getattr(self, "_loading", None):
            self._set_status(t("status.loading_photo"))
            return
        if not self.scaler_left.has_image:
            messagebox.showwarning(t("msg.no_image_title"),
                                   t("msg.no_image"))
            return
        reason = t("corr.skip_board") if self.current_image_path in self._board_photos() else \
            self._skip_reason(self._corr_info.get(self.current_image_path))
        if reason:
            messagebox.showwarning(t("corr.skip_title"), reason, parent=self.root)
            return
        self._run_single(name, self.scaler_left.original,
                         self.current_image_path or "imagen")

    def _run_single(self, name: str, image: np.ndarray, path: str):
        self._with_examples(self._build_params(), lambda params: self._start_single(name, image, path, params))

    def _start_single(self, name: str, image: np.ndarray, path: str, params: dict):
        self._set_status(t("status.running", name=_analysis_label(name)))
        self.root.config(cursor="watch")

        def worker():
            t0 = time.perf_counter()
            try:
                result = ANALYSES[name].func(image, params)
            except Exception as e:
                _log.exception("%s falló en %s", name, path)
                result = AnalysisResult(status="error", error=str(e))
            result.extra["params"] = params
            result.extra["processing_s"] = round(time.perf_counter() - t0, 3)
            self.root.after(0, lambda: self._on_result(name, result, path))
        threading.Thread(target=worker, daemon=True).start()

    def _run_batch(self, name: str):
        """Todas las fotos en procesos aparte (core.batch): vuelven tablas y marcas, no
        imágenes; la vista de una foto se recalcula al abrirla."""
        if not self.batch_paths:
            messagebox.showwarning(t("msg.no_batch_title"), t("msg.no_batch"))
            return
        if self._batch is not None or getattr(self, "_preparing", False):
            return
        board = self._board_photos()
        paths = [p for p in self.batch_paths if p not in board]
        if not paths:
            messagebox.showwarning(t("msg.no_batch_title"), t("msg.no_batch"))
            return
        self._preparing = True                     # no se lanza otro mientras se juntan los ejemplos

        def go(params):
            self._preparing = False
            self._start_batch(name, paths, params)
        self._with_examples(self._build_params(), go, on_fail=lambda: setattr(self, "_preparing", False))

    def _start_batch(self, name: str, paths: list, params: dict):
        total = len(paths)
        self._set_status(t("status.running_batch", name=_analysis_label(name), n=total))
        jobs = []
        for path in paths:
            sc = self.project.scale_for(path)
            aruco = sc.source == "aruco"
            seg = params.get("segmentation") or {}
            if seg.get("method") == "model":
                seg = dict(seg, model=self._model_for(seg["model"], path))
            jobs.append(batch.Job(path, dict(params, segmentation=seg, mm_per_pixel=None if aruco else sc.mm_per_pixel,
                                             roi_shapes=self.project.roi_for(path),
                                             drop_points=self.project.excluded_for(path),
                                             mark_points=self.project.marks_for("highlighted", path)),
                                  aruco_scale=aruco,
                                  max_mpx=self.project.max_mpx))
        self._pending = set(paths)                 # al volver a correr, las marcas se rehacen
        self._skipped -= self._pending
        self._progress(0, total)
        self._update_batch_list()
        self._sync_listbox()
        self._batch_t0 = time.perf_counter()
        after = self.root.after
        self._batch = batch.BatchRunner(
            name, jobs, self.project.corrections, workers=int(settings.get("workers", 0) or 0) or None,
            on_outcome=lambda i, n, o: after(0, lambda: self._on_batch_outcome(name, i, n, o)),
            on_done=lambda cancelled: after(0, lambda: self._on_batch_done(name, total, cancelled))).start()

    def _cancel_batch(self):
        if self._batch is not None:
            self._batch.cancel()
            self._set_status(t("status.cancelling"))

    def _on_batch_outcome(self, name: str, i: int, n: int, out: "batch.Outcome"):
        path = out.path
        self._pending.discard(path)
        if out.status == "skipped":
            d = self.project.corrections.distortion
            reason = t("corr.skip_distortion", size="×".join(map(str, d.size or [])) or "?") \
                if out.message == "distortion" else t("corr.skip_color") if out.message == "color" else \
                t("corr.skip_aruco", ids=", ".join(map(str, out.aruco_missing)) or "?")
            self._mark_skipped(path, reason)
        elif out.status == "unreadable":
            self._mark_skipped(path, t("msg.image_unreadable", name=Path(path).name))
        else:
            result = out.result or AnalysisResult(status="error", error=out.message)
            if out.status == "error" and not result.error:
                result.error = out.message
            self._cache_result(name, path, result)
        self._set_status(t("status.batch_progress", name=_analysis_label(name), i=i, n=n))
        self._progress(i, n)
        self._update_batch_list()
        if path in self.batch_paths:                 # la lista sigue al lote (si el usuario no la mueve)
            self.image_list.reveal(self.batch_paths.index(path))

    def _cache_result(self, name: str, path: str, result: AnalysisResult):
        """Guarda el resultado del análisis que corre (aunque el usuario cambie de
        análisis) y, si es la foto que se ve, lo muestra."""
        views = list(result.step_images) or result.extra.get("views", [])
        self._results.setdefault(name, {})[path] = result
        self._step_names.setdefault(name, {})[path] = views
        if path == self.current_image_path and result.status == "ok" and name == self.active_analysis:
            self._display_result(result, views, fresh=True)
            self._refresh_history()

    def _on_batch_done(self, name: str, total: int, cancelled: bool):
        self._batch = None
        self._write_preview()
        if not cancelled:
            self._mark_ran(name)
        if cancelled:
            self._rerun_queue = []
        elif getattr(self, "_rerun_queue", None):
            self.root.after(300, self._rerun_next)
        self._pending.clear()
        self._progress(None)
        self._update_batch_list()
        store = self._results.get(name, {})
        ok = sum(1 for p in self.batch_paths if p in store and store[p].status == "ok")
        failed = sum(1 for p in self.batch_paths if p in store and store[p].status != "ok")
        n_skipped = sum(1 for p in self.batch_paths if p in self._skipped)
        key = "status.batch_cancelled" if cancelled else "status.batch_done"
        msg = t(key, name=_analysis_label(name), ok=ok, n=total)
        if n_skipped:
            msg += ". " + t("status.skipped", n=n_skipped)
        if failed:
            msg += ". " + t("status.failed", n=failed)
        elapsed = time.perf_counter() - getattr(self, "_batch_t0", time.perf_counter())
        done = ok + failed
        if done:
            msg += ". " + t("status.batch_time", total=_duration(elapsed), per=_duration(elapsed / done))
        self._set_status(msg)
        if n_skipped and not cancelled:
            messagebox.showwarning(t("corr.skip_title"), t("corr.skip_summary", n=n_skipped), parent=self.root)

    def _log_process(self, name: str):
        """Agrega al log del panel izquierdo el proceso ejecutado."""
        if hasattr(self, 'log_listbox'):
            self.log_listbox.insert(tk.END, f"✓  {name}")
            self.log_listbox.see(tk.END)

    def _progress(self, i: int | None, n: int = 0):
        """Barra de avance del lote en la barra de estado (None = ocultarla)."""
        if i is None:
            self._progress_bar.pack_forget()
            self._cancel_link.pack_forget()
            return
        self._progress_bar.config(maximum=max(n, 1), value=i)
        if not self._progress_bar.winfo_ismapped():
            self._cancel_link.pack(side=tk.RIGHT, padx=(0, SPACE["md"]), before=self._status_label)
            self._progress_bar.pack(side=tk.RIGHT, padx=(SPACE["md"], SPACE["xs"]), before=self._status_label)

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.root.update_idletasks()
