"""Lote en paralelo (procesos): cada foto se carga, se corrige y se analiza en un proceso
aparte. Vuelve solo lo liviano (tablas, estadísticas, marcas), no las imágenes: un
resultado completo pesa ~70 MB en una foto de 8 MP; sin imágenes, ~0.1 MB.

Sin tkinter: la interfaz recibe los resultados por callbacks (desde un hilo) y los pasa
a su hilo principal."""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from fenotit import log

_log = log.get("batch")

HEAVY = ("base_image",)          # extra que no viaja de vuelta


@dataclass
class Job:
    path: str
    params: dict
    aruco_scale: bool = False     # escala de la foto = la de sus marcadores ArUco


@dataclass
class Outcome:
    path: str
    status: str                   # ok | error | skipped | unreadable
    result: Any = None            # AnalysisResult liviano
    message: str = ""
    aruco_missing: list = field(default_factory=list)
    mm_per_px: float | None = None


def light(result):
    """El resultado sin imágenes; `views` guarda los nombres de las vistas."""
    extra = {k: v for k, v in result.extra.items() if k not in HEAVY}
    extra["views"] = list(result.step_images)
    extra["light"] = True
    return replace(result, step_images={}, extra=extra)


def default_workers() -> int:
    """Núcleos − 1, máx. 6 (cada proceso puede usar ~1 GB con fotos grandes)."""
    return max(1, min((os.cpu_count() or 2) - 1, 6))


def process(analysis: str, job: Job, corrections, keep_images: bool = False) -> Outcome:
    """Una foto: leer → corregir → analizar. Se ejecuta en el proceso de trabajo."""
    from fenotit.core.analysis.registry import ANALYSES, AnalysisResult
    from fenotit.core.corrections import pipeline as corr
    from fenotit.core.image_io import load_image

    img = load_image(job.path)
    if img is None:
        return Outcome(job.path, "unreadable")
    img, info = corr.apply(img, corrections)
    if corrections.perspective.enabled and "perspective" not in info.applied:
        return Outcome(job.path, "skipped", aruco_missing=info.aruco_missing)
    params = dict(job.params)
    if job.aruco_scale:
        params["mm_per_pixel"] = info.mm_per_px
    try:
        result = ANALYSES[analysis].func(img, params)
    except Exception as e:
        _log.exception("%s falló en %s", analysis, job.path)
        result = AnalysisResult(status="error", error=str(e))
    result.extra["params"] = params          # para recalcular la vista o exportarla igual
    if result.status != "ok":
        return Outcome(job.path, "error", message=result.error, mm_per_px=params.get("mm_per_pixel"))
    return Outcome(job.path, "ok", result if keep_images else light(result),
                   mm_per_px=params.get("mm_per_pixel"))


def export_views(folder: str, display: dict, analysis: str, job: Job, corrections) -> Outcome:
    """Recalcula la foto con sus mismos parámetros y guarda las imágenes de sus vistas."""
    from pathlib import Path
    from fenotit.core.export.exporter import save_views
    out = process(analysis, job, corrections, keep_images=True)
    if out.status == "ok":
        try:
            save_views(out.result, Path(folder), Path(job.path).name, display)
        except Exception as e:
            _log.exception("Vistas de %s", job.path)
            out.status, out.message = "error", str(e)
        out.result = light(out.result)
    return out


def _init_worker(language: str | None = None):
    """El idioma de la app (los nombres de las vistas y las leyendas salen de él) y los
    errores al mismo registro, sin rotarlo (en Windows varios procesos no pueden renombrar
    el archivo a la vez)."""
    import logging
    from fenotit import i18n
    if language:
        i18n.load(language)
    root = logging.getLogger("fenotit")
    if root.handlers:
        return
    root.setLevel(logging.WARNING)
    try:
        log.log_dir().mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log.log_file(), encoding="utf-8", delay=True)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s [lote]: %(message)s"))
        root.addHandler(fh)
    except OSError:
        pass


class BatchRunner:
    """Corre los trabajos en procesos y avisa con `on_outcome(i, n, outcome)` y
    `on_done(cancelled)` (desde un hilo: la interfaz debe pasarlo a su hilo principal)."""

    def __init__(self, analysis: str, jobs: list[Job], corrections,
                 on_outcome: Callable[[int, int, Outcome], None], on_done: Callable[[bool], None],
                 workers: int | None = None, task: Callable = process):
        self.analysis, self.jobs, self.corrections = analysis, jobs, corrections
        self.on_outcome, self.on_done = on_outcome, on_done
        self.workers = workers or default_workers()
        self.task = task
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _run(self):
        n = len(self.jobs)
        done = 0
        workers = min(self.workers, n) or 1
        from fenotit import i18n
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(i18n.current(),),
                                   mp_context=mp.get_context("spawn"))   # igual en Windows y Linux
        try:
            todo = iter(self.jobs)
            running = {}

            def submit():                       # de a pocos: cancelar no deja una cola larga
                while len(running) < workers * 2 and not self._cancel.is_set():
                    job = next(todo, None)
                    if job is None:
                        return
                    running[pool.submit(self.task, self.analysis, job, self.corrections)] = job

            submit()
            while running and not self._cancel.is_set():
                finished, _ = wait(running, timeout=0.2, return_when=FIRST_COMPLETED)
                for fut in finished:
                    job = running.pop(fut)
                    try:
                        out = fut.result()
                    except Exception as e:          # proceso caído (memoria, etc.)
                        _log.exception("Lote: %s", job.path)
                        out = Outcome(job.path, "error", message=str(e))
                    done += 1
                    if not self._cancel.is_set():
                        self.on_outcome(done, n, out)
                submit()
        except Exception:
            _log.exception("Lote")
        cancelled = self._cancel.is_set()
        if cancelled:                               # no esperar a las fotos que ya corrían
            procs = list((getattr(pool, "_processes", None) or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    _log.debug("terminate", exc_info=True)
        else:
            pool.shutdown(wait=True)
        self.on_done(cancelled)
