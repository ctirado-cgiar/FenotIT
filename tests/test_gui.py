"""Interfaz completa: recorre el flujo de trabajo en la ventana real y se detiene en el primer error.

Abre la aplicación, agrega fotos, segmenta, analiza una foto y el lote, recorre las capas,
elige y excluye un objeto, deshace, abre gráficos y ventanas, y exporta. Cualquier
excepción de la interfaz, de un hilo o del registro (nivel ERROR) se imprime con su traza y
la prueba termina con código 1.

En Linux sin pantalla se ejecuta dentro de Xvfb (`xvfb-run`). Uso:

    python tests/test_gui.py            # flujo normal (~1 min)
    python tests/test_gui.py --sam      # agrega MobileSAM (~1 min más)
"""
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = [str(ROOT / "tests" / "images" / n) for n in ("01.jpg", "02.jpg")]

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    if not shutil.which("xvfb-run"):
        print("skip: sin pantalla ni xvfb-run")
        sys.exit(0)
    sys.exit(subprocess.call(["xvfb-run", "-a", "-s", "-screen 0 1600x1000x24",
                              sys.executable, *sys.argv]))

os.environ["LOCALAPPDATA"] = tempfile.mkdtemp()          # configuración del usuario aparte
os.environ["APPDATA"] = os.environ["LOCALAPPDATA"]
sys.path.insert(0, str(ROOT))

import tkinter as tk                                    # noqa: E402
from tkinter import filedialog, messagebox              # noqa: E402

from fenotit import i18n, settings                      # noqa: E402

ERRORS: list[str] = []
NOTES: list[str] = []


class _Collect(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            ERRORS.append(f"log {record.name}: {record.getMessage()}")


def _answer(value):
    def ask(title, message=None, **_):
        NOTES.append(f"{title}: {message}")
        return value
    return ask


def _quiet_dialogs():
    """Las ventanas de aviso y de pregunta responden solas y quedan anotadas."""
    for name in ("showinfo", "showwarning", "showerror"):
        setattr(messagebox, name, _answer("ok"))
    for name in ("askyesno", "askokcancel", "askyesnocancel", "askretrycancel"):
        setattr(messagebox, name, _answer(False))


class Flow:
    """Pasos que esperan una condición (con tiempo máximo) antes de seguir."""

    def __init__(self, root):
        self.root = root
        self.steps = []
        self.t0 = time.perf_counter()

    def step(self, name, wait=None, timeout=60):
        def deco(fn):
            self.steps.append((name, fn, wait, timeout))
            return fn
        return deco

    def run(self):
        self._next(0)
        self.root.mainloop()

    def _next(self, i):
        if i >= len(self.steps):
            self.root.after(100, self.root.destroy)
            return
        name, fn, wait, timeout = self.steps[i]
        try:
            fn()
        except Exception:
            ERRORS.append(f"{name}: {traceback.format_exc()}")
            self.root.after(100, self.root.destroy)
            return
        t0 = time.perf_counter()

        def poll():
            try:
                ready = wait is None or wait()
            except Exception:
                ready = False
            if ready:
                print(f"ok  {name}  ({time.perf_counter() - self.t0:.0f} s)", flush=True)
                self.root.after(150, lambda: self._next(i + 1))
            elif time.perf_counter() - t0 > timeout:
                ERRORS.append(f"{name}: no terminó en {timeout} s")
                self.root.after(100, self.root.destroy)
            else:
                self.root.after(100, poll)
        self.root.after(200, poll)


def _close_windows(root):
    for w in root.winfo_children():
        if isinstance(w, tk.Toplevel):
            w.destroy()


def main(with_sam=False):
    _quiet_dialogs()
    logging.getLogger().addHandler(_Collect())
    threading.excepthook = lambda a: ERRORS.append(f"hilo: {a.exc_type.__name__}: {a.exc_value}")
    settings.set("tour_seen", True)
    settings.set("tour_work_seen", True)
    i18n.set_language("es")
    i18n.load("es")

    from fenotit.core.analysis.registry import ANALYSES
    from fenotit.gui.main_window import MainWindow

    root = tk.Tk()
    root.geometry("1500x900+0+0")
    root.report_callback_exception = lambda *exc: ERRORS.append("".join(traceback.format_exception(*exc)))
    w = MainWindow(root)
    flow = Flow(root)
    objects, image_color = list(ANALYSES)[:2]
    out = Path(tempfile.mkdtemp())

    def result(name=objects, path=None):
        return w._results.get(name, {}).get(path or w.current_image_path)

    @flow.step("agregar fotos", wait=lambda: w.scaler_left.original is not None)
    def _():
        w._add_images(IMAGES)

    @flow.step("elegir análisis y umbral Otsu")
    def _():
        w._open_analysis(objects)
        w.cs_var.set("LAB")
        w._on_cs_change()
        w.ch_var.set(2)
        w.auto_var.set(True)
        w._on_auto_change()

    @flow.step("afinar máscara")
    def _():
        w.refine_list._add("open")
        w.refine_list.set([{"op": "fill_holes"}])
        w._on_refine_change()

    @flow.step("analizar una foto", wait=lambda: result() is not None, timeout=90)
    def _():
        w._run_analysis(False)

    @flow.step("recorrer capas")
    def _():
        for _ in range(8):
            w._step_view(1)
            root.update()
        for _ in range(8):
            w._step_view(-1)
            root.update()

    @flow.step("elegir y excluir un objeto", wait=lambda: 1 in w.project.excluded_for(w.current_image_path)
               or bool(w.project.excluded_for(w.current_image_path)), timeout=60)
    def _():
        w._select_object(1)
        w._toggle_exclude()

    @flow.step("deshacer", wait=lambda: not w.project.excluded_for(w.current_image_path), timeout=60)
    def _():
        w._undo()

    @flow.step("tabla y gráficos")
    def _():
        w._select_bottom_tab("charts")
        root.update()
        w._chart_fullscreen()
        root.update()
        _close_windows(root)
        w._select_bottom_tab("table")

    @flow.step("analizar el lote", wait=lambda: w._batch is None and all(result(path=p) for p in IMAGES),
               timeout=180)
    def _():
        w._run_analysis(True)

    @flow.step("comparar fotos y colores")
    def _():
        w._show_batch_chart()
        root.update()
        w._show_color3d()
        root.update()
        _close_windows(root)

    @flow.step("cambiar de foto", wait=lambda: w.current_image_path == IMAGES[1])
    def _():
        w._select_image(1)

    @flow.step("exportar")
    def _():
        class Dialog:
            def finished(self, text, ok=True, folder=None):
                if not ok:
                    ERRORS.append(f"exportar: {text}")

            def progress(self, *a):
                pass
        w._run_export(str(out), [objects], True, False, Dialog())
        files = {p.name for p in out.rglob("*")}
        for need in ("image.csv", "objects.csv", "metadata.csv"):
            if need not in files:
                ERRORS.append(f"exportar: falta {need}")

    @flow.step("guardar y volver a abrir el proyecto",
               wait=lambda: w.current_image_path is not None and w.scaler_left.original is not None)
    def _():
        filedialog.asksaveasfilename = lambda **_: str(out / "prueba.fenotit")
        w._save_project_as()
        saved = w.project.file
        if not saved or not Path(saved).exists():
            ERRORS.append("guardar: no se creó el proyecto")
            return
        w.open_project_path(str(saved))

    @flow.step("ventanas y diálogos")
    def _():
        for open_ in (w._open_preferences, w._about, w._show_shortcuts, w._open_corrections,
                      w._open_scale_dialog):
            open_()
            root.update()
            _close_windows(root)

    @flow.step("segmentación por ejemplos y de vuelta a umbral")
    def _():
        w._set_seg_method("examples")
        root.update()
        w._set_seg_method("threshold")

    @flow.step("quitar una foto y deshacer", wait=lambda: len(w.batch_paths) == 2)
    def _():
        w._remove_current_image()
        root.update()
        w._undo()

    @flow.step("color de imagen", wait=lambda: result(image_color) is not None, timeout=90)
    def _():
        w._open_analysis(image_color)
        w._run_analysis(False)

    if with_sam:
        @flow.step("MobileSAM", wait=lambda: "…" not in w.sam_status.get() and w._sam_labels() is not None,
                   timeout=240)
        def _():
            w._open_analysis(objects)
            w._set_seg_method("model")
            w._select_right_tab("seg")
            w.category_list.set([{"name": "A", "color": "#2E9E44"}, {"name": "B", "color": "#D62728"}])
            w._on_categories_change()
            w._sam_try_now()

    flow.run()
    shutil.rmtree(out, ignore_errors=True)
    for n in NOTES:
        print("aviso:", n)
    if ERRORS:
        print("\n".join(["", "ERRORES:"] + ERRORS))
        sys.exit(1)
    print("ok  interfaz completa")


if __name__ == "__main__":
    main(with_sam="--sam" in sys.argv)
