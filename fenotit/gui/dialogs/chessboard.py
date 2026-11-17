"""Calibración del lente con fotos de un tablero de ajedrez, dentro de la pestaña
Distorsión: lista de fotos con su estado (tablero encontrado o no, error de cada una),
miniatura de la foto que se está leyendo o de la elegida, avance y cancelar. Usa como
máximo MAX_PHOTOS fotos."""
from __future__ import annotations

import os
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from fenotit import log
from fenotit.core.corrections import distortion
from fenotit.core.project import IMAGE_EXTS
from fenotit.gui.style.components import button
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.i18n import t

_log = log.get("gui.chessboard")
PREVIEW = (300, 210)


class ChessboardPanel(tk.Frame):
    """Va dentro de la pestaña Distorsión (no abre otra ventana). Al calibrar con éxito
    llama on_result(resultado, fotos_del_tablero_entre_las_cargadas)."""

    def __init__(self, parent, cols: int = 7, rows: int = 6, on_result=None, project_paths=None,
                 extra=None, status: str = ""):
        super().__init__(parent, bg=COLORS["bg_card"])
        self.on_result = on_result
        self.paths: list[str] = []
        self.project_paths = list(project_paths or [])
        self.from_project: set[str] = set()         # fotos del tablero que estaban entre las fotos cargadas
        self._dets: dict[tuple, distortion.Detection] = {}     # (foto, columnas, filas) -> detección
        self._result: distortion.CalibrationResult | None = None
        self._cancel = threading.Event()
        self._busy = False
        self._photo = None
        bg = COLORS["bg_card"]

        row = tk.Frame(self, bg=bg)
        row.pack(fill=tk.X)
        tk.Label(row, text=t("board.corners"), bg=bg, fg=COLORS["text"], font=FONTS["body"]).pack(side=tk.LEFT)
        self.cols, self.rows = tk.StringVar(value=str(cols)), tk.StringVar(value=str(rows))
        for i, var in enumerate((self.cols, self.rows)):
            if i:
                tk.Label(row, text="×", bg=bg, fg=COLORS["text_muted"], font=FONTS["body"]).pack(side=tk.LEFT)
            tk.Entry(row, textvariable=var, width=4, bg=COLORS["bg_panel"], fg=COLORS["text"], relief="solid",
                     bd=1, font=FONTS["mono"], justify="center").pack(side=tk.LEFT, padx=SPACE["sm"])
        tk.Label(row, text=t("board.corners_hint"), bg=bg, fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))

        btns = tk.Frame(self, bg=bg)
        btns.pack(fill=tk.X, pady=(SPACE["md"], 0))
        self._btn(btns, t("board.add"), self._add).pack(side=tk.LEFT)
        if self.project_paths:
            self._btn(btns, t("board.scan", n=len(self.project_paths)), self._scan).pack(side=tk.LEFT, padx=(SPACE["md"], 0))
        self._btn(btns, t("board.remove"), self._remove).pack(side=tk.LEFT, padx=(SPACE["md"], 0))
        self._go = button(btns, text=t("board.run"), command=self._run, kind="primary")
        self._go.pack(side=tk.LEFT, padx=(SPACE["md"], 0))
        if extra:
            text, cmd = extra
            self._btn(btns, text, cmd).pack(side=tk.RIGHT)

        mid = tk.Frame(self, bg=bg)
        mid.pack(fill=tk.BOTH, expand=True, pady=(SPACE["md"], 0))
        left = tk.Frame(mid, bg=bg)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(left, orient="vertical")
        self.tree = ttk.Treeview(left, columns=("state", "error"), show="tree headings", height=8,
                                 yscrollcommand=scrollbar.set, selectmode="extended")
        scrollbar.config(command=self.tree.yview)
        self.tree.heading("#0", text=t("board.col_photo"), anchor="w")
        self.tree.heading("state", text=t("board.col_board"), anchor="center")
        self.tree.heading("error", text=t("board.col_error"), anchor="e")
        self.tree.column("#0", width=170, stretch=True)
        self.tree.column("state", width=60, anchor="center", stretch=False)
        self.tree.column("error", width=70, anchor="e", stretch=False)
        self.tree.tag_configure("bad", foreground=COLORS["error"])
        self.tree.tag_configure("warn", foreground=COLORS["warning"])
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda event: self._show_selected())
        self.tree.bind("<Delete>", lambda event: self._remove())
        self._empty = tk.Label(left, text=t("board.empty"), bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                               font=FONTS["small"])

        right = tk.Frame(mid, bg=bg)
        right.pack(side=tk.LEFT, padx=(SPACE["lg"], 0), anchor="n")
        self.canvas = tk.Canvas(right, width=PREVIEW[0], height=PREVIEW[1], bg=COLORS["bg_panel"],
                                highlightthickness=1, highlightbackground=COLORS["border"])
        self.canvas.pack()
        self.caption = tk.StringVar(value="")
        tk.Label(right, textvariable=self.caption, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                 anchor="w").pack(fill=tk.X, pady=(SPACE["xs"], 0))

        bottom = tk.Frame(self, bg=bg)
        bottom.pack(fill=tk.X, pady=(SPACE["md"], 0))
        self.bar = ttk.Progressbar(bottom, mode="determinate", length=160)
        self._cancel_btn = self._btn(bottom, t("common.cancel"), self.cancel)
        self.status = tk.StringVar(value=status)
        self._status_lbl = tk.Label(bottom, textvariable=self.status, bg=bg, fg=COLORS["text"],
                                    font=FONTS["small"], anchor="w", justify="left", wraplength=620)
        self._status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.bind("<Destroy>", lambda event: self._cancel.set() if event.widget is self else None)
        self._refresh()

    def _ui(self, func):
        """Desde el hilo de trabajo: a la interfaz, si la ventana sigue abierta."""
        try:
            self.after(0, func)
        except (tk.TclError, RuntimeError):
            self._cancel.set()

    def _btn(self, parent, text, cmd):
        return button(parent, text=text, command=cmd, kind="secondary")

    # ── fotos ─────────────────────────────────────────────────────────────────

    def _grid(self) -> tuple[int, int] | None:
        try:
            cols, rows = int(self.cols.get()), int(self.rows.get())
        except ValueError:
            return None
        return (cols, rows) if cols >= 2 and rows >= 2 else None

    def _add(self):
        exts = " ".join(f"*{event}" for event in sorted(IMAGE_EXTS))
        files = filedialog.askopenfilenames(title=t("board.add"), parent=self,
                                            filetypes=[(t("common.images"), exts)])
        if not files:
            return
        self._scan_note = None
        chosen = list(dict.fromkeys(self.paths + [str(Path(file)) for file in files]))
        self.paths = distortion.spread(chosen, distortion.MAX_PHOTOS)
        self._result = None
        self._refresh()
        if len(chosen) > distortion.MAX_PHOTOS:
            self.status.set(t("board.too_many", n=len(chosen), max=distortion.MAX_PHOTOS))
        else:
            self.status.set(t("board.n_photos", n=len(self.paths)))
        if self.paths:
            self.tree.selection_set(self.paths[0])

    def _remove(self):
        if self._busy:
            return
        sel = set(self.tree.selection())
        if sel:
            self.paths = [path for path in self.paths if path not in sel]
            self._result = None
            self._refresh()
            self.status.set(t("board.n_photos", n=len(self.paths)))

    def _refresh(self):
        """Lista con el estado de cada foto para el tablero elegido."""
        grid = self._grid()
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        errs = [detection.error_px for detection in (self._result.detections if self._result else []) if detection.error_px]
        high = max(2.5 * float(np.median(errs)), 0.5) if errs else None     # bajo 0.5 px todo está bien
        for path in self.paths:
            detection = self._dets.get((path, *grid)) if grid else None
            state, err, tag = "—", "", ()
            if detection is not None:
                state = "✓" if detection.found else "✗"
                tag = () if detection.found else ("bad",)
                if detection.error_px is not None and self._result is not None:
                    err = f"{detection.error_px:.2f} px"
                    if high and detection.error_px > high:
                        state, tag = "⚠", ("warn",)
            self.tree.insert("", "end", iid=path, text=Path(path).name, values=(state, err), tags=tag)
        keep = [item for item in sel if self.tree.exists(item)]
        if keep:
            self.tree.selection_set(keep)
        self._empty.place(relx=0.5, rely=0.45, anchor="center") if not self.paths else self._empty.place_forget()
        self._go.config(state="normal" if len(self.paths) >= 3 and not self._busy else "disabled")

    # ── miniatura ─────────────────────────────────────────────────────────────

    def _show(self, bgr: np.ndarray | None, caption: str):
        self.canvas.delete("all")
        self.caption.set(caption)
        if bgr is None:
            return
        picture = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        picture.thumbnail(PREVIEW)
        self._photo = ImageTk.PhotoImage(picture)
        self.canvas.create_image(PREVIEW[0] // 2, PREVIEW[1] // 2, image=self._photo)

    def _show_selected(self):
        sel = self.tree.selection()
        if not sel or self._busy:
            return
        path = sel[0]
        grid = self._grid()
        detection = self._dets.get((path, *grid)) if grid else None
        if detection is not None and detection.preview is not None:
            note = t("board.found") if detection.found else t("board.not_found")
            if detection.error_px is not None and self._result is not None:
                note += f", {detection.error_px:.2f} px"
            self._show(detection.preview, f"{Path(path).name} — {note}")
            return
        try:
            with Image.open(path) as file:
                file.draft("RGB", (PREVIEW[0] * 2, PREVIEW[1] * 2))
                file = ImageOps.exif_transpose(file).convert("RGB")
                file.thumbnail(PREVIEW)
                bgr = cv2.cvtColor(np.asarray(file), cv2.COLOR_RGB2BGR)
        except Exception:
            _log.debug("miniatura %s", path, exc_info=True)
            bgr = None
        self._show(bgr, Path(path).name)

    # ── calibrar ──────────────────────────────────────────────────────────────

    def _scan(self):
        """Buscar el tablero entre las fotos cargadas: primero una revisión rápida de todas
        (copia pequeña, sin afinar); luego, solo en las que lo tienen, la detección
        completa. Esas fotos se usan para calibrar y no se analizan."""
        grid = self._grid()
        if self._busy:
            return
        if grid is None:
            self.status.set(t("corr.distortion.bad_grid"))
            return
        paths = list(dict.fromkeys(self.project_paths))
        self._busy = True
        self._cancel.clear()
        self._go.config(state="disabled")
        self._cancel_btn.pack(side=tk.RIGHT, padx=(SPACE["md"], 0), before=self._status_lbl)
        self.bar.pack(side=tk.RIGHT, before=self._status_lbl)
        self.bar.config(maximum=len(paths), value=0)
        self.status.set(t("board.checking", i=0, n=len(paths)))

        def work():
            found = []
            workers = max(1, min((os.cpu_count() or 2) - 1, 6))
            with ThreadPoolExecutor(workers) as pool:
                futures = {pool.submit(distortion.has_board, path, *grid): path for path in paths}
                for done_count, fut in enumerate(futures, 1):
                    if self._cancel.is_set():
                        for future in futures:
                            future.cancel()
                        break
                    try:
                        if fut.result():
                            found.append(futures[fut])
                    except Exception:
                        _log.debug("revisión rápida", exc_info=True)
                    if done_count % 5 == 0 or done_count == len(paths):
                        self._ui(lambda done_count=done_count: (self.bar.config(value=done_count),
                                              self.status.set(t("board.checking", i=done_count, n=len(paths)))))
            self._ui(lambda: self._scan_done(found, len(paths)))
        threading.Thread(target=work, daemon=True).start()

    def _scan_done(self, found: list[str], total: int):
        self._busy = False
        if self._cancel.is_set():
            self._finish(t("board.cancelled"))
            return
        self.from_project = set(found)
        self.paths = distortion.spread(found, distortion.MAX_PHOTOS)
        self._scan_note = t("board.scan_found", found=len(found), n=total)
        self._refresh()
        if len(self.paths) < 3:
            self._finish(self._scan_note)
            return
        self._run()

    def _run(self, keep_found: bool = False):
        grid = self._grid()
        if grid is None:
            self.status.set(t("corr.distortion.bad_grid"))
            return
        self._busy = True
        self._cancel.clear()
        self._result = None
        self._refresh()
        self._go.config(state="disabled")
        self._cancel_btn.pack(side=tk.RIGHT, padx=(SPACE["md"], 0), before=self._status_lbl)
        self.bar.pack(side=tk.RIGHT, before=self._status_lbl)
        paths = list(self.paths)
        todo = [path for path in paths if (path, *grid) not in self._dets]
        total, done = len(paths), len(paths) - len(todo)
        self.bar.config(maximum=total, value=done)
        self.status.set(t("board.reading", i=done, n=total))

        def work():
            workers = max(1, min((os.cpu_count() or 2) - 1, 4))     # OpenCV suelta el GIL
            with ThreadPoolExecutor(workers) as pool:
                futures = [pool.submit(distortion.detect, path, *grid) for path in todo]
                for done_count, fut in enumerate(futures, done + 1):
                    if self._cancel.is_set():
                        for future in futures:
                            future.cancel()
                        break
                    try:
                        det = fut.result()
                    except Exception:
                        _log.exception("Tablero")
                        continue
                    self._dets[(det.path, *grid)] = det          # antes de calibrar (no esperar a la interfaz)
                    self._ui(lambda det=det, done_count=done_count: self._on_detection(det, grid, done_count, total))
            if self._cancel.is_set():
                self._ui(lambda: self._finish(t("board.cancelled")))
                return
            use = paths
            if keep_found:
                found = [path for path in paths if (path, *grid) in self._dets and self._dets[(path, *grid)].found]
                use = distortion.spread(found, distortion.MAX_PHOTOS)
                self._ui(lambda: self._keep(use, found, len(paths)))
            self._ui(lambda: self.status.set(t("board.solving")))
            try:
                res = distortion.solve([self._dets[(path, *grid)] for path in use if (path, *grid) in self._dets], *grid)
            except Exception as e:
                _log.exception("Calibración")
                msg = t("board.error", error=e)
                self._ui(lambda: self._finish(msg))
                return
            self._ui(lambda: self._solved(res))
        threading.Thread(target=work, daemon=True).start()

    def _keep(self, use: list[str], found: list[str], total: int):
        """Tras buscar entre las fotos cargadas: quedan las del tablero (hasta el máximo)."""
        self.paths = use
        self.from_project = set(found)
        self._refresh()
        self._scan_note = t("board.scan_found", found=len(found), n=total)
        self.status.set(self._scan_note)

    def _on_detection(self, det: distortion.Detection, grid, done_count: int, total: int):
        if not self.winfo_exists():
            return
        if self.tree.exists(det.path):
            self.tree.item(det.path, values=("✓" if det.found else "✗", ""), tags=() if det.found else ("bad",))
            self.tree.see(det.path)
        self._show(det.preview, f"{Path(det.path).name} — "
                                f"{t('board.found') if det.found else t('board.not_found')}")
        self.bar.config(value=done_count)
        self.status.set(t("board.reading", i=done_count, n=total))

    def _solved(self, res: distortion.CalibrationResult):
        if not self.winfo_exists():
            return
        self._result = res if res.success else None
        if res.success:
            quality = t("board.good") if res.rms_error < 1 else t("board.fair") if res.rms_error < 2 else t("board.poor")
            text = t("board.done", used=res.n_images_used, n=res.n_images_total, rms=f"{res.rms_error:.2f}") \
                + ". " + quality
        else:
            text = t("corr.distortion.failed", used=res.n_images_used, total=res.n_images_total)
        if getattr(self, "_scan_note", None):
            text = self._scan_note + "\n" + text
        self._finish(text)

    def _finish(self, text: str):
        self._busy = False
        self.bar.pack_forget()
        self._cancel_btn.pack_forget()
        self._refresh()
        self.status.set(text)
        if self._result and self.on_result:          # se usa en cuanto sale bien (Aplicar la guarda)
            self.on_result(self._result, sorted(self.from_project))

    def cancel(self):
        if self._busy:
            self._cancel.set()
            self.status.set(t("status.cancelling"))
