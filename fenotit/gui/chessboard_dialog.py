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
from fenotit.core.corrections import distortion as D
from fenotit.core.project import IMAGE_EXTS
from fenotit.gui.theme import COLORS, FONTS
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
        self._dets: dict[tuple, D.Detection] = {}     # (foto, columnas, filas) -> detección
        self._result: D.CalibrationResult | None = None
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
                     bd=1, font=FONTS["mono"], justify="center").pack(side=tk.LEFT, padx=4)
        tk.Label(row, text=t("board.corners_hint"), bg=bg, fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(4, 0))

        btns = tk.Frame(self, bg=bg)
        btns.pack(fill=tk.X, pady=(6, 0))
        self._btn(btns, t("board.add"), self._add).pack(side=tk.LEFT)
        if self.project_paths:
            self._btn(btns, t("board.scan", n=len(self.project_paths)), self._scan).pack(side=tk.LEFT, padx=(6, 0))
        self._btn(btns, t("board.remove"), self._remove).pack(side=tk.LEFT, padx=(6, 0))
        self._go = tk.Button(btns, text=t("board.run"), command=self._run, bg=COLORS["accent"], fg="#FFFFFF",
                             activebackground="#1A5390", activeforeground="#FFFFFF", relief="flat",
                             font=("Segoe UI", 9, "bold"), cursor="hand2", padx=12)
        self._go.pack(side=tk.LEFT, padx=(6, 0))
        if extra:
            text, cmd = extra
            self._btn(btns, text, cmd).pack(side=tk.RIGHT)

        mid = tk.Frame(self, bg=bg)
        mid.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        left = tk.Frame(mid, bg=bg)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(left, orient="vertical")
        self.tree = ttk.Treeview(left, columns=("state", "error"), show="tree headings", height=8,
                                 yscrollcommand=sb.set, selectmode="extended")
        sb.config(command=self.tree.yview)
        self.tree.heading("#0", text=t("board.col_photo"), anchor="w")
        self.tree.heading("state", text=t("board.col_board"), anchor="center")
        self.tree.heading("error", text=t("board.col_error"), anchor="e")
        self.tree.column("#0", width=170, stretch=True)
        self.tree.column("state", width=60, anchor="center", stretch=False)
        self.tree.column("error", width=70, anchor="e", stretch=False)
        self.tree.tag_configure("bad", foreground="#C0392B")
        self.tree.tag_configure("warn", foreground="#E67E00")
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._show_selected())
        self.tree.bind("<Delete>", lambda e: self._remove())
        self._empty = tk.Label(left, text=t("board.empty"), bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                               font=FONTS["small"])

        right = tk.Frame(mid, bg=bg)
        right.pack(side=tk.LEFT, padx=(10, 0), anchor="n")
        self.canvas = tk.Canvas(right, width=PREVIEW[0], height=PREVIEW[1], bg=COLORS["bg_panel"],
                                highlightthickness=1, highlightbackground=COLORS["border"])
        self.canvas.pack()
        self.caption = tk.StringVar(value="")
        tk.Label(right, textvariable=self.caption, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                 anchor="w").pack(fill=tk.X, pady=(2, 0))

        bottom = tk.Frame(self, bg=bg)
        bottom.pack(fill=tk.X, pady=(6, 0))
        self.bar = ttk.Progressbar(bottom, mode="determinate", length=160)
        self._cancel_btn = self._btn(bottom, t("common.cancel"), self.cancel)
        self.status = tk.StringVar(value=status)
        self._status_lbl = tk.Label(bottom, textvariable=self.status, bg=bg, fg=COLORS["text"],
                                    font=FONTS["small"], anchor="w", justify="left", wraplength=620)
        self._status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.bind("<Destroy>", lambda e: self._cancel.set() if e.widget is self else None)
        self._refresh()

    def _ui(self, fn):
        """Desde el hilo de trabajo: a la interfaz, si la ventana sigue abierta."""
        try:
            self.after(0, fn)
        except (tk.TclError, RuntimeError):
            self._cancel.set()

    def _btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd, bg=COLORS["btn_bg"], fg=COLORS["accent"], relief="flat",
                         font=FONTS["body"], cursor="hand2", padx=10)

    # ── fotos ─────────────────────────────────────────────────────────────────

    def _grid(self) -> tuple[int, int] | None:
        try:
            c, r = int(self.cols.get()), int(self.rows.get())
        except ValueError:
            return None
        return (c, r) if c >= 2 and r >= 2 else None

    def _add(self):
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        files = filedialog.askopenfilenames(title=t("board.add"), parent=self,
                                            filetypes=[(t("common.images"), exts)])
        if not files:
            return
        self._scan_note = None
        chosen = list(dict.fromkeys(self.paths + [str(Path(f)) for f in files]))
        self.paths = D.spread(chosen, D.MAX_PHOTOS)
        self._result = None
        self._refresh()
        if len(chosen) > D.MAX_PHOTOS:
            self.status.set(t("board.too_many", n=len(chosen), max=D.MAX_PHOTOS))
        else:
            self.status.set(t("board.n_photos", n=len(self.paths)))
        if self.paths:
            self.tree.selection_set(self.paths[0])

    def _remove(self):
        if self._busy:
            return
        sel = set(self.tree.selection())
        if sel:
            self.paths = [p for p in self.paths if p not in sel]
            self._result = None
            self._refresh()
            self.status.set(t("board.n_photos", n=len(self.paths)))

    def _refresh(self):
        """Lista con el estado de cada foto para el tablero elegido."""
        grid = self._grid()
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        errs = [d.error_px for d in (self._result.detections if self._result else []) if d.error_px]
        high = max(2.5 * float(np.median(errs)), 0.5) if errs else None     # bajo 0.5 px todo está bien
        for p in self.paths:
            d = self._dets.get((p, *grid)) if grid else None
            state, err, tag = "—", "", ()
            if d is not None:
                state = "✓" if d.found else "✗"
                tag = () if d.found else ("bad",)
                if d.error_px is not None and self._result is not None:
                    err = f"{d.error_px:.2f} px"
                    if high and d.error_px > high:
                        state, tag = "⚠", ("warn",)
            self.tree.insert("", "end", iid=p, text=Path(p).name, values=(state, err), tags=tag)
        keep = [s for s in sel if self.tree.exists(s)]
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
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        im.thumbnail(PREVIEW)
        self._photo = ImageTk.PhotoImage(im)
        self.canvas.create_image(PREVIEW[0] // 2, PREVIEW[1] // 2, image=self._photo)

    def _show_selected(self):
        sel = self.tree.selection()
        if not sel or self._busy:
            return
        p = sel[0]
        grid = self._grid()
        d = self._dets.get((p, *grid)) if grid else None
        if d is not None and d.preview is not None:
            note = t("board.found") if d.found else t("board.not_found")
            if d.error_px is not None and self._result is not None:
                note += f" · {d.error_px:.2f} px"
            self._show(d.preview, f"{Path(p).name} — {note}")
            return
        try:
            with Image.open(p) as im:
                im.draft("RGB", (PREVIEW[0] * 2, PREVIEW[1] * 2))
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail(PREVIEW)
                bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        except Exception:
            _log.debug("miniatura %s", p, exc_info=True)
            bgr = None
        self._show(bgr, Path(p).name)

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
        self._cancel_btn.pack(side=tk.RIGHT, padx=(6, 0), before=self._status_lbl)
        self.bar.pack(side=tk.RIGHT, before=self._status_lbl)
        self.bar.config(maximum=len(paths), value=0)
        self.status.set(t("board.checking", i=0, n=len(paths)))

        def work():
            found = []
            workers = max(1, min((os.cpu_count() or 2) - 1, 6))
            with ThreadPoolExecutor(workers) as pool:
                futures = {pool.submit(D.has_board, p, *grid): p for p in paths}
                for k, fut in enumerate(futures, 1):
                    if self._cancel.is_set():
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        if fut.result():
                            found.append(futures[fut])
                    except Exception:
                        _log.debug("revisión rápida", exc_info=True)
                    if k % 5 == 0 or k == len(paths):
                        self._ui(lambda k=k: (self.bar.config(value=k),
                                              self.status.set(t("board.checking", i=k, n=len(paths)))))
            self._ui(lambda: self._scan_done(found, len(paths)))
        threading.Thread(target=work, daemon=True).start()

    def _scan_done(self, found: list[str], n: int):
        self._busy = False
        if self._cancel.is_set():
            self._finish(t("board.cancelled"))
            return
        self.from_project = set(found)
        self.paths = D.spread(found, D.MAX_PHOTOS)
        self._scan_note = t("board.scan_found", found=len(found), n=n)
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
        self._cancel_btn.pack(side=tk.RIGHT, padx=(6, 0), before=self._status_lbl)
        self.bar.pack(side=tk.RIGHT, before=self._status_lbl)
        paths = list(self.paths)
        todo = [p for p in paths if (p, *grid) not in self._dets]
        n, done = len(paths), len(paths) - len(todo)
        self.bar.config(maximum=n, value=done)
        self.status.set(t("board.reading", i=done, n=n))

        def work():
            workers = max(1, min((os.cpu_count() or 2) - 1, 4))     # OpenCV suelta el GIL
            with ThreadPoolExecutor(workers) as pool:
                futures = [pool.submit(D.detect, p, *grid) for p in todo]
                for k, fut in enumerate(futures, done + 1):
                    if self._cancel.is_set():
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        det = fut.result()
                    except Exception:
                        _log.exception("Tablero")
                        continue
                    self._dets[(det.path, *grid)] = det          # antes de calibrar (no esperar a la interfaz)
                    self._ui(lambda det=det, k=k: self._on_detection(det, grid, k, n))
            if self._cancel.is_set():
                self._ui(lambda: self._finish(t("board.cancelled")))
                return
            use = paths
            if keep_found:
                found = [p for p in paths if (p, *grid) in self._dets and self._dets[(p, *grid)].found]
                use = D.spread(found, D.MAX_PHOTOS)
                self._ui(lambda: self._keep(use, found, len(paths)))
            self._ui(lambda: self.status.set(t("board.solving")))
            try:
                res = D.solve([self._dets[(p, *grid)] for p in use if (p, *grid) in self._dets], *grid)
            except Exception as e:
                _log.exception("Calibración")
                msg = t("board.error", error=e)
                self._ui(lambda: self._finish(msg))
                return
            self._ui(lambda: self._solved(res))
        threading.Thread(target=work, daemon=True).start()

    def _keep(self, use: list[str], found: list[str], n: int):
        """Tras buscar entre las fotos cargadas: quedan las del tablero (hasta el máximo)."""
        self.paths = use
        self.from_project = set(found)
        self._refresh()
        self._scan_note = t("board.scan_found", found=len(found), n=n)
        self.status.set(self._scan_note)

    def _on_detection(self, det: D.Detection, grid, k: int, n: int):
        if not self.winfo_exists():
            return
        if self.tree.exists(det.path):
            self.tree.item(det.path, values=("✓" if det.found else "✗", ""), tags=() if det.found else ("bad",))
            self.tree.see(det.path)
        self._show(det.preview, f"{Path(det.path).name} — "
                                f"{t('board.found') if det.found else t('board.not_found')}")
        self.bar.config(value=k)
        self.status.set(t("board.reading", i=k, n=n))

    def _solved(self, res: D.CalibrationResult):
        if not self.winfo_exists():
            return
        self._result = res if res.success else None
        if res.success:
            quality = t("board.good") if res.rms_error < 1 else t("board.fair") if res.rms_error < 2 else t("board.poor")
            text = t("board.done", used=res.n_images_used, n=res.n_images_total, rms=f"{res.rms_error:.2f}") \
                + "  ·  " + quality
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
