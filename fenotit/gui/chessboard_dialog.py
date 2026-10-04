"""Calibración del lente con fotos de un tablero de ajedrez: lista de fotos con su
estado (tablero encontrado o no, error de cada una), miniatura de la foto que se está
leyendo o de la elegida, avance y cancelar. Usa como máximo MAX_PHOTOS fotos."""
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
PREVIEW = (360, 270)


class ChessboardDialog(tk.Toplevel):

    def __init__(self, parent, cols: int = 7, rows: int = 6, on_use=None):
        super().__init__(parent)
        self.title(t("board.title"))
        self.configure(bg=COLORS["bg_card"])
        self.transient(parent)
        self.on_use = on_use
        self.paths: list[str] = []
        self._dets: dict[tuple, D.Detection] = {}     # (foto, columnas, filas) -> detección
        self._result: D.CalibrationResult | None = None
        self._cancel = threading.Event()
        self._busy = False
        self._photo = None
        bg = COLORS["bg_card"]

        tk.Frame(self, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        top = tk.Frame(self, bg=bg)
        top.pack(fill=tk.X, padx=20, pady=(12, 0))
        tk.Label(top, text=t("board.title"), bg=bg, fg=COLORS["accent"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(top, text=t("board.help", n=D.MAX_PHOTOS), bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                 justify="left", wraplength=700).pack(anchor="w", pady=(2, 8))

        row = tk.Frame(top, bg=bg)
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
        self._btn(row, t("board.remove"), self._remove).pack(side=tk.RIGHT)
        self._btn(row, t("board.add"), self._add).pack(side=tk.RIGHT, padx=(0, 6))

        mid = tk.Frame(self, bg=bg)
        mid.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        left = tk.Frame(mid, bg=bg)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(left, orient="vertical")
        self.tree = ttk.Treeview(left, columns=("state", "error"), show="tree headings", height=12,
                                 yscrollcommand=sb.set, selectmode="extended")
        sb.config(command=self.tree.yview)
        self.tree.heading("#0", text=t("board.col_photo"), anchor="w")
        self.tree.heading("state", text=t("board.col_board"), anchor="center")
        self.tree.heading("error", text=t("board.col_error"), anchor="e")
        self.tree.column("#0", width=210, stretch=True)
        self.tree.column("state", width=70, anchor="center", stretch=False)
        self.tree.column("error", width=80, anchor="e", stretch=False)
        self.tree.tag_configure("bad", foreground="#C0392B")
        self.tree.tag_configure("warn", foreground="#E67E00")
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._show_selected())
        self.tree.bind("<Delete>", lambda e: self._remove())
        self._empty = tk.Label(left, text=t("board.empty"), bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                               font=FONTS["small"])

        right = tk.Frame(mid, bg=bg)
        right.pack(side=tk.LEFT, padx=(14, 0), anchor="n")
        self.canvas = tk.Canvas(right, width=PREVIEW[0], height=PREVIEW[1], bg=COLORS["bg_panel"],
                                highlightthickness=1, highlightbackground=COLORS["border"])
        self.canvas.pack()
        self.caption = tk.StringVar(value="")
        tk.Label(right, textvariable=self.caption, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                 anchor="w").pack(fill=tk.X, pady=(4, 0))

        bottom = tk.Frame(self, bg=bg)
        bottom.pack(fill=tk.X, padx=20)
        self.bar = ttk.Progressbar(bottom, mode="determinate", length=220)
        self.status = tk.StringVar(value="")
        self._status_lbl = tk.Label(bottom, textvariable=self.status, bg=bg, fg=COLORS["text"], font=FONTS["body"],
                                    anchor="w", justify="left", wraplength=700)
        self._status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=(0, 6))

        tk.Frame(self, bg=COLORS["border"], height=1).pack(fill=tk.X)
        btns = tk.Frame(self, bg=bg)
        btns.pack(fill=tk.X, padx=20, pady=10)
        self._close = self._btn(btns, t("common.close"), self._on_close)
        self._close.pack(side=tk.RIGHT, padx=(6, 0))
        self._use = tk.Button(btns, text=t("board.use"), command=self._use_result, bg=COLORS["accent"],
                              fg="#FFFFFF", activebackground="#1A5390", activeforeground="#FFFFFF", relief="flat",
                              font=("Segoe UI", 9, "bold"), cursor="hand2", padx=14, state="disabled")
        self._use.pack(side=tk.RIGHT)
        self._go = self._btn(btns, t("board.run"), self._run)
        self._go.pack(side=tk.RIGHT, padx=(0, 6))

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda e: self._on_close())
        self._refresh()
        self.update_idletasks()
        w, h = max(760, self.winfo_reqwidth()), self.winfo_reqheight()
        self.geometry(f"{w}x{h}+{parent.winfo_rootx() + max(0, (parent.winfo_width() - w) // 2)}"
                      f"+{parent.winfo_rooty() + max(0, (parent.winfo_height() - h) // 3)}")
        self.grab_set()

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
        self._use.config(state="normal" if self._result and self._result.success and not self._busy else "disabled")

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

    def _run(self):
        grid = self._grid()
        if grid is None:
            self.status.set(t("corr.distortion.bad_grid"))
            return
        self._busy = True
        self._cancel.clear()
        self._result = None
        self._refresh()
        self._go.config(state="disabled")
        self._close.config(text=t("common.cancel"))
        self.bar.pack(side=tk.RIGHT, pady=(0, 6), before=self._status_lbl)
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
                    self.after(0, lambda det=det, k=k: self._on_detection(det, grid, k, n))
            if self._cancel.is_set():
                self.after(0, lambda: self._finish(t("board.cancelled")))
                return
            self.after(0, lambda: self.status.set(t("board.solving")))
            try:
                res = D.solve([self._dets[(p, *grid)] for p in paths if (p, *grid) in self._dets], *grid)
            except Exception as e:
                _log.exception("Calibración")
                msg = t("board.error", error=e)
                self.after(0, lambda: self._finish(msg))
                return
            self.after(0, lambda: self._solved(res))
        threading.Thread(target=work, daemon=True).start()

    def _on_detection(self, det: D.Detection, grid, k: int, n: int):
        if not self.winfo_exists():
            return
        self._dets[(det.path, *grid)] = det
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
        self._finish(text)

    def _finish(self, text: str):
        self._busy = False
        self.bar.pack_forget()
        self._close.config(text=t("common.close"))
        self._refresh()
        self.status.set(text)

    def _use_result(self):
        if self._result and self.on_use:
            self.on_use(self._result)
        self.destroy()

    def _on_close(self):
        if self._busy:
            self._cancel.set()
            self.status.set(t("status.cancelling"))
            return
        self.destroy()
