"""Exportar: carpeta, qué análisis y qué incluir. Siempre un CSV por tabla y los
metadatos; opcional, Excel e imágenes de las vistas. El trabajo lo hace la ventana
principal (`run`), que avisa el avance con `progress` y el final con `finished`."""
from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from fenotit import log
from fenotit.gui.theme import COLORS, FONTS
from fenotit.i18n import t

_log = log.get("gui.export")


def open_folder(path: str):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)                                    # noqa: S606
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
    except Exception:
        _log.debug("abrir carpeta", exc_info=True)


class ExportDialog(tk.Toplevel):

    def __init__(self, parent, analyses: list[tuple[str, str, int]], folder: str, run, cancel,
                 views_default: bool = False, scopes: list[tuple[str, str]] | None = None,
                 stamp: bool = True, existing=None):
        """analyses: (nombre, etiqueta, nº de fotos con resultado); scopes: (clave, texto) de
        qué fotos exportar (la primera = todas). run(folder, nombres, excel, views, dialog,
        scope, stamp); cancel(). existing(folder, nombres) = lo que se reemplazaría."""
        super().__init__(parent)
        self.title(t("export.title"))
        self.configure(bg=COLORS["bg_card"])
        self.resizable(False, False)
        self.transient(parent)
        self._run, self._cancel, self._existing = run, cancel, existing
        self._busy = False
        bg = COLORS["bg_card"]

        tk.Frame(self, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        body = tk.Frame(self, bg=bg)
        body.pack(fill=tk.BOTH, expand=True, padx=22, pady=(14, 8))
        tk.Label(body, text=t("export.title"), bg=bg, fg=COLORS["accent"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")

        def section(text):
            row = tk.Frame(body, bg=bg)
            row.pack(fill=tk.X, pady=(14, 4))
            tk.Label(row, text=text, bg=bg, fg=COLORS["accent"], font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
            tk.Frame(row, bg=COLORS["border"], height=1).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                                             padx=(8, 0), pady=(2, 0))

        section(t("export.folder"))
        row = tk.Frame(body, bg=bg)
        row.pack(fill=tk.X)
        self.folder = tk.StringVar(value=folder)
        tk.Entry(row, textvariable=self.folder, width=46, bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1, font=FONTS["small"]).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=2)
        tk.Button(row, text="…", command=self._pick, bg=COLORS["btn_bg"], fg=COLORS["accent"], relief="flat",
                  cursor="hand2", font=FONTS["body"], width=3).pack(side=tk.LEFT, padx=(4, 0))
        self.stamp = tk.BooleanVar(value=stamp)
        tk.Checkbutton(body, variable=self.stamp, text=t("export.stamp"), bg=bg, fg=COLORS["text"],
                       selectcolor=COLORS["bg_panel"], activebackground=bg, font=FONTS["small"],
                       anchor="w").pack(fill=tk.X, pady=(2, 0))

        self.scope = tk.StringVar(value=(scopes or [("all", "")])[0][0])
        if scopes and len(scopes) > 1:
            section(t("export.photos"))
            for key, text in scopes:
                tk.Radiobutton(body, variable=self.scope, value=key, text=text, bg=bg, fg=COLORS["text"],
                               selectcolor=COLORS["bg_panel"], activebackground=bg, font=FONTS["body"],
                               anchor="w").pack(fill=tk.X)

        section(t("export.analyses"))
        self.chosen: dict[str, tk.BooleanVar] = {}
        for name, label, n in analyses:
            var = self.chosen[name] = tk.BooleanVar(value=True)
            tk.Checkbutton(body, variable=var, text=f"{label}  ·  {t('export.n_images', n=n)}", bg=bg,
                           fg=COLORS["text"], selectcolor=COLORS["bg_panel"], activebackground=bg,
                           font=FONTS["body"], anchor="w").pack(fill=tk.X)

        section(t("export.include"))
        tk.Label(body, text=t("export.always"), bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                 anchor="w", justify="left", wraplength=420).pack(fill=tk.X, pady=(0, 2))
        self.excel = tk.BooleanVar(value=True)
        self.views = tk.BooleanVar(value=views_default)
        for var, text in ((self.excel, t("export.excel")), (self.views, t("export.views"))):
            tk.Checkbutton(body, variable=var, text=text, bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_panel"],
                           activebackground=bg, font=FONTS["body"], anchor="w").pack(fill=tk.X)
        tk.Label(body, text=t("export.views_note"), bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                 anchor="w", justify="left", wraplength=420).pack(fill=tk.X, padx=(22, 0))

        self.msg = tk.StringVar(value="")
        self._bar = ttk.Progressbar(body, mode="determinate", length=420)
        self._msg_lbl = tk.Label(body, textvariable=self.msg, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                                 anchor="w", justify="left", wraplength=420)
        self._msg_lbl.pack(fill=tk.X, pady=(14, 0))

        tk.Frame(self, bg=COLORS["border"], height=1).pack(fill=tk.X)
        btns = tk.Frame(self, bg=bg)
        btns.pack(fill=tk.X, padx=22, pady=10)
        self._close = tk.Button(btns, text=t("common.close"), command=self._on_close, bg=COLORS["btn_bg"],
                                fg=COLORS["accent"], relief="flat", font=FONTS["body"], cursor="hand2", padx=12)
        self._close.pack(side=tk.RIGHT, padx=(6, 0))
        self._go = tk.Button(btns, text=t("export.run"), command=self._start, bg=COLORS["accent"], fg="#FFFFFF",
                             activebackground="#1A5390", activeforeground="#FFFFFF", relief="flat",
                             font=("Segoe UI", 9, "bold"), cursor="hand2", padx=14)
        self._go.pack(side=tk.RIGHT)
        self._open = tk.Button(btns, text=t("export.open_folder"), bg=bg, fg=COLORS["accent"], relief="flat",
                               font=FONTS["body"], cursor="hand2",
                               command=lambda: open_folder(getattr(self, "_done_folder", None) or self.folder.get()))
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda e: self._on_close())

        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        self.geometry(f"+{px + (pw - w) // 2}+{py + max(0, (ph - h) // 3)}")
        self.grab_set()

    def _pick(self):
        d = filedialog.askdirectory(title=t("export.folder"), parent=self, initialdir=self.folder.get() or None)
        if d:
            self.folder.set(d)

    def _start(self):
        folder = self.folder.get().strip()
        names = [n for n, v in self.chosen.items() if v.get()]
        if not folder:
            self.msg.set(t("export.no_folder"))
            return
        if not names:
            self.msg.set(t("export.no_analysis"))
            return
        if not self.stamp.get() and self._existing:
            found = self._existing(folder, names)
            if found and not messagebox.askyesno(t("export.title"), t("export.overwrite", items=found), parent=self):
                return
        self._busy = True
        self._go.config(state="disabled")
        self._close.config(text=t("common.cancel"))
        self._open.pack_forget()
        self.msg.set(t("export.writing"))
        self._run(folder, names, self.excel.get(), self.views.get(), self, self.scope.get(), self.stamp.get())

    # llamados por la ventana principal (hilo principal)
    def progress(self, i: int, n: int, text: str):
        if not self.winfo_exists():
            return
        if not self._bar.winfo_ismapped():
            self._bar.pack(fill=tk.X, pady=(10, 0), before=self._msg_lbl)
        self._bar.config(maximum=max(n, 1), value=i)
        self.msg.set(text)

    def finished(self, text: str, ok: bool = True, folder: str | None = None):
        if not self.winfo_exists():
            return
        self._done_folder = folder
        self._busy = False
        self._bar.pack_forget()
        self.msg.set(text)
        self._go.config(state="normal")
        self._close.config(text=t("common.close"))
        if ok:
            self._open.pack(side=tk.LEFT)

    def _on_close(self):
        if self._busy:
            self._cancel()
            return
        self.destroy()
