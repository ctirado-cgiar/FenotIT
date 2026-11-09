"""Exportar: carpeta, qué análisis y qué incluir. Siempre un CSV por tabla y los
metadatos; opcional, Excel e imágenes de las vistas. El trabajo lo hace la ventana
principal (`run`), que avisa el avance con `progress` y el final con `finished`."""
from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fenotit import log
from fenotit.gui.widgets.help import HelpIcon
from fenotit.gui.style.components import button, divider, section as title
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
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
        body.pack(fill=tk.BOTH, expand=True, padx=SPACE["xxl"], pady=(SPACE["lg"], SPACE["md"]))
        tk.Label(body, text=t("export.title"), bg=bg, fg=COLORS["accent"],
                 font=FONTS["heading_large"]).pack(anchor="w")

        def section(text, help_key=None):
            title(body, text, bg=bg, help_title=text.capitalize(),
                  help_text=t(help_key) if help_key else None).pack(fill=tk.X, pady=(SPACE["lg"], SPACE["sm"]))

        section(t("export.folder"))
        row = tk.Frame(body, bg=bg)
        row.pack(fill=tk.X)
        self.folder = tk.StringVar(value=folder)
        tk.Entry(row, textvariable=self.folder, width=46, bg=COLORS["bg_panel"], fg=COLORS["text"],
                 relief="solid", bd=1, font=FONTS["small"]).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=2)
        button(row, text="…", command=self._pick, width=3, kind="secondary").pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self.stamp = tk.BooleanVar(value=stamp)
        row = tk.Frame(body, bg=bg)
        row.pack(fill=tk.X, pady=(SPACE["xs"], 0))
        tk.Checkbutton(row, variable=self.stamp, text=t("export.stamp"), bg=bg, fg=COLORS["text"],
                       selectcolor=COLORS["bg_panel"], activebackground=bg, font=FONTS["small"]).pack(side=tk.LEFT)
        HelpIcon(row, t("export.stamp"), t("export.stamp_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))

        self.scope = tk.StringVar(value=(scopes or [("all", "")])[0][0])
        if scopes and len(scopes) > 1:
            section(t("export.photos"), "export.photos_help")
            row = tk.Frame(body, bg=bg)
            row.pack(fill=tk.X)
            for key, text in scopes:
                tk.Radiobutton(row, variable=self.scope, value=key, text=text, bg=bg, fg=COLORS["text"],
                               selectcolor=COLORS["bg_panel"], activebackground=bg,
                               font=FONTS["body"]).pack(side=tk.LEFT, padx=(0, SPACE["lg"]))

        section(t("export.analyses"))
        self.chosen: dict[str, tk.BooleanVar] = {}
        for name, label, n in analyses:
            var = self.chosen[name] = tk.BooleanVar(value=True)
            tk.Checkbutton(body, variable=var, text=f"{label} ({t('export.n_images_one' if n == 1 else 'export.n_images', n=n)})", bg=bg,
                           fg=COLORS["text"], selectcolor=COLORS["bg_panel"], activebackground=bg,
                           font=FONTS["body"], anchor="w").pack(fill=tk.X)

        section(t("export.include"), "export.always")
        self.excel = tk.BooleanVar(value=True)
        self.views = tk.BooleanVar(value=views_default)
        for var, text, help_key in ((self.excel, t("export.excel"), None),
                                    (self.views, t("export.views"), "export.views_note")):
            row = tk.Frame(body, bg=bg)
            row.pack(fill=tk.X)
            tk.Checkbutton(row, variable=var, text=text, bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_panel"],
                           activebackground=bg, font=FONTS["body"]).pack(side=tk.LEFT)
            if help_key:
                HelpIcon(row, text, t(help_key), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))

        self.msg = tk.StringVar(value="")
        self._bar = ttk.Progressbar(body, mode="determinate", length=420)
        self._msg_lbl = tk.Label(body, textvariable=self.msg, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                                 anchor="w", justify="left", wraplength=420)
        self._msg_lbl.pack(fill=tk.X, pady=(SPACE["lg"], 0))

        divider(self).pack(fill=tk.X)
        btns = tk.Frame(self, bg=bg)
        btns.pack(fill=tk.X, padx=SPACE["xxl"], pady=SPACE["lg"])
        self._close = button(btns, text=t("common.close"), command=self._on_close, kind="secondary")
        self._close.pack(side=tk.RIGHT, padx=(SPACE["md"], 0))
        self._go = button(btns, text=t("export.run"), command=self._start, kind="primary")
        self._go.pack(side=tk.RIGHT)
        self._open = button(btns, text=t("export.open_folder"), command=lambda: open_folder(getattr(self, "_done_folder", None) or self.folder.get()), bg=bg, kind="link")
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
            self._bar.pack(fill=tk.X, pady=(SPACE["lg"], 0), before=self._msg_lbl)
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
