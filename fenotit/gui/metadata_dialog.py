"""Tabla de datos del usuario: archivo (CSV o Excel), columna que identifica la foto,
vista previa y qué fotos/filas no coinciden. Aplicar = `on_apply(file, key_column)`;
Quitar = `on_apply(None, None)`."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from fenotit import log
from fenotit.core import metadata
from fenotit.gui.help import HelpIcon
from fenotit.gui.theme import COLORS, FONTS
from fenotit.i18n import t

_log = log.get("gui.metadata")

PREVIEW_ROWS = 8
WARN = "#B35806"


def _short(items: list[str], n: int = 4) -> str:
    s = ", ".join(items[:n])
    return s + (f" … (+{len(items) - n})" if len(items) > n else "")


class MetadataDialog(tk.Toplevel):

    def __init__(self, parent, image_names: list[str], current: dict, on_apply):
        super().__init__(parent)
        self.title(t("meta.title"))
        self.configure(bg=COLORS["bg_card"])
        self.transient(parent)
        self.minsize(520, 0)
        self.names, self.on_apply = image_names, on_apply
        self.cols: list[str] = []
        self.rows: list[dict] = []
        bg = COLORS["bg_card"]

        tk.Frame(self, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        body = tk.Frame(self, bg=bg)
        body.pack(fill=tk.BOTH, expand=True, padx=22, pady=(14, 8))
        head = tk.Frame(body, bg=bg)
        head.pack(fill=tk.X)
        tk.Label(head, text=t("meta.title"), bg=bg, fg=COLORS["accent"],
                 font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)
        HelpIcon(head, t("meta.title"), t("meta.help"), bg=bg).pack(side=tk.LEFT, padx=(6, 0))

        grid = tk.Frame(body, bg=bg)
        grid.pack(fill=tk.X, pady=(12, 0))
        grid.columnconfigure(1, weight=1)
        tk.Label(grid, text=t("meta.file"), bg=bg, fg=COLORS["text"], font=FONTS["body"]).grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        self.file = tk.StringVar(value=current.get("file", ""))
        tk.Entry(grid, textvariable=self.file, state="readonly", readonlybackground=COLORS["bg_panel"],
                 fg=COLORS["text"], relief="solid", bd=1, font=FONTS["small"]).grid(row=0, column=1, sticky="ew", ipady=2)
        tk.Button(grid, text="…", command=self._pick, bg=COLORS["btn_bg"], fg=COLORS["accent"], relief="flat",
                  cursor="hand2", font=FONTS["body"], width=3).grid(row=0, column=2, padx=(4, 0))
        tk.Label(grid, text=t("meta.key"), bg=bg, fg=COLORS["text"], font=FONTS["body"]).grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        krow = tk.Frame(grid, bg=bg)
        krow.grid(row=1, column=1, columnspan=2, sticky="w", pady=(8, 0))
        self.key = tk.StringVar(value=current.get("key_column", ""))
        self._key_cb = ttk.Combobox(krow, textvariable=self.key, state="readonly", width=24, font=FONTS["small"])
        self._key_cb.pack(side=tk.LEFT)
        self._key_cb.bind("<<ComboboxSelected>>", lambda e: self._check())
        HelpIcon(krow, t("meta.key"), t("meta.key_help"), bg=bg).pack(side=tk.LEFT, padx=(6, 0))

        pf = tk.Frame(body, bg=bg)
        pf.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.tree = ttk.Treeview(pf, show="headings", height=PREVIEW_ROWS, selectmode="none")
        xsb = ttk.Scrollbar(pf, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=xsb.set)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.summary = tk.Label(body, text="", bg=bg, fg=COLORS["text"], font=FONTS["body"], anchor="w")
        self.summary.pack(fill=tk.X, pady=(10, 0))
        self.warn = tk.Label(body, text="", bg=bg, fg=WARN, font=FONTS["small"], anchor="w", justify="left",
                             wraplength=520)
        self.warn.pack(fill=tk.X)

        tk.Frame(self, bg=COLORS["border"], height=1).pack(fill=tk.X)
        btns = tk.Frame(self, bg=bg)
        btns.pack(fill=tk.X, padx=22, pady=10)
        self._apply_btn = tk.Button(btns, text=t("common.apply"), command=self._apply, bg=COLORS["accent"],
                                    fg="#FFFFFF", relief="flat", font=FONTS["body"], cursor="hand2", padx=12)
        self._apply_btn.pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(btns, text=t("common.cancel"), command=self.destroy, bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"], cursor="hand2", padx=12).pack(side=tk.RIGHT)
        if current.get("file"):
            tk.Button(btns, text=t("meta.remove"), command=self._remove, bg=COLORS["btn_bg"], fg=COLORS["text"],
                      relief="flat", font=FONTS["body"], cursor="hand2", padx=12).pack(side=tk.LEFT)
        self.bind("<Escape>", lambda e: self.destroy())

        if current.get("file"):
            self._read(current["file"], current.get("key_column"))
        else:
            self._check()
        self.update_idletasks()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        w, h = max(self.winfo_reqwidth(), 560), self.winfo_reqheight()
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + max(0, (ph - h) // 3)}")
        if not current.get("file"):
            self.after(100, self._pick)

    def _pick(self):
        start = Path(self.file.get()).parent if self.file.get() else None
        path = filedialog.askopenfilename(parent=self, title=t("meta.file"), initialdir=start or None,
                                          filetypes=[(t("meta.filetypes"), "*.csv *.txt *.xlsx *.xlsm"),
                                                     ("CSV", "*.csv *.txt"), ("Excel", "*.xlsx *.xlsm")])
        if path:
            self._read(path, None)

    def _read(self, path: str, key_column: str | None):
        try:
            self.cols, self.rows = metadata.read_table(path)
        except Exception as e:
            _log.warning("Tabla de datos %s: %s", path, e)
            self.cols, self.rows = [], []
            self.file.set(path)
            self._fill_preview()
            self._key_cb.configure(values=[])
            self.key.set("")
            self.summary.configure(text=t("meta.read_error", error=e), fg=WARN)
            self.warn.configure(text="")
            self._apply_btn.configure(state="disabled")
            return
        self.file.set(path)
        self._key_cb.configure(values=self.cols)
        if key_column not in self.cols:
            key_column = metadata.guess_key_column(self.cols, self.rows, self.names) or (self.cols[0] if self.cols else "")
        self.key.set(key_column)
        self._fill_preview()
        self._check()

    def _fill_preview(self):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = self.cols
        for c in self.cols:
            self.tree.heading(c, text=c, anchor="w")
            self.tree.column(c, width=max(70, min(160, 9 * len(c) + 20)), stretch=False, anchor="w")
        for r in self.rows[:PREVIEW_ROWS]:
            self.tree.insert("", "end", values=["" if r.get(c) is None else r.get(c) for c in self.cols])

    def _check(self):
        if not self.rows or not self.key.get():
            self.summary.configure(text=t("meta.none") if not self.file.get() else t("meta.empty"),
                                   fg=COLORS["text_muted"])
            self.warn.configure(text="")
            self._apply_btn.configure(state="disabled")
            return
        m = metadata.match(self.names, self.rows, self.key.get())
        n = len(m.by_image)
        self.summary.configure(text=t("meta.matched", n=n, total=len(self.names), rows=len(self.rows)),
                               fg=COLORS["text"] if n else WARN)
        lines = []
        if m.images_without_row:
            lines.append(t("meta.no_row", n=len(m.images_without_row), items=_short(m.images_without_row)))
        if m.rows_without_image:
            lines.append(t("meta.no_image", n=len(m.rows_without_image), items=_short(m.rows_without_image)))
        if m.duplicated_keys:
            lines.append(t("meta.duplicated", n=len(m.duplicated_keys), items=_short(m.duplicated_keys)))
        self.warn.configure(text="\n".join(lines))
        self._apply_btn.configure(state="normal" if n else "disabled")

    def _apply(self):
        self.on_apply(self.file.get(), self.key.get())
        self.destroy()

    def _remove(self):
        self.on_apply(None, None)
        self.destroy()
