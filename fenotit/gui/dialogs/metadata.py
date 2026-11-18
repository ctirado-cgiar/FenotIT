"""Tabla de datos del usuario: archivo (CSV o Excel), columna que identifica la foto,
vista previa y qué fotos/filas no coinciden. Aplicar = `on_apply(file, key_column)`;
Quitar = `on_apply(None, None)`."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

# log escribe los mensajes y errores en el archivo de registro.
from fenotit import log
# metadata lee la tabla de datos del usuario, adivina la columna de la foto y une cada fila con su
# foto.
from fenotit.core import metadata
# HelpIcon es el «?» que explica una opción al pasar el mouse o al hacer clic.
from fenotit.gui.widgets.help import HelpIcon
# button crea un botón con el estilo del tema; divider dibuja una línea separadora.
from fenotit.gui.style.components import button, divider
# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras); SPACE (márgenes).
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
# t devuelve cada texto de la ventana en el idioma elegido (fenotit/lang).
from fenotit.i18n import t

_log = log.get("gui.metadata")

PREVIEW_ROWS = 8
WARN = COLORS["warning_text"]


def _short(items: list[str], limit: int = 4) -> str:
    text = ", ".join(items[:limit])
    return text + (f" … (+{len(items) - limit})" if len(items) > limit else "")


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
        body.pack(fill=tk.BOTH, expand=True, padx=SPACE["xxl"], pady=(SPACE["lg"], SPACE["md"]))
        head = tk.Frame(body, bg=bg)
        head.pack(fill=tk.X)
        tk.Label(head, text=t("meta.title"), bg=bg, fg=COLORS["accent"],
                 font=FONTS["heading_large"]).pack(side=tk.LEFT)
        HelpIcon(head, t("meta.title"), t("meta.help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["md"], 0))

        grid = tk.Frame(body, bg=bg)
        grid.pack(fill=tk.X, pady=(SPACE["lg"], 0))
        grid.columnconfigure(1, weight=1)
        tk.Label(grid, text=t("meta.file"), bg=bg, fg=COLORS["text"], font=FONTS["body"]).grid(
            row=0, column=0, sticky="w", padx=(0, SPACE["md"]))
        self.file = tk.StringVar(value=current.get("file", ""))
        tk.Entry(grid, textvariable=self.file, state="readonly", readonlybackground=COLORS["bg_panel"],
                 fg=COLORS["text"], relief="solid", bd=1, font=FONTS["small"]).grid(row=0, column=1, sticky="ew", ipady=2)
        button(grid, text="…", command=self._pick, width=3, kind="secondary").grid(row=0, column=2, padx=(SPACE["sm"], 0))
        tk.Label(grid, text=t("meta.key"), bg=bg, fg=COLORS["text"], font=FONTS["body"]).grid(
            row=1, column=0, sticky="w", padx=(0, SPACE["md"]), pady=(SPACE["md"], 0))
        krow = tk.Frame(grid, bg=bg)
        krow.grid(row=1, column=1, columnspan=2, sticky="w", pady=(SPACE["md"], 0))
        self.key = tk.StringVar(value=current.get("key_column", ""))
        self._key_cb = ttk.Combobox(krow, textvariable=self.key, state="readonly", width=24, font=FONTS["small"])
        self._key_cb.pack(side=tk.LEFT)
        self._key_cb.bind("<<ComboboxSelected>>", lambda event: self._check())
        HelpIcon(krow, t("meta.key"), t("meta.key_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["md"], 0))

        frame = tk.Frame(body, bg=bg)
        frame.pack(fill=tk.BOTH, expand=True, pady=(SPACE["lg"], 0))
        self.tree = ttk.Treeview(frame, show="headings", height=PREVIEW_ROWS, selectmode="none")
        xsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=xsb.set)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.summary = tk.Label(body, text="", bg=bg, fg=COLORS["text"], font=FONTS["body"], anchor="w")
        self.summary.pack(fill=tk.X, pady=(SPACE["lg"], 0))
        self.warn = tk.Label(body, text="", bg=bg, fg=WARN, font=FONTS["small"], anchor="w", justify="left",
                             wraplength=520)
        self.warn.pack(fill=tk.X)

        divider(self).pack(fill=tk.X)
        btns = tk.Frame(self, bg=bg)
        btns.pack(fill=tk.X, padx=SPACE["xxl"], pady=SPACE["lg"])
        self._apply_btn = button(btns, text=t("common.apply"), command=self._apply, kind="primary")
        self._apply_btn.pack(side=tk.RIGHT, padx=(SPACE["md"], 0))
        button(btns, text=t("common.cancel"), command=self.destroy, kind="secondary").pack(side=tk.RIGHT)
        if current.get("file"):
            button(btns, text=t("meta.remove"), command=self._remove, kind="quiet").pack(side=tk.LEFT)
        self.bind("<Escape>", lambda event: self.destroy())

        if current.get("file"):
            self._read(current["file"], current.get("key_column"))
        else:
            self._check()
        self.update_idletasks()
        pixel_x, pixel_y = parent.winfo_rootx(), parent.winfo_rooty()
        panel_width, panel_height = parent.winfo_width(), parent.winfo_height()
        width, height = max(self.winfo_reqwidth(), 560), self.winfo_reqheight()
        self.minsize(width, 0)
        self.geometry(f"+{pixel_x + (panel_width - width) // 2}+{pixel_y + max(0, (panel_height - height) // 3)}")      # el alto sigue al contenido

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
        for column in self.cols:
            self.tree.heading(column, text=column, anchor="w")
            self.tree.column(column, width=max(70, min(160, 9 * len(column) + 20)), stretch=False, anchor="w")
        for row in self.rows[:PREVIEW_ROWS]:
            self.tree.insert("", "end", values=["" if row.get(column) is None else row.get(column) for column in self.cols])

    def _check(self):
        if not self.rows or not self.key.get():
            self.summary.configure(text=t("meta.none") if not self.file.get() else t("meta.empty"),
                                   fg=COLORS["text_muted"])
            self.warn.configure(text="")
            self._apply_btn.configure(state="disabled")
            return
        match = metadata.match(self.names, self.rows, self.key.get())
        n_matched = len(match.by_image)
        self.summary.configure(text=t("meta.matched", n=n_matched, total=len(self.names), rows=len(self.rows)),
                               fg=COLORS["text"] if n_matched else WARN)
        lines = []
        if match.images_without_row:
            lines.append(t("meta.no_row", n=len(match.images_without_row), items=_short(match.images_without_row)))
        if match.rows_without_image:
            lines.append(t("meta.no_image", n=len(match.rows_without_image), items=_short(match.rows_without_image)))
        if match.duplicated_keys:
            lines.append(t("meta.duplicated", n=len(match.duplicated_keys), items=_short(match.duplicated_keys)))
        self.warn.configure(text="\n".join(lines))
        self._apply_btn.configure(state="normal" if n_matched else "disabled")

    def _apply(self):
        self.on_apply(self.file.get(), self.key.get())
        self.destroy()

    def _remove(self):
        self.on_apply(None, None)
        self.destroy()
