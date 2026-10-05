"""Controles reutilizables."""
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from fenotit.core.project import IMAGE_EXTS
from fenotit.gui.theme import COLORS, FONTS
from fenotit.i18n import t


class ImagePicker(tk.Frame):
    """Combo compacto: imágenes cargadas + «Otra foto…». Llama on_change(path)."""

    def __init__(self, parent, paths: list[str], current: str | None, on_change, width: int = 28):
        super().__init__(parent, bg=COLORS["bg_card"])
        self._paths = list(paths)
        self._on_change = on_change
        self._other = t("picker.other")
        tk.Label(self, text=t("picker.label"), bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"]).pack(side=tk.LEFT)
        self._var = tk.StringVar()
        self._combo = ttk.Combobox(self, textvariable=self._var, state="readonly", width=width)
        self._combo.pack(side=tk.LEFT, padx=6)
        self._combo.bind("<<ComboboxSelected>>", self._selected)
        self._refresh(current)

    def _label(self, i: int, p: str) -> str:
        return f"{i + 1}. {Path(p).name}"

    def _refresh(self, current: str | None):
        self._combo["values"] = [self._label(i, p) for i, p in enumerate(self._paths)] + [self._other]
        if current in self._paths:
            self._combo.current(self._paths.index(current))
        else:
            self._var.set("")
        self._last = self._var.get()

    def _selected(self, _e=None):
        idx = self._combo.current()
        if idx == len(self._paths):
            exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
            path = filedialog.askopenfilename(
                parent=self, filetypes=[(t("common.images"), exts), (t("common.all_files"), "*.*")])
            if not path:
                self._var.set(self._last)
                return
            if path not in self._paths:
                self._paths.append(path)
            self._refresh(path)
        else:
            path = self._paths[idx]
            self._last = self._var.get()
        self._on_change(path)
