"""Guardar preset: nombre, descripción y qué partes de la configuración lleva."""
import tkinter as tk

from fenotit.core import preset
from fenotit.gui.dialogs.scale import BaseDialog
from fenotit.gui.style.components import button
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.gui.widgets.help import HelpIcon
from fenotit.i18n import t


class PresetDialog(BaseDialog):
    """`on_save(nombre, descripción, partes)` recibe lo elegido al pulsar Guardar.
    `available`: partes que el proyecto tiene configuradas; `checked`: las que empiezan marcadas;
    `cautions`: parte → aviso que se muestra mientras esa parte esté marcada."""
    W, H = 440, 410

    def __init__(self, parent, name: str, available: tuple, checked: tuple, cautions: dict, on_save):
        self._on_save = on_save
        self._cautions = cautions
        super().__init__(parent, t("preset.save_title"))
        bg, small = COLORS["bg_card"], FONTS["small"]
        tk.Label(self.body, text=t("preset.name"), bg=bg, fg=COLORS["text_muted"], font=small, anchor="w").pack(fill=tk.X)
        self.name_var = tk.StringVar(value=name)
        name_entry = tk.Entry(self.body, textvariable=self.name_var, font=FONTS["body"], relief="flat",
                              highlightthickness=1, highlightbackground=COLORS["border"])
        name_entry.pack(fill=tk.X, pady=(SPACE["xs"], SPACE["md"]))
        tk.Label(self.body, text=t("preset.description"), bg=bg, fg=COLORS["text_muted"], font=small,
                 anchor="w").pack(fill=tk.X)
        self.description_var = tk.StringVar()
        tk.Entry(self.body, textvariable=self.description_var, font=FONTS["body"], relief="flat",
                 highlightthickness=1, highlightbackground=COLORS["border"]).pack(
            fill=tk.X, pady=(SPACE["xs"], SPACE["md"]))
        head = tk.Frame(self.body, bg=bg)
        head.pack(fill=tk.X)
        tk.Label(head, text=t("preset.parts"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(head, t("preset.parts"), t("preset.parts_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self.part_vars = {}
        for part in preset.PARTS:
            if part not in available:
                continue
            self.part_vars[part] = tk.BooleanVar(value=part in checked)
            tk.Checkbutton(self.body, text=t(f"preset.part.{part}"), variable=self.part_vars[part], bg=bg,
                           activebackground=bg, selectcolor=COLORS["bg_card"], font=FONTS["body"],
                           anchor="w", command=self._show_cautions).pack(fill=tk.X)
        self._caution = tk.Label(self.body, text="", bg=bg, fg=COLORS["warning"], font=small, anchor="w",
                                 justify=tk.LEFT, wraplength=self.W - 2 * SPACE["xl"])
        self._caution.pack(fill=tk.X, pady=(SPACE["sm"], 0))
        self._show_cautions()
        self._error = tk.Label(self.body, text="", bg=bg, fg=COLORS["warning"], font=small, anchor="w")
        self._error.pack(fill=tk.X, pady=(SPACE["sm"], 0))
        name_entry.focus_set()
        name_entry.select_range(0, tk.END)
        self.bind("<Return>", lambda event: self._save())
        self._fit_content()

    def _build_buttons(self, parent):
        button(parent, text=t("common.save"), command=self._save, kind="primary").pack(side=tk.RIGHT)
        button(parent, text=t("common.cancel"), command=self.destroy, kind="quiet").pack(
            side=tk.RIGHT, padx=(0, SPACE["md"]))

    def _show_cautions(self):
        self._caution.config(text="\n".join(self._cautions[part] for part in self.chosen_parts()
                                            if part in self._cautions))
        self.update_idletasks()
        shown = self._caution.winfo_reqheight() if self._caution.cget("text") else 0
        self.geometry(f"{max(self.winfo_width(), self.W)}x{self.H + shown}")

    def chosen_parts(self) -> tuple:
        return tuple(part for part, variable in self.part_vars.items() if variable.get())

    def _save(self):
        name = self.name_var.get().strip()
        if not name:
            self._error.config(text=t("preset.need_name"))
            return
        if not self.chosen_parts():
            self._error.config(text=t("preset.need_part"))
            return
        self.destroy()
        self._on_save(name, self.description_var.get().strip(), self.chosen_parts())
