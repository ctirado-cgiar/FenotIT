"""Preferencias (Configuración → Preferencias…): idioma, resolución de trabajo, procesos en
paralelo, tabla al ejecutar y autoguardado. Se guardan en la configuración del usuario; la
resolución de trabajo también queda en el proyecto (cambia los resultados)."""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Callable

from fenotit import i18n, settings
from fenotit.gui.help import HelpIcon
from fenotit.gui.theme import COLORS, FONTS
from fenotit.i18n import t

WORK_RES = (("0", "workres.full"), ("100.0", "workres.100"), ("50.0", "workres.50"),
            ("25.0", "workres.25"), ("12.0", "workres.12"))
AUTOSAVE = (0, 1, 2, 3, 5, 10)


def defaults() -> dict:
    return {"autosave_min": 3, "show_table_on_run": True, "workers": 0, "default_max_mpx": 50.0}


def get(key: str):
    return settings.get(key, defaults()[key])


def show(root: tk.Misc, max_mpx: float | None, on_ok: Callable[[dict], None]) -> tk.Toplevel:
    bg = COLORS["bg_card"]
    win = tk.Toplevel(root)
    win.title(t("prefs.title"))
    win.configure(bg=bg)
    win.resizable(False, False)
    win.transient(root)
    tk.Frame(win, bg=COLORS["accent"], height=4).pack(fill=tk.X)
    body = tk.Frame(win, bg=bg)
    body.pack(fill=tk.BOTH, padx=20, pady=(12, 4))
    tk.Label(body, text=t("prefs.title"), bg=bg, fg=COLORS["accent"],
             font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
    row = [1]

    def section(key):
        tk.Label(body, text=t(key).upper(), bg=bg, fg=COLORS["text_muted"],
                 font=("Segoe UI", 8, "bold")).grid(row=row[0], column=0, columnspan=3, sticky="w", pady=(10, 2))
        row[0] += 1

    def line(key, widget, help_key=None):
        tk.Label(body, text=t(key), bg=bg, fg=COLORS["text"], font=FONTS["body"]).grid(
            row=row[0], column=0, sticky="w", padx=(0, 12), pady=2)
        widget.grid(row=row[0], column=1, sticky="w", pady=2)
        if help_key:
            HelpIcon(body, t(key), t(help_key), bg=bg).grid(row=row[0], column=2, sticky="w", padx=6)
        row[0] += 1

    def combo(values, current, width=26):
        var = tk.StringVar(value=current)
        cb = ttk.Combobox(body, textvariable=var, values=values, state="readonly", width=width)
        return var, cb

    langs = i18n.available()
    section("prefs.general")
    lang_var, cb = combo(list(langs.values()), langs.get(i18n.current(), ""))
    line("prefs.language", cb)

    section("prefs.images")
    res_names = {v: t(k) for v, k in WORK_RES}
    cur = str(max_mpx or 0)
    res_var, cb = combo(list(res_names.values()), res_names.get(cur, res_names["50.0"]), 34)
    line("workres.title", cb, "prefs.workres_help")
    cpus = max(1, (os.cpu_count() or 2) - 1)
    work_names = [t("prefs.auto")] + [str(n) for n in range(1, cpus + 1)]
    w = int(get("workers") or 0)
    work_var, cb = combo(work_names, work_names[w] if 0 < w <= cpus else work_names[0], 10)
    line("prefs.workers", cb, "prefs.workers_help")

    section("prefs.results")
    table_var = tk.BooleanVar(value=bool(get("show_table_on_run")))
    line("prefs.show_table", tk.Checkbutton(body, variable=table_var, bg=bg, activebackground=bg,
                                            selectcolor=COLORS["bg_panel"]), "prefs.show_table_help")

    section("prefs.saving")
    names = [t("prefs.autosave_off")] + [t("prefs.every_min", n=n) for n in AUTOSAVE[1:]]
    a = int(get("autosave_min") or 0)
    save_var, cb = combo(names, names[AUTOSAVE.index(a)] if a in AUTOSAVE else names[3], 16)
    line("prefs.autosave", cb, "prefs.autosave_help")

    def ok():
        code = next((c for c, n in langs.items() if n == lang_var.get()), i18n.current())
        res = next(v for v, n in res_names.items() if n == res_var.get())
        values = {"language": code,
                  "max_mpx": float(res) or None,
                  "workers": work_names.index(work_var.get()),
                  "show_table_on_run": bool(table_var.get()),
                  "autosave_min": AUTOSAVE[names.index(save_var.get())]}
        win.destroy()
        on_ok(values)

    btns = tk.Frame(win, bg=bg)
    btns.pack(fill=tk.X, padx=20, pady=(10, 14))
    tk.Button(btns, text=t("common.cancel"), command=win.destroy, bg=COLORS["btn_bg"], fg=COLORS["accent"],
              relief="flat", font=FONTS["body"], cursor="hand2", padx=10).pack(side=tk.RIGHT, padx=(6, 0))
    tk.Button(btns, text=t("common.ok"), command=ok, bg=COLORS["accent"], fg="#FFFFFF", relief="flat",
              font=("Segoe UI", 9, "bold"), cursor="hand2", padx=14).pack(side=tk.RIGHT)
    win.bind("<Escape>", lambda e: win.destroy())
    win.update_idletasks()
    win.geometry(f"+{root.winfo_rootx() + (root.winfo_width() - win.winfo_reqwidth()) // 2}"
                 f"+{root.winfo_rooty() + (root.winfo_height() - win.winfo_reqheight()) // 3}")
    win.grab_set()
    return win
