"""
ui/config_panel.py  v1.2
Panel de configuración compacto.
- Entrada de texto + botones ▲▼ en lugar de sliders grandes
- Tooltips inteligentes (detectan borde de pantalla)
- Menos padding, más compacto
"""

import tkinter as tk
from tkinter import ttk
from typing import Any

from fenotit import log

_log = log.get("gui.config_panel")


class ConfigPanel(tk.Frame):

    def __init__(self, parent, schema: list[dict],
                 colors: dict, **kwargs):
        super().__init__(parent, bg=colors["bg_panel"], **kwargs)
        self.colors = colors
        self._vars: dict[str, tk.Variable] = {}
        self._tips: list[tk.Toplevel]      = []
        self._build(schema)

    def _build(self, schema):
        canvas = tk.Canvas(self, bg=self.colors["bg_panel"],
                           highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical",
                           command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)

        self.inner = tk.Frame(canvas, bg=self.colors["bg_panel"])
        win_id = canvas.create_window((0,0), window=self.inner,
                                      anchor="nw")

        self.inner.bind("<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e: canvas.itemconfig(win_id, width=e.width))

        def _enter(e): canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(
                int(-1*(e.delta/120)), "units"))
        def _leave(e): canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _enter)
        canvas.bind("<Leave>", _leave)

        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        if not schema:
            tk.Label(self.inner,
                     text="Sin parámetros configurables.",
                     bg=self.colors["bg_panel"],
                     fg=self.colors["text_muted"],
                     font=("Segoe UI", 8),
                     pady=8).pack(fill=tk.X, padx=8)
            return

        for item in schema:
            self._add_param(self.inner, item)

    def _add_param(self, parent, item: dict):
        key     = item["key"]
        label   = item.get("label", key)
        typ     = item.get("type", "int")
        default = item.get("default", 0)
        tooltip = item.get("tooltip", "")
        vmin    = item.get("min")
        vmax    = item.get("max")

        row = tk.Frame(parent, bg=self.colors["bg_panel"])
        row.pack(fill=tk.X, padx=8, pady=2)

        # Etiqueta + ?
        hdr = tk.Frame(row, bg=self.colors["bg_panel"])
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=label,
                 bg=self.colors["bg_panel"],
                 fg=self.colors["text"],
                 font=("Segoe UI", 8),
                 anchor="w").pack(side=tk.LEFT)
        if tooltip:
            tip_btn = tk.Label(hdr, text=" ?",
                               bg=self.colors["bg_panel"],
                               fg=self.colors["accent"],
                               font=("Segoe UI", 8, "bold"),
                               cursor="hand2")
            tip_btn.pack(side=tk.LEFT)
            tip_btn.bind("<Button-1>",
                lambda e, t=tooltip, l=label:
                self._show_tip(e, l, t))

        if typ == "bool":
            var = tk.BooleanVar(value=bool(default))
            tk.Checkbutton(row, variable=var,
                           bg=self.colors["bg_panel"],
                           fg=self.colors["text"],
                           selectcolor=self.colors["bg_card"],
                           activebackground=self.colors["bg_panel"],
                           font=("Segoe UI", 8)).pack(anchor="w")
        else:
            # Entrada de texto + botones ▲▼
            var = tk.DoubleVar(value=float(default)) \
                  if typ == "float" else tk.IntVar(value=int(default))

            ctrl_row = tk.Frame(row, bg=self.colors["bg_panel"])
            ctrl_row.pack(fill=tk.X)

            entry = tk.Entry(ctrl_row, textvariable=var, width=7,
                             bg=self.colors["bg_card"],
                             fg=self.colors["text"],
                             insertbackground=self.colors["text"],
                             relief="solid", bd=1,
                             font=("Consolas", 8))
            entry.pack(side=tk.LEFT, ipady=2)

            # Botones ▲▼ compactos
            btn_frame = tk.Frame(ctrl_row, bg=self.colors["bg_panel"])
            btn_frame.pack(side=tk.LEFT, padx=2)

            step = 0.1 if typ == "float" else 1

            def make_up(v=var, s=step, mn=vmin, mx=vmax):
                def up():
                    try:
                        cur = v.get()
                        nv  = cur + s
                        if mx is not None:
                            nv = min(nv, mx)
                        v.set(round(nv, 4) if isinstance(s, float) else int(nv))
                    except Exception:
                        _log.debug("ignorado", exc_info=True)
                return up

            def make_dn(v=var, s=step, mn=vmin, mx=vmax):
                def dn():
                    try:
                        cur = v.get()
                        nv  = cur - s
                        if mn is not None:
                            nv = max(nv, mn)
                        v.set(round(nv, 4) if isinstance(s, float) else int(nv))
                    except Exception:
                        _log.debug("ignorado", exc_info=True)
                return dn

            btn_kw = dict(bg=self.colors["bg_panel"],
                          fg=self.colors["accent"],
                          relief="flat",
                          font=("Segoe UI", 7),
                          width=2, height=1,
                          cursor="hand2",
                          padx=0, pady=0)
            tk.Button(btn_frame, text="▲",
                      command=make_up(),
                      **btn_kw).pack(side=tk.TOP)
            tk.Button(btn_frame, text="▼",
                      command=make_dn(),
                      **btn_kw).pack(side=tk.TOP)

        self._vars[key] = var
        tk.Frame(parent, bg=self.colors["border"],
                 height=1).pack(fill=tk.X, padx=6, pady=(2,0))

    def get_values(self) -> dict[str, Any]:
        out = {}
        for k, v in self._vars.items():
            try:
                out[k] = v.get()
            except Exception:
                _log.warning("Valor inválido en '%s'; se usa el valor por defecto", k)
        return out

    def _show_tip(self, event, label: str, text: str):
        for tw in self._tips:
            try:
                tw.destroy()
            except Exception:
                _log.debug("ignorado", exc_info=True)
        self._tips.clear()

        tw = tk.Toplevel()
        tw.wm_overrideredirect(True)
        tw.configure(bg=self.colors["border"])
        tw.attributes("-topmost", True)

        inner = tk.Frame(tw, bg=self.colors["bg_card"],
                         padx=10, pady=8)
        inner.pack(padx=1, pady=1)

        tk.Label(inner, text=label,
                 bg=self.colors["bg_card"],
                 fg=self.colors["accent"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(inner, text=text,
                 bg=self.colors["bg_card"],
                 fg=self.colors["text"],
                 font=("Segoe UI", 8),
                 justify="left",
                 wraplength=260).pack(anchor="w", pady=(3,0))
        tk.Button(inner, text="Cerrar",
                  bg=self.colors["btn_bg"],
                  fg=self.colors["accent"],
                  font=("Segoe UI", 7),
                  relief="flat", padx=6,
                  cursor="hand2",
                  command=tw.destroy).pack(
                      anchor="e", pady=(6,0))

        # Posición inteligente
        tw.update_idletasks()
        tw_w = tw.winfo_reqwidth()
        tw_h = tw.winfo_reqheight()
        sw   = tw.winfo_screenwidth()
        sh   = tw.winfo_screenheight()
        x    = event.widget.winfo_rootx() + 24
        y    = event.widget.winfo_rooty() + 4
        if x + tw_w > sw - 20: x = event.widget.winfo_rootx() - tw_w - 8
        if y + tw_h > sh - 40: y = sh - tw_h - 40
        x = max(10, x)
        y = max(10, y)
        tw.wm_geometry(f"+{x}+{y}")
        self._tips.append(tw)
        tw.focus_set()
        tw.bind("<FocusOut>", lambda e: tw.destroy())