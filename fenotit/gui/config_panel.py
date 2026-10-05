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
from fenotit.i18n import t

_log = log.get("gui.config_panel")


class ConfigPanel(tk.Frame):

    def __init__(self, parent, schema: list[dict],
                 colors: dict, prefix: str = "", units: dict | None = None, **kwargs):
        super().__init__(parent, bg=colors["bg_panel"], **kwargs)
        self.colors = colors
        self._vars: dict[str, tk.Variable] = {}
        self._tips: list[tk.Toplevel]      = []
        self._prefix = prefix
        self._units = units or {}
        self._labels: dict[str, tuple[tk.Widget, dict]] = {}
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
            lambda e=None: canvas.configure(
                scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e=None: canvas.itemconfig(win_id, width=e.width))

        def _enter(e): canvas.bind_all(
            "<MouseWheel>",
            lambda e=None: canvas.yview_scroll(
                int(-1*(e.delta/120)), "units"))
        def _leave(e): canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _enter)
        canvas.bind("<Leave>", _leave)

        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        if not schema:
            tk.Label(self.inner,
                     text=t("config.no_params"),
                     bg=self.colors["bg_panel"],
                     fg=self.colors["text_muted"],
                     font=("Segoe UI", 8),
                     pady=8).pack(fill=tk.X, padx=8)
            return

        self._schema = schema
        self._rows: dict[str, tk.Frame] = {}
        self._arrows: dict[str, tk.Label] = {}
        self._children = {i["group"] for i in schema if i.get("group")}
        self._open = {g: False for g in self._children}
        for item in schema:
            self._add_param(self.inner, item)
            self._rows[item["key"]] = self._last_row
        for item in schema:          # casillas con opciones: se refresca al marcar/desmarcar
            if item["key"] in self._children and item["key"] in self._vars:
                self._vars[item["key"]].trace_add("write", lambda *_: self._refresh_rows())
            parent = item.get("requires")
            if parent in self._vars:
                self._vars[parent].trace_add("write", lambda *_: self._refresh_rows())
        self._refresh_rows()

    def _toggle_group(self, key: str):
        self._open[key] = not self._open[key]
        self._refresh_rows()

    def _group_active(self, key: str) -> bool:
        """Una casilla con opciones solo las muestra si está marcada; una sección siempre."""
        return key not in self._vars or bool(self._vars[key].get())

    def _refresh_rows(self):
        """Empaca las filas en el orden del esquema. Las opciones de un grupo (ícono de ajustes) solo se
        ven si el grupo está abierto y activo; las de "requires", si esa casilla está marcada."""
        for row in self._rows.values():
            row.pack_forget()
        for key, btn in self._arrows.items():
            active, opened = self._group_active(key), self._open[key]
            if not active:
                btn.config(image="", bg=self.colors["bg_panel"], cursor="")
            else:
                btn.config(image=self._tune_icons[opened], cursor="hand2",
                           bg=self.colors["accent_light"] if opened else self.colors["bg_panel"])
        for item in self._schema:
            group, parent = item.get("group"), item.get("requires")
            if group and not (self._open.get(group) and self._group_active(group)):
                continue
            if parent in self._vars and not self._vars[parent].get():
                continue
            self._rows[item["key"]].pack(fill=tk.X, padx=(24, 8) if group else 8, pady=2)

    def set_units(self, units: dict):
        """Cambia las unidades en las etiquetas (p. ej. {"area": "mm²"})."""
        self._units = units
        for key, (widget, item) in self._labels.items():
            widget.config(text=self._label_text(item))

    def _label_text(self, item: dict) -> str:
        return t(f"param.{self._prefix}.{item['key']}.label", item.get("label", item["key"]),
                 unit=self._units.get(item.get("unit"), ""))

    def _add_param(self, parent, item: dict):
        key     = item["key"]
        label   = self._label_text(item)
        typ     = item.get("type", "int")
        default = item.get("default", 0)
        tooltip = t(f"param.{self._prefix}.{key}.tip", item.get("tooltip", ""))
        vmin    = item.get("min")
        vmax    = item.get("max")

        row = tk.Frame(parent, bg=self.colors["bg_panel"])
        self._last_row = row

        # Etiqueta + ?  (en los sí/no la casilla va en la misma línea)
        hdr = tk.Frame(row, bg=self.colors["bg_panel"])
        hdr.pack(fill=tk.X)
        if typ == "header":
            row.config(pady=0)
            tk.Label(hdr, text=label, bg=self.colors["bg_panel"], fg=self.colors["accent"],
                     font=("Segoe UI", 8, "bold"), pady=4).pack(side=tk.LEFT)
            tk.Frame(row, bg=self.colors["border"], height=1).pack(fill=tk.X)
            return
        if typ == "section":
            lbl = tk.Label(hdr, text=label, bg=self.colors["bg_panel"], fg=self.colors["text"],
                           font=("Segoe UI", 8), anchor="w", cursor="hand2")
            lbl.bind("<Button-1>", lambda e=None, k=key: self._toggle_group(k))
        elif typ == "bool":
            var = tk.BooleanVar(value=bool(default))
            lbl = tk.Checkbutton(hdr, variable=var, text=label,
                                 bg=self.colors["bg_panel"],
                                 fg=self.colors["text"],
                                 selectcolor=self.colors["bg_card"],
                                 activebackground=self.colors["bg_panel"],
                                 font=("Segoe UI", 8), padx=0)
        else:
            lbl = tk.Label(hdr, text=label,
                           bg=self.colors["bg_panel"],
                           fg=self.colors["text"],
                           font=("Segoe UI", 8),
                           anchor="w")
        lbl.pack(side=tk.LEFT)
        self._labels[key] = (lbl, item)
        if key in self._children:
            if not hasattr(self, "_tune_icons"):           # cerrado: tenue; abierto: azul sobre fondo claro
                from PIL import ImageTk
                from fenotit.gui.toolbar import icon
                self._tune_icons = {False: ImageTk.PhotoImage(icon("tune", 13, "#7FA6CF")),
                                    True: ImageTk.PhotoImage(icon("tune", 13, self.colors["accent"]))}
            arrow = tk.Label(hdr, image=self._tune_icons[False], bg=self.colors["bg_panel"], cursor="hand2",
                             padx=3, pady=1)
            arrow.pack(side=tk.LEFT, padx=(4, 0))
            arrow.bind("<Button-1>", lambda e=None, k=key: self._toggle_group(k))
            from fenotit.gui.toolbar import Tooltip
            Tooltip(arrow, t("panel.options"))
            self._arrows[key] = arrow
        if tooltip:
            from fenotit.gui.help import HelpIcon
            HelpIcon(hdr, label, tooltip, bg=self.colors["bg_panel"]).pack(side=tk.LEFT, padx=(4, 0))

        if typ in ("bool", "section"):
            pass
        elif typ == "choice":
            # Lista de opciones: se guarda la clave, se muestra el texto traducido
            var = tk.StringVar(value=str(default))
            labels = {c: t(f"param.{self._prefix}.{key}.{c}", c) for c in item["choices"]}
            shown = tk.StringVar(value=labels.get(str(default), str(default)))
            combo = ttk.Combobox(row, textvariable=shown, values=list(labels.values()),
                                 state="readonly", width=30, font=("Segoe UI", 8))
            combo.pack(anchor="w", pady=(1, 0))
            combo.bind("<<ComboboxSelected>>", lambda e=None, v=var, sv=shown, lb=labels: v.set(
                next(k for k, l in lb.items() if l == sv.get())))
            var.trace_add("write", lambda *_, v=var, sv=shown, lb=labels: sv.set(lb.get(v.get(), v.get())))
        else:
            # Entrada de texto + botones ▲▼
            var = tk.DoubleVar(value=float(default)) \
                  if typ == "float" else tk.IntVar(value=int(default))

            ctrl_row = tk.Frame(row, bg=self.colors["bg_panel"])
            ctrl_row.pack(fill=tk.X)

            entry = tk.Entry(ctrl_row, textvariable=var, width=10,
                             bg=self.colors["bg_card"],
                             fg=self.colors["text"],
                             insertbackground=self.colors["text"],
                             relief="solid", bd=1,
                             font=("Consolas", 8))
            entry.pack(side=tk.LEFT, ipady=2)

            # Botones ▲▼ compactos
            btn_frame = tk.Frame(ctrl_row, bg=self.colors["bg_panel"])
            btn_frame.pack(side=tk.LEFT, padx=2)

            step = item.get("step", 0.1 if typ == "float" else 1)

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

        if typ != "section":
            self._vars[key] = var
        tk.Frame(row, bg=self.colors["border"],
                 height=1).pack(fill=tk.X, pady=(2, 0))

    def get_values(self, warn: bool = True) -> dict[str, Any]:
        out = {}
        for k, v in self._vars.items():
            try:
                out[k] = v.get()
            except Exception:
                if warn:
                    _log.warning("Valor inválido en '%s'; se usa el valor por defecto", k)
        return out

    def set_values(self, values: dict[str, Any]):
        for k, v in (values or {}).items():
            if k in self._vars:
                try:
                    self._vars[k].set(v)
                except Exception:
                    _log.warning("No se pudo restaurar '%s' = %r", k, v)
