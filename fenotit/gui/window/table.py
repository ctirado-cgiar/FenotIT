"""Tabla de resultados: columnas, filas, orden y fila elegida."""
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont

from fenotit.core.analysis.registry import AnalysisResult
from fenotit.gui.window.common import _cell, _row_objects
from fenotit.i18n import t


class TableMixin:
    def _update_table(self, result: AnalysisResult, view: str | None = None):
        """Cada vista tiene su tabla (Conteo, Morfometría, Forma, Color...). Columnas de
        ancho fijo (las que no caben se ven con la barra de abajo) y clic en el título
        ordena de menor a mayor / de mayor a menor."""
        self.table.delete(*self.table.get_children())
        rows = (result.extra.get("view_tables") or {}).get(view)
        if rows is None:
            rows = result.measurements
            cols = result.table_columns or (list(rows[0].keys()) if rows else [])
        else:
            cols = list(dict.fromkeys(k for r in rows for k in r))
        self.table["columns"] = cols
        self._table_rows, self._table_cols, self._table_sort = list(rows or []), cols, (None, False)
        if not rows:
            return
        from fenotit.gui.widgets.labels import column as _col
        from fenotit.core.measurements.classes import slug
        self._col_alias = {slug(r["class"]): r["class"] for r in
                           ((result.extra.get("tables") or {}).get("classes") or [])}
        column = lambda c: _col(c, self._col_alias)
        font, head = tkfont.nametofont("TkDefaultFont"), tkfont.nametofont("TkHeadingFont")
        self._table_widths = {}
        for col in cols:
            sample = [_cell(r.get(col)) for r in rows[:200]]
            width = max([head.measure(column(col) + " ▼") + 20] + [font.measure(v) + 14 for v in sample])
            self._table_widths[col] = min(max(width, 60), 260)
            self.table.heading(col, text=column(col), anchor="w", command=lambda c=col: self._sort_table(c))
        first = cols[0]                    # ▲ en la primera columna si ya viene ordenada: se ve que se puede ordenar
        vals = [r.get(first) for r in rows]
        if all(isinstance(v, (int, float)) for v in vals) and vals == sorted(vals):
            self._table_sort = (first, False)
            self.table.heading(first, text=column(first) + " ▲")
        self._fit_table_columns()
        self._fill_table()

    def _fit_table_columns(self, _event=None):
        """Pocas columnas: se reparten el ancho disponible (sin espacio en blanco).
        Muchas: cada una con su ancho y barra para desplazarse."""
        widths = getattr(self, "_table_widths", None)
        if not widths:
            self._table_hscroll(False)
            return
        avail = self.table.winfo_width() - 2
        total = sum(widths.values())
        if avail < 50:                       # todavía sin dibujar
            return
        if total <= avail:                   # sobra: se reparte
            extra = (avail - total) / len(widths)
            sized = {c: w + extra for c, w in widths.items()}
        elif total <= avail * 1.3:           # falta poco: se angostan un poco (sin barra)
            sized = {c: max(44, w * avail / total) for c, w in widths.items()}
        else:
            sized = widths
        for col, w in sized.items():
            self.table.column(col, width=int(w), minwidth=40, stretch=False, anchor="w")
        self._table_hscroll(sum(int(w) for w in sized.values()) > avail + 1)

    def _table_hscroll(self, need: bool):
        xsb = self._table_xsb
        if need and not xsb.winfo_ismapped():
            xsb.pack(side=tk.BOTTOM, fill=tk.X, before=self.table)
        elif not need and xsb.winfo_ismapped():
            xsb.pack_forget()
            self.table.xview_moveto(0)

    _YES_NO = ("touching", "crosses_object")

    @staticmethod
    def _class_aliases(result) -> dict:
        """Nombre ASCII de cada clase → su nombre; el fondo, marcado con "\0" (no se resume)."""
        from fenotit.core.measurements.classes import slug
        out = {}
        for r in (result.extra.get("tables") or {}).get("classes") or []:
            out[slug(r["class"])] = r["class"]
        return out

    def _shown(self, col: str, v) -> str:
        """Valor para la pantalla: estados y sí/no traducidos; números con pocos decimales."""
        if col == "status" and v:
            return t(f"insp.status.{v}", str(v))
        if col in self._YES_NO and v in (0, 1, True, False):
            return t("common.yes") if v else t("common.no")
        return _cell(v)

    def _fill_table(self):
        self.table.delete(*self.table.get_children())
        self._table_items = {}
        for row in self._table_rows:
            iid = self.table.insert("", "end", values=[self._shown(c, row.get(c)) for c in self._table_cols])
            self._table_items[iid] = row
        self._sync_table_selection()

    def _sync_table_selection(self):
        items = getattr(self, "_table_items", {})
        self._table_syncing = True
        try:
            want = [iid for iid, row in items.items() if self._sel is not None and self._sel in _row_objects(row)]
            self.table.selection_set(want)
            if want:
                self.table.see(want[0])
        finally:
            self.root.after_idle(lambda: setattr(self, "_table_syncing", False))

    def _on_table_select(self, _e=None):
        if getattr(self, "_table_syncing", False):
            return
        sel = self.table.selection()
        if not sel:
            return
        ids = _row_objects(getattr(self, "_table_items", {}).get(sel[0], {}))
        if ids and self.zoom_ctrl and self.zoom_ctrl.tool != "inspect":
            self.zoom_ctrl.set_tool("inspect")          # la tabla elige objetos: también la imagen
        if ids:                         # en Distancias (una fila por par) se elige el primero
            self._select_object(self._sel if self._sel in ids else ids[0], from_table=True)

    def _sort_table(self, col: str):
        """Ordena por esa columna; otro clic invierte el orden (▲ / ▼ en el título)."""
        last, desc = self._table_sort
        desc = not desc if last == col else False
        self._table_sort = (col, desc)

        def key(r):
            v = r.get(col)
            if isinstance(v, (int, float)) and v == v:
                return (0, float(v), "")
            try:
                return (0, float(v), "")
            except (TypeError, ValueError):
                return (1, 0.0, str(v or ""))
        self._table_rows.sort(key=key, reverse=desc)
        from fenotit.gui.widgets.labels import column
        for c in self._table_cols:
            alias = getattr(self, "_col_alias", {})
            mark = (" ▼" if desc else " ▲") if c == col else ""
            self.table.heading(c, text=column(c, alias) + mark)
        self._fill_table()
