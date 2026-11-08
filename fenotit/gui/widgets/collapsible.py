"""Panel lateral que se puede plegar."""
from __future__ import annotations

import tkinter as tk

from PIL import ImageTk

from fenotit.gui.style.icons import icon
from fenotit.gui.widgets.toolbar import Tooltip
from fenotit.gui.window.common import _peek_tab
from fenotit.i18n import t


class CollapsiblePanel(tk.Frame):
    """Panel lateral que se puede ocultar. Abierto: un ícono pequeño en su esquina de
    arriba, del lado del centro (izquierdo: a la derecha; derecho: a la izquierda).
    Oculto: el panel desaparece y queda una pestañita en el borde de la zona central."""
    COLLAPSED_W = 14

    def __init__(self, parent, side: str, title: str,
                 colors: dict, default_width: int = 200, on_toggle=None, max_width: int = 420, **kw):
        super().__init__(parent, bg=colors["bg_panel"], **kw)
        self.max_width = max_width
        self.colors        = colors
        self._on_toggle    = on_toggle
        self.side          = side          # "left" o "right"
        self.title         = title
        self.default_width = default_width
        self._expanded     = True
        self._tab = None
        self.tab_host: tk.Widget | None = None    # dónde se asoma la pestañita (zona central)
        # ícono de panel (no una flecha: las flechas quedan para anterior/siguiente)
        self._icon = ImageTk.PhotoImage(icon(f"panel_{side}", 12, colors["accent"]))

        self.content = tk.Frame(self, bg=colors["bg_panel"])
        self.content.pack(fill=tk.BOTH, expand=True)
        self._corner = tk.Label(self, image=self._icon, bg=colors["bg_panel"], cursor="hand2", padx=2, pady=2)
        self._corner_place = (dict(relx=1.0, x=-4, y=5, anchor="ne") if side == "left"
                              else dict(x=4, y=5, anchor="nw"))
        self._corner.place(**self._corner_place)
        self._corner.bind("<Button-1>", lambda e=None: self.toggle())
        Tooltip(self._corner, t("view.toggle_panel"))

        self.configure(width=default_width)
        # el ancho lo decide el usuario (barra divisoria), no el contenido: así el panel no
        # crece solo cuando aparece un texto o un menú largo
        self.pack_propagate(False)
        self.bind("<Map>", lambda e: self._corner.lift(), add="+")

    def toggle(self):
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def _pane(self, **kw):
        if isinstance(self.master, tk.PanedWindow):
            self.master.paneconfigure(self, **kw)

    def collapse(self):
        """Oculta el panel y deja la pestañita para volver a abrirlo."""
        self._expanded = False
        width = self.winfo_width()
        if width > self.COLLAPSED_W * 3:          # recordar el ancho que dejó el usuario
            self.default_width = width
        self._pane(hide=True)
        if self.tab_host is not None:
            if self._tab is None:
                self._tab = _peek_tab(self.tab_host, self._icon, self.expand, self.side)
            self._tab.show()
        if self._on_toggle:
            self._on_toggle()

    def expand(self):
        self._expanded = True
        if self._tab is not None:
            self._tab.place_forget()
        self._pane(hide=False, width=self.default_width)
        self._corner.place(**self._corner_place)
        self._corner.lift()
        if self._on_toggle:
            self._on_toggle()

    def hide_all(self):
        """Sin panel ni pestañita (p. ej. en la pantalla de inicio)."""
        self._pane(hide=True)
        if self._tab is not None:
            self._tab.place_forget()
