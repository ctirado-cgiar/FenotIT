"""Menú desplegable de la barra superior, con submenús."""
from __future__ import annotations

import tkinter as tk

# divider dibuja una línea separadora.
from fenotit.gui.style.components import divider
# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras); SPACE (márgenes).
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
# t devuelve cada texto de la ventana en el idioma elegido (fenotit/lang).
from fenotit.i18n import t


class Sub:
    """Submenú que se arma al abrirlo (p. ej. Recientes)."""
    def __init__(self, func):
        self.fn = func


class DropMenu(tk.Frame):
    """Botón de la barra superior. `items` = lista de elementos del menú, o una función que
    se ejecuta al tocarlo (sin menú)."""
    def __init__(self, parent, text, items, colors, arrow_only=False, **kwargs):
        bg = COLORS["bg_card"] if arrow_only else colors["bg_topbar"]
        super().__init__(parent, bg=bg, **kwargs)
        self.colors = colors
        self.items  = items
        self._popup = None
        self._arrow_only = arrow_only
        self.btn = tk.Button(
            self, text="▼" if arrow_only else f"  {text}  ",
            bg=bg, fg=colors["accent"] if arrow_only else COLORS["text_topbar"],
            font=FONTS["subtitle"] if arrow_only else FONTS["topbar_menu"], relief="flat", bd=0,
            **({"padx": 8} if arrow_only else {}),
            activebackground=COLORS["accent_dark"] if arrow_only else COLORS["topbar_pressed"],
            activeforeground=COLORS["on_accent"] if arrow_only else COLORS["text_topbar"],
            cursor="hand2", highlightthickness=0, command=self._toggle)
        self.btn.pack()

    def _toggle(self):
        if callable(self.items):                    # botón sin menú: ejecuta la acción
            self.items()
            return
        if self._popup and self._popup.winfo_exists():
            self._close()
        else:
            self._open()

    def _open(self):
        """Menú desplegable. Un elemento (texto, [elementos]) abre un submenú a la derecha
        al pasar el mouse (como en QGIS)."""
        self.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        popup = self._panel(self, self.items)
        self._popup = popup
        self._subs: list[tk.Toplevel] = []
        popup.update_idletasks()
        panel_width = popup.winfo_reqwidth()
        top = self.winfo_toplevel()
        right = top.winfo_rootx() + top.winfo_width()
        if self._arrow_only or x + panel_width > right - 4:          # el de ▼ junto al borde derecho se abre hacia la izquierda
            x = self.winfo_rootx() + self.winfo_width() - panel_width
        x = max(top.winfo_rootx() + 4, min(x, right - panel_width - 4))
        popup.wm_geometry(f"+{x}+{y}")
        popup.bind("<FocusOut>", lambda event: self.after(150, self._maybe_close))
        popup.focus_set()

    def _family(self):
        return [widget for widget in [self._popup, *getattr(self, "_subs", [])] if widget is not None]

    def _maybe_close(self):
        """Se cierra al perder el foco, salvo que el foco esté en uno de sus submenús."""
        try:
            focused = self.focus_get()
        except (KeyError, tk.TclError):
            focused = None
        fam = self._family()
        if focused is not None and any(str(focused).startswith(str(widget)) for widget in fam):
            return
        self._close()

    def _close(self):
        for widget in reversed(self._family()):
            try:
                widget.destroy()
            except tk.TclError:
                pass
        self._popup, self._subs = None, []

    def _close_subs(self, level: int):
        """Cierra los submenús desde `level` (0 = todos)."""
        for widget in self._subs[level:]:
            try:
                widget.destroy()
            except tk.TclError:
                pass
        del self._subs[level:]

    def _panel(self, master, items, level: int = 0) -> tk.Toplevel:
        colors = self.colors
        popup = tk.Toplevel(master)
        popup.wm_overrideredirect(True)
        popup.configure(bg=colors["border"])
        popup.attributes("-topmost", True)
        inner = tk.Frame(popup, bg=colors["bg_card"], padx=1, pady=1)
        inner.pack(padx=1, pady=1)
        for item in items:
            if item is None:
                divider(inner).pack(fill=tk.X, pady=SPACE["xs"])
                continue
            label, cmd, *rest = item
            label = (label() if callable(label) else label).strip()
            sub = cmd if isinstance(cmd, (list, tuple, Sub)) else None
            disabled = bool(rest and rest[0] is True)
            shortcut = "▸" if sub else next((part for part in rest if isinstance(part, str)), "")
            row = tk.Frame(inner, bg=colors["bg_card"])
            row.pack(fill=tk.X)
            lbl = tk.Label(row, text=f"  {label}   ", bg=colors["bg_card"],
                           fg=colors["text_muted"] if disabled else colors["text"],
                           font=FONTS["body"], anchor="w", pady=SPACE["sm"], padx=SPACE["sm"])
            lbl.pack(fill=tk.X)
            if shortcut:
                tk.Label(row, text=shortcut, bg=colors["bg_card"], fg=colors["text_muted"],
                         font=FONTS["small"], padx=SPACE["md"]).place(relx=1.0, rely=0.5, anchor="e")
                lbl.config(text=f"  {label}" + " " * (len(shortcut) + 10))
            if disabled:
                continue
            self._bind_row(row, cmd, sub, level)
        return popup

    def _bind_row(self, row, cmd, sub, level):
        colors = self.colors

        def paint(color):
            for widget in [row] + list(row.winfo_children()):
                try:
                    widget.config(bg=color)
                except tk.TclError:
                    pass

        def enter(event):
            paint(colors["accent_light"])
            self._close_subs(level)
            if sub:
                items = sub.fn() if isinstance(sub, Sub) else sub
                child = self._panel(self._popup, items or [(t("menu.none"), None, True)], level + 1)
                child.update_idletasks()
                x, y = row.winfo_rootx() + row.winfo_width(), row.winfo_rooty() - 2
                right = self.winfo_toplevel().winfo_rootx() + self.winfo_toplevel().winfo_width()
                if x + child.winfo_reqwidth() > right:
                    x = row.winfo_rootx() - child.winfo_reqwidth()
                child.wm_geometry(f"+{x}+{y}")
                self._subs.append(child)

        def click(event):
            if sub:
                return
            self._close()
            cmd()

        for widget in [row] + list(row.winfo_children()):
            widget.bind("<Enter>", enter)
            widget.bind("<Leave>", lambda event: paint(colors["bg_card"]))
            widget.bind("<Button-1>", click)
