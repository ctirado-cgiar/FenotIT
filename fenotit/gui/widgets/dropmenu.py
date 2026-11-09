"""Menú desplegable de la barra superior, con submenús."""
from __future__ import annotations

import tkinter as tk

from fenotit.gui.style.components import divider
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.i18n import t


class Sub:
    """Submenú que se arma al abrirlo (p. ej. Recientes)."""
    def __init__(self, fn):
        self.fn = fn


class DropMenu(tk.Frame):
    def __init__(self, parent, text, items, colors, arrow_only=False, **kw):
        bg = COLORS["bg_card"] if arrow_only else colors["bg_topbar"]
        super().__init__(parent, bg=bg, **kw)
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
        pw = popup.winfo_reqwidth()
        top = self.winfo_toplevel()
        right = top.winfo_rootx() + top.winfo_width()
        if self._arrow_only or x + pw > right - 4:          # el de ▼ junto al borde derecho se abre hacia la izquierda
            x = self.winfo_rootx() + self.winfo_width() - pw
        x = max(top.winfo_rootx() + 4, min(x, right - pw - 4))
        popup.wm_geometry(f"+{x}+{y}")
        popup.bind("<FocusOut>", lambda e: self.after(150, self._maybe_close))
        popup.focus_set()

    def _family(self):
        return [w for w in [self._popup, *getattr(self, "_subs", [])] if w is not None]

    def _maybe_close(self):
        """Se cierra al perder el foco, salvo que el foco esté en uno de sus submenús."""
        try:
            f = self.focus_get()
        except (KeyError, tk.TclError):
            f = None
        fam = self._family()
        if f is not None and any(str(f).startswith(str(w)) for w in fam):
            return
        self._close()

    def _close(self):
        for w in reversed(self._family()):
            try:
                w.destroy()
            except tk.TclError:
                pass
        self._popup, self._subs = None, []

    def _close_subs(self, level: int):
        """Cierra los submenús desde `level` (0 = todos)."""
        for w in self._subs[level:]:
            try:
                w.destroy()
            except tk.TclError:
                pass
        del self._subs[level:]

    def _panel(self, master, items, level: int = 0) -> tk.Toplevel:
        c = self.colors
        popup = tk.Toplevel(master)
        popup.wm_overrideredirect(True)
        popup.configure(bg=c["border"])
        popup.attributes("-topmost", True)
        inner = tk.Frame(popup, bg=c["bg_card"], padx=1, pady=1)
        inner.pack(padx=1, pady=1)
        for item in items:
            if item is None:
                divider(inner).pack(fill=tk.X, pady=SPACE["xs"])
                continue
            label, cmd, *rest = item
            label = (label() if callable(label) else label).strip()
            sub = cmd if isinstance(cmd, (list, tuple, Sub)) else None
            disabled = bool(rest and rest[0] is True)
            shortcut = "▸" if sub else next((r for r in rest if isinstance(r, str)), "")
            row = tk.Frame(inner, bg=c["bg_card"])
            row.pack(fill=tk.X)
            lbl = tk.Label(row, text=f"  {label}   ", bg=c["bg_card"],
                           fg=c["text_muted"] if disabled else c["text"],
                           font=FONTS["body"], anchor="w", pady=SPACE["sm"], padx=SPACE["sm"])
            lbl.pack(fill=tk.X)
            if shortcut:
                tk.Label(row, text=shortcut, bg=c["bg_card"], fg=c["text_muted"],
                         font=FONTS["small"], padx=SPACE["md"]).place(relx=1.0, rely=0.5, anchor="e")
                lbl.config(text=f"  {label}" + " " * (len(shortcut) + 10))
            if disabled:
                continue
            self._bind_row(row, cmd, sub, level)
        return popup

    def _bind_row(self, row, cmd, sub, level):
        c = self.colors

        def paint(color):
            for w in [row] + list(row.winfo_children()):
                try:
                    w.config(bg=color)
                except tk.TclError:
                    pass

        def enter(e):
            paint(c["accent_light"])
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

        def click(e):
            if sub:
                return
            self._close()
            cmd()

        for w in [row] + list(row.winfo_children()):
            w.bind("<Enter>", enter)
            w.bind("<Leave>", lambda e: paint(c["bg_card"]))
            w.bind("<Button-1>", click)
