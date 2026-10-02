"""Pantalla de carga: aparece antes de importar lo pesado y la barra avanza con lo que
de verdad se va cargando (módulos, ventana). Se ve al menos `min_ms` para no parpadear."""
from __future__ import annotations

import importlib
import time
import tkinter as tk
from pathlib import Path

from fenotit import APP_NAME, __version__, log
from fenotit.gui.decor import BLUE, mosaic
from fenotit.i18n import t

_log = log.get("gui.splash")


def _assets() -> Path:
    return Path(__file__).parent.parent / "assets"


class SplashScreen:

    def __init__(self, root: tk.Tk, min_ms: int = 1000):
        self.root = root
        self.min_ms = min_ms
        self.window: tk.Toplevel | None = None
        self._frac = 0.0

    def show(self):
        self._t0 = time.monotonic()
        w = self.window = tk.Toplevel(self.root)
        W, H = 500, 300
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w.geometry(f"{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}")
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        c = self._c = tk.Canvas(w, width=W, height=H, bg="#FFFFFF", highlightthickness=1,
                                highlightbackground="#D5DEE8")
        c.pack(fill=tk.BOTH, expand=True)
        c.create_rectangle(0, 0, W, 6, fill=BLUE, width=0)
        mosaic(c, W, 6, 13, [7, 5, 4, 3, 2, 2, 1], mirror=True)
        x0 = 40
        try:
            from PIL import Image, ImageTk
            img = Image.open(_assets() / "logo.ico").convert("RGBA").resize((64, 64), Image.LANCZOS)
            self._logo = ImageTk.PhotoImage(img)
            c.create_image(x0, 70, image=self._logo, anchor="nw")
            x1 = x0 + 78
        except Exception:
            _log.debug("logo", exc_info=True)
            x1 = x0
        c.create_text(x1, 66, text=APP_NAME, anchor="nw", fill=BLUE, font=("Segoe UI", 28, "bold"))
        c.create_text(x1 + 2, 112, text=t("start.subtitle"), anchor="nw", fill="#4A4A4A",
                      font=("Georgia", 11, "italic"))
        c.create_text(x0, 160, text="Alliance of Bioversity International & CIAT", anchor="nw",
                      fill="#8A8A8A", font=("Segoe UI", 8))
        c.create_line(x0, 186, W - x0, 186, fill="#E3E8EE")
        self._msg = c.create_text(x0, 204, text=t("splash.loading"), anchor="nw", fill="#666666",
                                  font=("Segoe UI", 8))
        self._bar = (x0, 226, W - x0, 230)
        c.create_rectangle(*self._bar, fill="#E8EEF5", width=0)
        self._fill = c.create_rectangle(x0, 226, x0, 230, fill=BLUE, width=0)
        c.create_text(W - x0, H - 18, text=f"v{__version__}", anchor="e", fill="#B0B0B0",
                      font=("Segoe UI", 7))
        c.create_rectangle(0, H - 4, W, H, fill=BLUE, width=0)
        self._set(0.03)
        w.update()

    def _set(self, frac: float, msg: str | None = None):
        self._frac = max(self._frac, min(1.0, frac))
        if msg:
            self._c.itemconfig(self._msg, text=msg)
        x0, y0, x1, y1 = self._bar
        self._c.coords(self._fill, x0, y0, x0 + (x1 - x0) * self._frac, y1)
        self.window.update_idletasks()

    def load(self, stages):
        """`stages`: (fracción al terminar, mensaje, módulos o función). Importa / ejecuta
        cada etapa y adelanta la barra."""
        out = None
        for frac, msg, work in stages:
            self._set(self._frac + 0.04, msg)
            self.window.update()
            if callable(work):
                out = work()
            else:
                for mod in work:
                    importlib.import_module(mod)
            self._set(frac)
        return out

    def finish(self, then):
        """Completa la barra con suavidad (respetando el tiempo mínimo) y cierra."""
        left = max(0.0, self.min_ms / 1000 - (time.monotonic() - self._t0))
        steps = max(1, int(left / 0.03))
        start = self._frac

        def tick(i=1):
            if self.window is None:
                return
            self._set(start + (1 - start) * i / steps)
            if i < steps:
                self.window.after(30, lambda: tick(i + 1))
            else:
                self.window.after(80, lambda: (self.close(), then()))
        tick()

    def close(self):
        if self.window:
            try:
                self.window.destroy()
            except tk.TclError:
                _log.debug("ignorado", exc_info=True)
            self.window = None
