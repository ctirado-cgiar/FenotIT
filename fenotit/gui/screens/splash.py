"""Pantalla de carga: aparece antes de importar lo pesado y la barra avanza con lo que
de verdad se va cargando (módulos, ventana). Se ve al menos `min_ms` para no parpadear."""
from __future__ import annotations

import importlib
import time
import tkinter as tk
from pathlib import Path

from fenotit import APP_NAME, __version__, log
from fenotit.gui.style.decor import BLUE, mosaic
from fenotit.i18n import t
from fenotit.gui.style.theme import COLORS, FONTS

_log = log.get("gui.splash")


def _assets() -> Path:
    return Path(__file__).parents[2] / "assets"


class SplashScreen:

    def __init__(self, root: tk.Tk, min_ms: int = 1000):
        self.root = root
        self.min_ms = min_ms
        self.window: tk.Toplevel | None = None
        self._frac = 0.0

    def show(self):
        self._t0 = time.monotonic()
        window = self.window = tk.Toplevel(self.root)
        width, height = 500, 300
        screen_width, screen_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        window.geometry(f"{width}x{height}+{(screen_width - width) // 2}+{(screen_height - height) // 2}")
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        canvas = self._c = tk.Canvas(window, width=width, height=height, bg=COLORS["bg_card"], highlightthickness=1,
                                highlightbackground=COLORS["divider"])
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.create_rectangle(0, 0, width, 6, fill=BLUE, width=0)
        mosaic(canvas, width, 6, 13, [7, 5, 4, 3, 2, 2, 1], mirror=True)
        x0 = 40
        try:
            from PIL import Image, ImageTk
            img = Image.open(_assets() / "logo.ico").convert("RGBA").resize((64, 64), Image.LANCZOS)
            self._logo = ImageTk.PhotoImage(img)
            canvas.create_image(x0, 70, image=self._logo, anchor="nw")
            x1 = x0 + 78
        except Exception:
            _log.debug("logo", exc_info=True)
            x1 = x0
        canvas.create_text(x1, 66, text=APP_NAME, anchor="nw", fill=BLUE, font=FONTS["display_xl"])
        canvas.create_text(x1 + 2, 112, text=t("start.subtitle"), anchor="nw", fill=COLORS["text_muted"],
                      font=FONTS["motto"])
        canvas.create_text(x0, 160, text="Alliance of Bioversity International & CIAT", anchor="nw",
                      fill=COLORS["text_faint"], font=FONTS["small"])
        canvas.create_line(x0, 186, width - x0, 186, fill=COLORS["divider"])
        self._msg = canvas.create_text(x0, 204, text=t("splash.loading"), anchor="nw", fill=COLORS["text_muted"],
                                  font=FONTS["small"])
        self._bar = (x0, 226, width - x0, 230)
        canvas.create_rectangle(*self._bar, fill=COLORS["placeholder"], width=0)
        self._fill = canvas.create_rectangle(x0, 226, x0, 230, fill=BLUE, width=0)
        canvas.create_text(width - x0, height - 18, text=f"v{__version__}", anchor="e", fill=COLORS["text_faint"],
                      font=FONTS["tiny"])
        canvas.create_rectangle(0, height - 4, width, height, fill=BLUE, width=0)
        self._set(0.03)
        window.update()

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
                self.root.after(30, lambda: tick(i + 1))
            else:                                  # en la ventana principal: el callback cierra esta
                self.root.after(80, lambda: (self.close(), then()))
        tick()

    def close(self):
        if self.window:
            try:
                self.window.destroy()
            except tk.TclError:
                _log.debug("ignorado", exc_info=True)
            self.window = None
