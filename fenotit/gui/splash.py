"""Pantalla de carga: aparece antes de importar lo pesado y la barra avanza con lo que
de verdad se va cargando (módulos, ventana). Se ve al menos `min_ms` para no parpadear."""
from __future__ import annotations

import importlib
import time
import tkinter as tk
from pathlib import Path

from fenotit import APP_NAME, __version__, log
from fenotit.i18n import t

_log = log.get("gui.splash")
_BAR, _TRACK = "#2A7587", "#E6ECEE"       # azul petróleo del logo
_TEXT, _MUTED = "#22343B", "#7A858A"


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
        W, H = 440, 290
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w.geometry(f"{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}")
        w.overrideredirect(True)
        w.configure(bg="#FFFFFF", highlightthickness=1, highlightbackground="#D9DEE0")
        w.attributes("-topmost", True)
        ico = _assets() / "logo.ico"
        try:
            from PIL import Image, ImageTk
            img = Image.open(ico).convert("RGBA").resize((64, 64), Image.LANCZOS)
            self._logo = ImageTk.PhotoImage(img)
            tk.Label(w, image=self._logo, bg="#FFFFFF").pack(pady=(34, 8))
        except Exception:
            _log.debug("logo", exc_info=True)
        tk.Label(w, text=APP_NAME, bg="#FFFFFF", fg=_TEXT, font=("Segoe UI", 24, "bold")).pack()
        tk.Label(w, text=t("start.subtitle"), bg="#FFFFFF", fg=_BAR,
                 font=("Georgia", 11, "italic")).pack(pady=(2, 0))

        foot = tk.Frame(w, bg="#FFFFFF")
        foot.pack(side=tk.BOTTOM, fill=tk.X, padx=36, pady=(0, 14))
        tk.Label(foot, text=f"v{__version__}  ·  Alliance Bioversity & CIAT", bg="#FFFFFF", fg="#A9B1B5",
                 font=("Segoe UI", 7)).pack(side=tk.RIGHT)
        self._track = tk.Frame(w, bg=_TRACK, height=4)
        self._track.pack(side=tk.BOTTOM, fill=tk.X, padx=36, pady=(4, 14))
        self._fill = tk.Frame(self._track, bg=_BAR)
        self._fill.place(x=0, y=0, relheight=1, relwidth=0)
        self.msg = tk.StringVar(value=t("splash.loading"))
        tk.Label(w, textvariable=self.msg, bg="#FFFFFF", fg=_MUTED, font=("Segoe UI", 8),
                 anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=36)
        self._set(0.03)
        w.update()

    def _set(self, frac: float, msg: str | None = None):
        self._frac = max(self._frac, min(1.0, frac))
        if msg:
            self.msg.set(msg)
        self._fill.place_configure(relwidth=self._frac)
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
