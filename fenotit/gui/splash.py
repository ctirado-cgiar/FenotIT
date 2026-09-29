"""
ui/splash.py  —  Splash screen con tema claro científico + logo .ico
"""

import tkinter as tk
from pathlib import Path

from fenotit import __version__, log

_log = log.get("gui.splash")


def _assets() -> Path:
    return Path(__file__).parent.parent / "assets"


class SplashScreen:

    def __init__(self, root: tk.Tk, duration_ms: int = 2200):
        self.root        = root
        self.duration_ms = duration_ms
        self.window: tk.Toplevel | None = None

    def show(self):
        self.window = tk.Toplevel(self.root)
        W, H = 480, 300

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.window.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
        self.window.overrideredirect(True)
        self.window.configure(bg="#FFFFFF")
        self.window.attributes("-topmost", True)

        # Ícono en barra de tareas
        ico = _assets() / "logo.ico"
        if ico.exists():
            try:
                self.window.iconbitmap(str(ico))
            except Exception:
                _log.debug("ignorado", exc_info=True)

        # ── Borde azul superior (banda de color) ──────────────────────────────
        tk.Frame(self.window, bg="#2166AC", height=6).pack(fill=tk.X)

        body = tk.Frame(self.window, bg="#FFFFFF")
        body.pack(fill=tk.BOTH, expand=True, padx=32, pady=0)

        # Logo + nombre
        logo_row = tk.Frame(body, bg="#FFFFFF")
        logo_row.pack(pady=(24, 0))

        # Intentar mostrar ícono como imagen
        try:
            from PIL import Image, ImageTk
            img = Image.open(str(ico)).resize((48, 48), Image.LANCZOS)
            self._logo_photo = ImageTk.PhotoImage(img)
            tk.Label(logo_row, image=self._logo_photo,
                     bg="#FFFFFF").pack(side=tk.LEFT, padx=(0, 12))
        except Exception:
            _log.debug("ignorado", exc_info=True)

        name_col = tk.Frame(logo_row, bg="#FFFFFF")
        name_col.pack(side=tk.LEFT)

        tk.Label(name_col, text="FenotIT",
                 bg="#FFFFFF", fg="#2166AC",
                 font=("Segoe UI", 30, "bold")).pack(anchor="w")
        tk.Label(name_col, text="Digital Phenotyping Platform",
                 bg="#FFFFFF", fg="#444444",
                 font=("Segoe UI", 10)).pack(anchor="w")

        # Institución
        tk.Label(body,
                 text="Alliance of Bioversity International & CIAT  —  Bean Breeding Program",
                 bg="#FFFFFF", fg="#888888",
                 font=("Segoe UI", 8)).pack(pady=(6, 0))

        # Línea separadora
        tk.Frame(body, bg="#DDDDDD", height=1).pack(fill=tk.X, pady=(16, 8))

        # Mensaje de carga
        self.msg_var = tk.StringVar(value="Cargando módulos de análisis…")
        tk.Label(body, textvariable=self.msg_var,
                 bg="#FFFFFF", fg="#666666",
                 font=("Segoe UI", 8)).pack(anchor="w")

        # Barra de progreso
        prog_bg = tk.Frame(body, bg="#EEEEEE", height=4)
        prog_bg.pack(fill=tk.X, pady=(6, 0))
        self._prog_fill = tk.Frame(prog_bg, bg="#2166AC", height=4, width=0)
        self._prog_fill.place(x=0, y=0, relheight=1)
        self._prog_bg   = prog_bg

        # Versión
        tk.Label(body, text=f"v{__version__}",
                 bg="#FFFFFF", fg="#BBBBBB",
                 font=("Segoe UI", 7)).pack(side=tk.BOTTOM, anchor="e", pady=6)

        # Banda azul inferior
        tk.Frame(self.window, bg="#2166AC", height=4).pack(fill=tk.X, side=tk.BOTTOM)

        self._animate(0, 40)

    def _animate(self, step: int, total: int):
        if self.window is None:
            return
        frac = step / total
        try:
            w = self._prog_bg.winfo_width()
            self._prog_fill.place(x=0, y=0, relheight=1, width=int(w * frac))
        except Exception:
            _log.debug("ignorado", exc_info=True)

        msgs = {
            5:  "Registrando módulo: Morfometría…",
            12: "Registrando módulo: Color KMeans…",
            20: "Registrando módulo: Contador de semillas…",
            30: "Preparando interfaz gráfica…",
            37: "Casi listo…",
        }
        if step in msgs:
            self.msg_var.set(msgs[step])

        if step < total:
            self.window.after(
                self.duration_ms // total,
                lambda: self._animate(step + 1, total)
            )

    def close(self):
        if self.window:
            try:
                self.window.destroy()
            except Exception:
                _log.debug("ignorado", exc_info=True)
            self.window = None
