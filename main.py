"""
main.py
Punto de entrada de FenotIT.
"""

import tkinter as tk
from tkinter import simpledialog  # necesario para calibración en main_window
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ui.splash import SplashScreen
from ui.main_window import MainWindow


def main():
    root = tk.Tk()
    root.withdraw()

    splash = SplashScreen(root, duration_ms=2000)
    splash.show()

    # Programar lanzamiento desde root directamente, no desde splash.window
    root.after(2200, lambda: _launch(root, splash))
    root.mainloop()


def _launch(root: tk.Tk, splash: SplashScreen):
    # Cerrar splash si aún está abierto
    splash.close()
    root.deiconify()
    app = MainWindow(root)  # noqa: F841


if __name__ == "__main__":
    main()
