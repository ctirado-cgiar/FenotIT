import tkinter as tk
from tkinter import messagebox

from fenotit import log
from fenotit.gui.splash import SplashScreen
from fenotit.gui.main_window import MainWindow

_log = log.get("gui")


def _report_error(root, exc_type, exc, tb):
    _log.error("Error en la interfaz", exc_info=(exc_type, exc, tb))
    messagebox.showerror(
        "Error",
        f"{exc_type.__name__}: {exc}\n\nDetalle guardado en:\n{log.log_file()}",
        parent=root)


def main():
    root = tk.Tk()
    root.report_callback_exception = lambda *a: _report_error(root, *a)
    root.withdraw()

    splash = SplashScreen(root, duration_ms=2000)
    splash.show()
    root.after(2200, lambda: _launch(root, splash))
    root.mainloop()


def _launch(root: tk.Tk, splash: SplashScreen):
    splash.close()
    root.deiconify()
    MainWindow(root)
