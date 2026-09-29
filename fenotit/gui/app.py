import tkinter as tk
from tkinter import messagebox

from fenotit import i18n, log
from fenotit.gui.splash import SplashScreen
from fenotit.gui.main_window import MainWindow

_log = log.get("gui")


def _report_error(root, exc_type, exc, tb):
    _log.error("Error en la interfaz", exc_info=(exc_type, exc, tb))
    messagebox.showerror(
        i18n.t("common.error"),
        i18n.t("error.unhandled", error=f"{exc_type.__name__}: {exc}", log=log.log_file()),
        parent=root)


def main(project_path: str | None = None):
    lang = i18n.load()
    _log.info("Idioma: %s", lang)
    root = tk.Tk()
    root.report_callback_exception = lambda *a: _report_error(root, *a)
    root.withdraw()

    splash = SplashScreen(root, duration_ms=2000)
    splash.show()
    root.after(2200, lambda: _launch(root, splash, project_path))
    root.mainloop()


def _launch(root: tk.Tk, splash: SplashScreen, project_path: str | None):
    splash.close()
    root.deiconify()
    window = MainWindow(root)
    if project_path:
        root.after(100, lambda: window.open_project_path(project_path))
