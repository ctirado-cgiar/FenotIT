import tkinter as tk
from tkinter import messagebox

from fenotit import i18n, log
from fenotit.gui.splash import SplashScreen

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

    splash = SplashScreen(root)
    splash.show()

    def build():
        from fenotit.gui.main_window import MainWindow
        return MainWindow(root)

    t = i18n.t
    window = splash.load([
        (0.30, t("splash.images"), ("numpy", "cv2")),
        (0.50, t("splash.analysis"), ("scipy.ndimage", "scipy.spatial", "sklearn.cluster")),
        (0.68, t("splash.charts"), ("matplotlib.pyplot", "openpyxl")),
        (0.80, t("splash.modules"), ("fenotit.gui.main_window",)),
        (0.95, t("splash.ui"), build),
    ])

    def show():
        root.deiconify()
        if project_path:
            root.after(100, lambda: window.open_project_path(project_path))
    splash.finish(show)
    root.mainloop()
