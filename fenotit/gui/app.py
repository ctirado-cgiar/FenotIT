import tkinter as tk

from fenotit.gui.splash import SplashScreen
from fenotit.gui.main_window import MainWindow


def main():
    root = tk.Tk()
    root.withdraw()

    splash = SplashScreen(root, duration_ms=2000)
    splash.show()
    root.after(2200, lambda: _launch(root, splash))
    root.mainloop()


def _launch(root: tk.Tk, splash: SplashScreen):
    splash.close()
    root.deiconify()
    MainWindow(root)
