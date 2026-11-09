"""Conversor de modelos: instalar, ver el estado y desinstalar el Python privado que pasa
modelos .pt a ONNX (core/segmentation/converter_setup.py)."""
import threading
import tkinter as tk
from tkinter import ttk

from fenotit import log
from fenotit.gui.dialogs.scale import BaseDialog
from fenotit.gui.style.components import button
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.i18n import t

_log = log.get("gui.converter")


class ConverterDialog(BaseDialog):
    """`on_ready()` se llama cuando el conversor queda instalado."""
    W, H = 500, 400

    def __init__(self, parent, on_ready=None):
        self._on_ready = on_ready
        self._cancel = threading.Event()
        self._busy = False
        super().__init__(parent, t("conv.title"))
        bg, wrap = COLORS["bg_card"], self.W - 70
        from fenotit.core.segmentation import converter_setup as cs
        tk.Label(self.body, text=t("conv.intro", gb=cs.SIZE_GB), bg=bg, fg=COLORS["text"], font=FONTS["small"],
                 justify="left", wraplength=wrap, anchor="w").pack(fill=tk.X)
        tk.Label(self.body, text=t("conv.license"), bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                 justify="left", wraplength=wrap, anchor="w").pack(fill=tk.X, pady=(SPACE["md"], 0))
        self.state = tk.StringVar()
        tk.Label(self.body, textvariable=self.state, bg=bg, fg=COLORS["text"], font=FONTS["small_bold"],
                 anchor="w").pack(fill=tk.X, pady=(SPACE["lg"], SPACE["xs"]))
        self.bar = ttk.Progressbar(self.body, mode="indeterminate", style="Thin.Horizontal.TProgressbar")
        self.line = tk.StringVar()
        self._line = tk.Label(self.body, textvariable=self.line, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                              anchor="w", justify="left", wraplength=wrap)
        self._line.pack(fill=tk.X)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._refresh()

    def _build_buttons(self, parent):
        self._close_btn = button(parent, text=t("common.close"), command=self._close, kind="quiet")
        self._close_btn.pack(side=tk.RIGHT)
        self._main = button(parent, text="", command=self._main_action, kind="primary")
        self._main.pack(side=tk.RIGHT, padx=(0, SPACE["md"]))

    def _refresh(self):
        from fenotit.core.segmentation import converter_setup as cs
        if self._busy:
            self.state.set(t("conv.installing"))
            self._main.config(text=t("common.cancel"))
        elif cs.installed():
            self.state.set(t("conv.ready", mb=f"{cs.size_mb():,.0f}"))
            self._main.config(text=t("conv.uninstall"))
        else:
            self.state.set(t("conv.not_installed"))
            self._main.config(text=t("conv.install"))

    def _main_action(self):
        from fenotit.core.segmentation import converter_setup as cs
        if self._busy:
            self._cancel.set()
            self.line.set(t("conv.cancelling"))
        elif cs.installed():
            from tkinter import messagebox
            if messagebox.askyesno(t("conv.title"), t("conv.uninstall_q"), parent=self):
                cs.uninstall()
                self.line.set("")
                self._refresh()
        else:
            self._start()

    def _start(self):
        from fenotit.core.segmentation import converter_setup as cs
        self._busy = True
        self._cancel.clear()
        self.bar.pack(fill=tk.X, pady=(0, SPACE["sm"]), before=self._line)
        self.bar.start(12)
        self._refresh()

        def post(fn):
            try:
                self.after(0, fn)
            except (tk.TclError, RuntimeError):            # la ventana ya se cerró
                pass

        def work():
            try:
                cs.install(lambda text: post(lambda: self.line.set(text[:160])), self._cancel)
                post(lambda: self._finished(None))
            except cs.Cancelled:
                post(lambda: self._finished(t("conv.cancelled")))
            except Exception as e:
                _log.exception("Instalar conversor")
                msg = t("conv.failed", error=str(e))
                post(lambda: self._finished(msg))
        threading.Thread(target=work, daemon=True).start()

    def _finished(self, error):
        self._busy = False
        self.bar.stop()
        self.bar.pack_forget()
        self.line.set(error or t("conv.done"))
        self._refresh()
        if error is None and self._on_ready:
            self._on_ready()

    def _close(self):
        if self._busy:
            from tkinter import messagebox
            if not messagebox.askyesno(t("conv.title"), t("conv.close_q"), parent=self):
                return
            self._cancel.set()
        self.destroy()
