"""Ventana principal: combina las partes de fenotit/gui/window y fenotit/gui/layout.py y
arma la ventana al iniciar."""
from __future__ import annotations

import dataclasses
import tkinter as tk

from fenotit import APP_NAME, log, settings
from fenotit.gui.layout import LayoutMixin
from fenotit.gui.style import ttk_styles
from fenotit.gui.style.theme import COLORS
from fenotit.gui.window.analyze import AnalyzeMixin
from fenotit.gui.window.areas import AreasMixin
from fenotit.gui.window.bottom import BottomMixin
from fenotit.gui.window.calibration import CalibrationMixin
from fenotit.gui.window.charts import ChartsMixin
from fenotit.gui.window.common import _assets
from fenotit.gui.window.dialogs import DialogsMixin
from fenotit.gui.window.examples import ExamplesMixin
from fenotit.gui.window.export import ExportMixin
from fenotit.gui.window.history import HistoryMixin
from fenotit.gui.window.images import ImagesMixin
from fenotit.gui.window.inspector import InspectorMixin
from fenotit.gui.window.keys import KeysMixin
from fenotit.gui.window.layers import LayersMixin
from fenotit.gui.window.metadata import MetadataMixin
from fenotit.gui.window.model import ModelMixin
from fenotit.gui.window.project import ProjectMixin
from fenotit.gui.window.results import ResultsMixin
from fenotit.gui.window.run import RunMixin
from fenotit.gui.window.segment import SegmentMixin
from fenotit.gui.window.table import TableMixin
from fenotit.gui.window.threshold import ThresholdMixin
from fenotit.gui.window.topbar import TopbarMixin
from fenotit.gui.window.tour import TourMixin
from fenotit.gui.window.viewer import ViewerMixin


_log = log.get("gui.main_window")


class MainWindow(
        ProjectMixin,
        DialogsMixin,
        CalibrationMixin,
        HistoryMixin,
        ImagesMixin,
        RunMixin,
        ResultsMixin,
        LayersMixin,
        ViewerMixin,
        BottomMixin,
        TableMixin,
        ChartsMixin,
        InspectorMixin,
        AreasMixin,
        KeysMixin,
        MetadataMixin,
        ExportMixin,
        TopbarMixin,
        AnalyzeMixin,
        SegmentMixin,
        ThresholdMixin,
        ExamplesMixin,
        ModelMixin,
        TourMixin,
        LayoutMixin,
):
    def __init__(self, root: tk.Tk):
        self.root = root
        self._setup_window()

        self._init_project()
        self._init_images()
        self._init_results()
        self._init_run()
        self._init_analyze()
        self._reset_history()

        self._build_layout()
        self._build_topbar()
        self._build_center_panel()
        self._build_panels()
        self._bind_resize()
        self._bind_shortcuts()
        self._populate_analysis_menu()
        # Al abrir no hay nada que mostrar: los paneles aparecen al agregar imágenes
        # (izquierdo) o al elegir un análisis (derecho).
        self._show_panel(self.left_panel, False)
        self._show_panel(self.right_panel, False)
        self._set_seg_state(dataclasses.asdict(self.project.segmentation))
        self._update_start()
        self._mark_saved()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(20000, self._autosave_tick)
        self.root.after(900, self._offer_recovery)
        self._update_title()
        if not settings.get("tour_seen", False):            # primera vez que se abre la app
            self.root.after(1800, lambda: not self.project.images and self._start_tour())

    def _setup_window(self):
        self.root.title(f"{APP_NAME} — Digital Phenotyping")
        self.root.configure(bg=COLORS["bg"])
        ico = _assets() / "logo.ico"
        if ico.exists():
            try: self.root.iconbitmap(default=str(ico))
            except Exception: _log.debug("ignorado", exc_info=True)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w = min(int(sw * 0.92), 1600)
        h = min(int(sh * 0.88), 960)
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.root.minsize(900, 600)
        ttk_styles.apply(self.root)
