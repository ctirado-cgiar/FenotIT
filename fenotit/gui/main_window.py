"""Ventana principal: combina las partes de fenotit/gui/window y fenotit/gui/layout.py y
arma la ventana al iniciar."""
from __future__ import annotations

import dataclasses
import os
import time
import tkinter as tk
from tkinter import ttk

from fenotit import APP_NAME, log, settings
from fenotit.core import batch
from fenotit.core.analysis.registry import AnalysisResult
from fenotit.core.corrections import pipeline as corrections
from fenotit.core.image_io import ImageScaler
from fenotit.gui.canvas.areas import AreaEditor
from fenotit.gui.canvas.zoom import ZoomController
from fenotit.gui.charts.panels import IntraImageChartPanel
from fenotit.gui.layout import LayoutMixin
from fenotit.gui.style.theme import COLORS, FONTS
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


class MainWindow(ProjectMixin, DialogsMixin, CalibrationMixin, HistoryMixin, ImagesMixin, RunMixin, ResultsMixin, LayersMixin, ViewerMixin, BottomMixin, TableMixin, ChartsMixin, InspectorMixin, AreasMixin, KeysMixin, MetadataMixin, ExportMixin, TopbarMixin, AnalyzeMixin, SegmentMixin, ThresholdMixin, ExamplesMixin, ModelMixin, TourMixin, LayoutMixin):
    def __init__(self, root: tk.Tk):
        self.root = root
        self._setup_window()

        self.scaler_left   = ImageScaler()
        self.scaler_right  = ImageScaler()
        self.project = self._blank_project()
        self._corr_info: dict[str, corrections.CorrectionInfo] = {}
        self._saved_state = self.project.to_dict()
        self.current_image_path: str | None = None
        self.last_result:  AnalysisResult | None = None
        self.step_names:   list[str] = []
        self.step_idx:     int = 0
        # Cache de resultados por imagen — para restaurar al navegar
        # resultados por análisis y por foto: al cambiar de análisis no se pierden los otros
        self._results: dict[str, dict[str, AnalysisResult]] = {}
        self._skipped: set[str] = set()          # fotos omitidas (⚠ en la lista)
        self._pending: set[str] = set()          # en cola del lote (… en la lista)
        self._step_names: dict[str, dict[str, list]] = {}
        self.active_analysis: str | None = None
        self._after_resize_id  = None
        self._after_preview_id = None
        self.zoom_ctrl: ZoomController | None = None
        self._zoom_pct_var = tk.StringVar(value="100%")
        self._batch: batch.BatchRunner | None = None
        self._full_paths: list[str] = []      # fotos con imágenes en memoria (las demás, solo datos)
        self._rehydrating = None
        self._rehydrate_job = None
        self._chart_panel: IntraImageChartPanel | None = None
        self._photo_left  = None
        self._photo_right = None
        self.areas: AreaEditor | None = None

        self._build_layout()
        self._build_topbar()
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()
        self._bind_resize()
        self._bind_shortcuts()
        self._populate_analysis_menu()
        # Al abrir no hay nada que mostrar: los paneles aparecen al agregar imágenes
        # (izquierdo) o al elegir un análisis (derecho).
        self._show_panel(self.left_panel, False)
        self._show_panel(self.right_panel, False)
        self._set_seg_state(dataclasses.asdict(self.project.segmentation))
        self._examples = {}
        self._update_start()
        self._saved_state = self._collect_state()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._session = f"{os.getpid()}-{int(time.time())}"
        self._recovery_file = None
        self._last_autosave = time.monotonic()
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
        self._apply_styles()

    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame",      background=COLORS["bg"])
        s.configure("TLabel",      background=COLORS["bg"],
                    foreground=COLORS["text"], font=FONTS["body"])
        s.configure("TButton",     background=COLORS["btn_bg"],
                    foreground=COLORS["accent"], font=FONTS["body"],
                    borderwidth=1, relief="flat", padding=(8, 3))
        s.map("TButton",
              background=[("active", COLORS["btn_hover"])],
              foreground=[("active", COLORS["accent"])])
        s.configure("TSeparator",  background=COLORS["border"])
        for name in ("Horizontal.TProgressbar", "Thin.Horizontal.TProgressbar"):   # barras de avance azules
            s.configure(name, background=COLORS["accent"], troughcolor=COLORS["bg_panel"],
                        bordercolor=COLORS["border"], lightcolor=COLORS["accent"], darkcolor=COLORS["accent"])
        s.configure("TScrollbar",  background=COLORS["bg_panel"],
                    troughcolor=COLORS["bg"], borderwidth=0, arrowsize=12)
        s.configure("TCombobox",
                    fieldbackground=COLORS["bg_card"],
                    background=COLORS["bg_card"],
                    foreground=COLORS["text"],
                    selectbackground=COLORS["accent_light"],
                    selectforeground=COLORS["text"])
        s.configure("Treeview",
                    background=COLORS["bg_card"],
                    fieldbackground=COLORS["bg_card"],
                    foreground=COLORS["text"],
                    font=FONTS["small"], rowheight=20)
        s.configure("Treeview.Heading",
                    background=COLORS["bg_panel"],
                    foreground=COLORS["accent"],
                    font=FONTS["small_bold"], relief="flat")
        s.map("Treeview",
              background=[("selected", COLORS["bg_table_sel"])],
              foreground=[("selected", COLORS["text"])])
