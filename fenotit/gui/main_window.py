"""
ui/main_window.py  —  FenotIT v1.3
Cambios vs v1.2:
  - Preview en tiempo real: al mover sliders se superpone la máscara
    sobre el canvas izquierdo (objetos resaltados en verde)
  - Paneles izquierdo y derecho colapsables con botón ◀/▶
  - PanedWindow para redimensionar arrastrando el separador
  - Tooltips del config_panel ya no se salen de pantalla (fix en config_panel.py)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import platform
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from fenotit.core.image_io import ImageScaler, load_image
from fenotit.core.project import IMAGE_EXTS, PROJECT_EXT, Project, Scale, Segmentation
from fenotit.gui.roi.selectors import ROISelector
from fenotit.core.analysis.registry import ANALYSES, AnalysisResult
from fenotit.core.export.exporter import Exporter
from fenotit.gui.config_panel import ConfigPanel
from fenotit.gui.zoom_controller import ZoomController
from fenotit.gui.calibration_dialogs import ScaleDialog
from fenotit.gui.corrections_dialog import CorrectionsDialog
from fenotit.core.corrections import pipeline as corrections
from fenotit.gui.export_dialog import ExportDialog
from fenotit.gui.charts import IntraImageChartPanel, BatchChartWindow
from fenotit.core.export.exporter import Exporter, quick_export_csv
from fenotit.gui.theme import COLORS, FONTS

from fenotit import APP_NAME, __version__, log
from fenotit.i18n import t
from fenotit import i18n

_log = log.get("gui.main_window")

CHANNEL_NAMES = {
    "BGR":   ["B", "G", "R"],
    "HSV":   ["H", "S", "V"],
    "LAB":   ["L", "A", "B"],
    "YCrCb": ["Y", "Cr", "Cb"],
    "HLS":   ["H", "L", "S"],
    "XYZ":   ["X", "Y", "Z"],
    "YUV":   ["Y", "U", "V"],
    "LUV":   ["L", "U", "V"],
}

CV2_CODES = {
    "HSV":   cv2.COLOR_BGR2HSV,
    "LAB":   cv2.COLOR_BGR2LAB,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
    "LUV":   cv2.COLOR_BGR2LUV,
    "HLS":   cv2.COLOR_BGR2HLS,
    "XYZ":   cv2.COLOR_BGR2XYZ,
    "YUV":   cv2.COLOR_BGR2YUV,
}


def _analysis_key(name: str) -> str:
    return ANALYSES[name].func.__module__.rsplit(".", 1)[-1]


def _analysis_label(name: str) -> str:
    return t(f"analysis.{_analysis_key(name)}.name", name)


def _analysis_name(key: str | None) -> str | None:
    return next((n for n in ANALYSES if _analysis_key(n) == key), None)


def _assets() -> Path:
    return Path(__file__).parent.parent / "assets"


# ── DropMenu (igual que v1.2) ─────────────────────────────────────────────────

class DropMenu(tk.Frame):
    def __init__(self, parent, text, items, colors, **kw):
        super().__init__(parent, bg=colors["bg_topbar"], **kw)
        self.colors = colors
        self.items  = items
        self._popup = None
        self.btn = tk.Button(
            self, text=f"  {text}  ▾",
            bg=colors["bg_topbar"], fg="#FFFFFF",
            font=FONTS["body"], relief="flat", bd=0,
            activebackground="#1a5490",
            activeforeground="#FFFFFF",
            cursor="hand2", command=self._toggle)
        self.btn.pack()

    def _toggle(self):
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
            self._popup = None
        else:
            self._open()

    def _open(self):
        self.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        popup = tk.Toplevel(self)
        popup.wm_overrideredirect(True)
        popup.configure(bg=self.colors["border"])
        popup.wm_geometry(f"+{x}+{y}")
        popup.attributes("-topmost", True)
        self._popup = popup

        inner = tk.Frame(popup, bg=self.colors["bg_card"], padx=1, pady=1)
        inner.pack(padx=1, pady=1)

        for item in self.items:
            if item is None:
                tk.Frame(inner, bg=self.colors["border"],
                         height=1).pack(fill=tk.X, pady=2)
                continue
            label, cmd, *rest = item
            disabled = rest[0] if rest else False
            fg = self.colors["text_muted"] if disabled else self.colors["text"]
            row = tk.Frame(inner, bg=self.colors["bg_card"])
            row.pack(fill=tk.X)
            lbl = tk.Label(row, text=f"  {label.strip()}   ",
                           bg=self.colors["bg_card"], fg=fg,
                           font=FONTS["body"], anchor="w", pady=4, padx=4)
            lbl.pack(fill=tk.X)
            if disabled:
                tk.Label(row, text=t("common.wip_badge"),
                         bg="#FFF3CD", fg="#856404",
                         font=("Segoe UI", 7),
                         padx=4).place(relx=1.0, rely=0.5,
                                       anchor="e", x=-8)
            if not disabled and cmd:
                def _bind(w, c=cmd):
                    def enter(e):
                        w.config(bg=self.colors["accent_light"])
                        for ch in w.winfo_children():
                            try: ch.config(bg=self.colors["accent_light"])
                            except Exception: _log.debug("ignorado", exc_info=True)
                    def leave(e):
                        w.config(bg=self.colors["bg_card"])
                        for ch in w.winfo_children():
                            try: ch.config(bg=self.colors["bg_card"])
                            except Exception: _log.debug("ignorado", exc_info=True)
                    def click(e):
                        popup.destroy()
                        c()
                    for widget in [w] + list(w.winfo_children()):
                        widget.bind("<Enter>", enter)
                        widget.bind("<Leave>", leave)
                        widget.bind("<Button-1>", click)
                _bind(row)

        popup.bind("<FocusOut>", lambda e: popup.destroy())
        popup.focus_set()


# ── Panel colapsable ──────────────────────────────────────────────────────────

class CollapsiblePanel(tk.Frame):
    """
    Panel lateral que se puede colapsar/expandir con un botón.
    Cuando está colapsado solo muestra una tira con el botón.
    """
    COLLAPSED_W = 18

    def __init__(self, parent, side: str, title: str,
                 colors: dict, default_width: int = 200, **kw):
        super().__init__(parent, bg=colors["bg_panel"], **kw)
        self.colors        = colors
        self.side          = side          # "left" o "right"
        self.title         = title
        self.default_width = default_width
        self._expanded     = True

        # Botón de colapsar (tira lateral)
        self._strip = tk.Frame(self, bg=colors["bg_panel"],
                               width=self.COLLAPSED_W)
        self._strip.pack(
            side=tk.RIGHT if side == "left" else tk.LEFT,
            fill=tk.Y)
        self._strip.pack_propagate(False)

        arrow = "◀" if side == "left" else "▶"
        self._toggle_btn = tk.Label(
            self._strip, text=arrow,
            bg=colors["bg_panel"], fg=colors["accent"],
            font=("Segoe UI", 9), cursor="hand2")
        self._toggle_btn.place(relx=0.5, rely=0.5, anchor="center")
        self._toggle_btn.bind("<Button-1>", lambda e: self.toggle())

        # Contenedor del contenido
        self.content = tk.Frame(self, bg=colors["bg_panel"])
        self.content.pack(
            side=tk.LEFT if side == "left" else tk.RIGHT,
            fill=tk.BOTH, expand=True)

        self.configure(width=default_width)

    def toggle(self):
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def collapse(self):
        self._expanded = False
        self.content.pack_forget()
        self.configure(width=self.COLLAPSED_W)
        arrow = "▶" if self.side == "left" else "◀"
        self._toggle_btn.config(text=arrow)

    def expand(self):
        self._expanded = True
        self.content.pack(
            side=tk.LEFT if self.side == "left" else tk.RIGHT,
            fill=tk.BOTH, expand=True)
        self.configure(width=self.default_width)
        arrow = "◀" if self.side == "left" else "▶"
        self._toggle_btn.config(text=arrow)


# ── Ventana principal ─────────────────────────────────────────────────────────

class MainWindow:

    def __init__(self, root: tk.Tk):
        self.root = root
        self._setup_window()

        self.scaler_left   = ImageScaler()
        self.scaler_right  = ImageScaler()
        self.project = Project(name=t("project.untitled"))
        self._corr_info: dict[str, corrections.CorrectionInfo] = {}
        self._saved_state = self.project.to_dict()
        self.current_image_path: str | None = None
        self.last_result:  AnalysisResult | None = None
        self.step_names:   list[str] = []
        self.step_idx:     int = 0
        self.step_history: dict[str, list] = {}
        # Cache de resultados por imagen — para restaurar al navegar
        self.results_cache: dict[str, AnalysisResult] = {}
        self.step_names_cache: dict[str, list] = {}
        self.output_root:  str | None = None
        self.active_analysis: str | None = None
        self._after_resize_id  = None
        self._after_preview_id = None
        self.zoom_ctrl: ZoomController | None = None
        self._zoom_pct_var = tk.StringVar(value="100%")
        # Exporter acumula resultados en sesión
        self._exporter: Exporter | None = None
        self._chart_panel: IntraImageChartPanel | None = None
        # Resultados por análisis para gráficos combinados
        self.all_results_by_analysis: dict[str, dict] = {}
        self._photo_left  = None
        self._photo_right = None
        self.roi_selector: ROISelector | None = None

        self._build_layout()
        self._build_topbar()
        self._build_controls_bar()
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()
        self._bind_resize()
        self._populate_analysis_menu()
        self._saved_state = self._collect_state()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_title()

    # ── Estado del proyecto ───────────────────────────────────────────────────

    @property
    def batch_paths(self) -> list[str]:
        return [str(p) for p in self.project.images]

    @property
    def batch_index(self) -> int:
        return self.project.current_index

    @batch_index.setter
    def batch_index(self, value: int):
        self.project.current_index = value

    @property
    def mm_per_pixel(self) -> float | None:
        return self._scale_for(self.current_image_path)

    def _scale_for(self, path: str | None) -> float | None:
        if self.project.scale.source == "aruco":
            info = self._corr_info.get(path) if path else None
            return info.mm_per_px if info else None
        return self.project.scale.mm_per_pixel

    def _load_corrected(self, path: str):
        img = load_image(path)
        if img is None:
            return None, None
        img, info = corrections.apply(img, self.project.corrections)
        self._corr_info[path] = info
        return img, info

    def _load_corrected_image(self, path: str):
        return self._load_corrected(path)[0]

    def _open_corrections(self):
        raw = load_image(self.current_image_path) if self.current_image_path else None
        CorrectionsDialog(self.root, self.project.corrections, raw,
                          aruco_scale=self.project.scale.source == "aruco",
                          on_apply=self._on_corrections_applied)

    def _on_corrections_applied(self, corr, use_aruco_scale: bool):
        self.project.corrections = corr
        if use_aruco_scale:
            self.project.scale = Scale(None, "aruco")
        elif self.project.scale.source == "aruco":
            self.project.scale = Scale()
        self._invalidate_results()
        if self.current_image_path:
            self._load_single(self.current_image_path)
        self._update_corr_indicator()

    def _invalidate_results(self):
        self._corr_info.clear()
        self.results_cache.clear()
        self.step_names_cache.clear()
        self.all_results_by_analysis.clear()
        self.last_result = None
        self.canvas_right.delete("all")

    def _clear_scale(self):
        self.project.scale = Scale()
        self._set_status(t("status.scale_cleared"))
        self._update_corr_indicator()

    def _update_corr_indicator(self):
        parts = [t(f"corr.short.{n}") for n in self.project.corrections.active()]
        mm = self.mm_per_pixel
        if self.project.scale.source != "none":
            parts.append(f"{mm:.4f} mm/px" if mm else t("corr.short.scale_pending"))
        info = self._corr_info.get(self.current_image_path) if self.current_image_path else None
        warn = bool(info and info.warnings)
        text = ("⚠ " if warn else "") + "  ·  ".join(parts)
        self.corr_var.set(text)
        self._corr_label.config(fg=COLORS["warning"] if warn else COLORS["accent"])

    def _store_panel_params(self):
        panel = getattr(self, "config_panel", None)
        if panel is not None and self.active_analysis in ANALYSES:
            self.project.params[_analysis_key(self.active_analysis)] = panel.get_values(warn=False)

    def _collect_state(self) -> dict:
        self._store_panel_params()
        self.project.mode = self.mode_var.get()
        self.project.segmentation = Segmentation(
            self.cs_var.get(), int(self.ch_var.get()),
            int(self.min_slider.get()), int(self.max_slider.get()))
        if self.roi_selector:
            self.project.roi = self.roi_selector.to_dict()
        return self.project.to_dict()

    def _is_dirty(self) -> bool:
        return self._collect_state() != self._saved_state

    def _update_title(self):
        mark = " *" if self._is_dirty() else ""
        self.root.title(f"{self.project.name}{mark} — {APP_NAME}")
        self.root.after(1000, self._update_title)

    def _confirm_discard(self) -> bool:
        if not self._is_dirty():
            return True
        ans = messagebox.askyesnocancel(
            t("project.unsaved_title"),
            t("project.unsaved_msg", name=self.project.name), parent=self.root)
        if ans is None:
            return False
        return self._save_project() if ans else True

    def _save_project(self) -> bool:
        if self.project.folder is None:
            return self._save_project_as()
        self._collect_state()
        try:
            self.project.save()
        except Exception as e:
            _log.exception("Error guardando proyecto")
            messagebox.showerror(t("common.error"), t("project.save_error", error=e), parent=self.root)
            return False
        self._saved_state = self.project.to_dict()
        self._set_status(t("project.saved", path=self.project.file))
        return True

    def _save_project_as(self) -> bool:
        folder = filedialog.askdirectory(
            title=t("project.folder_title", ext=PROJECT_EXT), mustexist=False)
        if not folder:
            return False
        self.project.folder = Path(folder)
        self.project.name = Path(folder).name
        return self._save_project()

    def _new_project(self):
        if not self._confirm_discard():
            return
        self._apply_project(Project(name=t("project.untitled")))

    def _open_project(self):
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(
            title=t("project.open_title"),
            filetypes=[(t("project.filetype"), f"*{PROJECT_EXT}"), (t("common.all_files"), "*.*")])
        if path:
            self.open_project_path(path)

    def open_project_path(self, path):
        try:
            project = Project.load(path)
        except Exception as e:
            _log.exception("Error abriendo proyecto %s", path)
            messagebox.showerror(t("common.error"), t("project.open_error", error=e), parent=self.root)
            return
        missing = project.missing_images()
        if missing:
            messagebox.showwarning(
                t("project.missing_title"),
                t("project.missing_msg", n=len(missing), total=len(project.images)) + "\n"
                + "\n".join(str(m) for m in missing[:5])
                + ("\n…" if len(missing) > 5 else ""), parent=self.root)
            project.images = [p for p in project.images if p.exists()]
            project.current_index = min(project.current_index, max(len(project.images) - 1, 0))
        self._apply_project(project)
        self._set_status(t("project.opened", path=project.file))

    def _apply_project(self, project: Project):
        self.config_panel = None
        self.project = project
        seg = project.segmentation
        self.cs_var.set(seg.color_space)
        self._update_channel_names()
        self.ch_var.set(seg.channel)
        self.min_slider.set(seg.min_val)
        self.max_slider.set(seg.max_val)
        self.mode_var.set(project.mode)

        name = _analysis_name(project.analysis) or self._selected_analysis()
        self.analysis_var.set(_analysis_label(name))
        self._on_analysis_selected(None)

        self.results_cache.clear()
        self.step_names_cache.clear()
        self.all_results_by_analysis.clear()
        self._corr_info.clear()
        self._exporter = None
        self.output_root = None
        self.canvas_right.delete("all")
        if self.roi_selector:
            self.roi_selector.restore_when_ready(project.roi if project.current_image else None)
        self._update_batch_list()
        if project.current_image:
            if project.mode == "batch":
                self.output_root = str(project.images[0].parent)
                self._exporter = Exporter(self.output_root)
            self._load_single(str(project.current_image))
            self._sync_listbox()
        else:
            self.current_image_path = None
            self.canvas_left.delete("all")
        self._update_corr_indicator()
        self._saved_state = self._collect_state()

    def _on_close(self):
        if self._confirm_discard():
            self.root.destroy()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_window(self):
        self.root.title(f"{APP_NAME} — Digital Phenotyping")
        self.root.configure(bg=COLORS["bg"])
        ico = _assets() / "logo.ico"
        if ico.exists():
            try: self.root.iconbitmap(str(ico))
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
                    font=("Segoe UI", 8, "bold"), relief="flat")
        s.map("Treeview",
              background=[("selected", COLORS["bg_table_sel"])],
              foreground=[("selected", COLORS["text"])])

    # ── Layout con PanedWindow ────────────────────────────────────────────────

    def _build_layout(self):
        self.topbar = tk.Frame(self.root, bg=COLORS["bg_topbar"], height=40)
        self.topbar.pack(fill=tk.X, side=tk.TOP)
        self.topbar.pack_propagate(False)

        self.controls_bar = tk.Frame(self.root, bg=COLORS["bg_panel"],
                                     height=36)
        self.controls_bar.pack(fill=tk.X, side=tk.TOP)
        self.controls_bar.pack_propagate(False)
        tk.Frame(self.controls_bar, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, side=tk.BOTTOM)

        # PanedWindow horizontal para los 3 paneles
        self.paned = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bg=COLORS["border"],
            sashwidth=5,
            sashrelief="flat",
            sashpad=0,
            showhandle=False,
            opaqueresize=True)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo colapsable
        self.left_panel = CollapsiblePanel(
            self.paned, side="left",
            title=t("panel.left"), colors=COLORS,
            default_width=200)
        self.paned.add(self.left_panel, minsize=CollapsiblePanel.COLLAPSED_W,
                       width=200)

        # Centro
        self.center_frame = tk.Frame(self.paned, bg=COLORS["bg"])
        self.paned.add(self.center_frame, minsize=400, stretch="always")

        # Panel derecho colapsable
        self.right_panel = CollapsiblePanel(
            self.paned, side="right",
            title=t("panel.right"), colors=COLORS,
            default_width=260)
        self.paned.add(self.right_panel, minsize=CollapsiblePanel.COLLAPSED_W,
                       width=260)

        # Statusbar
        self.statusbar = tk.Frame(self.root, bg=COLORS["bg_panel"], height=24)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)
        self.statusbar.pack_propagate(False)
        tk.Frame(self.statusbar, bg=COLORS["border"],
                 height=1).pack(fill=tk.X)
        self.status_var = tk.StringVar(value=t("status.ready"))
        self.corr_var = tk.StringVar(value="")
        self._corr_label = tk.Label(self.statusbar, textvariable=self.corr_var,
                                    bg=COLORS["bg_panel"], fg=COLORS["accent"],
                                    font=FONTS["small"], padx=10)
        self._corr_label.pack(side=tk.RIGHT)
        tk.Label(self.statusbar, textvariable=self.status_var,
                 bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                 font=FONTS["small"], anchor="w",
                 padx=10).pack(fill=tk.X)

    # ── Topbar ────────────────────────────────────────────────────────────────

    def _build_topbar(self):
        try:
            ico  = _assets() / "logo.ico"
            img  = Image.open(str(ico)).resize((26, 26), Image.LANCZOS)
            self._topbar_logo = ImageTk.PhotoImage(img)
            tk.Label(self.topbar, image=self._topbar_logo,
                     bg=COLORS["bg_topbar"]).pack(
                         side=tk.LEFT, padx=(10, 2), pady=6)
        except Exception:
            _log.debug("ignorado", exc_info=True)

        tk.Label(self.topbar, text=APP_NAME,
                 bg=COLORS["bg_topbar"], fg="#FFFFFF",
                 font=("Segoe UI", 12, "bold")).pack(
                     side=tk.LEFT, padx=(0, 8))
        self._vdiv()

        # Archivo
        self._drop(self.topbar, t("menu.file"), [
            (t("menu.new_project"),      self._new_project),
            (t("menu.open_project"),     self._open_project),
            (t("menu.save_project"),    self._save_project),
            (t("menu.save_project_as"), self._save_project_as),
            None,
            (t("menu.load_image"),       self._open_image),
            (t("menu.load_folder"),      self._open_folder),
            None,
            (t("menu.export_image"),     self._export_results),
            (t("menu.export_batch"),       self._export_batch),
            None,
            (t("menu.exit"),              self._on_close),
        ])

        # Configuración
        self._drop(self.topbar, t("menu.settings"), [
            (t("menu.corrections"), self._open_corrections),
            None,
            (t("menu.cal_scale"),
             lambda: ScaleDialog(
                 self.root,
                 current_image=self.scaler_left.original
                     if self.scaler_left.has_image else None,
                 current_path=self.current_image_path,
                 on_scale_set=self._on_scale_set,
                 loader=self._load_corrected_image)),
            (t("menu.scale_manual"), self._calibrate_scale),
            (t("menu.scale_clear"), self._clear_scale),
            None,
            (t("menu.export_prefs"),
             lambda: self._wip(t("menu.export_prefs"), t("wip.export_prefs")), True),
            (t("menu.language"), self._choose_language),
            None,
            (t("menu.about"),      self._about),
        ])

        # ROI
        self._drop(self.topbar, t("menu.roi"), [
            (t("roi.menu.rect"),
             lambda: self._set_roi_mode("rectángulo")),
            (t("roi.menu.square"),
             lambda: self._set_roi_mode("cuadrado")),
            (t("roi.menu.polygon"),
             lambda: self._set_roi_mode("polígono")),
            (t("roi.menu.hole"),
             lambda: self._set_roi_mode("hueco")),
            None,
            (t("roi.menu.exclusion"),
             lambda: self._set_roi_mode("exclusión")),
            (t("roi.menu.exclusion_color"),
             self._pick_exclusion_color),
            (t("roi.menu.clear_exclusions"),
             self._clear_exclusions),
            None,
            (t("roi.menu.ai"),    self._ai_segmentation),
            None,
            (t("roi.menu.clear_all"),
             lambda: self.roi_selector.clear()
             if self.roi_selector else None),
        ])

        self._vdiv()

        # Análisis
        tk.Label(self.topbar, text=t("topbar.analysis"),
                 bg=COLORS["bg_topbar"], fg="#CCCCCC",
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(6, 2))
        self.analysis_var = tk.StringVar()
        self.analysis_combo = ttk.Combobox(
            self.topbar, textvariable=self.analysis_var,
            state="readonly", width=20, font=FONTS["body"])
        self.analysis_combo.pack(side=tk.LEFT, padx=4, pady=8)
        self.analysis_combo.bind("<<ComboboxSelected>>",
                                 self._on_analysis_selected)

        tk.Button(self.topbar, text=t("topbar.run"),
                  command=self._run_analysis,
                  bg="#FFFFFF", fg=COLORS["accent"],
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", bd=0,
                  activebackground=COLORS["accent_light"],
                  activeforeground=COLORS["accent"],
                  padx=10, pady=2,
                  cursor="hand2").pack(side=tk.LEFT, padx=6, pady=8)

        self._vdiv()

        tk.Label(self.topbar, text=t("topbar.mode"),
                 bg=COLORS["bg_topbar"], fg="#CCCCCC",
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(8, 2))
        self.mode_var = tk.StringVar(value="individual")
        for val, lbl in [("individual", t("mode.individual")), ("batch", t("mode.batch"))]:
            tk.Radiobutton(
                self.topbar, text=lbl,
                variable=self.mode_var, value=val,
                bg=COLORS["bg_topbar"], fg="#FFFFFF",
                selectcolor=COLORS["bg_topbar"],
                activebackground=COLORS["bg_topbar"],
                font=FONTS["small"]).pack(side=tk.LEFT, padx=2)

    def _vdiv(self):
        tk.Frame(self.topbar, bg="#4a8fd4", width=1).pack(
            side=tk.LEFT, fill=tk.Y, pady=6, padx=4)

    def _drop(self, parent, text, items):
        DropMenu(parent, text, items, COLORS).pack(side=tk.LEFT)

    # ── Barra de controles compacta ───────────────────────────────────────────

    def _build_controls_bar(self):
        bar = self.controls_bar

        tk.Label(bar, text=t("controls.space"),
                 bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(10, 2))

        self.cs_var = tk.StringVar(value="BGR")
        cs_combo = ttk.Combobox(bar, textvariable=self.cs_var,
                                values=list(CHANNEL_NAMES.keys()),
                                state="readonly", width=7,
                                font=FONTS["small"])
        cs_combo.pack(side=tk.LEFT, padx=(0, 8), pady=4)
        cs_combo.bind("<<ComboboxSelected>>", self._on_cs_change)

        tk.Label(bar, text=t("controls.channel"),
                 bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(0, 4))

        self.ch_var = tk.IntVar(value=0)
        self._ch_radios = []
        for i in range(3):
            rb = tk.Radiobutton(
                bar, text="",
                variable=self.ch_var, value=i,
                command=self._on_slider_change,
                bg=COLORS["bg_panel"], fg=COLORS["text"],
                selectcolor=COLORS["bg_card"],
                activebackground=COLORS["bg_panel"],
                font=FONTS["small"])
            rb.pack(side=tk.LEFT, padx=1)
            self._ch_radios.append(rb)

        self._update_channel_names()

        tk.Frame(bar, bg=COLORS["border"], width=1).pack(
            side=tk.LEFT, fill=tk.Y, pady=6, padx=8)

        tk.Label(bar, text="Min:",
                 bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(0, 2))
        self.min_slider = tk.Scale(
            bar, from_=0, to=255, orient=tk.HORIZONTAL,
            bg=COLORS["bg_panel"], fg=COLORS["text"],
            troughcolor=COLORS["slider_trough"],
            activebackground=COLORS["accent"],
            highlightthickness=0, showvalue=True,
            font=FONTS["small"], length=120,
            command=lambda _: self._on_slider_change())
        self.min_slider.set(0)
        self.min_slider.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(bar, text="Max:",
                 bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(0, 2))
        self.max_slider = tk.Scale(
            bar, from_=0, to=255, orient=tk.HORIZONTAL,
            bg=COLORS["bg_panel"], fg=COLORS["text"],
            troughcolor=COLORS["slider_trough"],
            activebackground=COLORS["accent"],
            highlightthickness=0, showvalue=True,
            font=FONTS["small"], length=120,
            command=lambda _: self._on_slider_change())
        self.max_slider.set(255)
        self.max_slider.pack(side=tk.LEFT)

        # ── Controles de zoom ─────────────────────────────────────────────
        tk.Frame(bar, bg=COLORS["border"], width=1).pack(
            side=tk.LEFT, fill=tk.Y, pady=6, padx=6)

        # Guardar referencias a los botones para poder actualizarlos
        # Los comandos usan self.do_zoom_* que son métodos reales
        for label, method_name in [
            ("−",  "do_zoom_out"),
            ("+",  "do_zoom_in"),
            ("⊡",  "do_zoom_fit"),
        ]:
            btn = tk.Button(bar, text=label,
                            command=lambda m=method_name: getattr(self, m)(),
                            bg=COLORS["btn_bg"], fg=COLORS["accent"],
                            relief="flat", font=("Segoe UI", 10, "bold"),
                            width=2, cursor="hand2")
            btn.pack(side=tk.LEFT, padx=1)
            btn.bind("<Enter>",
                     lambda e, b=btn: b.config(bg=COLORS["btn_hover"]))
            btn.bind("<Leave>",
                     lambda e, b=btn: b.config(bg=COLORS["btn_bg"]))

        tk.Label(bar, textvariable=self._zoom_pct_var,
                 bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                 font=FONTS["small"], width=5).pack(
                     side=tk.LEFT, padx=2)



        # Indicador de modo preview
        self.preview_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.preview_var,
                 bg=COLORS["bg_panel"], fg=COLORS["accent2"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=8)

    def _update_channel_names(self):
        cs    = self.cs_var.get()
        names = CHANNEL_NAMES.get(cs, ["0","1","2"])
        for i, rb in enumerate(self._ch_radios):
            rb.config(text=names[i])

    def _on_cs_change(self, _=None):
        self._update_channel_names()
        self.ch_var.set(0)
        self._on_slider_change()

    # ── Preview en tiempo real ────────────────────────────────────────────────

    def _on_slider_change(self):
        """
        Llamado cada vez que cambia un slider, canal o espacio de color.
        Debounce de 60ms para no saturar mientras se arrastra.
        """
        if self._after_preview_id:
            self.root.after_cancel(self._after_preview_id)
        self._after_preview_id = self.root.after(60, self._show_preview)

    def _show_preview(self):
        """
        Muestra en el canvas izquierdo la imagen original con la máscara
        superpuesta en verde semitransparente, respetando el zoom actual.
        """
        if not self.scaler_left.has_image or self.zoom_ctrl is None:
            return

        img_original = self.scaler_left.original
        cs_code = CV2_CODES.get(self.cs_var.get())
        ch_idx  = self.ch_var.get()
        min_val = self.min_slider.get()
        max_val = self.max_slider.get()

        try:
            if cs_code is not None:
                converted = cv2.cvtColor(img_original, cs_code)
            else:
                converted = img_original

            channel = converted[:, :, min(ch_idx, converted.shape[2]-1)] \
                      if converted.ndim == 3 else converted

            _, t_min = cv2.threshold(channel, min_val-1, 255,
                                     cv2.THRESH_BINARY)
            _, t_max = cv2.threshold(channel, max_val,   255,
                                     cv2.THRESH_BINARY_INV)
            mask = cv2.bitwise_and(t_min, t_max)

            # Aplicar ROI (en coords originales)
            roi = self.roi_selector.mask if self.roi_selector else None
            if roi is not None:
                roi_full = self.scaler_left.scale_mask_to_original(roi)
                if roi_full.shape[:2] != mask.shape[:2]:
                    roi_full = cv2.resize(
                        roi_full,
                        (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
                mask = cv2.bitwise_and(mask, mask, mask=roi_full)

            # Superposición verde sobre imagen original
            overlay = img_original.copy()
            overlay[mask > 0] = (
                overlay[mask > 0] * 0.5 +
                np.array([0, 180, 80], dtype=np.float32) * 0.5
            ).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 160, 60), 1)

            # Pasar al zoom controller — renderiza con zoom actual
            self.zoom_ctrl.redraw_with_overlay(overlay_left=overlay)

            n_px = int(np.sum(mask > 0))
            pct  = round(n_px / mask.size * 100, 1)
            self.preview_var.set(t("status.preview", px=f"{n_px:,}", pct=pct))

        except Exception:
            _log.debug("Preview falló", exc_info=True)
            if self.zoom_ctrl:
                self.zoom_ctrl.redraw_with_overlay(overlay_left=img_original)
            self.preview_var.set("")

    # ── Panel izquierdo ───────────────────────────────────────────────────────

    def _build_left_panel(self):
        lf = self.left_panel.content

        self._section_lbl(lf, t("left.workflow"))
        self.step_labels = []
        for num, name in [("①", t("step.corrections")), ("②", t("step.load")),
                          ("③", t("step.roi")), ("④", t("step.analysis")), ("⑤", t("step.export"))]:
            f = tk.Frame(lf, bg=COLORS["bg_panel"])
            f.pack(fill=tk.X, padx=10, pady=1)
            tk.Label(f, text=num, bg=COLORS["bg_panel"],
                     fg=COLORS["accent"],
                     font=("Segoe UI", 9, "bold"),
                     width=2).pack(side=tk.LEFT)
            lbl = tk.Label(f, text=name, bg=COLORS["bg_panel"],
                           fg=COLORS["text_muted"], font=FONTS["body"],
                           anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.step_labels.append(lbl)

        self._divider(lf)
        self._section_lbl(lf, t("left.history"))
        self.history_frame = tk.Frame(lf, bg=COLORS["bg_panel"])
        self.history_frame.pack(fill=tk.X, padx=6)

        self._divider(lf)
        self._section_lbl(lf, t("left.batch"))

        nav = tk.Frame(lf, bg=COLORS["bg_panel"])
        nav.pack(fill=tk.X, padx=10, pady=2)
        tk.Button(nav, text="◀", command=self._prev_image,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["small"],
                  width=2, cursor="hand2").pack(side=tk.LEFT)
        self.batch_label = tk.Label(nav, text="—",
                                    bg=COLORS["bg_panel"],
                                    fg=COLORS["text_muted"],
                                    font=FONTS["small"])
        self.batch_label.pack(side=tk.LEFT, expand=True)
        tk.Button(nav, text="▶", command=self._next_image,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["small"],
                  width=2, cursor="hand2").pack(side=tk.RIGHT)

        list_f = tk.Frame(lf, bg=COLORS["bg_panel"])
        list_f.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        sb = ttk.Scrollbar(list_f, orient="vertical")
        self.batch_listbox = tk.Listbox(
            list_f, bg=COLORS["bg_card"], fg=COLORS["text"],
            font=FONTS["small"],
            selectbackground=COLORS["accent"],
            selectforeground="#FFFFFF",
            borderwidth=1, relief="flat",
            highlightthickness=1,
            highlightcolor=COLORS["border"],
            highlightbackground=COLORS["border"],
            yscrollcommand=sb.set)
        sb.config(command=self.batch_listbox.yview)
        self.batch_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.batch_listbox.bind("<<ListboxSelect>>", self._on_batch_select)

    # ── Panel central ─────────────────────────────────────────────────────────

    def _build_center_panel(self):
        cf = self.center_frame

        lbl_row = tk.Frame(cf, bg=COLORS["bg"])
        lbl_row.pack(fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(lbl_row, text=t("center.input"),
                 bg=COLORS["bg"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT, expand=True)
        tk.Label(lbl_row, text=t("center.result"),
                 bg=COLORS["bg"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.RIGHT, expand=True)

        canvas_row = tk.Frame(cf, bg=COLORS["bg"])
        canvas_row.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

        self.canvas_left = tk.Canvas(
            canvas_row, bg=COLORS["bg_card"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            cursor="crosshair")
        self.canvas_right = tk.Canvas(
            canvas_row, bg=COLORS["bg_card"],
            highlightthickness=1,
            highlightbackground=COLORS["border"])
        self.canvas_left.pack(side=tk.LEFT, fill=tk.BOTH,
                              expand=True, padx=(0, 3))
        self.canvas_right.pack(side=tk.RIGHT, fill=tk.BOTH,
                               expand=True, padx=(3, 0))

        self.roi_selector = ROISelector(
            self.canvas_left, self._on_roi_change)

        # ZoomController sincroniza ambos canvas
        self.zoom_ctrl = ZoomController(
            self.canvas_left, self.canvas_right,
            on_redraw=self._on_zoom_redraw)
        self.zoom_ctrl.set_zoom_var(self._zoom_pct_var)
        # Referencia para que ZoomController informe al ROI
        self.canvas_left._roi_selector_ref = self.roi_selector

        # Barra inferior: pan + navegador de pasos
        bottom_bar = tk.Frame(cf, bg=COLORS["bg_panel"], height=28)
        bottom_bar.pack(fill=tk.X)
        bottom_bar.pack_propagate(False)
        tk.Frame(bottom_bar, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, side=tk.TOP)

        # Botones de pan — izquierda de la barra inferior
        pan_kw = dict(bg=COLORS["bg_panel"], fg=COLORS["accent"],
                      relief="flat", font=("Segoe UI", 7),
                      width=2, height=1, cursor="hand2",
                      padx=0, pady=0)
        tk.Button(bottom_bar, text="◀",
                  command=lambda: self._do_pan(30, 0),
                  **pan_kw).pack(side=tk.LEFT, padx=1)
        tk.Button(bottom_bar, text="▲",
                  command=lambda: self._do_pan(0, 30),
                  **pan_kw).pack(side=tk.LEFT, padx=1)
        tk.Button(bottom_bar, text="▼",
                  command=lambda: self._do_pan(0, -30),
                  **pan_kw).pack(side=tk.LEFT, padx=1)
        tk.Button(bottom_bar, text="▶",
                  command=lambda: self._do_pan(-30, 0),
                  **pan_kw).pack(side=tk.LEFT, padx=1)

        # Navegador de pasos
        step_nav = bottom_bar
        step_nav.pack(fill=tk.X)
        step_nav.pack_propagate(False)
        tk.Frame(step_nav, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, side=tk.TOP)
        tk.Button(step_nav, text="◀", command=self._prev_step,
                  bg=COLORS["bg_panel"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["small"],
                  cursor="hand2", width=2).pack(side=tk.LEFT, padx=4)
        self.step_label_var = tk.StringVar(value="—")
        tk.Label(step_nav, textvariable=self.step_label_var,
                 bg=COLORS["bg_panel"], fg=COLORS["text"],
                 font=FONTS["small"]).pack(side=tk.LEFT, expand=True)
        tk.Button(step_nav, text="▶", command=self._next_step,
                  bg=COLORS["bg_panel"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["small"],
                  cursor="hand2", width=2).pack(side=tk.RIGHT, padx=4)

        # ── Notebook: Tabla | Gráficos ───────────────────────────────────
        nb_style = ttk.Style()
        nb_style.configure("Bottom.TNotebook",
                           background=COLORS["bg_panel"],
                           tabmargins=[0,0,0,0])
        nb_style.configure("Bottom.TNotebook.Tab",
                           background=COLORS["bg_panel"],
                           foreground=COLORS["text_muted"],
                           font=FONTS["small"],
                           padding=[8, 3])
        nb_style.map("Bottom.TNotebook.Tab",
                     background=[("selected", COLORS["bg_card"])],
                     foreground=[("selected", COLORS["accent"])])

        bottom_nb = ttk.Notebook(cf, style="Bottom.TNotebook")
        bottom_nb.pack(fill=tk.BOTH, expand=False, padx=6, pady=(2,0))
        # Altura mínima del panel de tabla/gráficos
        bottom_nb.configure(height=220)

        # ── Pestaña Tabla ─────────────────────────────────────────────────
        tab_table = tk.Frame(bottom_nb, bg=COLORS["bg_card"])
        bottom_nb.add(tab_table, text=f"  {t('tab.table')}  ")

        tv_f = tk.Frame(tab_table, bg=COLORS["bg_card"])
        tv_f.pack(fill=tk.BOTH, expand=True)
        xsb = ttk.Scrollbar(tv_f, orient="horizontal")
        ysb = ttk.Scrollbar(tv_f, orient="vertical")
        self.table = ttk.Treeview(tv_f, show="headings",
                                   xscrollcommand=xsb.set,
                                   yscrollcommand=ysb.set, height=4)
        xsb.config(command=self.table.xview)
        ysb.config(command=self.table.yview)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)
        ysb.pack(side=tk.RIGHT,  fill=tk.Y)
        self.table.pack(fill=tk.BOTH)

        self.stats_var = tk.StringVar(value="")
        tk.Label(tab_table, textvariable=self.stats_var,
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["small"], anchor="w",
                 padx=8).pack(fill=tk.X)

        # ── Pestaña Gráficos ──────────────────────────────────────────────
        tab_charts = tk.Frame(bottom_nb, bg=COLORS["bg_card"])
        bottom_nb.add(tab_charts, text=f"  {t('tab.charts')}  ")

        chart_ctrl = tk.Frame(tab_charts, bg=COLORS["bg_panel"])
        chart_ctrl.pack(fill=tk.X)
        tk.Button(chart_ctrl,
                  text=t("charts.this_image"),
                  command=self._show_intra_chart,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["small"],
                  cursor="hand2", pady=3).pack(
                      side=tk.LEFT, padx=6, pady=3)
        tk.Button(chart_ctrl,
                  text=t("charts.whole_batch"),
                  command=self._show_batch_chart,
                  bg=COLORS["btn_bg"], fg=COLORS["accent2"],
                  relief="flat", font=FONTS["small"],
                  cursor="hand2", pady=3).pack(
                      side=tk.LEFT, padx=2, pady=3)
        tk.Button(chart_ctrl,
                  text=t("common.save"),
                  command=self._save_intra_chart,
                  bg=COLORS["btn_bg"], fg=COLORS["text_muted"],
                  relief="flat", font=FONTS["small"],
                  cursor="hand2", pady=3).pack(
                      side=tk.RIGHT, padx=6, pady=3)

        self._chart_panel = IntraImageChartPanel(
            tab_charts, colors=COLORS)
        self._chart_panel.pack(fill=tk.BOTH, expand=True)

    # ── Panel derecho ─────────────────────────────────────────────────────────

    def _build_right_panel(self):
        rf = self.right_panel.content
        self._section_lbl(rf, t("right.title"))
        tk.Label(rf, text=t("right.subtitle"),
                 bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                 font=FONTS["small"], padx=10).pack(anchor="w")
        self._divider(rf)
        self.config_container = tk.Frame(rf, bg=COLORS["bg_panel"])
        self.config_container.pack(fill=tk.BOTH, expand=True, padx=4)
        self.config_panel: ConfigPanel | None = None

    # ── Helpers layout ────────────────────────────────────────────────────────

    def _section_lbl(self, parent, text):
        f = tk.Frame(parent, bg=COLORS["bg_panel"])
        f.pack(fill=tk.X)
        tk.Label(f, text=text, bg=COLORS["bg_panel"],
                 fg=COLORS["accent"],
                 font=("Segoe UI", 8, "bold"),
                 pady=5, padx=10).pack(side=tk.LEFT)
        tk.Frame(f, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, side=tk.BOTTOM)

    def _divider(self, parent):
        tk.Frame(parent, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, pady=4)

    # ── Responsividad ─────────────────────────────────────────────────────────

    def _bind_resize(self):
        self.root.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        if event.widget != self.root:
            return
        if self._after_resize_id:
            self.root.after_cancel(self._after_resize_id)
        self._after_resize_id = self.root.after(80, self._do_resize)

    def _do_resize(self):
        self.root.update_idletasks()
        # Actualizar roi selector con nuevo tamaño del canvas izquierdo
        self.canvas_left.update_idletasks()
        cw = max(self.canvas_left.winfo_width(),  100)
        ch = max(self.canvas_left.winfo_height(), 100)
        if self.roi_selector:
            self.roi_selector.set_canvas_size(cw, ch)
        # Redibujar con zoom actual
        self._show_preview()

    # ── Carga ─────────────────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title=t("dlg.select_image"),
            filetypes=[(t("common.images"),"*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       (t("common.all_files"),"*.*")])
        if path:
            self.project.set_images([path], "individual")
            self._update_batch_list()
            self._load_single(path)
            self.mode_var.set("individual")

    def _open_folder(self):
        folder = filedialog.askdirectory(title=t("dlg.image_folder"))
        if not folder:
            return
        paths = sorted(str(p) for p in Path(folder).iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            messagebox.showwarning(t("msg.no_images_title"),
                                   t("msg.no_images"))
            return
        self.project.set_images(paths, "batch")
        self.output_root = folder
        self._update_batch_list()
        self._load_single(paths[0])
        self.mode_var.set("batch")
        self._exporter = Exporter(folder)
        self._set_status(t("status.folder_loaded", n=len(paths), folder=folder))

    # ── Métodos de zoom (llamados por botones) ───────────────────────────────

    def _do_pan(self, dx: int, dy: int):
        """Pan de la imagen — llamado por botones de flecha."""
        if self.zoom_ctrl is None:
            return
        img = self.zoom_ctrl._img_left if self.zoom_ctrl._img_left is not None \
              else self.zoom_ctrl._img_right
        if img is None:
            return
        self.zoom_ctrl.cl.update_idletasks()
        cw = max(self.zoom_ctrl.cl.winfo_width(),  1)
        ch = max(self.zoom_ctrl.cl.winfo_height(), 1)
        ih, iw = img.shape[:2]
        self.zoom_ctrl.state.pan(dx, dy, cw, ch, iw, ih)
        # Redibujar con overlay si hay preview activo
        self._show_preview()

    def do_zoom_in(self):
        if self.zoom_ctrl is None:
            return
        self.zoom_ctrl.zoom_in()

    def do_zoom_out(self):
        if self.zoom_ctrl is None:
            return
        self.zoom_ctrl.zoom_out()

    def do_zoom_fit(self):
        if self.zoom_ctrl is None:
            return
        self.zoom_ctrl.fit()

    def _on_zoom_redraw(self):
        """Callback del ZoomController — actualiza offset del ROI."""
        if self.zoom_ctrl and self.roi_selector:
            ox, oy, dw, dh = self.zoom_ctrl.current_image_offset()
            self.roi_selector.set_image_offset(ox, oy, dw, dh)

    def _load_single(self, path: str):
        self.current_image_path = path
        img, info = self._load_corrected(path)
        if img is None:
            messagebox.showerror(t("common.error"), t("msg.image_unreadable", name=Path(path).name),
                                 parent=self.root)
            return
        self.scaler_left.set_image(img)
        self._update_corr_indicator()
        self.preview_var.set("")
        self._update_step_active(1)
        # Al cargar imagen nueva: fit-to-canvas (imagen completa visible)
        if self.zoom_ctrl:
            self.zoom_ctrl._img_right = None
            self.zoom_ctrl.set_left(self.scaler_left.original)
        else:
            self._show_preview()

        # Restaurar resultado previo si existe en cache
        if path in self.results_cache:
            self.last_result = self.results_cache[path]
            self.step_names  = self.step_names_cache.get(path, [])
            self.step_idx    = len(self.step_names) - 1

            # Mostrar último paso del resultado en canvas derecho
            if self.step_names and self.last_result:
                last_name = self.step_names[-1]
                last_img  = self.last_result.step_images.get(last_name)
                if last_img is not None:
                    self.scaler_right = ImageScaler()
                    self.scaler_right.set_image(last_img)
                    self.canvas_right.update_idletasks()
                    cw = max(self.canvas_right.winfo_width(), 100)
                    ch = max(self.canvas_right.winfo_height(), 100)
                    self.scaler_right.set_canvas_size(cw, ch)
                    self._show_on(self.canvas_right,
                                  self.scaler_right.display, "_photo_right")
                self.step_label_var.set(
                    f"{self.step_names[-1]}  ({len(self.step_names)} pasos)")
            self._update_table(self.last_result)
            self.stats_var.set("   |   ".join(
                f"{k}: {v}" for k, v in self.last_result.stats.items()))
        else:
            # Nueva imagen sin resultados previos
            self.last_result = None
            self.step_names  = []
            self.step_idx    = 0
            self.scaler_right = ImageScaler()
            self.step_label_var.set("—")
            # No limpiar canvas derecho — mantener último resultado visible
            # (solo se limpia si no hay nada cacheado en ninguna imagen)
            if not any(self.results_cache.values()):
                self.canvas_right.delete("all")

        self._refresh_history()
        self._set_status(t("status.image", name=Path(path).name))

    # ── Display ───────────────────────────────────────────────────────────────

    def _refresh_display(self):
        self._show_preview()
        # El canvas derecho se actualiza via zoom_ctrl en _show_preview
        # Si no hay imagen derecha, aseguramos que zoom_ctrl la tenga
        if self.zoom_ctrl and self.scaler_right.has_image:
            if self.zoom_ctrl._img_right is None:
                self.zoom_ctrl.set_right(self.scaler_right.original)

    def _show_on(self, canvas: tk.Canvas,
                 img_bgr: np.ndarray, attr: str):
        rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        setattr(self, attr, photo)
        canvas.delete("all")
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        x  = max(0, (cw - img_bgr.shape[1]) // 2)
        y  = max(0, (ch - img_bgr.shape[0]) // 2)
        canvas.create_image(x, y, anchor="nw", image=photo)
        # Informar al ROI selector el offset real de la imagen
        if canvas is self.canvas_left and self.roi_selector:
            self.roi_selector.set_image_offset(
                x, y,
                img_bgr.shape[1],
                img_bgr.shape[0])

    # ── Análisis ──────────────────────────────────────────────────────────────

    def _populate_analysis_menu(self):
        names = list(ANALYSES.keys())
        self.analysis_combo["values"] = [_analysis_label(n) for n in names]
        if names:
            self.analysis_var.set(_analysis_label(names[0]))
            self.active_analysis = names[0]
            self._on_analysis_selected(None)

    def _selected_analysis(self) -> str | None:
        label = self.analysis_var.get()
        return next((n for n in ANALYSES if _analysis_label(n) == label), None)

    def _on_analysis_selected(self, _event):
        name = self._selected_analysis()
        if name not in ANALYSES:
            return
        self._store_panel_params()
        self.active_analysis = name
        self.project.analysis = _analysis_key(name)
        for w in self.config_container.winfo_children():
            w.destroy()
        self.config_panel = ConfigPanel(
            self.config_container,
            schema=ANALYSES[name].params_schema,
            colors=COLORS, prefix=_analysis_key(name))
        self.config_panel.set_values(self.project.params.get(self.project.analysis, {}))
        self.config_panel.pack(fill=tk.BOTH, expand=True)

    def _build_params(self) -> dict:
        p = {
            "color_space_code": CV2_CODES.get(self.cs_var.get()),
            "channel_idx":      self.ch_var.get(),
            "min_val":          self.min_slider.get(),
            "max_val":          self.max_slider.get(),
            "mm_per_pixel":     self.mm_per_pixel,
        }
        roi = self.roi_selector.mask if self.roi_selector else None
        if roi is not None and self.scaler_left.has_image:
            p["roi_mask"] = self.scaler_left.scale_mask_to_original(roi)
        else:
            p["roi_mask"] = None
        # Exclusiones
        if self.roi_selector and self.roi_selector.has_exclusions:
            p["_roi_selector"] = self.roi_selector
            p["_scaler_left"]  = self.scaler_left
        if self.config_panel:
            p.update(self.config_panel.get_values())
        return p

    def _run_analysis(self):
        name = self._selected_analysis()
        if name not in ANALYSES:
            messagebox.showwarning(t("msg.no_analysis_title"),
                                   t("msg.no_analysis"))
            return
        if self.mode_var.get() == "batch":
            self._run_batch(name)
        else:
            if not self.scaler_left.has_image:
                messagebox.showwarning(t("msg.no_image_title"),
                                       t("msg.no_image"))
                return
            self._run_single(name, self.scaler_left.original,
                             self.current_image_path or "imagen")

    def _run_single(self, name: str, image: np.ndarray, path: str):
        self._set_status(t("status.running", name=_analysis_label(name)))
        self.root.config(cursor="watch")
        params = self._build_params()
        def worker():
            try:
                result = ANALYSES[name].func(image, params)
            except Exception as e:
                _log.exception("%s falló en %s", name, path)
                result = AnalysisResult(status="error", error=str(e))
            self.root.after(0, lambda: self._on_result(name, result, path))
        threading.Thread(target=worker, daemon=True).start()

    def _run_batch(self, name: str):
        if not self.batch_paths:
            messagebox.showwarning(t("msg.no_batch_title"), t("msg.no_batch"))
            return
        # Advertencia si hay resoluciones diferentes
        self._check_batch_resolutions()
        total = len(self.batch_paths)
        self._set_status(t("status.running_batch", name=_analysis_label(name), n=total))
        self.root.config(cursor="watch")
        params = self._build_params()

        def worker():
            results = []
            for i, path in enumerate(self.batch_paths):
                img, _ = self._load_corrected(path)
                if img is None:
                    continue
                p = dict(params, mm_per_pixel=self._scale_for(path))
                try:
                    r = ANALYSES[name].func(img, p)
                except Exception as e:
                    _log.exception("%s falló en %s", name, path)
                    r = AnalysisResult(status="error", error=str(e))
                results.append((path, r))

                # Cachear inmediatamente — así al navegar ya está disponible
                self.root.after(0, lambda p=path, r=r, i=i:
                    self._cache_result(name, p, r, i, total))

            self.root.after(0, lambda: self._on_batch_result(name, results))

        threading.Thread(target=worker, daemon=True).start()

    def _check_batch_resolutions(self):
        """Avisa si las imágenes del lote tienen resoluciones distintas."""
        resolutions = set()
        for path in self.batch_paths[:20]:  # revisar primeras 20
            try:
                raw = Path(path).read_bytes()
                arr = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    resolutions.add(img.shape[:2])
            except Exception:
                _log.warning("No se pudo leer %s", path, exc_info=True)
        if len(resolutions) > 1:
            res_list = ', '.join(f'{w}×{h}' for h,w in resolutions)
            messagebox.showwarning(
                t("msg.res_title"),
                t("msg.res_body", n=len(resolutions), list=res_list),
                parent=self.root)

    def _cache_result(self, name: str, path: str,
                      result: AnalysisResult, i: int, total: int):
        """
        Guarda el resultado en cache inmediatamente al procesarse.
        Si la imagen activa es esta, actualiza el display.
        """
        step_names = list(result.step_images.keys())
        self.results_cache[path]    = result
        self.step_names_cache[path] = step_names
        self.step_history[path]     = step_names
        # Guardar en índice combinado por análisis
        if name not in self.all_results_by_analysis:
            self.all_results_by_analysis[name] = {}
        self.all_results_by_analysis[name][path] = result

        # Exportar CSV en tiempo real y guardar imágenes de pasos
        if self.output_root and result.status == 'ok':
            if self._exporter is None:
                self._exporter = Exporter(self.output_root)
            self._exporter.save_result(
                name, Path(path).name, result,
                save_step_images=True)
            self._exporter.append_to_csv(
                name, Path(path).name, result)

        self._set_status(f"{name}: {i+1}/{total}…")

        # Si es la imagen que está visible ahora, actualizar display
        if path == self.current_image_path and result.status == "ok":
            self.last_result = result
            self.step_names  = step_names
            self.step_idx    = len(step_names) - 1
            if step_names:
                last_img = result.step_images.get(step_names[-1])
                if last_img is not None:
                    self._show_step_img(step_names[-1], last_img)
                    self.step_label_var.set(
                        f"{step_names[-1]}  ({len(step_names)} pasos)")
            self._update_table(result)
            self.stats_var.set("   |   ".join(
                f"{k}: {v}" for k, v in result.stats.items()))

    def _on_result(self, name: str, result: AnalysisResult, path: str):
        self.root.config(cursor="")
        self.last_result = result
        if result.status == "error":
            messagebox.showerror(t("common.error"), result.error)
            self._set_status(f"{t('common.error')}: {result.error}")
            return
        self.step_names = list(result.step_images.keys())
        self.step_idx   = len(self.step_names) - 1
        self.step_history[path] = self.step_names
        # Guardar en cache para restaurar al navegar
        self.results_cache[path]    = result
        self.step_names_cache[path] = self.step_names
        # Índice combinado por análisis
        if name not in self.all_results_by_analysis:
            self.all_results_by_analysis[name] = {}
        self.all_results_by_analysis[name][path] = result
        if self.step_names:
            self._show_step_img(self.step_names[-1],
                                result.step_images[self.step_names[-1]])
            self.step_label_var.set(
                f"{self.step_names[-1]}  ({len(self.step_names)} pasos)")
        self._update_table(result)
        self.stats_var.set("   |   ".join(
            f"{k}: {v}" for k, v in result.stats.items()))
        if self.output_root:
            if self._exporter is None:
                self._exporter = Exporter(self.output_root)
            self._exporter.save_result(name, Path(path).name, result,
                                       save_step_images=True)
            self._exporter.append_to_csv(name, Path(path).name, result)
            self._set_status(
                t("status.done", name=_analysis_label(name), detail=self._exporter.results_dir))
        else:
            self._set_status(t("status.done", name=_analysis_label(name), detail=result.stats))
        self._refresh_history()
        self._update_step_active(3)
        self._log_process(name)
        # Auto-actualizar panel de gráficos
        if self._chart_panel and result.measurements:
            self._chart_panel.load(result.measurements)

    def _on_batch_result(self, name: str, results: list):
        self.root.config(cursor="")
        if not results:
            self._set_status(t("msg.no_results_status"))
            return

        self.root.config(cursor="")
        ok_count = sum(1 for _, r in results if r.status == "ok")

        # Asegurar que la imagen activa muestra su resultado
        cur   = self.current_image_path
        if cur and cur in self.results_cache:
            result = self.results_cache[cur]
            if result.status == "ok":
                self.last_result = result
                self.step_names  = self.step_names_cache.get(cur, [])
                self.step_idx    = len(self.step_names) - 1
                if self.step_names:
                    last_img = result.step_images.get(self.step_names[-1])
                    if last_img is not None:
                        self._show_step_img(self.step_names[-1], last_img)
                        self.step_label_var.set(
                            f"{self.step_names[-1]}  "
                            f"({len(self.step_names)} pasos)")
                self._update_table(result)
                self.stats_var.set("   |   ".join(
                    f"{k}: {v}" for k, v in result.stats.items()))
                self._refresh_history()
                self._update_step_active(3)

        if self.output_root:
            self._set_status(
                t("status.batch_done", name=_analysis_label(name), ok=ok_count, n=len(results))
                + f" — {Path(self.output_root) / 'resultados'}")
        else:
            self._set_status(
                t("status.batch_done", name=_analysis_label(name), ok=ok_count, n=len(results)))

    # ── Tabla ─────────────────────────────────────────────────────────────────

    def _update_table(self, result: AnalysisResult):
        self.table.delete(*self.table.get_children())
        if not result.measurements:
            return
        cols = result.table_columns or list(result.measurements[0].keys())
        self.table["columns"] = cols
        for col in cols:
            self.table.heading(col, text=col, anchor="w")
            self.table.column(col, width=max(80, len(col)*8), anchor="w")
        for row in result.measurements:
            self.table.insert("", "end",
                              values=[str(row.get(c,"")) for c in cols])

    # ── Navegación pasos ──────────────────────────────────────────────────────

    def _prev_step(self):
        if self.last_result and self.step_names:
            self.step_idx = max(0, self.step_idx - 1)
            n = self.step_names[self.step_idx]
            self._show_step_img(n, self.last_result.step_images[n])
            self.step_label_var.set(
                f"{n}  ({self.step_idx+1}/{len(self.step_names)})")

    def _next_step(self):
        if self.last_result and self.step_names:
            self.step_idx = min(len(self.step_names)-1, self.step_idx+1)
            n = self.step_names[self.step_idx]
            self._show_step_img(n, self.last_result.step_images[n])
            self.step_label_var.set(
                f"{n}  ({self.step_idx+1}/{len(self.step_names)})")

    def _show_step_img(self, name: str, img: np.ndarray):
        self.scaler_right.set_image(img)
        if self.zoom_ctrl:
            self.zoom_ctrl.set_right(img)
        else:
            self.canvas_right.update_idletasks()
            cw = max(self.canvas_right.winfo_width(),  100)
            ch = max(self.canvas_right.winfo_height(), 100)
            self.scaler_right.set_canvas_size(cw, ch)
            self._show_on(self.canvas_right,
                          self.scaler_right.display, "_photo_right")

    # ── Lote ──────────────────────────────────────────────────────────────────

    def _prev_image(self):
        if self.batch_paths and self.batch_index > 0:
            self.batch_index -= 1
            self._load_single(self.batch_paths[self.batch_index])
            self._sync_listbox()

    def _next_image(self):
        if self.batch_paths and self.batch_index < len(self.batch_paths)-1:
            self.batch_index += 1
            self._load_single(self.batch_paths[self.batch_index])
            self._sync_listbox()

    def _on_batch_select(self, _event):
        sel = self.batch_listbox.curselection()
        if sel:
            self.batch_index = sel[0]
            self._load_single(self.batch_paths[self.batch_index])
            self._update_batch_label()

    def _sync_listbox(self):
        self.batch_listbox.selection_clear(0, tk.END)
        self.batch_listbox.selection_set(self.batch_index)
        self.batch_listbox.see(self.batch_index)
        self._update_batch_label()

    def _update_batch_list(self):
        self.batch_listbox.delete(0, tk.END)
        for p in self.batch_paths:
            self.batch_listbox.insert(tk.END, Path(p).name)
        if self.batch_paths:
            self.batch_listbox.selection_set(0)
        self._update_batch_label()

    def _update_batch_label(self):
        n = len(self.batch_paths)
        self.batch_label.config(
            text=f"{self.batch_index+1} / {n}" if n else "—")

    # ── Historial ─────────────────────────────────────────────────────────────

    def _refresh_history(self):
        for w in self.history_frame.winfo_children():
            w.destroy()
        path = self.current_image_path
        if not path or path not in self.step_history:
            return
        for i, step_name in enumerate(self.step_history[path]):
            btn = tk.Label(self.history_frame,
                           text=f"  ▸  {step_name}",
                           bg=COLORS["bg_panel"], fg=COLORS["text"],
                           font=FONTS["small"], anchor="w",
                           padx=4, pady=2, cursor="hand2")
            btn.pack(fill=tk.X)
            btn.bind("<Button-1>",
                     lambda e, idx=i: self._jump_to_step(idx))
            btn.bind("<Enter>",
                     lambda e, b=btn: b.config(bg=COLORS["accent_light"]))
            btn.bind("<Leave>",
                     lambda e, b=btn: b.config(bg=COLORS["bg_panel"]))

    def _jump_to_step(self, idx: int):
        if not self.last_result or idx >= len(self.step_names):
            return
        self.step_idx = idx
        name = self.step_names[idx]
        self._show_step_img(name, self.last_result.step_images[name])
        self.step_label_var.set(
            f"{name}  ({idx+1}/{len(self.step_names)})")

    # ── ROI ───────────────────────────────────────────────────────────────────

    def _pick_exclusion_color(self):
        """Abre selector de color para zona de exclusión."""
        if self.roi_selector:
            self.roi_selector.pick_exclusion_color(self.root)

    def _clear_exclusions(self):
        """Limpia solo las zonas de exclusión."""
        if self.roi_selector:
            self.roi_selector.clear_exclusions()

    def _set_roi_mode(self, mode: str):
        if self.roi_selector:
            self.roi_selector.set_mode(mode)
        self._update_step_active(2)
        self._set_status(t("status.roi_mode", mode=t(f"roi.mode.{mode}", mode)))

    def _on_roi_change(self, mask):
        self._update_step_active(2)
        self._show_preview()

    # ── Calibración / diálogos ────────────────────────────────────────────────

    def _on_scale_set(self, scale_result):
        """Callback desde ScaleDialog — aplica la escala al estado."""
        from fenotit.core.corrections.scale import UNIT_TO_MM
        # Guardamos mm/px para compatibilidad con los módulos de análisis
        self.project.scale = Scale(
            scale_result.unit_per_px * UNIT_TO_MM.get(scale_result.unit, 1.0),
            "two_points", scale_result.format())
        self._update_corr_indicator()
        self.scale_result = scale_result
        self._set_status(t("status.scale", scale=scale_result.format()))

    def _calibrate_scale(self):
        val = simpledialog.askfloat(
            t("menu.scale_manual").rstrip("…"),
            t("scale.manual_prompt"),
            minvalue=0.0001)
        if val:
            self.project.scale = Scale(val, "manual")
            self._update_corr_indicator()
            self._set_status(t("status.scale", scale=f"{val:.6f} mm/px"))

    def _wip(self, title: str, desc: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=COLORS["bg_card"])
        win.resizable(False, False)
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"380x200+{(sw-380)//2}+{(sh-200)//2}")
        try: win.iconbitmap(str(_assets()/"logo.ico"))
        except Exception: _log.debug("ignorado", exc_info=True)
        tk.Frame(win, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        body = tk.Frame(win, bg=COLORS["bg_card"])
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        tk.Label(body, text=title,
                 bg=COLORS["bg_card"], fg=COLORS["accent"],
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(body, text=desc,
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"], justify="left").pack(
                     anchor="w", pady=(8, 12))
        tk.Label(body, text=t("wip.notice"),
                 bg=COLORS["bg_card"], fg=COLORS["warning"],
                 font=FONTS["small"]).pack(anchor="w")
        tk.Button(body, text=t("common.close"), command=win.destroy,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=12).pack(anchor="e", pady=(12, 0))

    def _ai_segmentation(self):
        win = tk.Toplevel(self.root)
        win.title(t("ai.title"))
        win.configure(bg=COLORS["bg_card"])
        win.resizable(False, False)
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"400x280+{(sw-400)//2}+{(sh-280)//2}")
        try: win.iconbitmap(str(_assets()/"logo.ico"))
        except Exception: _log.debug("ignorado", exc_info=True)
        tk.Frame(win, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        body = tk.Frame(win, bg=COLORS["bg_card"])
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        tk.Label(body, text=f"🤖  {t('ai.title')}",
                 bg=COLORS["bg_card"], fg=COLORS["accent"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(body, text=t("ai.select_model"),
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"]).pack(anchor="w", pady=(8, 4))
        model_var = tk.StringVar(value="SAM (Segment Anything)")
        for m in ["SAM (Segment Anything)",
                  "YOLOv8-seg (custom)", t("ai.other_model")]:
            tk.Radiobutton(body, text=m,
                           variable=model_var, value=m,
                           bg=COLORS["bg_card"], fg=COLORS["text"],
                           selectcolor=COLORS["bg_panel"],
                           activebackground=COLORS["bg_card"],
                           font=FONTS["body"]).pack(anchor="w")
        tk.Label(body,
                 text=t("wip.module"),
                 bg=COLORS["bg_card"], fg=COLORS["warning"],
                 font=FONTS["small"]).pack(anchor="w", pady=(12, 0))
        tk.Button(body, text=t("common.close"), command=win.destroy,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=12).pack(anchor="e", pady=(12, 0))

    def _choose_language(self):
        langs = i18n.available()
        win = tk.Toplevel(self.root)
        win.title(t("lang.title"))
        win.configure(bg=COLORS["bg_card"])
        win.resizable(False, False)
        win.transient(self.root)
        tk.Frame(win, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        body = tk.Frame(win, bg=COLORS["bg_card"])
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        var = tk.StringVar(value=i18n.current())
        for code, name in langs.items():
            tk.Radiobutton(body, text=name, variable=var, value=code,
                           bg=COLORS["bg_card"], fg=COLORS["text"],
                           selectcolor=COLORS["bg_panel"],
                           activebackground=COLORS["bg_card"],
                           font=FONTS["body"]).pack(anchor="w")

        def apply():
            code = var.get()
            win.destroy()
            if code == i18n.current():
                return
            i18n.set_language(code)
            messagebox.showinfo(t("lang.title"), t("lang.restart"), parent=self.root)

        tk.Button(body, text="OK", command=apply,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=12).pack(anchor="e", pady=(12, 0))
        win.update_idletasks()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = max(win.winfo_reqwidth(), 260), win.winfo_reqheight()
        win.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def _system_info(self) -> str:
        try:
            import ctypes
            scale = ctypes.windll.shcore.GetScaleFactorForDevice(0)
        except Exception:
            scale = round(self.root.winfo_fpixels("1i") / 96 * 100)
        return t("about.system", version=__version__, python=platform.python_version(),
                 os=f"{platform.system()} {platform.release()}",
                 screen=f"{self.root.winfo_screenwidth()}×{self.root.winfo_screenheight()}",
                 scale=scale, log=log.log_file())

    def _about(self):
        about_file = _assets() / f"about_{i18n.current()}.txt"
        if not about_file.exists():
            about_file = _assets() / "about_en.txt"
        text = about_file.read_text(encoding="utf-8") if about_file.exists() else ""
        win = tk.Toplevel(self.root)
        win.title(t("menu.about"))
        win.configure(bg=COLORS["bg_card"])
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        try: win.iconbitmap(str(_assets()/"logo.ico"))
        except Exception: _log.debug("ignorado", exc_info=True)
        tk.Frame(win, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        body = tk.Frame(win, bg=COLORS["bg_card"])
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        tk.Label(body, text=APP_NAME,
                 bg=COLORS["bg_card"], fg=COLORS["accent"],
                 font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(body, text="Digital Phenotyping Platform",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["body"]).pack(anchor="w")
        tk.Frame(body, bg=COLORS["border"], height=1).pack(
            fill=tk.X, pady=10)
        tk.Label(body, text=text,
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["small"], justify="left",
                 wraplength=380).pack(anchor="w")
        tk.Frame(body, bg=COLORS["border"], height=1).pack(
            fill=tk.X, pady=10)
        tk.Label(body, text=self._system_info(),
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["small"], justify="left",
                 wraplength=380).pack(anchor="w")
        tk.Button(body, text=t("common.close"), command=win.destroy,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=12).pack(anchor="e", pady=(12, 0))
        win.update_idletasks()
        w, h = win.winfo_reqwidth(), min(win.winfo_reqheight(), sh - 80)
        win.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    # ── Exportación ───────────────────────────────────────────────────────────

    def _export_results(self):
        """Abre el diálogo de exportación."""
        available = list(self.results_cache.keys())
        if not available and not self._exporter:
            messagebox.showwarning(
                t("msg.no_results_title"),
                t("msg.no_results"),
                parent=self.root)
            return
        # Inferir análisis disponibles del cache
        analysis_names = list({
            name for name in (self.active_analysis,)
            if name
        }) or ["Análisis"]
        ExportDialog(
            self.root,
            available_analyses=analysis_names,
            exporter=self._exporter,
            output_root=self.output_root)

    def _export_batch(self):
        """Exportar lote — mismo diálogo que exportar imagen."""
        self._export_results()

    # ── Helpers generales ─────────────────────────────────────────────────────

    def _show_intra_chart(self):
        """Actualiza el panel de gráficos con la imagen actual."""
        if self._chart_panel and self.last_result \
                and self.last_result.measurements:
            self._chart_panel.load(self.last_result.measurements)
        elif self._chart_panel:
            messagebox.showinfo(
                t("msg.no_data_title"),
                t("msg.no_data_image"),
                parent=self.root)

    def _show_batch_chart(self):
        """Abre ventana de gráficos del lote con todos los análisis."""
        if not self.results_cache:
            messagebox.showinfo(
                t("msg.no_data_title"),
                t("msg.no_data_batch"),
                parent=self.root)
            return
        # Agrupar results_cache por análisis
        # results_cache: {path: AnalysisResult} del último análisis
        # Para multi-análisis, usamos all_results_by_analysis
        results_by_analysis = getattr(
            self, "all_results_by_analysis", {})
        if not results_by_analysis:
            # Fallback: solo el análisis activo
            name = self.active_analysis or "Análisis"
            results_by_analysis = {name: self.results_cache}
        BatchChartWindow(self.root, results_by_analysis,
                         self.active_analysis or "")

    def _save_intra_chart(self):
        """Guarda el gráfico intra-imagen actual."""
        if self._chart_panel:
            self._chart_panel.save_figure()

    def _log_process(self, name: str):
        """Agrega al log del panel izquierdo el proceso ejecutado."""
        if hasattr(self, 'log_listbox'):
            self.log_listbox.insert(tk.END, f"✓  {name}")
            self.log_listbox.see(tk.END)

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _update_step_active(self, active: int):
        # step_labels ya no se usan — no-op
        pass