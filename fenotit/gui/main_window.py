"""
ui/main_window.py  —  FenotIT v1.3
Cambios vs v1.2:
  - Preview en tiempo real: al mover sliders se superpone la máscara
    sobre el canvas izquierdo (objetos resaltados en verde)
  - Paneles izquierdo y derecho colapsables con botón ◀/▶
  - PanedWindow para redimensionar arrastrando el separador
  - Tooltips del config_panel ya no se salen de pantalla (fix en config_panel.py)
"""

import os
import platform
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from fenotit.core.image_io import ImageScaler, load_image
from fenotit.core.project import IMAGE_EXTS, PROJECT_EXT, Project, Scale, Segmentation
from fenotit.gui.roi.selectors import ROISelector
from fenotit.core.analysis.registry import ANALYSES, AnalysisResult
from fenotit.core import pipeline
from fenotit.core.export.exporter import Exporter
from fenotit.gui.config_panel import ConfigPanel
from fenotit.gui.zoom_controller import ZoomController
from fenotit.gui.toolbar import IconButton, Tooltip
from fenotit.gui.calibration_dialogs import ScaleDialog
from fenotit.gui.corrections_dialog import CorrectionsDialog
from fenotit.core.corrections import pipeline as corrections
from fenotit.gui.export_dialog import ExportDialog
from fenotit.gui.charts import IntraImageChartPanel, BatchChartWindow
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
    def __init__(self, parent, text, items, colors, arrow_only=False, **kw):
        bg = "#FFFFFF" if arrow_only else colors["bg_topbar"]
        super().__init__(parent, bg=bg, **kw)
        self.colors = colors
        self.items  = items
        self._popup = None
        self._arrow_only = arrow_only
        self.btn = tk.Button(
            self, text=" ▾ " if arrow_only else f"  {text}  ▾",
            bg=bg, fg=colors["accent"] if arrow_only else "#FFFFFF",
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
            disabled = bool(rest and rest[0] is True)
            shortcut = next((r for r in rest if isinstance(r, str)), "")
            fg = self.colors["text_muted"] if disabled else self.colors["text"]
            row = tk.Frame(inner, bg=self.colors["bg_card"])
            row.pack(fill=tk.X)
            lbl = tk.Label(row, text=f"  {label.strip()}   ",
                           bg=self.colors["bg_card"], fg=fg,
                           font=FONTS["body"], anchor="w", pady=4, padx=4)
            lbl.pack(fill=tk.X)
            if shortcut:
                tk.Label(row, text=shortcut, bg=self.colors["bg_card"], fg=self.colors["text_muted"],
                         font=("Segoe UI", 8), padx=8).place(relx=1.0, rely=0.5, anchor="e")
                lbl.config(text=f"  {label.strip()}" + " " * (len(shortcut) + 10))
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

        # Que quede dentro de la ventana: el menú de ▾ (junto al borde derecho) se abre
        # hacia la izquierda, alineado con el botón.
        popup.update_idletasks()
        pw = popup.winfo_reqwidth()
        top = self.winfo_toplevel()
        right = top.winfo_rootx() + top.winfo_width()
        if self._arrow_only or x + pw > right - 4:
            x = self.winfo_rootx() + self.winfo_width() - pw
        x = max(top.winfo_rootx() + 4, min(x, right - pw - 4))
        popup.wm_geometry(f"+{x}+{y}")
        popup.bind("<FocusOut>", lambda e=None: popup.destroy())
        popup.focus_set()


# ── Panel colapsable ──────────────────────────────────────────────────────────

class CollapsiblePanel(tk.Frame):
    """
    Panel lateral que se puede colapsar/expandir con un botón.
    Cuando está colapsado solo muestra una tira con el botón.
    """
    COLLAPSED_W = 18

    def __init__(self, parent, side: str, title: str,
                 colors: dict, default_width: int = 200, on_toggle=None, **kw):
        super().__init__(parent, bg=colors["bg_panel"], **kw)
        self.colors        = colors
        self._on_toggle    = on_toggle
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
        self._toggle_btn.bind("<Button-1>", lambda e=None: self.toggle())

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

    def _set_width(self, width: int):
        """El ancho real lo manda el PanedWindow que contiene al panel."""
        if isinstance(self.master, tk.PanedWindow):
            self.master.paneconfigure(self, width=width)
        else:
            self.configure(width=width)
        if self._on_toggle:
            self._on_toggle()

    def collapse(self):
        self._expanded = False
        width = self.winfo_width()
        if width > self.COLLAPSED_W * 3:          # recordar el ancho que dejó el usuario
            self.default_width = width
        self.content.pack_forget()
        arrow = "▶" if self.side == "left" else "◀"
        self._toggle_btn.config(text=arrow)
        self._set_width(self.COLLAPSED_W)

    def expand(self):
        self._expanded = True
        self.content.pack(
            side=tk.LEFT if self.side == "left" else tk.RIGHT,
            fill=tk.BOTH, expand=True)
        arrow = "◀" if self.side == "left" else "▶"
        self._toggle_btn.config(text=arrow)
        self._set_width(self.default_width)


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

    def _skip_reason(self, info) -> str | None:
        """Motivo para no analizar una imagen (perspectiva activa pero sin los 4 ArUco)."""
        if info and self.project.corrections.perspective.enabled and "perspective" not in info.applied:
            return t("corr.skip_aruco", ids=", ".join(map(str, info.aruco_missing)) or "?")
        return None

    def _mark_skipped(self, path: str, reason: str):
        if path in self.batch_paths:
            i = self.batch_paths.index(path)
            self.batch_listbox.delete(i)
            self.batch_listbox.insert(i, f"⚠ {Path(path).name}")
            self.batch_listbox.itemconfig(i, fg=COLORS["warning"])
        _log.warning("%s omitida: %s", Path(path).name, reason)

    def _open_corrections(self):
        CorrectionsDialog(self.root, self.project.corrections, self.batch_paths, self.current_image_path,
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
        self._sync_area_units()

    def _invalidate_results(self):
        self._corr_info.clear()
        self.results_cache.clear()
        self.step_names_cache.clear()
        self.all_results_by_analysis.clear()
        self.last_result = None
        self.canvas_right.delete("all")
        self._show_results_ui(False)

    def _clear_scale(self):
        self.project.scale = Scale()
        self._set_status(t("status.scale_cleared"))
        self._update_corr_indicator()
        self._sync_area_units()

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
            values = panel.get_values(warn=False)
            if self._has_area_units():
                values["_area_unit"] = self._panel_area_unit
            self.project.params[_analysis_key(self.active_analysis)] = values

    # ── Unidades de área en el panel (px² sin escala, mm² con escala) ────────

    def _area_unit(self) -> str:
        return "mm" if self.project.scale.source != "none" else "px"

    def _has_area_units(self) -> bool:
        return any(i.get("unit") == "area"
                   for i in ANALYSES[self.active_analysis].params_schema) if self.active_analysis in ANALYSES else False

    def _sync_area_units(self):
        """Si cambió la escala, convierte las áreas del panel y cambia la etiqueta."""
        panel = getattr(self, "config_panel", None)
        if panel is None or not self._has_area_units():
            return
        new = self._area_unit()
        if new != self._panel_area_unit:
            # al quitar la escala se convierte con la última escala usada
            mpp = self.mm_per_pixel if new == "mm" else getattr(self, "_panel_mpp", None)
            if mpp:
                factor = mpp * mpp if new == "mm" else 1 / (mpp * mpp)
                keys = [i["key"] for i in ANALYSES[self.active_analysis].params_schema if i.get("unit") == "area"]
                vals = panel.get_values(warn=False)
                panel.set_values({k: round(vals[k] * factor, 4) for k in keys if k in vals})
            else:
                _log.info("Sin mm/px para convertir áreas; se conservan los números")
            self._panel_area_unit = new
        if new == "mm" and self.mm_per_pixel:
            self._panel_mpp = self.mm_per_pixel
        panel.set_units({"area": "mm²" if new == "mm" else "px²"})

    def _collect_state(self) -> dict:
        self._store_panel_params()
        self.project.segmentation = Segmentation(
            self.cs_var.get(), int(self.ch_var.get()),
            int(self.min_slider.get()), int(self.max_slider.get()), bool(self.auto_var.get()))
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
        self.auto_var.set(seg.auto)
        self._on_auto_change()
        self.legend_var.set(project.display.get("legend", True))
        self.color_fmt_var.set(project.display.get("color_format", "RGB"))

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
        self._show_panel(self.left_panel, bool(project.images))
        self._show_panel(self.right_panel, bool(project.images))
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
        if self._is_dirty():
            if not self._confirm_discard():
                return
        elif not messagebox.askyesno(t("close.title"), t("close.msg"), parent=self.root):
            return
        self.root.destroy()

    # ── Setup ─────────────────────────────────────────────────────────────────

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
            default_width=200, on_toggle=self._after_panel_toggle)
        self.paned.add(self.left_panel, minsize=CollapsiblePanel.COLLAPSED_W,
                       width=200)

        # Centro
        self.center_frame = tk.Frame(self.paned, bg=COLORS["bg"])
        self.paned.add(self.center_frame, minsize=400, stretch="always")

        # Panel derecho colapsable
        self.right_panel = CollapsiblePanel(
            self.paned, side="right",
            title=t("panel.right"), colors=COLORS,
            default_width=260, on_toggle=self._after_panel_toggle)
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
        self.preview_var = tk.StringVar(value="")
        tk.Label(self.statusbar, textvariable=self.preview_var, bg=COLORS["bg_panel"],
                 fg=COLORS["accent2"], font=FONTS["small"], padx=8).pack(side=tk.RIGHT)
        self.image_info_var = tk.StringVar(value="")
        tk.Label(self.statusbar, textvariable=self.image_info_var, bg=COLORS["bg_panel"],
                 fg=COLORS["text"], font=FONTS["small"], padx=8).pack(side=tk.RIGHT)
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

        self._drop(self.topbar, t("menu.file"), [
            (t("menu.new_project"), self._new_project),
            (t("menu.open_project"), self._open_project),
            (t("menu.save_project"), self._save_project, "Ctrl+S"),
            (t("menu.save_project_as"), self._save_project_as),
            None,
            (t("menu.add_images"), self._open_image, "Ctrl+O"),
            (t("menu.add_folder"), self._open_folder),
            (t("menu.remove_image"), self._remove_current_image),
            (t("menu.clear_images"), self._clear_images),
            None,
            (t("menu.export"), self._export_results, "Ctrl+E"),
            None,
            (t("menu.exit"), self._on_close),
        ])
        self._drop(self.topbar, t("menu.image"), [
            (t("menu.corrections"), self._open_corrections),
            None,
            (t("menu.cal_scale"), self._open_scale_dialog),
            (t("menu.scale_manual"), self._calibrate_scale),
            (t("menu.scale_clear"), self._clear_scale),
            None,
            (t("roi.area_rect"), lambda: self._set_roi_mode("rectángulo"), t("roi.shift_square")),
            (t("roi.area_polygon"), lambda: self._set_roi_mode("polígono")),
            (t("roi.exclude_rect"), lambda: self._set_roi_mode("exclusión_rect")),
            (t("roi.exclude_polygon"), lambda: self._set_roi_mode("exclusión")),
            (t("roi.clear"), self._clear_roi),
        ])
        self._drop(self.topbar, t("menu.analysis"), [
            (_analysis_label(n), lambda n=n: self._open_analysis(n)) for n in ANALYSES])
        self._drop(self.topbar, t("menu.view"), [
            (t("view.toggle_legend"), self._toggle_legend),
            (t("view.left_panel"), lambda: self._toggle_panel(self.left_panel)),
            (t("view.analysis_panel"), lambda: self._toggle_panel(self.right_panel)),
            (t("view.results_panel"), self._toggle_results_panel),
            None,
            (t("style.title"), self._open_style),
            None,
            (t("view.zoom_in"), self.do_zoom_in, "Ctrl++"),
            (t("view.zoom_out"), self.do_zoom_out, "Ctrl+−"),
            (t("view.zoom_fit"), self.do_zoom_fit, "Ctrl+0"),
            (t("view.zoom_100"), self.do_zoom_100, "Ctrl+1"),
            (t("view.zoom_area"), lambda: self._set_zoom_tool("zoom_area"), "Z"),
            (t("view.pan"), lambda: self._set_zoom_tool("pan"), "H"),
            None,
            (t("menu.language"), self._choose_language),
        ])
        self._drop(self.topbar, t("menu.help"), [
            (t("help.shortcuts"), self._show_shortcuts),
            (t("help.log"), self._open_log_folder),
            None,
            (t("menu.about"), self._about),
        ])

        # Derecha: tipo de análisis y Ejecutar (con margen para no quedar junto a la X)
        tk.Frame(self.topbar, bg=COLORS["bg_topbar"], width=24).pack(side=tk.RIGHT)
        run = tk.Frame(self.topbar, bg="#FFFFFF")
        run.pack(side=tk.RIGHT, padx=(6, 0), pady=8)
        tk.Button(run, text=t("topbar.run"), command=lambda: self._run_analysis(False),
                  bg="#FFFFFF", fg=COLORS["accent"], font=("Segoe UI", 9, "bold"),
                  relief="flat", bd=0, activebackground=COLORS["accent_light"],
                  activeforeground=COLORS["accent"], padx=10, pady=2,
                  cursor="hand2").pack(side=tk.LEFT)
        tk.Frame(run, bg=COLORS["border"], width=1).pack(side=tk.LEFT, fill=tk.Y, pady=3)
        DropMenu(run, "", [
            (t("run.current"), lambda: self._run_analysis(False), "Ctrl+Enter"),
            (t("run.all"), lambda: self._run_analysis(True), "Ctrl+Shift+Enter"),
        ], COLORS, arrow_only=True).pack(side=tk.LEFT)

        self.analysis_var = tk.StringVar()
        tk.Frame(self.topbar, bg="#4a8fd4", width=1).pack(side=tk.RIGHT, fill=tk.Y, pady=8, padx=6)
        self._build_zoom_tools()

    def _build_zoom_tools(self):
        """Zoom en la barra superior: acercar, alejar, ajustar, zoom a un área, mano y el %."""
        bar = tk.Frame(self.topbar, bg=COLORS["bg_topbar"])
        bar.pack(side=tk.RIGHT)
        size = max(16, int(round(18 * self.root.winfo_fpixels("1i") / 96)))
        kw = dict(bg=COLORS["bg_topbar"], hover="#3a7cc4", active="#0d3f75", size=size)
        pct = tk.Label(bar, textvariable=self._zoom_pct_var, bg=COLORS["bg_topbar"], fg="#FFFFFF",
                       font=FONTS["small"], width=6, cursor="hand2")
        self._zoom_tools = {}
        for name, cmd, tip in (
                ("zoom_in", self.do_zoom_in, f"{t('view.zoom_in')}  (Ctrl++)"),
                ("zoom_out", self.do_zoom_out, f"{t('view.zoom_out')}  (Ctrl+−)"),
                ("fit", self.do_zoom_fit, f"{t('view.zoom_fit')}  (Ctrl+0)"),
                ("zoom_area", lambda: self._set_zoom_tool("zoom_area"), f"{t('view.zoom_area')}  (Z)"),
                ("pan", lambda: self._set_zoom_tool("pan"), f"{t('view.pan')}  (H)")):
            b = IconButton(bar, name, cmd, tip, **kw)
            b.pack(side=tk.LEFT, padx=1, pady=6)
            if name in ("zoom_area", "pan"):
                self._zoom_tools[name] = b
        pct.pack(side=tk.LEFT, padx=(4, 0))
        tk.Frame(bar, bg="#4a8fd4", width=1).pack(side=tk.LEFT, fill=tk.Y, pady=8, padx=(6, 4))
        self._style_btn = IconButton(bar, "palette", self._open_style, f"{t('style.title')}", **kw)
        self._style_btn.pack(side=tk.LEFT, padx=1, pady=6)
        self._style_popup = None
        self.root.bind("<ButtonPress-1>", self._close_style_popup, add="+")
        pct.bind("<Button-1>", lambda e: self.do_zoom_100())
        Tooltip(pct, f"{t('view.zoom_100')}  (Ctrl+1)")
        self._zoom_tools["pan"].select(True)

    def _vdiv(self):
        tk.Frame(self.topbar, bg="#4a8fd4", width=1).pack(
            side=tk.LEFT, fill=tk.Y, pady=6, padx=4)

    def _drop(self, parent, text, items):
        DropMenu(parent, text, items, COLORS).pack(side=tk.LEFT)

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
        if not self._uses_threshold():
            self.zoom_ctrl.redraw_with_overlay(overlay_left=None)
            return

        img_original = self.scaler_left.original
        cs_code = CV2_CODES.get(self.cs_var.get())
        ch_idx  = self.ch_var.get()
        min_val = self.min_slider.get()
        max_val = self.max_slider.get()

        try:
            if self.auto_var.get():
                ctx = pipeline.run(img_original, [{"step": "otsu", "params": {
                    "color_space": self.cs_var.get(), "channel": int(ch_idx)}}])
                mask = ctx.mask
                self._otsu_value = int(ctx.extra.get("otsu_threshold", 0))
            else:
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

            from fenotit.core.pipeline import overlay as marks
            colors = marks.resolve(self.project.display.get("style"), img_original, mask)
            overlay = marks.tint(img_original, mask, colors)

            # Pasar al zoom controller — renderiza con zoom actual
            self.zoom_ctrl.redraw_with_overlay(overlay_left=overlay)

            n_px = int(np.sum(mask > 0))
            pct  = round(n_px / mask.size * 100, 1)
            text = t("status.preview", px=f"{n_px:,}", pct=pct)
            if self.auto_var.get():
                text += "  ·  Otsu = " + str(getattr(self, "_otsu_value", ""))
            self.preview_var.set(text)

        except Exception:
            _log.debug("Preview falló", exc_info=True)
            if self.zoom_ctrl:
                self.zoom_ctrl.redraw_with_overlay(overlay_left=img_original)
            self.preview_var.set("")

    # ── Panel izquierdo ───────────────────────────────────────────────────────

    def _build_left_panel(self):
        lf = self.left_panel.content

        # Imágenes: lista (agregar suma), contador y navegación
        self._section_lbl(lf, t("left.images"))
        nav = tk.Frame(lf, bg=COLORS["bg_panel"])
        nav.pack(fill=tk.X, padx=8, pady=2)
        btn = dict(bg=COLORS["btn_bg"], fg=COLORS["accent"], relief="flat",
                   font=FONTS["small"], cursor="hand2")
        tk.Button(nav, text="＋", command=self._open_image, width=2, **btn).pack(side=tk.LEFT)
        tk.Button(nav, text="◀", command=self._prev_image, width=2, **btn).pack(side=tk.LEFT, padx=(6, 0))
        self.batch_label = tk.Label(nav, text="—", bg=COLORS["bg_panel"],
                                    fg=COLORS["text_muted"], font=FONTS["small"])
        self.batch_label.pack(side=tk.LEFT, expand=True)
        tk.Button(nav, text="▶", command=self._next_image, width=2, **btn).pack(side=tk.RIGHT)

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
            activestyle="none", exportselection=False,
            yscrollcommand=sb.set)
        sb.config(command=self.batch_listbox.yview)
        self.batch_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.batch_listbox.bind("<<ListboxSelect>>", self._on_batch_select)
        self.batch_listbox.bind("<Delete>", lambda e=None: self._remove_current_image())

        # Vistas del resultado de la imagen actual
        self._section_lbl(lf, t("left.history"))
        self.history_frame = tk.Frame(lf, bg=COLORS["bg_panel"], height=150)
        self.history_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

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
                              expand=True, padx=(0, 2))
        self.canvas_right.pack(side=tk.RIGHT, fill=tk.BOTH,
                               expand=True, padx=(2, 0))

        self.roi_selector = ROISelector(
            self.canvas_left, self._on_roi_change)
        self.roi_selector.on_done = self._on_roi_tool_done

        # ZoomController sincroniza ambos canvas
        self.zoom_ctrl = ZoomController(
            self.canvas_left, self.canvas_right,
            on_redraw=self._on_zoom_redraw)
        self.zoom_ctrl.set_zoom_var(self._zoom_pct_var)
        self.zoom_ctrl.on_tool_change = self._on_zoom_tool_change
        # Referencia para que ZoomController informe al ROI
        self.canvas_left._roi_selector_ref = self.roi_selector

        self._canvas_row = canvas_row
        self._build_view_controls()

        # Barra de vistas (◀ vista ▶ · Resultados): solo cuando hay resultados
        step_nav = tk.Frame(cf, bg=COLORS["bg_panel"], height=28)
        step_nav.pack_propagate(False)
        self._results_bar = step_nav
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
        self._results_btn = tk.Button(step_nav, command=self._toggle_results_panel,
                                      bg=COLORS["bg_panel"], fg=COLORS["accent"],
                                      relief="flat", font=FONTS["small"], cursor="hand2")
        self._results_btn.pack(side=tk.RIGHT, padx=4)
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
        self._bottom_nb = bottom_nb
        self._results_visible = True          # preferencia del usuario (▾/▴)
        self._results_shown = False           # hay resultados para esta imagen
        self._results_btn.config(text=f"▾ {t('results.panel')}")
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
        self.table.pack(fill=tk.BOTH, expand=True)

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
        """ANÁLISIS (lista de tipos) → los bloques que pide el análisis elegido."""
        rf = self.right_panel.content
        self._section_lbl(rf, t("panel.analysis"))
        self._analysis_cards = tk.Frame(rf, bg=COLORS["bg_panel"])
        self._analysis_cards.pack(fill=tk.X, padx=6, pady=(2, 6))
        self._seg_block = tk.Frame(rf, bg=COLORS["bg_panel"])
        self._seg_block.pack(fill=tk.X)
        self._build_segmentation_block(self._seg_block)
        self.config_container = tk.Frame(rf, bg=COLORS["bg_panel"])
        self.config_container.pack(fill=tk.BOTH, expand=True, padx=4)
        self.config_panel: ConfigPanel | None = None

    def _build_segmentation_block(self, rf):
        """Espacio de color, canal y rango (o Otsu), con vista previa en vivo."""
        bg, small = COLORS["bg_panel"], FONTS["small"]
        self._section_lbl(rf, t("seg.title"))
        box = tk.Frame(rf, bg=bg)
        box.pack(fill=tk.X, padx=8, pady=(2, 4))

        row = tk.Frame(box, bg=bg)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=t("controls.space"), bg=bg, fg=COLORS["text_muted"],
                 font=small).pack(side=tk.LEFT)
        self.cs_var = tk.StringVar(value="BGR")
        cs_combo = ttk.Combobox(row, textvariable=self.cs_var, values=list(CHANNEL_NAMES.keys()),
                                state="readonly", width=7, font=small)
        cs_combo.pack(side=tk.LEFT, padx=(4, 8))
        cs_combo.bind("<<ComboboxSelected>>", self._on_cs_change)
        self.ch_var = tk.IntVar(value=0)
        self._ch_radios = []
        for i in range(3):
            rb = tk.Radiobutton(row, text="", variable=self.ch_var, value=i,
                                command=self._on_slider_change, bg=bg, fg=COLORS["text"],
                                selectcolor=COLORS["bg_card"], activebackground=bg, font=small)
            rb.pack(side=tk.LEFT)
            self._ch_radios.append(rb)
        self._update_channel_names()

        self.auto_var = tk.BooleanVar(value=False)
        tk.Checkbutton(box, text=t("seg.auto"), variable=self.auto_var, command=self._on_auto_change,
                       bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"],
                       activebackground=bg, font=small, padx=0).pack(anchor="w", pady=(2, 0))

        self.min_slider = self._range_row(box, t("seg.min"), 0)
        self.max_slider = self._range_row(box, t("seg.max"), 255)

    def _range_row(self, parent, label: str, value: int) -> tk.Scale:
        """Deslizador 0-255 con botones ‹ › para mover de a uno."""
        bg = COLORS["bg_panel"]
        row = tk.Frame(parent, bg=bg)
        row.pack(fill=tk.X)
        tk.Label(row, text=label, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                 width=4, anchor="w").pack(side=tk.LEFT, anchor="s", pady=(0, 4))
        scale = tk.Scale(row, from_=0, to=255, orient=tk.HORIZONTAL, bg=bg, fg=COLORS["text"],
                         troughcolor=COLORS["slider_trough"], activebackground=COLORS["accent"],
                         highlightthickness=0, showvalue=True, font=FONTS["small"],
                         command=lambda _: self._on_slider_change())
        scale.set(value)
        btn = dict(bg=COLORS["btn_bg"], fg=COLORS["accent"], relief="flat",
                   font=FONTS["small"], width=2, cursor="hand2", padx=0, pady=0)
        tk.Button(row, text="‹", command=lambda: scale.set(max(0, scale.get() - 1)),
                  **btn).pack(side=tk.LEFT, anchor="s", pady=(0, 4))
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(row, text="›", command=lambda: scale.set(min(255, scale.get() + 1)),
                  **btn).pack(side=tk.LEFT, anchor="s", pady=(0, 4))
        return scale

    def _on_auto_change(self):
        state = tk.DISABLED if self.auto_var.get() else tk.NORMAL
        for s in (self.min_slider, self.max_slider):
            s.config(state=state)
        self._on_slider_change()

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
        paths = filedialog.askopenfilenames(
            title=t("dlg.select_image"),
            filetypes=[(t("common.images"), "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       (t("common.all_files"), "*.*")])
        if paths:
            self._add_images(list(paths))

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
        if not self.output_root:
            self.output_root = folder
            self._exporter = Exporter(folder)
        self._add_images(paths)
        self._set_status(t("status.folder_loaded", n=len(paths), folder=folder))

    def _add_images(self, paths: list[str]):
        """Agrega imágenes a la lista (sin repetir) y muestra la primera nueva."""
        current = [str(p) for p in self.project.images]
        new = [p for p in paths if str(p) not in current]
        if not new:
            if paths:
                self.batch_index = current.index(str(paths[0]))
                self._load_single(self.batch_paths[self.batch_index])
                self._sync_listbox()
            return
        self.project.images = [Path(p) for p in current + new]
        self.project.current_index = len(current)
        self._update_batch_list()
        self._show_panel(self.left_panel, True)
        self._load_single(new[0])
        self._sync_listbox()

    def _remove_current_image(self):
        path = self.current_image_path
        if not path:
            return
        images = [p for p in self.project.images if str(p) != str(path)]
        self.results_cache.pop(path, None)
        self.project.images = images
        self.project.current_index = min(self.project.current_index, max(len(images) - 1, 0))
        self._update_batch_list()
        if images:
            self._load_single(self.batch_paths[self.batch_index])
            self._sync_listbox()

    def _clear_images(self):
        if self.project.images and messagebox.askyesno(
                t("menu.clear_images"), t("msg.clear_images"), parent=self.root):
            self.project.images = []
            self.project.current_index = 0
            self._invalidate_results()
            self._update_batch_list()
            self._show_panel(self.left_panel, False)

    # ── Métodos de zoom (llamados por botones) ───────────────────────────────

    def _set_zoom_tool(self, tool: str):
        if self.zoom_ctrl is None:
            return
        if self.roi_selector and self.roi_selector.active:
            self.roi_selector.stop()
        self.zoom_ctrl.set_tool(tool)
        if tool == "zoom_area":
            self._set_status(t("view.zoom_area_hint"))

    def _on_zoom_tool_change(self, tool: str):
        for name, btn in getattr(self, "_zoom_tools", {}).items():
            btn.select(name == tool)
        if tool == "pan":
            self._set_status(t("status.ready"))

    def _open_style(self):
        """Colores de la máscara, contornos y puntos (se guardan en el proyecto)."""
        from fenotit.gui.style_popup import StylePopup
        if self._style_popup is not None and self._style_popup.winfo_exists():
            self._style_popup.destroy()
            self._style_popup = None
            return
        self._style_popup = StylePopup(self.root, self._style_btn, self.project.display.get("style"),
                                       COLORS, self._on_style_change)

    def _close_style_popup(self, event=None):
        if event is not None and event.widget is self._style_btn:
            return
        if self._style_popup is not None and self._style_popup.winfo_exists():
            self._style_popup.destroy()
        self._style_popup = None

    def _on_style_change(self, style: dict):
        self.project.display = {**self.project.display, "style": style}
        if self.last_result and self.step_names:
            self._jump_to_step(self.step_idx)
        self._show_preview()

    def do_zoom_100(self):
        if self.zoom_ctrl:
            self.zoom_ctrl.actual_size()

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
            self.roi_selector.redraw_shapes()

    def _load_single(self, path: str):
        self.current_image_path = path
        self.image_info_var.set(Path(path).name)
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
        self._on_slider_change()          # vista previa de la segmentación en la nueva imagen

        # Restaurar resultado previo si existe en cache
        if path in self.results_cache:
            self._display_result(self.results_cache[path], self.step_names_cache.get(path, []))
        else:
            # Nueva imagen sin resultados previos
            self.last_result = None
            self.step_names  = []
            self.step_idx    = 0
            self.scaler_right = ImageScaler()
            self.step_label_var.set("—")
            self._show_results_ui(False)
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
        if names:
            self.analysis_var.set(_analysis_label(names[0]))
            self.active_analysis = names[0]
            self._on_analysis_selected(None)

    def _render_analysis_cards(self):
        """Encabezado del panel: solo el análisis elegido (se cambia en el menú Análisis)."""
        box = self._analysis_cards
        for w in box.winfo_children():
            w.destroy()
        name = self._selected_analysis()
        if name is None:
            return
        bg = COLORS["accent_light"]
        card = tk.Frame(box, bg=bg, highlightthickness=1, highlightbackground=COLORS["accent"])
        card.pack(fill=tk.X, pady=1)
        tk.Label(card, text=_analysis_label(name), bg=bg, fg=COLORS["accent"],
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(fill=tk.X, padx=8, pady=(4, 0))
        desc = t(f"analysis.{_analysis_key(name)}.desc", ANALYSES[name].description)
        tk.Label(card, text=desc, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"], anchor="w",
                 justify="left", wraplength=230).pack(fill=tk.X, padx=8, pady=(0, 4))

    def _open_analysis(self, name: str):
        if name != self._selected_analysis():
            self.analysis_var.set(_analysis_label(name))
            self._on_analysis_selected(None)
        self._show_panel(self.right_panel, True)

    def _show_panel(self, panel, on: bool):
        self.paned.paneconfigure(panel, hide=not on)
        if on and not panel._expanded:
            panel.expand()
        self._after_panel_toggle()

    def _toggle_panel(self, panel):
        hidden = str(self.paned.panecget(panel, "hide")) in ("1", "true")
        self._show_panel(panel, hidden)

    def _uses_threshold(self) -> bool:
        name = self._selected_analysis()
        return name is None or ANALYSES[name].segmentation == "threshold"

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
        self._render_analysis_cards()
        if self._uses_threshold():
            self._seg_block.pack(fill=tk.X, before=self.config_container)
        else:
            self._seg_block.pack_forget()
        for w in self.config_container.winfo_children():
            w.destroy()
        stored = self.project.params.get(self.project.analysis, {})
        self._panel_area_unit = stored.get("_area_unit", "px")
        self.config_panel = ConfigPanel(
            self.config_container,
            schema=ANALYSES[name].params_schema,
            colors=COLORS, prefix=_analysis_key(name),
            units={"area": "mm²" if self._panel_area_unit == "mm" else "px²"})
        self.config_panel.set_values({k: v for k, v in stored.items() if not k.startswith("_")})
        self.config_panel.pack(fill=tk.BOTH, expand=True)
        self._sync_area_units()

    def _build_params(self) -> dict:
        p = {
            "color_space":      self.cs_var.get(),
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
            p["exclusions_norm"] = self.roi_selector.exclusions_normalized()
            p["_roi_selector"] = self.roi_selector
            p["_scaler_left"]  = self.scaler_left
        if self.config_panel:
            p.update(self.config_panel.get_values())
            p["area_unit"] = getattr(self, "_panel_area_unit", "px")
        p["auto_threshold"] = bool(self.auto_var.get())
        return p

    def _run_analysis(self, all_images: bool = False):
        """Ejecutar: la imagen actual; con all_images, todas las de la lista."""
        name = self._selected_analysis()
        if name not in ANALYSES:
            messagebox.showwarning(t("msg.no_analysis_title"),
                                   t("msg.no_analysis"))
            return
        if all_images:
            self._run_batch(name)
            return
        if not self.scaler_left.has_image:
            messagebox.showwarning(t("msg.no_image_title"),
                                   t("msg.no_image"))
            return
        reason = self._skip_reason(self._corr_info.get(self.current_image_path))
        if reason:
            messagebox.showwarning(t("corr.skip_title"), reason, parent=self.root)
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

        skipped: list[str] = []
        self._update_batch_list()
        self._sync_listbox()

        def worker():
            results = []
            for i, path in enumerate(self.batch_paths):
                img, info = self._load_corrected(path)
                if img is None:
                    continue
                reason = self._skip_reason(info)
                if reason:
                    skipped.append(path)
                    self.root.after(0, lambda p=path, r=reason: self._mark_skipped(p, r))
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

            self.root.after(0, lambda: self._on_batch_result(name, results, len(skipped)))

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
        if result.status == 'ok':
            self._remember_result(name, path, result)

        self._set_status(f"{name}: {i+1}/{total}…")
        self._update_batch_list()

        # Si es la imagen que está visible ahora, actualizar display
        if path == self.current_image_path and result.status == "ok":
            self._display_result(result, step_names, fresh=True)

    def _remember_result(self, name: str, path: str, result: AnalysisResult):
        """Guarda el resultado para exportar. Con carpeta de salida, además escribe
        imágenes y CSV al momento; sin ella queda en memoria hasta Exportar."""
        if self._exporter is None:
            self._exporter = Exporter(self.output_root or ".")
        live = bool(self.output_root)
        self._exporter.save_result(name, Path(path).name, result, save_step_images=live,
                                   decorate=self._decorate_fn(result, Path(path).stem))
        if live:
            self._exporter.append_to_csv(name, Path(path).name, result)

    def _on_result(self, name: str, result: AnalysisResult, path: str):
        self.root.config(cursor="")
        self.last_result = result
        if result.status == "error":
            messagebox.showerror(t("common.error"), result.error)
            self._set_status(f"{t('common.error')}: {result.error}")
            return
        names = list(result.step_images.keys())
        self.step_history[path] = names
        # Guardar en cache para restaurar al navegar
        self.results_cache[path]    = result
        self.step_names_cache[path] = names
        # Índice combinado por análisis
        if name not in self.all_results_by_analysis:
            self.all_results_by_analysis[name] = {}
        self.all_results_by_analysis[name][path] = result
        self._display_result(result, names, fresh=True)
        self._update_batch_list()
        self._remember_result(name, path, result)
        self._set_status(t("status.done", name=_analysis_label(name),
                           detail=self._exporter.results_dir if self.output_root else result.stats))
        reused = result.extra.get("reused")
        if reused:
            items = ", ".join(t(f"step.{k}", k) for k in reused)
            self._set_status(self.status_var.get() + "  ·  " + t("status.reused", items=items))
        self._refresh_history()
        self._update_step_active(3)
        self._log_process(name)
        # Auto-actualizar panel de gráficos
        if self._chart_panel and result.measurements:
            self._chart_panel.load(result.measurements)

    def _on_batch_result(self, name: str, results: list, n_skipped: int = 0):
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
                self._display_result(result, self.step_names_cache.get(cur, []))
                self._refresh_history()
                self._update_step_active(3)

        msg = t("status.batch_done", name=_analysis_label(name), ok=ok_count, n=len(results) + n_skipped)
        if n_skipped:
            msg += "  ·  " + t("status.skipped", n=n_skipped)
        if self.output_root:
            msg += f" — {Path(self.output_root) / 'results'}"
        self._set_status(msg)
        if n_skipped:
            messagebox.showwarning(t("corr.skip_title"), t("corr.skip_summary", n=n_skipped), parent=self.root)

    def _toggle_results_panel(self):
        """Oculta o muestra la tabla y los gráficos para dar espacio a las imágenes."""
        self._results_visible = not self._results_visible
        self._layout_results()
        self.root.after(50, self._refit_images)

    def _layout_results(self):
        """Barra de vistas y tabla/gráficos: ocultas hasta que la imagen tenga resultados."""
        self._results_bar.pack_forget()
        self._bottom_nb.pack_forget()
        if self._results_shown:
            self._results_bar.pack(fill=tk.X, after=self._canvas_row)
            if self._results_visible:
                self._bottom_nb.pack(fill=tk.BOTH, expand=False, padx=6, pady=(2, 0), after=self._results_bar)
        arrow = "▾" if self._results_visible else "▴"
        self._results_btn.config(text=f"{arrow} {t('results.panel')}")

    def _show_results_ui(self, on: bool):
        if on == self._results_shown:
            return
        self._results_shown = on
        if not on:
            self._view_ctrl.place_forget()
        self._layout_results()
        self.root.after(50, self._refit_images)

    def _build_view_controls(self):
        """Controles de la leyenda dentro del recuadro de resultado (esquina inferior derecha)."""
        from fenotit.core.pipeline.views import COLOR_FORMATS
        bg = COLORS["bg_card"]
        box = tk.Frame(self.canvas_right, bg=bg, highlightthickness=1,
                       highlightbackground=COLORS["border"], padx=2)
        self._view_ctrl = box
        self.legend_var = tk.BooleanVar(value=True)
        tk.Checkbutton(box, text=t("view.legend"), variable=self.legend_var,
                       command=self._on_display_change, bg=bg, fg=COLORS["text"],
                       selectcolor=bg, activebackground=bg, font=FONTS["small"]).pack(side=tk.LEFT)
        size_kw = dict(bg=bg, fg=COLORS["accent"], relief="flat", bd=0,
                       font=FONTS["small"], cursor="hand2", padx=3, pady=0)
        a_minus = tk.Button(box, text="A−", command=lambda: self._legend_size(0.8), **size_kw)
        a_plus = tk.Button(box, text="A+", command=lambda: self._legend_size(1.25), **size_kw)
        a_minus.pack(side=tk.LEFT)
        a_plus.pack(side=tk.LEFT)
        Tooltip(a_minus, t("view.legend_smaller"))
        Tooltip(a_plus, t("view.legend_bigger"))
        self.color_fmt_var = tk.StringVar(value="RGB")
        self._fmt_combo = ttk.Combobox(box, textvariable=self.color_fmt_var, values=list(COLOR_FORMATS),
                                       state="readonly", width=4, font=FONTS["small"])
        self._fmt_combo.bind("<<ComboboxSelected>>", self._on_display_change)

    def _update_view_controls(self, name: str):
        """Solo si la vista tiene leyenda; el formato de color solo en la vista de color."""
        spec = (self.last_result.extra.get("legends") or {}).get(name) if self.last_result else None
        if not spec:
            self._view_ctrl.place_forget()
            return
        if spec.get("colors") is not None:
            self._fmt_combo.pack(side=tk.LEFT, padx=(2, 1), pady=1)
        else:
            self._fmt_combo.pack_forget()
        self._view_ctrl.place(relx=1.0, rely=1.0, anchor="se", x=-6, y=-6)

    def _after_panel_toggle(self):
        self.root.after(50, self._refit_images)

    def _refit_images(self):
        self._do_resize()
        if self.zoom_ctrl and self.scaler_left.has_image:
            self.do_zoom_fit()

    def _display_result(self, result: AnalysisResult, step_names: list[str], fresh: bool = False):
        """Muestra un resultado. Tras correr: la medición que se recalculó (si solo cambió
        esa); si no, se conserva la vista que se estaba viendo; si no existe, la primera."""
        prev = self.step_names[self.step_idx] if self.step_names and 0 <= self.step_idx < len(self.step_names) else None
        self.last_result = result
        self.step_names = list(step_names)
        self._show_results_ui(True)
        self.stats_var.set("   |   ".join(f"{k}: {v}" for k, v in result.stats.items()))
        if not self.step_names:
            self._update_table(result)
            return
        changed = result.extra.get("default_view") if fresh else None
        for name in (changed, prev, result.extra.get("first_view")):
            if name in self.step_names:
                self._jump_to_step(self.step_names.index(name))
                return
        self._jump_to_step(len(self.step_names) - 1)

    # ── Tabla ─────────────────────────────────────────────────────────────────

    def _update_table(self, result: AnalysisResult, view: str | None = None):
        """Cada vista tiene su tabla (Conteo, Morfometría, Forma, Color...)."""
        self.table.delete(*self.table.get_children())
        rows = (result.extra.get("view_tables") or {}).get(view)
        if rows is None:
            rows = result.measurements
            cols = result.table_columns or (list(rows[0].keys()) if rows else [])
        else:
            cols = list(dict.fromkeys(k for r in rows for k in r))
        self.table["columns"] = cols
        if not rows:
            return
        rows_src = rows
        self.table["columns"] = cols
        for col in cols:
            self.table.heading(col, text=col, anchor="w")
            self.table.column(col, width=max(80, len(col)*8), anchor="w")
        for row in rows_src:
            self.table.insert("", "end",
                              values=["" if row.get(c) is None else str(row.get(c)) for c in cols])

    # ── Navegación pasos ──────────────────────────────────────────────────────

    def _prev_step(self):
        if self.last_result and self.step_names:
            self._jump_to_step(max(0, self.step_idx - 1))

    def _next_step(self):
        if self.last_result and self.step_names:
            self._jump_to_step(min(len(self.step_names) - 1, self.step_idx + 1))

    def _mark_colors(self, result: AnalysisResult) -> dict:
        from fenotit.core.pipeline import overlay
        return overlay.resolve(self.project.display.get("style"), auto=result.extra.get("contrast"))

    def _decorate_fn(self, result: AnalysisResult, image_name: str | None = None):
        """Marcas (contornos, puntos, números) con el estilo elegido y leyenda si está
        activada. Al exportar se dibujan a escala de la imagen; en pantalla, aparte."""
        from fenotit.core.pipeline import overlay
        from fenotit.core.pipeline.views import render_legend
        legends = result.extra.get("legends") or {}
        marks = result.extra.get("overlays") or {}
        base = result.extra.get("base_image")
        disp = self.project.display

        def decorate(name: str, img: np.ndarray, with_marks: bool = True, box: list | None = None) -> np.ndarray:
            if name in marks and base is not None:
                img = base.copy()
                if with_marks:
                    img = overlay.draw(img, marks[name], self._mark_colors(result))
            spec = legends.get(name)
            if not spec or not disp.get("legend", True):
                return img
            if image_name:
                spec = {**spec, "footer": image_name}
            return render_legend(img, spec, disp.get("color_format", "RGB"), disp.get("legend_scale", 1.0),
                                 box=box)
        return decorate

    def _on_display_change(self, _=None):
        self.project.display = {**self.project.display, "legend": bool(self.legend_var.get()),
                                "color_format": self.color_fmt_var.get()}
        if self.last_result and self.step_names:
            self._jump_to_step(self.step_idx)

    def _legend_size(self, factor: float):
        """A− / A+: tamaño de la leyenda (se guarda en el proyecto)."""
        scale = self.project.display.get("legend_scale", 1.0) * factor
        self.project.display = {**self.project.display, "legend_scale": round(min(4.0, max(0.4, scale)), 2)}
        self._on_display_change()

    def _show_step_img(self, name: str, img: np.ndarray):
        marks = None
        if self.last_result is not None:
            ovs = self.last_result.extra.get("overlays") or {}
            keep = []
            shown = Path(self.current_image_path).stem if self.current_image_path else None
            img = self._decorate_fn(self.last_result, shown)(name, img, with_marks=False, box=keep)
            if name in ovs and self.last_result.extra.get("base_image") is not None:
                marks = (ovs[name], self._mark_colors(self.last_result), keep)
        self.scaler_right.set_image(img)
        if self.zoom_ctrl:
            self.zoom_ctrl.set_right(img, marks)
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
        """Lista de imágenes; ✓ = ya analizada en esta sesión."""
        self.batch_listbox.delete(0, tk.END)
        for p in self.batch_paths:
            mark = "✓ " if p in self.results_cache else "   "
            self.batch_listbox.insert(tk.END, mark + Path(p).name)
        if self.batch_paths:
            self.batch_listbox.selection_set(min(self.batch_index, len(self.batch_paths) - 1))
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
        self._update_view_controls(name)
        self.step_label_var.set(
            f"{name}  ({idx+1}/{len(self.step_names)})")
        self._update_table(self.last_result, name)

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
        """Activa una herramienta de ROI; al terminar la forma se vuelve a mover la imagen."""
        if not self.roi_selector:
            return
        self.roi_selector.set_mode(mode)
        if self.zoom_ctrl:
            if self.zoom_ctrl.tool != "pan":
                self.zoom_ctrl.set_tool("pan")
            self.zoom_ctrl._update_cursor(self.canvas_left)
        hint = "roi.hint_polygon" if mode in ("polígono", "exclusión") else "roi.hint_rect"
        self._set_status(t(hint))

    def _on_roi_tool_done(self):
        if self.zoom_ctrl:
            self.zoom_ctrl._update_cursor(self.canvas_left)
        self._set_status(t("status.ready"))

    def _clear_roi(self):
        if self.roi_selector:
            self.roi_selector.stop()
            self.roi_selector.clear()

    def _open_scale_dialog(self):
        ScaleDialog(self.root,
                    current_image=self.scaler_left.original if self.scaler_left.has_image else None,
                    current_path=self.current_image_path,
                    on_scale_set=self._on_scale_set,
                    loader=self._load_corrected_image,
                    paths=self.batch_paths)

    def _toggle_legend(self):
        self.legend_var.set(not self.legend_var.get())
        self._on_display_change()

    def _open_log_folder(self):
        folder = Path(log.log_file()).parent
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)            # noqa: S606 (solo Windows)
            else:
                subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", str(folder)])
        except Exception:
            _log.exception("No se pudo abrir %s", folder)
            messagebox.showinfo(t("help.log"), str(folder), parent=self.root)

    def _show_shortcuts(self):
        rows = [("Ctrl+Enter", t("run.current")), ("Ctrl+Shift+Enter", t("run.all")),
                ("Ctrl+O", t("menu.add_images")), ("Ctrl+S", t("menu.save_project")),
                ("Ctrl+E", t("menu.export")), ("← →", t("help.key_images")),
                ("Ctrl+ +  −", t("help.key_zoom")), ("Ctrl+0", t("view.zoom_fit")),
                ("Ctrl+1", t("view.zoom_100")), ("Z", t("view.zoom_area")), ("H", t("view.pan")),
                (t("help.mouse_wheel"), t("help.key_zoom")), ("Shift + " + t("help.mouse_wheel"), t("help.key_pan")),
                (t("help.mouse_drag"), t("help.key_pan")), ("Esc", t("help.key_esc"))]
        messagebox.showinfo(t("help.shortcuts"), "\n".join(f"{k:<18}  {v}" for k, v in rows),
                            parent=self.root)

    def _bind_shortcuts(self):
        def typing() -> bool:
            w = self.root.focus_get()
            return isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Spinbox, tk.Text))

        def key(handler):
            return lambda e=None: None if typing() else (handler(), "break")[1]
        r = self.root
        r.bind_all("<Control-Return>", lambda e=None: self._run_analysis(False))
        r.bind_all("<Control-Shift-Return>", lambda e=None: self._run_analysis(True))
        r.bind_all("<Control-o>", lambda e=None: self._open_image())
        r.bind_all("<Control-s>", lambda e=None: self._save_project())
        r.bind_all("<Control-e>", lambda e=None: self._export_results())
        r.bind_all("<Left>", key(self._prev_image))
        r.bind_all("<Right>", key(self._next_image))
        # Zoom: Ctrl + / − / 0 / 1 (también sin Ctrl y con el teclado numérico)
        for seq, fn in (("plus", self.do_zoom_in), ("equal", self.do_zoom_in), ("KP_Add", self.do_zoom_in),
                        ("minus", self.do_zoom_out), ("KP_Subtract", self.do_zoom_out),
                        ("Key-0", self.do_zoom_fit), ("KP_0", self.do_zoom_fit),
                        ("Key-1", self.do_zoom_100), ("KP_1", self.do_zoom_100)):
            r.bind_all(f"<Control-{seq}>", lambda e=None, f=fn: (f(), "break")[1])
            r.bind_all(f"<{seq}>", key(fn))
        r.bind_all("<Key-z>", key(lambda: self._set_zoom_tool("zoom_area")))
        r.bind_all("<Key-h>", key(lambda: self._set_zoom_tool("pan")))
        r.bind_all("<Escape>", lambda e=None: self._on_escape())

    def _on_escape(self):
        if self.roi_selector and self.roi_selector.active:
            self.roi_selector.stop()
        elif self.zoom_ctrl and self.zoom_ctrl.tool != "pan":
            self.zoom_ctrl.set_tool("pan")

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
        self._sync_area_units()
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
            self._sync_area_units()
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