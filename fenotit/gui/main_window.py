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
import tkinter.font as tkfont
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from fenotit.core.image_io import ImageScaler, load_image
from fenotit.core.project import IMAGE_EXTS, PROJECT_EXT, Project, Scale, Segmentation
from fenotit.gui.roi.editor import AreaEditor
from fenotit.core.analysis.registry import ANALYSES, AnalysisResult
from fenotit.core import pipeline, units
from fenotit.core import batch
from fenotit.gui.config_panel import ConfigPanel
from fenotit.gui.zoom_controller import ZoomController
from fenotit.gui.image_list import ImageList
from fenotit.gui.toolbar import IconButton, Tooltip, icon
from fenotit.gui.calibration_dialogs import ScaleDialog
from fenotit.gui.corrections_dialog import CorrectionsDialog
from fenotit.core.corrections import pipeline as corrections
from fenotit.gui.export_dialog import ExportDialog
from fenotit.gui.charts import IntraImageChartPanel, BatchChartWindow
from fenotit.gui.theme import COLORS, FONTS

from fenotit import APP_NAME, __version__, log, settings
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


def _cell(v) -> str:
    """Texto de una celda: decimales recortados para que la tabla se lea."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e-3 and v != 0 else f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


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

def _peek_tab(host, image, command, side: str):
    """Pestañita que se asoma en el borde cuando un panel está oculto (como la de una
    carpeta): clic = mostrar el panel. side = left | right | bottom."""
    tab = tk.Label(host, image=image, bg=COLORS["bg_panel"], cursor="hand2", padx=3, pady=4,
                   highlightthickness=1, highlightbackground=COLORS["border"])
    tab.bind("<Button-1>", lambda e: command())
    tab.bind("<Enter>", lambda e: tab.config(bg=COLORS["accent_light"]))
    tab.bind("<Leave>", lambda e: tab.config(bg=COLORS["bg_panel"]))
    place = {"left": dict(x=0, y=24, anchor="nw"),
             "right": dict(relx=1.0, x=0, y=24, anchor="ne"),
             "bottom": dict(relx=0.5, rely=1.0, y=0, anchor="s")}[side]
    tab.show = lambda: (tab.place(**place), tab.lift())
    return tab


_SUMMARY_SKIP = ("image_width_px", "image_height_px")


def _summary(stats: dict, limit: int = 6) -> str:
    """Resumen de la fila de la imagen para la barra de estado. Sirve para cualquier
    análisis (conteos, distancias, color, modelos…): los valores que el análisis puso, sin
    vacíos ni la resolución, con pocos decimales."""
    items = [(k, v) for k, v in stats.items() if v not in (None, "") and k not in _SUMMARY_SKIP
             and not k.endswith("_per_px")]
    text = "  ·  ".join(f"{k}: {_cell(v)}" for k, v in items[:limit])
    return text + ("  ·  …" if len(items) > limit else "")


class CollapsiblePanel(tk.Frame):
    """Panel lateral que se puede ocultar. Abierto: un ícono pequeño en su esquina de
    arriba, del lado del centro (izquierdo: a la derecha; derecho: a la izquierda).
    Oculto: el panel desaparece y queda una pestañita en el borde de la zona central."""
    COLLAPSED_W = 14

    def __init__(self, parent, side: str, title: str,
                 colors: dict, default_width: int = 200, on_toggle=None, max_width: int = 420, **kw):
        super().__init__(parent, bg=colors["bg_panel"], **kw)
        self.max_width = max_width
        self.colors        = colors
        self._on_toggle    = on_toggle
        self.side          = side          # "left" o "right"
        self.title         = title
        self.default_width = default_width
        self._expanded     = True
        self._tab = None
        self.tab_host: tk.Widget | None = None    # dónde se asoma la pestañita (zona central)
        # ícono de panel (no una flecha: las flechas quedan para anterior/siguiente)
        self._icon = ImageTk.PhotoImage(icon(f"panel_{side}", 12, colors["accent"]))

        self.content = tk.Frame(self, bg=colors["bg_panel"])
        self.content.pack(fill=tk.BOTH, expand=True)
        self._corner = tk.Label(self, image=self._icon, bg=colors["bg_panel"], cursor="hand2", padx=2, pady=2)
        self._corner_place = (dict(relx=1.0, x=-4, y=5, anchor="ne") if side == "left"
                              else dict(x=4, y=5, anchor="nw"))
        self._corner.place(**self._corner_place)
        self._corner.bind("<Button-1>", lambda e=None: self.toggle())
        Tooltip(self._corner, t("view.toggle_panel"))

        self.configure(width=default_width)
        # el ancho lo decide el usuario (barra divisoria), no el contenido: así el panel no
        # crece solo cuando aparece un texto o un menú largo
        self.pack_propagate(False)
        self.bind("<Map>", lambda e: self._corner.lift(), add="+")

    def toggle(self):
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def _pane(self, **kw):
        if isinstance(self.master, tk.PanedWindow):
            self.master.paneconfigure(self, **kw)

    def collapse(self):
        """Oculta el panel y deja la pestañita para volver a abrirlo."""
        self._expanded = False
        width = self.winfo_width()
        if width > self.COLLAPSED_W * 3:          # recordar el ancho que dejó el usuario
            self.default_width = width
        self._pane(hide=True)
        if self.tab_host is not None:
            if self._tab is None:
                self._tab = _peek_tab(self.tab_host, self._icon, self.expand, self.side)
            self._tab.show()
        if self._on_toggle:
            self._on_toggle()

    def expand(self):
        self._expanded = True
        if self._tab is not None:
            self._tab.place_forget()
        self._pane(hide=False, width=self.default_width)
        self._corner.place(**self._corner_place)
        self._corner.lift()
        if self._on_toggle:
            self._on_toggle()

    def hide_all(self):
        """Sin panel ni pestañita (p. ej. en la pantalla de inicio)."""
        self._pane(hide=True)
        if self._tab is not None:
            self._tab.place_forget()


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
        self._chart_panel: IntraImageChartPanel | None = None
        # Resultados por análisis para gráficos combinados
        self.all_results_by_analysis: dict[str, dict] = {}
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
        self._update_start()
        self._saved_state = self._collect_state()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_title()

    # ── Estado del proyecto ───────────────────────────────────────────────────

    @property
    def results_cache(self) -> dict[str, AnalysisResult]:
        """Resultados del análisis elegido (foto -> resultado)."""
        return self._results.setdefault(getattr(self, "active_analysis", None) or "", {})

    @property
    def step_names_cache(self) -> dict[str, list]:
        return self._step_names.setdefault(getattr(self, "active_analysis", None) or "", {})

    def _clear_all_results(self, path: str | None = None):
        for store in (self._results, self._step_names):
            for per in store.values():
                if path is None:
                    per.clear()
                else:
                    per.pop(path, None)

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
        """mm/px de una foto: su propia escala; si no, la de todas (ArUco: la de cada foto)."""
        sc = self.project.scale_for(path)
        if sc.source == "aruco":
            info = self._corr_info.get(path) if path else None
            return info.mm_per_px if info else None
        return sc.mm_per_pixel

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
        d = self.project.corrections.distortion
        if info and d.enabled and d.mtx and "distortion" not in info.applied:
            return t("corr.skip_distortion", size="×".join(map(str, d.size or [])) or "?")
        if info and self.project.corrections.perspective.enabled and "perspective" not in info.applied:
            return t("corr.skip_aruco", ids=", ".join(map(str, info.aruco_missing)) or "?")
        return None

    def _mark_skipped(self, path: str, reason: str):
        self._skipped.add(path)
        self._update_batch_list()
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
        self._clear_all_results()
        self.all_results_by_analysis.clear()
        self.last_result = None
        self.canvas_right.delete("all")
        self._show_results_ui(False)

    def _clear_scale(self):
        self.project.apply_scale_to_all(Scale())
        self._set_status(t("status.scale_cleared"))
        self._update_corr_indicator()
        self._sync_area_units()

    def _update_corr_indicator(self):
        parts = [t(f"corr.short.{n}") for n in self.project.corrections.active()]
        mm = self.mm_per_pixel
        if self.project.scale_for(self.current_image_path).source != "none":
            u = self.project.unit
            parts.append(f"{units.per_px(mm, u):.6g} {u}/px" if mm else t("corr.short.scale_pending"))
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

    # ── Unidades en el panel (px sin escala, mm con escala) ─────────────────

    _UNIT_POWER = {"area": 2, "length": 1}

    def _unit_labels(self, unit: str) -> dict:
        """Etiquetas del panel: la unidad de la escala (µm, mm, cm, m) o px sin escala."""
        return {"area": f"{units.display(unit)}²", "length": units.display(unit)}

    def _area_unit(self) -> str:
        """La unidad de la escala si hay alguna (de todas las fotos o de alguna); si no, px."""
        p = self.project
        return p.unit if p.scale.source != "none" or p.image_scale else "px"

    def _has_area_units(self) -> bool:
        return any(i.get("unit") in self._UNIT_POWER
                   for i in ANALYSES[self.active_analysis].params_schema) if self.active_analysis in ANALYSES else False

    def _sync_area_units(self):
        """Si cambió la escala o su unidad, convierte las medidas del panel y la etiqueta."""
        panel = getattr(self, "config_panel", None)
        if panel is None or not self._has_area_units():
            return
        old, new = self._panel_area_unit, self._area_unit()
        if new != old:
            # mm de cada unidad; para pasar de/a px se usa la escala (al quitarla, la última usada)
            mpp = self.mm_per_pixel or getattr(self, "_panel_mpp", None)
            if old != "px" and new != "px":
                f = units.TO_MM[old] / units.TO_MM[new]
            elif mpp:
                f = mpp / units.TO_MM[new] if old == "px" else units.TO_MM[old] / mpp
            else:
                f = None
            if f:
                vals = panel.get_values(warn=False)
                changed = {}
                for i in ANALYSES[self.active_analysis].params_schema:
                    power = self._UNIT_POWER.get(i.get("unit"))
                    if power and i["key"] in vals:
                        changed[i["key"]] = round(vals[i["key"]] * f ** power, 6)
                panel.set_values(changed)
            else:
                _log.info("Sin escala para convertir; se conservan los números")
            self._panel_area_unit = new
        if new != "px" and self.mm_per_pixel:
            self._panel_mpp = self.mm_per_pixel
        panel.set_units(self._unit_labels(new))

    def _collect_state(self) -> dict:
        self._store_panel_params()
        self.project.segmentation = Segmentation(
            self.cs_var.get(), int(self.ch_var.get()),
            int(self.min_slider.get()), int(self.max_slider.get()), bool(self.auto_var.get()))
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
        settings.add_recent_project(self.project.file)
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

    def _show_cached_result(self):
        """Al cambiar de análisis: el resultado que ya tenía esta foto con ese análisis."""
        path = getattr(self, "current_image_path", None)
        if not path or not hasattr(self, "_results_bar"):
            return
        if path in self.results_cache:
            self._display_result(self.results_cache[path], self.step_names_cache.get(path, []))
        else:
            self.last_result, self.step_names, self.step_idx = None, [], 0
            self._show_results_ui(False)
            self.canvas_right.delete("all")
        self._refresh_history()
        self._update_batch_list()
        self._sync_listbox()

    def _open_recent(self, path: str):
        if self._confirm_discard():
            self.open_project_path(path)

    def _update_start(self):
        """Pantalla de inicio mientras no haya fotos; con fotos, Entrada y Resultado."""
        start = getattr(self, "_start", None)
        if start is None:
            return
        if self.project.images:
            start.place_forget()
            if not self.topbar.winfo_ismapped():          # vuelven la barra y el estado
                self.topbar.pack(fill=tk.X, side=tk.TOP, before=self.paned)
                self.statusbar.pack(fill=tk.X, side=tk.BOTTOM, before=self.paned)
        else:
            start.refresh()
            start.place(relx=0, rely=0, relwidth=1, relheight=1)
            start.lift()
            self.topbar.pack_forget()                      # en el inicio no hacen falta
            self.statusbar.pack_forget()

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
        if project.file:
            settings.add_recent_project(project.file)
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
        self._paint_pos_grid()
        self.color_fmt_var.set(project.display.get("color_format", "RGB"))

        name = _analysis_name(project.analysis) or self._selected_analysis()
        self.analysis_var.set(_analysis_label(name))
        self._on_analysis_selected(None)

        self._clear_all_results()
        self.all_results_by_analysis.clear()
        self._corr_info.clear()
        self.canvas_right.delete("all")
        self._update_batch_list()
        self._show_panel(self.left_panel, bool(project.images))
        self._show_panel(self.right_panel, bool(project.images))
        if project.current_image:
            self._load_single(str(project.current_image))
            self._sync_listbox()
        else:
            self.current_image_path = None
            self.canvas_left.delete("all")
        self._update_corr_indicator()
        self._update_start()
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
            default_width=200, on_toggle=self._after_panel_toggle, max_width=420)
        self.paned.add(self.left_panel, minsize=CollapsiblePanel.COLLAPSED_W,
                       width=200, stretch="never")

        # Centro
        self.center_frame = tk.Frame(self.paned, bg=COLORS["bg"])
        self.left_panel.tab_host = self.center_frame
        self.paned.add(self.center_frame, minsize=400, stretch="always")

        # Panel derecho colapsable
        self.right_panel = CollapsiblePanel(
            self.paned, side="right",
            title=t("panel.right"), colors=COLORS,
            default_width=260, on_toggle=self._after_panel_toggle, max_width=380)
        self.right_panel.tab_host = self.center_frame
        for w in (self.paned, self.left_panel, self.right_panel):
            w.bind("<Configure>", self._limit_panels, add="+")
        self.paned.bind("<ButtonPress-1>", self._sash_press)
        self.paned.bind("<B1-Motion>", self._sash_motion)
        self.paned.bind("<ButtonRelease-1>", self._sash_release)
        self.paned.add(self.right_panel, minsize=CollapsiblePanel.COLLAPSED_W,
                       width=260, stretch="never")      # si no, el último panel se queda con todo el espacio que sobra

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
        ttk.Style(self.root).configure("Thin.Horizontal.TProgressbar", thickness=8)
        self._progress_bar = ttk.Progressbar(self.statusbar, length=160, mode="determinate",
                                             style="Thin.Horizontal.TProgressbar")
        self._cancel_link = tk.Label(self.statusbar, text=t("common.cancel"), bg=COLORS["bg_panel"],
                                     fg=COLORS["accent"], font=FONTS["small"], cursor="hand2", padx=4)
        self._cancel_link.bind("<Button-1>", lambda e: self._cancel_batch())
        self.preview_var = tk.StringVar(value="")
        tk.Label(self.statusbar, textvariable=self.preview_var, bg=COLORS["bg_panel"],
                 fg=COLORS["accent2"], font=FONTS["small"], padx=8).pack(side=tk.RIGHT)
        self.image_info_var = tk.StringVar(value="")
        tk.Label(self.statusbar, textvariable=self.image_info_var, bg=COLORS["bg_panel"],
                 fg=COLORS["text"], font=FONTS["small"], padx=8).pack(side=tk.RIGHT)
        self._status_label = tk.Label(self.statusbar, textvariable=self.status_var,
                                      bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                                      font=FONTS["small"], anchor="w", padx=10)
        self._status_label.pack(fill=tk.X)

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
        self._drop(self.topbar, t("menu.calibration"), [
            (t("menu.corrections"), self._open_corrections),
            None,
            (t("menu.cal_scale"), self._open_scale_dialog),
            (t("menu.scale_manual"), self._calibrate_scale),
            (t("menu.scale_clear"), self._clear_scale),
        ])
        self._drop(self.topbar, t("menu.areas"), [
            (t("roi.area_rect"), lambda: self._set_roi_mode("include_rect"), "R"),
            (t("roi.area_polygon"), lambda: self._set_roi_mode("include_poly"), "P"),
            (t("roi.exclude_rect"), lambda: self._set_roi_mode("exclude_rect"), "X"),
            (t("roi.exclude_polygon"), lambda: self._set_roi_mode("exclude_poly"), "Shift+X"),
            None,
            (t("roi.select"), lambda: self._set_roi_mode("select"), "S"),
            (t("roi.apply_all"), self._apply_areas_to_all),
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

            # Áreas de análisis y zonas excluidas de esta foto
            from fenotit.core import roi as areas
            inc, exc = areas.masks(self.project.roi_for(self.current_image_path),
                                   mask.shape[1], mask.shape[0])
            if inc is not None:
                mask = cv2.bitwise_and(mask, inc)
            if exc:
                mask = mask.copy()
                cv2.fillPoly(mask, [pts for pts, _ in exc], 0)

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
        self.batch_label = tk.Label(nav, text="—", bg=COLORS["bg_panel"],     # anterior/siguiente: ← → o clic
                                    fg=COLORS["text_muted"], font=FONTS["small"])
        self.batch_label.pack(side=tk.RIGHT, padx=4)

        # Lista (buscador, lista o cuadrícula): alto según las fotos. Si CAPAS (y, después,
        # el inspector) necesita espacio, la lista cede, hasta dejar ver un par de fotos
        self.image_list = ImageList(lf, COLORS, FONTS, on_select=self._select_image,
                                    on_delete=self._remove_current_image,
                                    mode=settings.get("image_view", "list"),
                                    on_mode=lambda m: settings.set("image_view", m),
                                    max_height=self._image_list_room)
        self.image_list.pack(fill=tk.X)

        # Capas del resultado de la imagen actual (con barra solo si no caben)
        views = self._layers = tk.Frame(lf, bg=COLORS["bg_panel"])
        views.pack(fill=tk.BOTH, expand=True)
        self._layers_title = self._section_lbl(views, t("left.history"))
        box = tk.Frame(views, bg=COLORS["bg_panel"])
        box.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self._layers_sb = ttk.Scrollbar(box, orient="vertical")
        cv = self._layers_canvas = tk.Canvas(box, bg=COLORS["bg_panel"], highlightthickness=0, bd=0,
                                             yscrollcommand=self._layers_sb.set)
        self._layers_sb.config(command=cv.yview)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.history_frame = tk.Frame(cv, bg=COLORS["bg_panel"])
        win = cv.create_window(0, 0, window=self.history_frame, anchor="nw")
        cv.bind("<Configure>", lambda e: (cv.itemconfig(win, width=e.width), self._layers_scroll()))
        self.history_frame.bind("<Configure>", lambda e: self._layers_scroll())
        for ev, d in (("<Button-4>", -1), ("<Button-5>", 1)):
            cv.bind_all(ev, lambda e, d=d: self._layers_wheel(e, d), add="+")
        cv.bind_all("<MouseWheel>", lambda e: self._layers_wheel(e, -1 if e.delta > 0 else 1), add="+")

    def _layers_need(self) -> int:
        """Alto que piden CAPAS (título + filas). El inspector sumará el suyo."""
        rows = self.history_frame.winfo_reqheight() if self.history_frame.winfo_children() else 0
        return self._layers_title.winfo_reqheight() + max(rows, 22) + 12

    def _image_list_room(self) -> int:
        """Tope de la lista de imágenes: 45 % del espacio, y menos si CAPAS lo necesita."""
        lf = self.left_panel.content
        room = lf.winfo_height() - self.image_list.winfo_y()
        return min(int(room * 0.45), room - self._layers_need())

    def _layers_scroll(self):
        cv = self._layers_canvas
        h = self.history_frame.winfo_reqheight()
        cv.config(scrollregion=(0, 0, cv.winfo_width(), h))
        need = h > cv.winfo_height() + 1
        if need != self._layers_sb.winfo_ismapped():
            if need:
                self._layers_sb.pack(side=tk.RIGHT, fill=tk.Y, before=cv)
            else:
                self._layers_sb.pack_forget()
                cv.yview_moveto(0)

    def _layers_wheel(self, e, d):
        w = e.widget
        while w is not None and w is not self._layers_canvas:
            w = getattr(w, "master", None)
        if w is not None and self._layers_sb.winfo_ismapped():
            self._layers_canvas.yview_scroll(d, "units")

    # ── Panel central ─────────────────────────────────────────────────────────

    def _build_center_panel(self):
        cf = self.center_frame

        canvas_row = tk.Frame(cf, bg=COLORS["bg"])
        canvas_row.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 2))
        canvas_row.rowconfigure(0, weight=1)
        # Cada lado: título + ojo (ocultar) y su imagen. Oculto, queda una franja con el
        # ojo tachado para volver a mostrarlo.
        self._cols, self._strips, self._hidden = {}, {}, {"input": False, "result": False}
        for side, title in (("input", t("center.input")), ("result", t("center.result"))):
            col = tk.Frame(canvas_row, bg=COLORS["bg"])
            head = tk.Frame(col, bg=COLORS["bg"])
            head.pack(fill=tk.X)
            tk.Label(head, text=title, bg=COLORS["bg"], fg=COLORS["text_muted"],
                     font=FONTS["small"]).pack(side=tk.LEFT, padx=2)
            IconButton(head, "eye", lambda sd=side: self._toggle_side(sd), t("view.hide_side", name=title),
                       bg=COLORS["bg"], hover=COLORS["btn_hover"], active=COLORS["accent_light"],
                       size=14, color=COLORS["text_muted"]).pack(side=tk.RIGHT)
            canvas = tk.Canvas(col, bg=COLORS["bg_card"], highlightthickness=1,
                               highlightbackground=COLORS["border"])
            canvas.pack(fill=tk.BOTH, expand=True)
            strip = tk.Frame(canvas_row, bg=COLORS["bg_panel"], width=22)
            IconButton(strip, "eye_off", lambda sd=side: self._toggle_side(sd), t("view.show_side", name=title),
                       bg=COLORS["bg_panel"], hover=COLORS["btn_hover"], active=COLORS["accent_light"],
                       size=14, color=COLORS["accent"]).pack(side=tk.TOP, pady=4)
            self._cols[side], self._strips[side] = col, strip
            if side == "input":
                self.canvas_left = canvas
                canvas.config(cursor="crosshair")
            else:
                self.canvas_right = canvas
        self._layout_sides()

        self.areas = AreaEditor(self.canvas_left, self._on_areas_change)
        self.areas.on_done = self._on_roi_tool_done

        # ZoomController sincroniza ambos canvas
        self.zoom_ctrl = ZoomController(
            self.canvas_left, self.canvas_right,
            on_redraw=self._on_zoom_redraw)
        self.zoom_ctrl.set_zoom_var(self._zoom_pct_var)
        self.zoom_ctrl.on_tool_change = self._on_zoom_tool_change
        # Referencia para que ZoomController informe al ROI
        self.canvas_left._roi_selector_ref = self.areas
        self._build_area_tools()

        self._canvas_row = canvas_row
        from fenotit.gui.start_screen import StartScreen
        self._start = StartScreen(cf, COLORS, FONTS, self._open_image, self._open_folder,
                                  self._open_project, self._open_recent,
                                  links=((t("menu.language"), self._choose_language), (t("menu.about"), self._about)))
        self._build_view_controls()

        # Tabla y gráficos: ocultables como los paneles (ícono en su esquina; oculta = tira delgada)
        self.step_label_var = tk.StringVar(value="—")
        self._panel_bottom_icon = ImageTk.PhotoImage(icon("panel_bottom", 12, COLORS["accent"]))
        self._results_bar = _peek_tab(cf, self._panel_bottom_icon, self._toggle_results_panel, "bottom")

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
        self._results_visible = True          # preferencia del usuario (mostrar u ocultar)
        self._results_shown = False           # hay resultados para esta imagen
        corner = tk.Label(bottom_nb, image=self._panel_bottom_icon, bg=COLORS["bg_panel"], cursor="hand2",
                          padx=2, pady=2)
        corner.place(relx=1.0, x=-4, y=3, anchor="ne")
        corner.bind("<Button-1>", lambda e: self._toggle_results_panel())
        Tooltip(corner, t("view.toggle_results"))
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
        ysb.pack(side=tk.RIGHT,  fill=tk.Y)
        self._table_xsb = xsb                   # solo aparece si las columnas no caben
        self.table.pack(fill=tk.BOTH, expand=True)
        self.table.bind("<Configure>", self._fit_table_columns, add="+")

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
        self._section_lbl(rf, t("panel.analysis"), pad_left=26)       # deja sitio al ícono del panel
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

    def _section_lbl(self, parent, text, pad_left: int = 10):
        f = tk.Frame(parent, bg=COLORS["bg_panel"])
        f.pack(fill=tk.X)
        tk.Label(f, text=text, bg=COLORS["bg_panel"],
                 fg=COLORS["accent"],
                 font=("Segoe UI", 8, "bold"),
                 pady=5).pack(side=tk.LEFT, padx=(pad_left, 10))
        tk.Frame(f, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, side=tk.BOTTOM)
        return f

    def _divider(self, parent):
        tk.Frame(parent, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, pady=4)

    # ── Responsividad ─────────────────────────────────────────────────────────

    def _visible_panes(self):
        """Paneles visibles, como widgets (panes() devuelve nombres de Tcl)."""
        pw = self.paned
        names = {str(p) for p in pw.panes() if str(pw.panecget(p, "hide")) not in ("1", "true")}
        return [w for w in (self.left_panel, self.center_frame, self.right_panel) if str(w) in names]

    def _sash_press(self, event):
        """Arrastre propio de las barras divisorias: el panel lateral llega como máximo a su
        `max_width` y el centro nunca baja de 400 px (Tk solo, empujaba las otras barras)."""
        hit = self.paned.identify(event.x, event.y)
        if not hit or "sash" not in str(hit):
            self._sash_drag = None
            return
        self._sash_drag = int(hit[0])
        return "break"

    def _sash_motion(self, event):
        index = getattr(self, "_sash_drag", None)
        if index is None:
            return
        pw = self.paned
        visible = self._visible_panes()
        if index + 1 >= len(visible):
            return "break"
        sw = int(pw.cget("sashwidth"))
        total = pw.winfo_width()
        lo, hi = CollapsiblePanel.COLLAPSED_W, total - CollapsiblePanel.COLLAPSED_W - sw
        if visible[index] is self.left_panel:                     # barra izquierda
            hi = min(self.left_panel.max_width,
                     (pw.sash_coord(index + 1)[0] if index + 1 < len(visible) - 1 else total) - 400 - sw)
        elif visible[index + 1] is self.right_panel:              # barra derecha
            start = pw.sash_coord(index - 1)[0] + sw if index > 0 else 0
            lo = max(total - self.right_panel.max_width - sw, start + 400)
        x = int(min(max(event.x, lo), max(lo, hi)))
        pw.sash_place(index, x, 1)
        return "break"

    def _sash_release(self, _event=None):
        if getattr(self, "_sash_drag", None) is None:
            return
        self._sash_drag = None
        self._after_panel_toggle()
        return "break"

    def _limit_panels(self, _event=None):
        """Si al achicar la ventana un panel quedó aplastado, al agrandarla recupera su ancho."""
        pw = self.paned
        if pw.winfo_width() <= 1 or getattr(self, "_restoring", False):
            return
        visible = self._visible_panes()
        room = self.center_frame.winfo_width() - 400
        for panel in (self.left_panel, self.right_panel):
            w_now = panel.winfo_width() if panel in visible else 0
            if (panel.winfo_ismapped() and panel._expanded and CollapsiblePanel.COLLAPSED_W <= w_now
                    < 0.6 * panel.default_width and room > 40):
                self._restoring = True
                target = int(min(panel.default_width, w_now + room))

                def restore(panel=panel, target=target):
                    self._restoring = False
                    vis = self._visible_panes()
                    if panel is self.left_panel:
                        pw.sash_place(0, target, 1)
                    else:
                        pw.sash_place(len(vis) - 2, pw.winfo_width() - target - int(pw.cget("sashwidth")), 1)
                self.root.after(60, restore)
                return

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
        self._update_start()
        self._load_single(new[0])
        self._sync_listbox()

    def _remove_current_image(self):
        path = self.current_image_path
        if not path:
            return
        images = [p for p in self.project.images if str(p) != str(path)]
        self._clear_all_results(path)
        self.project.image_roi.pop(self.project.key(path), None)
        self.project.image_scale.pop(self.project.key(path), None)
        self.project.images = images
        self.project.current_index = min(self.project.current_index, max(len(images) - 1, 0))
        self._update_batch_list()
        if images:
            self._load_single(self.batch_paths[self.batch_index])
            self._sync_listbox()
        else:
            self._show_no_image()

    def _clear_images(self):
        if self.project.images and messagebox.askyesno(
                t("menu.clear_images"), t("msg.clear_images"), parent=self.root):
            self.project.images = []
            self.project.current_index = 0
            self.project.image_roi, self.project.image_scale = {}, {}
            self._invalidate_results()
            self._update_batch_list()
            self._show_no_image()
            self._show_panel(self.left_panel, False)

    def _show_no_image(self):
        """Sin fotos: entrada y resultado vacíos y nada que ejecutar."""
        self.current_image_path = None
        self.scaler_left = ImageScaler()
        self.scaler_right = ImageScaler()
        self.last_result, self.step_names = None, []
        if self.zoom_ctrl:
            self.zoom_ctrl.set_images(None, None)
        self.canvas_left.delete("all")
        self.canvas_right.delete("all")
        if self.areas:
            self.areas.stop()
            self.areas.set_shapes([])
        self._update_area_scope()
        self._show_results_ui(False)
        self.image_info_var.set("")
        self.preview_var.set("")
        self._update_start()

    # ── Métodos de zoom (llamados por botones) ───────────────────────────────

    def _set_zoom_tool(self, tool: str):
        if self.zoom_ctrl is None:
            return
        if self.areas and self.areas.active:
            self.areas.stop()
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
        pass

    def _load_single(self, path: str):
        self.current_image_path = path
        self.image_info_var.set(Path(path).name)
        img, info = self._load_corrected(path)
        if img is None:
            messagebox.showerror(t("common.error"), t("msg.image_unreadable", name=Path(path).name),
                                 parent=self.root)
            return
        self.scaler_left.set_image(img)
        self.image_info_var.set(f"{Path(path).name}  ·  {img.shape[1]} × {img.shape[0]} px")
        self._update_corr_indicator()
        self.preview_var.set("")
        self._update_step_active(1)
        # Al cargar imagen nueva: fit-to-canvas (imagen completa visible)
        if self.zoom_ctrl:
            self.zoom_ctrl._img_right = None
            self.zoom_ctrl.set_left(self.scaler_left.original)
        if self.areas:
            self.areas.set_shapes(self.project.roi_for(path))
        self._update_area_scope()
        self._on_slider_change()          # vista previa de la segmentación en la nueva imagen

        self._set_status(t("status.ready"))           # el nombre ya está a la derecha de la barra
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
        if canvas is self.canvas_left and self.areas:
            self.areas.set_image_offset(
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
        if on:
            panel.expand()
        else:
            panel.hide_all()
            self._after_panel_toggle()

    def _toggle_panel(self, panel):
        hidden = str(self.paned.panecget(panel, "hide")) in ("1", "true")
        panel.expand() if hidden else panel.collapse()

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
            units=self._unit_labels(self._panel_area_unit))
        self.config_panel.set_values({k: v for k, v in stored.items() if not k.startswith("_")})
        self.config_panel.pack(fill=tk.BOTH, expand=True)
        self._show_cached_result()
        self._sync_area_units()

    def _build_params(self) -> dict:
        p = {
            "color_space":      self.cs_var.get(),
            "color_space_code": CV2_CODES.get(self.cs_var.get()),
            "channel_idx":      self.ch_var.get(),
            "min_val":          self.min_slider.get(),
            "max_val":          self.max_slider.get(),
            "mm_per_pixel":     self.mm_per_pixel,
            "length_unit":      self.project.unit,
        }
        p["roi_shapes"] = self.project.roi_for(self.current_image_path)
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
            result.extra["params"] = params
            self.root.after(0, lambda: self._on_result(name, result, path))
        threading.Thread(target=worker, daemon=True).start()

    def _run_batch(self, name: str):
        """Todas las fotos en procesos aparte (core.batch): vuelven tablas y marcas, no
        imágenes; la vista de una foto se recalcula al abrirla."""
        if not self.batch_paths:
            messagebox.showwarning(t("msg.no_batch_title"), t("msg.no_batch"))
            return
        if self._batch is not None:
            return
        paths = self.batch_paths
        total = len(paths)
        self._set_status(t("status.running_batch", name=_analysis_label(name), n=total))
        params = self._build_params()
        jobs = []
        for path in paths:
            sc = self.project.scale_for(path)
            aruco = sc.source == "aruco"
            jobs.append(batch.Job(path, dict(params, mm_per_pixel=None if aruco else sc.mm_per_pixel,
                                             roi_shapes=self.project.roi_for(path)), aruco_scale=aruco))
        self._pending = set(paths)                 # al volver a correr, las marcas se rehacen
        self._skipped -= self._pending
        self._progress(0, total)
        self._update_batch_list()
        self._sync_listbox()
        after = self.root.after
        self._batch = batch.BatchRunner(
            name, jobs, self.project.corrections,
            on_outcome=lambda i, n, o: after(0, lambda: self._on_batch_outcome(name, i, n, o)),
            on_done=lambda cancelled: after(0, lambda: self._on_batch_done(name, total, cancelled))).start()

    def _cancel_batch(self):
        if self._batch is not None:
            self._batch.cancel()
            self._set_status(t("status.cancelling"))

    def _on_batch_outcome(self, name: str, i: int, n: int, out: "batch.Outcome"):
        path = out.path
        self._pending.discard(path)
        if out.status == "skipped":
            d = self.project.corrections.distortion
            reason = t("corr.skip_distortion", size="×".join(map(str, d.size or [])) or "?") \
                if out.message == "distortion" else \
                t("corr.skip_aruco", ids=", ".join(map(str, out.aruco_missing)) or "?")
            self._mark_skipped(path, reason)
        elif out.status == "unreadable":
            self._mark_skipped(path, t("msg.image_unreadable", name=Path(path).name))
        else:
            result = out.result or AnalysisResult(status="error", error=out.message)
            if out.status == "error" and not result.error:
                result.error = out.message
            self._cache_result(name, path, result)
        self._set_status(t("status.batch_progress", name=_analysis_label(name), i=i, n=n))
        self._progress(i, n)
        self._update_batch_list()

    def _cache_result(self, name: str, path: str, result: AnalysisResult):
        """Guarda el resultado del análisis que corre (aunque el usuario cambie de
        análisis) y, si es la foto que se ve, lo muestra."""
        views = list(result.step_images) or result.extra.get("views", [])
        self._results.setdefault(name, {})[path] = result
        self._step_names.setdefault(name, {})[path] = views
        self.all_results_by_analysis.setdefault(name, {})[path] = result
        if path == self.current_image_path and result.status == "ok" and name == self.active_analysis:
            self._display_result(result, views, fresh=True)
            self._refresh_history()

    def _on_batch_done(self, name: str, total: int, cancelled: bool):
        self._batch = None
        self._pending.clear()
        self._progress(None)
        self._update_batch_list()
        store = self._results.get(name, {})
        ok = sum(1 for p in self.batch_paths if p in store and store[p].status == "ok")
        failed = sum(1 for p in self.batch_paths if p in store and store[p].status != "ok")
        n_skipped = sum(1 for p in self.batch_paths if p in self._skipped)
        key = "status.batch_cancelled" if cancelled else "status.batch_done"
        msg = t(key, name=_analysis_label(name), ok=ok, n=total)
        if n_skipped:
            msg += "  ·  " + t("status.skipped", n=n_skipped)
        if failed:
            msg += "  ·  " + t("status.failed", n=failed)
        self._set_status(msg)
        if n_skipped and not cancelled:
            messagebox.showwarning(t("corr.skip_title"), t("corr.skip_summary", n=n_skipped), parent=self.root)

    # ── Resultados livianos (lote): la vista se recalcula al abrir la foto ───────

    def _rehydrate(self, name: str, path: str, light: AnalysisResult):
        """Recalcula la foto con los mismos parámetros para tener sus imágenes."""
        if self._rehydrating == (name, path) or not self.scaler_left.has_image:
            return
        self._rehydrating = (name, path)
        image, params = self.scaler_left.original, light.extra.get("params") or self._build_params()
        self._set_status(t("status.preparing_view"))

        def worker():
            try:
                full = ANALYSES[name].func(image, params)
            except Exception as e:
                _log.exception("%s falló en %s", name, path)
                full = AnalysisResult(status="error", error=str(e))
            full.extra["params"] = params
            self.root.after(0, lambda: self._on_rehydrated(name, path, full))
        threading.Thread(target=worker, daemon=True).start()

    def _on_rehydrated(self, name: str, path: str, full: AnalysisResult):
        self._rehydrating = None
        if full.status != "ok" or self._results.get(name, {}).get(path) is None:
            return
        self._results[name][path] = full
        self.all_results_by_analysis.setdefault(name, {})[path] = full
        self._trim_memory(path)
        if path == self.current_image_path and name == self.active_analysis:
            self._display_result(full, list(full.step_images))
            self._refresh_history()
            if self.status_var.get() == t("status.preparing_view"):
                self._set_status(t("status.ready"))

    def _trim_memory(self, path: str):
        """Solo las 2 últimas fotos vistas guardan imágenes (cada una ~100 MB en 8 MP)."""
        if path in self._full_paths:
            self._full_paths.remove(path)
        self._full_paths.append(path)
        del self._full_paths[:-2]
        for name, per in self._results.items():
            for p, r in per.items():
                if p not in self._full_paths and r.step_images:
                    per[p] = batch.light(r)
                    self.all_results_by_analysis.get(name, {}).pop(p, None)

    def _on_result(self, name: str, result: AnalysisResult, path: str):
        self.root.config(cursor="")
        self.last_result = result
        if result.status == "error":
            messagebox.showerror(t("common.error"), result.error)
            self._set_status(f"{t('common.error')}: {result.error}")
            return
        names = list(result.step_images.keys())
        # Guardar en cache para restaurar al navegar
        self.results_cache[path]    = result
        self.step_names_cache[path] = names
        # Índice combinado por análisis
        if name not in self.all_results_by_analysis:
            self.all_results_by_analysis[name] = {}
        self.all_results_by_analysis[name][path] = result
        self._display_result(result, names, fresh=True)
        self._trim_memory(path)
        self._update_batch_list()
        self._set_status(t("status.done", name=_analysis_label(name),
                           detail=_summary(result.stats)))
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

    def _toggle_results_panel(self):
        """Oculta o muestra la tabla y los gráficos para dar espacio a las imágenes."""
        self._results_visible = not self._results_visible
        self._layout_results()
        self.root.after(50, self._refit_images)

    def _layout_results(self):
        """Barra de vistas y tabla/gráficos: ocultas hasta que la imagen tenga resultados."""
        self._results_bar.place_forget()
        self._bottom_nb.pack_forget()
        if self._results_shown:
            if self._results_visible:
                self._bottom_nb.pack(fill=tk.BOTH, expand=False, padx=6, pady=(4, 0), after=self._canvas_row)
            else:
                self._results_bar.show()

    def _show_results_ui(self, on: bool):
        if on == self._results_shown:
            return
        self._results_shown = on
        if not on:
            self._view_ctrl.place_forget()
        self._layout_results()
        self._layout_sides()
        self.root.after(50, self._refit_images)

    def _layout_sides(self):
        """Entrada y Resultado lado a lado. Sin resultados solo se ve la entrada; con
        resultados, el ojo de cada lado lo oculta (queda una franja para volver)."""
        has_result = getattr(self, "_results_shown", False)
        show = {"input": not (has_result and self._hidden["input"]),
                "result": has_result and not self._hidden["result"]}
        row = self._cols["input"].master
        for w in list(self._cols.values()) + list(self._strips.values()):
            w.grid_forget()
        col = 0
        for side in ("input", "result"):
            if show[side]:
                self._cols[side].grid(row=0, column=col, sticky="nsew", padx=2)
                row.columnconfigure(col, weight=1, uniform="side")
            elif has_result:
                self._strips[side].grid(row=0, column=col, sticky="ns")
                row.columnconfigure(col, weight=0, uniform="")
            col += 1

    def _toggle_side(self, side: str):
        other = "result" if side == "input" else "input"
        self._hidden[side] = not self._hidden[side]
        if self._hidden[side] and self._hidden[other]:      # nunca las dos ocultas
            self._hidden[other] = False
        self._layout_sides()
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
        # posición de la leyenda: cuadrícula 3×3 (el centro no se usa)
        self._pos_grid = tk.Canvas(box, width=25, height=19, bg=bg, highlightthickness=0, cursor="hand2")
        self._pos_grid.pack(side=tk.LEFT, padx=(4, 2), pady=1)
        self._pos_grid.bind("<Button-1>", self._pick_legend_pos)
        Tooltip(self._pos_grid, t("view.legend_pos"))
        self._paint_pos_grid()
        Tooltip(a_minus, t("view.legend_smaller"))
        Tooltip(a_plus, t("view.legend_bigger"))
        self.color_fmt_var = tk.StringVar(value="RGB")
        self._fmt_combo = ttk.Combobox(box, textvariable=self.color_fmt_var, values=list(COLOR_FORMATS),
                                       state="readonly", width=4, font=FONTS["small"])
        self._fmt_combo.bind("<<ComboboxSelected>>", self._on_display_change)

    _POS_GRID = (("tl", "tc", "tr"), ("ml", None, "mr"), ("bl", "bc", "br"))

    def _paint_pos_grid(self):
        c, cur = self._pos_grid, self.project.display.get("legend_pos", "tl") if hasattr(self, "project") else "tl"
        c.delete("all")
        for r, row in enumerate(self._POS_GRID):
            for k, pos in enumerate(row):
                if pos is None:
                    continue
                x, y = 2 + k * 8, 1 + r * 6
                c.create_rectangle(x, y, x + 6, y + 4, outline=COLORS["accent"],
                                   fill=COLORS["accent"] if pos == cur else "")

    def _pick_legend_pos(self, event):
        k, r = min(2, max(0, (event.x - 2) // 8)), min(2, max(0, (event.y - 1) // 6))
        pos = self._POS_GRID[r][k]
        if pos:
            self.project.display = {**self.project.display, "legend_pos": pos}
            self._paint_pos_grid()
            self._on_display_change()

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
        """Cada vista tiene su tabla (Conteo, Morfometría, Forma, Color...). Columnas de
        ancho fijo (las que no caben se ven con la barra de abajo) y clic en el título
        ordena de menor a mayor / de mayor a menor."""
        self.table.delete(*self.table.get_children())
        rows = (result.extra.get("view_tables") or {}).get(view)
        if rows is None:
            rows = result.measurements
            cols = result.table_columns or (list(rows[0].keys()) if rows else [])
        else:
            cols = list(dict.fromkeys(k for r in rows for k in r))
        self.table["columns"] = cols
        self._table_rows, self._table_cols, self._table_sort = list(rows or []), cols, (None, False)
        if not rows:
            return
        font, head = tkfont.nametofont("TkDefaultFont"), tkfont.nametofont("TkHeadingFont")
        self._table_widths = {}
        for col in cols:
            sample = [_cell(r.get(col)) for r in rows[:200]]
            width = max([head.measure(col + " ▼") + 20] + [font.measure(v) + 14 for v in sample])
            self._table_widths[col] = min(max(width, 60), 260)
            self.table.heading(col, text=col, anchor="w", command=lambda c=col: self._sort_table(c))
        self._fit_table_columns()
        self._fill_table()

    def _fit_table_columns(self, _event=None):
        """Pocas columnas: se reparten el ancho disponible (sin espacio en blanco).
        Muchas: cada una con su ancho y barra para desplazarse."""
        widths = getattr(self, "_table_widths", None)
        if not widths:
            self._table_hscroll(False)
            return
        avail = self.table.winfo_width()
        total = sum(widths.values())
        extra = (avail - total) / len(widths) if avail > total + 4 else 0
        for col, w in widths.items():
            self.table.column(col, width=int(w + extra), minwidth=50, stretch=False, anchor="w")
        self._table_hscroll(total > avail + 1)

    def _table_hscroll(self, need: bool):
        xsb = self._table_xsb
        if need and not xsb.winfo_ismapped():
            xsb.pack(side=tk.BOTTOM, fill=tk.X, before=self.table)
        elif not need and xsb.winfo_ismapped():
            xsb.pack_forget()
            self.table.xview_moveto(0)

    def _fill_table(self):
        self.table.delete(*self.table.get_children())
        for row in self._table_rows:
            self.table.insert("", "end", values=[_cell(row.get(c)) for c in self._table_cols])

    def _sort_table(self, col: str):
        """Ordena por esa columna; otro clic invierte el orden (▲ / ▼ en el título)."""
        last, desc = self._table_sort
        desc = not desc if last == col else False
        self._table_sort = (col, desc)

        def key(r):
            v = r.get(col)
            if isinstance(v, (int, float)) and v == v:
                return (0, float(v), "")
            try:
                return (0, float(v), "")
            except (TypeError, ValueError):
                return (1, 0.0, str(v or ""))
        self._table_rows.sort(key=key, reverse=desc)
        for c in self._table_cols:
            mark = (" ▼" if desc else " ▲") if c == col else ""
            self.table.heading(c, text=c + mark)
        self._fill_table()

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
        """Marcas con el estilo elegido y leyenda si está activada (core.pipeline.views)."""
        from fenotit.core.pipeline.views import decorate
        return decorate(result, self.project.display, image_name)

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
        self._step_image(-1)

    def _next_image(self):
        self._step_image(1)

    def _step_image(self, delta: int):
        """Anterior / siguiente entre las imágenes que se ven (respeta el buscador)."""
        if not self.batch_paths:
            return
        k = self.image_list.step(self.batch_index, delta)
        if k is not None and k != self.batch_index:
            self._select_image(k)

    def _select_image(self, index: int):
        self.batch_index = index
        self._load_single(self.batch_paths[index])
        self._sync_listbox()

    def _sync_listbox(self):
        self.image_list.set_current(self.batch_index)
        self._update_batch_label()

    def _update_batch_list(self):
        """Lista de imágenes; ✓ = ya analizada en esta sesión, ⚠ = omitida."""
        marks = {p: "error" for p in self._skipped}
        store = self.results_cache
        marks.update({p: "done" if store[p].status == "ok" else "error" for p in self.batch_paths if p in store})
        marks.update({p: "pending" for p in self._pending})
        if self.batch_paths:
            self.batch_index = min(self.batch_index, len(self.batch_paths) - 1)
        self.image_list.set_items(self.batch_paths, marks, self.batch_index if self.batch_paths else -1)
        self._update_batch_label()

    def _update_batch_label(self):
        n = len(self.batch_paths)
        self.batch_label.config(
            text=f"{self.batch_index+1} / {n}" if n else "—")

    # ── Historial ─────────────────────────────────────────────────────────────

    def _refresh_history(self):
        """Vistas de la foto actual, agrupadas por análisis (solo los que ya tienen
        resultado). Tocar una vista de otro análisis lo activa y la muestra al instante."""
        for w in self.history_frame.winfo_children():
            w.destroy()
        path = self.current_image_path
        if not path:
            return
        done = [n for n in ANALYSES if path in self._results.get(n, {})]
        for name in done:
            current = name == getattr(self, "active_analysis", None)
            if len(done) > 1:
                tk.Label(self.history_frame, text=_analysis_label(name), bg=COLORS["bg_panel"],
                         fg=COLORS["accent"] if current else COLORS["text_muted"],
                         font=("Segoe UI", 8, "italic"), anchor="w", padx=6).pack(fill=tk.X, pady=(4, 0))
            for i, view in enumerate(self._step_names.get(name, {}).get(path, [])):
                sel = current and i == self.step_idx
                bg = COLORS["accent_light"] if sel else COLORS["bg_panel"]
                row = tk.Frame(self.history_frame, bg=bg, cursor="hand2")
                row.pack(fill=tk.X)
                tk.Frame(row, bg=COLORS["accent"] if sel else bg, width=3).pack(side=tk.LEFT, fill=tk.Y)
                lbl = tk.Label(row, text=view, bg=bg, fg=COLORS["text"] if current else COLORS["text_muted"],
                               font=FONTS["small"], anchor="w", padx=8, pady=2, cursor="hand2")
                lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
                for wdg in (row, lbl):
                    wdg.bind("<Button-1>", lambda e, a=name, v=view: self._open_view(a, v))
                    wdg.bind("<Enter>", lambda e, r=row, l=lbl: (r.config(bg=COLORS["accent_light"]),
                                                                 l.config(bg=COLORS["accent_light"])))
                    wdg.bind("<Leave>", lambda e, r=row, l=lbl, c=bg: (r.config(bg=c), l.config(bg=c)))
        self.image_list.refit()                # más capas = la lista de fotos cede espacio

    def _step_view(self, delta: int):
        """↑ ↓: capa anterior / siguiente de la foto, pasando de un análisis al otro."""
        path = self.current_image_path
        if not path:
            return
        flat = [(n, v) for n in ANALYSES if path in self._results.get(n, {})
                for v in self._step_names.get(n, {}).get(path, [])]
        if not flat:
            return
        cur = (self.active_analysis, self.step_names[self.step_idx]) \
            if self.step_names and 0 <= self.step_idx < len(self.step_names) else None
        k = flat.index(cur) + delta if cur in flat else 0
        if 0 <= k < len(flat):
            self._open_view(*flat[k])

    def _open_view(self, analysis: str, view: str):
        if analysis != self.active_analysis:
            self.analysis_var.set(_analysis_label(analysis))
            self._on_analysis_selected(None)
        if view in self.step_names:
            self._jump_to_step(self.step_names.index(view))

    def _jump_to_step(self, idx: int):
        if not self.last_result or idx >= len(self.step_names):
            return
        self.step_idx = idx
        name = self.step_names[idx]
        img = self.last_result.step_images.get(name)
        if img is None:                    # resultado del lote: datos listos, marcas en camino
            if self.scaler_left.has_image:
                self._show_step_img(name, self.scaler_left.original)
            self._rehydrate(self.active_analysis, self.current_image_path, self.last_result)
        else:
            self._show_step_img(name, img)
        self._update_view_controls(name)
        self.step_label_var.set(
            f"{name}  ({idx+1}/{len(self.step_names)})")
        self._update_table(self.last_result, name)
        self._refresh_history()

    # ── ROI ───────────────────────────────────────────────────────────────────

    def _set_roi_mode(self, mode: str):
        """Herramienta de áreas; al terminar la forma se vuelve a mover la imagen
        (la de seleccionar queda activa hasta Esc)."""
        if not self.areas or not self.current_image_path:
            return
        self.areas.set_mode(mode)
        if self.zoom_ctrl:
            if self.zoom_ctrl.tool != "pan":
                self.zoom_ctrl.set_tool("pan")
            self.zoom_ctrl._update_cursor(self.canvas_left)
        self._paint_area_tools()
        hint = {"include_poly": "roi.hint_polygon", "exclude_poly": "roi.hint_polygon",
                "select": "roi.hint_select"}.get(mode, "roi.hint_rect")
        self._set_status(t(hint))

    def _on_roi_tool_done(self):
        if self.zoom_ctrl:
            self.zoom_ctrl._update_cursor(self.canvas_left)
        self._paint_area_tools()
        self._set_status(t("status.ready"))

    def _clear_roi(self):
        """Borra las áreas de esta foto (las demás no cambian)."""
        if self.areas and self.current_image_path:
            self.areas.stop()
            self.areas.clear()

    def _on_areas_change(self, shapes: list):
        """Lo que se dibuja o borra queda como áreas propias de esta foto."""
        if self.current_image_path:
            self.project.set_roi(self.current_image_path, shapes)
        self._update_area_scope()
        self._update_step_active(2)
        self._show_preview()

    def _apply_areas_to_all(self):
        path = self.current_image_path
        if not path:
            return
        n = self.project.others_with_own_roi(path)
        if n and not messagebox.askyesno(t("roi.apply_all"), t("roi.apply_all_warn", n=n), parent=self.root):
            return
        self.project.apply_roi_to_all(self.project.roi_for(path))
        self._update_area_scope()
        self._set_status(t("roi.applied_all", n=len(self.project.images)))

    def _build_area_tools(self):
        """Abajo a la izquierda de la entrada: de qué fotos son las áreas (esta foto /
        todas) con "Aplicar a todas" y "Borrar". Las herramientas están en el menú Áreas."""
        bg = COLORS["bg_card"]
        self._area_btns = {}
        scope = tk.Frame(self.canvas_left, bg=bg, highlightthickness=1, highlightbackground=COLORS["border"])
        self._area_scope = scope
        self._area_scope_var = tk.StringVar()
        tk.Label(scope, textvariable=self._area_scope_var, bg=bg, fg=COLORS["text"],
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(6, 4))
        self._apply_all_btn = tk.Button(scope, text=t("roi.apply_all_short"), command=self._apply_areas_to_all,
                                        bg=bg, fg=COLORS["accent"], relief="flat", bd=0, font=FONTS["small"],
                                        cursor="hand2", padx=4)
        self._clear_area_btn = tk.Button(scope, text=t("roi.clear_short"), command=self._clear_roi, bg=bg,
                                         fg=COLORS["text_muted"], relief="flat", bd=0, font=FONTS["small"],
                                         cursor="hand2", padx=4)
        self._clear_area_btn.pack(side=tk.RIGHT, padx=(0, 4))

    def _toggle_area_tool(self, mode: str):
        if self.areas and self.areas.mode == mode:
            self.areas.stop()
        else:
            self._set_roi_mode(mode)

    def _paint_area_tools(self):
        for mode, b in getattr(self, "_area_btns", {}).items():
            b.select(bool(self.areas) and self.areas.mode == mode)

    def _update_area_scope(self):
        """Indicador abajo a la izquierda, solo si esta foto tiene áreas: si son suyas o
        de todas las fotos, y "Aplicar a todas" cuando son suyas."""
        path = self.current_image_path
        shapes = self.project.roi_for(path) if path else []
        if not shapes:
            self._area_scope.place_forget()
            return
        own = self.project.has_own_roi(path)
        self._area_scope_var.set(t("roi.scope_image" if own else "roi.scope_all", n=len(shapes)))
        if own and len(self.project.images) > 1:
            self._apply_all_btn.pack(side=tk.LEFT, before=self._clear_area_btn)
        else:
            self._apply_all_btn.pack_forget()
        self._area_scope.place(x=6, rely=1.0, y=-6, anchor="sw")

    def _open_scale_dialog(self):
        ScaleDialog(self.root,
                    current_image=self.scaler_left.original if self.scaler_left.has_image else None,
                    current_path=self.current_image_path,
                    on_scale_set=self._on_scale_set,
                    loader=self._load_corrected_image,
                    paths=self.batch_paths, unit=self.project.unit)

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
                ("Ctrl+E", t("menu.export")), ("← →", t("help.key_images")), ("↑ ↓", t("help.key_layers")),
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

        def key_lists(handler):        # en listas y tablas ↑ ↓ siguen siendo de la lista
            return lambda e=None: None if typing() or isinstance(self.root.focus_get(), (tk.Listbox, ttk.Treeview)) \
                else (handler(), "break")[1]
        r.bind_all("<Up>", key_lists(lambda: self._step_view(-1)))
        r.bind_all("<Down>", key_lists(lambda: self._step_view(1)))
        # Zoom: Ctrl + / − / 0 / 1 (también sin Ctrl y con el teclado numérico)
        for seq, fn in (("plus", self.do_zoom_in), ("equal", self.do_zoom_in), ("KP_Add", self.do_zoom_in),
                        ("minus", self.do_zoom_out), ("KP_Subtract", self.do_zoom_out),
                        ("Key-0", self.do_zoom_fit), ("KP_0", self.do_zoom_fit),
                        ("Key-1", self.do_zoom_100), ("KP_1", self.do_zoom_100)):
            r.bind_all(f"<Control-{seq}>", lambda e=None, f=fn: (f(), "break")[1])
            r.bind_all(f"<{seq}>", key(fn))
        r.bind_all("<Key-z>", key(lambda: self._set_zoom_tool("zoom_area")))
        r.bind_all("<Key-h>", key(lambda: self._set_zoom_tool("pan")))
        r.bind_all("<Key-s>", key(lambda: self._toggle_area_tool("select")))
        r.bind_all("<Key-r>", key(lambda: self._toggle_area_tool("include_rect")))
        r.bind_all("<Key-p>", key(lambda: self._toggle_area_tool("include_poly")))
        r.bind_all("<Key-x>", key(lambda: self._toggle_area_tool("exclude_rect")))
        r.bind_all("<Key-X>", key(lambda: self._toggle_area_tool("exclude_poly")))
        r.bind_all("<Delete>", key(self._on_delete_key))
        r.bind_all("<Escape>", lambda e=None: self._on_escape())

    def _on_delete_key(self):
        """Supr: borra las áreas seleccionadas; si no hay, en la lista quita la foto."""
        if self.areas and self.areas.delete_selected():
            return "break"

    def _on_escape(self):
        if self.areas and self.areas.active:
            self.areas.stop()
        elif self.zoom_ctrl and self.zoom_ctrl.tool != "pan":
            self.zoom_ctrl.set_tool("pan")

    # ── Calibración / diálogos ────────────────────────────────────────────────

    def _on_scale_set(self, scale_result, scope: str = "all", path: str | None = None) -> bool:
        """Escala de la ventana de escala: solo para esa foto o para todas."""
        from fenotit.core.corrections.scale import UNIT_TO_MM
        sc = Scale(scale_result.unit_per_px * UNIT_TO_MM.get(scale_result.unit, 1.0),
                   "two_points", scale_result.format())
        if not self._set_scale(sc, scope, path or self.current_image_path):
            return False
        self.project.unit = scale_result.unit          # los resultados se reportan en esta unidad
        self._update_corr_indicator()
        self._sync_area_units()
        self.scale_result = scale_result
        self._set_status(t("status.scale", scale=scale_result.format()))
        return True

    def _set_scale(self, sc: Scale, scope: str, path: str | None) -> bool:
        if scope == "image" and path:
            self.project.set_scale(path, sc)
            return True
        n = self.project.others_with_own_scale(path)
        if n and not messagebox.askyesno(t("scale.apply_all"), t("scale.apply_all_warn", n=n), parent=self.root):
            return False
        odd = self._other_resolutions(path) if sc.mm_per_pixel else []
        if odd:
            names = "\n".join(f"  {name}  ({w}×{h})" for name, w, h in odd[:10])
            more = t("scale.res_more", n=len(odd) - 10) if len(odd) > 10 else ""
            if not messagebox.askyesno(t("scale.apply_all"), t("scale.res_warn", n=len(odd), names=names + more),
                                       parent=self.root):
                return False
        self.project.apply_scale_to_all(sc)
        return True

    def _other_resolutions(self, path: str | None) -> list[tuple[str, int, int]]:
        """Fotos con resolución distinta a la de `path` (solo se lee el encabezado)."""
        def size(p):
            try:
                with Image.open(p) as im:
                    return im.size
            except Exception:
                return None
        ref = size(path) if path else None
        if ref is None:
            return []
        out = []
        for p in self.project.images:
            sz = size(p)
            if sz and sz != ref and sz != ref[::-1]:
                out.append((Path(p).name, *sz))
        return out

    def _ask_scale_value(self) -> tuple[float, str] | None:
        """Escala escrita: valor y unidad por píxel (µm, mm, cm o m)."""
        win = tk.Toplevel(self.root)
        win.title(t("menu.scale_manual").rstrip("…"))
        win.configure(bg=COLORS["bg_card"])
        win.resizable(False, False)
        win.transient(self.root)
        body = tk.Frame(win, bg=COLORS["bg_card"])
        body.pack(padx=18, pady=14)
        tk.Label(body, text=t("scale.manual_prompt"), bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["small"], justify="left").pack(anchor="w", pady=(0, 8))
        row = tk.Frame(body, bg=COLORS["bg_card"])
        row.pack(anchor="w")
        value = tk.StringVar()
        entry = tk.Entry(row, textvariable=value, width=12, font=FONTS["body"])
        entry.pack(side=tk.LEFT)
        unit = tk.StringVar(value=self.project.unit)
        ttk.Combobox(row, textvariable=unit, values=units.UNITS, state="readonly", width=4,
                     font=FONTS["body"]).pack(side=tk.LEFT, padx=(6, 2))
        tk.Label(row, text="/ px", bg=COLORS["bg_card"], fg=COLORS["text"], font=FONTS["body"]).pack(side=tk.LEFT)
        out: list = []

        def ok(_e=None):
            try:
                v = float(value.get().replace(",", "."))
            except ValueError:
                entry.focus_set()
                return
            if v > 0:
                out.append((v, unit.get()))
                win.destroy()
        btns = tk.Frame(body, bg=COLORS["bg_card"])
        btns.pack(fill=tk.X, pady=(12, 0))
        tk.Button(btns, text=t("common.cancel"), command=win.destroy, relief="flat",
                  bg=COLORS["btn_bg"], fg=COLORS["accent"], font=FONTS["small"]).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(btns, text=t("common.ok"), command=ok, relief="flat", bg=COLORS["accent"], fg="#FFFFFF",
                  font=FONTS["small"]).pack(side=tk.RIGHT)
        win.bind("<Return>", ok)
        win.bind("<Escape>", lambda e: win.destroy())
        entry.focus_set()
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_reqwidth()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_reqheight()) // 3
        win.geometry(f"+{x}+{y}")
        win.grab_set()
        self.root.wait_window(win)
        return out[0] if out else None

    def _calibrate_scale(self):
        got = self._ask_scale_value()
        if got:
            val, unit = got
            scope = "all"
            if len(self.project.images) > 1:
                ans = messagebox.askyesnocancel(t("menu.scale_manual").rstrip("…"), t("scale.scope_question"),
                                                parent=self.root)
                if ans is None:
                    return
                scope = "all" if ans else "image"
            mm = val * units.TO_MM[unit]
            if not self._set_scale(Scale(mm, "manual", f"{val:g} {unit}/px"), scope, self.current_image_path):
                return
            self.project.unit = unit
            self._update_corr_indicator()
            self._sync_area_units()
            self._set_status(t("status.scale", scale=f"{val:g} {unit}/px"))

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
        from fenotit.gui import about_dialog
        about_dialog.show(self.root, _assets() / "logo.ico", self._system_info())

    # ── Exportación ───────────────────────────────────────────────────────────

    def _export_results(self):
        """Una sola exportación: tablas + metadatos por análisis (y, si se pide, Excel e
        imágenes de las vistas, que se recalculan en procesos aparte)."""
        done = [(n, _analysis_label(n), sum(1 for p in self.batch_paths if p in self._results.get(n, {})
                                            and self._results[n][p].status == "ok"))
                for n in ANALYSES if self._results.get(n)]
        done = [d for d in done if d[2]]
        if not done:
            messagebox.showwarning(t("msg.no_results_title"), t("msg.no_results"), parent=self.root)
            return
        folder = settings.get("export_dir") or (str(self.project.folder) if self.project.folder else "") \
            or (str(Path(self.batch_paths[0]).parent) if self.batch_paths else "")
        ExportDialog(self.root, done, folder, run=self._run_export, cancel=self._cancel_export)

    def _image_info(self, path: str) -> dict:
        sc = self.project.scale_for(path)
        shapes = self.project.roi_for(path)
        own = self.project.key(path) in self.project.image_scale
        return {"file": path, "mm_per_px": self._scale_for(path) or "",
                "scale_source": sc.source + (" (image)" if own else ""),
                "length_unit": units.ascii_name(self.project.unit),
                "n_analysis_areas": sum(1 for x in shapes if x["kind"] == "include"),
                "n_excluded_areas": sum(1 for x in shapes if x["kind"] == "exclude")}

    def _run_export(self, folder: str, names: list[str], excel: bool, views: bool, dlg):
        from functools import partial
        from fenotit.core.export import exporter
        root = Path(folder)
        settings.set("export_dir", folder)
        written = 0
        try:
            for name in names:
                store = self._results.get(name, {})
                key = _analysis_key(name)
                info = {p: self._image_info(p) for p in self.batch_paths}
                tables = exporter.collect(name, store, self.batch_paths, info, self._skipped)
                meta = exporter.metadata(key, store, self.project,
                                         {"view_images": "views/" if views else "no"})
                written += len(exporter.write_tables(root / key, tables, meta, excel, f"{key}.xlsx"))
        except Exception as e:
            _log.exception("Exportación")
            dlg.finished(t("export.error", error=e), ok=False)
            return
        if not views:
            dlg.finished(t("export.done", folder=folder))
            return
        queue = []
        for name in names:
            store = self._results.get(name, {})
            jobs = [batch.Job(p, store[p].extra.get("params") or self._build_params())
                    for p in self.batch_paths if p in store and store[p].status == "ok"]
            if jobs:
                queue.append((name, jobs))
        total = sum(len(j) for _, j in queue)
        state = {"i": 0, "errors": 0}

        def next_analysis(cancelled=False):
            if cancelled or not queue:
                self._export_runner = None
                text = t("export.cancelled") if cancelled else t("export.done", folder=folder)
                if state["errors"]:
                    text += "  ·  " + t("status.failed", n=state["errors"])
                dlg.finished(text, ok=not cancelled)
                return
            name, jobs = queue.pop(0)
            task = partial(batch.export_views, str(root / _analysis_key(name) / "views"), dict(self.project.display))

            def outcome(i, n, out):
                state["i"] += 1
                state["errors"] += out.status != "ok"
                dlg.progress(state["i"], total, t("export.views_progress", i=state["i"], n=total))
            after = self.root.after
            self._export_runner = batch.BatchRunner(
                name, jobs, self.project.corrections, task=task,
                on_outcome=lambda i, n, o: after(0, lambda: outcome(i, n, o)),
                on_done=lambda c: after(0, lambda: next_analysis(c))).start()
        dlg.progress(0, total, t("export.views_progress", i=0, n=total))
        next_analysis()

    def _cancel_export(self):
        runner = getattr(self, "_export_runner", None)
        if runner is not None:
            runner.cancel()

    def _export_batch(self):
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

    def _progress(self, i: int | None, n: int = 0):
        """Barra de avance del lote en la barra de estado (None = ocultarla)."""
        if i is None:
            self._progress_bar.pack_forget()
            self._cancel_link.pack_forget()
            return
        self._progress_bar.config(maximum=max(n, 1), value=i)
        if not self._progress_bar.winfo_ismapped():
            self._cancel_link.pack(side=tk.RIGHT, padx=(0, 6), before=self._status_label)
            self._progress_bar.pack(side=tk.RIGHT, padx=(8, 2), before=self._status_label)

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _update_step_active(self, active: int):
        # step_labels ya no se usan — no-op
        pass