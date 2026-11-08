"""Proyecto: estado, abrir, guardar, recientes, fotos faltantes, recalcular y copia de recuperación."""
from __future__ import annotations

import dataclasses
import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2

from fenotit import APP_NAME, log, settings
from fenotit.core import recovery
from fenotit.core import units
from fenotit.core.analysis.registry import ANALYSES, AnalysisResult
from fenotit.core.corrections import pipeline as corrections
from fenotit.core.image_io import load_image
from fenotit.core.project import PROJECT_EXT, Project, Segmentation
from fenotit.gui.style.theme import COLORS, FONTS
from fenotit.gui.window.common import _analysis_key, _on_base, _analysis_label, _analysis_name
from fenotit.i18n import t


_log = log.get("gui.window.project")


class ProjectMixin:
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
        info = self._corr_info.get(path) if path else None
        if sc.source == "aruco":
            return info.mm_per_px if info else None
        if sc.mm_per_pixel and info is not None:          # guardada para la foto completa
            return sc.mm_per_pixel / info.work_k
        return sc.mm_per_pixel

    def _scale_summary(self, path: str | None) -> str | None:
        """La escala que ya tiene la foto (para no cambiarla sin querer): px por unidad."""
        sc = self.project.scale_for(path)
        if sc.source == "aruco":
            mm = self._scale_for(path)
        else:
            mm = sc.mm_per_pixel
        if not mm:
            return None
        unit = self.project.unit
        per = mm / units.TO_MM[unit]
        own = bool(path) and self.project.key(path) in self.project.image_scale
        return t("scale.current", px=f"{1 / per:.2f}", unit=units.display(unit), per=f"{per:.5g}",
                 src=t(f"scale.src.{sc.source}", sc.source),
                 scope=t("scale.scope_image") if own else t("scale.scope_all"))

    def _load_corrected(self, path: str):
        img = load_image(path)
        if img is None:
            return None, None
        img, info = corrections.apply(img, self.project.corrections)
        img = corrections.to_working(img, info, self.project.max_mpx)
        self._corr_info[path] = info
        return img, info

    def _load_corrected_image(self, path: str):
        return self._load_corrected(path)[0]

    def _board_photos(self) -> set[str]:
        """Fotos del tablero de ajedrez que están entre las fotos cargadas: no se analizan."""
        return {str(p) for p in self.project.corrections.distortion.photos or []}

    def _skip_reason(self, info) -> str | None:
        """Motivo para no analizar una imagen (perspectiva activa pero sin los 4 ArUco)."""
        d = self.project.corrections.distortion
        if info and d.enabled and d.mtx and "distortion" not in info.applied:
            return t("corr.skip_distortion", size="×".join(map(str, d.size or [])) or "?")
        if info and self.project.corrections.perspective.enabled and "perspective" not in info.applied:
            return t("corr.skip_aruco", ids=", ".join(map(str, info.aruco_missing)) or "?")
        if info and self.project.corrections.color.enabled and "color" not in info.applied:
            return t("corr.skip_color")
        return None

    def _mark_skipped(self, path: str, reason: str):
        self._skipped.add(path)
        self._update_batch_list()
        _log.warning("%s omitida: %s", Path(path).name, reason)

    def _recent_items(self) -> list:
        cur = str(self.project.file) if self.project.file else None
        return [(Path(p).stem, lambda p=p: self._open_recent(p)) for p in settings.recent_projects()
                if p != cur]

    def _invalidate_results(self):
        self._corr_info.clear()
        self._clear_all_results()
        self.last_result = None
        self.canvas_right.delete("all")
        self._show_results_ui(False)

    def _blank_project(self) -> Project:
        p = Project(name=t("project.untitled"))
        p.max_mpx = self._default_max_mpx()
        p.segmentation.classes = self._default_classes()
        return p

    def _store_panel_params(self):
        panel = getattr(self, "config_panel", None)
        if panel is not None and self.active_analysis in ANALYSES:
            values = panel.get_values(warn=False)
            if self._has_area_units():
                values["_area_unit"] = self._panel_area_unit
            self.project.params[_analysis_key(self.active_analysis)] = values

    def _collect_state(self) -> dict:
        self._store_panel_params()
        self.project.segmentation = Segmentation(**self._seg_state())
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
        self._drop_recovery()
        self._write_preview()
        settings.add_recent_project(self.project.file)
        self._set_status(t("project.saved", path=self.project.file))
        return True

    def _save_project_as(self) -> bool:
        """Nombre y lugar como en cualquier programa; el proyecto es una carpeta con ese
        nombre (ahí van el .fenotit, calibration/ y results/)."""
        untitled = self.project.name in ("", t("project.untitled"), "Sin título")
        start = self.project.folder.parent if self.project.folder else (
            Path(self.batch_paths[0]).parent if self.batch_paths else None)
        path = filedialog.asksaveasfilename(
            title=t("project.save_as_title"), parent=self.root, defaultextension=PROJECT_EXT,
            initialfile=("" if untitled else self.project.name) + PROJECT_EXT,
            initialdir=str(start) if start else None,
            filetypes=[(t("project.filetype"), f"*{PROJECT_EXT}")])
        if not path:
            return False
        path = Path(path)
        name = path.name[:-len(PROJECT_EXT)] if path.name.endswith(PROJECT_EXT) else path.stem
        name = name.strip() or t("project.untitled")
        # si el usuario ya entró a la carpeta del proyecto, no crear otra adentro
        folder = path.parent if path.parent.name == name else path.parent / name
        if (folder / f"{name}{PROJECT_EXT}").exists() and folder != self.project.folder \
                and not path.exists():                      # si existe, el diálogo del sistema ya preguntó
            if not messagebox.askyesno(t("project.save_as_title"), t("project.exists", name=name), parent=self.root):
                return False
        self.project.folder = folder
        self.project.name = name
        return self._save_project()

    def _new_project(self):
        if not self._confirm_discard():
            return
        self._apply_project(self._blank_project())

    def _go_home(self):
        """Volver a la pantalla de inicio (proyectos recientes); pregunta si guardar."""
        if not self.project.images:
            return
        if not self._confirm_discard():
            return
        self._apply_project(self._blank_project())

    def _open_project(self):
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(
            title=t("project.open_title"),
            filetypes=[(t("project.filetype"), f"*{PROJECT_EXT}"), (t("common.all_files"), "*.*")])
        if path:
            self.open_project_path(path)

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
        self._relinked = False
        if project.missing_images() and not self._resolve_missing(project):
            return
        self._apply_project(project)
        if self._relinked:                                # rutas nuevas: el proyecto queda por guardar
            self._saved_state = {}
            self._update_title()
        if project.file:
            settings.add_recent_project(project.file)
        self._set_status(t("project.opened", path=project.file))

    def _find_missing(self):
        """Imágenes → Buscar faltantes: lo mismo que al abrir, con el proyecto abierto."""
        if not self.project.missing_images():
            messagebox.showinfo(t("missing.title"), t("missing.none"), parent=self.root)
            return
        self._relinked = False
        if not self._resolve_missing(self.project):
            return
        self._invalidate_results()
        self._update_batch_list()
        paths = self.batch_paths
        if paths:
            self._load_single(paths[min(self.project.current_index, len(paths) - 1)])
        else:
            self._show_no_image()

    def _resolve_missing(self, project: Project) -> bool:
        """Fotos que no están donde las dejó quien guardó el proyecto (otro computador,
        carpeta movida o compartida). Se buscan por nombre en la carpeta que indique el
        usuario (y sus subcarpetas); las que no aparezcan se pueden quitar. False = no abrir."""
        from fenotit.core.project import find_by_name
        relinked = 0
        while True:
            missing = project.missing_images()
            if not missing:
                break
            ans = self._missing_dialog(len(missing), len(project.images), missing[:4], relinked)
            if ans == "search":
                folder = filedialog.askdirectory(title=t("missing.pick_folder"), parent=self.root)
                if folder:
                    relinked += project.relink(find_by_name(missing, folder))
                continue
            if ans == "remove":
                project.images = [p for p in project.images if p.exists()]
                project.current_index = min(project.current_index, max(len(project.images) - 1, 0))
                break
            return False
        if relinked:
            self._relinked = True                       # al abrir queda "con cambios": conviene guardar
        return True

    def _missing_dialog(self, n: int, total: int, examples: list, relinked: int) -> str | None:
        """search | remove | None (cancelar)."""
        win = tk.Toplevel(self.root)
        win.title(t("missing.title"))
        win.transient(self.root)
        win.resizable(False, False)
        win.configure(bg=COLORS["bg_panel"])
        box = tk.Frame(win, bg=COLORS["bg_panel"], padx=18, pady=14)
        box.pack(fill=tk.BOTH)
        head = t("missing.found", k=relinked) + "  " if relinked else ""
        tk.Label(box, text=head + t("missing.msg", n=n, total=total), bg=COLORS["bg_panel"], fg=COLORS["text"],
                 font=FONTS["body"], justify="left", wraplength=420).pack(anchor="w")
        tk.Label(box, text="\n".join(Path(p).name for p in examples) + ("\n…" if n > len(examples) else ""),
                 bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONTS["small"], justify="left").pack(
                     anchor="w", pady=(6, 10))
        tk.Label(box, text=t("missing.hint"), bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONTS["small"],
                 justify="left", wraplength=420).pack(anchor="w", pady=(0, 12))
        res = {"v": None}

        def pick(v):
            res["v"] = v
            win.destroy()
        bar = tk.Frame(box, bg=COLORS["bg_panel"])
        bar.pack(fill=tk.X)
        btn = dict(relief="flat", bd=0, cursor="hand2", font=FONTS["body"], padx=12, pady=4)
        tk.Button(bar, text=t("missing.search"), command=lambda: pick("search"), bg=COLORS["accent"], fg=COLORS["on_accent"],
                  activebackground=COLORS["accent"], activeforeground=COLORS["on_accent"], **btn).pack(side=tk.LEFT)
        tk.Button(bar, text=t("missing.remove"), command=lambda: pick("remove"), bg=COLORS["btn_bg"],
                  fg=COLORS["accent"], **btn).pack(side=tk.LEFT, padx=6)
        tk.Button(bar, text=t("common.cancel"), command=lambda: pick(None), bg=COLORS["btn_bg"],
                  fg=COLORS["text"], **btn).pack(side=tk.RIGHT)
        win.bind("<Escape>", lambda e: pick(None))
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        win.grab_set()
        self.root.wait_window(win)
        return res["v"]

    def _apply_project(self, project: Project):
        self._drop_recovery()                    # el usuario ya guardó o descartó el anterior
        self._undo_stack, self._redo_stack, self._settled = [], [], None
        self.config_panel = None
        self.project = project
        self._set_seg_state(dataclasses.asdict(project.segmentation))
        self._examples = {}
        self._on_auto_change()
        self.legend_var.set(project.display.get("legend", True))
        self._paint_pos_grid()
        self.color_fmt_var.set(project.display.get("color_format", "RGB"))

        name = _analysis_name(project.analysis)
        self._analysis_chosen = False
        if name:
            self.analysis_var.set(_analysis_label(name))
        self._on_analysis_selected(None, choose=bool(name))

        self._clear_all_results()
        self._corr_info.clear()
        self.canvas_right.delete("all")
        self._update_batch_list()
        self._show_panel(self._panel_of("images"), bool(project.images))
        self._show_panel(self._panel_of("steps"), bool(project.images) and self._analysis_chosen)
        if project.current_image:
            self._load_single(str(project.current_image))
            self._sync_listbox()
        else:
            self.current_image_path = None
            self.canvas_left.delete("all")
        self._update_corr_indicator()
        self._update_start()
        self._saved_state = self._collect_state()
        if project.images and project.ran:
            self.root.after(500, self._offer_rerun)

    def _offer_rerun(self):
        """Los resultados no se guardan en el proyecto: al abrirlo se ofrece recalcularlos."""
        names = [n for n in (_analysis_name(k) for k in self.project.ran) if n]
        if not names or not messagebox.askyesno(
                t("rerun.title"), t("rerun.msg", analyses=", ".join(_analysis_label(n) for n in names),
                                    n=len(self.batch_paths)), parent=self.root):
            return
        self._rerun_queue = names
        self._rerun_next()

    def _rerun_next(self):
        queue = getattr(self, "_rerun_queue", [])
        if not queue:
            return
        name = queue.pop(0)
        self._open_analysis(name)
        self._run_batch(name)

    def _mark_ran(self, name: str):
        key = _analysis_key(name)
        if key not in self.project.ran:
            self.project.ran.append(key)

    def _restart(self):
        """Reinicia la aplicación (p. ej. al cambiar el idioma): pregunta si guardar, abre otro
        proceso con el mismo proyecto y cierra este. Si el usuario cancela, el cambio queda
        para la próxima vez."""
        import subprocess
        if self._is_dirty() and not self._confirm_discard():
            messagebox.showinfo(t("lang.title"), t("lang.restart"), parent=self.root)
            return
        project = str(self.project.file) if self.project.file and Path(self.project.file).exists() else None
        if getattr(sys, "frozen", False):                  # instalado (.exe)
            cmd = [sys.executable]
        elif Path(sys.argv[0]).name == "__main__.py":       # python -m fenotit
            cmd = [sys.executable, "-m", "fenotit"]
        else:                                               # python main.py
            cmd = [sys.executable, str(Path(sys.argv[0]).resolve())]
        try:
            subprocess.Popen(cmd + ([project] if project else []), cwd=os.getcwd(), close_fds=True)
        except Exception:
            _log.exception("No se pudo reiniciar")
            messagebox.showinfo(t("lang.title"), t("lang.restart"), parent=self.root)
            return
        self._drop_recovery()
        self.root.destroy()

    def _on_close(self):
        if self._is_dirty():
            if not self._confirm_discard():
                return
        elif not messagebox.askyesno(t("close.title"), t("close.msg"), parent=self.root):
            return
        self._drop_recovery()
        self.root.destroy()

    def _autosave_tick(self):
        """Cada 20 s mira si ya toca: guarda una copia aparte solo si hay cambios sin guardar.
        Nunca escribe sobre el proyecto del usuario."""
        try:
            every = int(settings.get("autosave_min", 3) or 0)
            if every and time.monotonic() - self._last_autosave >= every * 60:
                self._last_autosave = time.monotonic()
                if self.project.images and self._is_dirty():
                    self._recovery_file = recovery.write(self.project, self._session)
        except Exception:
            _log.exception("Autoguardado")
        self.root.after(20000, self._autosave_tick)

    def _drop_recovery(self):
        recovery.discard(getattr(self, "_recovery_file", None))
        self._recovery_file = None

    def _recovery_dialog(self, name: str, when: str, n: int):
        """True = recuperar, False = descartar la copia, None = decidir después."""
        bg = COLORS["bg_card"]
        win = tk.Toplevel(self.root)
        win.title(t("recovery.title"))
        win.transient(self.root)
        win.resizable(False, False)
        win.configure(bg=bg)
        tk.Frame(win, bg=COLORS["accent"], height=4).pack(fill=tk.X)
        box = tk.Frame(win, bg=bg, padx=20, pady=14)
        box.pack(fill=tk.BOTH)
        tk.Label(box, text=t("recovery.title"), bg=bg, fg=COLORS["accent"],
                 font=FONTS["heading"]).pack(anchor="w")
        tk.Label(box, text=t("recovery.msg", name=name, when=when, n=n), bg=bg, fg=COLORS["text"],
                 font=FONTS["body"], justify="left", wraplength=380).pack(anchor="w", pady=(6, 14))
        res = {"v": None}

        def pick(v):
            res["v"] = v
            win.destroy()
        bar = tk.Frame(box, bg=bg)
        bar.pack(fill=tk.X)
        for text, v, accent in ((t("recovery.later"), None, False), (t("recovery.discard"), False, False),
                                (t("recovery.recover"), True, True)):
            tk.Button(bar, text=text, command=lambda v=v: pick(v), relief="flat", cursor="hand2", padx=12,
                      bg=COLORS["accent"] if accent else COLORS["btn_bg"], fg=COLORS["on_accent"] if accent else COLORS["accent"],
                      font=FONTS["body_bold"] if accent else FONTS["body"]).pack(side=tk.RIGHT, padx=(6, 0))
        win.protocol("WM_DELETE_WINDOW", lambda: pick(None))
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_reqwidth()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_reqheight()) // 3
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        win.grab_set()
        self.root.wait_window(win)
        return res["v"]

    def _offer_recovery(self):
        """Al abrir: si quedó trabajo sin guardar de una sesión que se cerró mal, se ofrece."""
        for item in recovery.pending():
            when = item["saved"].replace("T", " ")[:16]
            ans = self._recovery_dialog(item["name"], when, item["n_images"])
            if ans is None:                      # Más tarde: se vuelve a ofrecer la próxima vez
                continue
            if ans:
                try:
                    project = recovery.load(item)
                except Exception:
                    _log.exception("No se pudo recuperar")
                    continue
                self._apply_project(project)
                self._saved_state = {}           # recuperado = sin guardar
                self._recovery_file = item["file"]
                return
            recovery.discard(item["file"])

    PREVIEW_SIDE = 360

    def _write_preview(self):
        """Miniatura de la vista que se está viendo (con sus marcas, sin leyenda) en la
        carpeta del proyecto: los recientes de la pantalla de inicio la muestran."""
        r, target = self.last_result, self.project.preview
        if target is None or not r or not self.step_names or not 0 <= self.step_idx < len(self.step_names):
            return
        name = self.step_names[self.step_idx]
        ovs = r.extra.get("overlays") or {}
        base = r.extra.get("base_image") if _on_base(r, name) else r.step_images.get(name)
        if base is None:
            return
        try:
            from fenotit.core.pipeline import overlay
            k = self.PREVIEW_SIDE / max(base.shape[:2])
            small = cv2.resize(base, (max(1, round(base.shape[1] * k)), max(1, round(base.shape[0] * k))),
                               interpolation=cv2.INTER_AREA)
            if name in ovs:
                overlay.draw(small, ovs[name], self._mark_colors(r), (0, 0), k, screen=True)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                target.write_bytes(buf.tobytes())
        except Exception:
            _log.debug("Miniatura del proyecto", exc_info=True)
