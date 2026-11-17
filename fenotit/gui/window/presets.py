"""Presets: guardar la configuración del proyecto en un archivo, cargarla y restaurarla."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox

from fenotit import log, settings
from fenotit.core import preset
from fenotit.core.analysis.registry import ANALYSES
from fenotit.i18n import t

_log = log.get("gui.window.presets")


class PresetsMixin:
    PRESET_LABEL_MS = 1000       # cada cuánto se revisa si el preset aplicado se modificó

    def _preset_items(self) -> list:
        """Entradas del submenú Preset."""
        items = [(t("preset.load"), self._load_preset), (t("preset.save"), self._save_preset)]
        recent = [(Path(path).stem, lambda path=path: self._load_preset(path)) for path in settings.recent_presets()]
        if recent:
            items.insert(1, (t("menu.recent"), recent))
        if self.project.preset.get("data"):
            items += [None, (t("preset.restore"), self._restore_preset), (t("preset.detach"), self._detach_preset)]
        return items

    def _preset_folder(self) -> str | None:
        folder = settings.get("preset_dir")
        return folder if folder and Path(folder).is_dir() else None

    def _save_preset(self):
        from fenotit.gui.dialogs.preset import PresetDialog
        self._collect_state()
        photo = self.current_image_path
        chosen = self._analysis_chosen and self.project.analysis
        available = tuple(part for part in preset.applicable(self.project, photo) if part != "analysis" or chosen)
        if not available:
            messagebox.showinfo(t("preset.save_title"), t("preset.nothing_to_save"), parent=self.root)
            return
        checked = tuple(part for part in preset.suggested(self.project, photo) if part in available)
        PresetDialog(self.root, self.project.preset.get("name", ""), available, checked, self._write_preset)

    def _write_preset(self, name: str, description: str, parts: tuple):
        path = filedialog.asksaveasfilename(
            title=t("preset.save_title"), parent=self.root, defaultextension=preset.PRESET_EXT,
            initialfile=name + preset.PRESET_EXT, initialdir=self._preset_folder(),
            filetypes=[(t("preset.filetype"), f"*{preset.PRESET_EXT}")])
        if not path:
            return
        self._collect_state()
        data = preset.from_project(self.project, name, parts, description,
                                   areas=self.project.roi_for(self.current_image_path),
                                   scale=self.project.scale_for(self.current_image_path))
        try:
            saved = preset.save(data, path)
        except OSError as error:
            _log.exception("Guardar preset")
            messagebox.showerror(t("common.error"), str(error), parent=self.root)
            return
        settings.set("preset_dir", str(saved.parent))
        settings.add_recent_preset(saved)
        self.project.preset = {"name": name, "data": data}
        self._show_preset_name()
        self._set_status(t("preset.saved", name=name))

    def _load_preset(self, path: str | None = None):
        path = path or filedialog.askopenfilename(
            title=t("preset.load_title"), parent=self.root, initialdir=self._preset_folder(),
            filetypes=[(t("preset.filetype"), f"*{preset.PRESET_EXT}"), (t("common.all_files"), "*.*")])
        if not path:
            return
        try:
            data = preset.load(path)
        except (OSError, ValueError):
            messagebox.showerror(t("common.error"), t("preset.not_a_preset", file=Path(path).name), parent=self.root)
            return
        settings.set("preset_dir", str(Path(path).parent))
        settings.add_recent_preset(path)
        self._apply_preset(data)

    def _restore_preset(self):
        self._apply_preset(self.project.preset["data"])

    def _detach_preset(self):
        self.project.preset = {}
        self._show_preset_name()

    def _apply_preset(self, data: dict):
        """Aplica las partes del preset como un solo cambio (Ctrl+Z lo devuelve)."""
        self._collect_state()
        self._remember("undo.preset")
        target = copy.deepcopy(self.project)
        parts = preset.apply(data, target)
        state = self._snapshot()
        for field in ("roi", "image_roi", "image_scale", "scale", "unit", "max_mpx", "corrections"):
            state[field] = getattr(target, field)
        analysis = data.get("analysis")
        if "analysis" in parts and analysis in ANALYSES:
            self._open_analysis(analysis)
            values = {key: value for key, value in target.params[analysis].items() if not key.startswith("_")}
            state["_params"] = (analysis, {**state["_params"][1], **values} if state["_params"][0] == analysis
                                else values)
        if "segmentation" in parts:
            wanted = asdict(target.segmentation)
            state["_seg"] = {**self._seg_state(), **wanted,
                             "model": {**self._seg_state()["model"], **wanted["model"]}}
        self._apply_state(state)
        self.project.preset = target.preset
        self._collect_state()
        preset.rebase(self.project)
        self._mark_settled()
        self._show_preset_name()
        self._set_status(t("preset.applied", name=data.get("name", "")))
        wanted_model = (data.get("segmentation") or {}).get("model", {}).get("name")
        if "segmentation" in parts and data["segmentation"].get("method") == "model" and self._seg_method != "model":
            messagebox.showwarning(t("preset.load_title"), t("preset.model_missing", model=wanted_model),
                                   parent=self.root)
        if "analysis" in parts and analysis not in ANALYSES:
            messagebox.showwarning(t("preset.load_title"), t("preset.analysis_missing", analysis=analysis),
                                   parent=self.root)

    def _show_preset_name(self):
        """Nombre del preset aplicado bajo el del análisis, con «(modificado)» si algo cambió."""
        name = self.project.preset.get("name")
        text = ""
        if name:
            self._collect_state()
            changed = preset.modified(self.project)
            text = t("preset.label_modified" if changed else "preset.label", name=name)
        if text != self._preset_label.cget("text"):
            self._preset_label.config(text=text)
            self._select_right_tab(self._right_tab)

    def _watch_preset(self):
        self._show_preset_name()
        self.root.after(self.PRESET_LABEL_MS, self._watch_preset)
