"""Calibración: correcciones de imagen y escala."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

from PIL import Image

# units convierte entre unidades de longitud y arma los textos con unidad.
from fenotit.core import units
# Scale es la escala guardada en el proyecto; aquí se cambia al elegir ArUco o quitar la escala.
from fenotit.core.project import Scale
# La ventana de cada corrección (lente, perspectiva, tarjeta de color) que se abre desde el menú
# Calibración.
from fenotit.gui.dialogs.corrections import CorrectionsDialog
# La ventana para fijar la escala con dos puntos o con un valor escrito.
from fenotit.gui.dialogs.scale import ScaleDialog
# button crea un botón con el estilo del tema.
from fenotit.gui.style.components import button
# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras); SPACE (márgenes).
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
# t devuelve cada texto de la ventana en el idioma elegido (fenotit/lang).
from fenotit.i18n import t


class CalibrationMixin:
    def _open_corrections(self, kind: str = "distortion"):
        """Abre la ventana de una corrección (distortion, perspective o color)."""
        self._single(f"corr_{kind}", lambda: CorrectionsDialog(
            self.root, self.project.corrections, self.batch_paths, self.current_image_path,
            aruco_scale=self.project.scale.source == "aruco", on_apply=self._on_corrections_applied, kind=kind))

    def _on_corrections_applied(self, corr, use_aruco_scale: bool):
        self._remember("undo.corrections")
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
        self._update_batch_list()

    def _clear_scale(self):
        self._remember("undo.scale")
        self.project.apply_scale_to_all(Scale())
        self._set_status(t("status.scale_cleared"))
        self._update_corr_indicator()
        self._sync_area_units()

    def _update_corr_indicator(self):
        """Correcciones activas y si se aplicaron a esta foto: «Color ✓» o «⚠ Color» (con
        el motivo al pasar el mouse)."""
        info = self._corr_info.get(self.current_image_path) if self.current_image_path else None
        parts, missing = [], []
        for correction in self.project.corrections.active():
            name = t(f"corr.short.{correction}")
            if info is None:
                parts.append(name)
            elif correction in info.applied:
                parts.append(name + " ✓")
            else:
                parts.append("⚠ " + name)
                missing.append(correction)
        millimeters = self.mm_per_pixel
        if self.project.scale_for(self.current_image_path).source != "none":
            unit = self.project.unit
            parts.append(f"{units.per_px(millimeters, unit):.6g} {unit}/px" if millimeters else t("corr.short.scale_pending"))
        self.corr_var.set(", ".join(parts))
        self._corr_label.config(fg=COLORS["warning"] if missing else COLORS["accent"])
        why = [t(f"corr.not_applied.{correction}") for correction in missing]
        self._corr_tip.text = "\n".join(why) if why else t("corr.applied_tip")

    def _open_scale_dialog(self):
        ScaleDialog(self.root,
                    current_image=self.scaler_left.original if self.scaler_left.has_image else None,
                    current_path=self.current_image_path,
                    on_scale_set=self._on_scale_set,
                    loader=self._load_corrected_image,
                    paths=self.batch_paths, unit=self.project.unit, current=self._scale_summary)

    def _on_scale_set(self, scale_result, scope: str = "all", path: str | None = None) -> bool:
        """Escala de la ventana de escala: solo para esa foto o para todas."""
        from fenotit.core.corrections.scale import UNIT_TO_MM
        info = self._corr_info.get(path or self.current_image_path)
        work_factor = info.work_k if info else 1.0    # medida en la foto de trabajo -> foto completa
        scale = Scale(scale_result.unit_per_px * UNIT_TO_MM.get(scale_result.unit, 1.0) * work_factor,
                   "two_points", scale_result.format())
        if not self._set_scale(scale, scope, path or self.current_image_path):
            return False
        self.project.unit = scale_result.unit          # los resultados se reportan en esta unidad
        self._update_corr_indicator()
        self._sync_area_units()
        self.scale_result = scale_result
        self._set_status(t("status.scale", scale=scale_result.format()))
        return True

    def _set_scale(self, scale: Scale, scope: str, path: str | None) -> bool:
        if scope == "image" and path:
            self._remember("undo.scale")
            self.project.set_scale(path, scale)
            return True
        n_own = self.project.others_with_own_scale(path)
        if n_own and not messagebox.askyesno(t("scale.apply_all"), t("scale.apply_all_warn", n=n_own), parent=self.root):
            return False
        odd = self._other_resolutions(path) if scale.mm_per_pixel else []
        if odd:
            names = "\n".join(f"  {name}  ({width}×{height})" for name, width, height in odd[:10])
            more = t("scale.res_more", n=len(odd) - 10) if len(odd) > 10 else ""
            if not messagebox.askyesno(t("scale.apply_all"), t("scale.res_warn", n=len(odd), names=names + more),
                                       parent=self.root):
                return False
        self._remember("undo.scale")
        self.project.apply_scale_to_all(scale)
        return True

    def _other_resolutions(self, path: str | None) -> list[tuple[str, int, int]]:
        """Fotos con resolución distinta a la de `path` (solo se lee el encabezado)."""
        def size(image):
            try:
                with Image.open(image) as picture:
                    return picture.size
            except Exception:
                return None
        ref = size(path) if path else None
        if ref is None:
            return []
        out = []
        for image in self.project.images:
            image_size = size(image)
            if image_size and image_size != ref and image_size != ref[::-1]:
                out.append((Path(image).name, *image_size))
        return out

    def _ask_scale_value(self) -> tuple[float, str] | None:
        """Escala escrita: valor y unidad por píxel (µm, mm, cm o m)."""
        win = tk.Toplevel(self.root)
        win.title(t("menu.scale_manual").rstrip("…"))
        win.configure(bg=COLORS["bg_card"])
        win.resizable(False, False)
        win.transient(self.root)
        body = tk.Frame(win, bg=COLORS["bg_card"])
        body.pack(padx=SPACE["xl"], pady=SPACE["lg"])
        current = self._scale_summary(self.current_image_path)
        if current:
            tk.Label(body, text=current, bg=COLORS["accent_light"], fg=COLORS["accent"], font=FONTS["body_bold"],
                     justify="left", anchor="w", padx=SPACE["md"], pady=SPACE["sm"]).pack(fill=tk.X, pady=(0, SPACE["lg"]))
        tk.Label(body, text=t("scale.manual_prompt"), bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["small"], justify="left").pack(anchor="w", pady=(0, SPACE["md"]))
        row = tk.Frame(body, bg=COLORS["bg_card"])
        row.pack(anchor="w")
        value = tk.StringVar()
        entry = tk.Entry(row, textvariable=value, width=12, font=FONTS["body"])
        entry.pack(side=tk.LEFT)
        unit = tk.StringVar(value=self.project.unit)
        ttk.Combobox(row, textvariable=unit, values=units.UNITS, state="readonly", width=4,
                     font=FONTS["body"]).pack(side=tk.LEFT, padx=(SPACE["md"], SPACE["xs"]))
        tk.Label(row, text="/ px", bg=COLORS["bg_card"], fg=COLORS["text"], font=FONTS["body"]).pack(side=tk.LEFT)
        out: list = []

        def ok(_event=None):
            try:
                number = float(value.get().replace(",", "."))
            except ValueError:
                entry.focus_set()
                return
            if number > 0:
                out.append((number, unit.get()))
                win.destroy()
        btns = tk.Frame(body, bg=COLORS["bg_card"])
        btns.pack(fill=tk.X, pady=(SPACE["lg"], 0))
        button(btns, text=t("common.cancel"), command=win.destroy, kind="secondary", size="small").pack(side=tk.RIGHT, padx=(SPACE["md"], 0))
        button(btns, text=t("common.ok"), command=ok, kind="primary", size="small").pack(side=tk.RIGHT)
        win.bind("<Return>", ok)
        win.bind("<Escape>", lambda event: win.destroy())
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
            number, unit = got
            scope = "all"
            if len(self.project.images) > 1:
                answer = messagebox.askyesnocancel(t("menu.scale_manual").rstrip("…"), t("scale.scope_question"),
                                                parent=self.root)
                if answer is None:
                    return
                scope = "all" if answer else "image"
            millimeters = number * units.TO_MM[unit]
            if not self._set_scale(Scale(millimeters, "manual", f"{number:g} {unit}/px"), scope, self.current_image_path):
                return
            self.project.unit = unit
            self._update_corr_indicator()
            self._sync_area_units()
            self._set_status(t("status.scale", scale=f"{number:g} {unit}/px"))
