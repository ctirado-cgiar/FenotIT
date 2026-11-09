"""Calibración: correcciones de imagen y escala."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

from PIL import Image

from fenotit.core import units
from fenotit.core.project import Scale
from fenotit.gui.dialogs.corrections import CorrectionsDialog
from fenotit.gui.dialogs.scale import ScaleDialog
from fenotit.gui.style.components import button
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.i18n import t


class CalibrationMixin:
    def _open_corrections(self, tab: str = "distortion"):
        CorrectionsDialog(self.root, self.project.corrections, self.batch_paths, self.current_image_path,
                          aruco_scale=self.project.scale.source == "aruco",
                          on_apply=self._on_corrections_applied, tab=tab)

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
        for n in self.project.corrections.active():
            name = t(f"corr.short.{n}")
            if info is None:
                parts.append(name)
            elif n in info.applied:
                parts.append(name + " ✓")
            else:
                parts.append("⚠ " + name)
                missing.append(n)
        mm = self.mm_per_pixel
        if self.project.scale_for(self.current_image_path).source != "none":
            u = self.project.unit
            parts.append(f"{units.per_px(mm, u):.6g} {u}/px" if mm else t("corr.short.scale_pending"))
        self.corr_var.set(", ".join(parts))
        self._corr_label.config(fg=COLORS["warning"] if missing else COLORS["accent"])
        why = [t(f"corr.not_applied.{n}") for n in missing]
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
        k = info.work_k if info else 1.0              # medida en la foto de trabajo -> foto completa
        sc = Scale(scale_result.unit_per_px * UNIT_TO_MM.get(scale_result.unit, 1.0) * k,
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
            self._remember("undo.scale")
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
        self._remember("undo.scale")
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
        body.pack(padx=SPACE["xl"], pady=SPACE["lg"])
        cur = self._scale_summary(self.current_image_path)
        if cur:
            tk.Label(body, text=cur, bg=COLORS["accent_light"], fg=COLORS["accent"], font=FONTS["body_bold"],
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
        btns.pack(fill=tk.X, pady=(SPACE["lg"], 0))
        button(btns, text=t("common.cancel"), command=win.destroy, kind="secondary", size="small").pack(side=tk.RIGHT, padx=(SPACE["md"], 0))
        button(btns, text=t("common.ok"), command=ok, kind="primary", size="small").pack(side=tk.RIGHT)
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
