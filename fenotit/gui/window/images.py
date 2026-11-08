"""Fotos del proyecto: agregar, quitar, cargar y cambiar de foto."""
from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog, messagebox

from fenotit import log, settings
from fenotit.core.image_io import ImageScaler, natural_key
from fenotit.core.project import IMAGE_EXTS
from fenotit.i18n import t


_log = log.get("gui.window.images")


class ImagesMixin:
    def _image_list_room(self) -> int:
        """Tope de la lista de imágenes: 45 % del espacio, y menos si CAPAS lo necesita."""
        lf = self.left_panel.content
        room = lf.winfo_height() - self.image_list.winfo_y()
        return min(int(room * 0.45), room - self._layers_need())

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
        paths = sorted((str(p) for p in Path(folder).iterdir()
                        if p.suffix.lower() in IMAGE_EXTS), key=natural_key)
        if not paths:
            messagebox.showwarning(t("msg.no_images_title"),
                                   t("msg.no_images"))
            return
        self._add_images(paths)
        self._set_status(t("status.folder_loaded", n=len(paths), folder=folder))

    def _add_images(self, paths: list[str]):
        """Agrega imágenes a la lista (sin repetir) y muestra la primera nueva."""
        current = [str(p) for p in self.project.images]
        new = sorted(dict.fromkeys(str(p) for p in paths if str(p) not in current), key=natural_key)
        if not new:
            if paths:
                self.batch_index = current.index(str(paths[0]))
                self._load_single(self.batch_paths[self.batch_index])
                self._sync_listbox()
            return
        self.project.images = [Path(p) for p in current + new]
        self.project.current_index = len(current)
        self._update_batch_list()
        first = not current
        self._show_panel(self.left_panel, True)
        if first:                                  # paso 1 a la vista: segmentar
            self._show_panel(self.right_panel, True)
            self._select_right_tab("seg" if not getattr(self, "_analysis_chosen", False) else self._right_tab)
        self._update_start()
        self._load_single(new[0])
        self._sync_listbox()
        if not settings.get("tour_work_seen", False):     # primera vez con fotos: recorrido guiado
            self.root.after(1500, self._start_tour)

    def _remove_current_image(self):
        path = self.current_image_path
        if not path:
            return
        self._remember("undo.remove_image")
        images = [p for p in self.project.images if str(p) != str(path)]
        self._clear_all_results(path)
        self.project.image_roi.pop(self.project.key(path), None)
        self.project.image_scale.pop(self.project.key(path), None)
        self.project.excluded.pop(self.project.key(path), None)
        self.project.highlighted.pop(self.project.key(path), None)
        self.project.strokes.pop(self.project.key(path), None)
        self.project.prompts.pop(self.project.key(path), None)
        self.project.regions.pop(self.project.key(path), None)
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
            self._remember("undo.clear_images")
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

    def _load_single(self, path: str):
        """Lee y corrige la foto en otro hilo (una foto grande o en OneDrive puede tardar
        varios segundos y la ventana dejaría de responder). Si se pide otra mientras tanto,
        se carga solo la última."""
        self.current_image_path = path
        self.image_info_var.set(Path(path).name)
        if getattr(self, "_loading", None):
            self._load_next = path
            return
        self._loading = path
        self._load_next = None
        self._set_status(t("status.loading_photo"))

        def work():
            try:
                img, info = self._load_corrected(path)
            except Exception:
                _log.exception("Cargando %s", path)
                img, info = None, None
            self.root.after(0, lambda: self._on_photo_loaded(path, img, info))
        threading.Thread(target=work, daemon=True).start()

    def _on_photo_loaded(self, path: str, img, info):
        self._loading = None
        nxt = self._load_next
        if nxt:                                    # se pidió otra (o la misma con otros ajustes)
            self._load_single(nxt)
            return
        if path != self.current_image_path:
            return
        if img is None:
            messagebox.showerror(t("common.error"), t("msg.image_unreadable", name=Path(path).name),
                                 parent=self.root)
            return
        if path != getattr(self, "_sel_path", None):          # otra foto: sin objeto elegido
            self._clear_selection()
        self._sel_path = path
        self.scaler_left.set_image(img)
        self._on_canvas_row_resize()               # otra proporción: quizá conviene arriba y abajo
        self._fit_docked_chart()
        self.image_info_var.set(f"{Path(path).name}  ·  {img.shape[1]} × {img.shape[0]} px")
        self._show_meta()
        self._update_corr_indicator()
        self.preview_var.set("")
        # Al cargar imagen nueva: fit-to-canvas (imagen completa visible)
        if self.zoom_ctrl:
            self.zoom_ctrl._img_right = None
            self.zoom_ctrl.set_left(self.scaler_left.original)
        if self.areas:
            self.areas.set_shapes(self.project.roi_for(path))
        self.strokes.set_strokes(self.project.strokes.get(self.project.key(path)))
        self.prompts.set_points(self.project.prompts.get(self.project.key(path)))
        self._update_stroke_tool()
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
            if self._chart_panel:
                self._chart_panel.load([])
            self.step_label_var.set("—")
            self._show_results_ui(False)
            # No limpiar canvas derecho — mantener último resultado visible
            # (solo se limpia si no hay nada cacheado en ninguna imagen)
            if not any(self.results_cache.values()):
                self.canvas_right.delete("all")

        self._refresh_history()
        self._sync_area_units()

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
        marks = {p: "board" for p in self._board_photos()}
        marks.update({p: "error" for p in self._skipped})
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
