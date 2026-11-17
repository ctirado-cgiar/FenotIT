"""Tabla de datos del usuario (genotipo, repetición…)."""
from __future__ import annotations

from pathlib import Path

from fenotit import log
from fenotit.i18n import t


_log = log.get("gui.window.metadata")


class MetadataMixin:
    def _open_metadata(self):
        from fenotit.gui.dialogs.metadata import MetadataDialog
        self._single("metadata", lambda: MetadataDialog(
            self.root, [Path(path).name for path in self.batch_paths], dict(self.project.metadata), self._set_metadata))

    def _forget_meta(self):
        """La tabla del usuario se vuelve a leer la próxima vez que se pida."""
        self._meta_cache = None

    def _set_metadata(self, file: str | None, key_column: str | None):
        self._remember("undo.metadata")
        self.project.metadata = {"file": file, "key_column": key_column} if file else {}
        self._forget_meta()
        self.image_list._filter()
        self._show_meta()
        match = self._meta_match()
        self._set_status(t("meta.applied", n=len(match.by_image), total=len(self.batch_paths)) if match
                         else t("meta.removed"))

    def _meta_table(self):
        """(columnas, filas) del archivo; se vuelve a leer si cambió en el disco."""
        from fenotit.core import metadata
        file = self.project.metadata.get("file")
        if not file:
            return None
        try:
            stamp = (file, Path(file).stat().st_mtime)
        except OSError:
            return None
        cache = getattr(self, "_meta_file_cache", None)
        if cache and cache[0] == stamp:
            return cache[1]
        try:
            table = metadata.read_table(file)
        except Exception:
            _log.warning("Tabla de datos %s", file, exc_info=True)
            table = None
        self._meta_file_cache = (stamp, table)
        return table

    def _meta_match(self):
        from fenotit.core import metadata
        table = self._meta_table()
        col = self.project.metadata.get("key_column")
        if not table or col not in table[0]:
            return None
        names = tuple(Path(path).name for path in self.batch_paths)
        key = (id(table), names, col)
        cache = getattr(self, "_meta_cache", None)
        if cache and cache[0] == key:
            return cache[1]
        match = metadata.match(list(names), table[1], col)
        self._meta_cache = (key, match)
        return match

    def _meta_columns(self) -> list[str]:
        table = self._meta_table()
        col = self.project.metadata.get("key_column")
        return [column for column in table[0] if column != col] if table and self._meta_match() else []

    def _meta_row(self, path) -> dict:
        """Valores de la tabla del usuario para una foto (sin la columna que la identifica)."""
        match = self._meta_match() if path else None
        if not match:
            return {}
        row = match.by_image.get(Path(path).name) or {}
        return {column: ("" if row.get(column) is None else row.get(column)) for column in self._meta_columns()}

    def _show_meta(self):
        if not hasattr(self, "_meta_var"):
            return
        row = self._meta_row(getattr(self, "current_image_path", None))
        self._meta_var.set(", ".join(f"{key}: {value}" for key, value in row.items() if value != ""))
