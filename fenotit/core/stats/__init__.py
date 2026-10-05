"""Análisis posteriores sobre las tablas extraídas (una imagen, un lote o grupos).

La extracción (fenotit.core.pipeline) da filas por objeto y por imagen. Aquí se
combinan y se comparan con cualquier agrupación: por imagen, por lote o por una
columna del usuario (genotipo, tratamiento...). Nada de esto toca imágenes.
"""
from fenotit.core.stats.dataset import combine, groups, summarize

__all__ = ["combine", "groups", "summarize"]
