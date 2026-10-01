"""
analysis/registry.py
Registro central de módulos de análisis.

Para agregar un análisis nuevo:
  1. Crear analysis/mi_analisis.py con una función run(image, params) -> AnalysisResult
  2. Decorar con @register(...) o llamar register() al final del módulo
  3. Importar el módulo aquí abajo — eso es todo, la UI lo detecta sola.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np

from fenotit import log

_log = log.get("analysis")


# ── Estructura de resultado estándar ─────────────────────────────────────────

@dataclass
class AnalysisResult:
    """
    Lo que devuelve cualquier módulo de análisis.
    Todos los campos son opcionales — cada módulo llena lo que aplica.
    """
    # Imágenes intermedias con nombre descriptivo del paso
    # Ej: {"máscara binaria": arr, "contornos": arr, "segmentado": arr}
    step_images: dict[str, np.ndarray] = field(default_factory=dict)

    # Mediciones por objeto: lista de dicts con métricas
    # Ej: [{"index": 1, "area_mm": 12.3, "largo_mm": 5.1, ...}, ...]
    measurements: list[dict[str, Any]] = field(default_factory=list)

    # Estadísticas globales del resultado
    # Ej: {"total": 47, "area_promedio": 10.2, "std_area": 1.4}
    stats: dict[str, Any] = field(default_factory=dict)

    # Columnas para mostrar en la tabla de resultados de la UI
    # Si está vacío, se usan todas las claves del primer measurement
    table_columns: list[str] = field(default_factory=list)

    # Datos extra para exportación (p.ej. color_data, efa_coefficients)
    extra: dict[str, Any] = field(default_factory=dict)

    # Mensaje de estado / error
    status: str = "ok"
    error: str = ""


# ── Descriptor de análisis ────────────────────────────────────────────────────

@dataclass
class AnalysisDescriptor:
    name: str
    func: Callable                    # run(image: np.ndarray, params: dict) -> AnalysisResult
    description: str = ""
    supports_batch: bool = True
    # Parámetros que este análisis expone en el panel de configuración
    # Cada entrada: {"key": str, "label": str, "type": "int"|"float"|"bool",
    #                "default": Any, "min": Any, "max": Any, "tooltip": str}
    params_schema: list[dict] = field(default_factory=list)
    # Íconos Tabler para el botón del menú (nombre sin 'ti-')
    icon: str = "chart-bar"
    # Bloque de segmentación que usa ("threshold" = umbral/Otsu compartido; None = no segmenta,
    # p. ej. color de la foto completa). Un modelo de IA tendría su propio bloque.
    segmentation: str | None = "threshold"


# ── Registro global ───────────────────────────────────────────────────────────

ANALYSES: dict[str, AnalysisDescriptor] = {}


def register(
    name: str,
    func: Callable,
    description: str = "",
    supports_batch: bool = True,
    params_schema: list[dict] | None = None,
    icon: str = "chart-bar",
    segmentation: str | None = "threshold",
) -> AnalysisDescriptor:
    """
    Registra un módulo de análisis.

    Uso típico al final de cada módulo:
        register(
            name="Morfometría",
            func=run,
            description="Área, largo, ancho, perímetro y 37 métricas adicionales",
            params_schema=[
                {"key": "area_min", "label": "Área mínima (px²)", "type": "int",
                 "default": 200, "min": 1, "max": 50000,
                 "tooltip": "Objetos más pequeños que este valor serán ignorados. "
                            "Sube este número si el fondo tiene ruido."},
                ...
            ],
            icon="ruler-2",
        )
    """
    descriptor = AnalysisDescriptor(
        name=name,
        func=func,
        description=description,
        supports_batch=supports_batch,
        params_schema=params_schema or [],
        icon=icon,
        segmentation=segmentation,
    )
    ANALYSES[name] = descriptor
    _log.debug("Análisis registrado: %s", name)
    return descriptor


def get(name: str) -> AnalysisDescriptor | None:
    return ANALYSES.get(name)


def list_names() -> list[str]:
    return list(ANALYSES.keys())


# ── Importar módulos — cada import auto-registra vía register() ───────────────
# Para agregar uno nuevo: crear el archivo y agregar el import aquí.

def _load_modules():
    try:
        from . import objects          # noqa: F401  cadena de piezas (etapa 4)
    except Exception as e:
        _log.error("objects no disponible: %s", e)


_load_modules()
