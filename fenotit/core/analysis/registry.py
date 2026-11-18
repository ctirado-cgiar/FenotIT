"""Registro de los análisis disponibles.

Un análisis es un módulo de este paquete con `run(image, params) -> AnalysisResult` que se
registra con `register(key=...)` y aparece en `ANALYSIS_MODULES`. La clave identifica el
análisis en el proyecto, en la exportación y en los textos de `lang/`:

    analysis.<clave>.name / .desc / .step     nombre, descripción y nombre del paso
    param.<clave>.<parámetro>.label / .tip    etiqueta y ayuda de cada parámetro
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
    key: str                          # identificador: "objects", "cover"…
    func: Callable                    # run(image: np.ndarray, params: dict) -> AnalysisResult
    supports_batch: bool = True
    # Parámetros del panel de configuración. Cada entrada:
    # {"key": str, "type": "int"|"float"|"bool"|"choice"|"header", "default": Any, "min": Any, "max": Any}
    params_schema: list[dict] = field(default_factory=list)
    # Íconos Tabler para el botón del menú (nombre sin 'ti-')
    icon: str = "chart-bar"
    # Bloque de segmentación que usa ("threshold" = umbral/Otsu compartido; None = no segmenta,
    # p. ej. color de la foto completa). Un modelo de IA tendría su propio bloque.
    segmentation: str | None = "threshold"
    # Pasos del panel derecho, en orden: "segment" (método de segmentación y afinar máscara),
    # "detect" (detector de cajas) y "params" (parámetros del análisis).
    steps: tuple[str, ...] = ("segment", "params")
    # Pasos que solo aparecen con una casilla del panel marcada: {paso: parámetro}. Se
    # muestran antes de "params".
    optional_steps: dict[str, str] = field(default_factory=dict)
    # Paso que un paso opcional reemplaza cuando aparece: {paso: paso reemplazado}
    replaced_steps: dict[str, str] = field(default_factory=dict)
    # Análisis cuyos textos de parámetros (param.<clave>.*) usa; None = los propios
    param_texts: str | None = None
    # Métodos de segmentación que admite (None = todos)
    seg_methods: tuple[str, ...] | None = None
    # Trabaja con objetos: muestra «Afinar máscara» y la casilla «Objeto» de las clases
    objects: bool = True


# ── Registro global ───────────────────────────────────────────────────────────

ANALYSES: dict[str, AnalysisDescriptor] = {}


def register(
    key: str,
    func: Callable,
    supports_batch: bool = True,
    params_schema: list[dict] | None = None,
    icon: str = "chart-bar",
    segmentation: str | None = "threshold",
    steps: tuple[str, ...] | None = None,
    seg_methods: tuple[str, ...] | None = None,
    objects: bool = True,
    optional_steps: dict[str, str] | None = None,
    replaced_steps: dict[str, str] | None = None,
    param_texts: str | None = None,
) -> AnalysisDescriptor:
    """Registra un análisis con su clave. Se llama al final de cada módulo de análisis."""
    default_steps = ("segment", "params") if segmentation else ("params",)
    descriptor = AnalysisDescriptor(
        key=key,
        func=func,
        supports_batch=supports_batch,
        params_schema=params_schema or [],
        icon=icon,
        segmentation=segmentation,
        steps=tuple(steps) if steps else default_steps,
        seg_methods=tuple(seg_methods) if seg_methods else None,
        objects=objects,
        optional_steps=dict(optional_steps or {}),
        replaced_steps=dict(replaced_steps or {}),
        param_texts=param_texts,
    )
    ANALYSES[key] = descriptor
    _log.debug("Análisis registrado: %s", key)
    return descriptor


def get(key: str) -> AnalysisDescriptor | None:
    return ANALYSES.get(key)


# ── Módulos de análisis: cada uno se registra con register() al importarse ─────
# Orden = orden en el menú Análisis.
ANALYSIS_MODULES = ("objects", "detection", "cover", "image_color")


def _load_modules():
    import importlib
    for name in ANALYSIS_MODULES:
        try:
            importlib.import_module(f"{__package__}.{name}")
        except Exception as e:
            _log.error("%s no disponible: %s", name, e)


_load_modules()
