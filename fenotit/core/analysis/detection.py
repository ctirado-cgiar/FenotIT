"""Análisis de detección: un detector de cajas (modelo de IA) encuentra los objetos y su
clase; se cuentan y se miden con las mismas mediciones del análisis de objetos. Las cajas
pueden pasar a máscaras (MobileSAM) para medir el contorno, la forma y el color."""
from __future__ import annotations

import numpy as np

from fenotit.core.analysis import objects     # módulo (no nombres): evita el import circular
from fenotit.core.analysis.object_params import PARAMS_SCHEMA
from fenotit.core.analysis.registry import AnalysisResult, register

# Parámetros del análisis de objetos que no aplican a cajas de un detector
NOT_FOR_DETECTION = ("touching", "measure_touching", "measure_classes", "measure_detections")


def segmentation(detector: dict) -> dict:
    """La segmentación que usa la cadena de objetos: el detector como modelo de cajas."""
    return {"method": "model", "refine": [],
            "model": {**detector, "output": "boxes", "name": detector.get("name", "")}}


def run(image: np.ndarray, params: dict) -> AnalysisResult:
    detector = params.get("detector") or {}
    if not detector.get("name"):
        raise FileNotFoundError("No detection model chosen")
    settings = {key: value for key, value in params.items() if key not in NOT_FOR_DETECTION}
    settings["segmentation"] = segmentation(detector)
    return objects.run(image, settings)


register(
    key="detection",
    func=run,
    icon="target",
    segmentation=None,
    steps=("detect", "params"),
    param_texts="objects",
    params_schema=[item for item in PARAMS_SCHEMA if item["key"] not in NOT_FOR_DETECTION],
)
