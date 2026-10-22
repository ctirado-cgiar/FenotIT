"""Incertidumbre de medición por discretización (píxeles) y escala.

Un borde que cae al azar dentro de los píxeles agrega a cada píxel del contorno un error
uniforme de varianza 1/12 px²; sumado sobre el perímetro P (px):

    σ_A ≈ √(P / 12)  px²          →   ε_A = σ_A / A   (desviación relativa del área)

Con la escala s (mm/px) de error relativo ε_s, el área en mm² (A·s²) suma ambos:

    ε = √(ε_A² + (2·ε_s)²)

Para planear la captura: con P = k·√A (círculo k = 2√π), el área mínima en px para un
error relativo ε es A = (√(k/12) / ε)^(4/3)."""
from __future__ import annotations

import math

CIRCLE_K = 2 * math.sqrt(math.pi)


def area_rel_sd(area_px: float, perimeter_px: float) -> float:
    if area_px <= 0:
        return float("nan")
    return math.sqrt(max(perimeter_px, 0) / 12) / area_px


def combined(area_rel: float, scale_rel: float = 0.0) -> float:
    return math.hypot(area_rel, 2 * scale_rel)


def min_area_px(target_rel: float, k: float = CIRCLE_K) -> float:
    return (math.sqrt(k / 12) / target_rel) ** (4 / 3)
