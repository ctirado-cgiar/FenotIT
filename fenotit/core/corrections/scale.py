"""
corrections/scale.py
Calibración de escala interactiva.
El usuario marca dos puntos sobre una imagen y especifica la distancia real
y la unidad (µm, mm, cm). El factor se aplica a todo el lote.
"""

import cv2
import numpy as np
from dataclasses import dataclass


from fenotit.core.units import TO_MM as UNIT_TO_MM, UNITS  # noqa: F401


@dataclass
class ScaleCalibration:
    px_per_unit: float        # píxeles por unidad
    unit_per_px: float        # unidad por píxel
    unit: str                 # "mm", "µm", etc.
    distance_px: float        # distancia marcada en px
    distance_real: float      # distancia real ingresada
    reference_image: str = "" # path de la imagen usada

    def format(self) -> str:
        return (f"1 px = {self.unit_per_px:.6f} {self.unit}   |   "
                f"1 {self.unit} = {self.px_per_unit:.2f} px")


def compute_scale(
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    real_distance: float,
    unit: str = "mm",
) -> ScaleCalibration:
    """
    Calcula el factor de escala a partir de dos puntos y la distancia real.
    """
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    dist_px = float(np.sqrt(dx**2 + dy**2))

    if dist_px == 0:
        raise ValueError("Los dos puntos son idénticos.")
    if real_distance <= 0:
        raise ValueError("La distancia real debe ser mayor a 0.")

    unit_per_px = real_distance / dist_px
    px_per_unit = dist_px / real_distance

    return ScaleCalibration(
        px_per_unit=px_per_unit,
        unit_per_px=unit_per_px,
        unit=unit,
        distance_px=dist_px,
        distance_real=real_distance,
    )


def draw_scale_annotation(
    image: np.ndarray,
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    label: str = "",
    color: tuple = (0, 100, 220),
) -> np.ndarray:
    """
    Dibuja la línea de calibración sobre la imagen con etiqueta.
    """
    out = image.copy()
    cv2.line(out, point_a, point_b, color, 2, cv2.LINE_AA)

    # Extremos
    for pt in [point_a, point_b]:
        cv2.circle(out, pt, 6, color, -1)
        cv2.circle(out, pt, 7, (255, 255, 255), 1)

    # Etiqueta centrada
    if label:
        mx = (point_a[0] + point_b[0]) // 2
        my = (point_a[1] + point_b[1]) // 2 - 14
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out,
                      (mx - tw//2 - 4, my - th - 4),
                      (mx + tw//2 + 4, my + 4),
                      (255, 255, 255), -1)
        cv2.putText(out, label,
                    (mx - tw//2, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2, cv2.LINE_AA)
    return out
