"""
corrections/scale.py
Calibración de escala interactiva.
El usuario marca dos puntos sobre una imagen y especifica la distancia real
y la unidad (µm, mm, cm). El factor se aplica a todo el lote.
"""

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
    """Factor de escala a partir de dos puntos y la distancia real entre ellos.
    ValueError("same_points") si los puntos coinciden; ValueError("distance") si la
    distancia real no es mayor que cero."""
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    dist_px = float(np.sqrt(dx**2 + dy**2))

    if dist_px == 0:
        raise ValueError("same_points")
    if real_distance <= 0:
        raise ValueError("distance")

    unit_per_px = real_distance / dist_px
    px_per_unit = dist_px / real_distance

    return ScaleCalibration(
        px_per_unit=px_per_unit,
        unit_per_px=unit_per_px,
        unit=unit,
        distance_px=dist_px,
        distance_real=real_distance,
    )
