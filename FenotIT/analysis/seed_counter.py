"""
analysis/seed_counter.py
Conteo de semillas por transformada de distancia + picos locales.
Adaptado de 15_contadorSemillas.py

Diferencias:
- Parámetros vienen de params{} en lugar de constantes globales
- roi_mask aplicado antes del análisis
- Retorna AnalysisResult con imágenes de pasos y conteo
"""

import cv2
import numpy as np
from .registry import register, AnalysisResult

try:
    from skimage.feature import peak_local_max
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False


def run(image: np.ndarray, params: dict) -> AnalysisResult:
    """
    params esperados:
        threshold_val:   int    umbral en canal Cr de YCrCb (0-255)
        min_distance:    int    distancia mínima entre semillas en px
        peak_threshold:  float  fracción del máximo de la distancia transform (0-1)
        roi_mask:        np.ndarray | None
        scale_factor:    int    factor de reducción de imagen (1=original, 4=1/4)
    """
    result = AnalysisResult()

    if not SKIMAGE_OK:
        result.status = "error"
        result.error  = "scikit-image no está instalado. Instala con: pip install scikit-image"
        return result

    threshold_val  = int(params.get("threshold_val",  125))
    min_distance   = int(params.get("min_distance",    10))
    peak_threshold = float(params.get("peak_threshold", 0.20))
    scale_factor   = int(params.get("scale_factor",     4))

    result.step_images["imagen original"] = image.copy()

    # ── 1. Reducir resolución ─────────────────────────────────────────────────
    sf = max(1, scale_factor)
    img_small = cv2.resize(image,
                           (image.shape[1] // sf, image.shape[0] // sf),
                           interpolation=cv2.INTER_AREA)

    # ── 2. Aplicar ROI (en escala reducida) ───────────────────────────────────
    roi_mask = params.get("roi_mask")
    if roi_mask is not None:
        roi_small = cv2.resize(roi_mask,
                               (img_small.shape[1], img_small.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    else:
        roi_small = None

    # ── 3. Umbralización en canal Cr ─────────────────────────────────────────
    cr_channel = cv2.cvtColor(img_small, cv2.COLOR_BGR2YCrCb)[:, :, 1]
    result.step_images["canal Cr (YCrCb)"] = cv2.cvtColor(cr_channel, cv2.COLOR_GRAY2BGR)

    _, thresh = cv2.threshold(cr_channel, threshold_val, 255, cv2.THRESH_BINARY)

    if roi_small is not None:
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_small)

    kernel  = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    result.step_images["máscara umbralizada"] = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    # ── 4. Transformada de distancia ─────────────────────────────────────────
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    result.step_images["transformada distancia"] = cv2.cvtColor(dist_norm, cv2.COLOR_GRAY2BGR)

    # ── 5. Picos locales = centros de semillas ────────────────────────────────
    threshold_abs = peak_threshold * dist.max() if dist.max() > 0 else 0.1
    coords = peak_local_max(dist,
                            min_distance=min_distance,
                            threshold_abs=threshold_abs)

    # Filtrar coords fuera de imagen
    h, w = img_small.shape[:2]
    valid = (coords[:, 0] >= 0) & (coords[:, 0] < h) & \
            (coords[:, 1] >= 0) & (coords[:, 1] < w)
    coords = coords[valid]

    count = len(coords)

    # ── 6. Imagen anotada ─────────────────────────────────────────────────────
    annotated = img_small.copy()
    for i, (row, col) in enumerate(coords):
        cv2.circle(annotated, (int(col), int(row)), 6, (0, 220, 80), 2)
        cv2.putText(annotated, str(i + 1),
                    (int(col) - 6, int(row) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Escalar anotada al tamaño original para mostrar
    annotated_full = cv2.resize(annotated,
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    result.step_images["semillas detectadas"] = annotated_full

    # ── 7. Resultados ─────────────────────────────────────────────────────────
    result.measurements = [
        {"index": i + 1,
         "centroid_x (display)": int(coords[i][1]) * sf,
         "centroid_y (display)": int(coords[i][0]) * sf}
        for i in range(count)
    ]
    result.table_columns = ["index", "centroid_x (display)", "centroid_y (display)"]
    result.stats = {"total semillas": count}
    result.extra["coords"]      = coords
    result.extra["scale_factor"] = sf

    return result


# ── Registro ──────────────────────────────────────────────────────────────────

register(
    name="Contador de semillas",
    func=run,
    description="Cuenta semillas usando transformada de distancia y picos locales",
    supports_batch=True,
    icon="seed",
    params_schema=[
        {
            "key": "threshold_val", "label": "Umbral canal Cr",
            "type": "int", "default": 125, "min": 0, "max": 255,
            "tooltip": (
                "Umbral en el canal Cr del espacio YCrCb para separar semillas del fondo.\n"
                "Las semillas de frijol suelen tener Cr > 125.\n"
                "Baja este valor si se detectan pocas semillas; súbelo si hay falsos positivos."
            ),
        },
        {
            "key": "min_distance", "label": "Distancia mínima entre semillas (px)",
            "type": "int", "default": 10, "min": 1, "max": 200,
            "tooltip": (
                "Distancia mínima en píxeles entre dos centros de semillas detectados.\n"
                "Baja este valor si las semillas están muy juntas.\n"
                "Sube si una semilla está siendo contada dos veces."
            ),
        },
        {
            "key": "peak_threshold", "label": "Umbral de picos (fracción)",
            "type": "float", "default": 0.20, "min": 0.05, "max": 0.95,
            "tooltip": (
                "Fracción del valor máximo de la transformada de distancia.\n"
                "Solo se detectan picos por encima de este porcentaje del máximo.\n"
                "Sube para detectar solo semillas grandes; baja para incluir las pequeñas."
            ),
        },
        {
            "key": "scale_factor", "label": "Factor de reducción",
            "type": "int", "default": 4, "min": 1, "max": 8,
            "tooltip": (
                "La imagen se reduce a 1/N para acelerar el procesamiento.\n"
                "Factor 4 = imagen al 25% del tamaño original.\n"
                "Usa 1 para máxima precisión (más lento con imágenes grandes)."
            ),
        },
    ],
)
