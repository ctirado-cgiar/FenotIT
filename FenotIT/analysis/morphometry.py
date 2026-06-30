"""
analysis/morphometry.py
Morfometría completa de objetos segmentados.
Adaptado de scripts 08_analisisMorfometria.py y 09_summarizeMorfometria.py.

Diferencias respecto al script original:
- Los parámetros de umbralización vienen de params{} (UI) en lugar de hardcode
- El ROI viene de params["roi_mask"] en lugar de imagen completa
- mm_per_pixel viene de params["mm_per_pixel"] de la calibración de FenotIT
- Los filtros de área/ancho/largo son configurables desde el panel de config
"""

import cv2
import numpy as np
from .registry import register, AnalysisResult


# ── Función principal ─────────────────────────────────────────────────────────

def run(image: np.ndarray, params: dict) -> AnalysisResult:
    """
    Analiza morfometría de objetos en la imagen.

    params esperados:
        color_space_code: int | None    cv2 conversion code (None = BGR)
        channel_idx:      int           índice del canal a umbralizar (0,1,2)
        min_val:          int           umbral mínimo (0-255)
        max_val:          int           umbral máximo (0-255)
        roi_mask:         np.ndarray|None   máscara ROI (None = imagen completa)
        mm_per_pixel:     float|None    factor de escala (None = medir en px)
        area_min:         int           área mínima en px² a considerar
        area_max:         int           área máxima en px² a considerar
        ancho_min:        int           ancho mínimo (px)
        ancho_max:        int           ancho máximo (px)
        largo_min:        int           largo mínimo (px)
        largo_max:        int           largo máximo (px)
        ar_max:           float         relación largo/ancho máxima
    """
    result = AnalysisResult()

    # ── 1. Aplicar espacio de color y extraer canal ───────────────────────────
    cs_code = params.get("color_space_code")
    ch_idx  = params.get("channel_idx", 0)

    if cs_code is not None:
        try:
            converted = cv2.cvtColor(image, cs_code)
        except Exception:
            converted = image
    else:
        converted = image

    if converted.ndim == 3:
        ch_idx = min(ch_idx, converted.shape[2] - 1)
        channel = converted[:, :, ch_idx]
    else:
        channel = converted

    result.step_images["canal seleccionado"] = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)

    # ── 2. Aplicar ROI ────────────────────────────────────────────────────────
    roi_mask = params.get("roi_mask")
    if roi_mask is not None:
        # Asegurarse de que la máscara tenga el mismo tamaño que la imagen
        if roi_mask.shape[:2] != image.shape[:2]:
            roi_mask = cv2.resize(roi_mask, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        channel_roi = cv2.bitwise_and(channel, channel, mask=roi_mask)
    else:
        channel_roi = channel

    # ── 3. Umbralización ──────────────────────────────────────────────────────
    min_val = int(params.get("min_val", 0))
    max_val = int(params.get("max_val", 255))

    _, thresh_min = cv2.threshold(channel_roi, min_val - 1, 255, cv2.THRESH_BINARY)
    _, thresh_max = cv2.threshold(channel_roi, max_val,     255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_and(thresh_min, thresh_max)

    if roi_mask is not None:
        binary = cv2.bitwise_and(binary, binary, mask=roi_mask)

    result.step_images["máscara binaria"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ── 4. Encontrar contornos ────────────────────────────────────────────────
    contours_raw, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── 5. Filtrar contornos ──────────────────────────────────────────────────
    area_min  = int(params.get("area_min",  200))
    area_max  = int(params.get("area_max",  500000))
    ancho_min = int(params.get("ancho_min", 5))
    ancho_max = int(params.get("ancho_max", 99999))
    largo_min = int(params.get("largo_min", 10))
    largo_max = int(params.get("largo_max", 99999))
    ar_max    = float(params.get("ar_max",  10.0))

    valid_contours = []
    for cnt in contours_raw:
        area = cv2.contourArea(cnt)
        if not (area_min <= area <= area_max):
            continue
        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), _ = rect
        w, h = (h, w) if w > h else (w, h)   # w = ancho (menor), h = largo (mayor)
        if not (ancho_min <= w <= ancho_max):
            continue
        if not (largo_min <= h <= largo_max):
            continue
        if w > 0 and (h / w) > ar_max:
            continue
        valid_contours.append(cnt)

    # ── 6. Medir objetos ──────────────────────────────────────────────────────
    mm_per_pixel = params.get("mm_per_pixel")
    factor = mm_per_pixel if mm_per_pixel else 1.0
    unit   = "mm" if mm_per_pixel else "px"

    measurements = []
    img_contours = image.copy()

    for i, cnt in enumerate(valid_contours):
        area_px  = cv2.contourArea(cnt)
        rect     = cv2.minAreaRect(cnt)
        (cx, cy), (w_px, h_px), angle = rect
        w_px, h_px = (h_px, w_px) if w_px > h_px else (w_px, h_px)
        perim_px = cv2.arcLength(cnt, True)

        # Métricas adicionales (del script 08)
        hull         = cv2.convexHull(cnt)
        hull_area    = cv2.contourArea(hull)
        convexity    = area_px / hull_area if hull_area > 0 else 0
        circularity  = (4 * np.pi * area_px) / (perim_px ** 2) if perim_px > 0 else 0
        ar           = h_px / w_px if w_px > 0 else 0
        solidity     = area_px / hull_area if hull_area > 0 else 0
        elongation   = 1 - (w_px / h_px) if h_px > 0 else 0
        rectangularity = area_px / (w_px * h_px) if (w_px * h_px) > 0 else 0

        # Elipse (para Major/Minor axis y excentricidad)
        eccentricity = 0.0
        major_axis = h_px
        minor_axis = w_px
        if len(cnt) >= 5:
            try:
                (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
                major_axis   = max(MA, ma)
                minor_axis   = min(MA, ma)
                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            except Exception:
                pass

        # Momentos → centroide
        M  = cv2.moments(cnt)
        cx_c = int(M["m10"] / M["m00"]) if M["m00"] != 0 else int(cx)
        cy_c = int(M["m01"] / M["m00"]) if M["m00"] != 0 else int(cy)

        # Radios desde centroide
        pts   = cnt.reshape(-1, 2).astype(float)
        dists = np.sqrt((pts[:, 0] - cx_c)**2 + (pts[:, 1] - cy_c)**2)
        r_min, r_mean, r_max = dists.min(), dists.mean(), dists.max()
        r_ratio = r_min / r_max if r_max > 0 else 0

        # Diámetros equivalentes
        d_mean = 2 * r_mean
        d_min  = 2 * r_min
        d_max  = 2 * r_max

        m = {
            "index":       i + 1,
            f"área ({unit}²)":    round(area_px  * factor**2, 2),
            f"largo ({unit})":    round(h_px     * factor,    2),
            f"ancho ({unit})":    round(w_px     * factor,    2),
            f"perímetro ({unit})": round(perim_px * factor,   2),
            "AR":          round(ar,           3),
            "circularidad": round(circularity, 3),
            "solidez":     round(solidity,     3),
            "convexidad":  round(convexity,    3),
            "elongación":  round(elongation,   3),
            "rectangularidad": round(rectangularity, 3),
            "excentricidad": round(eccentricity, 3),
            f"eje_mayor ({unit})": round(major_axis * factor, 2),
            f"eje_menor ({unit})": round(minor_axis * factor, 2),
            f"radio_min ({unit})": round(r_min  * factor, 2),
            f"radio_mean ({unit})": round(r_mean * factor, 2),
            f"radio_max ({unit})": round(r_max  * factor, 2),
            "radio_ratio": round(r_ratio, 3),
            f"diam_min ({unit})": round(d_min  * factor, 2),
            f"diam_mean ({unit})": round(d_mean * factor, 2),
            f"diam_max ({unit})": round(d_max  * factor, 2),
            "centroid_x":  cx_c,
            "centroid_y":  cy_c,
            "_contour":    cnt,   # solo para dibujar — no se exporta
        }
        measurements.append(m)

        # Dibujar en imagen de resultados
        cv2.drawContours(img_contours, [cnt], -1, (0, 220, 100), 2)
        cv2.putText(img_contours, str(i + 1),
                    (cx_c - 8, cy_c + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    result.step_images["contornos detectados"] = img_contours

    # ── 7. Stats globales ─────────────────────────────────────────────────────
    n = len(measurements)
    if n > 0:
        key_area = f"área ({unit}²)"
        key_largo = f"largo ({unit})"
        areas  = [m[key_area]  for m in measurements]
        largos = [m[key_largo] for m in measurements]
        result.stats = {
            "total objetos": n,
            f"área promedio ({unit}²)": round(float(np.mean(areas)),  2),
            f"área std ({unit}²)":      round(float(np.std(areas)),   2),
            f"largo promedio ({unit})": round(float(np.mean(largos)), 2),
            f"largo std ({unit})":      round(float(np.std(largos)),  2),
        }
    else:
        result.stats = {"total objetos": 0}
        result.status = "sin objetos detectados"

    # Quitar _contour de los datos de medición (no exportar)
    result.measurements = [
        {k: v for k, v in m.items() if k != "_contour"}
        for m in measurements
    ]
    # Guardar contornos en extra para que la UI los use al dibujar
    result.extra["contours"] = valid_contours
    result.extra["binary"]   = binary

    # Columnas prioritarias para la tabla de la UI
    result.table_columns = [
        "index",
        f"área ({unit}²)",
        f"largo ({unit})",
        f"ancho ({unit})",
        f"perímetro ({unit})",
        "AR", "circularidad", "solidez",
    ]

    return result


# ── Registro ──────────────────────────────────────────────────────────────────

register(
    name="Morfometría",
    func=run,
    description="Mide área, largo, ancho, perímetro y 17 métricas de forma por objeto",
    supports_batch=True,
    icon="ruler-2",
    params_schema=[
        {
            "key": "area_min", "label": "Área mínima (px²)",
            "type": "int", "default": 200, "min": 1, "max": 100000,
            "tooltip": (
                "Objetos con área menor a este valor serán ignorados.\n"
                "Sube este número para filtrar ruido de fondo pequeño.\n"
                "Ejemplo: si tus semillas tienen ~500 px² sube a 200-300."
            ),
        },
        {
            "key": "area_max", "label": "Área máxima (px²)",
            "type": "int", "default": 500000, "min": 100, "max": 10000000,
            "tooltip": (
                "Objetos con área mayor a este valor serán ignorados.\n"
                "Útil para excluir la bandeja o artefactos grandes.\n"
                "Deja el valor alto si no hay ruido grande."
            ),
        },
        {
            "key": "ancho_min", "label": "Ancho mínimo (px)",
            "type": "int", "default": 5, "min": 1, "max": 10000,
            "tooltip": (
                "Ancho mínimo del rectángulo envolvente de cada objeto.\n"
                "Filtra objetos muy delgados (líneas, artefactos)."
            ),
        },
        {
            "key": "ancho_max", "label": "Ancho máximo (px)",
            "type": "int", "default": 99999, "min": 1, "max": 99999,
            "tooltip": "Ancho máximo permitido. Deja alto si no necesitas filtrar.",
        },
        {
            "key": "largo_min", "label": "Largo mínimo (px)",
            "type": "int", "default": 10, "min": 1, "max": 10000,
            "tooltip": "Largo mínimo del rectángulo envolvente.",
        },
        {
            "key": "largo_max", "label": "Largo máximo (px)",
            "type": "int", "default": 99999, "min": 1, "max": 99999,
            "tooltip": "Largo máximo permitido.",
        },
        {
            "key": "ar_max", "label": "Relación largo/ancho máx.",
            "type": "float", "default": 10.0, "min": 1.0, "max": 50.0,
            "tooltip": (
                "Filtra objetos muy alargados (relación largo/ancho).\n"
                "Semillas de frijol típicamente < 3.0.\n"
                "Baja este valor para excluir tallos, raíces delgadas."
            ),
        },
    ],
)
