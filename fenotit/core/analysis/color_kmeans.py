"""
analysis/color_kmeans.py
Análisis de color por KMeans.

Flujo correcto:
  1. Segmentar objetos usando los mismos sliders Min/Max + espacio de color
     de la UI (igual que morfometría) → fg_mask
  2. Aplicar ROI si existe
  3. Mostrar objetos en color original, fondo en blanco → imagen "segmentada"
  4. Correr KMeans SOLO sobre píxeles de objetos
  5. Leyenda configurable (tamaño, espacios a exportar)
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from .registry import register, AnalysisResult

from fenotit import log

_log = log.get("analysis.color_kmeans")


# ── Función de segmentación (igual que morfometría) ───────────────────────────

def _build_fg_mask(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Construye máscara de primer plano usando los sliders y espacio de color
    de la UI. Retorna máscara uint8 (255 = objeto, 0 = fondo).
    """
    cs_code = params.get("color_space_code")
    ch_idx  = int(params.get("channel_idx", 0))
    min_val = int(params.get("min_val", 0))
    max_val = int(params.get("max_val", 255))

    # Convertir espacio de color
    if cs_code is not None:
        try:
            converted = cv2.cvtColor(image, cs_code)
        except Exception:
            _log.warning("Conversión de color %s falló; se usa BGR", cs_code, exc_info=True)
            converted = image
    else:
        converted = image

    # Extraer canal
    if converted.ndim == 3:
        ch_idx  = min(ch_idx, converted.shape[2] - 1)
        channel = converted[:, :, ch_idx]
    else:
        channel = converted

    # Umbralización Min/Max
    _, t_min = cv2.threshold(channel, min_val - 1, 255, cv2.THRESH_BINARY)
    _, t_max = cv2.threshold(channel, max_val,     255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(t_min, t_max)

    # Aplicar ROI
    roi = params.get("roi_mask")
    if roi is not None:
        if roi.shape[:2] != image.shape[:2]:
            roi = cv2.resize(roi, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        mask = cv2.bitwise_and(mask, mask, mask=roi)

    return mask


# ── Conversiones de espacio de color para exportación ────────────────────────

def _rgb_to_lab(rgb: tuple) -> tuple:
    """Convierte RGB (0-255) → CIELab."""
    swatch = np.uint8([[list(rgb)]])
    bgr    = cv2.cvtColor(swatch, cv2.COLOR_RGB2BGR)
    lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = (int(v) for v in lab[0, 0])
    # Desnormalizar: OpenCV usa L*100/255, a/b centrados en 128
    L_real = round(L * 100 / 255, 1)
    a_real = a - 128
    b_real = b - 128
    return (L_real, a_real, b_real)


def _luminance(rgb: tuple) -> float:
    return round(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2], 1)


# ── Función principal ─────────────────────────────────────────────────────────

def run(image: np.ndarray, params: dict) -> AnalysisResult:
    """
    params esperados (además de los comunes del UI):
        n_colors:          int    clusters KMeans (1-10)
        legend_scale:      int    escala de la leyenda 1-5 (1=pequeña, 5=grande)
        export_rgb:        bool   incluir RGB en resultados
        export_lab:        bool   incluir CIELab en resultados
        export_pct:        bool   incluir % de área
        export_luminance:  bool   incluir luminancia
    """
    result = AnalysisResult()

    n_colors        = int(params.get("n_colors",       3))
    legend_scale    = int(params.get("legend_scale",   3))
    export_rgb      = bool(params.get("export_rgb",    True))
    export_lab      = bool(params.get("export_lab",    True))
    export_pct      = bool(params.get("export_pct",    True))
    export_lum      = bool(params.get("export_luminance", True))

    result.step_images["imagen original"] = image.copy()

    # ── 1. Segmentar usando los sliders (igual que morfometría) ───────────────
    fg_mask = _build_fg_mask(image, params)

    # Mostrar canal umbralizado como paso intermedio
    cs_code = params.get("color_space_code")
    ch_idx  = int(params.get("channel_idx", 0))
    if cs_code is not None:
        try:
            conv    = cv2.cvtColor(image, cs_code)
            channel = conv[:, :, min(ch_idx, conv.shape[2]-1)]
        except Exception:
            _log.warning("Conversión de color %s falló; se usa canal B", cs_code, exc_info=True)
            channel = image[:, :, 0]
    else:
        channel = image[:, :, min(ch_idx, 2)]
    result.step_images["canal umbralizado"] = cv2.cvtColor(
        channel, cv2.COLOR_GRAY2BGR)
    result.step_images["máscara objetos"] = cv2.cvtColor(
        fg_mask, cv2.COLOR_GRAY2BGR)

    # ── 2. Imagen con objetos en color original, fondo blanco ─────────────────
    img_rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masked_rgb  = np.full_like(img_rgb, 255)          # fondo blanco
    masked_rgb[fg_mask > 0] = img_rgb[fg_mask > 0]   # objetos en color real
    masked_bgr  = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)
    result.step_images["objetos segmentados"] = masked_bgr

    # ── 3. Extraer píxeles de objetos ─────────────────────────────────────────
    pixels = img_rgb[fg_mask > 0].reshape(-1, 3)
    if len(pixels) < n_colors:
        result.status = "sin píxeles suficientes para KMeans"
        result.stats  = {"total píxeles objetos": len(pixels)}
        return result

    # ── 4. KMeans ─────────────────────────────────────────────────────────────
    n_clusters = min(n_colors, len(np.unique(pixels, axis=0)))
    kmeans     = KMeans(n_clusters=n_clusters, n_init=10, random_state=134)
    kmeans.fit(pixels)

    total_pixels = len(pixels)
    colors_data  = []

    for cid in range(n_clusters):
        mask_c = (kmeans.labels_ == cid)
        size   = int(np.sum(mask_c))
        rgb    = tuple(int(v) for v in kmeans.cluster_centers_[cid].astype(int))
        pct    = round(size / total_pixels * 100, 1)
        hex_c  = '#%02x%02x%02x' % rgb
        lum    = _luminance(rgb)
        lab    = _rgb_to_lab(rgb)

        colors_data.append({
            "cluster":     cid + 1,
            "RGB":         str(rgb)   if export_rgb  else None,
            "HEX":         hex_c,
            "L* (CIELab)": lab[0]    if export_lab  else None,
            "a* (CIELab)": lab[1]    if export_lab  else None,
            "b* (CIELab)": lab[2]    if export_lab  else None,
            "luminancia":  lum       if export_lum  else None,
            "% área":      pct       if export_pct  else None,
            "_rgb":        rgb,       # para render (no se exporta)
        })

    # Ordenar por % descendente
    colors_data.sort(key=lambda x: -(x["% área"] or 0))

    # ── 5. Imagen segmentada KMeans ───────────────────────────────────────────
    segmented = np.full_like(img_rgb, 255)
    fg_flat   = (fg_mask > 0).reshape(-1)
    seg_flat  = segmented.reshape(-1, 3)
    seg_flat[fg_flat] = kmeans.cluster_centers_[kmeans.labels_].astype(int)
    segmented = seg_flat.reshape(img_rgb.shape)
    seg_bgr   = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
    result.step_images["colores KMeans"] = seg_bgr

    # ── 6. Imagen con leyenda ─────────────────────────────────────────────────
    legend_img = _draw_legend(seg_bgr, colors_data, legend_scale,
                              export_rgb, export_lab, export_lum, export_pct)
    result.step_images["resultado con leyenda"] = legend_img

    # ── 7. Mediciones (sin campos None) ───────────────────────────────────────
    measurements = []
    for c in colors_data:
        row = {k: v for k, v in c.items()
               if not k.startswith("_") and v is not None}
        measurements.append(row)
    result.measurements = measurements

    # Columnas para la tabla
    cols = ["cluster", "HEX"]
    if export_rgb:  cols.append("RGB")
    if export_lab:  cols += ["L* (CIELab)", "a* (CIELab)", "b* (CIELab)"]
    if export_lum:  cols.append("luminancia")
    if export_pct:  cols.append("% área")
    result.table_columns = cols

    result.stats = {
        "total clusters":       n_clusters,
        "píxeles analizados":   total_pixels,
        "color dominante":      colors_data[0]["HEX"] if colors_data else "—",
    }
    result.extra["colors_data"] = colors_data
    result.extra["fg_mask"]     = fg_mask

    return result


# ── Leyenda configurable ──────────────────────────────────────────────────────

def _draw_legend(image: np.ndarray, colors_data: list,
                 scale: int = 3,
                 show_rgb: bool = True,
                 show_lab: bool = True,
                 show_lum: bool = True,
                 show_pct: bool = True) -> np.ndarray:
    """
    Dibuja leyenda de colores sobre la imagen.
    scale 1-5: controla el tamaño del cuadro de leyenda.
    """
    out = image.copy()
    h, w = out.shape[:2]

    # Tamaño base proporcional a la imagen y al scale
    base       = max(12, min(w, h) // 20)
    swatch_sz  = int(base * 0.8 * scale)
    font_scale = max(0.3, 0.25 * scale)
    font_thick = max(1, scale // 2)
    pad        = max(6, scale * 4)
    line_h     = swatch_sz + pad

    # Construir líneas de texto
    lines_per_color = []
    for c in colors_data:
        parts = [f"  {c['HEX']}"]
        if show_pct and c.get("% área") is not None:
            parts.append(f"  {c['% área']}%")
        if show_rgb and c.get("RGB") is not None:
            parts.append(f"  RGB{c['RGB']}")
        if show_lab and c.get("L* (CIELab)") is not None:
            parts.append(
                f"  Lab({c['L* (CIELab)']}, "
                f"{c['a* (CIELab)']}, {c['b* (CIELab)']})")
        if show_lum and c.get("luminancia") is not None:
            parts.append(f"  Lum: {c['luminancia']}")
        lines_per_color.append(parts)

    # Calcular ancho máximo del texto
    max_text_w = 0
    for parts in lines_per_color:
        for part in parts:
            (tw, _), _ = cv2.getTextSize(
                part, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            max_text_w = max(max_text_w, tw)

    box_w = swatch_sz + max_text_w + pad * 3
    n_lines_total = sum(len(p) for p in lines_per_color)
    box_h = pad * 2 + n_lines_total * (line_h // len(lines_per_color[0])
                                       if lines_per_color else 1) + \
            len(colors_data) * swatch_sz + pad * len(colors_data)

    # Recalcular altura correctamente
    box_h = pad * 2
    for parts in lines_per_color:
        box_h += swatch_sz + (len(parts) - 1) * int(line_h * 0.55) + pad

    # Posición: esquina inferior derecha
    x0 = max(0, w - box_w - pad)
    y0 = max(0, h - box_h - pad)

    # Fondo semitransparente
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0),
                  (x0 + box_w, y0 + box_h),
                  (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.88, out, 0.12, 0, out)
    cv2.rectangle(out, (x0, y0),
                  (x0 + box_w, y0 + box_h),
                  (180, 180, 180), 1)

    # Dibujar colores
    cy = y0 + pad
    for c, parts in zip(colors_data, lines_per_color):
        rgb = c["_rgb"]
        bgr = (rgb[2], rgb[1], rgb[0])

        # Swatch de color
        cv2.rectangle(out,
                      (x0 + pad, cy),
                      (x0 + pad + swatch_sz, cy + swatch_sz),
                      bgr, -1)
        cv2.rectangle(out,
                      (x0 + pad, cy),
                      (x0 + pad + swatch_sz, cy + swatch_sz),
                      (150, 150, 150), 1)

        # Texto a la derecha del swatch
        tx      = x0 + pad + swatch_sz + pad
        line_sp = int(swatch_sz / max(1, len(parts) - 0.5))
        for j, part in enumerate(parts):
            ty = cy + int(swatch_sz * 0.55) + j * line_sp - \
                 (len(parts) - 1) * line_sp // 2
            cv2.putText(out, part,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (30, 30, 30), font_thick,
                        cv2.LINE_AA)

        cy += swatch_sz + pad

    return out


# ── Registro ──────────────────────────────────────────────────────────────────

register(
    name="Color KMeans",
    func=run,
    description=(
        "Segmenta objetos (con los mismos sliders de la UI) y extrae "
        "colores dominantes por KMeans. Exporta RGB, CIELab, luminancia y %."
    ),
    supports_batch=True,
    icon="palette",
    params_schema=[
        {
            "key": "n_colors", "label": "Número de colores (clusters)",
            "type": "int", "default": 3, "min": 1, "max": 10,
            "tooltip": (
                "Cantidad de grupos de color a detectar (clusters KMeans).\n"
                "Más clusters = más detalle, más tiempo de cómputo.\n"
                "Para semillas de frijol se recomiendan 2-4."
            ),
        },
        {
            "key": "legend_scale", "label": "Tamaño leyenda (1-5)",
            "type": "int", "default": 3, "min": 1, "max": 5,
            "tooltip": (
                "Controla el tamaño del cuadro de leyenda sobre la imagen.\n"
                "1 = muy pequeño, 5 = grande y legible.\n"
                "Ajusta según la resolución de tus imágenes."
            ),
        },
        {
            "key": "export_rgb", "label": "Exportar RGB",
            "type": "bool", "default": True,
            "tooltip": "Incluir valores RGB (0-255) en resultados y leyenda.",
        },
        {
            "key": "export_lab", "label": "Exportar CIELab",
            "type": "bool", "default": True,
            "tooltip": (
                "Incluir valores CIELab (L*, a*, b*) en resultados.\n"
                "CIELab es perceptualmente uniforme — ideal para comparar colores."
            ),
        },
        {
            "key": "export_pct", "label": "Exportar % de área",
            "type": "bool", "default": True,
            "tooltip": (
                "Incluir el porcentaje de píxeles que ocupa cada color\n"
                "respecto al total de píxeles de objetos detectados."
            ),
        },
        {
            "key": "export_luminance", "label": "Exportar luminancia",
            "type": "bool", "default": True,
            "tooltip": (
                "Luminancia perceptual: 0.2126·R + 0.7152·G + 0.0722·B.\n"
                "Valores altos = colores claros, valores bajos = colores oscuros."
            ),
        },
    ],
)
