# FenotIT — Estructura del proyecto

## Cómo agregar un nuevo análisis

1. **Crear el módulo** en `fenotit/core/analysis/mi_analisis.py`

```python
import numpy as np
from .registry import register, AnalysisResult

def run(image: np.ndarray, params: dict) -> AnalysisResult:
    result = AnalysisResult()
    # ... lógica del análisis ...
    result.step_images["paso 1"] = imagen_intermedia
    result.measurements = [{"index": 1, "valor": 42}]
    result.stats = {"total": 1}
    return result

register(
    name="Mi Análisis",
    func=run,
    description="Qué hace este análisis",
    supports_batch=True,
    icon="chart-bar",
    params_schema=[
        {
            "key":     "mi_parametro",
            "label":   "Nombre visible en UI",
            "type":    "int",           # int | float | bool
            "default": 100,
            "min":     1,
            "max":     1000,
            "tooltip": "Explica qué hace este parámetro y cómo afecta el resultado.",
        },
    ],
)
```

2. **Registrar el import** en `fenotit/core/analysis/registry.py`, sección `_load_modules()`:

```python
try:
    from . import mi_analisis   # noqa: F401
except Exception as e:
    print(f"[registry] mi_analisis no disponible: {e}")
```

Eso es todo. La UI detecta el nuevo análisis automáticamente.

---

## Estructura de carpetas

```
FenotIT/
├── main.py                       # Lanzador (equivale a `python -m fenotit`)
├── pyproject.toml
├── fenotit/
│   ├── __init__.py               # __version__
│   ├── cli.py                    # Línea de comandos; sin argumentos abre la GUI
│   ├── assets/                   # logo.ico, about.txt
│   ├── core/                     # Sin tkinter
│   │   ├── image_io.py           # Loader Unicode-safe + ImageScaler
│   │   ├── analysis/
│   │   │   ├── registry.py       # Registro central + AnalysisResult
│   │   │   ├── morphometry.py
│   │   │   ├── color_kmeans.py
│   │   │   └── seed_counter.py
│   │   ├── corrections/          # distortion, color_card, aruco, scale
│   │   └── export/exporter.py
│   └── gui/                      # tkinter
│       ├── app.py                # Splash + ventana principal
│       ├── main_window.py
│       ├── config_panel.py, charts.py, calibration_dialogs.py, ...
│       └── roi/selectors.py
└── tests/
    ├── images/                   # Imágenes de prueba
    ├── reference/                # Resultados de referencia
    └── reference.py              # Compara contra la referencia
```

## Estructura de params{}

Todos los módulos de análisis reciben un dict `params` con:

| Clave              | Fuente           | Descripción                          |
|--------------------|------------------|--------------------------------------|
| `color_space_code` | selector UI      | Código cv2 de conversión de espacio  |
| `channel_idx`      | radio buttons    | Índice de canal (0, 1, 2)            |
| `min_val`          | slider           | Umbral mínimo (0-255)                |
| `max_val`          | slider           | Umbral máximo (0-255)                |
| `roi_mask`         | ROI selector     | Máscara en coordenadas originales    |
| `mm_per_pixel`     | calibración      | Factor de escala (None = píxeles)    |
| `*`                | config panel     | Parámetros específicos del análisis  |

## Escalado de imágenes

`fenotit/core/image_io.ImageScaler` maneja todo el escalado:
- `scaler.original` → imagen full-res para análisis
- `scaler.display`  → imagen escalada al canvas para visualización
- `scaler.display_to_original(x, y)` → convierte coords de click → originales
- `scaler.scale_mask_to_original(mask)` → ROI display → ROI para análisis

Los módulos de análisis **siempre reciben la imagen original** y el ROI
convertido a coordenadas originales. El escalado es 100% transparente.

## Exportación

La estructura de carpetas de salida es **idéntica al pipeline**:

```
RUTA/
├── Morfometria/
│   ├── metricasCompletas.csv
│   └── pasos/
│       └── nombre_imagen/
│           ├── canal_seleccionado.jpg
│           ├── mascara_binaria.jpg
│           └── contornos_detectados.jpg
├── Colorimetria/
│   ├── analisis_colores.csv
│   └── pasos/...
└── conteo/
    └── reporte_20260101_120000.csv
```
