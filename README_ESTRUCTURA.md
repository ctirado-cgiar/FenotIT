# FenotIT — Estructura del proyecto

## Cómo funciona un análisis

Un análisis es una **cadena de piezas**. Cada pieza vive en el archivo de su método:

| Tipo | Hace | Piezas | Carpeta |
|---|---|---|---|
| `segmenter` | produce la máscara | `threshold`, `otsu`, `whole`, `classify`, `model_segment` | `core/segmentation/` |
| `processor` | transforma máscara u objetos | `refine` (afinar máscara) | `core/segmentation/` |
| `processor` | de la máscara a objetos | `roi`, `clean`, `label`, `separate`, `filter` | `core/objects/` |
| `measurement` | lee los objetos y agrega tablas e imágenes | `count`, `morphometry`, `shape`, `color`, `classes`, `distances` | `core/measurements/` |

Todo pasa por un `Context` (imagen, `mask`, `labels`, `groups`, tablas `objects`/`image`/..., imágenes de vista).
`pipeline.run(image, chain)` corre la cadena y valida que cada pieza tenga lo que necesita.
`pipeline.ChainCache` recalcula solo lo que cambió.

```python
from fenotit.core import pipeline
chain = [{"step": "otsu", "params": {"color_space": "LAB", "channel": 2}},
         {"step": "clean"}, {"step": "separate"}, {"step": "filter"},
         {"step": "morphometry"}, {"step": "count"}]
ctx = pipeline.run(image, chain, mm_per_px=0.05)
ctx.tables["objects"]      # una fila por objeto
```

## Cómo agregar una pieza

Un archivo en `fenotit/core/segmentation/`, `fenotit/core/objects/` o
`fenotit/core/measurements/` (según el tipo); se descubre sola al iniciar
(`pipeline.base.STEP_PACKAGES`):

```python
from fenotit.core.pipeline.base import step

@step("mi_medicion", "measurement", requires=("labels",), params=[
    {"key": "umbral", "type": "float", "default": 0.5, "min": 0, "max": 1},
])
def mi_medicion(ctx, p):
    rows = ctx.object_rows()                     # {object_id: fila}
    for oid in rows:
        rows[oid]["mi_valor"] = ...
    ctx.images["mi_medicion"] = vista            # imagen para revisar el resultado
```

Columnas en inglés y con unidad (`length_mm`, `area_px2`). Una medición solo lee objetos:
no depende de otras mediciones.

## Cómo agregar un análisis a la interfaz

Un análisis de la lista es un **tipo de muestra** (hoy: objetos; después: raíces, conteo con IA).
Cada uno es un módulo en `fenotit/core/analysis/` que arma su cadena y llama `register(...)`
con su esquema de parámetros; ver `objects.py`. Se importa en `registry._load_modules()`.

Esquema de parámetros del panel: `type` (`int`, `float`, `bool`, `section`), `default`, `min`, `max`,
`step`, `group` (se muestra con ▸ bajo esa casilla), `requires`, `unit` (`area` → px² o mm²).
Textos en `fenotit/lang/*.json`: `param.<análisis>.<clave>.label` y `.tip`.

## Análisis posterior

`fenotit/core/stats/` trabaja sobre tablas, sin imágenes: `combine` (varias imágenes, con
`Image_ID`/`Image_name` y columnas de la tabla del usuario), `summarize(by=...)`,
`stats.shape.mean_shapes(by=...)`. `fenotit/core/metadata.py` lee la tabla del usuario (CSV/Excel).

## Carpetas

```
FenotIT/
├── main.py                        lanzador (equivale a `python -m fenotit`)
├── fenotit/
│   ├── core/                      sin interfaz: se puede usar desde Python
│   │   ├── corrections/           distortion.py · aruco.py (perspectiva) · colorcard.py · scale.py
│   │   │                          pipeline.py (orden y aplicación de las correcciones)
│   │   ├── segmentation/          threshold.py · classifier.py (por ejemplos) · sam.py (MobileSAM)
│   │   │                          regions.py (categorías de regiones) · refine_mask.py (afinar máscara)
│   │   │                          __init__.py (configuración de la segmentación → piezas)
│   │   ├── objects/               mask.py (área de análisis y limpieza) · label.py · separate.py · filter.py
│   │   ├── measurements/          morphometry.py · shape.py · color.py · count.py · classes.py · distances.py
│   │   ├── analysis/              objects.py · distances.py · image_color.py (análisis de la lista)
│   │   ├── pipeline/              base.py (contexto y registro de piezas) · cache.py · overlay.py · views.py
│   │   ├── stats/                 análisis posterior (tablas, sin imágenes)
│   │   ├── export/                CSV, Excel y metadatos
│   │   └── project.py, batch.py, image_io.py, efd.py, units.py, uncertainty.py …
│   ├── gui/                       interfaz (tkinter)
│   │   ├── main_window.py         arma la ventana con las partes de window/
│   │   ├── layout.py              qué sección va en cada panel lateral (PANELS)
│   │   ├── window/                la ventana por áreas: project, images, run, results, segment,
│   │   │                          threshold, examples, model, analyze, layers, viewer, bottom,
│   │   │                          table, charts, inspector, areas, calibration, metadata,
│   │   │                          export, history, dialogs, keys, topbar, tour, common
│   │   ├── style/                 theme.py (colores y letras) · icons.py · decor.py
│   │   ├── widgets/               piezas reutilizables (ayuda «?», lista de fotos, inspector, menús…)
│   │   ├── canvas/                sobre la foto: zoom.py · areas.py · tools.py (pincel, clics, regiones)
│   │   ├── dialogs/               una ventana por archivo
│   │   ├── screens/               start.py · splash.py · tour.py
│   │   └── charts/                gráficos y colores 3D
│   ├── assets/                    logo e íconos
│   └── lang/                      en.json, es.json
├── docs/
│   ├── methods/                   descripción de cada método (inglés)
│   └── design/                    guía de diseño de la interfaz
└── tests/
    ├── images/                    fotos de prueba
    ├── reference/                 tablas de referencia de las fotos de prueba
    └── reference.py, test_*.py
```

## Exportación

```
RUTA/
├── mask/  count/  morphometry/  shape/  color/     imágenes de cada vista
└── results/
    ├── objects.csv  image.csv  object_colors.csv  image_colors.csv  object_shape.csv
    └── results.xlsx                                una hoja por tabla
```
