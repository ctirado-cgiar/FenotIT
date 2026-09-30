# FenotIT — Estructura del proyecto

## Cómo funciona un análisis

Un análisis es una **cadena de piezas** (`fenotit/core/pipeline/`):

| Tipo | Hace | Piezas actuales |
|---|---|---|
| `segmenter` | produce la máscara (o detecciones) | `threshold`, `otsu` |
| `processor` | transforma máscara u objetos | `roi`, `clean`, `label`, `separate`, `filter` |
| `measurement` | lee los objetos y agrega tablas e imágenes | `count`, `morphometry`, `shape`, `color` |

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

Un archivo en `fenotit/core/pipeline/steps/`; se descubre sola:

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
├── main.py                       # Lanzador (equivale a `python -m fenotit`)
├── fenotit/
│   ├── core/                     # Sin tkinter
│   │   ├── pipeline/             # base, cache, views, steps/
│   │   ├── analysis/             # registry + objects (análisis de la interfaz)
│   │   ├── stats/                # análisis posterior
│   │   ├── corrections/          # distorsión, perspectiva ArUco, tarjeta de color, escala
│   │   ├── efd.py, metadata.py, colorspaces.py, project.py, image_io.py
│   │   └── export/exporter.py    # una tabla CSV por nivel + Excel
│   ├── gui/                      # tkinter
│   └── lang/                     # en.json, es.json
└── tests/
    ├── images/                   # 01-10 (sueltas) y 23xxx (pegadas)
    ├── reference/                # tablas de referencia de las 10 fotos
    ├── reference.py, test_pipeline.py, test_counts.py, ...
```

## Exportación

```
RUTA/
├── mask/  count/  morphometry/  shape/  color/     imágenes de cada vista
└── results/
    ├── objects.csv  image.csv  object_colors.csv  image_colors.csv  object_shape.csv
    └── results.xlsx                                una hoja por tabla
```
