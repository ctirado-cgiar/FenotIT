# Desarrollo de FenotIT

Dónde está cada cosa, cómo agregar un método, cómo probar un cambio y cómo enviarlo.

**Grafo de dependencias:** `docs/map/index.html` (abrir en el navegador). Paquetes, módulos,
clases y funciones, qué importa cada módulo y quién lo importa; en los mixins de la ventana,
su estado propio y los métodos de otros mixins que llama. Se genera con
`python docs/map/make_map.py`.

## Preparar el entorno

```bash
git clone https://github.com/ctirado-cgiar/FenotIT.git
cd FenotIT
conda create -n fenotit python=3.12 -y
conda activate fenotit
pip install -r requirements.txt
python main.py
```

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

## Dónde está cada cosa

Los métodos de análisis están en `fenotit/core/` y no dependen de la interfaz. Cada método
es un archivo:

| Quiero mejorar… | Archivo | Prueba |
|---|---|---|
| Umbral y Otsu | `core/segmentation/threshold.py` | `tests/test_pipeline.py` |
| Segmentación por ejemplos | `core/segmentation/classifier.py` | `tests/test_classifier.py` |
| MobileSAM y sus regiones | `core/segmentation/sam.py`, `regions.py` | `tests/test_sam.py`, `test_regions.py` |
| Afinar máscara | `core/segmentation/refine_mask.py`, `core/objects/mask.py` | `tests/test_refine.py` |
| Separar objetos que se tocan | `core/objects/separate.py` | `tests/test_counts.py`, `test_pipeline.py` |
| Filtro de objetos | `core/objects/filter.py` | `tests/test_pipeline.py`, `test_exclude.py` |
| Morfometría | `core/measurements/morphometry.py` | `tests/test_pipeline.py`, `reference.py` |
| Forma (Fourier) | `core/measurements/shape.py`, `core/efd.py` | `tests/test_pipeline.py`, `reference.py` |
| Color de los objetos | `core/measurements/color.py` | `tests/test_colors.py`, `test_pipeline.py` |
| Distancias entre objetos | `core/measurements/distances.py` | `tests/test_pipeline.py` |
| Corrección de color (tarjeta) | `core/corrections/colorcard.py` | `tests/test_colorcard.py` |
| Distorsión del lente | `core/corrections/distortion.py` | `tests/test_distortion.py` |
| Perspectiva (ArUco) | `core/corrections/aruco.py` | `tests/test_corrections.py` |
| Aspecto de la interfaz (tema) | `fenotit/assets/themes/<tema>.toml` | ver `docs/design/README.md` |
| Aspecto de botones, títulos de sección y separadores | `gui/style/components.py` y `[buttons]` del tema | `tests/test_gui.py` |
| Íconos | `fenotit/assets/icons/`, `gui/style/icons.py` | ver `docs/design/README.md` |
| Orden de los controles dentro de una zona | `gui/window/<zona>.py` (p. ej. `topbar.py`, `segment.py`) | `tests/test_gui.py` |
| Mover una sección a otro panel | `gui/layout.py` (`PANELS`) | `tests/test_gui.py` |
| Una parte de la ventana | `gui/window/<área>.py` (p. ej. `table.py`, `inspector.py`) | `tests/test_gui.py` |
| Textos de la interfaz | `fenotit/lang/es.json`, `en.json` | `tests/test_i18n.py` |

## Agregar algo nuevo

| Quiero agregar… | Crear | Conectar |
|---|---|---|
| Un método de segmentación | `core/segmentation/<nombre>.py` con una función `@step("<clave>", "segmenter", …)` | Nada: se carga solo |
| Una medición | `core/measurements/<nombre>.py` con `@step("<clave>", "measurement", requires=("labels",))` | Nada: se carga solo; se activa desde la cadena del análisis |
| Un paso sobre objetos | `core/objects/<nombre>.py` con `@step("<clave>", "processor", …)` | Nada: se carga solo |
| Un análisis (tipo de muestra) | `core/analysis/<nombre>.py` que llame a `register(...)` | Su nombre en `ANALYSIS_MODULES` (`core/analysis/registry.py`) |
| Una parte de la ventana | `gui/window/<nombre>.py` con `class <Nombre>Mixin` | Agregarla a `MainWindow` (`gui/main_window.py`); si crea estado, un `_init_<nombre>()` llamado desde `MainWindow.__init__` |
| Una sección de un panel lateral | `_build_<sección>(parent)` en `gui/layout.py` | Su nombre en `PANELS` |
| Un diálogo | `gui/dialogs/<nombre>.py` | Abrirlo desde la parte de la ventana que corresponda |
| Un modelo de IA | Una carpeta en `models/` con el `.onnx` y su `model.yaml` con la ficha (para qué sirve, propietario, licencia, fuente, cita; ver `core/segmentation/models.py`) | Nada: aparece en Segmentar, Modelo (IA). Los detectores también se agregan desde la ventana con «Agregar modelo» (.onnx o .pt, que se convierte con `core/segmentation/convert.py`) |
| Un tema | `fenotit/assets/themes/<nombre>.toml` | Nada: aparece en Preferencias |
| Un texto | La clave en `fenotit/lang/es.json` y `en.json` | `t("clave")` en el código |

### Cómo agregar una pieza

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

### Cómo agregar un análisis a la interfaz

Un análisis es un **tipo de muestra** (hoy: Objetos, Detección, Distancias, Clasificar áreas y
Color de imagen; después: raíces). Cada uno es un módulo en `fenotit/core/analysis/` que arma su
cadena y llama `register(...)` con su esquema de parámetros, sus pasos en el panel derecho
(`steps`: `"segment"`, `"detect"` y/o `"params"`), los pasos que aparecen con una casilla
marcada (`optional_steps`, p. ej. Detectar en Objetos con «Detecciones por objeto»), los métodos
de segmentación que admite (`seg_methods`) y si trabaja con objetos (`objects`). Ver `objects.py`
(dos pasos y uno opcional), `detection.py` (Detectar y Medir) y `cover.py` (un paso). Se carga desde `ANALYSIS_MODULES` en `registry.py`. En la interfaz, el usuario elige
primero el análisis (tarjetas del panel derecho) y luego ve solo sus pasos.

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
│   │   ├── corrections/           distortion.py, aruco.py (perspectiva), colorcard.py, scale.py
│   │   │                          pipeline.py (orden y aplicación de las correcciones)
│   │   ├── segmentation/          threshold.py, classifier.py (por ejemplos), sam.py (MobileSAM)
│   │   │                          models.py (modelos de IA disponibles), detect.py (detectores de cajas)
│   │   │                          convert.py (.pt → ONNX), converter_setup.py (instalar el conversor)
│   │   │                          vault.py (modelos cifrados que solo abre FenotIT)
│   │   │                          regions.py (categorías de regiones), refine_mask.py (afinar máscara)
│   │   │                          __init__.py (configuración de la segmentación → piezas)
│   │   ├── objects/               mask.py (área de análisis y limpieza), label.py, separate.py, filter.py
│   │   │                          boxes.py (objetos a partir de cajas)
│   │   ├── measurements/          morphometry.py, shape.py, color.py, count.py, classes.py, distances.py
│   │   │                          boxes.py (conteo y tamaño con cajas), detections.py
│   │   │                          (detecciones por objeto), cover.py
│   │   ├── analysis/              objects.py, detection.py, distances.py, cover.py, image_color.py
│   │   ├── pipeline/              base.py (contexto y registro de piezas), cache.py, overlay.py, views.py
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
│   │   ├── style/                 theme.py (lee el tema), ttk_styles.py, icons.py, decor.py
│   │   ├── widgets/               piezas reutilizables (ayuda «?», lista de fotos, inspector, menús…)
│   │   ├── canvas/                sobre la foto: zoom.py, areas.py, tools.py (pincel, clics, regiones)
│   │   ├── dialogs/               una ventana por archivo
│   │   ├── screens/               start.py, splash.py, tour.py
│   │   └── charts/                gráficos y colores 3D
│   ├── assets/                    logo, íconos, themes/ (temas .toml) y models/ (miniaturas máscara/caja)
│   └── lang/                      en.json, es.json
├── models/                        modelos de IA (una carpeta por modelo con model.yaml)
├── docs/
│   ├── desarrollo.md              este archivo
│   ├── methods/                   descripción de cada método (inglés)
│   ├── design/                    guía de diseño de la interfaz y de los temas
│   └── map/                       mapa de conexiones (index.html, se genera con make_map.py)
└── tests/
    ├── reference.py, test_*.py    (test_gui.py: la ventana completa; test_architecture.py: reglas)
    └── data/
        ├── images/                fotos de prueba
        ├── reference/             tablas de referencia de las fotos de prueba
        ├── models/                detectores ONNX de prueba (make_models.py)
        └── Cgiar_*/, Kaggle_*/    fotos de ejemplo por cultivo y origen
```

## La ventana principal

`MainWindow` (`gui/main_window.py`) junta las partes de `gui/window/`: cada archivo es una
clase `<Área>Mixin` con los métodos de un panel o una función. Cada parte crea su estado en
`_init_<área>()` (llamado desde `MainWindow.__init__`) o al armar su panel, y es la única que
lo cambia; las demás lo leen o llaman a un método de esa parte (p. ej. `_clear_result()` de
`results.py`, `_set_step()` de `layers.py`). El mapa (`docs/map/index.html`) muestra, para
cada parte, su estado y a qué métodos de otras partes llama.

## Probar un cambio

Antes de enviar un cambio, corre todas las pruebas:

```bash
python tests/reference.py                      # resultados de las fotos de prueba
for f in tests/test_*.py; do python "$f"; done  # pruebas de cada parte
```

`tests/test_gui.py` abre la ventana real y recorre el flujo completo (agregar fotos,
segmentar, analizar una foto y el lote, capas, inspector, deshacer, gráficos, exportar,
guardar y abrir el proyecto) y muestra la traza del primer error que aparezca. `tests/test_architecture.py` revisa
las reglas de abajo.

`tests/reference.py` compara las tablas de las fotos de `tests/data/images/` con las guardadas en
`tests/data/reference/`. Si tu cambio mejora un método y los números cambian a propósito,
regrábalas con `python tests/reference.py --save` y explica en el envío por qué cambiaron.

Si agregas un método, agrega también su prueba en `tests/test_<método>.py`: una imagen
(real o sintética) y el resultado que se espera.

## Reglas del código

- `fenotit/core/` no importa nada de la interfaz (tkinter).
- Cada variable de la ventana (`self.algo`) se crea y se cambia en un solo archivo de
  `gui/window/`; las demás partes la leen o llaman a un método de ese archivo.
- Dos partes de la ventana no definen métodos con el mismo nombre.
- Columnas de resultados en inglés, en ASCII y con la unidad en el nombre (`area_mm2`,
  `length_px`).
- Textos que ve el usuario siempre con `t("clave")` y en `fenotit/lang/es.json` y `en.json`.
- Botones con `button()`, títulos de sección con `section()` y separadores con `divider()`
  (`gui/style/components.py`).
- Colores, letras y tamaños de la interfaz solo desde el tema (`COLORS`, `FONTS`, `SIZES`
  de `gui/style/theme.py`, que lee `fenotit/assets/themes/`).
- Cada archivo empieza con una línea que dice qué hace; no quedan imports sin uso.
- Variables, parámetros y alias con su nombre completo (`height`, `class_rows`, `mm_per_pixel`),
  no letras sueltas ni abreviaturas de dos letras. Se admiten `x`, `y`, `z`, `i`, `j`, `x0`…`y1`,
  `dx`, `dy`, `cx`, `cy`, `ok`, `id`, `t` (traducción), `np` y `tk`. Los números y colores con
  significado llevan nombre (constante en mayúsculas al inicio del módulo).
- El núcleo no lleva textos para el usuario: los análisis y sus parámetros se identifican por
  clave (`register(key="objects")`) y sus textos están en `fenotit/lang/`
  (`analysis.<clave>.name`, `param.<clave>.<parámetro>.label`).
- Encabezados, docstrings y comentarios dicen qué hace el código.
- Cambios pequeños, uno por envío, con un mensaje corto que diga qué cambia.

## Enviar un cambio

1. Crea una rama: `git checkout -b mejora-separacion`.
2. Haz el cambio, corre las pruebas y actualiza el mapa (`python docs/map/make_map.py`).
3. Sube la rama y abre un *pull request* en GitHub explicando qué cambia y cómo lo probaste.

Si un cambio modifica cómo se mide algo, actualiza también su descripción en `docs/methods/`.
