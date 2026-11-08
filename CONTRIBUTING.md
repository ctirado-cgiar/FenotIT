# Cómo contribuir a FenotIT

Gracias por ayudar. Esta guía dice dónde está cada cosa, cómo probar un cambio y cómo
enviarlo. La arquitectura completa está en [README_ESTRUCTURA.md](README_ESTRUCTURA.md).

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
| Íconos | `fenotit/assets/icons/`, `gui/style/icons.py` | ver `docs/design/README.md` |
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
| Un tema | `fenotit/assets/themes/<nombre>.toml` | Nada: aparece en Preferencias |
| Un texto | La clave en `fenotit/lang/es.json` y `en.json` | `t("clave")` en el código |

Las piezas `@step` reciben un `Context` (`from fenotit.core.pipeline.base import step`) y
escriben en sus tablas (`ctx.object_rows()`, `ctx.image_row()`). Ver «Cómo agregar una pieza»
en [README_ESTRUCTURA.md](README_ESTRUCTURA.md); el mapa muestra un ejemplo de cada tipo.

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

`tests/reference.py` compara las tablas de las fotos de `tests/images/` con las guardadas en
`tests/reference/`. Si tu cambio mejora un método y los números cambian a propósito,
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
- Colores, letras y tamaños de la interfaz solo desde el tema (`COLORS`, `FONTS`, `SIZES`
  de `gui/style/theme.py`, que lee `fenotit/assets/themes/`).
- Cada archivo empieza con una línea que dice qué hace; no quedan imports sin uso.
- Encabezados, docstrings y comentarios dicen qué hace el código.
- Cambios pequeños, uno por envío, con un mensaje corto que diga qué cambia.

## Enviar un cambio

1. Crea una rama: `git checkout -b mejora-separacion`.
2. Haz el cambio, corre las pruebas y actualiza el mapa (`python docs/map/make_map.py`).
3. Sube la rama y abre un *pull request* en GitHub explicando qué cambia y cómo lo probaste.

Si un cambio modifica cómo se mide algo, actualiza también su descripción en `docs/methods/`.
