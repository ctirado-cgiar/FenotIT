# Cómo contribuir a FenotIT

Gracias por ayudar. Esta guía dice dónde está cada cosa, cómo probar un cambio y cómo
enviarlo. La arquitectura completa está en [README_ESTRUCTURA.md](README_ESTRUCTURA.md).

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
| Aspecto de la interfaz | `gui/theme.py`, `gui/icons.py` | ver `docs/design/README.md` |
| Textos de la interfaz | `fenotit/lang/es.json`, `en.json` | `tests/test_i18n.py` |

Para agregar un método nuevo (otra segmentación, otra medición), basta un archivo nuevo en
la carpeta que corresponda: las piezas se registran solas con `@step`. Ver «Cómo agregar
una pieza» en [README_ESTRUCTURA.md](README_ESTRUCTURA.md).

## Probar un cambio

Antes de enviar un cambio, corre todas las pruebas:

```bash
python tests/reference.py                      # resultados de las fotos de prueba
for f in tests/test_*.py; do python "$f"; done  # pruebas de cada parte
```

`tests/reference.py` compara las tablas de las fotos de `tests/images/` con las guardadas en
`tests/reference/`. Si tu cambio mejora un método y los números cambian a propósito,
regrábalas con `python tests/reference.py --save` y explica en el envío por qué cambiaron.

Si agregas un método, agrega también su prueba en `tests/test_<método>.py`: una imagen
(real o sintética) y el resultado que se espera.

## Reglas del código

- `fenotit/core/` no importa nada de la interfaz (tkinter).
- Columnas de resultados en inglés, en ASCII y con la unidad en el nombre (`area_mm2`,
  `length_px`).
- Textos que ve el usuario siempre con `t("clave")` y en `fenotit/lang/es.json` y `en.json`.
- Colores y letras de la interfaz solo desde `gui/theme.py`.
- Encabezados, docstrings y comentarios dicen qué hace el código.
- Cambios pequeños, uno por envío, con un mensaje corto que diga qué cambia.

## Enviar un cambio

1. Crea una rama: `git checkout -b mejora-separacion`.
2. Haz el cambio y corre las pruebas.
3. Sube la rama y abre un *pull request* en GitHub explicando qué cambia y cómo lo probaste.

Si un cambio modifica cómo se mide algo, actualiza también su descripción en `docs/methods/`.
