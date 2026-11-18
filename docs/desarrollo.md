# Desarrollo de FenotIT

## Entorno

```bash
git clone https://github.com/ctirado-cgiar/FenotIT.git
cd FenotIT
conda create -n fenotit python=3.12 -y
conda activate fenotit
pip install -r requirements.txt
python main.py
```

## Cómo está armado

`fenotit/core/` tiene los métodos de análisis y no usa la interfaz; `fenotit/gui/` es la
ventana (tkinter). Un análisis es una cadena de piezas que pasan un `Context` de una a otra:
segmentación (`core/segmentation/`), objetos (`core/objects/`) y mediciones
(`core/measurements/`). Cada pieza está en el archivo de su método.

```python
from fenotit.core import pipeline
chain = [{"step": "otsu", "params": {"color_space": "LAB", "channel": 2}},
         {"step": "clean"}, {"step": "separate"}, {"step": "filter"},
         {"step": "morphometry"}, {"step": "count"}]
ctx = pipeline.run(image, chain, mm_per_px=0.05)
ctx.tables["objects"]      # una fila por objeto
```

`docs/map/index.html` muestra qué importa cada módulo y quién lo usa
(`python docs/map/make_map.py` lo actualiza).

## Agregar una medición

Un archivo nuevo en `fenotit/core/measurements/`. Se carga solo al iniciar:

```python
from fenotit.core.pipeline.base import step

@step("mi_medicion", "measurement", requires=("labels",))
def mi_medicion(ctx, params):
    rows = ctx.object_rows()
    for object_id in rows:
        rows[object_id]["mi_valor_mm"] = ...
```

Lo mismo para una segmentación (`"segmenter"`, en `core/segmentation/`) o un paso sobre
objetos (`"processor"`, en `core/objects/`). Un análisis nuevo es un módulo en
`core/analysis/` que llama a `register(...)` y se agrega a `ANALYSIS_MODULES` en
`registry.py`; `cover.py` es el ejemplo más corto.

## Probar

```bash
python tests/reference.py
python -m pytest tests
```

`reference.py` compara los resultados de las fotos de `tests/data/images/` con los guardados.
Si un cambio mejora un método y los números cambian a propósito:
`python tests/reference.py --save`. `test_gui.py` recorre la ventana completa y
`test_architecture.py` revisa las reglas de abajo.

## Reglas

- `core/` no importa nada de `gui/`.
- Columnas de resultados en inglés, con la unidad en el nombre (`area_mm2`).
- Textos de la interfaz con `t("clave")`, en `fenotit/lang/es.json` y `en.json`.
- Colores, letras y tamaños desde el tema (`fenotit/assets/themes/`).
- Nombres completos, sin variables de una letra.
- Si cambia cómo se mide algo, se actualiza su descripción en `docs/methods/`.

Para enviar un cambio: una rama, las pruebas pasando y un *pull request* que diga qué cambia.
