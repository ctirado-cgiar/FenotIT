# Diseño de la interfaz

Todo lo que define el aspecto de FenotIT está en pocos lugares. Cambiarlo no requiere
tocar el resto del código.

| Qué | Dónde |
|---|---|
| Colores, tipos de letra y paletas | `fenotit/gui/style/theme.py` |
| Íconos | `fenotit/assets/icons/` (y los dibujados por defecto en `fenotit/gui/style/icons.py`) |
| Logo | `fenotit/assets/logo.ico` |
| Mosaico y píxeles decorativos (carga, inicio, Acerca de) | `fenotit/gui/style/decor.py` (tonos en `theme.py`) |

## Colores y letras

`fenotit/gui/style/theme.py` tiene cuatro grupos:

- `COLORS`: cada color de la interfaz con un nombre que dice dónde se usa
  (`bg_panel`, `text_muted`, `accent`, `warning`…). Cambiar un valor cambia todos los
  lugares que lo usan.
- `FONTS`: tipos de letra por nombre (`small`, `body`, `title`, `display`…). La familia
  de letra está una sola vez al inicio del bloque (`_SANS`, `_SERIF`, `_MONO`).
- `CLASS_COLORS` y `CHART_COLORS`: colores que se reparten en orden a las clases nuevas y
  a las series de los gráficos.
- `LOGO_TINTS` y `LOGO_EDGES`: tonos del mosaico del logo.

`docs/design/theme.png` muestra todos los colores y letras con su nombre. Para
actualizarla después de un cambio:

```
python docs/design/theme_sheet.py
```

## Íconos

Cada ícono tiene un nombre (`home`, `zoom_in`, `eye`…) y es de un solo color: la
aplicación lo pinta del color que necesita (blanco sobre la barra azul, gris en los
paneles…). `docs/design/icons.png` muestra todos los íconos con su nombre:

```
python docs/design/icon_sheet.py
```

Para reemplazar un ícono:

1. Dibujarlo en SVG, de un solo color sobre fondo transparente, en un cuadro de
   128 × 128 (el color no importa: se usa solo la forma).
2. Guardarlo como `fenotit/assets/icons/svg/<nombre>.svg`, con el nombre exacto de la hoja
   de íconos.
3. Convertirlo:

   ```
   python docs/design/icons_svg_to_png.py
   ```

   Esto crea `fenotit/assets/icons/<nombre>.png`, que es lo que lee la aplicación.

Un ícono sin archivo se sigue dibujando como hoy. Se pueden reemplazar de uno en uno.

## Ver los cambios

```
python main.py
```
