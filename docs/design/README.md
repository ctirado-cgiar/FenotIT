# Diseño de la interfaz

Todo lo que define el aspecto de FenotIT está en pocos lugares. Cambiarlo no requiere
tocar el resto del código.

| Qué | Dónde |
|---|---|
| Temas: colores, letras, tamaños y paletas | `fenotit/assets/themes/<tema>.toml` |
| Cómo se leen los temas | `fenotit/gui/style/theme.py` |
| Botones, títulos de sección y separadores | `fenotit/gui/style/components.py` (estilos en `[buttons]` del tema) |
| Estilos de tablas, listas y barras (ttk) | `fenotit/gui/style/ttk_styles.py` |
| Íconos | `fenotit/assets/icons/` (y los dibujados por defecto en `fenotit/gui/style/icons.py`) |
| Logo | `fenotit/assets/logo.ico` |
| Mosaico y píxeles decorativos (carga, inicio, Acerca de) | `fenotit/gui/style/decor.py` (tonos en el tema) |

## Temas

Un tema es un archivo `.toml` en `fenotit/assets/themes/`. `classic.toml` es el tema por
defecto y tiene todos los valores, cada uno con un comentario que dice dónde se usa.
`light.toml` es un ejemplo de tema que cambia solo algunos valores.

Secciones de un tema:

- `[info]`: nombre del tema en español (`name`) e inglés (`name_en`), como aparece en
  Preferencias.
- `[colors]`: cada color de la interfaz con un nombre por uso (`bg_panel`, `text_muted`,
  `accent`, `bg_topbar`, `text_topbar`, `warning`…).
- `[families]`: familias de letra `sans`, `serif` y `mono`. Con una lista se usa la primera
  instalada en el equipo: `sans = ["Inter", "Segoe UI"]`.
- `[fonts]`: tipos de letra por nombre (`small`, `body`, `title`, `display`…), escritos como
  `"familia tamaño [bold|italic]"`, p. ej. `"sans 9 bold"`.
- `[sizes]`: alto de las filas de tabla, márgenes de los botones, márgenes de los paneles,
  borde de foco…
- `[buttons.<tipo>]`: color de fondo, de texto, al pasar el mouse y al presionar, y letra de
  cada tipo de botón: `primary` (acción principal), `secondary`, `quiet` (Cancelar, Cerrar),
  `run` (Ejecutar) y `link` (sin fondo).
- `[palettes]`: colores que se reparten a las clases y a las series de los gráficos, y tonos
  del logo.

### Crear un tema

1. Copiar `classic.toml` (o `light.toml`, si se van a cambiar pocos valores) con otro nombre,
   p. ej. `fenotit/assets/themes/modern.toml`. Lo que el tema no traiga se toma de
   `classic.toml`.
2. Cambiar `[info]` y los valores.
3. Probarlo sin cambiar las preferencias:

   ```
   FENOTIT_THEME=modern python main.py          # Linux / macOS
   set FENOTIT_THEME=modern && python main.py   # Windows (cmd)
   ```

4. Ver todos sus colores y letras juntos: `python docs/design/theme_sheet.py modern`
   (crea `docs/design/theme_modern.png`).
5. Para dejarlo elegido: Editar → Preferencias → Tema.

`tests/test_architecture.py` revisa que cada color, letra y tamaño que usa la interfaz exista
en `classic.toml` y que `classic.toml` no tenga valores sin uso.

### Qué cambia un tema y qué no

Un tema cambia colores, letras, tamaños y el aspecto de cada tipo de botón. La forma de los
botones, títulos de sección y separadores está en `fenotit/gui/style/components.py`: todas
las pantallas los crean con `button()`, `section()` y `divider()`. La distribución (qué
sección va en cada panel) está en `fenotit/gui/layout.py`, y el orden de los controles dentro
de cada zona, en su archivo de `fenotit/gui/window/`.

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
