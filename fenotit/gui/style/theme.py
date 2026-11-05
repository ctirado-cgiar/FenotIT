"""Tema visual de la interfaz: todos los colores, letras y paletas, con nombre.

Ningún archivo de la interfaz escribe un color ni un tipo de letra directamente: los
toma de aquí. Para cambiar el aspecto de la aplicación basta con editar este archivo.

COLORS        colores de la interfaz (#RRGGBB)
FONTS         tipos de letra: (familia, tamaño[, estilo])
CLASS_COLORS  colores que se asignan a las clases y categorías nuevas
CHART_COLORS  colores de las series en los gráficos
LOGO_TINTS    tonos del mosaico del logo (de oscuro a claro)
LOGO_EDGES    bordes de los píxeles decorativos (de oscuro a claro)
"""

COLORS = {
    # Fondos
    "bg":            "#F7F7F7",   # ventana principal
    "bg_panel":      "#ECECEC",   # paneles laterales
    "bg_card":       "#FFFFFF",   # tarjetas, lienzos, campos de texto
    "bg_topbar":     "#2166AC",   # barra superior
    "bg_table":      "#FFFFFF",
    "bg_table_sel":  "#D6E8F7",   # fila elegida
    "bg_chart":      "#FAFAFA",   # área de dibujo de los gráficos
    "bg_tooltip":    "#FFFFE6",   # texto al pasar el mouse
    "placeholder":   "#EBEEF2",   # miniatura que aún no carga, pista de la barra de carga

    # Texto
    "text":          "#1A1A1A",   # texto principal
    "text_muted":    "#666666",   # texto secundario, rótulos
    "text_faint":    "#999999",   # texto terciario: versión, separadores, notas
    "text_topbar":   "#FFFFFF",   # texto sobre la barra superior
    "text_accent":   "#2166AC",
    "on_accent":     "#FFFFFF",   # texto e íconos sobre fondos de acento

    # Acento
    "accent":        "#2166AC",   # azul principal
    "accent_hover":  "#3A7CC4",   # botón de la barra superior con el mouse encima
    "accent_dark":   "#1A5490",   # botón presionado, recuadro de zoom
    "accent_darker": "#0D3F75",   # herramienta activa en la barra superior, rótulos sobre la foto
    "accent_faint":  "#7FA6CF",   # ícono de opciones cerrado
    "accent_light":  "#D6E8F7",   # fondo suave de acento
    "accent2":       "#006837",   # verde (botones secundarios)
    "topbar_divider": "#4A8FD4",  # separadores de la barra superior

    # Bordes y líneas
    "border":        "#CCCCCC",
    "border_focus":  "#2166AC",
    "divider":       "#E3E8EE",   # líneas finas que separan secciones
    "grid":          "#EEEEEE",   # cuadrícula de los gráficos

    # Botones
    "btn_bg":        "#DDEEFF",
    "btn_hover":     "#B8D4EE",
    "btn_accent_bg": "#2166AC",
    "btn_accent_fg": "#FFFFFF",

    # Estados
    "warning":       "#E67E00",
    "warning_text":  "#B35806",   # avisos escritos
    "error":         "#C0392B",
    "success":       "#006837",

    # Deslizadores
    "slider_trough": "#DDDDDD",
    "slider_active": "#2166AC",

    # Marcas sobre la foto
    "white":         "#FFFFFF",   # bordes de regiones, halos, puntos medios
    "black":         "#000000",   # oscurecido del recorrido guiado, borde del minimapa
    "highlight":     "#D73027",   # anillo del punto elegido en un gráfico
    "selection":     "#FFD400",   # vértices de un área elegida
    "area_include":  "#0B2E59",   # área de análisis
    "area_exclude":  "#E67E00",   # zona excluida
    "delete":        "#D32F2F",   # × para borrar un área, con el mouse encima
    "point_positive": "#2E9E44",  # clic sobre un objeto (MobileSAM)
    "point_negative": "#D62728",  # clic sobre el fondo (MobileSAM)
    "scale_line":    "#0078DC",   # línea entre los dos puntos de la escala
    "scale_point_1": "#0050DC",
    "scale_point_2": "#DCA000",
    "scale_label":   "#0050A0",
}

_SANS, _SERIF, _MONO = "Segoe UI", "Georgia", "Consolas"

FONTS = {
    "tiny":          (_SANS, 7),
    "tiny_bold":     (_SANS, 7, "bold"),
    "small":         (_SANS, 8),
    "small_bold":    (_SANS, 8, "bold"),
    "small_italic":  (_SANS, 8, "italic"),
    "body":          (_SANS, 9),
    "body_bold":     (_SANS, 9, "bold"),
    "large":         (_SANS, 10),
    "topbar":        (_SANS, 10, "bold"),
    "title":         (_SANS, 11, "bold"),
    "subtitle":      (_SANS, 11),
    "heading":       (_SANS, 12, "bold"),
    "heading_large": (_SANS, 13, "bold"),
    "logo":          (_SANS, 15, "bold"),
    "display":       (_SANS, 22, "bold"),
    "display_large": (_SANS, 26, "bold"),
    "display_xl":    (_SANS, 28, "bold"),
    "splash_title":  (_SANS, 32, "bold"),
    "splash_sub":    (_SANS, 10),
    "motto":         (_SERIF, 11, "italic"),
    "motto_large":   (_SERIF, 12, "italic"),
    "motto_xl":      (_SERIF, 14, "italic"),
    "mono":          (_MONO, 8),
}

CLASS_COLORS = ("#9E9E9E", "#2166AC", "#2E9E44", "#D62728", "#F2B705", "#8E44AD", "#E67E22", "#17BECF")

CHART_COLORS = ("#2166AC", "#D73027", "#1A9850", "#8073AC", "#E08214", "#01665E",
                "#762A83", "#4DAC26", "#F4A582", "#92C5DE", "#A6D96A", "#FDAE61")

LOGO_TINTS = ("#2166AC", "#3B7DC0", "#6A9FD3", "#A9C8E8", "#D6E5F4")
LOGO_EDGES = ("#5A93CC", "#7DA8D8", "#9DBEE3", "#B9D1EC", "#CFE0F2")


def font(size: int, weight: str | None = None) -> tuple:
    """Tipo de letra de la interfaz en un tamaño calculado (p. ej. rótulos que cambian
    de tamaño con el zoom)."""
    return (_SANS, int(size), weight) if weight else (_SANS, int(size))
