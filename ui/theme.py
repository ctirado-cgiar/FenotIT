"""
ui/theme.py
Tema visual de FenotIT — estilo científico claro, inspirado en RStudio.

Paleta:
  Fondo principal:   blanco roto / gris muy claro  (#F7F7F7 / #FFFFFF)
  Paneles laterales: gris claro                    (#ECECEC)
  Cards / entradas:  blanco                        (#FFFFFF)
  Texto primario:    casi negro                    (#1A1A1A)
  Texto secundario:  gris medio                    (#555555)
  Acento primario:   azul CGIAR / científico       (#2166AC)
  Acento secundario: verde pino                    (#006837)
  Borde:             gris suave                    (#CCCCCC)
  Advertencia:       naranja suave                 (#E67E00)
  Error:             rojo suave                    (#C0392B)
  Selección tabla:   azul muy suave                (#D6E8F7)
"""

COLORS = {
    # Fondos
    "bg":           "#F7F7F7",   # ventana principal
    "bg_panel":     "#ECECEC",   # paneles laterales
    "bg_card":      "#FFFFFF",   # áreas de trabajo, canvas, entradas
    "bg_topbar":    "#2166AC",   # barra superior (azul científico)
    "bg_table":     "#FFFFFF",
    "bg_table_sel": "#D6E8F7",

    # Texto
    "text":         "#1A1A1A",
    "text_muted":   "#666666",
    "text_topbar":  "#FFFFFF",   # texto sobre barra azul
    "text_accent":  "#2166AC",

    # Acento
    "accent":       "#2166AC",   # azul científico
    "accent2":      "#006837",   # verde pino (botones secundarios)
    "accent_light": "#D6E8F7",   # fondo suave de acento

    # Bordes
    "border":       "#CCCCCC",
    "border_focus": "#2166AC",

    # Botones
    "btn_bg":       "#DDEEFF",
    "btn_hover":    "#B8D4EE",
    "btn_accent_bg": "#2166AC",
    "btn_accent_fg": "#FFFFFF",

    # Estados
    "warning":      "#E67E00",
    "error":        "#C0392B",
    "success":      "#006837",

    # Sliders
    "slider_trough": "#DDDDDD",
    "slider_active": "#2166AC",
}

FONTS = {
    "title":   ("Segoe UI", 11, "bold"),
    "body":    ("Segoe UI", 9),
    "small":   ("Segoe UI", 8),
    "topbar":  ("Segoe UI", 10, "bold"),
    "mono":    ("Consolas", 8),
    "logo":    ("Segoe UI", 15, "bold"),
    "splash_title": ("Segoe UI", 32, "bold"),
    "splash_sub":   ("Segoe UI", 10),
}

LOGO_PATH = None   # se resuelve en runtime desde Path(__file__)
