"""Estilos de los controles ttk (botones, tablas, listas desplegables, barras) según el tema."""
import tkinter as tk
from tkinter import ttk

# Valores del tema (fenotit/assets/themes): COLORS (colores); FONTS (letras); SIZES (tamaños);
# use_installed_fonts (elige las letras instaladas).
from fenotit.gui.style.theme import COLORS, FONTS, SIZES, use_installed_fonts


def apply(root: tk.Misc) -> None:
    """Aplica el tema a la ventana `root`: familias de letra instaladas, opciones de los
    controles tk y estilos ttk."""
    use_installed_fonts(root)
    for cls in ("Checkbutton", "Radiobutton", "Button"):
        root.option_add(f"*{cls}.highlightThickness", SIZES["focus_ring"])
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=FONTS["body"])
    style.configure("TButton", background=COLORS["btn_bg"], foreground=COLORS["accent"], font=FONTS["body"],
                borderwidth=1, relief="flat", padding=(SIZES["button_small_pad_x"], SIZES["button_pad_y"]))
    style.map("TButton", background=[("active", COLORS["btn_hover"])], foreground=[("active", COLORS["accent"])])
    style.configure("TSeparator", background=COLORS["border"])
    for name in ("Horizontal.TProgressbar", "Thin.Horizontal.TProgressbar"):
        style.configure(name, background=COLORS["accent"], troughcolor=COLORS["bg_panel"],
                    bordercolor=COLORS["border"], lightcolor=COLORS["accent"], darkcolor=COLORS["accent"])
    style.configure("Thin.Horizontal.TProgressbar", thickness=SIZES["progress_thin"])
    style.configure("TScrollbar", background=COLORS["bg_panel"], troughcolor=COLORS["bg"], borderwidth=0,
                arrowsize=SIZES["scroll_arrow"])
    style.configure("TCombobox", fieldbackground=COLORS["bg_card"], background=COLORS["bg_card"],
                foreground=COLORS["text"], selectbackground=COLORS["accent_light"],
                selectforeground=COLORS["text"])
    style.configure("Treeview", background=COLORS["bg_card"], fieldbackground=COLORS["bg_card"],
                foreground=COLORS["text"], font=FONTS["small"], rowheight=SIZES["row_height"])
    style.configure("Treeview.Heading", background=COLORS["bg_panel"], foreground=COLORS["table_heading"],
                font=FONTS["small_bold"], relief="flat")
    style.map("Treeview", background=[("selected", COLORS["bg_table_sel"])],
          foreground=[("selected", COLORS["text"])])
    style.configure("TNotebook", background=COLORS["bg_card"], bordercolor=COLORS["border"])
    style.configure("TNotebook.Tab", background=COLORS["bg_panel"], foreground=COLORS["text_muted"],
                bordercolor=COLORS["border"], lightcolor=COLORS["bg_panel"], font=FONTS["body"])
    style.map("TNotebook.Tab", background=[("selected", COLORS["bg_card"])], foreground=[("selected", COLORS["text"])],
          lightcolor=[("selected", COLORS["bg_card"])])
    style.map("TCombobox", fieldbackground=[("readonly", COLORS["bg_card"])], background=[("readonly", COLORS["bg_card"])],
          selectbackground=[("readonly", COLORS["bg_card"])], selectforeground=[("readonly", COLORS["text"])],
          bordercolor=[("focus", COLORS["accent"])])
    style.configure("TCombobox", bordercolor=COLORS["border"], lightcolor=COLORS["bg_card"], darkcolor=COLORS["bg_card"],
                arrowcolor=COLORS["text_muted"])
    style.configure("Horizontal.TScale", background=COLORS["bg_card"], troughcolor=COLORS["divider"],
                bordercolor=COLORS["border"], lightcolor=COLORS["bg_card"], darkcolor=COLORS["bg_card"])
