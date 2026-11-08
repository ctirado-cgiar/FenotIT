"""Estilos de los controles ttk (botones, tablas, listas desplegables, barras) según el tema."""
import tkinter as tk
from tkinter import ttk

from fenotit.gui.style.theme import COLORS, FONTS, SIZES, use_installed_fonts


def apply(root: tk.Misc) -> None:
    """Aplica el tema a la ventana `root`: familias de letra instaladas, opciones de los
    controles tk y estilos ttk."""
    use_installed_fonts(root)
    for cls in ("Checkbutton", "Radiobutton", "Button"):
        root.option_add(f"*{cls}.highlightThickness", SIZES["focus_ring"])
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure("TFrame", background=COLORS["bg"])
    s.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=FONTS["body"])
    s.configure("TButton", background=COLORS["btn_bg"], foreground=COLORS["accent"], font=FONTS["body"],
                borderwidth=1, relief="flat", padding=(SIZES["button_pad_x"], SIZES["button_pad_y"]))
    s.map("TButton", background=[("active", COLORS["btn_hover"])], foreground=[("active", COLORS["accent"])])
    s.configure("TSeparator", background=COLORS["border"])
    for name in ("Horizontal.TProgressbar", "Thin.Horizontal.TProgressbar"):
        s.configure(name, background=COLORS["accent"], troughcolor=COLORS["bg_panel"],
                    bordercolor=COLORS["border"], lightcolor=COLORS["accent"], darkcolor=COLORS["accent"])
    s.configure("Thin.Horizontal.TProgressbar", thickness=SIZES["progress_thin"])
    s.configure("TScrollbar", background=COLORS["bg_panel"], troughcolor=COLORS["bg"], borderwidth=0,
                arrowsize=SIZES["scroll_arrow"])
    s.configure("TCombobox", fieldbackground=COLORS["bg_card"], background=COLORS["bg_card"],
                foreground=COLORS["text"], selectbackground=COLORS["accent_light"],
                selectforeground=COLORS["text"])
    s.configure("Treeview", background=COLORS["bg_card"], fieldbackground=COLORS["bg_card"],
                foreground=COLORS["text"], font=FONTS["small"], rowheight=SIZES["row_height"])
    s.configure("Treeview.Heading", background=COLORS["bg_panel"], foreground=COLORS["accent"],
                font=FONTS["small_bold"], relief="flat")
    s.map("Treeview", background=[("selected", COLORS["bg_table_sel"])],
          foreground=[("selected", COLORS["text"])])
