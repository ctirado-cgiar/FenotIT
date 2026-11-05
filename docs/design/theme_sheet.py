"""Hoja con los colores y tipos de letra del tema (fenotit/gui/theme.py):
docs/design/theme.png.

Uso:  python docs/design/theme_sheet.py"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fenotit.gui.theme import CHART_COLORS, CLASS_COLORS, COLORS, FONTS  # noqa: E402


def main():
    root = tk.Tk()
    root.title("Tema")
    root.configure(bg=COLORS["bg_card"])
    cols = tk.Frame(root, bg=COLORS["bg_card"])
    cols.pack(padx=16, pady=16)
    box = tk.Frame(cols, bg=COLORS["bg_card"])
    box.grid(row=0, column=0, sticky="n", padx=(0, 24))
    tk.Label(box, text="COLORS", bg=COLORS["bg_card"], fg=COLORS["text"], font=FONTS["title"]).pack(anchor="w")
    for name, hexa in COLORS.items():
        row = tk.Frame(box, bg=COLORS["bg_card"])
        row.pack(anchor="w", pady=1)
        tk.Frame(row, bg=hexa, width=36, height=16, highlightthickness=1,
                 highlightbackground=COLORS["border"]).pack(side=tk.LEFT)
        tk.Label(row, text=f"  {name}  {hexa}", bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["mono"]).pack(side=tk.LEFT)
    box = tk.Frame(cols, bg=COLORS["bg_card"])
    box.grid(row=0, column=1, sticky="n")
    tk.Label(box, text="FONTS", bg=COLORS["bg_card"], fg=COLORS["text"], font=FONTS["title"]).pack(anchor="w")
    for name, f in FONTS.items():
        tk.Label(box, text=f"{name}  {f}", bg=COLORS["bg_card"], fg=COLORS["text"], font=f).pack(anchor="w")
    for title, pal in (("CLASS_COLORS", CLASS_COLORS), ("CHART_COLORS", CHART_COLORS)):
        tk.Label(box, text=title, bg=COLORS["bg_card"], fg=COLORS["text"], font=FONTS["title"]).pack(anchor="w", pady=(12, 2))
        row = tk.Frame(box, bg=COLORS["bg_card"])
        row.pack(anchor="w")
        for c in pal:
            tk.Frame(row, bg=c, width=22, height=22).pack(side=tk.LEFT, padx=1)
    root.update()
    out = Path(__file__).with_name("theme.png")
    try:
        from PIL import ImageGrab
        x, y = root.winfo_rootx(), root.winfo_rooty()
        import os
        disp = os.environ.get("DISPLAY") if sys.platform.startswith("linux") else None
        ImageGrab.grab((x, y, x + root.winfo_width(), y + root.winfo_height()), xdisplay=disp).save(out)
        print(out)
    except Exception as e:
        print(e)
        root.mainloop()
        return
    root.destroy()


if __name__ == "__main__":
    main()
