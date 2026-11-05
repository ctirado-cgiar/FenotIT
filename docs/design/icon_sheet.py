"""Hoja con todos los íconos de la aplicación y su nombre: docs/design/icons.png.

Uso:  python docs/design/icon_sheet.py"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fenotit.gui.icons import NAMES, icon  # noqa: E402
from fenotit.gui.theme import COLORS  # noqa: E402

CELL, SIZE, COLS = 150, 48, 6


def main():
    rows = (len(NAMES) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * CELL, rows * CELL), COLORS["bg_card"])
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for i, name in enumerate(NAMES):
        x, y = (i % COLS) * CELL, (i // COLS) * CELL
        d.rounded_rectangle((x + 35, y + 20, x + 115, y + 100), 10, fill=COLORS["accent"])
        im = icon(name, SIZE, COLORS["on_accent"])
        sheet.paste(im, (x + 75 - SIZE // 2, y + 60 - SIZE // 2), im)
        w = d.textlength(name, font=font)
        d.text((x + (CELL - w) / 2, y + 112), name, fill=COLORS["text"], font=font)
    out = Path(__file__).with_name("icons.png")
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
