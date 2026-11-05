"""Convierte los íconos de fenotit/assets/icons/svg/<nombre>.svg en
fenotit/assets/icons/<nombre>.png (128 × 128 px), que es lo que lee la aplicación.

Uso:  python docs/design/icons_svg_to_png.py

Necesita cairosvg (pip install cairosvg) o Inkscape en el PATH."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "fenotit" / "assets" / "icons" / "svg"
DST = SRC.parent
SIZE = 128


def convert(svg: Path, png: Path):
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=SIZE, output_height=SIZE)
        return
    except (ImportError, OSError):
        pass
    exe = shutil.which("inkscape")
    if not exe:
        sys.exit("Falta cairosvg o Inkscape.")
    subprocess.run([exe, str(svg), "--export-type=png", f"--export-filename={png}",
                    f"--export-width={SIZE}", f"--export-height={SIZE}"], check=True)


def main():
    sys.path.insert(0, str(ROOT))
    from fenotit.gui.icons import NAMES
    for svg in sorted(SRC.glob("*.svg")):
        if svg.stem not in NAMES:
            print(f"  {svg.name}: no es un nombre de ícono de la app (ver docs/design/README.md)")
            continue
        convert(svg, DST / f"{svg.stem}.png")
        print(f"  {svg.stem}.png")


if __name__ == "__main__":
    main()
