"""Fig. 2: ventana principal con una foto de semillas de frijol del ensayo de sequía y las
zonas de la interfaz marcadas con letras. Recibe la captura de la ventana como argumento."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figure_style as style  # noqa: E402

OUT = Path(__file__).parent / "out"
# Letra y su lugar (x, y en píxeles de la captura): lista de fotos y capas, imágenes de entrada
# y de resultado, panel del análisis, tabla de resultados e inspector del objeto elegido
LABELS = {"(a)": (150, 520), "(b)": (232, 76), "(c)": (1500, 480), "(d)": (1250, 708), "(e)": (112, 622)}


def main(capture: str):
    style.apply()
    screen = Image.open(capture).convert("RGB")
    width, height = screen.size
    fig = plt.figure(figsize=(style.DOUBLE, style.DOUBLE * height / width))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(screen)
    style.photo_axes(ax)
    for text, (x, y) in LABELS.items():
        ax.text(x, y, text, ha="left", va="top", fontsize=style.FONT_SIZE + 1, fontweight="bold",
                color=style.WHITE, bbox={"boxstyle": "square,pad=0.25", "facecolor": style.BLACK,
                                          "edgecolor": "none"})
    style.save(fig, OUT / "fig2_interface.png", dpi=style.PHOTO_DPI)


if __name__ == "__main__":
    main(sys.argv[1])
