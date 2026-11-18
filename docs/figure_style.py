"""Estilo común de las figuras de la documentación y del artículo.

Usa el tema "ticks" de seaborn con letra Arial (o Liberation Sans, de las mismas medidas) de
7 pt, ejes separados del área de datos y recortados al rango de los valores. Los colores son el
azul de FenotIT, grises y negro. Los anchos son los de Elsevier: 90 mm (una columna), 140 mm
(columna y media) y 190 mm (dos columnas).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BLUE = "#2166AC"
BLACK = "#000000"
DARK_GRAY = "#4D4D4D"
GRAY = "#8C8C8C"
LIGHT_GRAY = "#C8C8C8"
WHITE = "#FFFFFF"

MM = 1 / 25.4
SINGLE = 90 * MM
ONE_AND_HALF = 140 * MM
DOUBLE = 190 * MM

PHOTO_DPI = 300           # figuras solo con fotos
CHART_DPI = 500           # figuras con gráficos (o fotos y gráficos)
FONT_SIZE = 7
FONTS = ("Arial", "Liberation Sans", "DejaVu Sans")


def font() -> str:
    """La primera letra de FONTS instalada en el equipo."""
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    return next(name for name in FONTS if name in installed)


def apply():
    """Pone el estilo en matplotlib; se llama una vez antes de dibujar."""
    sns.set_theme(context="paper", style="ticks", font=font(), rc={
        "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "lines.linewidth": 0.8,
        "axes.titlelocation": "left", "axes.titlepad": 4, "legend.frameon": False,
        "axes.edgecolor": BLACK, "axes.labelcolor": BLACK, "text.color": BLACK,
        "xtick.color": BLACK, "ytick.color": BLACK, "image.cmap": "gray",
    })


def chart_axes(*axes):
    """Ejes de un gráfico al estilo de R: separados del área de datos y recortados."""
    for ax in axes:
        sns.despine(ax=ax, offset=4, trim=True)


def plain_log_labels(*axes):
    """Números simples (0.1, 1, 10) en los ejes logarítmicos."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    for ax in axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "log":
                axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                axis.set_minor_formatter(NullFormatter())


def photo_axes(*axes):
    """Paneles con una foto: sin ejes."""
    for ax in axes:
        ax.set_axis_off()


def save(fig, path: Path, dpi: int = CHART_DPI):
    """Guarda la figura recortada a su contenido."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", path)
