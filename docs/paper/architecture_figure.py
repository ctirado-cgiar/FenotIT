"""Figura A.1: arquitectura de FenotIT con íconos (usuario, archivos, bases de datos, módulos).

Escribe docs/paper/out/architecture.svg y, con el Chromium de Playwright, architecture.png a
190 mm de ancho y 600 dpi.

    python docs/paper/architecture_figure.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).with_name("out")
WIDTH, HEIGHT = 1900, 1260
DPI = 600
WIDTH_MM = 190

BLUE = "#2166AC"
BLUE_FILL = "#DCE8F5"
BLUE_HEADER = "#9EC1E6"
GRAY = "#6E6E6E"
GRAY_FILL = "#E8E8E8"
DARK = "#222222"
FONT = "Liberation Sans, Arial, sans-serif"

NAME_SIZE = 25
NOTE_SIZE = 21
LABEL_SIZE = 21
TITLE_SIZE = 26

LEFT_X = 165          # centro de la columna izquierda
RIGHT_X = 1745        # centro de la columna derecha
INTERFACE = (330, 20, 1570, 232)     # x0, y0, x1, y1 de cada marco
CORE = (330, 300, 1570, 965)

parts: list[str] = []


def text(x, y, content, size=NOTE_SIZE, weight="normal", anchor="middle", color=DARK, italic=False):
    style = ' font-style="italic"' if italic else ""
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                 f'text-anchor="{anchor}" fill="{color}"{style}>{content}</text>')


def lines(x, y, rows, size=NOTE_SIZE, gap=None, **kwargs):
    gap = gap or size * 1.2
    for index, row in enumerate(rows):
        text(x, y + index * gap, row, size=size, **kwargs)


def group(box, title):
    x0, y0, x1, y1 = box
    parts.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" '
                 f'stroke="{GRAY}" stroke-width="2" stroke-dasharray="12 7"/>')
    text(x0 + 16, y0 + 32, title, size=NAME_SIZE, anchor="start", color=GRAY)


def module(x, y, width, height, name, notes=(), fill=BLUE_FILL, stroke=BLUE, name_size=NAME_SIZE, weight="bold", note_size=NOTE_SIZE):
    """Caja con barras laterales, el símbolo clásico de un módulo de software."""
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="2"/>')
    for side in (x + 14, x + width - 14):
        parts.append(f'<line x1="{side}" y1="{y}" x2="{side}" y2="{y + height}" stroke="{stroke}" stroke-width="2"/>')
    middle = x + width / 2
    first = y + height / 2 - (len(notes) * note_size * 1.2) / 2 + name_size * 0.35
    text(middle, first, name, size=name_size, weight=weight)
    lines(middle, first + name_size * 1.15, notes, size=note_size)


def stacked(x, y, width, height, name, notes=()):
    for shift in (16, 8):
        parts.append(f'<rect x="{x + shift}" y="{y - shift}" width="{width}" height="{height}" '
                     f'fill="{BLUE_FILL}" stroke="{BLUE}" stroke-width="2"/>')
    module(x, y, width, height, name, notes)


def window(x, y, width, height, name, notes=()):
    """Caja con barra de título, para las partes de la interfaz."""
    bar = 26
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="white" '
                 f'stroke="{BLUE}" stroke-width="2"/>')
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{bar}" fill="{BLUE_HEADER}" '
                 f'stroke="{BLUE}" stroke-width="2"/>')
    for index in range(3):
        cx = x + width - 22 - index * 18
        parts.append(f'<rect x="{cx - 6}" y="{y + 7}" width="12" height="12" fill="white" stroke="{BLUE}" stroke-width="1.5"/>')
    middle = x + width / 2
    text(middle, y + bar + 36, name, size=NAME_SIZE, weight="bold")
    lines(middle, y + bar + 36 + NAME_SIZE * 1.15, notes)


def cylinder(cx, cy, width, height, name_rows, note_rows=(), fill=GRAY_FILL, stroke=GRAY, stack=1):
    rx, ry = width / 2, 14
    for layer in range(stack - 1, -1, -1):
        x = cx - rx + layer * 10
        top = cy - height / 2 - layer * 10
        bottom = top + height
        parts.append(f'<path d="M{x},{top} v{height} a{rx},{ry} 0 0 0 {width},0 v{-height}" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<ellipse cx="{x + rx}" cy="{top}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    lines(cx, cy - (len(name_rows) - 1) * NAME_SIZE * 0.6 + 12, name_rows, size=NAME_SIZE, weight="bold")
    lines(cx, cy + height / 2 + ry + 30, note_rows)


def person(cx, cy, label):
    parts.append(f'<circle cx="{cx}" cy="{cy - 38}" r="24" fill="{GRAY_FILL}" stroke="{GRAY}" stroke-width="2.5"/>')
    parts.append(f'<path d="M{cx - 46},{cy + 46} v-14 a46,40 0 0 1 92,0 v14 z" fill="{GRAY_FILL}" '
                 f'stroke="{GRAY}" stroke-width="2.5"/>')
    text(cx, cy + 82, label, size=NAME_SIZE, weight="bold")


def photos(cx, cy):
    for shift in (24, 12, 0):
        x, y = cx - 58 + shift, cy - 48 - shift
        parts.append(f'<rect x="{x}" y="{y}" width="92" height="70" fill="white" stroke="{GRAY}" stroke-width="2.5"/>')
    x, y = cx - 58, cy - 48
    parts.append(f'<path d="M{x + 8},{y + 62} l24,-30 l16,18 l12,-12 l24,24 z" fill="{GRAY_FILL}" stroke="{GRAY}" stroke-width="2"/>')
    parts.append(f'<circle cx="{x + 70}" cy="{y + 18}" r="7" fill="{GRAY_FILL}" stroke="{GRAY}" stroke-width="2"/>')


def document(cx, cy, tag=None):
    x, y, width, height, fold = cx - 38, cy - 50, 76, 96, 20
    parts.append(f'<path d="M{x},{y} h{width - fold} l{fold},{fold} v{height - fold} h{-width} z" fill="white" '
                 f'stroke="{GRAY}" stroke-width="2.5"/>')
    parts.append(f'<path d="M{x + width - fold},{y} v{fold} h{fold}" fill="none" stroke="{GRAY}" stroke-width="2"/>')
    if tag:
        text(cx, cy + 16, tag, size=NAME_SIZE, weight="bold", color=GRAY)
    else:
        for row in range(4):
            parts.append(f'<line x1="{x + 12}" y1="{y + 34 + row * 14}" x2="{x + width - 12}" '
                         f'y2="{y + 34 + row * 14}" stroke="{GRAY}" stroke-width="2"/>')


def arrow(points, label=None, at=None, anchor="middle", both=False, dashed=False, width=2.2, main=False):
    """Flecha; main = camino principal de la foto (azul y más gruesa)."""
    path = " ".join(f"{x},{y}" for x, y in points)
    head = "main" if main else "head"
    marks = f' marker-end="url(#{head})"' + (' marker-start="url(#tail)"' if both else "")
    dash = ' stroke-dasharray="10 7"' if dashed else ""
    color = BLUE if main else DARK
    parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="{width}"{dash}{marks}/>')
    if label:
        rows = label if isinstance(label, (list, tuple)) else [label]
        lines(at[0], at[1], rows, size=LABEL_SIZE, anchor=anchor)


def draw():
    # Usuario e interfaz
    person(LEFT_X, 150, "User")
    group(INTERFACE, "Interface (Tkinter)")
    for column, (name, notes) in zip((360, 660, 960, 1260), (
            ("Main window", ["photo list, views,", "analysis panel"]),
            ("Dialogs", ["corrections, models,", "presets, export"]),
            ("Image views", ["zoom, areas,", "strokes, clicks"]),
            ("Charts", ["boxes, histograms,", "scatter, color"]))):
        window(column, 80, 270, 128, name, notes)
    arrow([(215, 145), (358, 145)], label="uses", at=(287, 130), both=True)
    cylinder(RIGHT_X, 140, 190, 92, ["Languages"], ["one JSON file", "per language"])
    arrow([(1648, 145), (1572, 145)], label="texts", at=(1610, 130))

    # Núcleo: la cadena de análisis en una fila
    group(CORE, "Analysis core (no interface code)")
    arrow([(950, 232), (950, 298)], width=3)
    text(968, 272, "runs analyses", size=LABEL_SIZE, anchor="start")
    text(360, 372, "Analysis steps, run in order on each photograph", size=NOTE_SIZE, anchor="start", italic=True)
    chain = [("Corrections", ["lens, perspective,", "color, scale"]),
             ("Segmentation", ["threshold,", "by example,", "MobileSAM"]),
             ("Detection", ["boxes from", "trained models"]),
             ("Objects", ["separation,", "filters"]),
             ("Measurements", ["size, shape,", "color, classes"]),
             ("Export", ["tables, workbook,", "metadata"])]
    box_width, gap, top, height = 180, 22, 392, 120
    for index, (name, notes) in enumerate(chain):
        x = 360 + index * (box_width + gap)
        module(x, top, box_width, height, name, notes, name_size=21, note_size=18)
        if index:
            arrow([(x - gap, top + height / 2), (x - 1, top + height / 2)], width=3.5, main=True)
    chain_end = 360 + len(chain) * (box_width + gap) - gap
    photos(LEFT_X, 470)
    lines(LEFT_X, 545, ["Photographs"], size=NAME_SIZE, weight="bold")
    arrow([(LEFT_X + 70, 452), (358, 452)], width=3.5, main=True)
    text(290, 438, "reads", size=LABEL_SIZE)
    cylinder(RIGHT_X, 452, 190, 92, ["Results"], ["tables, images,", "metadata"], stack=3)
    arrow([(chain_end + 2, 452), (1645, 452)], width=3.5, main=True)
    text(1612, 438, "writes", size=LABEL_SIZE)

    # Núcleo: lo que acompaña a la cadena
    module(360, 590, 280, 100, "Project", ["photos, areas, scale,", "parameters"])
    module(360, 720, 280, 100, "Saved settings", ["presets shared", "between laboratories"])
    cylinder(LEFT_X, 705, 200, 92, ["Project and", "preset files"], ["readable text files"])
    arrow([(265, 690), (358, 640)], both=True, label="opens, saves", at=(215, 618))
    arrow([(265, 720), (358, 770)], both=True)
    arrow([(450, 588), (450, 514)])
    text(466, 560, "settings of each step", size=LABEL_SIZE, anchor="start")
    arrow([(500, 718), (500, 692)])
    text(516, 712, "loaded into", size=LABEL_SIZE, anchor="start")
    stacked(700, 600, 300, 110, "Batch processing", ["several photos at once"])
    arrow([(850, 584), (850, 514)])
    text(866, 560, "runs these steps on each photo", size=LABEL_SIZE, anchor="start")
    module(1100, 600, 300, 110, "Statistics and charts", ["by photo or group"])
    arrow([(1258, 514), (1258, 598)])
    text(1274, 560, "results", size=LABEL_SIZE, anchor="start")
    cylinder(RIGHT_X, 655, 190, 92, ["Field book"], ["genotype or ID", "of each photo"])
    arrow([(1648, 655), (1402, 655)], label="adds", at=(1610, 640))

    # Bibliotecas
    module(360, 880, 560, 62, "OpenCV, NumPy, SciPy, scikit-image, scikit-learn", fill=GRAY_FILL, stroke=GRAY,
           name_size=NOTE_SIZE, weight="normal")
    module(960, 880, 580, 62, "ONNX Runtime (runs trained models)", fill=GRAY_FILL, stroke=GRAY,
           name_size=NOTE_SIZE, weight="normal")
    text(360, 868, "Libraries used by every step", size=NOTE_SIZE, anchor="start", italic=True)

    # Modelos entrenados y su conversión
    cylinder(RIGHT_X, 910, 190, 92, ["Trained", "models"], [])
    arrow([(1648, 910), (1542, 910)], label="loads", at=(1603, 895))
    module(1600, 1095, 290, 110, "Converter", ["separate Python", "environment, to ONNX"])
    arrow([(RIGHT_X, 1093), (RIGHT_X, 975)], label="adds", at=(RIGHT_X + 40, 1040), anchor="start")
    document(1440, 1150, tag=".pt")
    text(1385, 1160, "PyTorch model", size=NOTE_SIZE, anchor="end")
    document(1440, 1030, tag=".onnx")
    text(1385, 1040, "ONNX model", size=NOTE_SIZE, anchor="end")
    arrow([(1482, 1015), (1655, 960)])
    text(1590, 1018, "adds", size=LABEL_SIZE)
    arrow([(1482, 1150), (1598, 1150)])


def svg() -> str:
    defs = (f'<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{DARK}"/></marker>'
            f'<marker id="tail" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M10,0 L0,5 L10,10 z" fill="{DARK}"/></marker>'
            f'<marker id="main" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/></marker></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'font-family="{FONT}">{defs}<rect width="100%" height="100%" fill="white"/>{"".join(parts)}</svg>')


def render(svg_path: Path, png_path: Path):
    from playwright.sync_api import sync_playwright
    scale = WIDTH_MM / 25.4 * DPI / WIDTH
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=scale)
        page.set_content(f'<html><body style="margin:0">{svg_path.read_text(encoding="utf-8")}</body></html>')
        page.screenshot(path=str(png_path), clip={"x": 0, "y": 0, "width": WIDTH, "height": HEIGHT})
        browser.close()


def main():
    OUT.mkdir(exist_ok=True)
    draw()
    svg_path = OUT / "architecture.svg"
    svg_path.write_text(svg(), encoding="utf-8")
    render(svg_path, OUT / "architecture.png")
    print(OUT / "architecture.png")


if __name__ == "__main__":
    main()
