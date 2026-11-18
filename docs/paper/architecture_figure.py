"""Figura A.1: arquitectura de FenotIT con íconos (usuario, archivos, bases de datos, módulos).

Escribe docs/paper/out/architecture.svg y, con el Chromium de Playwright, architecture.png a
190 mm de ancho y 600 dpi.

    python docs/paper/architecture_figure.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).with_name("out")
WIDTH, HEIGHT = 1900, 1270
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
APP = (340, 40, 1530, 1010)          # x0, y0, x1, y1 de la aplicación
INTERFACE = (370, 100, 1500, 300)
CORE = (370, 350, 1500, 980)
CONVERTER = (340, 1070, 1530, 1240)

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


def module(x, y, width, height, name, notes=(), fill=BLUE_FILL, stroke=BLUE, name_size=NAME_SIZE, weight="bold"):
    """Caja con barras laterales, el símbolo clásico de un módulo de software."""
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="2"/>')
    for side in (x + 14, x + width - 14):
        parts.append(f'<line x1="{side}" y1="{y}" x2="{side}" y2="{y + height}" stroke="{stroke}" stroke-width="2"/>')
    middle = x + width / 2
    first = y + height / 2 - (len(notes) * NOTE_SIZE * 1.2) / 2 + name_size * 0.35
    text(middle, first, name, size=name_size, weight=weight)
    lines(middle, first + NAME_SIZE * 1.15, notes)


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


def arrow(points, label=None, at=None, anchor="middle", both=False, dashed=False, width=2.2):
    path = " ".join(f"{x},{y}" for x, y in points)
    marks = ' marker-end="url(#head)"' + (' marker-start="url(#tail)"' if both else "")
    dash = ' stroke-dasharray="10 7"' if dashed else ""
    parts.append(f'<polyline points="{path}" fill="none" stroke="{DARK}" stroke-width="{width}"{dash}{marks}/>')
    if label:
        rows = label if isinstance(label, (list, tuple)) else [label]
        lines(at[0], at[1], rows, size=LABEL_SIZE, anchor=anchor)


def draw():
    # Columna izquierda: usuario y lo que entra
    person(LEFT_X, 165, "User")
    cylinder(LEFT_X, 470, 200, 92, ["Project and", "preset files"], ["YAML (.fnt, .fntp)"])
    photos(LEFT_X, 715)
    lines(LEFT_X, 790, ["Photographs"], size=NAME_SIZE, weight="bold")
    document(LEFT_X, 1150, tag=".pt")
    lines(LEFT_X, 1232, ["Trained model"], size=NAME_SIZE, weight="bold")

    # Aplicación
    group(APP, "FenotIT desktop application (Python 3.12)")
    group(INTERFACE, "Interface layer (Tkinter)")
    columns = [395, 670, 945, 1220]
    box_width = 255
    window(columns[0], 150, box_width, 132, "Main window", ["photo list, views,", "step panels"])
    window(columns[1], 150, box_width, 132, "Dialogs", ["corrections, models,", "presets, export"])
    window(columns[2], 150, box_width, 132, "Image canvases", ["zoom, areas,", "strokes, clicks"])
    window(columns[3], 150, box_width, 132, "Charts", ["boxes, histograms,", "scatter, color 3D"])

    group(CORE, "Analysis core (no interface code)")
    arrow([(935, 300), (935, 352)], width=4)
    text(955, 333, "calls", size=LABEL_SIZE, anchor="start")

    # Núcleo: columna izquierda
    module(395, 400, 255, 110, "Project", ["areas, scale, strokes,", "parameters"])
    module(395, 545, 255, 100, "Presets", ["capture setup"])
    stacked(395, 700, 225, 110, "Batch runner", ["worker processes"])

    # Motor con la cadena de pasos
    engine = (710, 400, 1140, 880)
    parts.append(f'<rect x="{engine[0]}" y="{engine[1]}" width="{engine[2] - engine[0]}" '
                 f'height="{engine[3] - engine[1]}" fill="white" stroke="{BLUE}" stroke-width="2.5"/>')
    text((engine[0] + engine[2]) / 2, engine[1] + 34, "Pipeline engine", size=NAME_SIZE, weight="bold")
    text((engine[0] + engine[2]) / 2, engine[1] + 62, "registered steps, shared context, cache")
    steps = [("Corrections", "lens, perspective, color, scale"),
             ("Segmentation", "threshold, by example, MobileSAM"),
             ("Detection", "boxes of trained models"),
             ("Objects", "separation, filters"),
             ("Measurements", "count, size, shape, color, classes")]
    top, height, gap = 485, 62, 15
    for index, (name, note) in enumerate(steps):
        y = top + index * (height + gap)
        parts.append(f'<rect x="{engine[0] + 25}" y="{y}" width="{engine[2] - engine[0] - 50}" height="{height}" '
                     f'fill="{BLUE_FILL}" stroke="{BLUE}" stroke-width="2"/>')
        text((engine[0] + engine[2]) / 2, y + 26, name, size=NAME_SIZE, weight="bold")
        text((engine[0] + engine[2]) / 2, y + 52, note)
        if index:
            arrow([((engine[0] + engine[2]) / 2, y - gap - 1), ((engine[0] + engine[2]) / 2, y - 1)], width=2)

    # Núcleo: columna derecha
    module(1230, 400, 255, 110, "Statistics", ["groups, summaries,", "mean shapes"])
    module(1230, 590, 255, 110, "Export", ["tables, workbook,", "metadata, views"])

    # Bibliotecas
    module(395, 905, 520, 62, "OpenCV, NumPy, SciPy, scikit-image, scikit-learn", fill=GRAY_FILL, stroke=GRAY,
           name_size=NOTE_SIZE, weight="normal")
    module(965, 905, 520, 62, "ONNX Runtime", fill=GRAY_FILL, stroke=GRAY, name_size=NOTE_SIZE, weight="normal")

    # Relaciones dentro del núcleo
    arrow([(636, 755), (708, 755)], label=["one", "photo"], at=(672, 716))
    arrow([(1140, 455), (1228, 455)], label="results", at=(1184, 440))
    arrow([(1140, 645), (1228, 645)])
    arrow([(810, 880), (810, 903)])
    arrow([(1080, 880), (1080, 903)])

    # Columna derecha: almacenamiento
    cylinder(RIGHT_X, 200, 190, 92, ["Languages"], ["JSON, one file", "per language"])
    cylinder(RIGHT_X, 470, 190, 92, ["Field book"], ["CSV, XLSX"])
    cylinder(RIGHT_X, 680, 190, 92, ["Results"], ["CSV, XLSX, PNG,", "metadata"], stack=3)
    cylinder(RIGHT_X, 935, 190, 92, ["Model", "library"])
    text(RIGHT_X, 1032, "ONNX models")

    # Conexiones con el exterior
    arrow([(225, 165), (393, 165)], label=["edits,", "views"], at=(309, 120), both=True)
    arrow([(265, 470), (393, 455)], both=True, label="open, save", at=(318, 440))
    arrow([(265, 500), (393, 590)], both=True)
    arrow([(LEFT_X + 70, 700), (392, 740)], label="read", at=(318, 704))
    arrow([(1648, 200), (1502, 200)], label="texts", at=(1575, 186))
    arrow([(1648, 470), (1488, 460)], label="joined", at=(1568, 450))
    arrow([(1650, 500), (1488, 610)])
    arrow([(1487, 645), (1640, 668)], label="write", at=(1565, 642))
    arrow([(1648, 935), (1487, 935)], label="load", at=(1568, 920))

    # Conversión de modelos
    group(CONVERTER, "Model converter (separate Python environment, installed on request)")
    module(560, 1135, 420, 80, "PyTorch with the model library", ["Ultralytics or RF-DETR"])
    module(1060, 1135, 380, 80, "ONNX export", ["opset 17, output layout read"])
    arrow([(LEFT_X + 45, 1150), (558, 1175)])
    arrow([(980, 1175), (1058, 1175)])
    arrow([(1440, 1175), (1870, 1175), (1870, 935), (1842, 935)])


def svg() -> str:
    defs = (f'<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{DARK}"/></marker>'
            f'<marker id="tail" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M10,0 L0,5 L10,10 z" fill="{DARK}"/></marker></defs>')
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
