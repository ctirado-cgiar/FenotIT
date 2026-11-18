"""Fig. 1: de la foto a los datos con una foto real de semillas de frijol del ensayo de sequía
(muchas semillas en contacto y tarjeta de color) y los resultados del pipeline."""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))
import figure_style as style  # noqa: E402
from fenotit.core.analysis import objects
from fenotit.core.corrections import pipeline as corrections_pipeline
from fenotit.core.image_io import load_image
from fenotit.core.pipeline import overlay

OUT = Path(__file__).parent / "out"
PHOTO = ROOT / "tests" / "data" / "Cgiar_Bean_Sequia_Seeds" / "In_row=23379.jpg"
TRAY = (0.03, 0.97, 0.12, 0.86)            # recorte de la bandeja: y0, y1, x0, x1 (fracciones)
CARD = [[0.62, 0.0], [0.84, 0.25]]         # zona excluida: tarjeta de color (en la bandeja)
ZOOM = (0.34, 0.72, 0.02, 0.37)            # detalle con semillas en contacto: y0, y1, x0, x1
style.apply()


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop(image, box):
    height, width = image.shape[:2]
    y0, y1, x0, x1 = box
    return image[int(height * y0):int(height * y1), int(width * x0):int(width * x1)]


def data():
    raw = load_image(str(PHOTO))
    corrections = corrections_pipeline.Corrections()
    corrections.color.enabled = True
    fixed, _ = corrections_pipeline.apply(raw, corrections)
    raw_tray, tray = crop(raw, TRAY), crop(fixed, TRAY)
    segmentation = {"method": "threshold", "color_space": "LAB", "channel": 2, "auto": True,
                    "refine": [{"op": "fill_holes"}]}
    areas = [{"kind": "exclude", "type": "rect", "points": CARD}]
    result = objects.run(tray, {"segmentation": segmentation, "touching": True, "measure_size": True,
                                "measure_color": True, "roi_shapes": areas, "exclude_border": True})
    folders = result.extra["step_folders"]

    def view(folder):
        return next(name for name, kind in folders.items() if kind == folder)
    colors = overlay.resolve({"mask": "white", "outline": "white", "dot": "white", "mask_alpha": 0.55})
    mask = result.step_images[view("mask")][..., 0] > 0
    segmented = overlay.tint(tray, (mask * 255).astype(np.uint8), colors)
    measured = tray.copy()
    overlay.draw(measured, dict(result.extra["overlays"][view("morphometry")]), colors)
    return raw_tray, tray, crop(segmented, ZOOM), crop(measured, ZOOM), result


def main():
    raw_tray, tray, segmented, measured, result = data()
    rows = [row for row in result.extra["tables"]["objects"] if "length_px" in row]
    length = np.array([row["length_px"] for row in rows], float)
    width = np.array([row["width_px"] for row in rows], float)

    fig = plt.figure(figsize=(style.DOUBLE, 50 * style.MM))
    grid = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.32, 1.1], wspace=0.08)
    panels = [rgb(raw_tray), rgb(tray), rgb(segmented), rgb(measured)]
    for column, image in enumerate(panels):
        ax = fig.add_subplot(grid[column])
        ax.imshow(image)
        style.photo_axes(ax)
        ax.set_title(f"({'abcd'[column]})")
        if column == 1:                                   # lugar del detalle de (c) y (d)
            image_height, image_width = image.shape[:2]
            y0, y1, x0, x1 = ZOOM
            ax.add_patch(plt.Rectangle((x0 * image_width, y0 * image_height), (x1 - x0) * image_width,
                                       (y1 - y0) * image_height, fill=False, ec=style.WHITE, lw=0.8))
    ax = fig.add_subplot(grid[5])
    ax.scatter(length, width, s=6, facecolors="none", edgecolors=style.BLUE, linewidths=0.5)
    ax.set_xlabel("Length (px)")
    ax.set_ylabel("Width (px)")
    ax.set_title("(e)")
    ax.set_box_aspect(1.0)
    style.chart_axes(ax)
    style.save(fig, OUT / "fig1_workflow.png")
    stats = result.stats
    print(OUT / "fig1_workflow.png", {key: stats.get(key) for key in ("n_objects", "n_touching", "n_measured")})


if __name__ == "__main__":
    main()
