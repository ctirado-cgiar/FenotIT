"""Fig. 1: de la foto a los datos con una foto real de semillas de frijol del ensayo de sequía
(muchas semillas en contacto y tarjeta de color) y los resultados del pipeline."""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fenotit.core.analysis import objects
from fenotit.core.corrections import pipeline as corrections_pipeline
from fenotit.core.image_io import load_image
from fenotit.core.pipeline import overlay

OUT = Path(__file__).parent / "out"
PHOTO = ROOT / "tests" / "data" / "Cgiar_Bean_Sequia_Seeds" / "In_row=23379.jpg"
TRAY = (0.03, 0.97, 0.12, 0.86)            # recorte de la bandeja: y0, y1, x0, x1 (fracciones)
CARD = [[0.62, 0.0], [0.84, 0.25]]         # zona excluida: tarjeta de color (en la bandeja)
ZOOM = (0.34, 0.72, 0.02, 0.37)            # detalle con semillas en contacto: y0, y1, x0, x1
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7,
                     "axes.titlelocation": "left", "axes.titlepad": 3})


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
    colors = overlay.resolve(None, auto=result.extra.get("contrast"))
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
    seed_colors = [(row["mean_R"] / 255, row["mean_G"] / 255, row["mean_B"] / 255) for row in rows]

    fig = plt.figure(figsize=(7.2, 1.95), dpi=300)
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.1], wspace=0.1,
                            left=0.005, right=0.985, top=0.9, bottom=0.14)
    panels = [("(a) Photograph", rgb(raw_tray)), ("(b) Color corrected", rgb(tray)),
              ("(c) Segmentation", rgb(segmented)), ("(d) Separated and measured", rgb(measured))]
    for column, (title, image) in enumerate(panels):
        ax = fig.add_subplot(grid[column])
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(title)
        if column == 1:                                   # lugar del detalle de (c) y (d)
            image_height, image_width = image.shape[:2]
            y0, y1, x0, x1 = ZOOM
            ax.add_patch(plt.Rectangle((x0 * image_width, y0 * image_height), (x1 - x0) * image_width,
                                       (y1 - y0) * image_height,
                                       fill=False, ec="white", lw=0.8))
    ax = fig.add_subplot(grid[4])
    ax.scatter(length, width, s=4, c=seed_colors, linewidths=0.2, edgecolors="#555555")
    ax.set_xlabel("Length (px)", labelpad=1)
    ax.set_ylabel("Width (px)", labelpad=1)
    ax.set_title(f"(e) {len(rows)} measured seeds")
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.set_box_aspect(1.0)
    box = ax.get_position()
    ax.set_position([box.x0 + 0.035, box.y0, box.width - 0.035, box.height])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig1_workflow.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig1_workflow.svg", bbox_inches="tight", pad_inches=0.02)
    stats = result.stats
    print(OUT / "fig1_workflow.png", {key: stats.get(key) for key in ("n_objects", "n_touching", "n_measured")})


if __name__ == "__main__":
    main()
