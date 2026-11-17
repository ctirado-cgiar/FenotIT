"""Figures and numbers for docs/methods. Every figure is made with FenotIT's own functions on
real photos of the Alliance bean and banana programs (folders tests/data/Cgiar_*), so the
pictures show what the software computes on real material.

    python docs/make_figures.py                      (writes docs/img/*.png)
    FENOTIT_DOCS_DETECTOR=<model id> python docs/make_figures.py
                                                     (also the detection example, with a
                                                      detection model added in FenotIT)

A figure whose photos are not on this computer is skipped with a message.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fenotit.core import colorspaces, efd  # noqa: E402
from fenotit.core import roi as areas  # noqa: E402
from fenotit.core.image_io import load_image  # noqa: E402
from fenotit.core.pipeline import run  # noqa: E402

OUT = ROOT / "docs" / "img"
DATA = ROOT / "tests" / "data"
CALIBRATION = DATA / "Cgiar_Bean_Durango_Seeds" / "calibracion"
BOARD_CORNERS = (7, 6)
TRAY_PHOTO = DATA / "Cgiar_Bean_Sequia_Seeds" / "In_row=23377.jpg"
CLUMP_PHOTO = DATA / "Cgiar_Bean_Sequia_Seeds" / "In_row=23379.jpg"
# Bandeja azul sin la tarjeta de color (fracciones de la foto)
TRAY_AREA = [{"kind": "include", "type": "polygon",
              "points": [[0.07, 0.14], [0.59, 0.14], [0.59, 0.26], [0.72, 0.26], [0.72, 0.93], [0.07, 0.93]]}]
TRAY_SEG = {"step": "otsu", "params": {"color_space": "LAB", "channel": 2}}
DISH_PHOTO = DATA / "Cgiar_Bean_Pest_Seeds" / "552.jpg"
DISH_SEG = {"step": "threshold", "params": {"color_space": "LAB", "channel": 0, "min_val": 0, "max_val": 160}}
BANANA = DATA / "Cgiar_Banana_Leaf"
CARD_HEIGHT_MM = 100.0                      # alto del cuadro rojo de las fotos de banano
PEST = DATA / "Cgiar_Bean_Pest_Seeds"
SEED_PHOTOS = ("051", "052", "184", "185", "552", "553")       # semillas en la caja de Petri
INSECT_PHOTOS = ("053", "186", "554")                          # insectos y residuos sobre papel
DPI = 300
BLUE, RED, GREEN, GRAY = "#2166AC", "#D73027", "#1A9850", "#9A9A9A"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "savefig.dpi": DPI, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def panels(n, w=2.8, h=2.8):
    fig, axes = plt.subplots(1, n, figsize=(w * n, h))
    for ax in np.atleast_1d(axes):
        ax.set_axis_off()
    return fig, np.atleast_1d(axes)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def have(*paths) -> bool:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        print("skipped (photos not found):", ", ".join(missing))
    return not missing


def crop_box(labels, ids, pad=40):
    ys, xs = np.nonzero(np.isin(labels, ids))
    return slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad)


def outline(mask: np.ndarray) -> np.ndarray:
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(found, key=cv2.contourArea)[:, 0]
    return np.vstack([contour, contour[:1]])


def dish_area(img: np.ndarray, shrink: float = 0.94) -> list[dict]:
    """Analysis area: a polygon inside the Petri dish (the white region of the photo)."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(cv2.resize(img, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 140)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    biggest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    points = np.column_stack(np.nonzero(labels == biggest))[:, ::-1].astype(np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(points)
    cx, cy, radius = cx * 4, cy * 4, radius * 4 * shrink
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    return [{"kind": "include", "type": "polygon",
             "points": [[float((cx + radius * np.cos(a)) / w), float((cy + radius * np.sin(a)) / h)] for a in angles]}]


def objects_in(img, seg, shapes, extra=()):
    h, w = img.shape[:2]
    inc, exc = areas.masks(shapes, w, h)
    return run(img, [seg, {"step": "roi"}, {"step": "clean"}, {"step": "label"},
                     {"step": "filter", "params": {"exclude_border": True}}, *extra], roi=inc, exclusions=exc)


# ── 1. Corrections ──────────────────────────────────────────────────────────

def line_deviation(points: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Distance (px) from each chessboard corner to the straight line fitted to its row or column."""
    grid = points.reshape(rows, cols, 2)
    out = []
    for line in list(grid) + list(grid.transpose(1, 0, 2)):
        center = line.mean(0)
        _, _, axes = np.linalg.svd(line - center)
        out.append((line - center) @ axes[1])
    return np.concatenate(out)


def lens_evaluation():
    """Calibration with all board photos and, for each photo, the deviation of its corners from
    straight lines before and after a correction calibrated without that photo."""
    from fenotit.core.corrections import distortion
    photos = sorted(str(p) for p in CALIBRATION.glob("[0-9].jpg"))
    found = [d for d in (distortion.detect(p, *BOARD_CORNERS) for p in photos) if d.found]
    full = distortion.solve(found, *BOARD_CORNERS)
    before, after = [], []
    for index, detection in enumerate(found):
        rest = distortion.solve(found[:index] + found[index + 1:], *BOARD_CORNERS)
        size = tuple(rest.size)
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(rest.mtx, rest.dist, size, 1, size)
        corners = detection.corners.reshape(-1, 1, 2).astype(np.float64)
        fixed = cv2.undistortPoints(corners, rest.mtx, rest.dist, P=new_matrix).reshape(-1, 2)
        before.append(line_deviation(detection.corners.reshape(-1, 2), *BOARD_CORNERS))
        after.append(line_deviation(fixed, *BOARD_CORNERS))
    return full, found, before, after


def fig_lens():
    if not have(CALIBRATION):
        return
    full, found, before, after = lens_evaluation()
    rms = lambda values: float(np.sqrt(np.mean(np.square(values))))
    rms_before, rms_after = [rms(v) for v in before], [rms(v) for v in after]
    photo = load_image(found[0].path)
    corners = found[0].corners.reshape(-1, 2)
    pad = 0.6 * np.ptp(corners[:, 0]) / BOARD_CORNERS[0]
    x0, y0 = np.maximum(corners.min(0) - pad, 0).astype(int)
    x1, y1 = (corners.max(0) + pad).astype(int)
    fig = plt.figure(figsize=(8.4, 3.0))
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.35], wspace=0.25)
    ax = fig.add_subplot(grid[0])
    ax.set_axis_off()
    ax.imshow(rgb(photo[y0:y1, x0:x1]))
    ax.plot(corners[:, 0] - x0, corners[:, 1] - y0, "+", color=RED, ms=5, mew=1)
    ax.set_title(f"(a) Chessboard, {len(corners)} inner corners")
    ax = fig.add_subplot(grid[1])
    positions = np.arange(len(found))
    ax.bar(positions - 0.2, rms_before, 0.4, color=GRAY, label=f"Original photo (mean {np.mean(rms_before):.2f} px)")
    ax.bar(positions + 0.2, rms_after, 0.4, color=BLUE, label=f"Corrected photo (mean {np.mean(rms_after):.2f} px)")
    ax.set_xticks(positions, [str(i + 1) for i in positions])
    ax.set_xlabel("Chessboard photo (left out of the calibration)")
    ax.set_ylabel("Distance of the corners to a\nstraight line, RMS (px)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(b) Straightness of the chessboard rows and columns")
    save(fig, "lens.png")
    print(f"lens: {len(found)} photos, reprojection RMS {full.rms_error:.2f} px, "
          f"straightness {np.mean(rms_before):.2f} -> {np.mean(rms_after):.2f} px "
          f"(max {max(np.abs(v).max() for v in before):.2f} -> {max(np.abs(v).max() for v in after):.2f} px)")


def color_evaluation(path):
    """Color of the 24 patches as measured, after the correction fitted with all patches and
    after a correction fitted without each patch (leave-one-out), with their CIEDE2000
    difference to the reference."""
    from skimage.color import deltaE_ciede2000, rgb2lab
    from fenotit.core.corrections import colorcard
    reference = colorcard.COLORCHECKER24
    image = load_image(str(path))
    card = colorcard.detect(image)
    measured, turns = colorcard.orient(colorcard.measure(image, card), reference)
    source, target = measured.reshape(-1, 3), reference.reshape(-1, 3)
    left_out = np.empty_like(source)
    for index in range(len(source)):
        keep = np.arange(len(source)) != index
        fit = colorcard._affine(source[keep], target[keep])
        left_out[index] = np.append(source[index], 1) @ fit
    _, dropped = colorcard.matrix(measured, reference)

    def difference(colors):
        return deltaE_ciede2000(rgb2lab(np.clip(colors, 0, 1)[None]), rgb2lab(target[None]))[0]
    centers = np.rot90(card.centers, turns).reshape(-1, 2)
    return {"image": image, "card": card, "centers": centers, "measured": source, "corrected": left_out,
            "before": difference(source), "after": difference(left_out), "dropped": dropped}


def patch_grid(ax, inner, outer, values, title):
    """4 × 6 patches: the outer square is the reference color, the inner square the color shown."""
    ax.set_axis_off()
    for index in range(24):
        row, col = divmod(index, 6)
        y = -1.32 * row
        ax.add_patch(plt.Rectangle((col, y), 0.94, 0.94, color=np.clip(outer[index], 0, 1)))
        ax.add_patch(plt.Rectangle((col + 0.22, y + 0.22), 0.5, 0.5, color=np.clip(inner[index], 0, 1)))
        ax.text(col + 0.47, y - 0.06, f"{values[index]:.1f}", ha="center", va="top", fontsize=6)
    ax.set_xlim(-0.1, 6)
    ax.set_ylim(-1.32 * 3 - 0.4, 1.0)
    ax.set_aspect("equal")
    ax.set_title(title)


def fig_colorcard():
    path = CALIBRATION / "colorCard.jpg"
    if not have(path):
        return
    from fenotit.core.corrections import colorcard
    data = color_evaluation(path)
    card, centers = data["card"], data["centers"]
    pad = 1.6 * card.chip
    x0, y0 = np.maximum(centers.min(0) - pad, 0).astype(int)
    x1, y1 = (centers.max(0) + pad).astype(int)
    reference = colorcard.COLORCHECKER24.reshape(-1, 3)
    fig = plt.figure(figsize=(9.0, 5.4))
    grid = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.85], width_ratios=[0.8, 1, 1], hspace=0.35, wspace=0.15)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_axis_off()
    ax.imshow(rgb(data["image"][y0:y1, x0:x1]))
    for index, (x, y) in enumerate(centers):
        ax.text(x - x0, y - y0, str(index + 1), ha="center", va="center", fontsize=5.5, color="white",
                bbox={"boxstyle": "circle,pad=0.15", "fc": "black", "ec": "none", "alpha": 0.55})
    ax.set_title("(a) The 24 patches located")
    patch_grid(fig.add_subplot(grid[0, 1]), data["measured"], reference, data["before"],
               "(b) Before correction")
    patch_grid(fig.add_subplot(grid[0, 2]), data["corrected"], reference, data["after"],
               "(c) After correction")
    ax = fig.add_subplot(grid[1, :])
    positions = np.arange(24)
    ax.bar(positions - 0.2, data["before"], 0.4, color=GRAY, label=f"Before (mean {data['before'].mean():.1f})")
    ax.bar(positions + 0.2, data["after"], 0.4, color=BLUE, label=f"After (mean {data['after'].mean():.1f})")
    ax.set_xticks(positions, [str(i + 1) for i in positions])
    ax.set_xlabel("Patch")
    ax.set_ylabel("Color difference ΔE₀₀")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("(d) Difference between each patch and its reference color")
    save(fig, "colorcard.png")
    for photo in (CALIBRATION / "colorCard.jpg", DATA / "Cgiar_Bean_Sequia_Seeds" / "calibracion" / "colorCard.jpg",
                  DATA / "images" / "card_cameratrax.jpg"):
        if photo.exists():
            result = color_evaluation(photo)
            print(f"color {photo.parent.parent.name}/{photo.name}: ΔE00 {result['before'].mean():.1f} -> "
                  f"{result['after'].mean():.1f} (max {result['before'].max():.1f} -> {result['after'].max():.1f}); "
                  f"grays {result['before'][18:].mean():.1f} -> {result['after'][18:].mean():.1f}; "
                  f"left out of the fit by the glare rule: {[i + 1 for i in result['dropped']]}")


# ── 2. Segmentation ─────────────────────────────────────────────────────────

def tray():
    img = load_image(str(TRAY_PHOTO))
    ctx = objects_in(img, TRAY_SEG, TRAY_AREA)
    h, w = img.shape[:2]
    points = np.array(TRAY_AREA[0]["points"]) * (w, h)
    view = (slice(int(points[:, 1].min()), int(points[:, 1].max())), slice(int(points[:, 0].min()), int(points[:, 0].max())))
    return img, ctx, points, view


def fig_segmentation():
    if not have(TRAY_PHOTO):
        return
    img, ctx, points, view = tray()
    channel = colorspaces.channel(img, "LAB", 2)
    top, left = view[0].start, view[1].start
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    for ax in axes.ravel():
        ax.set_axis_off()
    axes[0, 0].imshow(rgb(img))
    closed = np.vstack([points, points[:1]])
    axes[0, 0].plot(closed[:, 0], closed[:, 1], color="yellow", lw=1.2)
    axes[0, 0].set_title("(a) Photo and analysis area (yellow)")
    axes[0, 1].imshow(channel[view], cmap="gray")
    axes[0, 1].set_title("(b) b* channel of CIELAB")
    axes[1, 0].imshow(ctx.mask[view], cmap="gray")
    axes[1, 0].set_title(f"(c) Mask, Otsu threshold b* ≥ {ctx.extra.get('otsu_threshold', 0):.0f}")
    axes[1, 1].imshow(rgb(img[view]))
    for object_id in range(1, int(ctx.labels.max()) + 1):
        mask = ctx.labels[view] == object_id
        if mask.any():
            contour = outline(mask)
            axes[1, 1].plot(contour[:, 0], contour[:, 1], color="yellow", lw=0.7)
    axes[1, 1].set_title(f"(d) {int(ctx.labels.max())} objects")
    fig.tight_layout()
    save(fig, "segmentation.png")

    h, w = img.shape[:2]
    inside, _ = areas.masks(TRAY_AREA, w, h)
    values = channel[inside > 0]
    threshold = ctx.extra.get("otsu_threshold", 0)
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    ax.hist(values, bins=256, range=(0, 255), color=BLUE)
    ax.axvline(threshold, color=RED, lw=1.2, label=f"Otsu threshold = {threshold:.0f}")
    ax.set_yscale("log")
    ax.set_xlim(values.min() - 5, values.max() + 5)
    ax.set_xlabel("b* value (8-bit scale; lower = bluer)")
    ax.set_ylabel("Pixels (log scale)")
    ax.text(0.03, 0.92, "blue tray", transform=ax.transAxes, fontsize=7)
    ax.text(0.97, 0.92, "seeds", transform=ax.transAxes, fontsize=7, ha="right")
    ax.legend(frameon=False, fontsize=7, loc="upper center")
    save(fig, "otsu_histogram.png")


def banana_strokes():
    """Brush strokes of four classes (background and three leaf colors: green, dark and yellow) on two leaves, as drawn
    in the interface; coordinates are fractions of the photo."""
    def young(px, py):
        return [round(0.40 + px / 352 * 0.54, 4), round(0.02 + py / 900 * 0.78, 4)]

    def old(px, py):
        return [round(0.52 + px / 422 * 0.30, 4), round(0.44 + py / 900 * 0.36, 4)]

    def stroke(cls, *pts, w=0.012):
        return {"c": cls, "pts": list(pts), "w": w}
    return {
        "20251203_095757.jpg": [
            stroke(0, [0.15, 0.20], [0.15, 0.60]), stroke(0, [0.62, 0.84], [0.80, 0.86]),
            stroke(0, [0.31, 0.80], [0.40, 0.90]),
            stroke(1, young(90, 640), young(120, 720)), stroke(1, young(250, 700), young(230, 760)),
            stroke(1, young(222, 110), young(235, 160)),
            stroke(2, young(80, 350), young(120, 400)), stroke(2, young(200, 300), young(240, 310))],
        "20251203_095619.jpg": [
            stroke(0, old(40, 600), old(40, 800)), stroke(0, old(265, 120), old(300, 180)),
            stroke(0, old(100, 880), old(350, 880)),
            stroke(1, old(215, 210), old(225, 250), w=0.006), stroke(1, old(250, 540), old(262, 568), w=0.006),
            stroke(2, old(170, 590), old(180, 640), w=0.006), stroke(2, old(240, 430), old(280, 440), w=0.006),
            stroke(3, old(115, 240), old(122, 300), w=0.006), stroke(3, old(290, 290), old(300, 320), w=0.006),
            stroke(3, old(320, 440), old(325, 520), w=0.006)],
    }


def red_card_height(img: np.ndarray) -> float | None:
    """Height in px of the red card (the side of its minimum-area rectangle closest to vertical)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 12) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 60)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(red)
    if len(stats) < 2:
        return None
    biggest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    if stats[biggest, cv2.CC_STAT_AREA] < 0.01 * img.shape[0] * img.shape[1]:
        return None
    ys, xs = np.nonzero(labels == biggest)
    box = cv2.boxPoints(cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32)))
    sides = [(box[k], box[(k + 1) % 4]) for k in range(4)]
    vertical = max(sides, key=lambda side: abs(side[1][1] - side[0][1]) / (np.linalg.norm(side[1] - side[0]) + 1e-9))
    return float(np.linalg.norm(vertical[1] - vertical[0]))


BANANA_CLASSES = ["background", "green", "dark", "yellow"]
BANANA_COLORS = ["#3B5BA5", "#4DAF4A", "#7B3F3F", "#E0B050"]


def banana_results():
    """Leaf area, length, width and share of each class for every banana photo."""
    import tempfile
    from fenotit.core.analysis import objects
    from fenotit.core.segmentation import classifier
    photos = sorted(BANANA.glob("*.jpg"))
    heights = [height for height in (red_card_height(load_image(str(p))) for p in photos) if height]
    mm_per_px = CARD_HEIGHT_MM / float(np.median(heights))
    strokes = banana_strokes()
    features, labels = [], []
    for name, marks in strokes.items():
        small = classifier.work_copy(load_image(str(BANANA / name)), 1.0)
        found_features, found_labels = classifier.samples(classifier.features(small), marks)
        features.append(found_features)
        labels.append(found_labels)
    features, labels = classifier.balance(np.concatenate(features), np.concatenate(labels))
    path, key = classifier.store(features, labels, Path(tempfile.mkdtemp()))
    segmentation = {"method": "classifier", "refine": [{"op": "fill_holes"}],
                    "classifier": {"samples": path, "hash": key, "objects": [1, 2, 3], "names": BANANA_CLASSES,
                                   "colors": BANANA_COLORS, "detail": 1.0}}
    area_above_card = [{"kind": "include", "type": "rect", "points": [[0.0, 0.0], [1.0, 0.76]]}]
    rows, views = [], {}
    for photo in photos:
        image = load_image(str(photo))
        result = objects.run(image, {"segmentation": segmentation, "measure_size": True, "measure_classes": True,
                                     "mm_per_pixel": mm_per_px, "length_unit": "cm",
                                     "roi_shapes": area_above_card, "exclude_border": True})
        leaf = max(result.extra["tables"]["objects"], key=lambda row: row.get("area_cm2", 0))
        rows.append({"photo": photo.stem.split("_")[-1], **leaf})
        views[photo.name] = (image, result)
    return mm_per_px, heights, rows, views


def fig_banana():
    if not have(BANANA):
        return
    mm_per_px, heights, rows, views = banana_results()
    shown = ["20251203_095757.jpg", "20251203_095645.jpg", "20251203_095543.jpg"]
    fig = plt.figure(figsize=(9.0, 6.6))
    grid = GridSpec(2, 6, figure=fig, height_ratios=[1.35, 1], hspace=0.3, wspace=0.08)
    for index, name in enumerate(shown):
        image, result = views[name]
        leaf = max(result.extra["tables"]["objects"], key=lambda row: row.get("area_cm2", 0))
        classes_view = result.step_images[next(v for v in result.step_images if "las" in v.lower() or "class" in v.lower())]
        mask = result.extra["geometry"][leaf["object_id"]]
        x0, y0 = np.maximum(mask.min(0) - 80, 0)
        x1, y1 = mask.max(0) + 80
        for offset, picture, label in ((0, image, "Photo"), (1, classes_view, "Classes")):
            ax = fig.add_subplot(grid[0, 2 * index + offset])
            ax.set_axis_off()
            ax.imshow(rgb(picture[y0:y1, x0:x1]))
            if offset == 0:
                ax.set_title(f"({'abc'[index]}) {leaf['area_cm2']:.0f} cm², {leaf['length_cm']:.1f} × "
                             f"{leaf['width_cm']:.1f} cm", fontsize=8, loc="left")
    ax = fig.add_subplot(grid[1, :])
    positions = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for name, color in zip(BANANA_CLASSES[1:], BANANA_COLORS[1:]):
        share = np.array([row.get(f"{name}_pct", 0) for row in rows])
        ax.bar(positions, share, bottom=bottom, color=color, label=name.capitalize(), width=0.7)
        bottom += share
    ax.set_xticks(positions, [row["photo"] for row in rows], rotation=45, fontsize=6.5)
    ax.set_ylabel("Share of the leaf (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    twin = ax.twinx()
    twin.plot(positions, [row["area_cm2"] for row in rows], "o-", color="black", ms=3, lw=0.8)
    twin.set_ylabel("Leaf area (cm²)")
    twin.spines["top"].set_visible(False)
    ax.set_xlabel("Photo (time of capture)")
    ax.set_title("(d) Leaf area and share of each color in the 13 photos", loc="left")
    save(fig, "banana.png")
    print(f"banana: card {np.median(heights):.1f} px tall (range {min(heights):.1f}-{max(heights):.1f}, "
          f"{len(heights)} photos) -> {mm_per_px:.4f} mm/px")
    for row in rows:
        print(f"  {row['photo']}: area {row['area_cm2']:.1f} cm2, length {row['length_cm']:.1f} cm, "
              f"width {row['width_cm']:.1f} cm, green {row.get('green_pct', 0):.1f} %, "
              f"dark {row.get('dark_pct', 0):.1f} %, yellow {row.get('yellow_pct', 0):.1f} %")


# ── 3. Separation of touching objects ───────────────────────────────────────

def touching_clump():
    """A crop of the tray photo where three to five seeds touch and form one component."""
    img = load_image(str(CLUMP_PHOTO))
    h, w = img.shape[:2]
    inside, exclusions = areas.masks(TRAY_AREA, w, h)
    ctx = run(img, [TRAY_SEG, {"step": "roi"}, {"step": "clean"}, {"step": "label"}], roi=inside, exclusions=exclusions)
    sizes = np.bincount(ctx.labels.ravel())[1:]
    typical = np.median(sizes[sizes > 0.2 * np.median(sizes)])
    threshold = {"step": "threshold", "params": {"color_space": "LAB", "channel": 2,
                                                 "min_val": int(ctx.extra["otsu_threshold"]), "max_val": 255}}
    from scipy.ndimage import find_objects
    boxes = find_objects(ctx.labels)
    candidates = sorted((abs(size / typical - 4), index + 1) for index, size in enumerate(sizes)
                        if 2.6 < size / typical < 5.5)
    for _, object_id in candidates:
        bounds = boxes[object_id - 1]
        window = (slice(max(0, bounds[0].start - 30), bounds[0].stop + 30),
                  slice(max(0, bounds[1].start - 30), bounds[1].stop + 30))
        crop, component = img[window], (ctx.labels[window] == object_id).astype(np.uint8)
        expected = int(round(sizes[object_id - 1] / typical))
        found = run(crop, [threshold, {"step": "roi"}, {"step": "clean"}, {"step": "separate"}], roi=component * 255)
        parts = len(np.unique(found.labels)) - 1
        if parts == expected and parts >= 3:
            return crop, component, threshold
    raise RuntimeError("no clump found")


def fig_separation():
    if not have(CLUMP_PHOTO):
        return
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    clump, component, threshold = touching_clump()
    base = run(clump, [threshold, {"step": "roi"}, {"step": "clean"}], roi=component * 255)
    binary = (base.mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, components = cv2.connectedComponents(binary, connectivity=8)
    rough = peak_local_max(distance, min_distance=5, threshold_abs=0.5, labels=components)
    spacing = max(3, int(0.9 * np.median(distance[tuple(rough.T)])))
    peaks = peak_local_max(distance, min_distance=spacing, threshold_abs=0.5, labels=components)
    peaks = peaks[distance[tuple(peaks.T)] >= 0.2 * distance.max()]
    markers = np.zeros(binary.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    basins = watershed(-distance, markers, mask=binary > 0)
    final = run(clump, [threshold, {"step": "roi"}, {"step": "clean"}, {"step": "separate"}], roi=component * 255)
    fig, ax = panels(5, 2.0, 2.0)
    shown = rgb(clump).copy()
    shown[component == 0] = (0.35 * shown[component == 0] + 0.65 * 255).astype(np.uint8)
    ax[0].imshow(shown)
    ax[0].set_title("(a) Touching seeds")
    ax[1].imshow(binary, cmap="gray")
    ax[1].set_title("(b) Mask: one component")
    ax[2].imshow(distance, cmap="magma")
    ax[2].plot(peaks[:, 1], peaks[:, 0], "c+", ms=8, mew=1.5)
    ax[2].set_title("(c) Distance to the\nbackground and its peaks")
    ax[3].imshow(np.ma.masked_equal(basins, 0), cmap="tab10", interpolation="nearest")
    ax[3].set_title("(d) Watershed from\nthe peaks")
    ax[4].imshow(shown)
    for object_id in range(1, int(final.labels.max()) + 1):
        if (final.labels == object_id).any():
            contour = outline(final.labels == object_id)
            ax[4].plot(contour[:, 0], contour[:, 1], lw=1.2, color="yellow")
    ax[4].set_title(f"(e) Final objects: {len(np.unique(final.labels)) - 1}")
    save(fig, "separation.png")


# ── 4–6. One seed: morphometry, outline and color ───────────────────────────

def dish():
    img = load_image(str(DISH_PHOTO))
    shapes = dish_area(img)
    return img, shapes


def chosen_seed(ctx):
    """The whole seed (solidity ≥ 0.97) closest to the median area."""
    rows = [row for row in ctx.tables["objects"] if row.get("solidity", 0) >= 0.97]
    median = np.median([row["area_px2"] for row in rows])
    return min(rows, key=lambda row: abs(row["area_px2"] - median))["object_id"]


def seed_crop(ctx, img, pad=30):
    object_id = chosen_seed(ctx)
    bounds = crop_box(ctx.labels, [object_id], pad=pad)
    return object_id, (ctx.labels[bounds] == object_id).astype(np.uint8), img[bounds], bounds


def fig_morphometry():
    if not have(DISH_PHOTO):
        return
    img, shapes = dish()
    ctx = objects_in(img, DISH_SEG, shapes, [{"step": "morphometry"}])
    _, mask, patch, _ = seed_crop(ctx, img)
    contour = outline(mask)
    hull = cv2.convexHull(contour[:-1].astype(np.int32))[:, 0]
    rect = cv2.boxPoints(cv2.minAreaRect(contour[:-1].astype(np.float32)))
    (ex, ey), (major, minor), angle = cv2.fitEllipse(contour[:-1].astype(np.int32))
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    fig, ax = panels(3, 2.8, 2.6)
    for axis in ax:
        axis.imshow(rgb(patch))
    ax[0].plot(contour[:, 0], contour[:, 1], color="#FFD700", lw=1)
    ax[0].plot(*np.vstack([rect, rect[:1]]).T, color=RED, lw=1)
    ax[0].plot(cx, cy, "w+", ms=8)
    ax[0].set_title("(a) Outline, smallest enclosing\nrectangle and centroid")
    ax[1].plot(*np.vstack([hull, hull[:1]]).T, color=GREEN, lw=1.2)
    ax[1].plot(contour[:, 0], contour[:, 1], color="#FFD700", lw=0.8)
    ax[1].set_title("(b) Convex hull")
    t = np.linspace(0, 2 * np.pi, 200)
    a = np.deg2rad(angle)
    x = ex + major / 2 * np.cos(t) * np.cos(a) - minor / 2 * np.sin(t) * np.sin(a)
    y = ey + major / 2 * np.cos(t) * np.sin(a) + minor / 2 * np.sin(t) * np.cos(a)
    ax[2].plot(x, y, color=BLUE, lw=1.2)
    points = contour[:-1].astype(float)
    radius = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    for index, color in ((radius.argmin(), RED), (radius.argmax(), GREEN)):
        ax[2].plot([cx, points[index, 0]], [cy, points[index, 1]], color=color, lw=1)
    ax[2].set_title("(c) Fitted ellipse, shortest and\nlongest radius from the centroid")
    save(fig, "morphometry.png")


def fig_efd():
    if not have(DISH_PHOTO):
        return
    img, shapes = dish()
    ctx = objects_in(img, DISH_SEG, shapes, [{"step": "morphometry"}])
    _, mask, _, _ = seed_crop(ctx, img)
    contour = outline(mask)[:-1].astype(float)
    fig, ax = panels(5, 1.9, 1.9)
    for axis, harmonics in zip(ax, (1, 2, 4, 8, 20)):
        coefficients = efd.efd(contour, harmonics)
        points = efd.contour_points(coefficients, 300)
        points = points + contour.mean(0) - points.mean(0)
        axis.plot(*np.vstack([contour, contour[:1]]).T, color="#BBBBBB", lw=2.2)
        axis.plot(*np.vstack([points, points[:1]]).T, color=RED, lw=1.1)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_title(f"{harmonics} harmonic{'s' if harmonics > 1 else ''}")
    save(fig, "efd_harmonics.png")


def seed_pixels(mask, patch):
    from fenotit.core.measurements.color import _core
    core = _core(mask, -1)
    return core, rgb(patch)[core > 0]


def fig_color():
    if not have(DISH_PHOTO):
        return
    from fenotit.core.measurements.color import _kmeans
    img, shapes = dish()
    ctx = objects_in(img, DISH_SEG, shapes, [{"step": "morphometry"}])
    _, mask, patch, _ = seed_crop(ctx, img)
    core, pixels = seed_pixels(mask, patch)
    ring = (mask > 0) & (core == 0)
    shown = rgb(patch).copy()
    shown[ring] = (255, 0, 255)
    clusters, labels = _kmeans(pixels, 3)
    painted = rgb(patch).copy()
    painted[mask == 0] = 255
    centers = np.array([color for color, _ in clusters], np.uint8)
    painted[core > 0] = centers[labels]
    fig, ax = panels(3, 2.9, 2.6)
    ax[0].imshow(shown)
    trim = int(np.clip(round(0.015 * np.sqrt(mask.sum())), 1, 3))
    ax[0].set_title(f"(a) Edge removed\n({trim} px ring, magenta)")
    ax[1].imshow(painted)
    ax[1].set_title("(b) Each pixel replaced by\nthe color of its group (k = 3)")
    ax[2].set_axis_on()
    left = 0.0
    from fenotit.core.measurements.color import _lab
    for color, share in clusters:
        lightness, a_value, b_value = _lab(color)
        hex_color = "#%02x%02x%02x" % color
        ax[2].barh(0, 100 * share, left=left, color=hex_color, edgecolor="white")
        ax[2].text(left + 50 * share, 0, f"{100 * share:.0f} %", ha="center", va="center", fontsize=7,
                   color="white" if lightness < 60 else "black")
        ax[2].text(left + 50 * share, -0.62, f"L* {lightness:.0f}\na* {a_value}\nb* {b_value}", ha="center",
                   va="top", fontsize=6)
        left += 100 * share
    ax[2].set_xlim(0, 100)
    ax[2].set_ylim(-1.7, 0.6)
    ax[2].set_yticks([])
    ax[2].set_xticks([])
    for side in ax[2].spines.values():
        side.set_visible(False)
    ax[2].set_title("(c) The three colors and their\nshare of the seed")
    save(fig, "color.png")


def k_curve(pixels, k_max=6):
    """For k = 1 … k_max: mean ΔE00 from each pixel to its group color, smallest ΔE00 between two
    group colors and smallest share of a group (the quantities used by the automatic k)."""
    from skimage.color import deltaE_ciede2000
    from fenotit.core.measurements import color as color_step
    sample = pixels
    if len(sample) > 1500:
        sample = sample[np.random.default_rng(134).choice(len(sample), 1500, replace=False)]
    lab = cv2.cvtColor(sample.reshape(-1, 1, 3).astype(np.float32) / 255, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    rows = []
    for k in range(1, k_max + 1):
        cv2.setRNGSeed(134)
        _, labels, centers = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1),
                                        3, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        error = float(np.mean(deltaE_ciede2000(lab, centers[labels])))
        separation = min((deltaE_ciede2000(centers[i], centers[j]) for i in range(k) for j in range(i + 1, k)),
                         default=np.nan)
        share = float((np.bincount(labels, minlength=k) / len(labels)).min())
        rows.append((k, error, float(separation), share))
    return rows, color_step.auto_k(pixels)


def fig_color_k():
    if not have(DISH_PHOTO):
        return
    from fenotit.core.measurements import color as color_step
    img, shapes = dish()
    ctx = objects_in(img, DISH_SEG, shapes, [{"step": "morphometry"}])
    _, mask, patch, _ = seed_crop(ctx, img)
    core, pixels = seed_pixels(mask, patch)
    rows, chosen = k_curve(pixels)
    fig = plt.figure(figsize=(9.0, 4.2))
    grid = GridSpec(2, 6, figure=fig, height_ratios=[0.55, 1], hspace=0.5, wspace=0.6)
    for index, k in enumerate(range(1, 7)):
        ax = fig.add_subplot(grid[0, index])
        ax.set_axis_off()
        clusters, labels = color_step._kmeans(pixels, k)
        painted = rgb(patch).copy()
        painted[mask == 0] = 255
        painted[core > 0] = np.array([color for color, _ in clusters], np.uint8)[labels]
        ax.imshow(painted)
        ax.set_title(f"k = {k}" + ("  (chosen)" if k == chosen else ""), fontsize=8,
                     color=BLUE if k == chosen else "black")
    ax = fig.add_subplot(grid[1, :3])
    ks = [row[0] for row in rows]
    ax.plot(ks, [row[1] for row in rows], "o-", color=BLUE)
    ax.axhline(color_step.AUTO_ERR, color=GRAY, ls="--", lw=1)
    ax.text(1, color_step.AUTO_ERR - 0.6, "limit 5", fontsize=7, color=GRAY)
    ax.axvline(chosen, color=BLUE, lw=0.8, alpha=0.4)
    ax.set_xlabel("Number of colors k")
    ax.set_ylabel("Mean ΔE₀₀ from each pixel\nto the color of its group")
    ax.set_title("(a) How well k colors describe the seed", loc="left")
    ax = fig.add_subplot(grid[1, 3:])
    ax.plot(ks[1:], [row[2] for row in rows[1:]], "o-", color=RED, label="Closest two colors (ΔE₀₀)")
    ax.axhline(color_step.AUTO_SEP, color=GRAY, ls="--", lw=1)
    ax.text(ks[-1], color_step.AUTO_SEP + 0.6, "limit 10", ha="right", fontsize=7, color=GRAY)
    ax.axvline(chosen, color=BLUE, lw=0.8, alpha=0.4)
    ax.set_xticks(ks[1:])
    ax.set_xlabel("Number of colors k")
    ax.set_ylabel("ΔE₀₀ between the two\nclosest colors")
    ax.set_title("(b) How distinct the k colors are", loc="left")
    save(fig, "color_k.png")
    counts = {}
    median = np.median([row["area_px2"] for row in ctx.tables["objects"] if row.get("area_px2")])
    seeds = [row["object_id"] for row in ctx.tables["objects"] if 0.4 * median < row.get("area_px2", 0) < 2.5 * median]
    for object_id in seeds:
        bounds = crop_box(ctx.labels, [object_id], pad=5)
        object_mask = (ctx.labels[bounds] == object_id).astype(np.uint8)
        _, object_pixels = seed_pixels(object_mask, img[bounds])
        k = color_step.auto_k(object_pixels)
        counts[k] = counts.get(k, 0) + 1
    print("color k:", [(k, round(e, 2), None if np.isnan(s) else round(s, 1), round(100 * m, 1)) for k, e, s, m in rows],
          "chosen", chosen, "| chosen k per seed in the photo:", dict(sorted(counts.items())))


# ── 7. Distances ────────────────────────────────────────────────────────────

def fig_distances():
    if not have(TRAY_PHOTO):
        return
    from scipy.spatial import Delaunay
    from fenotit.core.measurements.distances import _centroids, _edge_on_line
    img, _, points, view = tray()
    ctx = objects_in(img, TRAY_SEG, TRAY_AREA, [{"step": "distances", "params": {"neighbors": "voronoi"}}])
    centers = _centroids(ctx.labels)
    centroids = np.array([centers[i] for i in sorted(centers)])
    top, left = view[0].start, view[1].start
    fig = plt.figure(figsize=(8.4, 4.2))
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1], wspace=0.08)
    ax = fig.add_subplot(grid[0])
    ax.set_axis_off()
    ax.imshow(rgb(img[view]))
    triangles = Delaunay(centroids)
    ax.triplot(centroids[:, 0] - left, centroids[:, 1] - top, triangles.simplices, color="#FFD700", lw=0.6)
    ax.plot(centroids[:, 0] - left, centroids[:, 1] - top, "o", color=RED, ms=2)
    ax.set_title("(a) Neighbors: Delaunay triangulation of the centroids")
    pairs = sorted(ctx.tables["distances"], key=lambda row: row.get("center_distance_px", 1e9))
    pair = pairs[len(pairs) // 4]
    first, second = pair["object_a"], pair["object_b"]
    bounds = crop_box(ctx.labels, [first, second], pad=40)
    ax = fig.add_subplot(grid[1])
    ax.set_axis_off()
    ax.imshow(rgb(img[bounds]))
    oy, ox = bounds[0].start, bounds[1].start
    point_a, point_b = _edge_on_line(ctx.labels, first, second, centers[first], centers[second])
    ax.plot([centers[first][0] - ox, centers[second][0] - ox], [centers[first][1] - oy, centers[second][1] - oy],
            color="#DDDDDD", lw=1, ls="--")
    ax.plot([point_a[0] - ox, point_b[0] - ox], [point_a[1] - oy, point_b[1] - oy], color=RED, lw=2.2)
    ax.set_title("(b) Free space between two neighbors\nalong the line joining their centroids")
    save(fig, "distances.png")


# ── 8. Detection ────────────────────────────────────────────────────────────

def fig_detection():
    model_id = os.environ.get("FENOTIT_DOCS_DETECTOR", "")
    if not model_id:
        print("skipped detection.png (set FENOTIT_DOCS_DETECTOR to a detection model id)")
        return
    if not have(PEST):
        return
    from fenotit.core.analysis import detection, objects
    from fenotit.core.pipeline import overlay
    from fenotit.core.segmentation import models
    model = models.get(model_id)
    colors = ["#E41A1C", "#FF7F00", "#984EA3", "#4DAF4A", "#377EB8"]
    detector = {"name": model_id, "conf": 0.5, "use": [], "classes": model["classes"], "colors": colors}
    columns = [f"n_{name.lower()}" for name in model["classes"]]
    per_seed, seed_view = [], None
    for photo in SEED_PHOTOS:
        image = load_image(str(PEST / f"{photo}.jpg"))
        params = {"segmentation": {"method": "threshold", **DISH_SEG["params"], "refine": [{"op": "fill_holes"}]},
                  "roi_shapes": dish_area(image, 0.9), "exclude_border": True, "measure_size": True,
                  "measure_detections": True, "detector": detector}
        result = objects.run(image, params)
        rows = [row for row in result.extra["tables"]["objects"] if row.get("area_px2")]
        median = np.median([row["area_px2"] for row in rows])
        seeds = [row for row in rows if 0.4 * median < row["area_px2"] < 2.5 * median]
        per_seed += [[row.get(column, 0) for column in columns] for row in seeds]
        if photo == "051":
            name = next(view for view in result.step_images if view == result.extra.get("first_view") or "etec" in view)
            name = next(view for view in result.step_images if "etec" in view)
            seed_view = result.step_images[name].copy()
            overlay.draw(seed_view, result.extra["overlays"][name], overlay.resolve(None, seed_view))
    per_seed = np.array(per_seed)
    insects, insect_view = [], None
    for photo in INSECT_PHOTOS:
        image = load_image(str(PEST / f"{photo}.jpg"))
        result = detection.run(image, {"detector": {**detector, "use": [3]}, "roi_shapes": dish_area(image, 0.9),
                                       "measure_size": True})
        insects.append(int(result.stats.get("n_objects", 0)))
        if photo == "186":
            insect_view = result.extra["base_image"].copy()
            for row in result.extra["tables"]["objects"]:
                box = result.extra["geometry"][row["object_id"]]
                cv2.polylines(insect_view, [box.reshape(-1, 1, 2)], True, overlay.hex_to_bgr(colors[3]), 8)
    holes = per_seed[:, 0]
    fig = plt.figure(figsize=(9.6, 3.4))
    grid = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.9], wspace=0.3)
    h, w = seed_view.shape[:2]
    ax = fig.add_subplot(grid[0])
    ax.set_axis_off()
    ax.imshow(rgb(seed_view[int(0.06 * h):int(0.94 * h), int(0.2 * w):int(0.82 * w)]))
    ax.set_title("(a) Detections per seed (Objects)", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="none", ec=color, lw=1.5) for color in colors]
    ax.legend(handles, model["classes"], frameon=False, fontsize=5.5, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 0.02))
    h, w = insect_view.shape[:2]
    ax = fig.add_subplot(grid[1])
    ax.set_axis_off()
    ax.imshow(rgb(insect_view[int(0.3 * h):int(0.85 * h), int(0.25 * w):int(0.65 * w)]))
    ax.set_title("(b) Adult insects in a dish (Detection)", loc="left")
    ax = fig.add_subplot(grid[2])
    values, counts = np.unique(holes, return_counts=True)
    ax.bar(values, counts, color=BLUE, width=0.7)
    ax.set_xticks(values)
    ax.set_xlabel("Holes (class Agujero) per seed")
    ax.set_ylabel("Seeds")
    ax.set_title(f"(c) {len(holes)} seeds, {len(SEED_PHOTOS)} photos", loc="left")
    save(fig, "detection.png")
    totals = dict(zip(model["classes"], per_seed.sum(0).tolist()))
    print(f"detection: {len(holes)} seeds, holes per seed mean {holes.mean():.2f} (max {holes.max()}), "
          f"seeds with at least one hole {100 * (holes > 0).mean():.0f} %, per-seed totals {totals}; "
          f"insects (Bruquido) per dish photo {dict(zip(INSECT_PHOTOS, insects))}")


def main():
    fig_lens()
    fig_colorcard()
    fig_segmentation()
    fig_banana()
    fig_separation()
    fig_morphometry()
    fig_efd()
    fig_color()
    fig_color_k()
    fig_distances()
    fig_detection()


if __name__ == "__main__":
    main()
