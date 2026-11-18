"""Paper figure for the perspective correction with ArUco markers: photographs of common bean
plants in the field, taken in front of a board with four markers, before and after FenotIT
rectifies them. The four markers are identical squares, so the variation of their side lengths
and the deviation of their corners from 90° measure the perspective left in each image.

    python docs/paper/corrections_figures.py            (writes docs/paper/out/corr_aruco_field.png)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))
import figure_style as style  # noqa: E402
from fenotit.core.corrections import pipeline as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
FIELD = ROOT / "tests" / "data" / "Cgiar_Bean_FieldPlant_aruco"
FIELD_EXAMPLE = "1773256030039.jpg"
BLUE, GRAY = style.BLUE, style.GRAY


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def marker_geometry(corners):
    """Variación de los lados de los cuatro marcadores (coeficiente de variación, %) y desvío medio de
    sus esquinas respecto de 90° (grados). Los cuatro marcadores son cuadrados iguales."""
    sides, angles = [], []
    for quad in corners:
        for i in range(4):
            before, here, after = quad[i - 1], quad[i], quad[(i + 1) % 4]
            sides.append(np.linalg.norm(after - here))
            first, second = before - here, after - here
            cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
            angles.append(abs(np.degrees(np.arccos(np.clip(cosine, -1, 1))) - 90))
    sides = np.array(sides)
    return 100 * sides.std() / sides.mean(), float(np.mean(angles))


def fig_aruco_field():
    """Plantas de frijol en campo delante de una lámina con cuatro ArUco: forma de los marcadores en la
    foto y en la foto enderezada por FenotIT."""
    from fenotit.core.image_io import load_image
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    rows, example = [], None
    for path in sorted(FIELD.glob("*.jpg")):
        photo = load_image(str(path))
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY))
        if ids is None or set(ids.ravel()) < {0, 1, 2, 3}:
            print("aruco field: skipped", path.name)
            continue
        quads = [corners[list(ids.ravel()).index(i)][0] for i in range(4)]
        transform, size, _, _ = P.rectify_transform(photo, P.Perspective(True))
        straight = [cv2.perspectiveTransform(quad.reshape(-1, 1, 2), transform).reshape(4, 2) for quad in quads]
        rows.append((path.name, *marker_geometry(quads), *marker_geometry(straight)))
        if path.name == FIELD_EXAMPLE:
            example = (photo, cv2.warpPerspective(photo, transform, size, borderValue=(255, 255, 255)), quads)
    data = np.array([row[1:] for row in rows])
    style.apply()
    fig = plt.figure(figsize=(style.DOUBLE, 75 * style.MM), layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=[0.75, 0.75, 1, 1])
    photo, straight, quads = example
    ax = fig.add_subplot(grid[0])
    ax.imshow(rgb(photo))
    for index, quad in enumerate(quads):
        ax.add_patch(plt.Polygon(quad, fill=False, ec=style.PHOTO_COLORS[index], lw=1.0))
    style.photo_axes(ax)
    ax.set_title("(a)")
    ax = fig.add_subplot(grid[1])
    ax.imshow(rgb(straight))
    style.photo_axes(ax)
    ax.set_title("(b)")
    for column, (before, after, label, title) in enumerate((
            (0, 2, "Marker side, CV (%)", "(c)"), (1, 3, "Corner angle, deviation (°)", "(d)"))):
        ax = fig.add_subplot(grid[2 + column])
        for value_before, value_after in data[:, [before, after]]:
            ax.plot([0, 1], [value_before, value_after], color=GRAY, lw=0.5)
        ax.scatter([0] * len(data), data[:, before], s=14, facecolors=style.WHITE, edgecolors=BLUE, linewidths=0.7, zorder=3)
        ax.scatter([1] * len(data), data[:, after], s=14, color=BLUE, zorder=3)
        ax.set_xticks([0, 1], ["Photograph", "Rectified"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(0, None)
        ax.set_ylabel(label)
        ax.set_title(title)
        style.chart_axes(ax)
    style.save(fig, OUT / "corr_aruco_field.png")
    print("aruco field (side CV %, angle deg; photo -> rectified):")
    for row in rows:
        print("  ", row[0], [round(value, 2) for value in row[1:]])
    print("   mean", data.mean(0).round(2).tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_aruco_field()


if __name__ == "__main__":
    main()
