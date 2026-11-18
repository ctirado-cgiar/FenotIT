"""Separation of touching seeds on real photographs (paper figure).

Twelve clusters of touching bean seeds were taken from the drought trial photographs
(tests/data/Cgiar_Bean_Sequia_Seeds) and their seeds were counted by hand
(docs/paper/separation_areas.csv). Each cluster is separated with successive versions of the
algorithm, from plain connected components to the current FenotIT step, and the number of
objects found is compared with the manual count.

    python docs/paper/separation_benchmark.py            (writes docs/paper/out/separation.png)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))
import figure_style as style  # noqa: E402
from fenotit.core.image_io import load_image  # noqa: E402
from fenotit.core.objects import separate as ob  # noqa: E402
from fenotit.core.pipeline.base import Context  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
PHOTOS = ROOT / "tests" / "data" / "Cgiar_Bean_Sequia_Seeds"
AREAS = Path(__file__).resolve().parent / "separation_areas.csv"
TRAY_RIGHT = 0.62                     # fracción del ancho donde termina la bandeja (a la derecha, la tarjeta)
PAD = 40                              # margen alrededor de cada grupo, en px
VERSIONS = ["Connected components", "Watershed, fixed spacing", "+ spacing from object size",
            "+ neck merge", "+ notch and size rules", "FenotIT (current)"]
COLORS = ["#D0D0D0", "#B0B0B0", "#8C8C8C", "#666666", "#333333", style.BLUE]     # versiones: grises, actual en azul


def segment(img):
    """Seeds on the blue tray: Otsu on the Cb channel (blue background = high Cb)."""
    cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 2]
    t, _ = cv2.threshold(cb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (cb < t).astype(np.uint8)
    m[:, int(TRAY_RIGHT * m.shape[1]):] = 0
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def cluster(row):
    """Recorte de la foto y máscara del grupo: el componente con más píxeles dentro del recuadro."""
    img = load_image(str(PHOTOS / row["photo"]))
    x, y, width, height = (int(row[key]) for key in ("x", "y", "width", "height"))
    _, labels = cv2.connectedComponents(segment(img), connectivity=8)
    inside = labels[y:y + height, x:x + width]
    biggest = np.bincount(inside[inside > 0]).argmax()
    window = (slice(y - PAD, y + height + PAD), slice(x - PAD, x + width + PAD))
    return img[window], (labels[window] == biggest).astype(np.uint8)


def version(mask, img, v):
    binary = (mask > 0).astype(np.uint8)
    _, comps = cv2.connectedComponents(binary, connectivity=8)
    if v == 0:
        return comps
    if v == 5:
        ctx = Context(image=img, mask=binary * 255)
        ob.separate(ctx, {"min_distance": 0, "peak_threshold": 0.2, "merge_ratio": 0.9,
                          "split_large": True, "use_shadows": True})
        return ctx.labels
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if v == 1:                                          # scikit-image tutorial style
        peaks = peak_local_max(dist, min_distance=5, threshold_abs=0.5, labels=comps)
        radius = 0
    else:
        comp_max = ob._label_max(dist, comps)

        def pk(md):
            f = peak_local_max(dist, min_distance=md, threshold_abs=0.5, labels=comps)
            return f[dist[tuple(f.T)] >= 0.2 * comp_max[comps[tuple(f.T)]]]
        rough = pk(5)
        md = max(3, int(0.9 * np.median(dist[tuple(rough.T)]))) if len(rough) else 5
        peaks = pk(md)
        radius = float(np.median(dist[tuple(peaks.T)])) if len(peaks) else 0
    markers = np.zeros(binary.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-dist, markers, mask=binary > 0).astype(np.int32)
    if v >= 3:
        labels = ob._merge_flat_necks(labels, dist, 0.9, None)
    if v >= 4:
        labels = ob._merge_without_notches(labels, binary, radius, None, dist)
        labels = ob._merge_fragments(labels, comps)
        labels = ob._keep_single_sized_groups(labels, comps)
    return labels


def count(labels):
    return len([i for i in np.unique(labels) if i])


def outline(ax, img, labels, color):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in np.unique(labels):
        if not i:
            continue
        cnts, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            ax.plot(*np.vstack([c[:, 0], c[:1, 0]]).T, lw=0.7, color=color)
    style.photo_axes(ax)


def main():
    import pandas as pd
    import seaborn as sns
    style.apply()
    rows, example = [], None
    with open(AREAS, newline="") as table:
        areas = [row for row in csv.DictReader(table) if row["seeds"]]
    for row in areas:
        img, mask = cluster(row)
        labels = [version(mask, img, v) for v in range(len(VERSIONS))]
        counts = [count(item) for item in labels]
        manual = int(row["seeds"])
        for v, found in enumerate(counts):
            rows.append({"area": int(row["area"]), "seeds": manual, "version": VERSIONS[v], "found": found,
                         "count_error": 100 * (found - manual) / manual})
        spread = len(set(counts)) + 2 * (counts[-1] == manual)
        if manual <= 13 and (example is None or spread > example[0]):
            example = (spread, img, labels, manual)
    data = pd.DataFrame(rows)
    data.to_csv(OUT / "separation_real.csv", index=False)

    fig = plt.figure(figsize=(style.DOUBLE, 98 * style.MM), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[31, 62])
    axes = top.subplots(1, len(VERSIONS))
    _, img, labels, manual = example
    for v, ax in enumerate(axes):
        outline(ax, img, labels[v], style.WHITE)
        found = count(labels[v])
        ax.set_title(f"({chr(97 + v)}) {VERSIONS[v]}\n{found} object{'s' if found != 1 else ''}", fontsize=6)
    left, right = bottom.subplots(1, 2, width_ratios=[1, 1.6])
    current = data[data.version == VERSIONS[-1]]
    limit = max(data.seeds.max(), current.found.max()) + 2
    left.plot([0, limit], [0, limit], color=style.GRAY, lw=0.6, ls="--")
    left.set_box_aspect(1)
    left.scatter(current.seeds, current.found, s=16, color=style.BLUE, zorder=3)
    left.set(xlim=(0, limit), ylim=(0, limit), xlabel="Seeds counted by hand", ylabel="Objects found by FenotIT")
    left.set_title("(g)")
    right.axvline(0, color=style.GRAY, lw=0.6)
    sns.stripplot(data, y="version", x="count_error", hue="version", palette=COLORS, size=3.5, jitter=0.15,
                  legend=False, ax=right)
    right.set(ylabel="", xlabel="Count error (%)")
    right.set_title("(h)")
    style.chart_axes(left, right)
    style.save(fig, OUT / "separation.png")
    summary = data.groupby("version", sort=False).agg(mean_abs_error=("count_error", lambda e: e.abs().mean()),
                                                     exact=("count_error", lambda e: (e == 0).sum()))
    print(summary.round(1).to_string())
    print(current[["area", "seeds", "found"]].to_string(index=False))


if __name__ == "__main__":
    main()
