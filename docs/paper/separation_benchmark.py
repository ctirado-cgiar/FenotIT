"""Benchmark of the separation of touching objects (paper figures).

Synthetic scenes are composed from real bean seeds cut out of the test photos, with a known
number of seeds and a controlled degree of contact. Each scene is segmented once and then
separated with successive versions of the algorithm, from plain connected components to the
current FenotIT step, so the figures show what each rule adds.

    python docs/paper/separation_benchmark.py            (writes docs/paper/out/*)
"""
from __future__ import annotations

import csv
import glob
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))
import figure_style as style  # noqa: E402
from fenotit.core.pipeline.base import Context  # noqa: E402
from fenotit.core.objects import separate as ob  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
RNG = np.random.default_rng(134)
LEVELS = [("Isolated", None), ("Touching", (0.0, 0.03)), ("Overlap 3–10 %", (0.03, 0.10)),
          ("Overlap 10–20 %", (0.10, 0.20)), ("Overlap 20–30 %", (0.20, 0.30))]
N_SEEDS, N_SCENES, SIZE, CLUSTER = 30, 20, 1300, 5
VERSIONS = ["Connected\ncomponents", "Watershed,\nfixed spacing", "+ spacing from\nobject size",
            "+ neck\nmerge", "+ notch and\nsize rules", "FenotIT\n(current)"]
COLORS = ["#D0D0D0", "#B0B0B0", "#8C8C8C", "#666666", "#333333", style.BLUE]     # versiones: grises, actual en azul
MARKERS = ["o", "s", "^", "D", "v", "o"]


# ── scenes ──────────────────────────────────────────────────────────────────

def segment(img):
    """Seeds on the blue tray: Otsu on the Cb channel (blue background = high Cb)."""
    cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 2]
    t, _ = cv2.threshold(cb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (cb < t).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def seed_library():
    seeds = []
    for f in sorted(glob.glob(str(ROOT / "tests" / "data" / "images" / "[01]*.jpg")))[:10]:
        img = cv2.imread(f)
        m = segment(img)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        a = st[1:, 4]
        med = np.median(a[a > 200])
        H, W = m.shape
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if not 0.6 * med < area < 1.5 * med or x < 5 or y < 5 or x + w > W - 5 or y + h > H - 5:
                continue
            p = 6
            sl = (slice(y - p, y + h + p), slice(x - p, x + w + p))
            seeds.append(((lab[sl] == i).astype(np.uint8), img[sl].copy()))
    return seeds


def background():
    img = cv2.imread(str(ROOT / "tests" / "data" / "images" / "01.jpg"))
    tile = img[1900:2500, 150:750]                      # empty tray, real texture
    reps = SIZE // tile.shape[0] + 1
    return np.tile(tile, (reps, reps, 1))[:SIZE, :SIZE].copy()


def rotate(mask, patch, angle):
    h, w = mask.shape
    d = int(np.hypot(h, w)) + 4
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += [(d - w) / 2, (d - h) / 2]
    m = cv2.warpAffine(mask, M, (d, d), flags=cv2.INTER_NEAREST)
    p = cv2.warpAffine(patch, M, (d, d), flags=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(m)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return m[sl], p[sl]


def compose(lib, level, bg):
    """Scene with N_SEEDS seeds; returns image and the visible mask of each seed."""
    img = bg.copy()
    owner = np.zeros((SIZE, SIZE), np.int32)            # which seed is on top at each pixel
    margin = 40
    k = 0
    attempts = 0
    while k < N_SEEDS and attempts < 40000:
        attempts += 1
        m, p = rotate(*lib[RNG.integers(len(lib))], RNG.uniform(0, 360))
        h, w = m.shape
        y, x = RNG.integers(margin, SIZE - margin - h), RNG.integers(margin, SIZE - margin - w)
        under = owner[y:y + h, x:x + w]
        area = m.sum()
        ov = ((under > 0) & (m > 0)).sum() / area
        near = cv2.dilate((under > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
        contact = (near & m).any()
        rng = LEVELS[level][1]
        new_cluster = k % CLUSTER == 0
        if rng is None or new_cluster:
            gap = cv2.dilate((under > 0).astype(np.uint8), np.ones((25, 25), np.uint8))
            if (gap & m).any():
                continue
        else:
            lo, hi = rng
            if not contact or not (lo <= ov <= hi):
                continue
            # contact must be with the current cluster
            ids = np.unique(under[cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0])
            if not any(i > (k // CLUSTER) * CLUSTER for i in ids):
                continue
        k += 1
        reg = img[y:y + h, x:x + w]
        reg[m > 0] = p[m > 0]
        under[m > 0] = k
    visible = [owner == i for i in range(1, k + 1)]
    return img, visible


# ── algorithm versions ──────────────────────────────────────────────────────

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


def score(labels, visible):
    """Count and one-to-one matching with IoU ≥ 0.5 against the visible part of each seed."""
    ids = [i for i in np.unique(labels) if i]
    areas = np.bincount(labels.ravel())
    noise = 0.15 * np.median([v.sum() for v in visible])
    ids = [i for i in ids if areas[i] >= noise]
    tp = 0
    used = set()
    for v in visible:
        if v.sum() < 0.3 * noise / 0.15 * 0.15:
            continue
        cand = np.bincount(labels[v].ravel())
        cand[0] = 0
        j = int(cand.argmax())
        if j == 0 or j in used:
            continue
        inter = cand[j]
        iou = inter / (v.sum() + areas[j] - inter)
        if iou >= 0.5:
            tp += 1
            used.add(j)
    n_true = len(visible)
    prec = tp / max(len(ids), 1)
    rec = tp / n_true
    return len(ids), prec, rec


# ── run ─────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lib = seed_library()
    bg = background()
    print(f"seed library: {len(lib)} seeds")
    rows, examples = [], {}
    for li, (name, _) in enumerate(LEVELS):
        for s in range(N_SCENES):
            img, visible = compose(lib, li, bg)
            mask = segment(img)
            for v in range(len(VERSIONS)):
                t0 = time.perf_counter()
                lab = version(mask, img, v)
                dt = time.perf_counter() - t0
                n, p, r = score(lab, visible)
                rows.append({"level": li, "level_name": name, "scene": s, "version": v,
                             "n_true": len(visible), "n_found": n, "precision": p, "recall": r,
                             "f1": 2 * p * r / max(p + r, 1e-9), "seconds": dt})
            if s == 0:
                examples[li] = (img, mask, visible)
        print(name, "done")
    with open(OUT / "separation_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    figures(rows, examples)


def outline(ax, img, labels, color):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in np.unique(labels):
        if not i:
            continue
        cnts, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            ax.plot(*np.vstack([c[:, 0], c[:1, 0]]).T, lw=0.7, color=color)
    style.photo_axes(ax)


def figures(rows, examples):
    import pandas as pd
    style.apply()
    df = pd.DataFrame(rows)
    df["count_error"] = 100 * (df.n_found - df.n_true) / df.n_true
    fig = plt.figure(figsize=(style.DOUBLE, 165 * style.MM))
    rows_grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.35], hspace=0.22)

    levels = rows_grid[0].subgridspec(1, len(LEVELS), wspace=0.06)      # escenas de cada nivel
    for li, (name, _) in enumerate(LEVELS):
        ax = fig.add_subplot(levels[li])
        ax.imshow(cv2.cvtColor(examples[li][0], cv2.COLOR_BGR2RGB))
        style.photo_axes(ax)
        ax.set_title(f"({chr(97 + li)}) {name}")

    img, mask, visible = examples[3]                                    # el mismo grupo en cada versión
    ys, xs = np.nonzero(visible[0] | visible[1] | visible[2] | visible[3] | visible[4])
    sl = (slice(max(ys.min() - 30, 0), ys.max() + 30), slice(max(xs.min() - 30, 0), xs.max() + 30))
    versions = rows_grid[1].subgridspec(1, len(VERSIONS), wspace=0.06)
    for v in range(len(VERSIONS)):
        ax = fig.add_subplot(versions[v])
        sub = version(mask, img, v)[sl]
        n = len([i for i in np.unique(sub) if i])
        outline(ax, img[sl], sub, style.WHITE)
        ax.set_title(f"({chr(102 + v)}) {n} object{'s' if n != 1 else ''}", fontsize=6.5)
        ax.text(0.5, -0.04, VERSIONS[v], transform=ax.transAxes, ha="center", va="top", fontsize=6)

    stats = rows_grid[2].subgridspec(1, 2, wspace=0.3)
    x = np.arange(len(LEVELS))
    left, right = fig.add_subplot(stats[0]), fig.add_subplot(stats[1])
    for v in range(len(VERSIONS)):
        g = df[df.version == v].groupby("level")
        for ax, column in ((left, "f1"), (right, "count_error")):
            ax.errorbar(x, g[column].mean(), yerr=g[column].std(), color=COLORS[v], marker=MARKERS[v], ms=3,
                        mfc=style.WHITE if v < len(VERSIONS) - 1 else COLORS[v], lw=1.0 if v == len(VERSIONS) - 1 else 0.7,
                        capsize=1.5, label=VERSIONS[v].replace("\n", " "))
    for ax in (left, right):
        ax.set_xticks(x, [name.replace(" ", "\n", 1) for name, _ in LEVELS])
    left.set_ylabel("F1 (IoU ≥ 0.5)")
    left.set_ylim(0, 1.03)
    left.set_title("(l)")
    right.axhline(0, color=style.GRAY, lw=0.6)
    right.set_ylabel("Count error (%)")
    right.set_title("(m)")
    left.legend(loc="lower left", fontsize=6)
    style.chart_axes(left, right)
    style.save(fig, OUT / "separation.png")

    summary = df.groupby(["level_name", "version"], sort=False).agg(
        f1=("f1", "mean"), f1_sd=("f1", "std"), count_error=("count_error", "mean"),
        precision=("precision", "mean"), recall=("recall", "mean"), seconds=("seconds", "mean")).round(3)
    summary.to_csv(OUT / "separation_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
