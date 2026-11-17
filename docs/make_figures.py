"""Figures for docs/methods: each one is produced with FenotIT's own pipeline functions, so
the pictures show what the software actually does.

    python docs/make_figures.py          (writes docs/img/*.png)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fenotit.core import colorspaces, efd  # noqa: E402
from fenotit.core.image_io import load_image  # noqa: E402
from fenotit.core.pipeline import run  # noqa: E402

OUT = ROOT / "docs" / "img"
IMAGE = ROOT / "tests" / "data" / "images" / "01.jpg"
CALIBRATION = ROOT / "tests" / "data" / "Cgiar_Bean_Durango_Seeds" / "calibracion"
BOARD_CORNERS = (7, 6)
CLUMP_IMAGE = ROOT / "tests" / "data" / "images" / "23455.jpg"
CLUMP_BOX = (985, 943, 227, 260)                 # top, left, height, width
CLUMP_SEG = {"step": "otsu", "params": {"color_space": "LAB", "channel": 2}}
SEG = {"step": "threshold", "params": {"color_space": "YCrCb", "channel": 1, "min_val": 122, "max_val": 255}}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def panels(n, w=2.6, h=2.6):
    fig, axes = plt.subplots(1, n, figsize=(w * n, h))
    for ax in np.atleast_1d(axes):
        ax.set_axis_off()
    return fig, np.atleast_1d(axes)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def crop_box(labels, ids, pad=40):
    ys, xs = np.nonzero(np.isin(labels, ids))
    return slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad)


# ── 0. Corrections ──────────────────────────────────────────────────────────

def fig_lens():
    """Chessboard photos of the capture box: detected corners, corrected photo and displacement."""
    from fenotit.core.corrections import distortion, pipeline
    photos = sorted(str(p) for p in CALIBRATION.glob("[0-9].jpg"))
    result = distortion.calibrate(photos, *BOARD_CORNERS)
    config = pipeline.Distortion(enabled=True, mtx=result.mtx.tolist(),
                                 dist=np.ravel(result.dist).tolist(), size=list(result.size))
    detection = next(d for d in result.detections if d.found)
    photo = load_image(detection.path)
    height, width = photo.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(0, width, 8, dtype=np.float32), np.arange(0, height, 8, dtype=np.float32))
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(points, result.mtx, result.dist, P=result.mtx).reshape(-1, 2)
    shift = np.linalg.norm(ideal - points.reshape(-1, 2), axis=1).reshape(grid_x.shape)
    fig, ax = panels(3, 3.2, 2.5)
    ax[0].imshow(rgb(photo))
    corners = detection.corners.reshape(-1, 2)
    ax[0].plot(corners[:, 0], corners[:, 1], "+", color="#D73027", ms=4, mew=0.8)
    ax[0].set_title(f"(a) Chessboard photo, {len(corners)} inner corners")
    ax[1].imshow(rgb(pipeline.undistort(photo, config)))
    ax[1].set_title(f"(b) Corrected ({result.n_images_used} photos, RMS {result.rms_error:.2f} px)")
    shown = ax[2].imshow(shift, cmap="viridis", extent=(0, width, height, 0))
    for axis in (ax[2].xaxis, ax[2].yaxis):
        axis.set_visible(False)
    ax[2].set_title("(c) Displacement of each pixel (px)")
    fig.colorbar(shown, ax=ax[2], fraction=0.035, pad=0.02)
    save(fig, "lens.png")


def fig_colorcard():
    """Color card photographed in the capture box: detection, patch colors and corrected photo."""
    from fenotit.core.corrections import colorcard
    photo = load_image(str(CALIBRATION / "colorCard.jpg"))
    fixed, info = colorcard.correct(photo)
    card = info["card"]
    centers = card.centers.reshape(-1, 2)
    pad = 2.2 * card.chip
    x0, y0 = np.maximum(centers.min(axis=0) - pad, 0).astype(int)
    x1, y1 = (centers.max(axis=0) + pad).astype(int)
    before = np.rot90(colorcard.measure(photo, card), info["turns"]).reshape(-1, 3)
    after = np.rot90(colorcard.measure(fixed, card), info["turns"]).reshape(-1, 3)
    reference = colorcard.COLORCHECKER24.reshape(-1, 3)
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9), gridspec_kw={"width_ratios": [1, 1.6, 1], "wspace": 0.45})
    for axis in (ax[0], ax[2]):
        axis.set_axis_off()
    ax[0].imshow(rgb(photo[y0:y1, x0:x1]))
    half = card.chip * 0.2
    for x, y in centers:
        ax[0].add_patch(plt.Rectangle((x - x0 - half, y - y0 - half), 2 * half, 2 * half,
                                      fill=False, ec="white", lw=0.8))
    ax[0].set_title(f"(a) Card found ({card.found} of {card.rows * card.cols} patches seen)")
    strips = np.clip(np.stack([before, after, reference]), 0, 1)
    ax[1].imshow(strips, aspect="auto", interpolation="nearest")
    ax[1].set_yticks(range(3), ("measured", "corrected", "reference"))
    ax[1].set_xticks([])
    ax[1].tick_params(length=0)
    for side in ax[1].spines.values():
        side.set_visible(False)
    for index in info["dropped"]:
        ax[1].plot(index, 0, "x", color="white", ms=5, mew=1.2)
    ax[1].set_title(f"(b) Patch colors, × = not used (error {info['mean_error_rgb']:.1f} of 255)")
    ax[2].imshow(rgb(fixed[y0:y1, x0:x1]))
    ax[2].set_title("(c) Corrected photo")
    save(fig, "colorcard.png")


# ── 1. Segmentation ─────────────────────────────────────────────────────────

def fig_segmentation(img):
    ctx = run(img, [SEG, {"step": "roi"}, {"step": "clean"}, {"step": "label"},
                    {"step": "filter", "params": {"exclude_border": True}}])
    sl = crop_box(ctx.labels, list(range(1, int(ctx.labels.max()) + 1)), pad=60)
    ch = colorspaces.channel(img, "YCrCb", 1)
    fig, ax = panels(4)
    ax[0].imshow(rgb(img[sl]))
    ax[0].set_title("(a) Image (RGB)")
    ax[1].imshow(ch[sl], cmap="gray")
    ax[1].set_title("(b) Cr channel (YCrCb)")
    ax[2].imshow(ctx.mask[sl], cmap="gray")
    ax[2].set_title("(c) Binary mask, 122 ≤ Cr ≤ 255")
    ax[3].imshow(ctx.labels[sl] % 20, cmap="tab20", interpolation="nearest")
    ax[3].set_title("(d) Connected components")
    save(fig, "segmentation.png")

    fig, axh = plt.subplots(figsize=(4.2, 2.2))
    axh.hist(ch.ravel(), bins=256, range=(0, 255), color="#2166AC")
    t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    axh.axvline(t, color="#D73027", lw=1.2, label=f"Otsu threshold = {t:.0f}")
    axh.axvline(122, color="#333333", lw=1, ls="--", label="manual threshold = 122")
    axh.set_yscale("log")
    axh.set_xlabel("Cr value")
    axh.set_ylabel("pixels (log)")
    axh.legend(frameon=False, fontsize=7)
    save(fig, "otsu_histogram.png")


# ── 2. Separation of touching objects ───────────────────────────────────────

def touching_clump():
    """Crop of a test photo where four seeds touch and form one component."""
    top, left, height, width = CLUMP_BOX
    return load_image(str(CLUMP_IMAGE))[top:top + height, left:left + width]


def fig_separation():
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    clump = touching_clump()
    base = run(clump, [CLUMP_SEG, {"step": "roi"}, {"step": "clean"}])
    binary = (base.mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, comps = cv2.connectedComponents(binary, connectivity=8)
    rough = peak_local_max(dist, min_distance=5, threshold_abs=0.5, labels=comps)
    md = max(3, int(0.9 * np.median(dist[tuple(rough.T)])))
    peaks = peak_local_max(dist, min_distance=md, threshold_abs=0.5, labels=comps)
    peaks = peaks[dist[tuple(peaks.T)] >= 0.2 * dist.max()]
    markers = np.zeros(binary.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    ws = watershed(-dist, markers, mask=binary > 0)
    final = run(clump, [CLUMP_SEG, {"step": "roi"}, {"step": "clean"}, {"step": "separate"}])
    fig, ax = panels(5, 2.3, 2.3)
    ax[0].imshow(rgb(clump))
    ax[0].set_title("(a) Touching objects")
    ax[1].imshow(binary, cmap="gray")
    ax[1].set_title("(b) Mask: seeds joined")
    ax[2].imshow(dist, cmap="magma")
    ax[2].plot(peaks[:, 1], peaks[:, 0], "c+", ms=8, mew=1.5)
    ax[2].set_title(f"(c) Distance transform\n+ maxima (d_min = {md} px)")
    ax[3].imshow(np.ma.masked_equal(ws, 0), cmap="tab10", interpolation="nearest")
    ax[3].set_title("(d) Marker watershed")
    ax[4].imshow(rgb(clump))
    for oid in range(1, int(final.labels.max()) + 1):
        m = (final.labels == oid).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            ax[4].plot(*np.vstack([c[:, 0], c[:1, 0]]).T, lw=1.2, color="yellow")
    n_final = len(np.unique(final.labels)) - 1
    ax[4].set_title(f"(e) After merge rules: {n_final} objects")
    save(fig, "separation.png")


# ── 3. Morphometry ──────────────────────────────────────────────────────────

def one_object(img, oid=5):
    ctx = run(img, [SEG, {"step": "roi"}, {"step": "clean"}, {"step": "label"},
                    {"step": "filter", "params": {"exclude_border": True}}])
    m = (ctx.labels == oid).astype(np.uint8)
    sl = crop_box(ctx.labels, [oid], pad=25)
    return m[sl], img[sl]


def fig_morphometry(img):
    m, patch = one_object(img)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    rect = cv2.boxPoints(cv2.minAreaRect(cnt))
    ell = cv2.fitEllipse(cnt)
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    fig, ax = panels(3, 2.8, 2.8)
    ax[0].imshow(rgb(patch))
    ax[0].plot(*np.vstack([cnt[:, 0], cnt[:1, 0]]).T, color="#FFD700", lw=1)
    ax[0].plot(*np.vstack([rect, rect[:1]]).T, color="#D73027", lw=1)
    ax[0].plot(cx, cy, "w+", ms=8)
    ax[0].set_title("(a) Contour, minimum-area\nrectangle (length × width)")
    ax[1].imshow(rgb(patch))
    ax[1].plot(*np.vstack([hull[:, 0], hull[:1, 0]]).T, color="#1A9850", lw=1.2)
    ax[1].plot(*np.vstack([cnt[:, 0], cnt[:1, 0]]).T, color="#FFD700", lw=0.8)
    ax[1].set_title("(b) Convex hull\n(solidity, convexity)")
    ax[2].imshow(rgb(patch))
    t = np.linspace(0, 2 * np.pi, 200)
    (ex, ey), (ma, mi), ang = ell
    a = np.deg2rad(ang)
    x = ex + ma / 2 * np.cos(t) * np.cos(a) - mi / 2 * np.sin(t) * np.sin(a)
    y = ey + ma / 2 * np.cos(t) * np.sin(a) + mi / 2 * np.sin(t) * np.cos(a)
    ax[2].plot(x, y, color="#2166AC", lw=1.2)
    pts = cnt[:, 0].astype(float)
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    for k, c in ((r.argmin(), "#D73027"), (r.argmax(), "#1A9850")):
        ax[2].plot([cx, pts[k, 0]], [cy, pts[k, 1]], color=c, lw=1)
    ax[2].set_title("(c) Fitted ellipse, min/max\nradius from the centroid")
    save(fig, "morphometry.png")


# ── 4. Elliptic Fourier descriptors ─────────────────────────────────────────

def fig_efd(img):
    m, _ = one_object(img)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    fig, ax = panels(5, 2.0, 2.0)
    for a, n in zip(ax, (1, 2, 4, 8, 20)):
        c = efd.efd(cnt, n)
        pts = efd.contour_points(c, 200) + cnt.mean(0) - efd.contour_points(c, 200).mean(0)
        a.plot(*np.vstack([cnt, cnt[:1]]).T, color="#BBBBBB", lw=2)
        a.plot(*np.vstack([pts, pts[:1]]).T, color="#D73027", lw=1.2)
        a.set_aspect("equal")
        a.invert_yaxis()
        a.set_title(f"{n} harmonic{'s' if n > 1 else ''}")
    save(fig, "efd_harmonics.png")


# ── 5. Color ────────────────────────────────────────────────────────────────

def fig_color(img):
    ctx = run(img, [SEG, {"step": "roi"}, {"step": "clean"}, {"step": "label"},
                    {"step": "filter", "params": {"exclude_border": True}},
                    {"step": "color", "params": {"n_colors": 3}}])
    oid = 5
    sl = crop_box(ctx.labels, [oid], pad=25)
    m = (ctx.labels[sl] == oid).astype(np.uint8)
    trim = int(np.clip(round(0.015 * np.sqrt(m.sum())), 1, 3))
    inner = cv2.erode(np.pad(m, 1), np.ones((3, 3), np.uint8), iterations=trim)[1:-1, 1:-1]
    ring = (m > 0) & (inner == 0)
    show = rgb(img[sl]).copy()
    show[ring] = (255, 0, 255)
    rows = [r for r in ctx.tables["object_colors"] if r["object_id"] == oid]
    fig, ax = panels(3, 2.9, 2.6)
    ax[0].imshow(show)
    ax[0].set_title(f"(a) Edge removed\n({trim} px ring, magenta)")
    ax[1].imshow(rgb(ctx.images["color"][sl]))
    ax[1].set_title("(b) Each pixel replaced\nby its k-means center")
    ax[2].set_axis_on()
    left = 0.0
    for r in rows:
        ax[2].barh(0, r["pct"], left=left, color=r["hex"], edgecolor="white")
        ax[2].text(left + r["pct"] / 2, 0, f"{r['pct']:.0f}%", ha="center", va="center", fontsize=7,
                   color="white" if r["L"] < 60 else "black")
        ax[2].text(left + r["pct"] / 2, -0.62, f"L*{r['L']:.0f}\na*{r['a']}\nb*{r['b']}", ha="center",
                   va="top", fontsize=6)
        left += r["pct"]
    ax[2].set_xlim(0, 100)
    ax[2].set_ylim(-1.6, 0.6)
    ax[2].set_yticks([])
    ax[2].set_xticks([])
    for sp in ax[2].spines.values():
        sp.set_visible(False)
    ax[2].set_title("(c) Dominant colors (k = 3)")
    save(fig, "color.png")


# ── 6. Distances ────────────────────────────────────────────────────────────

def fig_distances(img):
    from scipy.spatial import Delaunay
    ctx = run(img, [SEG, {"step": "roi"}, {"step": "clean"}, {"step": "label"},
                    {"step": "filter", "params": {"exclude_border": True}},
                    {"step": "distances", "params": {"neighbors": "voronoi"}}])
    from fenotit.core.measurements.distances import _centroids, _edge_on_line
    cents = _centroids(ctx.labels)
    ids = sorted(cents)
    pts = np.array([cents[i] for i in ids])
    fig, ax = panels(2, 3.6, 3.6)
    ax[0].imshow(rgb(img))
    tri = Delaunay(pts)
    ax[0].triplot(pts[:, 0], pts[:, 1], tri.simplices, color="#FFD700", lw=0.6)
    ax[0].plot(pts[:, 0], pts[:, 1], "o", color="#D73027", ms=2)
    ax[0].set_title("(a) Delaunay neighbors of the centroids")
    a, b = ctx.tables["distances"][0]["object_a"], ctx.tables["distances"][0]["object_b"]
    sl = crop_box(ctx.labels, [a, b], pad=30)
    ax[1].imshow(rgb(img[sl]))
    ca, cb = cents[a], cents[b]
    p1, p2 = _edge_on_line(ctx.labels, a, b, ca, cb)
    oy, ox = sl[0].start, sl[1].start
    ax[1].plot([ca[0] - ox, cb[0] - ox], [ca[1] - oy, cb[1] - oy], color="#BBBBBB", lw=1, ls="--")
    ax[1].plot([p1[0] - ox, p2[0] - ox], [p1[1] - oy, p2[1] - oy], color="#D73027", lw=2)
    ax[1].set_title("(b) Edge-to-edge distance along\nthe line between centroids")
    save(fig, "distances.png")


if __name__ == "__main__":
    image = load_image(str(IMAGE))
    if CALIBRATION.exists():
        fig_lens()
        fig_colorcard()
    else:
        print(f"Sin {CALIBRATION}: no se generan lens.png ni colorcard.png")
    fig_segmentation(image)
    fig_separation()
    fig_morphometry(image)
    fig_efd(image)
    fig_color(image)
    fig_distances(image)
