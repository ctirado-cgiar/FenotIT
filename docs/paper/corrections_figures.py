"""Paper figure for the perspective and scale correction with ArUco markers. A real photo of
bean seeds is placed on a board with four markers, photographed with a simulated camera at
several tilts and distances, and the seeds are measured with one fixed scale and with the
FenotIT rectification, so the error removed by the correction is known.

    python docs/paper/corrections_figures.py            (writes docs/paper/out/corr_aruco.png)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))
import figure_style as style  # noqa: E402
from fenotit.core.corrections import pipeline as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
IMG = ROOT / "tests" / "data" / "images"
BLUE, BLACK, GRAY = style.BLUE, style.BLACK, style.GRAY
rows_out: list[dict] = []


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def seeds(img, keep=None):
    """Seed mask on the blue tray (Otsu on Cb), labels and stats of seed-sized components."""
    cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 2]
    t, _ = cv2.threshold(cb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((cb < t) & (keep if keep is not None else True)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    a = st[:, 4].astype(float)
    a[0] = 0
    med = np.median(a[a > 100]) if (a > 100).any() else 0
    good = [i for i in range(1, n) if 0.5 * med < a[i] < 1.6 * med]
    return lab, good, a, cen


def lengths(lab, ids):
    out = []
    for i in ids:
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (_, _), (w, h), _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        out.append((max(w, h), min(w, h)))
    return np.array(out)


def zoom(ax, src_ax, img, box, color=style.WHITE):
    x0, y0, x1, y1 = box
    ax.imshow(rgb(img[y0:y1, x0:x1]))
    style.photo_axes(ax)
    src_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=0.8))


def label(ax, s):
    ax.set_title(s, loc="left")


# ── Perspective and scale with ArUco markers ─────────────────────────────

MM_PER_PX = 0.10                                         # board units: 0.1 mm per board pixel


def aruco_board():
    src = cv2.imread(str(IMG / "23505.jpg"))
    h0, w0 = src.shape[:2]
    s = min(h0, w0) - 400
    scene = cv2.resize(src[(h0 - s) // 2:(h0 + s) // 2, (w0 - s) // 2:(w0 + s) // 2], (1400, 1400),
                       interpolation=cv2.INTER_AREA)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    size, pad = 110, 18
    centers = {0: (90, 90), 1: (1310, 90), 2: (1310, 1310), 3: (90, 1310)}
    keep = np.ones(scene.shape[:2], bool)
    for i, (cx, cy) in centers.items():
        mk = cv2.aruco.generateImageMarker(d, i, size)
        x0, y0 = cx - size // 2, cy - size // 2
        scene[y0 - pad:y0 + size + pad, x0 - pad:x0 + size + pad] = 255
        scene[y0:y0 + size, x0:x0 + size] = cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR)
        keep[y0 - pad - 10:y0 + size + pad + 10, x0 - pad - 10:x0 + size + pad + 10] = False
    width_mm = (centers[1][0] - centers[0][0]) * MM_PER_PX
    height_mm = (centers[3][1] - centers[0][1]) * MM_PER_PX
    return scene, keep, width_mm, height_mm, centers


def photograph(scene, tilt_deg, distance=1.0, out=(1600, 1200)):
    """Pinhole camera looking at the flat board, tilted about the horizontal axis."""
    bh, bw = scene.shape[:2]
    f = 1300.0
    Kc = np.array([[f, 0, out[0] / 2], [0, f, out[1] / 2], [0, 0, 1]])
    th = np.radians(tilt_deg)
    R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    t = np.array([0, 0, 2000.0 * distance])
    obj = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    X = np.hstack([obj - [bw / 2, bh / 2], np.zeros((4, 1))])
    p = (Kc @ (R @ X.T + t[:, None])).T
    p = (p[:, :2] / p[:, 2:]).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(obj, p)
    bg = tuple(int(v) for v in np.median(scene[200:400, 200:400].reshape(-1, 3), 0))
    return cv2.warpPerspective(scene, Hm, out, borderValue=bg), Hm


def fig_aruco():
    scene, keep, wmm, hmm, centers = aruco_board()
    cfg = P.Perspective(True, 20, wmm, hmm)

    def measure(photo, Hm, mm_px):
        k = cv2.warpPerspective(keep.astype(np.uint8), Hm, (photo.shape[1], photo.shape[0])) > 0
        lab, ids, a, _ = seeds(photo, k)
        return a[ids].mean() * mm_px ** 2, lengths(lab, ids)[:, 0].mean() * mm_px

    ref_photo, Href = photograph(scene, 0)
    pts = np.float32([centers[i] for i in range(4)]).reshape(-1, 1, 2)
    pc = cv2.perspectiveTransform(pts, Href).reshape(4, 2)
    fixed_scale = wmm / np.linalg.norm(pc[1] - pc[0])       # one scale for all photos, set on the first
    # reference: the same seeds photographed perpendicular to the tray, with its exact scale
    true_area, true_len = measure(ref_photo, Href, fixed_scale)
    tilts = [0, 10, 20, 30, 40, 50]
    res = []
    for th in tilts:
        photo, Hm = photograph(scene, th)
        a_raw, l_raw = measure(photo, Hm, fixed_scale)
        rect, mm_px, err = P.rectify(photo, cfg)
        if rect is None:
            print("tilt", th, err)
            continue
        lab, ids, a, _ = seeds(rect, _rect_keep(rect, keep, cfg))
        a_cor = a[ids].mean() * mm_px ** 2
        l_cor = lengths(lab, ids)[:, 0].mean() * mm_px
        res.append((th, 100 * (a_raw / true_area - 1), 100 * (a_cor / true_area - 1),
                    100 * (l_raw / true_len - 1), 100 * (l_cor / true_len - 1)))
        rows_out.append({"experiment": "tilt", "x": th, "before": round(res[-1][1], 2), "after": round(res[-1][2], 2)})
    res = np.array(res)
    dists = [0.8, 0.9, 1.0, 1.2, 1.4, 1.6]
    rd = []
    for dd in dists:
        photo, Hm = photograph(scene, 0, dd)
        a_raw, _ = measure(photo, Hm, fixed_scale)
        rect, mm_px, err = P.rectify(photo, cfg)
        if rect is None:
            print("distance", dd, err)
            continue
        lab, ids, a, _ = seeds(rect, _rect_keep(rect, keep, cfg))
        rd.append((dd, 100 * (a_raw / true_area - 1), 100 * (a[ids].mean() * mm_px ** 2 / true_area - 1)))
        rows_out.append({"experiment": "distance", "x": dd, "before": round(rd[-1][1], 2), "after": round(rd[-1][2], 2)})
    rd = np.array(rd)

    tilted, Ht = photograph(scene, 40)
    rect, _, _ = P.rectify(tilted, cfg)
    style.apply()
    fig = plt.figure(figsize=(style.DOUBLE, 120 * style.MM))
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb(scene))
    style.photo_axes(ax)
    label(ax, "(a)")
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(rgb(tilted))
    style.photo_axes(ax1)
    label(ax1, "(b)")
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(rgb(rect))
    style.photo_axes(ax2)
    label(ax2, "(c)")
    # zoom on the same seeds near the far edge
    q = cv2.perspectiveTransform(np.float32([[[700, 260]]]), Ht).reshape(2).round().astype(int)
    zb = (q[0] - 120, q[1] - 45, q[0] + 120, q[1] + 45)
    ax3 = fig.add_subplot(gs[0, 3])
    style.photo_axes(ax3)
    sub = fig.add_gridspec(2, 1, left=gs[0, 3].get_position(fig).x0, right=gs[0, 3].get_position(fig).x1,
                           bottom=gs[0, 3].get_position(fig).y0, top=gs[0, 3].get_position(fig).y1, hspace=0.25)
    z1 = fig.add_subplot(sub[0])
    zoom(z1, ax1, tilted, zb, style.WHITE)
    label(z1, "(d)")
    m = cfg.margin_px
    sc = (rect.shape[1] - 2 * m) / (centers[1][0] - centers[0][0])
    rx = int(m + (700 - centers[0][0]) * sc)
    ry = int(m + (260 - centers[0][1]) * sc)
    zb2 = (rx - 120, ry - 70, rx + 120, ry + 70)
    z2 = fig.add_subplot(sub[1])
    zoom(z2, ax2, rect, zb2, style.WHITE)
    charts = gs[1, :].subgridspec(1, 2, wspace=0.35)
    ax4 = fig.add_subplot(charts[0])
    ax4.plot(res[:, 0], res[:, 1], "o-", color=BLACK, ms=3, mfc=style.WHITE, label="Area, one scale for all photos")
    ax4.plot(res[:, 0], res[:, 2], "o-", color=BLUE, ms=3, label="Area, ArUco rectification")
    ax4.plot(res[:, 0], res[:, 3], "s--", color=BLACK, ms=2.5, mfc=style.WHITE, label="Length, one scale")
    ax4.plot(res[:, 0], res[:, 4], "s--", color=BLUE, ms=2.5, label="Length, ArUco")
    ax4.axhline(0, color=GRAY, lw=0.6)
    ax4.set_xlabel("Camera tilt (°)")
    ax4.set_ylabel("Error of mean seed size (%)")
    ax4.legend(fontsize=6)
    label(ax4, "(e)")
    ax5 = fig.add_subplot(charts[1])
    ax5.plot(rd[:, 0], rd[:, 1], "o-", color=BLACK, ms=3, mfc=style.WHITE, label="One scale for all photos")
    ax5.plot(rd[:, 0], rd[:, 2], "o-", color=BLUE, ms=3, label="Scale from ArUco in each photo")
    ax5.axhline(0, color=GRAY, lw=0.6)
    ax5.set_xlabel("Camera distance (relative to the first photo)")
    ax5.set_ylabel("Error of mean seed area (%)")
    ax5.legend(fontsize=6)
    label(ax5, "(f)")
    style.chart_axes(ax4, ax5)
    style.save(fig, OUT / "corr_aruco.png")
    print("aruco tilt", res.round(2).tolist())
    print("aruco distance", rd.round(2).tolist())


def _rect_keep(rect, keep, cfg):
    """Mask of the rectified photo without the markers and the white border."""
    centers = {0: (90, 90), 1: (1310, 90), 2: (1310, 1310), 3: (90, 1310)}
    m = cfg.margin_px
    sc = (rect.shape[1] - 2 * m) / (centers[1][0] - centers[0][0])
    A = np.float32([[sc, 0, m - centers[0][0] * sc], [0, sc, m - centers[0][1] * sc]])
    k = cv2.warpAffine(keep.astype(np.uint8), A, (rect.shape[1], rect.shape[0]), borderValue=0) > 0
    return k



def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_aruco()
    with open(OUT / "corrections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "x", "before", "after"])
        w.writeheader()
        w.writerows(rows_out)


if __name__ == "__main__":
    main()
