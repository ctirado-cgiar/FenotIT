"""Paper figures for the image corrections: lens distortion (chessboard), perspective and
scale (ArUco markers) and color (color card). Each experiment starts from a real photo of
bean seeds, degrades it in a controlled way and measures the seeds with and without the
FenotIT correction, so the error that each correction removes is known.

    python docs/paper/corrections_figures.py            (writes docs/paper/out/*)
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fenotit.core.corrections import colorcard as cc  # noqa: E402
from fenotit.core.corrections import distortion as D  # noqa: E402
from fenotit.core.corrections import pipeline as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
IMG = ROOT / "tests" / "images"
BLUE, RED, GRAY = "#2166AC", "#D6604D", "#7F7F7F"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
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


def zoom(ax, src_ax, img, box, color="white"):
    x0, y0, x1, y1 = box
    ax.imshow(rgb(img[y0:y1, x0:x1]))
    ax.set_axis_off()
    src_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1))


def label(ax, s):
    ax.set_title(s, loc="left")


# ── 1. Lens distortion ──────────────────────────────────────────────────────

W, H = 1600, 1200
K = np.array([[1300.0, 0, W / 2], [0, 1300.0, H / 2], [0, 0, 1]])
DIST = np.array([-0.20, 0.05, 0.0, 0.0, 0.0])       # strong barrel distortion of a wide-angle lens


def distort(ideal):
    """Image that a lens with DIST would record of the scene `ideal`."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, DIST, P=K).reshape(H, W, 2)
    bg = tuple(int(v) for v in np.median(ideal.reshape(-1, 3), 0))
    return cv2.remap(ideal, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=bg)


def chessboard_views(folder: Path, n=24):
    sq, cols, rows = 90, 8, 7                          # 7 x 6 inner corners
    board = np.full(((rows + 2) * sq, (cols + 2) * sq), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 0
    board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(7)
    bh, bw = board.shape[:2]
    paths = []
    for k in range(n):
        rv = rng.normal(0, 0.25, 3)
        R, _ = cv2.Rodrigues(rv)
        t = np.array([rng.uniform(-420, 420), rng.uniform(-300, 300), rng.uniform(1250, 1900)])
        obj = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
        X = np.hstack([obj - [bw / 2, bh / 2], np.zeros((4, 1))])
        img_pts, _ = cv2.projectPoints(X, rv, t, K, None)
        Hm = cv2.getPerspectiveTransform(obj, img_pts.reshape(4, 2).astype(np.float32))
        view = cv2.warpPerspective(board, Hm, (W, H), borderValue=(200, 200, 200))
        p = folder / f"board_{k:02d}.jpg"
        cv2.imwrite(str(p), distort(view))
        paths.append(str(p))
    return paths


def fig_lens():
    src = cv2.imread(str(IMG / "23379.jpg"))
    h0, w0 = src.shape[:2]
    crop = src[(h0 - int(w0 * 0.75)) // 2:(h0 + int(w0 * 0.75)) // 2]
    ideal = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
    dist_img = distort(ideal)
    with tempfile.TemporaryDirectory() as tmp:
        paths = chessboard_views(Path(tmp))
        res = D.calibrate(paths, 7, 6)
        board_example = cv2.imread(paths[3])
    cfg = P.Distortion(True, res.mtx.tolist(), res.dist.tolist(), res.rms_error, "synthetic", [W, H])
    fixed = P.undistort(dist_img, cfg)

    # per-seed area relative to the centre of the image, before and after
    lab_i, ids_i, a_i, c_i = seeds(ideal)
    lab_d, ids_d, a_d, _ = seeds(dist_img)
    lab_f, ids_f, a_f, _ = seeds(fixed)
    ids_d, ids_f = set(ids_d), set(ids_f)
    mtx, dc = res.mtx, res.dist
    new_mtx, (rx, ry, _, _) = cv2.getOptimalNewCameraMatrix(mtx, dc, (W, H), 1, (W, H))
    pts = np.float32([c_i[i] for i in ids_i]).reshape(-1, 1, 2)
    # where each ideal seed lands in the distorted photo and in the corrected one
    norm = cv2.undistortPoints(pts, K, None)
    xy = norm.reshape(-1, 2)
    r2 = (xy ** 2).sum(1)
    radial = 1 + DIST[0] * r2 + DIST[1] * r2 ** 2
    pd = (xy * radial[:, None]) @ K[:2, :2].T + K[:2, 2]
    pf = cv2.undistortPoints(pd.reshape(-1, 1, 2).astype(np.float32), mtx, dc, P=new_mtx).reshape(-1, 2) - [rx, ry]
    r = np.linalg.norm((pts.reshape(-1, 2) - [W / 2, H / 2]) / [W / 2, H / 2], axis=1) / np.sqrt(2)
    rows = []
    for k, i in enumerate(ids_i):
        xd, yd = pd[k].round().astype(int)
        xf, yf = pf[k].round().astype(int)
        jd = lab_d[yd, xd] if 0 <= yd < H and 0 <= xd < W else 0
        jf = lab_f[yf, xf] if 0 <= yf < lab_f.shape[0] and 0 <= xf < lab_f.shape[1] else 0
        if jd in ids_d and jf in ids_f:
            rows.append((r[k], a_i[i], a_d[jd], a_f[jf]))
    rows = np.array(rows)
    centre = rows[:, 0] < 0.25                          # scale set on the centre of the photo, as a user would
    kd = np.median(rows[centre, 2] / rows[centre, 1])
    kf = np.median(rows[centre, 3] / rows[centre, 1])
    err_d = 100 * (rows[:, 2] / kd / rows[:, 1] - 1)
    err_f = 100 * (rows[:, 3] / kf / rows[:, 1] - 1)
    for rr, ed, ef in zip(rows[:, 0], err_d, err_f):
        rows_out.append({"experiment": "lens", "x": round(rr, 3), "before": round(ed, 2), "after": round(ef, 2)})

    fig = plt.figure(figsize=(9.6, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25], hspace=0.3, wspace=0.25)
    a0 = fig.add_subplot(gs[0, 0])
    a0.imshow(rgb(board_example))
    a0.set_axis_off()
    label(a0, f"A  Chessboard photo (1 of {len(paths)})")
    a1 = fig.add_subplot(gs[0, 1])
    a1.imshow(rgb(dist_img))
    a1.set_axis_off()
    label(a1, "B  Photo with barrel distortion")
    a2 = fig.add_subplot(gs[0, 2])
    a2.imshow(rgb(fixed))
    a2.set_axis_off()
    label(a2, "C  Corrected by FenotIT")
    box_d = (W - 330, H - 250, W - 10, H - 10)
    a3 = fig.add_subplot(gs[0, 3])
    zoom(a3, a1, dist_img, box_d, "yellow")
    label(a3, "D  Corner of B")
    fh, fw = fixed.shape[:2]
    bx = np.float32([[box_d[0], box_d[1]], [box_d[2], box_d[1]], [box_d[2], box_d[3]], [box_d[0], box_d[3]]])
    q = cv2.undistortPoints(bx.reshape(-1, 1, 2), mtx, dc, P=new_mtx).reshape(-1, 2) - [rx, ry]
    box_f = (max(int(q[:, 0].min()), 0), max(int(q[:, 1].min()), 0),
             min(int(q[:, 0].max()), fw), min(int(q[:, 1].max()), fh))
    a4 = fig.add_subplot(gs[1, 0])
    zoom(a4, a2, fixed, box_f, "yellow")
    label(a4, "E  Same corner in C")
    a5 = fig.add_subplot(gs[1, 1:3])
    a5.scatter(rows[:, 0], err_d, s=6, color=RED, alpha=0.6, label="Distorted photo")
    a5.scatter(rows[:, 0], err_f, s=6, color=BLUE, alpha=0.6, label="After correction")
    a5.axhline(0, color="#333", lw=0.6)
    a5.set_xlabel("Distance from image centre (0 = centre, 1 = corner)")
    a5.set_ylabel("Seed area error (%)")
    a5.legend(frameon=False, fontsize=7)
    label(a5, "F  Error in seed area across the photo")
    a6 = fig.add_subplot(gs[1, 3])
    a6.set_axis_off()
    txt = (f"Calibration\n{len(paths)} chessboard photos\nRMS reprojection error\n{res.rms_error:.2f} px\n\n"
           f"k1 true {DIST[0]:+.3f}\nk1 estimated {res.dist.ravel()[0]:+.3f}\n\n"
           f"Mean |error| at corners (r > 0.7)\nbefore {np.abs(err_d[rows[:, 0] > 0.7]).mean():.1f} %\n"
           f"after {np.abs(err_f[rows[:, 0] > 0.7]).mean():.1f} %")
    a6.text(0.02, 0.98, txt, va="top", fontsize=7.5, family="DejaVu Sans")
    fig.savefig(OUT / "corr_lens.png")
    plt.close(fig)
    print("lens: rms", res.rms_error, "k1", res.dist.ravel()[:2])


# ── 2. Perspective and scale with ArUco markers ─────────────────────────────

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
    fig = plt.figure(figsize=(9.6, 5.8))
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb(scene))
    ax.set_axis_off()
    label(ax, "A  Tray with four ArUco markers")
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(rgb(tilted))
    ax1.set_axis_off()
    label(ax1, "B  Photographed at 40° tilt")
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(rgb(rect))
    ax2.set_axis_off()
    label(ax2, "C  Rectified by FenotIT")
    # zoom on the same seeds near the far edge
    q = cv2.perspectiveTransform(np.float32([[[700, 260]]]), Ht).reshape(2).round().astype(int)
    zb = (q[0] - 120, q[1] - 45, q[0] + 120, q[1] + 45)
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_axis_off()
    sub = fig.add_gridspec(2, 1, left=gs[0, 3].get_position(fig).x0, right=gs[0, 3].get_position(fig).x1,
                           bottom=gs[0, 3].get_position(fig).y0, top=gs[0, 3].get_position(fig).y1, hspace=0.25)
    z1 = fig.add_subplot(sub[0])
    zoom(z1, ax1, tilted, zb, "yellow")
    label(z1, "D  Far edge in B and C")
    m = cfg.margin_px
    sc = (rect.shape[1] - 2 * m) / (centers[1][0] - centers[0][0])
    rx = int(m + (700 - centers[0][0]) * sc)
    ry = int(m + (260 - centers[0][1]) * sc)
    zb2 = (rx - 120, ry - 70, rx + 120, ry + 70)
    z2 = fig.add_subplot(sub[1])
    zoom(z2, ax2, rect, zb2, "yellow")
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.plot(res[:, 0], res[:, 1], "o-", color=RED, ms=3.5, label="Area, one scale for all photos")
    ax4.plot(res[:, 0], res[:, 2], "o-", color=BLUE, ms=3.5, label="Area, ArUco rectification")
    ax4.plot(res[:, 0], res[:, 3], "s--", color=RED, ms=3, lw=0.9, label="Length, one scale")
    ax4.plot(res[:, 0], res[:, 4], "s--", color=BLUE, ms=3, lw=0.9, label="Length, ArUco")
    ax4.axhline(0, color="#333", lw=0.6)
    ax4.set_xlabel("Camera tilt (°)")
    ax4.set_ylabel("Error of mean seed size (%)")
    ax4.legend(frameon=False, fontsize=6.5)
    label(ax4, "E  Tilted camera")
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax5.plot(rd[:, 0], rd[:, 1], "o-", color=RED, ms=3.5, label="One scale for all photos")
    ax5.plot(rd[:, 0], rd[:, 2], "o-", color=BLUE, ms=3.5, label="Scale from ArUco in each photo")
    ax5.axhline(0, color="#333", lw=0.6)
    ax5.set_xlabel("Camera distance (relative to the first photo)")
    ax5.set_ylabel("Error of mean seed area (%)")
    ax5.legend(frameon=False, fontsize=6.5)
    label(ax5, "F  Camera moved closer or farther")
    fig.savefig(OUT / "corr_aruco.png")
    plt.close(fig)
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


# ── 3. Color card ───────────────────────────────────────────────────────────

LIGHTS = [("Warm light", (1.00, 0.80, 0.50)), ("Cool light", (0.72, 0.90, 1.12)),
          ("Green cast", (0.80, 1.05, 0.78)), ("Dim light", (0.45, 0.45, 0.45))]


MIX = np.array([[0.92, 0.06, 0.02], [0.05, 0.90, 0.05], [0.02, 0.08, 0.90]], np.float32)


def cast(img, gains, seed=0):
    """Another light source and camera: channel gains and crosstalk in linear RGB, a light
    falloff of 15 % across the tray, a slightly different tone curve, sensor noise and JPEG."""
    rng = np.random.default_rng(seed)
    lin = (img[..., ::-1].astype(np.float32) / 255) ** 2.2
    lin = lin @ MIX.T * np.float32(gains)
    h, w = lin.shape[:2]
    fall = np.linspace(1.0, 0.85, w, dtype=np.float32)[None, :, None]
    lin = lin * fall
    out = np.clip(lin, 0, 1) ** (1 / 2.4)
    out = out * 255 + rng.normal(0, 2.0, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)[..., ::-1]
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def fig_color():
    from skimage.color import deltaE_ciede2000, rgb2lab
    src = cv2.imread(str(IMG / "card_cameratrax.jpg"))
    base = cv2.resize(src, (1600, 1200), interpolation=cv2.INTER_AREA)
    ref, info = cc.correct(base)
    card = info["card"]
    xs, ys = card.centers[..., 0], card.centers[..., 1]
    pad = card.chip
    keep = np.ones(base.shape[:2], bool)
    keep[int(ys.min() - pad):int(ys.max() + pad), int(xs.min() - pad):int(xs.max() + pad)] = False
    keep[:, 1150:] = False                                      # white panel at the right
    lab, ids, _, _ = seeds(base, keep)

    def seed_lab(img):
        L = rgb2lab(rgb(img).astype(np.float32) / 255)
        return np.array([L[lab == i].mean(0) for i in ids])
    ref_lab = seed_lab(ref)
    res, shots = [], []
    for name, g in LIGHTS:
        c = cast(base, g)
        fixed, inf = cc.correct(c)
        before = deltaE_ciede2000(ref_lab, seed_lab(c))
        after = deltaE_ciede2000(ref_lab, seed_lab(fixed))
        res.append((name, before.mean(), before.std(), after.mean(), after.std(), inf.get("mean_error_rgb")))
        shots.append((name, c, fixed))
        rows_out.append({"experiment": "color", "x": name, "before": round(before.mean(), 2), "after": round(after.mean(), 2)})

    n = len(LIGHTS)
    fig = plt.figure(figsize=(9.6, 5.6))
    gs = fig.add_gridspec(3, n + 1, height_ratios=[1, 1, 1.3], hspace=0.22, wspace=0.08)
    a = fig.add_subplot(gs[0, 0])
    a.imshow(rgb(ref))
    a.set_axis_off()
    label(a, "A  Reference")
    zb = (int(xs.min() - pad), int(ys.min() - pad), int(xs.max() + pad), int(ys.max() + pad))
    for k, (name, c, fixed) in enumerate(shots):
        ax = fig.add_subplot(gs[0, k + 1])
        ax.imshow(rgb(c))
        ax.set_axis_off()
        label(ax, f"{chr(66 + k)}  {name}")
        ax2 = fig.add_subplot(gs[1, k + 1])
        ax2.imshow(rgb(fixed))
        ax2.set_axis_off()
        if k == 0:
            ax2.text(-0.06, 0.5, "Corrected", transform=ax2.transAxes, rotation=90, va="center", ha="right")
    # zoom: seeds and card under warm light, before and after
    name, c, fixed = shots[0]
    zx0, zy0 = 330, 330
    za = fig.add_subplot(gs[1, 0])
    za.imshow(np.hstack([rgb(c[zy0:zy0 + 260, zx0:zx0 + 200]), np.full((260, 8, 3), 255, np.uint8),
                         rgb(fixed[zy0:zy0 + 260, zx0:zx0 + 200])]))
    za.set_axis_off()
    label(za, "F  Seeds in B, before | after")
    ab = fig.add_subplot(gs[2, 0:3])
    x = np.arange(n)
    r = np.array([[v[1], v[2], v[3], v[4]] for v in res])
    ab.bar(x - 0.18, r[:, 0], 0.36, yerr=r[:, 1], color=RED, capsize=2, label="Without correction")
    ab.bar(x + 0.18, r[:, 2], 0.36, yerr=r[:, 3], color=BLUE, capsize=2, label="With color card")
    ab.axhline(2.3, color=GRAY, lw=0.8, ls="--")
    ab.text(-0.45, 2.6, "just noticeable difference (ΔE ≈ 2.3)", ha="left", fontsize=6.5, color=GRAY)
    ab.set_xticks(x, [v[0] for v in res])
    ab.set_ylabel("Seed color difference\nfrom reference (ΔE₀₀)")
    ab.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2)
    label(ab, "G  Seed color under each light")
    ac = fig.add_subplot(gs[2, 3:])
    ac.imshow(np.hstack([rgb(c[zb[1]:zb[3], zb[0]:zb[2]]), np.full((zb[3] - zb[1], 8, 3), 255, np.uint8),
                         rgb(fixed[zb[1]:zb[3], zb[0]:zb[2]])]))
    ac.set_axis_off()
    label(ac, "H  Card in B, before | after")
    fig.savefig(OUT / "corr_color.png")
    plt.close(fig)
    print("color", [(v[0], round(v[1], 1), round(v[3], 1)) for v in res])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_color()
    fig_aruco()
    fig_lens()
    with open(OUT / "corrections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "x", "before", "after"])
        w.writeheader()
        w.writerows(rows_out)


if __name__ == "__main__":
    main()
