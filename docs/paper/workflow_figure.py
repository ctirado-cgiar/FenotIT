"""Fig. 1: de la foto a los datos en cinco pasos, con una foto real (semillas de frijol con
tarjeta de color, tests/images/card_cameratrax.jpg) y los resultados reales del pipeline."""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fenotit.core.analysis import objects
from fenotit.core.corrections import pipeline as P
from fenotit.core.pipeline import overlay

OUT = Path(__file__).parent / "out"
BLUE = "#2166AC"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def data():
    raw = cv2.imread(str(ROOT / "tests" / "images" / "card_cameratrax.jpg"))
    c = P.Corrections()
    c.color.enabled = True
    fixed, _ = P.apply(raw, c)
    H, W = raw.shape[:2]
    tray = (slice(0, H), slice(int(W * .06), int(W * .76)))           # bandeja con la tarjeta
    seeds = (slice(int(H * .26), int(H * .84)), slice(int(W * .13), int(W * .65)))
    crop = fixed[seeds].copy()
    seg = {"method": "threshold", "color_space": "LAB", "channel": 2, "auto": True,
           "refine": [{"op": "open", "size": 5, "iter": 1}, {"op": "fill_holes"}]}
    res = objects.run(crop, {"segmentation": seg, "measure_size": True, "measure_color": True, "n_colors": 3,
                             "exclude_border": True})
    folders = res.extra["step_folders"]
    view = lambda f: next(k for k, v in folders.items() if v == f)
    colors = overlay.resolve(None, auto=res.extra.get("contrast"))
    mask = res.step_images[view("mask")][..., 0] > 0
    seg_img = overlay.tint(crop, (mask * 255).astype(np.uint8), colors)
    morph = crop.copy()
    ov = dict(res.extra["overlays"][view("morphometry")])
    overlay.draw(morph, ov, colors)
    rows = res.extra["tables"]["objects"]
    h, w = morph.shape[:2]
    zoom = morph[int(h * .36):int(h * .70), int(w * .20):int(w * .64)]   # de cerca: largo, ancho y número
    return raw[tray], fixed[tray], seg_img, zoom, rows, res


def split(before, after):
    """Antes / después en diagonal."""
    h, w = before.shape[:2]
    out = after.copy()
    yy, xx = np.mgrid[:h, :w]
    m = xx / w + yy / h < 1.0
    out[m] = before[m]
    cv2.line(out, (w, 0), (0, h), (255, 255, 255), max(3, w // 150), cv2.LINE_AA)
    return out


def main():
    raw, fixed, seg_img, morph, rows, res = data()
    unit = "px"
    area = np.array([r[f"area_{unit}2"] for r in rows if f"area_{unit}2" in r], float)
    length = np.array([r[f"length_{unit}"] for r in rows if f"length_{unit}" in r], float)
    width = np.array([r[f"width_{unit}"] for r in rows if f"width_{unit}" in r], float)
    cols = [(r["mean_R"] / 255, r["mean_G"] / 255, r["mean_B"] / 255) for r in rows if "mean_R" in r]

    fig = plt.figure(figsize=(7.2, 2.0), dpi=300)
    steps = [("Photograph", "any camera or phone", rgb(raw)),
             ("Correct", "lens · perspective · color · scale", rgb(split(raw, fixed))),
             ("Segment", "threshold · by example · AI model", rgb(seg_img)),
             ("Measure", "count · size · shape · color", rgb(morph)),
             ("Data", "tables, figures, annotated images", None)]
    n = len(steps)
    left, gap, top = 0.012, 0.03, 0.78
    w = (1 - 2 * left - gap * (n - 1)) / n
    h = 0.66
    for i, (title, sub, img) in enumerate(steps):
        x = left + i * (w + gap)
        card = FancyBboxPatch((x, 0.03), w, 0.94, boxstyle="round,pad=0,rounding_size=0.02",
                              transform=fig.transFigure, fc="#F4F8FC", ec="#C9DCEF", lw=0.8, zorder=-10)
        fig.add_artist(card)
        fig.text(x + 0.012, top + 0.105, str(i + 1), ha="center", va="center", color="white", fontsize=7,
                 fontweight="bold", bbox=dict(boxstyle="circle,pad=0.25", fc=BLUE, ec="none"))
        fig.text(x + 0.03, top + 0.105, title, ha="left", va="center", color=BLUE, fontsize=8.5, fontweight="bold")
        fig.text(x + w / 2, 0.075, sub, ha="center", va="center", color="#444444", fontsize=5.6)
        ax = fig.add_axes([x + 0.008, 0.13, w - 0.016, h])
        ax.set_axis_off()
        if img is not None:
            ax.imshow(img)
        else:
            data_panel(fig, x, w, area, length, width, cols)
        if i < n - 1:                                     # flecha entre tarjetas, sin cruzar nada
            ax2 = x + w + 0.004
            fig.add_artist(Polygon([[ax2, 0.47], [ax2 + gap - 0.008, 0.5], [ax2, 0.53]], closed=True,
                                   transform=fig.transFigure, fc=BLUE, ec="none"))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig1_workflow.png", dpi=300)
    fig.savefig(OUT / "fig1_workflow.svg")
    print(OUT / "fig1_workflow.png", len(area), "objects")


def data_panel(fig, x, w, area, length, width, cols):
    """Una tabla corta y dos gráficos pequeños: lo que el usuario se lleva."""
    tab = fig.add_axes([x + 0.008, 0.555, w - 0.016, 0.235])
    tab.set_axis_off()
    head = ["ID", "length", "width", "area"]
    body = [[str(i + 1), f"{length[i]:.0f}", f"{width[i]:.0f}", f"{area[i]:.0f}"] for i in range(4)] + \
        [["…", "…", "…", "…"]]
    t = tab.table(cellText=body, colLabels=head, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(4.6)
    t.scale(1, 0.62)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#C9DCEF")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
    ax = fig.add_axes([x + 0.03, 0.15, (w - 0.05) * 0.55, 0.28])
    bp = ax.boxplot(area, widths=0.55, patch_artist=True, showfliers=False)
    bp["boxes"][0].set(facecolor="#C9DCEF", edgecolor=BLUE, lw=0.6)
    for k in ("whiskers", "caps", "medians"):
        for ln in bp[k]:
            ln.set(color=BLUE, lw=0.6)
    ax.scatter(np.random.default_rng(1).normal(1, 0.06, len(area)), area, s=2, c=cols[:len(area)] if cols else BLUE,
               zorder=3, linewidths=0)
    ax.set_xticks([])
    ax.set_title("area", fontsize=5, pad=1)
    ax.tick_params(labelsize=4, length=1.5, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax2 = fig.add_axes([x + 0.03 + (w - 0.05) * 0.62, 0.15, (w - 0.05) * 0.38, 0.28])
    ax2.scatter(length, width, s=2.5, c=cols[:len(length)] if cols else BLUE, linewidths=0)
    ax2.set_title("length × width", fontsize=5, pad=1)
    ax2.tick_params(labelsize=4, length=1.5, pad=1)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)


if __name__ == "__main__":
    main()
