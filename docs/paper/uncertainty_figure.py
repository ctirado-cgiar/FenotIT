"""Validación del modelo de incertidumbre del área (Fig. de incertidumbre del paper).

Siluetas reales de semillas (tests/images/01.jpg) se vuelven a digitalizar con giro y
desplazamiento subpíxel al azar, a varias resoluciones. Para cada semilla y resolución se
compara la desviación relativa observada del área con la predicha, √(P/12)/A."""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fenotit.core import uncertainty as U
from fenotit.core.image_io import load_image as imread

OUT = Path(__file__).parent / "out"
SCALES = (1.0, 0.5, 0.3, 0.2, 0.12, 0.08)
REPS = 40


def seeds(n=30):
    from fenotit.core import pipeline
    img = imread(str(ROOT / "tests/images/01.jpg"))
    ctx = pipeline.run(img, [{"step": "otsu", "params": {"color_space": "LAB", "channel": 2}},
                             {"step": "clean", "params": {"open_iterations": 1}},
                             {"step": "label"}, {"step": "filter", "params": {"area_min": 300}}])
    lab = ctx.labels
    ids, a = np.unique(lab[lab > 0], return_counts=True)
    med = np.median(a)
    keep = [i for i, x in zip(ids, a) if 0.6 * med < x < 1.4 * med][:n]
    out = []
    for i in keep:
        ys, xs = np.nonzero(lab == i)
        p = 20
        crop = (lab[max(0, ys.min() - p):ys.max() + p, max(0, xs.min() - p):xs.max() + p] == i).astype(np.float32)
        out.append(cv2.GaussianBlur(crop, (0, 0), 1.0))       # contorno suave (forma "verdadera")
    return out


def digitize(shape, s, rng):
    h, w = shape.shape
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, rng.uniform(0, 360), s)
    d = int(np.hypot(w, h) * s) + 6                      # lienzo que cabe girado
    M[:, 2] += np.array([d / 2, d / 2]) - np.array(c) + rng.random(2)
    size = (d, d)
    return cv2.warpAffine(shape, M, size, flags=cv2.INTER_LINEAR) > 0.5


def main():
    rng = np.random.default_rng(7)
    obs, pred, area = [], [], []
    for shp in seeds():
        for s in SCALES:
            a, per = [], []
            for _ in range(REPS):
                m = digitize(shp, s, rng).astype(np.uint8)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                c = max(cnts, key=cv2.contourArea)
                a.append(m.sum())
                per.append(cv2.arcLength(cv2.approxPolyDP(c, 1.0, True), True))
            A, P = np.mean(a), np.mean(per)
            obs.append(np.std(a) / A); pred.append(U.area_rel_sd(A, P)); area.append(A)
    obs, pred, area = map(np.array, (obs, pred, area))
    ratio = obs / pred
    print(f"n={len(obs)}  observed/predicted median {np.median(ratio):.2f} "
          f"(IQR {np.percentile(ratio, 25):.2f}-{np.percentile(ratio, 75):.2f})")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), dpi=300)
    ax[0].loglog(pred * 100, obs * 100, "o", ms=3, color="#2166AC", alpha=.6)
    lim = [min(pred.min(), obs.min()) * 80, max(pred.max(), obs.max()) * 120]
    ax[0].plot(lim, lim, "k--", lw=.8)
    ax[0].set(xlabel="Predicted relative SD of area (%)", ylabel="Observed relative SD of area (%)",
              title="(a) Model vs. re-digitized seeds")
    aa = np.logspace(1, 4, 100)
    ax[1].loglog(area, obs * 100, "o", ms=3, color="#2166AC", alpha=.6, label="Seeds")
    ax[1].loglog(aa, [U.area_rel_sd(x, U.CIRCLE_K * np.sqrt(x)) * 100 for x in aa], "k-", lw=1,
                 label="Model, circle")
    for e in (0.01, 0.005):
        ax[1].axvline(U.min_area_px(e), color="gray", lw=.6, ls=":")
        ax[1].text(U.min_area_px(e), ax[1].get_ylim()[1] if False else 8, f" {e*100:g} %", fontsize=7, color="gray")
    ax[1].set(xlabel="Object area (px)", ylabel="Relative SD of area (%)", title="(b) Area needed for a target error")
    ax[1].legend(fontsize=7, frameon=False)
    for a_ in ax:
        a_.tick_params(labelsize=7); a_.title.set_size(9); a_.xaxis.label.set_size(8); a_.yaxis.label.set_size(8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "uncertainty.png")
    print(OUT / "uncertainty.png")


if __name__ == "__main__":
    main()
