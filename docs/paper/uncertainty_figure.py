"""Validación del modelo de incertidumbre del área (Fig. de incertidumbre del paper).

Siluetas reales de semillas (tests/data/images/01.jpg) se vuelven a digitalizar con giro y
desplazamiento subpíxel al azar, a varias resoluciones. Para cada semilla y resolución se
compara la desviación relativa observada del área con la predicha, √(P/12)/A."""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))
import figure_style as style  # noqa: E402
from fenotit.core import uncertainty as U
from fenotit.core.image_io import load_image as imread

OUT = Path(__file__).parent / "out"
SCALES = (1.0, 0.5, 0.3, 0.2, 0.12, 0.08)
REPS = 40


def seeds(n=30):
    from fenotit.core import pipeline
    img = imread(str(ROOT / "tests/data/images/01.jpg"))
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
    style.apply()
    fig, ax = plt.subplots(1, 2, figsize=(style.ONE_AND_HALF, 62 * style.MM), gridspec_kw={"wspace": 0.42})
    ax[0].scatter(pred * 100, obs * 100, s=7, facecolors="none", edgecolors=style.BLUE, linewidths=0.5)
    lim = [min(pred.min(), obs.min()) * 80, max(pred.max(), obs.max()) * 120]
    ax[0].plot(lim, lim, color=style.BLACK, ls="--", lw=0.7)
    ax[0].set(xscale="log", yscale="log", xlabel="Predicted relative SD of area (%)",
              ylabel="Observed relative SD of area (%)", title="(a)")
    areas = np.logspace(1, 4, 100)
    ax[1].scatter(area, obs * 100, s=7, facecolors="none", edgecolors=style.BLUE, linewidths=0.5)
    ax[1].plot(areas, [U.area_rel_sd(x, U.CIRCLE_K * np.sqrt(x)) * 100 for x in areas], color=style.BLACK, lw=0.8)
    for target in (0.01, 0.005):
        ax[1].axvline(U.min_area_px(target), color=style.GRAY, lw=0.6, ls=":")
        ax[1].text(U.min_area_px(target), 6 if target == 0.01 else 9, f" {target * 100:g} %", color=style.GRAY)
    ax[1].set(xscale="log", yscale="log", xlabel="Object area (px)", ylabel="Relative SD of area (%)", title="(b)")
    style.plain_log_labels(*ax)
    style.chart_axes(*ax)
    style.save(fig, OUT / "uncertainty.png")

if __name__ == "__main__":
    main()
