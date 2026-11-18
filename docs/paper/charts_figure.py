"""Figura con ejemplos de los gráficos de FenotIT (ventanas «Comparar fotos» y «Colores»),
dibujados por la propia aplicación con las fotos de semillas de la segunda caja.

    xvfb-run -a /usr/bin/python3.12 docs/paper/charts_figure.py [carpeta]

Guarda cada gráfico en la carpeta (por defecto docs/paper/out/charts) y la tira de cinco
paneles en docs/paper/out/charts.png.
"""
import os, sys, tempfile, pickle
from pathlib import Path
ROOT = Path("/home/claude/fenotit")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "paper" / "out" / "charts"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = tempfile.mkdtemp(); os.environ["APPDATA"] = os.environ["LOCALAPPDATA"]
sys.path.insert(0, str(ROOT))
from fenotit import i18n
i18n.set_language("en"); i18n.load("en")
cache = Path(tempfile.gettempdir()) / "fenotit_chart_results.pkl"
if cache.exists():
    results = pickle.load(open(cache, "rb"))
else:
    from fenotit.core.analysis import objects
    from fenotit.core.corrections import pipeline as cp
    from fenotit.core.image_io import load_image
    from fenotit.core import batch
    TRAY = (0.03, 0.97, 0.12, 0.86); CARD = [[0.62, 0.0], [0.84, 0.25]]
    results = {}
    for path in sorted((ROOT / "tests/data/Cgiar_Bean_Sequia_Seeds").glob("*.jpg")):
        if path.stem.split("=")[-1] in ("23421", "23507"):
            continue
        raw = load_image(str(path)); corr = cp.Corrections(); corr.color.enabled = True
        fixed, _ = cp.apply(raw, corr)
        h, w = fixed.shape[:2]
        tray = fixed[int(h*TRAY[0]):int(h*TRAY[1]), int(w*TRAY[2]):int(w*TRAY[3])]
        seg = {"method": "threshold", "color_space": "LAB", "channel": 2, "auto": True, "refine": [{"op": "fill_holes"}]}
        r = objects.run(tray, {"segmentation": seg, "touching": True, "measure_size": True, "measure_color": True,
                               "n_colors": 10, "n_colors_auto": True,
                               "roi_shapes": [{"kind": "exclude", "type": "rect", "points": CARD}], "exclude_border": True})
        results["S" + path.stem.split("=")[-1][-2:] + ".jpg" if False else str(path)] = batch.light(r)
        print(path.name, flush=True)
    pickle.dump(results, open(cache, "wb"))
results = {f"/samples/S{i}.jpg": value for i, (key, value) in enumerate(sorted(results.items()), 1)}
import tkinter as tk
from fenotit.gui.style import ttk_styles
from fenotit.gui.charts.panels import BatchChartWindow
from fenotit.gui.charts.color3d import Color3DWindow
root = tk.Tk(); root.withdraw(); ttk_styles.apply(root)
win = BatchChartWindow(root, results)
win.geometry("620x520+0+0")
root.update()
def shot(name, kind, cols, unit="images", sd=False):
    if win.unit.get() != unit:
        win.unit.set(unit); win._unit_changed()
    win.chooser.kind.set(kind); win.chooser.chosen[kind] = cols; win.chooser.count[kind] = len(cols)
    win.chooser._build_vars(); win.show_sd.set(sd); root.update(); win._refresh(); root.update()
    win._fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
shot("bars", "bars", ["length_px_mean"], sd=True)
shot("box", "box", ["length_px"], unit="objects")
shot("hist", "hist", ["area_px2"], unit="objects")
shot("scatter", "scatter", ["length_px", "width_px"], unit="objects")
images = [(path, i, Path(path).name, value.extra.get("tables") or {}) for i, (path, value) in enumerate(results.items(), 1)]
c3 = Color3DWindow(root, images[:1], images, scope="batch")
c3.geometry("700x560+0+0"); root.update()
c3.dims.set("1,2"); c3._redraw(); root.update()
c3._fig.savefig(OUT / "colors.png", dpi=300, bbox_inches="tight")
root.destroy()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, str(ROOT / "docs"))
import figure_style as style
style.apply()
panels = [Image.open(OUT / f"{name}.png").convert("RGB") for name in ("bars", "box", "hist", "scatter", "colors")]
ratios = [image.width / image.height for image in panels]
fig = plt.figure(figsize=(style.DOUBLE, style.DOUBLE / sum(ratios) * 1.12), layout="constrained")
grid = fig.add_gridspec(1, len(panels), width_ratios=ratios)
for index, image in enumerate(panels):
    axes = fig.add_subplot(grid[index])
    axes.imshow(image)
    axes.set_axis_off()
    axes.set_title(f"({'abcde'[index]})")
fig.savefig(OUT.parent / "charts.png", dpi=style.CHART_DPI)
print(OUT.parent / "charts.png")
