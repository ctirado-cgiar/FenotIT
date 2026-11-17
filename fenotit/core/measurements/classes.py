"""Área por clase (classes): % y área de cada clase dentro de cada objeto, o la
categoría de cada objeto cuando cada uno es de una sola clase."""
import numpy as np

from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import regions


def slug(name: str) -> str:
    """Nombre de clase → columna ASCII (sin tildes ni espacios)."""
    import re
    import unicodedata
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "class"


@step("classes", "measurement", requires=("labels", "classes"), params=[
    {"key": "names", "type": "list", "default": []},
    {"key": "colors", "type": "list", "default": []},
    {"key": "whole", "type": "bool", "default": False},
])
def classes(ctx, params):
    """Área de cada clase dentro de cada objeto (p. ej. sano / lesión / clorosis en cada
    hoja o fruto): % del objeto y área. En la foto, el total de cada clase en los objetos.
    whole = cada objeto es entero de una clase (categorías de las regiones del modelo):
    el objeto lleva su categoría y la foto, cuántos objetos y qué área hay de cada una."""
    names = list(params["names"]) or [f"class {i}" for i in range(int(ctx.classes.max()) + 1)]
    cols = [slug(pixel_counts) for pixel_counts in names]
    if params.get("whole"):
        _categories(ctx, names, cols, list(params["colors"]))
        return
    unit, scale, area_digits = ctx.unit, ctx.scale, ctx.digits(2)
    rows = ctx.object_rows()
    total = np.zeros(len(names), np.int64)
    for oid, bounds, mask in regions(ctx.labels):
        if oid in ctx.excluded:
            continue
        cls = ctx.classes[bounds][mask > 0]
        pixel_counts = np.bincount(cls, minlength=len(names))[:len(names)]
        total += pixel_counts
        area = max(int(pixel_counts[1:].sum()), 1)         # % sobre las clases del objeto (sin fondo): suman 100
        for i, column in enumerate(cols):
            if i == 0:
                continue                       # el fondo no es una parte del objeto
            rows[oid][f"{column}_pct"] = round(100 * pixel_counts[i] / area, 2)
            rows[oid][f"{column}_area_{unit}2"] = round(pixel_counts[i] * scale * scale, area_digits)
    total_px = max(int(total[1:].sum()), 1)
    img = ctx.image_row()
    for i, column in enumerate(cols):
        if i == 0:
            continue
        img[f"{column}_pct"] = round(100 * total[i] / total_px, 2)
        img[f"{column}_area_{unit}2"] = round(total[i] * scale * scale, area_digits)
    ctx.tables["classes"] = [{"class": names[i], "pct": round(100 * total[i] / total_px, 2),
                              f"area_{unit}2": round(total[i] * scale * scale, area_digits), "color": (params["colors"] or [""] * 99)[i]}
                             for i in range(1, len(names))]
    # vista: los colores de las clases dentro de cada objeto, sobre la foto atenuada
    from fenotit.core.segmentation.classifier import colorize
    view = (ctx.image * 0.35).astype(np.uint8)
    if params["colors"]:
        inside = ctx.labels > 0
        if ctx.excluded:
            inside &= ~np.isin(ctx.labels, list(ctx.excluded))
        view[inside] = (0.35 * ctx.image[inside] + 0.65 * colorize(ctx.classes, list(params["colors"]))[inside]
                        ).astype(np.uint8)
    ctx.images["classes"] = view


def _categories(ctx, names, cols, colors):
    unit, scale, area_digits = ctx.unit, ctx.scale, ctx.digits(2)
    rows = ctx.object_rows()
    counts = np.zeros(len(names), np.int64)
    area = np.zeros(len(names), np.int64)
    for oid, bounds, mask in regions(ctx.labels):
        if oid in ctx.excluded:
            continue
        category = int(np.bincount(ctx.classes[bounds][mask > 0], minlength=len(names)).argmax())
        rows[oid]["category"] = names[category]
        counts[category] += 1
        area[category] += int((mask > 0).sum())
    total_px = max(int(area[1:].sum()), 1)
    img = ctx.image_row()
    for i, category in enumerate(cols):
        if i == 0:
            continue
        img[f"n_{category}"] = int(counts[i])
        img[f"{category}_area_{unit}2"] = round(area[i] * scale * scale, area_digits)
        img[f"{category}_pct"] = round(100 * area[i] / total_px, 2)
    from fenotit.core.segmentation.classifier import colorize
    view = (ctx.image * 0.35).astype(np.uint8)
    inside = ctx.labels > 0
    if ctx.excluded:
        inside &= ~np.isin(ctx.labels, list(ctx.excluded))
    if colors:
        view[inside] = (0.35 * ctx.image[inside] + 0.65 * colorize(ctx.classes, colors)[inside]).astype(np.uint8)
    ctx.images["classes"] = view
    ctx.tables["classes"] = [{"class": names[i], "n": int(counts[i]), "pct": round(100 * area[i] / total_px, 2),
                              f"area_{unit}2": round(area[i] * scale * scale, area_digits), "color": (colors or [""] * 99)[i]}
                             for i in range(1, len(names))]
