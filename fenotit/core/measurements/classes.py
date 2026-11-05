"""Área por clase (classes): % y área de cada clase dentro de cada objeto, o la
categoría de cada objeto cuando cada uno es de una sola clase."""
import numpy as np

from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import regions


def slug(name: str) -> str:
    """Nombre de clase → columna ASCII (sin tildes ni espacios)."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_") or "class"


@step("classes", "measurement", requires=("labels", "classes"), params=[
    {"key": "names", "type": "list", "default": []},
    {"key": "colors", "type": "list", "default": []},
    {"key": "whole", "type": "bool", "default": False},
])
def classes(ctx, p):
    """Área de cada clase dentro de cada objeto (p. ej. sano / lesión / clorosis en cada
    hoja o fruto): % del objeto y área. En la foto, el total de cada clase en los objetos.
    whole = cada objeto es entero de una clase (categorías de las regiones del modelo):
    el objeto lleva su categoría y la foto, cuántos objetos y qué área hay de cada una."""
    names = list(p["names"]) or [f"class {i}" for i in range(int(ctx.classes.max()) + 1)]
    cols = [slug(n) for n in names]
    if p.get("whole"):
        _categories(ctx, names, cols, list(p["colors"]))
        return
    u, f, d2 = ctx.unit, ctx.scale, ctx.digits(2)
    rows = ctx.object_rows()
    total = np.zeros(len(names), np.int64)
    for oid, sl, m in regions(ctx.labels):
        if oid in ctx.excluded:
            continue
        cls = ctx.classes[sl][m > 0]
        n = np.bincount(cls, minlength=len(names))[:len(names)]
        total += n
        area = max(int(n[1:].sum()), 1)         # % sobre las clases del objeto (sin fondo): suman 100
        for i, c in enumerate(cols):
            if i == 0:
                continue                       # el fondo no es una parte del objeto
            rows[oid][f"{c}_pct"] = round(100 * n[i] / area, 2)
            rows[oid][f"{c}_area_{u}2"] = round(n[i] * f * f, d2)
    s = max(int(total[1:].sum()), 1)
    img = ctx.image_row()
    for i, c in enumerate(cols):
        if i == 0:
            continue
        img[f"{c}_pct"] = round(100 * total[i] / s, 2)
        img[f"{c}_area_{u}2"] = round(total[i] * f * f, d2)
    ctx.tables["classes"] = [{"class": names[i], "pct": round(100 * total[i] / s, 2),
                              f"area_{u}2": round(total[i] * f * f, d2), "color": (p["colors"] or [""] * 99)[i]}
                             for i in range(1, len(names))]
    # vista: los colores de las clases dentro de cada objeto, sobre la foto atenuada
    from fenotit.core.segmentation.classifier import colorize
    view = (ctx.image * 0.35).astype(np.uint8)
    if p["colors"]:
        inside = ctx.labels > 0
        if ctx.excluded:
            inside &= ~np.isin(ctx.labels, list(ctx.excluded))
        view[inside] = (0.35 * ctx.image[inside] + 0.65 * colorize(ctx.classes, list(p["colors"]))[inside]
                        ).astype(np.uint8)
    ctx.images["classes"] = view


def _categories(ctx, names, cols, colors):
    u, f, d2 = ctx.unit, ctx.scale, ctx.digits(2)
    rows = ctx.object_rows()
    n = np.zeros(len(names), np.int64)
    area = np.zeros(len(names), np.int64)
    for oid, sl, m in regions(ctx.labels):
        if oid in ctx.excluded:
            continue
        c = int(np.bincount(ctx.classes[sl][m > 0], minlength=len(names)).argmax())
        rows[oid]["category"] = names[c]
        n[c] += 1
        area[c] += int((m > 0).sum())
    s = max(int(area[1:].sum()), 1)
    img = ctx.image_row()
    for i, c in enumerate(cols):
        if i == 0:
            continue
        img[f"n_{c}"] = int(n[i])
        img[f"{c}_area_{u}2"] = round(area[i] * f * f, d2)
        img[f"{c}_pct"] = round(100 * area[i] / s, 2)
    from fenotit.core.segmentation.classifier import colorize
    view = (ctx.image * 0.35).astype(np.uint8)
    inside = ctx.labels > 0
    if ctx.excluded:
        inside &= ~np.isin(ctx.labels, list(ctx.excluded))
    if colors:
        view[inside] = (0.35 * ctx.image[inside] + 0.65 * colorize(ctx.classes, colors)[inside]).astype(np.uint8)
    ctx.images["classes"] = view
    ctx.tables["classes"] = [{"class": names[i], "n": int(n[i]), "pct": round(100 * area[i] / s, 2),
                              f"area_{u}2": round(area[i] * f * f, d2), "color": (colors or [""] * 99)[i]}
                             for i in range(1, len(names))]
