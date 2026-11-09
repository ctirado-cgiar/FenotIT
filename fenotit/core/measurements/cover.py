"""Cobertura por clase (cover): % y área de cada clase dentro del área de análisis, sin
separar objetos."""
import numpy as np

from fenotit.core.measurements.classes import slug
from fenotit.core.pipeline.base import step


@step("cover", "measurement", requires=("classes",), params=[
    {"key": "names", "type": "list", "default": []},
    {"key": "colors", "type": "list", "default": []},
])
def cover(ctx, p):
    """Cuenta los píxeles de cada clase dentro del área de análisis (toda la foto si no hay
    áreas; sin las zonas excluidas). Tabla `classes` (una fila por clase), columnas
    `<clase>_pct` y `<clase>_area_<unidad>2` en la fila de la foto y vista con los colores."""
    names = list(p["names"]) or [f"class {i}" for i in range(int(ctx.classes.max()) + 1)]
    region = ctx.analysis_region(ctx.classes.shape)
    inside = np.ones(ctx.classes.shape, bool) if region is None else region > 0
    n = np.bincount(ctx.classes[inside], minlength=len(names))[:len(names)]
    total = max(int(n.sum()), 1)
    u, f, d2 = ctx.unit, ctx.scale, ctx.digits(2)
    colors = list(p["colors"]) or [""] * len(names)
    img = ctx.image_row()
    rows = []
    for i, name in enumerate(names):
        col = slug(name)
        pct, area = round(100 * n[i] / total, 2), round(int(n[i]) * f * f, d2)
        img[f"{col}_pct"] = pct
        img[f"{col}_area_{u}2"] = area
        rows.append({"class": name, "pct": pct, f"area_{u}2": area, "color": colors[i] if i < len(colors) else ""})
    img[f"analyzed_area_{u}2"] = round(total * f * f, d2)
    ctx.tables["classes"] = rows
    view = (ctx.image * 0.35).astype(np.uint8)
    if p["colors"]:
        from fenotit.core.segmentation.classifier import colorize
        view[inside] = (0.35 * ctx.image[inside] + 0.65 * colorize(ctx.classes, colors)[inside]).astype(np.uint8)
    ctx.images["cover"] = view
