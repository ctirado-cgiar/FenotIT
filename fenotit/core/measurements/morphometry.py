"""Morfometría (morphometry): tamaño y descriptores de forma de cada objeto."""
import cv2
import numpy as np

from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import regions, included_view


@step("morphometry", "measurement", requires=("labels",), params=[
    {"key": "isolated_only", "type": "bool", "default": True},
])
def morphometry(ctx, p):
    """Tamaño y forma. Con isolated_only, los objetos que se tocaban con otro
    (desagrupados) se cuentan pero no se miden: su contorno no es confiable."""
    u, f = ctx.unit, ctx.scale
    d1, d2 = ctx.digits(1), ctx.digits(2)
    rows = ctx.object_rows()
    touching = ctx.touching_ids()
    axes = {}
    from fenotit.core.uncertainty import area_rel_sd
    unc = ctx.extra.setdefault("area_rel_sd", {})      # incertidumbre del área por píxeles (metadatos)
    for oid, sl, m in regions(ctx.labels):
        rows[oid]["touching"] = int(oid in touching)
        if oid in ctx.excluded or (p["isolated_only"] and oid in touching):
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        area = float(m.sum())                     # píxeles del objeto
        poly_area = cv2.contourArea(cnt)          # para razones de forma (consistente con el casco)
        perim = cv2.arcLength(cv2.approxPolyDP(cnt, 1.0, True), True)  # corrige el escalonado de píxeles
        hull = cv2.convexHull(cnt)
        hull_area, hull_perim = cv2.contourArea(hull), cv2.arcLength(hull, True)
        unc[oid] = area_rel_sd(area, perim)
        rect = cv2.minAreaRect(cnt)
        (a, b) = rect[1]
        width, length = min(a, b), max(a, b)
        axes[oid] = _axes(rect, sl)
        major, minor, ecc = length, width, 0.0
        if len(cnt) >= 5:
            (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
            major, minor = max(ma, mi), min(ma, mi)
            ecc = float(np.sqrt(1 - (minor / major) ** 2)) if major > 0 else 0.0
        ys, xs = np.nonzero(m)
        cx, cy = xs.mean(), ys.mean()
        pts = cnt.reshape(-1, 2).astype(float)
        r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        rows[oid].update({
            f"area_{u}2": round(area * f * f, d2),
            f"length_{u}": round(length * f, d1),
            f"width_{u}": round(width * f, d1),
            f"perimeter_{u}": round(perim * f, d1),
            "aspect_ratio": round(length / width, 4) if width else 0,
            "circularity": round(min(1.0, 4 * np.pi * poly_area / perim ** 2), 4) if perim else 0,
            "solidity": round(poly_area / hull_area, 4) if hull_area else 0,
            "convexity": round(min(1.0, hull_perim / perim), 4) if perim else 0,
            "elongation": round(1 - width / length, 4) if length else 0,
            "rectangularity": round(poly_area / (width * length), 4) if width * length else 0,
            "eccentricity": round(ecc, 4),
            f"major_axis_{u}": round(major * f, d1),
            f"minor_axis_{u}": round(minor * f, d1),
            f"radius_min_{u}": round(r.min() * f, d1),
            f"radius_mean_{u}": round(r.mean() * f, d1),
            f"radius_max_{u}": round(r.max() * f, d1),
            "radius_ratio": round(r.min() / r.max(), 4) if r.max() else 0,
            "centroid_x_px": int(round(cx + sl[1].start)),
            "centroid_y_px": int(round(cy + sl[0].start)),
        })
    measured = [r for r in rows.values() if f"area_{u}2" in r]
    ctx.image_row().update({
        "n_objects": len(set(rows) - ctx.excluded),
        "n_touching": len((touching & set(rows)) - ctx.excluded),
        "n_measured": len(measured),
    })
    included_view(ctx, "morphometry", {r["object_id"] for r in measured})
    ov = ctx.extra["overlays"]["morphometry"]       # lo que se midió: largo y ancho de cada objeto
    ov["thin_lines"] = True
    for seg in axes.values():
        ov["lines"] += [(p1, p2, 0.4) for p1, p2 in seg]


def _axes(rect, sl):
    """Largo y ancho (rectángulo de área mínima) como dos segmentos por el centro."""
    (cx, cy), (a, b), ang = rect
    cx, cy = cx + sl[1].start, cy + sl[0].start
    t = np.deg2rad(ang)
    u, v = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    (lu, lv) = (a, b) if a >= b else (b, a)
    if a < b:
        u, v = v, u
    c = np.array([cx, cy])
    return [(tuple(c - u * lu / 2), tuple(c + u * lu / 2)), (tuple(c - v * lv / 2), tuple(c + v * lv / 2))]
