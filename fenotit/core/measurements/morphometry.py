"""Morfometría (morphometry): tamaño y descriptores de forma de cada objeto."""
import cv2
import numpy as np

from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import regions, included_view


@step("morphometry", "measurement", requires=("labels",), params=[
    {"key": "isolated_only", "type": "bool", "default": True},
])
def morphometry(ctx, params):
    """Tamaño y forma. Con isolated_only, los objetos que se tocaban con otro
    (desagrupados) se cuentan pero no se miden: su contorno no es confiable."""
    unit, scale = ctx.unit, ctx.scale
    length_digits, area_digits = ctx.digits(1), ctx.digits(2)
    rows = ctx.object_rows()
    touching = ctx.touching_ids()
    axes = {}
    from fenotit.core.uncertainty import area_rel_sd
    unc = ctx.extra.setdefault("area_rel_sd", {})      # incertidumbre del área por píxeles (metadatos)
    for oid, bounds, mask in regions(ctx.labels):
        rows[oid]["touching"] = int(oid in touching)
        if oid in ctx.excluded or (params["isolated_only"] and oid in touching):
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        area = float(mask.sum())                     # píxeles del objeto
        poly_area = cv2.contourArea(cnt)          # para razones de forma (consistente con el casco)
        perim = cv2.arcLength(cv2.approxPolyDP(cnt, 1.0, True), True)  # corrige el escalonado de píxeles
        hull = cv2.convexHull(cnt)
        hull_area, hull_perim = cv2.contourArea(hull), cv2.arcLength(hull, True)
        unc[oid] = area_rel_sd(area, perim)
        rect = cv2.minAreaRect(cnt)
        (side_a, side_b) = rect[1]
        width, length = min(side_a, side_b), max(side_a, side_b)
        axes[oid] = _axes(rect, bounds)
        major, minor, ecc = length, width, 0.0
        if len(cnt) >= 5:
            (_, _), (ellipse_major, ellipse_minor), _ = cv2.fitEllipse(cnt)
            major, minor = max(ellipse_major, ellipse_minor), min(ellipse_major, ellipse_minor)
            ecc = float(np.sqrt(1 - (minor / major) ** 2)) if major > 0 else 0.0
        pixel_y, pixel_x = np.nonzero(mask)
        cx, cy = pixel_x.mean(), pixel_y.mean()
        pts = cnt.reshape(-1, 2).astype(float)
        radii = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        rows[oid].update({
            f"area_{unit}2": round(area * scale * scale, area_digits),
            f"length_{unit}": round(length * scale, length_digits),
            f"width_{unit}": round(width * scale, length_digits),
            f"perimeter_{unit}": round(perim * scale, length_digits),
            "aspect_ratio": round(length / width, 4) if width else 0,
            "circularity": round(min(1.0, 4 * np.pi * poly_area / perim ** 2), 4) if perim else 0,
            "solidity": round(poly_area / hull_area, 4) if hull_area else 0,
            "convexity": round(min(1.0, hull_perim / perim), 4) if perim else 0,
            "elongation": round(1 - width / length, 4) if length else 0,
            "rectangularity": round(poly_area / (width * length), 4) if width * length else 0,
            "eccentricity": round(ecc, 4),
            f"major_axis_{unit}": round(major * scale, length_digits),
            f"minor_axis_{unit}": round(minor * scale, length_digits),
            f"radius_min_{unit}": round(radii.min() * scale, length_digits),
            f"radius_mean_{unit}": round(radii.mean() * scale, length_digits),
            f"radius_max_{unit}": round(radii.max() * scale, length_digits),
            "radius_ratio": round(radii.min() / radii.max(), 4) if radii.max() else 0,
            "centroid_x_px": int(round(cx + bounds[1].start)),
            "centroid_y_px": int(round(cy + bounds[0].start)),
        })
    measured = [radii for radii in rows.values() if f"area_{unit}2" in radii]
    ctx.image_row().update({
        "n_objects": len(set(rows) - ctx.excluded),
        "n_touching": len((touching & set(rows)) - ctx.excluded),
        "n_measured": len(measured),
    })
    included_view(ctx, "morphometry", {radii["object_id"] for radii in measured})
    marks = ctx.extra["overlays"]["morphometry"]       # lo que se midió: largo y ancho de cada objeto
    marks["thin_lines"] = True
    for seg in axes.values():
        marks["lines"] += [(start, end, 0.4) for start, end in seg]


def _axes(rect, bounds):
    """Largo y ancho (rectángulo de área mínima) como dos segmentos por el centro."""
    (cx, cy), (side_a, side_b), ang = rect
    cx, cy = cx + bounds[1].start, cy + bounds[0].start
    t = np.deg2rad(ang)
    along, across = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    (long_side, short_side) = (side_a, side_b) if side_a >= side_b else (side_b, side_a)
    if side_a < side_b:
        along, across = across, along
    center_point = np.array([cx, cy])
    return [(tuple(center_point - along * long_side / 2), tuple(center_point + along * long_side / 2)), (tuple(center_point - across * short_side / 2), tuple(center_point + across * short_side / 2))]
