"""Mediciones: agregan columnas a las tablas 'objects' e 'image'."""
import cv2
import numpy as np
from scipy.ndimage import find_objects

from fenotit.core.pipeline.base import step


def _regions(labels):
    for oid, sl in enumerate(find_objects(labels), 1):
        if sl is not None:
            yield oid, sl, (labels[sl] == oid).astype(np.uint8)


def _draw_ids(ctx, name, color=(0, 220, 100), touching_color=(0, 150, 255)):
    """Contornos numerados; los objetos desagrupados (se tocaban) van en naranja."""
    out = ctx.image.copy()
    touching = ctx.touching_ids()
    for oid, sl, m in _regions(ctx.labels):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                   offset=(sl[1].start, sl[0].start))
        cv2.drawContours(out, cnts, -1, touching_color if oid in touching else color, 2)
        ys, xs = np.nonzero(m)
        cx, cy = int(xs.mean()) + sl[1].start, int(ys.mean()) + sl[0].start
        cv2.putText(out, str(oid), (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    ctx.images[name] = out


@step("morphometry", "measurement", requires=("labels",), params=[
    {"key": "isolated_only", "type": "bool", "default": True},
])
def morphometry(ctx, p):
    """Tamaño y forma. Con isolated_only, los objetos que se tocaban con otro
    (desagrupados) se cuentan pero no se miden: su contorno no es confiable."""
    u, f = ctx.unit, ctx.scale
    rows = ctx.object_rows()
    touching = ctx.touching_ids()
    for oid, sl, m in _regions(ctx.labels):
        rows[oid]["touching"] = int(oid in touching)
        if p["isolated_only"] and oid in touching:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        area = float(m.sum())                     # píxeles del objeto
        poly_area = cv2.contourArea(cnt)          # para razones de forma (consistente con el casco)
        perim = cv2.arcLength(cv2.approxPolyDP(cnt, 1.0, True), True)  # corrige el escalonado de píxeles
        hull = cv2.convexHull(cnt)
        hull_area, hull_perim = cv2.contourArea(hull), cv2.arcLength(hull, True)
        (_, _), (a, b), _ = cv2.minAreaRect(cnt)
        width, length = min(a, b), max(a, b)
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
            f"area_{u}2": round(area * f * f, 3),
            f"length_{u}": round(length * f, 3),
            f"width_{u}": round(width * f, 3),
            f"perimeter_{u}": round(perim * f, 3),
            "aspect_ratio": round(length / width, 4) if width else 0,
            "circularity": round(min(1.0, 4 * np.pi * poly_area / perim ** 2), 4) if perim else 0,
            "solidity": round(poly_area / hull_area, 4) if hull_area else 0,
            "convexity": round(min(1.0, hull_perim / perim), 4) if perim else 0,
            "elongation": round(1 - width / length, 4) if length else 0,
            "rectangularity": round(poly_area / (width * length), 4) if width * length else 0,
            "eccentricity": round(ecc, 4),
            f"major_axis_{u}": round(major * f, 3),
            f"minor_axis_{u}": round(minor * f, 3),
            f"radius_min_{u}": round(r.min() * f, 3),
            f"radius_mean_{u}": round(r.mean() * f, 3),
            f"radius_max_{u}": round(r.max() * f, 3),
            "radius_ratio": round(r.min() / r.max(), 4) if r.max() else 0,
            "centroid_x_px": int(round(cx + sl[1].start)),
            "centroid_y_px": int(round(cy + sl[0].start)),
        })
    measured = [r for r in rows.values() if f"area_{u}2" in r]
    ctx.image_row().update({
        "n_objects": len(rows),
        "n_touching": len(touching & set(rows)),
        "n_measured": len(measured),
    })
    _draw_ids(ctx, "objects")


def _lab(rgb):
    L, a, b = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(int)
    return round(L * 100 / 255, 2), a - 128, b - 128


def _kmeans(pixels_rgb, k):
    from sklearn.cluster import KMeans
    k = int(min(k, len(np.unique(pixels_rgb, axis=0))))
    km = KMeans(n_clusters=k, n_init=10, random_state=134).fit(pixels_rgb)
    counts = np.bincount(km.labels_, minlength=k)
    order = np.argsort(-counts)
    return [(tuple(int(v) for v in km.cluster_centers_[i]), counts[i] / counts.sum()) for i in order]


def _color_row(rgb, frac):
    L, a, b = _lab(rgb)
    return {"R": rgb[0], "G": rgb[1], "B": rgb[2], "hex": "#%02x%02x%02x" % rgb,
            "L": L, "a": a, "b": b, "pct": round(100 * frac, 2)}


@step("color", "measurement", requires=("labels",), params=[
    {"key": "n_colors", "type": "int", "default": 3, "min": 1, "max": 10},
])
def color(ctx, p):
    """Colores dominantes (KMeans) por objeto y de todos los objetos juntos."""
    rgb_img = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2RGB)
    rows = ctx.object_rows()
    per_obj, all_px = [], []
    for oid, sl, m in _regions(ctx.labels):
        px = rgb_img[sl][m > 0].reshape(-1, 3)
        all_px.append(px)
        mean = tuple(int(v) for v in px.mean(axis=0))
        L, a, b = _lab(mean)
        rows[oid].update({"mean_R": mean[0], "mean_G": mean[1], "mean_B": mean[2],
                          "mean_L": L, "mean_a": a, "mean_b": b})
        for i, (c, frac) in enumerate(_kmeans(px, p["n_colors"]), 1):
            per_obj.append({"object_id": oid, "cluster": i, **_color_row(c, frac)})
    ctx.tables["object_colors"] = per_obj
    if all_px:
        px = np.concatenate(all_px)
        ctx.tables["image_colors"] = [{"cluster": i, **_color_row(c, frac)}
                                      for i, (c, frac) in enumerate(_kmeans(px, p["n_colors"]), 1)]


@step("count", "measurement", requires=("labels",))
def count(ctx, p):
    ids = {i for i, s in enumerate(find_objects(ctx.labels), 1) if s is not None}
    ctx.image_row().update({"n_objects": len(ids), "n_touching": len(ctx.touching_ids() & ids)})
    if "objects" not in ctx.images:
        _draw_ids(ctx, "objects")
