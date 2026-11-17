"""Distancias euclidianas entre objetos vecinos.

Vecinos: los k más cercanos (por centro) o los de la red de Voronoi (triangulación de
Delaunay de los centros). Medida de cada par:
  edge_line     de borde a borde siguiendo la línea entre los centros (lo que se dibuja)
  edge_nearest  entre los puntos más cercanos de los dos contornos
  center        de centro a centro
Tablas: `distances` (un par por fila), por objeto (vecino más cercano y promedio) y por
imagen (media del vecino más cercano e índice de Clark-Evans, con centros). Con clases
(cajas de un detector o regiones con categoría), cada par lleva la clase de sus dos objetos
y la imagen, el vecino más cercano de la misma clase para cada clase. Con cajas, la
distancia es siempre de centro a centro.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import find_objects
from scipy.spatial import Delaunay, cKDTree

from fenotit.core.pipeline.base import step


def _centroids(labels):
    out = {}
    for oid, bounds in enumerate(find_objects(labels), 1):
        if bounds is None:
            continue
        pixel_y, pixel_x = np.nonzero(labels[bounds] == oid)
        out[oid] = (pixel_x.mean() + bounds[1].start, pixel_y.mean() + bounds[0].start)
    return out


def _pairs(ids, pts, mode, n_nearest):
    if len(ids) < 2:
        return []
    if mode == "voronoi" and len(ids) >= 3:
        try:
            tri = Delaunay(pts)
            edges = {tuple(sorted((simplex[i], simplex[j]))) for simplex in tri.simplices for i in range(3) for j in range(i + 1, 3)}
            return sorted(edges)
        except Exception:                      # puntos alineados: se usan vecinos más cercanos
            pass
    n_nearest = int(min(max(n_nearest, 1), len(ids) - 1))
    _, idx = cKDTree(pts).query(pts, k=n_nearest + 1)
    return sorted({tuple(sorted((i, int(j)))) for i, row in enumerate(idx) for j in row[1:]})


def _edge_on_line(labels, object_a, object_b, center_a, center_b):
    """Punto donde la línea entre centros sale de `a` y donde entra en `b`."""
    n_samples = int(max(abs(center_b[0] - center_a[0]), abs(center_b[1] - center_a[1]))) * 2 + 2
    line_x = np.linspace(center_a[0], center_b[0], n_samples)
    line_y = np.linspace(center_a[1], center_b[1], n_samples)
    height, width = labels.shape
    values = labels[np.clip(np.round(line_y).astype(int), 0, height - 1), np.clip(np.round(line_x).astype(int), 0, width - 1)]
    in_a = np.nonzero(values == object_a)[0]
    in_b = np.nonzero(values == object_b)[0]
    i = in_a.max() if len(in_a) else 0
    j = in_b.min() if len(in_b) else n_samples - 1
    if j <= i:                                 # se tocan
        return (line_x[i], line_y[i]), (line_x[i], line_y[i])
    return (line_x[i], line_y[i]), (line_x[j], line_y[j])


def _crosses(labels, object_a, object_b, start, end) -> bool:
    """¿El tramo entre p1 y p2 pasa por encima de otro objeto?"""
    n_samples = int(np.hypot(end[0] - start[0], end[1] - start[1])) + 2
    height, width = labels.shape
    line_x = np.clip(np.round(np.linspace(start[0], end[0], n_samples)).astype(int), 0, width - 1)
    line_y = np.clip(np.round(np.linspace(start[1], end[1], n_samples)).astype(int), 0, height - 1)
    values = labels[line_y, line_x]
    return bool(np.any((values != 0) & (values != object_a) & (values != object_b)))


def _contours(labels, ids):
    out = {}
    for oid, bounds in enumerate(find_objects(labels), 1):
        if bounds is None or oid not in ids:
            continue
        mask = (labels[bounds] == oid).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
                                   offset=(bounds[1].start, bounds[0].start))
        out[oid] = np.vstack([contour.reshape(-1, 2) for contour in cnts]).astype(float)
    return out


def _object_classes(ctx, ids) -> dict[int, str]:
    """Clase de cada objeto: la de su caja o la más común entre sus píxeles; vacío sin clases."""
    names = list(ctx.extra.get("class_names") or [])
    if ctx.detections is not None:
        return {box["id"]: box.get("class", "") for box in ctx.detections} if len({box["cls"] for box in ctx.detections}) > 1 \
            or len(names) > 1 else {}
    if ctx.classes is None or len(names) < 2:
        return {}
    out = {}
    for oid, bounds in enumerate(find_objects(ctx.labels), 1):
        if bounds is None or oid not in ids:
            continue
        class_index = int(np.bincount(ctx.classes[bounds][ctx.labels[bounds] == oid]).argmax())
        out[oid] = names[class_index - 1] if 0 < class_index <= len(names) else ""
    return out


@step("distances", "measurement", requires=("labels",), params=[
    {"key": "neighbors", "type": "choice", "choices": ["knn", "voronoi"], "default": "knn"},
    {"key": "k", "type": "int", "default": 5, "min": 1, "max": 50},
    {"key": "measure", "type": "choice", "choices": ["edge_line", "edge_nearest", "center"], "default": "edge_line"},
])
def distances(ctx, params):
    if ctx.detections is not None:
        from fenotit.core.objects.boxes import center
        cents = {box["id"]: center(box) for box in ctx.detections if box["id"] not in ctx.excluded}
        params = {**params, "measure": "center"}
    else:
        cents = {oid: point for oid, point in _centroids(ctx.labels).items() if oid not in ctx.excluded}
    ids = sorted(cents)
    klass = _object_classes(ctx, set(ids))
    pts = np.array([cents[i] for i in ids], float).reshape(-1, 2)
    pairs = _pairs(ids, pts, params["neighbors"], params["k"])
    unit, scale = ctx.unit, ctx.scale
    contours = _contours(ctx.labels, set(ids)) if params["measure"] == "edge_nearest" else {}
    trees = {}
    rows, lines = [], []          # lines: (punto 1, punto 2, distancia en px)
    for i, j in pairs:
        object_a, object_b = ids[i], ids[j]
        center_a, center_b = cents[object_a], cents[object_b]
        center_px = float(np.hypot(center_b[0] - center_a[0], center_b[1] - center_a[1]))
        if params["measure"] == "center":
            start, end = center_a, center_b
        elif params["measure"] == "edge_nearest":
            tree_a = trees.setdefault(object_a, cKDTree(contours[object_a]))
            distance, nearest_index = tree_a.query(contours[object_b], k=1)
            nearest_b = int(np.argmin(distance))
            start, end = tuple(contours[object_a][nearest_index[nearest_b]]), tuple(contours[object_b][nearest_b])
        else:
            start, end = _edge_on_line(ctx.labels, object_a, object_b, center_a, center_b)
        dist = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        crosses = _crosses(ctx.labels, object_a, object_b, start, end)
        row = {"object_a": object_a, "object_b": object_b}
        if klass:
            row.update({"class_a": klass.get(object_a, ""), "class_b": klass.get(object_b, "")})
        row.update({f"distance_{unit}": round(dist * scale, ctx.digits(1, 4)),
                    f"center_distance_{unit}": round(center_px * scale, ctx.digits(1, 4)), "crosses_object": int(crosses)})
        rows.append(row)
        if not crosses:                        # en la vista no se dibujan líneas que tapan otro objeto
            lines.append((start, end, dist))
    ctx.tables["distances"] = rows

    # por objeto: vecino más cercano y promedio a sus vecinos
    per = {i: [] for i in ids}
    for pair_row in rows:
        distance = pair_row[f"distance_{unit}"]
        per[pair_row["object_a"]].append(distance)
        per[pair_row["object_b"]].append(distance)
    obj = ctx.object_rows()
    for oid, pair_distances in per.items():
        obj[oid].update({f"nearest_{unit}": min(pair_distances) if pair_distances else None,
                         f"mean_neighbor_{unit}": round(float(np.mean(pair_distances)), 4) if pair_distances else None,
                         "n_neighbors": len(pair_distances)})

    # por imagen: vecino más cercano (centros) e índice de Clark-Evans
    img = {"n_objects": len(ids), "n_pairs": len(rows)}
    if len(ids) >= 2:
        nearest_px = cKDTree(pts).query(pts, k=2)[0][:, 1]
        area = float((ctx.roi > 0).sum()) if ctx.roi is not None else float(ctx.labels.size)
        expected = 0.5 / np.sqrt(len(ids) / area)
        near = [min(pair_distances) for pair_distances in per.values() if pair_distances]
        standard_error = 0.26136 / np.sqrt(len(ids) ** 2 / area)          # error estándar de Clark-Evans
        img.update({f"mean_nearest_{unit}": round(float(np.mean(near)), 4) if near else None,
                    f"sd_nearest_{unit}": round(float(np.std(near, ddof=1)), 4) if len(near) > 1 else None,
                    f"min_nearest_{unit}": round(float(np.min(near)), 4) if near else None,
                    f"max_nearest_{unit}": round(float(np.max(near)), 4) if near else None,
                    f"mean_pair_{unit}": round(float(np.mean([pair_row[f"distance_{unit}"] for pair_row in rows])), 4)
                    if rows else None,
                    "clark_evans_R": round(float(nearest_px.mean() / expected), 4),
                    "clark_evans_z": round(float((nearest_px.mean() - expected) / standard_error), 3)})
    if klass:                                  # vecino más cercano de la misma clase (centros)
        from fenotit.core.measurements.classes import slug
        for name in sorted(set(klass.values())):
            members = [i for i, oid in enumerate(ids) if klass.get(oid) == name]
            key = f"mean_nearest_{slug(name)}_{unit}"
            if len(members) < 2:
                img[key] = None
                continue
            sub_pts = pts[members]
            distance = cKDTree(sub_pts).query(sub_pts, k=2)[0][:, 1]
            img[key] = round(float(distance.mean()) * scale, 4)
    ctx.image_row().update(img)

    # vista: contornos y una línea por par, más tenue cuanto más lejos
    from fenotit.core.pipeline import overlay
    from fenotit.core.pipeline.views import add_overlay, regions
    marks = overlay.empty()
    if ctx.detections is not None:
        from fenotit.core.objects.boxes import rect
        marks["outlines"] = [(rect(box), box["id"] not in ctx.excluded) for box in ctx.detections]
    else:
        for oid, bounds, mask in regions(ctx.labels):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                       offset=(bounds[1].start, bounds[0].start))
            marks["outlines"] += [(contour.reshape(-1, 2), oid not in ctx.excluded) for contour in cnts]
    if lines:
        pair_distances = np.array([distance for *_, distance in lines])
        shortest, longest = float(pair_distances.min()), float(pair_distances.max())
        for start, end, distance in lines:
            strength = 1.0 if longest == shortest else 1.0 - 0.75 * (distance - shortest) / (longest - shortest)
            marks["lines"].append((start, end, strength))
    add_overlay(ctx, "distances", marks)
