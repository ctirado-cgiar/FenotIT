"""Distancias euclidianas entre objetos vecinos.

Vecinos: los k más cercanos (por centro) o los de la red de Voronoi (triangulación de
Delaunay de los centros). Medida de cada par:
  edge_line     de borde a borde siguiendo la línea entre los centros (lo que se dibuja)
  edge_nearest  entre los puntos más cercanos de los dos contornos
  center        de centro a centro
Tablas: `distances` (un par por fila), por objeto (vecino más cercano y promedio) y por
imagen (media del vecino más cercano e índice de Clark-Evans, con centros).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import find_objects
from scipy.spatial import Delaunay, cKDTree

from fenotit.core.pipeline.base import step


def _centroids(labels):
    out = {}
    for oid, sl in enumerate(find_objects(labels), 1):
        if sl is None:
            continue
        ys, xs = np.nonzero(labels[sl] == oid)
        out[oid] = (xs.mean() + sl[1].start, ys.mean() + sl[0].start)
    return out


def _pairs(ids, pts, mode, k):
    if len(ids) < 2:
        return []
    if mode == "voronoi" and len(ids) >= 3:
        try:
            tri = Delaunay(pts)
            edges = {tuple(sorted((s[i], s[j]))) for s in tri.simplices for i in range(3) for j in range(i + 1, 3)}
            return sorted(edges)
        except Exception:                      # puntos alineados: se usan vecinos más cercanos
            pass
    k = int(min(max(k, 1), len(ids) - 1))
    _, idx = cKDTree(pts).query(pts, k=k + 1)
    return sorted({tuple(sorted((i, int(j)))) for i, row in enumerate(idx) for j in row[1:]})


def _edge_on_line(labels, a, b, ca, cb):
    """Punto donde la línea entre centros sale de `a` y donde entra en `b`."""
    n = int(max(abs(cb[0] - ca[0]), abs(cb[1] - ca[1]))) * 2 + 2
    xs = np.linspace(ca[0], cb[0], n)
    ys = np.linspace(ca[1], cb[1], n)
    h, w = labels.shape
    v = labels[np.clip(np.round(ys).astype(int), 0, h - 1), np.clip(np.round(xs).astype(int), 0, w - 1)]
    in_a = np.nonzero(v == a)[0]
    in_b = np.nonzero(v == b)[0]
    i = in_a.max() if len(in_a) else 0
    j = in_b.min() if len(in_b) else n - 1
    if j <= i:                                 # se tocan
        return (xs[i], ys[i]), (xs[i], ys[i])
    return (xs[i], ys[i]), (xs[j], ys[j])


def _crosses(labels, a, b, p1, p2) -> bool:
    """¿El tramo entre p1 y p2 pasa por encima de otro objeto?"""
    n = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])) + 2
    h, w = labels.shape
    xs = np.clip(np.round(np.linspace(p1[0], p2[0], n)).astype(int), 0, w - 1)
    ys = np.clip(np.round(np.linspace(p1[1], p2[1], n)).astype(int), 0, h - 1)
    v = labels[ys, xs]
    return bool(np.any((v != 0) & (v != a) & (v != b)))


def _contours(labels, ids):
    out = {}
    for oid, sl in enumerate(find_objects(labels), 1):
        if sl is None or oid not in ids:
            continue
        m = (labels[sl] == oid).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
                                   offset=(sl[1].start, sl[0].start))
        out[oid] = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(float)
    return out


@step("distances", "measurement", requires=("labels",), params=[
    {"key": "neighbors", "type": "choice", "choices": ["knn", "voronoi"], "default": "knn"},
    {"key": "k", "type": "int", "default": 5, "min": 1, "max": 50},
    {"key": "measure", "type": "choice", "choices": ["edge_line", "edge_nearest", "center"], "default": "edge_line"},
])
def distances(ctx, p):
    cents = _centroids(ctx.labels)
    ids = sorted(cents)
    pts = np.array([cents[i] for i in ids], float).reshape(-1, 2)
    pairs = _pairs(ids, pts, p["neighbors"], p["k"])
    unit, f = ctx.unit, ctx.scale
    contours = _contours(ctx.labels, set(ids)) if p["measure"] == "edge_nearest" else {}
    trees = {}
    rows, lines = [], []          # lines: (punto 1, punto 2, distancia en px)
    for i, j in pairs:
        a, b = ids[i], ids[j]
        ca, cb = cents[a], cents[b]
        center = float(np.hypot(cb[0] - ca[0], cb[1] - ca[1]))
        if p["measure"] == "center":
            p1, p2 = ca, cb
        elif p["measure"] == "edge_nearest":
            ta = trees.setdefault(a, cKDTree(contours[a]))
            d, k = ta.query(contours[b], k=1)
            jb = int(np.argmin(d))
            p1, p2 = tuple(contours[a][k[jb]]), tuple(contours[b][jb])
        else:
            p1, p2 = _edge_on_line(ctx.labels, a, b, ca, cb)
        dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        crosses = _crosses(ctx.labels, a, b, p1, p2)
        rows.append({"object_a": a, "object_b": b, f"distance_{unit}": round(dist * f, ctx.digits(1, 4)),
                     f"center_distance_{unit}": round(center * f, ctx.digits(1, 4)), "crosses_object": int(crosses)})
        if not crosses:                        # en la vista no se dibujan líneas que tapan otro objeto
            lines.append((p1, p2, dist))
    ctx.tables["distances"] = rows

    # por objeto: vecino más cercano y promedio a sus vecinos
    per = {i: [] for i in ids}
    for r in rows:
        d = r[f"distance_{unit}"]
        per[r["object_a"]].append(d)
        per[r["object_b"]].append(d)
    obj = ctx.object_rows()
    for oid, ds in per.items():
        obj[oid].update({f"nearest_{unit}": min(ds) if ds else None,
                         f"mean_neighbor_{unit}": round(float(np.mean(ds)), 4) if ds else None,
                         "n_neighbors": len(ds)})

    # por imagen: vecino más cercano (centros) e índice de Clark-Evans
    img = {"n_objects": len(ids), "n_pairs": len(rows)}
    if len(ids) >= 2:
        nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
        area = float((ctx.roi > 0).sum()) if ctx.roi is not None else float(ctx.labels.size)
        expected = 0.5 / np.sqrt(len(ids) / area)
        near = [min(ds) for ds in per.values() if ds]
        se = 0.26136 / np.sqrt(len(ids) ** 2 / area)          # error estándar de Clark-Evans
        img.update({f"mean_nearest_{unit}": round(float(np.mean(near)), 4) if near else None,
                    f"sd_nearest_{unit}": round(float(np.std(near, ddof=1)), 4) if len(near) > 1 else None,
                    "clark_evans_R": round(float(nn.mean() / expected), 4),
                    "clark_evans_z": round(float((nn.mean() - expected) / se), 3)})
    ctx.image_row().update(img)

    # vista: contornos y una línea por par, más tenue cuanto más lejos
    from fenotit.core.pipeline import overlay
    from fenotit.core.pipeline.views import add_overlay, regions
    ov = overlay.empty()
    for oid, sl, m in regions(ctx.labels):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                   offset=(sl[1].start, sl[0].start))
        ov["outlines"] += [(c.reshape(-1, 2), True) for c in cnts]
    if lines:
        ds = np.array([d for *_, d in lines])
        lo, hi = float(ds.min()), float(ds.max())
        for p1, p2, d in lines:
            strength = 1.0 if hi == lo else 1.0 - 0.75 * (d - lo) / (hi - lo)
            ov["lines"].append((p1, p2, strength))
    add_overlay(ctx, "distances", ov)
