"""Procesadores: ROI, limpieza, etiquetado, separación y filtros de objetos."""
import cv2
import numpy as np

from fenotit.core.pipeline.base import step


def _fit(mask, shape):
    return mask if mask.shape[:2] == shape[:2] else cv2.resize(
        mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


@step("roi", "processor", requires=("mask",), provides=("mask",))
def roi(ctx, p):
    mask = ctx.mask
    if ctx.roi is not None:
        mask = cv2.bitwise_and(mask, _fit(ctx.roi, mask.shape))
    if ctx.exclusions:
        mask = mask.copy()
        for pts, _color in ctx.exclusions:
            cv2.fillPoly(mask, [np.asarray(pts, np.int32)], 0)
    ctx.mask = mask


@step("clean", "processor", requires=("mask",), provides=("mask",), params=[
    {"key": "open_iterations", "type": "int", "default": 0, "min": 0, "max": 10},
    {"key": "fill_holes", "type": "bool", "default": True},
])
def clean(ctx, p):
    mask = ctx.mask
    if p["open_iterations"]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
                                iterations=int(p["open_iterations"]))
    if p["fill_holes"]:
        mask = _fill_small_holes(mask)
    ctx.mask = mask


def _fill_small_holes(mask, frac=0.2):
    """Rellena huecos menores que frac × el área típica de un objeto. Los huecos
    grandes suelen ser fondo encerrado por un anillo de objetos que se tocan."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n < 2:
        return mask
    typical = float(np.median(stats[1:, cv2.CC_STAT_AREA]))
    inv = (mask == 0).astype(np.uint8)
    k, lab, hs, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = mask.shape
    out = mask.copy()
    for i in range(1, k):
        x, y, bw, bh, a = hs[i]
        if x == 0 or y == 0 or x + bw == w or y + bh == h:
            continue
        if a < frac * typical:
            out[lab == i] = 255
    return out


@step("label", "processor", requires=("mask",), provides=("labels",))
def label(ctx, p):
    _, ctx.labels = cv2.connectedComponents((ctx.mask > 0).astype(np.uint8), connectivity=8)


@step("separate", "processor", requires=("mask",), provides=("labels",), params=[
    {"key": "min_distance", "type": "int", "default": 0, "min": 0, "max": 1000},
    {"key": "peak_threshold", "type": "float", "default": 0.2, "min": 0.05, "max": 0.95},
    {"key": "merge_ratio", "type": "float", "default": 0.9, "min": 0.5, "max": 1.0},
    {"key": "split_large", "type": "bool", "default": True},
    {"key": "use_shadows", "type": "bool", "default": True},
])
def separate(ctx, p):
    """Separa objetos pegados: un pico de la transformada de distancia por objeto + watershed.

    min_distance (px) entre centros; 0 = automático (~ radio típico de los objetos).
    merge_ratio: vuelve a unir dos partes si el cuello entre ellas es al menos esa
    fracción del grosor de la más delgada (evita cortar objetos alargados).
    split_large: corta por las muescas los objetos de área ≥ 1.6 × la mediana
    (p. ej. dos semillas lado a lado, que la distancia ve como un solo objeto).
    use_shadows: en objetos claros, las sombras oscuras entre ellos marcan la
    frontera (solo si los píxeles oscuros son minoría dentro de la máscara).
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    binary = (ctx.mask > 0).astype(np.uint8)
    if not binary.any():
        ctx.labels = np.zeros(binary.shape, np.int32)
        return
    _, comps = cv2.connectedComponents(binary, connectivity=8)
    ctx.groups = comps
    core = _without_shadows(ctx.image, binary, comps) if p["use_shadows"] else binary
    dist = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    if dist.max() == 0:
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # umbral de picos relativo al grosor de CADA grupo (un objeto grande, p. ej. una
    # etiqueta, no debe esconder los picos de los objetos delgados)
    from scipy.ndimage import maximum
    comp_max = np.asarray(maximum(dist, comps, index=np.arange(comps.max() + 1)))
    rel = float(p["peak_threshold"])

    def _peaks(min_distance):
        found = peak_local_max(dist, min_distance=min_distance, threshold_abs=0.5, labels=comps)
        keep = dist[tuple(found.T)] >= rel * comp_max[comps[tuple(found.T)]]
        return found[keep]
    md = int(p["min_distance"])
    if md <= 0:
        rough = _peaks(5)
        md = max(3, int(0.9 * np.median(dist[tuple(rough.T)]))) if len(rough) else 5
    ctx.extra["separate_min_distance"] = md
    peaks = _peaks(md)
    markers = np.zeros(binary.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-dist, markers, mask=binary > 0).astype(np.int32)
    owner = None
    if core is not binary:   # partes separadas por sombras nunca se vuelven a unir
        _, cc = cv2.connectedComponents(core, connectivity=8)
        owner = _majority(labels, cc)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    labels = _merge_flat_necks(labels, dist, float(p["merge_ratio"]), owner)
    radius = float(np.median(dist[tuple(peaks.T)])) if len(peaks) else 0
    labels = _merge_without_notches(labels, binary, radius, owner,
                                    cv2.distanceTransform(binary, cv2.DIST_L2, 5))
    labels = _merge_fragments(labels, comps)
    labels = _keep_single_sized_groups(labels, comps)
    if p["split_large"]:
        labels = _split_large(labels, float(np.median(dist[tuple(peaks.T)])) if len(peaks) else 0)
    ctx.labels = labels
    ctx.images["distance"] = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _without_shadows(image, binary, comps, max_dark=0.35):
    """En grupos grandes (≥1.6× el área típica) quita las zonas oscuras conectadas con
    el fondo: sombras entre objetos claros. Los objetos sueltos no se tocan."""
    if image is None or image.ndim != 3:
        return binary
    areas = np.bincount(comps.ravel())
    areas[0] = 0
    if (areas > 0).sum() < 3:
        return binary
    big = np.flatnonzero(areas >= 1.6 * np.median(areas[areas > 0]))
    if not len(big):
        return binary
    in_groups = np.isin(comps, big)
    light = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    vals = light[binary > 0].reshape(-1, 1)
    t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = float((vals < t).mean())
    if not 0.02 < dark < max_dark:
        return binary
    dark_px = (in_groups & (light < t)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark_px, connectivity=8)
    edge = cv2.dilate((binary == 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    typical = float(np.median(areas[areas > 0]))
    shadow_ids = []
    for i in np.unique(lab[edge & (dark_px > 0)]):
        if i == 0:
            continue
        a, w, h = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # sombra = línea larga o zona grande; un hilo compacto (hilum) no cuenta
        if a >= 0.1 * typical or (a >= 0.01 * typical and max(w, h) ** 2 / max(a, 1) >= 4):
            shadow_ids.append(i)
    shadow = np.isin(lab, shadow_ids)
    core = ((binary > 0) & ~shadow).astype(np.uint8)
    return cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _solidity(m):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return cv2.contourArea(c) / hull if hull else 0.0


def _cut_by_notches(m, radius, single):
    """Corta una máscara en 2 partes por una muesca profunda: elige el corte recto
    (de muesca a muesca o de muesca al borde) que deja dos partes convexas y de área
    cercana a la de un objeto típico (single)."""
    min_area = 0.35 * single
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    try:
        defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
    except cv2.error:
        return None
    if defects is None:
        return None
    pts = [tuple(cnt[f][0]) for *_, f, d in defects[:, 0] if d / 256 >= 0.25 * radius]
    if not pts:
        return None
    border = [tuple(q[0]) for q in cnt[::3]]
    cands = {(a, b) for a in pts for b in pts + border if a != b}
    best, best_score = None, -np.inf
    for p0, p1 in cands:
        length = np.hypot(p0[0] - p1[0], p0[1] - p1[1])
        if not 3 <= length <= 4 * radius:
            continue
        line = np.zeros_like(m)
        cv2.line(line, p0, p1, 1, 1)
        if m[line > 0].mean() < 0.9:
            continue
        cut = m.copy()
        cv2.line(cut, p0, p1, 0, 2)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cut, connectivity=4)
        big = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_area]
        if len(big) != 2:
            continue
        score = (min(_solidity((lab == k).astype(np.uint8)) for k in big)
                 - 0.1 * sum(abs(stats[k, cv2.CC_STAT_AREA] / single - 1) for k in big)
                 - 0.01 * length / radius)
        if score > best_score:
            best, best_score = (lab, big), score
    if best is None:
        return None
    lab, big = best
    # el corte debe dejar partes claramente más convexas que el todo (dos semillas
    # pegadas sí; una vaina curva cortada en dos no)
    parts_solidity = min(_solidity((lab == k).astype(np.uint8)) for k in big)
    if parts_solidity < 0.85 or parts_solidity < _solidity(m) + 0.03:
        return None
    out = np.zeros(m.shape, np.int32)
    out[lab == big[0]], out[lab == big[1]] = 1, 2
    from skimage.segmentation import expand_labels
    return np.where(m > 0, expand_labels(out, 2), 0)


def _split_large(labels, radius):
    """Divide objetos ~2× la mediana cortando entre sus muescas (repite hasta 3 veces)."""
    from scipy.ndimage import find_objects
    if radius <= 0 or labels.max() < 3:
        return labels
    ids, areas = np.unique(labels[labels > 0], return_counts=True)
    single = float(np.median(areas))
    labels = labels.copy()
    nxt = int(labels.max()) + 1
    queue = [(int(i), 0) for i, a in zip(ids, areas) if a >= 1.6 * single]
    while queue:
        oid, level = queue.pop()
        sls = find_objects((labels == oid).astype(np.uint8))
        if not sls or sls[0] is None:
            continue
        sl = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in sls[0])
        m = (labels[sl] == oid).astype(np.uint8)
        parts = _cut_by_notches(m, radius, single)
        if parts is None:
            continue
        region = labels[sl]
        region[parts == 2] = nxt
        for k, a in ((oid, (parts == 1).sum()), (nxt, (parts == 2).sum())):
            if a >= 1.6 * single and level < 3:
                queue.append((k, level + 1))
        nxt += 1
    return labels


def _keep_single_sized_groups(labels, comps, max_ratio=1.5):
    """Un grupo del tamaño de un solo objeto no se parte: se vuelve a unir.
    El tamaño típico sale de los grupos que no se partieron (objetos sueltos);
    p. ej. una vaina con cintura no se cuenta como dos."""
    fg = labels > 0
    if not fg.any():
        return labels
    keys = np.unique(comps[fg].astype(np.int64) * (int(labels.max()) + 1) + labels[fg])
    grp = keys // (int(labels.max()) + 1)
    gid, nparts = np.unique(grp, return_counts=True)
    area = np.bincount(comps[fg], minlength=int(comps.max()) + 1)
    single = gid[nparts == 1]
    if len(single) < 3:
        return labels
    typical = float(np.median(area[single]))
    merge = gid[(nparts > 1) & (area[gid] < max_ratio * typical)]
    if not len(merge):
        return labels
    labels = labels.copy()
    sel = np.isin(comps, merge) & fg
    labels[sel] = (comps[sel] + int(labels.max()) + 1).astype(labels.dtype)
    return labels


def _merge_fragments(labels, comps, frac=0.35):
    """Une a su vecino las partes muy pequeñas (< frac × el área típica de un objeto
    suelto) que salieron de partir un grupo: puntas y colas de objetos alargados."""
    ids, areas = np.unique(labels[labels > 0], return_counts=True)
    if len(ids) < 2:
        return labels
    group = dict(zip(*np.unique(np.stack([labels[labels > 0], comps[labels > 0]]), axis=1)))
    per_group = np.bincount(np.array([group[i] for i in ids]))
    alone = [a for i, a in zip(ids, areas) if per_group[group[i]] == 1]
    typical = float(np.median(alone if alone else areas))
    labels = labels.copy()
    small = sorted((a, i) for i, a in zip(ids, areas) if a < frac * typical and per_group[group[i]] > 1)
    kernel = np.ones((3, 3), np.uint8)
    for _, i in small:
        m = (labels == i).astype(np.uint8)
        if not m.any():
            continue
        ring = (cv2.dilate(m, kernel) > 0) & (m == 0)
        neigh, n = np.unique(labels[ring], return_counts=True)
        keep = neigh > 0
        if keep.any():
            labels[m > 0] = neigh[keep][np.argmax(n[keep])]
    return labels


def _merge_without_notches(labels, binary, radius, owner=None, dist=None, rel_depth=0.2,
                           min_neck=0.6):
    """Une dos partes vecinas si el corte entre ellas no termina en muescas.

    Dos objetos que se tocan dejan una muesca a cada lado del contacto; un objeto
    alargado o curvo (vaina) cortado por la mitad no: al menos un extremo del corte
    queda sobre el borde convexo. Se mide la profundidad de cada extremo respecto al
    casco convexo de la unión de las dos partes. Solo se une si además el cuello es
    ancho (≥ min_neck del grosor de la parte más delgada): en montones de semillas
    encimadas el contacto es angosto aunque no deje muesca."""
    if radius <= 0 or labels.max() < 2:
        return labels
    min_depth = max(2.0, rel_depth * radius)
    bg = binary == 0
    near_bg = cv2.dilate(bg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pairs: dict[tuple[int, int], list] = {}
    for a, b, sl_a, sl_b in ((labels[:, :-1], labels[:, 1:], np.s_[:, :-1], np.s_[:, 1:]),
                             (labels[:-1, :], labels[1:, :], np.s_[:-1, :], np.s_[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        ys, xs = np.nonzero(m)
        for y, x, u, v in zip(ys, xs, a[m], b[m]):
            pairs.setdefault((min(u, v), max(u, v)), []).append((x, y))
    parent = np.arange(labels.max() + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    from scipy.ndimage import find_objects, maximum
    slices = find_objects(labels)
    peak = np.asarray(maximum(dist, labels, index=np.arange(labels.max() + 1))) if dist is not None else None
    for (a, b), pts in pairs.items():
        if owner is not None and owner[a] != owner[b]:
            continue
        pts = np.asarray(pts)
        if peak is not None and dist[pts[:, 1], pts[:, 0]].max() < min_neck * min(peak[a], peak[b]):
            continue
        ends = pts[near_bg[pts[:, 1], pts[:, 0]]]
        if len(ends) == 0:                       # corte interno (entre 3+ objetos): se respeta
            continue
        sa, sb = slices[a - 1], slices[b - 1]
        y0, y1 = min(sa[0].start, sb[0].start), max(sa[0].stop, sb[0].stop)
        x0, x1 = min(sa[1].start, sb[1].start), max(sa[1].stop, sb[1].stop)
        union = np.isin(labels[y0:y1, x0:x1], (a, b)).astype(np.uint8)
        cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        # extremos: los puntos del corte junto al fondo, agrupados en sus dos puntas
        far = ends[np.argmax(np.hypot(*(ends - ends[0]).T))]
        groups = [ends[np.hypot(*(ends - far).T) > np.hypot(*(ends - ends[0]).T)],
                  ends[np.hypot(*(ends - far).T) <= np.hypot(*(ends - ends[0]).T)]]
        depths = []
        for g in groups:
            if len(g):
                depths.append(max(cv2.pointPolygonTest(hull, (float(x - x0), float(y - y0)), True)
                                  for x, y in g))
        if min(depths) < min_depth:
            parent[root(a)] = root(b)
    roots = np.array([root(i) for i in range(len(parent))])
    return roots[labels].astype(np.int32)


def _majority(labels, other):
    """Para cada etiqueta, el valor más frecuente (≠0) de 'other' en sus píxeles."""
    m = (labels > 0) & (other > 0)
    base = int(other.max()) + 1
    keys, n = np.unique(labels[m].astype(np.int64) * base + other[m], return_counts=True)
    lab, o = keys // base, keys % base
    out = np.zeros(labels.max() + 1, np.int64)
    order = np.lexsort((n, lab))          # por etiqueta, la de más píxeles al final
    out[lab[order]] = o[order]
    return out


def _merge_flat_necks(labels, dist, ratio, owner=None):
    from scipy.ndimage import maximum
    if ratio >= 1.0 or labels.max() < 2:
        return labels
    xs, ys, vs = [], [], []
    for a, b, da, db in ((labels[:, :-1], labels[:, 1:], dist[:, :-1], dist[:, 1:]),
                         (labels[:-1, :], labels[1:, :], dist[:-1, :], dist[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        xs.append(np.minimum(a[m], b[m]))
        ys.append(np.maximum(a[m], b[m]))
        vs.append(np.minimum(da[m], db[m]))
    xs, ys, vs = np.concatenate(xs), np.concatenate(ys), np.concatenate(vs)
    if not len(xs):
        return labels
    key = xs.astype(np.int64) * (int(labels.max()) + 1) + ys
    uniq, inv = np.unique(key, return_inverse=True)
    top = np.zeros(len(uniq))
    np.maximum.at(top, inv, vs)
    n1 = int(labels.max()) + 1
    saddle = {(int(k // n1), int(k % n1)): v for k, v in zip(uniq, top)}
    peak = maximum(dist, labels, index=np.arange(labels.max() + 1))
    parent = np.arange(labels.max() + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (a, b), v in saddle.items():
        if owner is not None and owner[a] != owner[b]:
            continue
        if v >= ratio * min(peak[a], peak[b]):
            parent[root(a)] = root(b)
    roots = np.array([root(i) for i in range(len(parent))])
    return roots[labels].astype(np.int32)


@step("filter", "processor", requires=("labels",), provides=("labels",), params=[
    {"key": "area_min", "type": "int", "default": 1000, "min": 0, "max": 10_000_000},
    {"key": "area_max", "type": "int", "default": 500_000, "min": 1, "max": 100_000_000},
    {"key": "width_min", "type": "int", "default": 5, "min": 0, "max": 100_000},
    {"key": "width_max", "type": "int", "default": 99_999, "min": 1, "max": 100_000},
    {"key": "length_min", "type": "int", "default": 10, "min": 0, "max": 100_000},
    {"key": "length_max", "type": "int", "default": 99_999, "min": 1, "max": 100_000},
    {"key": "ar_max", "type": "float", "default": 10.0, "min": 1.0, "max": 100.0},
    {"key": "exclude_border", "type": "bool", "default": True},
])
def filter_objects(ctx, p):
    """Filtra objetos (tamaños en px) y renumera 1..n por filas, de izquierda a derecha."""
    from scipy.ndimage import find_objects
    labels = ctx.labels
    h, w = labels.shape
    keep = []
    for oid, sl in enumerate(find_objects(labels), 1):
        if sl is None:
            continue
        ys, xs = np.nonzero(labels[sl] == oid)
        area = len(xs)
        if not (p["area_min"] <= area <= p["area_max"]):
            continue
        y0, x0 = sl[0].start, sl[1].start
        if p["exclude_border"] and (x0 == 0 or y0 == 0 or sl[1].stop == w or sl[0].stop == h):
            continue
        (_, _), (a, b), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        width, length = min(a, b), max(a, b)
        if not (p["width_min"] <= width <= p["width_max"] and p["length_min"] <= length <= p["length_max"]):
            continue
        if width > 0 and length / width > p["ar_max"]:
            continue
        keep.append((y0 + ys.mean(), x0 + xs.mean(), sl[0].stop - sl[0].start, oid, sl))
    out = np.zeros_like(labels)
    row_h = float(np.median([k[2] for k in keep])) if keep else 1.0
    for new_id, (_, _, _, oid, sl) in enumerate(sorted(keep, key=lambda k: (round(k[0] / row_h), k[1])), 1):
        region = out[sl]
        region[labels[sl] == oid] = new_id
    ctx.labels = out
