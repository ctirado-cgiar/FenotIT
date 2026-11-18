"""Separación de objetos que se tocan (separate): picos de la transformada de distancia,
watershed, corte por muescas y unión de partes."""
import cv2
import numpy as np

# _instances usa los objetos que ya separó un modelo de IA, si los hay.
from fenotit.core.objects.label import _instances
# El decorador step registra la separación como pieza de la cadena.
from fenotit.core.pipeline.base import step


@step("separate", "processor", requires=("mask",), provides=("labels",), params=[
    {"key": "min_distance", "type": "int", "default": 0, "min": 0, "max": 1000},
    {"key": "peak_threshold", "type": "float", "default": 0.2, "min": 0.05, "max": 0.95},
    {"key": "merge_ratio", "type": "float", "default": 0.9, "min": 0.5, "max": 1.0},
    {"key": "split_large", "type": "bool", "default": True},
    {"key": "use_shadows", "type": "bool", "default": True},
])
def separate(ctx, params):
    """Separa objetos pegados: un pico de la transformada de distancia por objeto + watershed.

    min_distance (px) entre centros; 0 = automático (~ radio típico de los objetos).
    merge_ratio: vuelve a unir dos partes si el cuello entre ellas es al menos esa
    fracción del grosor de la más delgada (evita cortar objetos alargados).
    split_large: corta por las muescas los objetos de área ≥ 1.6 × la mediana
    (p. ej. dos semillas lado a lado, que la distancia ve como un solo objeto).
    Si la segmentación ya separó los objetos (modelo de IA), se usan esos.
    use_shadows: en objetos claros, las sombras oscuras entre ellos marcan la
    frontera (solo si los píxeles oscuros son minoría dentro de la máscara).
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    if _instances(ctx):
        return
    binary = (ctx.mask > 0).astype(np.uint8)
    if not binary.any():
        ctx.labels = np.zeros(binary.shape, np.int32)
        return
    _, comps = cv2.connectedComponents(binary, connectivity=8)
    ctx.groups = comps
    core = _without_shadows(ctx.image, binary, comps) if params["use_shadows"] else binary
    dist = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    if dist.max() == 0:
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # umbral de picos relativo al grosor de CADA grupo (un objeto grande, p. ej. una
    # etiqueta, no debe esconder los picos de los objetos delgados)
    comp_max = _label_max(dist, comps)
    rel = float(params["peak_threshold"])

    def _peaks(min_distance):
        found = peak_local_max(dist, min_distance=min_distance, threshold_abs=0.5, labels=comps)
        keep = dist[tuple(found.T)] >= rel * comp_max[comps[tuple(found.T)]]
        return found[keep]
    min_distance = int(params["min_distance"])
    if min_distance <= 0:
        rough = _peaks(5)
        min_distance = max(3, int(0.9 * np.median(dist[tuple(rough.T)]))) if len(rough) else 5
    ctx.extra["separate_min_distance"] = min_distance
    peaks = _peaks(min_distance)
    markers = np.zeros(binary.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-dist, markers, mask=binary > 0).astype(np.int32)
    owner = None
    if core is not binary:   # partes separadas por sombras nunca se vuelven a unir
        _, core_components = cv2.connectedComponents(core, connectivity=8)
        owner = _majority(labels, core_components)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    labels = _merge_flat_necks(labels, dist, float(params["merge_ratio"]), owner)
    radius = float(np.median(dist[tuple(peaks.T)])) if len(peaks) else 0
    labels = _merge_without_notches(labels, binary, radius, owner,
                                    cv2.distanceTransform(binary, cv2.DIST_L2, 5))
    labels = _merge_fragments(labels, comps)
    labels = _keep_single_sized_groups(labels, comps)
    if params["split_large"]:
        labels = _split_large(labels, float(np.median(dist[tuple(peaks.T)])) if len(peaks) else 0)
    ctx.labels = labels
    ctx.images["distance"] = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _label_max(values, labels):
    """Máximo de `values` en cada etiqueta (índice = etiqueta)."""
    out = np.zeros(int(labels.max()) + 1, values.dtype)     # mismo tipo: ufunc.at va por la vía rápida
    foreground = labels > 0
    np.maximum.at(out, labels[foreground], values[foreground])
    return out


def _areas(labels):
    """Área (px) de cada etiqueta; índice = etiqueta."""
    return np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)


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
    n_dark, lab, stats, _ = cv2.connectedComponentsWithStats(dark_px, connectivity=8)
    edge = cv2.dilate((binary == 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    typical = float(np.median(areas[areas > 0]))
    shadow_ids = []
    for i in np.unique(lab[edge & (dark_px > 0)]):
        if i == 0:
            continue
        area, width, height = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # sombra = línea larga o zona grande; un hilo compacto (hilum) no cuenta
        if area >= 0.1 * typical or (area >= 0.01 * typical and max(width, height) ** 2 / max(area, 1) >= 4):
            shadow_ids.append(i)
    shadow = np.isin(lab, shadow_ids)
    core = ((binary > 0) & ~shadow).astype(np.uint8)
    return cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _solidity(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(contour))
    return cv2.contourArea(contour) / hull if hull else 0.0


def _cut_by_notches(mask, radius, single):
    """Corta una máscara en 2 partes por una muesca profunda: elige el corte recto
    (de muesca a muesca o de muesca al borde) que deja dos partes convexas y de área
    cercana a la de un objeto típico (single)."""
    min_area = 0.35 * single
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    try:
        defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
    except cv2.error:
        return None
    if defects is None:
        return None
    pts = [tuple(cnt[far_index][0]) for *_, far_index, depth in defects[:, 0] if depth / 256 >= 0.25 * radius]
    if not pts:
        return None
    border = [tuple(point[0]) for point in cnt[::3]]
    cands = {(notch, other_point) for notch in pts for other_point in pts + border if notch != other_point}
    best, best_score = None, -np.inf
    for cut_start, cut_end in cands:
        length = np.hypot(cut_start[0] - cut_end[0], cut_start[1] - cut_end[1])
        if not 3 <= length <= 4 * radius:
            continue
        line = np.zeros_like(mask)
        cv2.line(line, cut_start, cut_end, 1, 1)
        if mask[line > 0].mean() < 0.9:
            continue
        cut = mask.copy()
        cv2.line(cut, cut_start, cut_end, 0, 2)
        n_parts, lab, stats, _ = cv2.connectedComponentsWithStats(cut, connectivity=4)
        big = [part for part in range(1, n_parts) if stats[part, cv2.CC_STAT_AREA] >= min_area]
        if len(big) != 2:
            continue
        score = (min(_solidity((lab == part).astype(np.uint8)) for part in big)
                 - 0.1 * sum(abs(stats[part, cv2.CC_STAT_AREA] / single - 1) for part in big)
                 - 0.01 * length / radius)
        if score > best_score:
            best, best_score = (lab, big), score
    if best is None:
        return None
    lab, big = best
    # el corte debe dejar partes claramente más convexas que el todo (dos semillas
    # pegadas sí; una vaina curva cortada en dos no)
    parts_solidity = min(_solidity((lab == part).astype(np.uint8)) for part in big)
    if parts_solidity < 0.85 or parts_solidity < _solidity(mask) + 0.03:
        return None
    out = np.zeros(mask.shape, np.int32)
    out[lab == big[0]], out[lab == big[1]] = 1, 2
    from skimage.segmentation import expand_labels
    return np.where(mask > 0, expand_labels(out, 2), 0)


def _split_large(labels, radius):
    """Divide objetos ~2× la mediana cortando entre sus muescas (repite hasta 3 veces)."""
    from scipy.ndimage import find_objects
    if radius <= 0 or labels.max() < 3:
        return labels
    all_areas = _areas(labels)
    ids = np.flatnonzero(all_areas[1:]) + 1
    areas = all_areas[ids]
    single = float(np.median(areas))
    labels = labels.copy()
    nxt = int(labels.max()) + 1
    boxes = dict(enumerate(find_objects(labels), 1))
    queue = [(int(i), 0) for i, part_area in zip(ids, areas) if part_area >= 1.6 * single]
    while queue:
        oid, level = queue.pop()
        box = boxes.get(oid)
        if box is None:
            continue
        bounds = tuple(slice(max(axis.start - 2, 0), axis.stop + 2) for axis in box)
        mask = (labels[bounds] == oid).astype(np.uint8)
        parts = _cut_by_notches(mask, radius, single)
        if parts is None:
            continue
        region = labels[bounds]
        region[parts == 2] = nxt
        boxes[nxt] = box
        for part_id, part_area in ((oid, (parts == 1).sum()), (nxt, (parts == 2).sum())):
            if part_area >= 1.6 * single and level < 3:
                queue.append((part_id, level + 1))
        nxt += 1
    return labels


def _keep_single_sized_groups(labels, comps, max_ratio=1.5):
    """Un grupo del tamaño de un solo objeto no se parte: se vuelve a unir.
    El tamaño típico sale de los grupos que no se partieron (objetos sueltos);
    p. ej. una vaina con cintura no se cuenta como dos."""
    foreground = labels > 0
    if not foreground.any():
        return labels
    stride = int(labels.max()) + 1
    present = np.bincount(comps[foreground].astype(np.int64) * stride + labels[foreground]) > 0
    grp = np.flatnonzero(present) // stride
    nparts_all = np.bincount(grp)
    gid = np.flatnonzero(nparts_all)
    nparts = nparts_all[gid]
    area = np.bincount(comps[foreground], minlength=int(comps.max()) + 1)
    single = gid[nparts == 1]
    if len(single) < 3:
        return labels
    typical = float(np.median(area[single]))
    merge = gid[(nparts > 1) & (area[gid] < max_ratio * typical)]
    if not len(merge):
        return labels
    labels = labels.copy()
    sel = np.isin(comps, merge) & foreground
    labels[sel] = (comps[sel] + int(labels.max()) + 1).astype(labels.dtype)
    return labels


def _merge_fragments(labels, comps, frac=0.35):
    """Une a su vecino las partes muy pequeñas (< frac × el área típica de un objeto
    suelto) que salieron de partir un grupo: puntas y colas de objetos alargados."""
    all_areas = _areas(labels)
    ids = np.flatnonzero(all_areas[1:]) + 1
    areas = all_areas[ids]
    if len(ids) < 2:
        return labels
    foreground = labels > 0
    group = np.zeros(len(all_areas), np.int64)       # cada parte cae dentro de un solo grupo
    group[labels[foreground]] = comps[foreground]
    per_group = np.bincount(group[ids])
    alone = [area for i, area in zip(ids, areas) if per_group[group[i]] == 1]
    typical = float(np.median(alone if alone else areas))
    labels = labels.copy()
    small = sorted((area, i) for i, area in zip(ids, areas) if area < frac * typical and per_group[group[i]] > 1)
    kernel = np.ones((3, 3), np.uint8)
    from scipy.ndimage import find_objects
    boxes = find_objects(labels)
    height, width = labels.shape
    for _, i in small:
        box = boxes[i - 1]
        if box is None:
            continue
        bounds = (slice(max(box[0].start - 1, 0), min(box[0].stop + 1, height)),
              slice(max(box[1].start - 1, 0), min(box[1].stop + 1, width)))
        region = labels[bounds]
        part_mask = (region == i).astype(np.uint8)
        if not part_mask.any():
            continue
        ring = (cv2.dilate(part_mask, kernel) > 0) & (part_mask == 0)
        neigh, counts = np.unique(region[ring], return_counts=True)
        keep = neigh > 0
        if keep.any():
            new = neigh[keep][np.argmax(counts[keep])]
            region[part_mask > 0] = new
            neighbor_box = boxes[new - 1]                       # la caja del vecino crece con la parte unida
            boxes[new - 1] = (slice(min(neighbor_box[0].start, box[0].start), max(neighbor_box[0].stop, box[0].stop)),
                              slice(min(neighbor_box[1].start, box[1].start), max(neighbor_box[1].stop, box[1].stop)))
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
    background = binary == 0
    near_bg = cv2.dilate(background.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pair_keys, pair_points = [], []
    stride = int(labels.max()) + 1
    for first, second in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        boundary = (first != second) & (first > 0) & (second > 0)
        pixel_y, pixel_x = np.nonzero(boundary)
        first_ids, second_ids = first[boundary].astype(np.int64), second[boundary].astype(np.int64)
        pair_keys.append(np.minimum(first_ids, second_ids) * stride + np.maximum(first_ids, second_ids))
        pair_points.append(np.stack([pixel_x, pixel_y], axis=1))
    pair_keys, pair_points = np.concatenate(pair_keys), np.concatenate(pair_points)
    order = np.argsort(pair_keys, kind="stable")
    pair_keys, pair_points = pair_keys[order], pair_points[order]
    cuts = np.flatnonzero(np.diff(pair_keys)) + 1
    pairs = {(int(key // stride), int(key % stride)): group
             for key, group in zip(pair_keys[np.r_[0, cuts]] if len(pair_keys) else [], np.split(pair_points, cuts))}
    parent = np.arange(labels.max() + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    from scipy.ndimage import find_objects
    slices = find_objects(labels)
    peak = _label_max(dist, labels) if dist is not None else None
    for (first, second), pts in pairs.items():
        if owner is not None and owner[first] != owner[second]:
            continue
        pts = np.asarray(pts)
        if peak is not None and dist[pts[:, 1], pts[:, 0]].max() < min_neck * min(peak[first], peak[second]):
            continue
        ends = pts[near_bg[pts[:, 1], pts[:, 0]]]
        if len(ends) == 0:                       # corte interno (entre 3+ objetos): se respeta
            continue
        bounds_a, bounds_b = slices[first - 1], slices[second - 1]
        y0, y1 = min(bounds_a[0].start, bounds_b[0].start), max(bounds_a[0].stop, bounds_b[0].stop)
        x0, x1 = min(bounds_a[1].start, bounds_b[1].start), max(bounds_a[1].stop, bounds_b[1].stop)
        union = np.isin(labels[y0:y1, x0:x1], (first, second)).astype(np.uint8)
        cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        # extremos: los puntos del corte junto al fondo, agrupados en sus dos puntas
        far = ends[np.argmax(np.hypot(*(ends - ends[0]).T))]
        groups = [ends[np.hypot(*(ends - far).T) > np.hypot(*(ends - ends[0]).T)],
                  ends[np.hypot(*(ends - far).T) <= np.hypot(*(ends - ends[0]).T)]]
        depths = []
        for group in groups:
            if len(group):
                depths.append(max(cv2.pointPolygonTest(hull, (float(x - x0), float(y - y0)), True)
                                  for x, y in group))
        if min(depths) < min_depth:
            parent[root(first)] = root(second)
    roots = np.array([root(i) for i in range(len(parent))])
    return roots[labels].astype(np.int32)


def _majority(labels, other):
    """Para cada etiqueta, el valor más frecuente (≠0) de 'other' en sus píxeles."""
    both = (labels > 0) & (other > 0)
    base = int(other.max()) + 1
    keys, counts = np.unique(labels[both].astype(np.int64) * base + other[both], return_counts=True)
    lab, value = keys // base, keys % base
    out = np.zeros(labels.max() + 1, np.int64)
    order = np.lexsort((counts, lab))          # por etiqueta, la de más píxeles al final
    out[lab[order]] = value[order]
    return out


def _merge_flat_necks(labels, dist, ratio, owner=None):
    if ratio >= 1.0 or labels.max() < 2:
        return labels
    smaller, larger, neck = [], [], []
    for first, second, dist_first, dist_second in ((labels[:, :-1], labels[:, 1:], dist[:, :-1], dist[:, 1:]),
                         (labels[:-1, :], labels[1:, :], dist[:-1, :], dist[1:, :])):
        boundary = (first != second) & (first > 0) & (second > 0)
        smaller.append(np.minimum(first[boundary], second[boundary]))
        larger.append(np.maximum(first[boundary], second[boundary]))
        neck.append(np.minimum(dist_first[boundary], dist_second[boundary]))
    smaller, larger, neck = np.concatenate(smaller), np.concatenate(larger), np.concatenate(neck)
    if not len(smaller):
        return labels
    key = smaller.astype(np.int64) * (int(labels.max()) + 1) + larger
    uniq, inv = np.unique(key, return_inverse=True)
    top = np.zeros(len(uniq), neck.dtype)
    np.maximum.at(top, inv, neck)
    stride = int(labels.max()) + 1
    saddle = {(int(pair_key // stride), int(pair_key % stride)): neck_value for pair_key, neck_value in zip(uniq, top)}
    peak = _label_max(dist, labels)
    parent = np.arange(labels.max() + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (first, second), neck_value in saddle.items():
        if owner is not None and owner[first] != owner[second]:
            continue
        if neck_value >= ratio * min(peak[first], peak[second]):
            parent[root(first)] = root(second)
    roots = np.array([root(i) for i in range(len(parent))])
    return roots[labels].astype(np.int32)
