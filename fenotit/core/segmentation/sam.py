"""Segmentación con MobileSAM (Segment Anything ligero, Apache 2.0) en ONNX Runtime, en
CPU y sin internet. Dos modos:

  points  el usuario hace clic en los objetos (y, si hace falta, en el fondo): una
          máscara por clic positivo; varios clics sobre el mismo objeto lo afinan
  auto    una cuadrícula de puntos sobre la foto; cada punto propone un objeto y se
          quedan las propuestas confiables y no repetidas (como el "automatic mask
          generator" de SAM): sirve para contar sin ajustar umbrales

La foto se reduce a 1024 px de lado mayor para el codificador (una vez por foto; el
resultado se guarda) y las máscaras vuelven al tamaño de la foto."""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

SIZE = 1024
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
FILES = ("mobilesam_encoder.onnx", "mobilesam_decoder.onnx")


def model_dir() -> Path | None:
    """models/mobilesam junto a la app (repositorio o instalación) o en la carpeta de datos."""
    from fenotit import settings
    here = Path(__file__).resolve()
    roots = [Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "frozen", False) else None,
             Path(sys.executable).parent if getattr(sys, "frozen", False) else None,
             here.parents[3], settings.data_dir()]
    for r in roots:
        if r is None:
            continue
        d = r / "models" / "mobilesam"
        if all((d / f).exists() for f in FILES):
            return d
    return None


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return model_dir() is not None


_SESSIONS: dict = {}
_EMB: "OrderedDict[str, tuple]" = OrderedDict()


def _sessions():
    if not _SESSIONS:
        import onnxruntime as ort
        d = model_dir()
        if d is None:
            raise FileNotFoundError("MobileSAM model files not found (models/mobilesam)")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.enable_cpu_mem_arena = False             # sin la reserva de ORT: ~1 GB menos
        _SESSIONS["enc"] = ort.InferenceSession(str(d / FILES[0]), opts, providers=["CPUExecutionProvider"])
        _SESSIONS["dec"] = ort.InferenceSession(str(d / FILES[1]), opts, providers=["CPUExecutionProvider"])
    return _SESSIONS["enc"], _SESSIONS["dec"]


def _key(img: np.ndarray) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=12)
    h.update(str(img.shape).encode())
    h.update(np.ascontiguousarray(img[::5, ::5]).tobytes())
    return h.hexdigest()


def embed(img: np.ndarray):
    """Embeddings de la foto (BGR) y su escala; se guardan los de las 4 últimas fotos."""
    key = _key(img)
    if key in _EMB:
        _EMB.move_to_end(key)
        return _EMB[key]
    enc, _ = _sessions()
    h, w = img.shape[:2]
    k = SIZE / max(h, w)
    nh, nw = int(h * k + 0.5), int(w * k + 0.5)
    rgb = cv2.cvtColor(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = np.zeros((SIZE, SIZE, 3), np.float32)
    x[:nh, :nw] = (rgb.astype(np.float32) - MEAN) / STD
    emb = enc.run(None, {"image": x.transpose(2, 0, 1)[None]})[0]
    out = (emb, k, (h, w), (nh, nw))
    _EMB[key] = out
    while len(_EMB) > 6:
        _EMB.popitem(last=False)
    return out


def _decode(emb, coords: np.ndarray, labels: np.ndarray):
    _, dec = _sessions()
    masks, iou = dec.run(None, {"embeddings": emb, "coords": coords.astype(np.float32),
                                "labels": labels.astype(np.float32)})
    return masks, iou


def _upscale(low: np.ndarray, pre: tuple, orig: tuple) -> np.ndarray:
    """Máscara 256×256 (logits) → tamaño de la foto (bool)."""
    big = cv2.resize(low, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)[:pre[0], :pre[1]]
    return cv2.resize(big, (orig[1], orig[0]), interpolation=cv2.INTER_LINEAR) > 0


def from_points(img: np.ndarray, points: list) -> np.ndarray:
    """Objetos por clics: points = [[x, y, positivo], ...] en coordenadas 0-1. Cada clic
    positivo es un objeto; los negativos (fondo) se agregan a todos. Devuelve etiquetas."""
    emb, k, orig, pre = embed(img)
    h, w = orig
    pos = [(x * w * k, y * h * k) for x, y, p in points if p]
    neg = [(x * w * k, y * h * k) for x, y, p in points if not p]
    labels = np.zeros(orig, np.int32)
    if not pos:
        return labels
    coords = np.array([[p] + neg + [(0, 0)] for p in pos], np.float32)
    lab = np.array([[1] + [0] * len(neg) + [-1] for _ in pos], np.float32)
    masks, iou = _decode(emb, coords, lab)
    for i in range(len(pos)):
        j = 1 + int(np.argmax(iou[i, 1:])) if not neg else 0      # un clic: la mejor de 3
        m = _upscale(masks[i, j], pre, orig)
        labels[m & (labels == 0)] = i + 1
    return labels


WORK = 2048          # lado mayor del marco común donde se comparan las regiones


def _seeds(img: np.ndarray, pre: tuple, grid: int, max_seeds: int = 600) -> np.ndarray:
    """Puntos de prueba en el marco del codificador: primero los lugares que más se
    distinguen de su entorno (color Lab frente a un fondo muy suavizado: semillas, hojas,
    lesiones, terrones…), luego una cuadrícula gruesa para lo grande y lo poco contrastado."""
    lo_h, lo_w = int(round(pre[0] / 4)), int(round(pre[1] / 4))
    sm = cv2.resize(img, (lo_w * 2, lo_h * 2), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = cv2.GaussianBlur(lab, (0, 0), max(lo_w, lo_h) / 8)
    d = cv2.GaussianBlur(np.sqrt(((lab - bg) ** 2).sum(2)), (0, 0), 1.5)
    peak = (d == cv2.dilate(d, np.ones((7, 7), np.uint8))) & (d > max(6.0, float(np.percentile(d, 60))))
    ys, xs = np.nonzero(peak)
    order = np.argsort(-d[ys, xs])[:max_seeds]
    seeds = np.stack([xs[order] * 2 + 1, ys[order] * 2 + 1], 1).astype(np.float32).reshape(-1, 2)
    xs = (np.arange(grid) + 0.5) / grid
    g = np.array([(x * pre[1], y * pre[0]) for y in xs for x in xs], np.float32)
    g = g[np.random.default_rng(134).permutation(len(g))]
    return np.concatenate([seeds, g])


def _proposals(img, box, frame, grid, min_iou, min_stability, batch, inner_edges):
    """Regiones propuestas en un recorte box = (x0, y0, x1, y1) de la foto, pasadas al marco
    común `frame` = (alto, ancho). Devuelve [(puntaje, (y0, x0, sub_máscara))]. Las que tocan
    un borde interior del recorte (objeto cortado) se descartan."""
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    emb, k, orig, pre = embed(crop)
    lo_h, lo_w = int(round(pre[0] / 4)), int(round(pre[1] / 4))
    H, W = img.shape[:2]
    fy, fx = frame[0] / H, frame[1] / W
    fx0, fy0 = int(round(x0 * fx)), int(round(y0 * fy))
    fw, fh = max(1, int(round(x1 * fx)) - fx0), max(1, int(round(y1 * fy)) - fy0)
    pts = _seeds(crop, pre, grid)
    covered = np.zeros((lo_h, lo_w), bool)
    out, found, i0 = [], [], 0
    while i0 < len(pts):
        p = []
        while i0 < len(pts) and len(p) < batch:       # se saltan los puntos que ya caen en una región
            x, y = pts[i0]
            i0 += 1
            if not covered[min(int(y / 4), lo_h - 1), min(int(x / 4), lo_w - 1)]:
                p.append((x, y))
        if not p:
            break
        p = np.array(p, np.float32)
        coords = np.concatenate([p[:, None, :], np.zeros((len(p), 1, 2), np.float32)], axis=1)
        lab = np.tile(np.array([[1, -1]], np.float32), (len(p), 1))
        masks, iou = _decode(emb, coords, lab)
        for i in range(len(p)):
            for j in (1, 2, 3):
                if iou[i, j] < min_iou:
                    continue
                lg = masks[i, j][:lo_h, :lo_w]
                hi, lo = (lg > 1.0).sum(), (lg > -1.0).sum()
                if lo == 0 or hi / lo < min_stability:
                    continue
                m = lg > 0
                if m.sum() < 4:
                    continue
                if _touches(m, inner_edges):
                    continue
                if m.sum() < 0.25 * m.size:           # las enormes (fondo) abarcan objetos: no tapan puntos
                    covered |= cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                score = float(iou[i, j]) * float(hi / lo)
                r = _box(m)
                dup = next((n for n, c in enumerate(found) if _iou(r, c[1]) > 0.85), None)
                if dup is None:
                    found.append((score, r, lg.copy()))
                elif score > found[dup][0]:           # repetida: queda la mejor
                    found[dup] = (score, r, lg.copy())
    for score, _, lg in found:                         # al marco común
        big = cv2.resize(lg, (fw, fh), interpolation=cv2.INTER_LINEAR) > 0
        r = _box(big)
        if r is not None:
            out.append((score, (fy0 + r[0], fx0 + r[1], r[2])))
    return out


def _box(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if not len(ys):
        return None
    return (int(ys.min()), int(xs.min()), m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def _touches(m: np.ndarray, edges) -> bool:
    side = {"top": m[0], "bottom": m[-1], "left": m[:, 0], "right": m[:, -1]}
    return any(side[e].any() for e in edges)


def _iou(a, b) -> float:
    ay, ax, am = a
    by, bx, bm = b
    y0, x0 = max(ay, by), max(ax, bx)
    y1, x1 = min(ay + am.shape[0], by + bm.shape[0]), min(ax + am.shape[1], bx + bm.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0.0
    inter = (am[y0 - ay:y1 - ay, x0 - ax:x1 - ax] & bm[y0 - by:y1 - by, x0 - bx:x1 - bx]).sum()
    return inter / (am.sum() + bm.sum() - inter)


def automatic(img: np.ndarray, grid: int = 16, detail: int = 1, min_iou: float = 0.86,
              min_stability: float = 0.92, max_iou: float = 0.7, batch: int = 16) -> np.ndarray:
    """Todas las regiones de la foto, sin filtros por tamaño ni por posición: una propuesta
    por punto de prueba (lo que se distingue de su entorno + cuadrícula grid×grid); quedan
    las de calidad estimada ≥ min_iou y estables (la máscara casi no cambia al mover el
    umbral), sin repetidas (IoU > max_iou con una mejor). Una región dentro de otra (una
    lesión en la hoja) se conserva. detail = 2 suma 2×2 recortes con solape
    analizados a mayor resolución. Devuelve
    etiquetas; las regiones pequeñas quedan encima de las grandes y lo que no cubrió
    ninguna propuesta forma regiones propias."""
    H, W = img.shape[:2]
    f = min(1.0, WORK / max(H, W))
    frame = (max(1, int(round(H * f))), max(1, int(round(W * f))))
    boxes = [((0, 0, W, H), ())]
    if detail >= 2:
        ox, oy = int(W * 0.1), int(H * 0.1)
        xs, ys = (0, W // 2 - ox, W // 2 + ox, W), (0, H // 2 - oy, H // 2 + oy, H)
        for r in range(2):
            for c in range(2):
                edges = (("bottom",) if r == 0 else ("top",)) + (("right",) if c == 0 else ("left",))
                boxes.append(((xs[2 * c], ys[2 * r], xs[2 * c + 1], ys[2 * r + 1]), edges))
    cands = []
    for box, edges in boxes:
        cands += _proposals(img, box, frame, grid, min_iou, min_stability, batch, edges)
    cands.sort(key=lambda c: -c[0])
    kept = []
    for _, r in cands:                                # sin repetidos (IoU de máscaras)
        if all(_iou(r, q) <= max_iou for q in kept):
            kept.append(r)
    labels = np.zeros((H, W), np.int32)
    for n, (y, x, m) in enumerate(sorted(kept, key=lambda r: -r[2].sum()), 1):
        Y0, X0 = int(y / f), int(x / f)
        Y1, X1 = min(H, int((y + m.shape[0]) / f + 0.5)), min(W, int((x + m.shape[1]) / f + 0.5))
        big = cv2.resize(m.astype(np.float32), (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR) > 0.5
        labels[Y0:Y1, X0:X1][big] = n
    return _relabel(_rest(labels))


def _rest(labels: np.ndarray) -> np.ndarray:
    """Lo que ninguna propuesta cubrió (p. ej. hojas muy finas) pasa a ser regiones
    también, una por zona continua; se descartan solo las franjas delgadas que quedan
    entre regiones vecinas (apertura de ~1/512 del lado mayor)."""
    r = max(1, int(round(max(labels.shape) / 512)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    free = cv2.morphologyEx((labels == 0).astype(np.uint8), cv2.MORPH_OPEN, k)
    n, cc, st, _ = cv2.connectedComponentsWithStats(free)
    out, m = labels.copy(), int(labels.max())
    for i in range(1, n):
        if st[i, 4] >= (4 * r) ** 2:
            m += 1
            out[cc == i] = m
    return out


def categorize(labels: np.ndarray, default: int = 1, picks: list | None = None) -> np.ndarray:
    """Categoría de cada región (índice = etiqueta; 0 = sin usar). Todas empiezan en
    `default`; cada clic [x, y, categoría] (0-1) pone esa categoría a la región que está
    en ese punto (la de encima: la más pequeña). Si hay varios clics en la misma región,
    vale el último."""
    n = int(labels.max())
    cats = np.full(n + 1, int(default), np.int32)
    cats[0] = 0
    h, w = labels.shape[:2]
    for x, y, c in picks or []:
        i = int(labels[min(h - 1, max(0, int(y * h))), min(w - 1, max(0, int(x * w)))])
        if i:
            cats[i] = int(c)
    return cats


def _relabel(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(int(labels.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[labels]
