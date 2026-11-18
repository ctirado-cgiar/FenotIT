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
    for root in roots:
        if root is None:
            continue
        folder = root / "models" / "mobilesam"
        if all((folder / file_name).exists() for file_name in FILES):
            return folder
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
        folder = model_dir()
        if folder is None:
            raise FileNotFoundError("MobileSAM model files not found (models/mobilesam)")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.enable_cpu_mem_arena = False             # sin la reserva de ORT: ~1 GB menos
        _SESSIONS["enc"] = ort.InferenceSession(str(folder / FILES[0]), opts, providers=["CPUExecutionProvider"])
        _SESSIONS["dec"] = ort.InferenceSession(str(folder / FILES[1]), opts, providers=["CPUExecutionProvider"])
    return _SESSIONS["enc"], _SESSIONS["dec"]


def _key(img: np.ndarray) -> str:
    import hashlib
    hasher = hashlib.blake2b(digest_size=12)
    hasher.update(str(img.shape).encode())
    hasher.update(np.ascontiguousarray(img[::5, ::5]).tobytes())
    return hasher.hexdigest()


def embed(img: np.ndarray):
    """Embeddings de la foto (BGR) y su escala; se guardan los de las 4 últimas fotos."""
    key = _key(img)
    if key in _EMB:
        _EMB.move_to_end(key)
        return _EMB[key]
    enc, _ = _sessions()
    height, width = img.shape[:2]
    enc_scale = SIZE / max(height, width)
    new_height, new_width = int(height * enc_scale + 0.5), int(width * enc_scale + 0.5)
    rgb = cv2.cvtColor(cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = np.zeros((SIZE, SIZE, 3), np.float32)
    x[:new_height, :new_width] = (rgb.astype(np.float32) - MEAN) / STD
    emb = enc.run(None, {"image": x.transpose(2, 0, 1)[None]})[0]
    out = (emb, enc_scale, (height, width), (new_height, new_width))
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
    emb, enc_scale, orig, pre = embed(img)
    height, width = orig
    pos = [(x * width * enc_scale, y * height * enc_scale) for x, y, positive in points if positive]
    neg = [(x * width * enc_scale, y * height * enc_scale) for x, y, positive in points if not positive]
    labels = np.zeros(orig, np.int32)
    if not pos:
        return labels
    coords = np.array([[point] + neg + [(0, 0)] for point in pos], np.float32)
    lab = np.array([[1] + [0] * len(neg) + [-1] for _ in pos], np.float32)
    masks, iou = _decode(emb, coords, lab)
    for i in range(len(pos)):
        j = 1 + int(np.argmax(iou[i, 1:])) if not neg else 0      # un clic: la mejor de 3
        mask = _upscale(masks[i, j], pre, orig)
        labels[mask & (labels == 0)] = i + 1
    return labels


def from_boxes(img: np.ndarray, boxes: list, batch: int = 16) -> np.ndarray:
    """Contorno del objeto dentro de cada caja (x0, y0, x1, y1 en px de la foto): la caja es
    la indicación para MobileSAM y la máscara se recorta a la caja con un margen del 5 %.
    Devuelve etiquetas (id = posición + 1); los objetos pequeños quedan encima de los grandes."""
    labels = np.zeros(img.shape[:2], np.int32)
    if not len(boxes):
        return labels
    emb, enc_scale, orig, pre = embed(img)
    height, width = orig
    masks = []
    for start in range(0, len(boxes), batch):
        part = boxes[start:start + batch]
        coords = np.array([[[x0 * enc_scale, y0 * enc_scale], [x1 * enc_scale, y1 * enc_scale]] for x0, y0, x1, y1 in part], np.float32)
        lab = np.tile(np.array([[2, 3]], np.float32), (len(part), 1))
        low, _ = _decode(emb, coords, lab)
        masks += [low[i, 0] for i in range(len(part))]
    order = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
    for i in order:
        x0, y0, x1, y1 = boxes[i]
        margin_x, margin_y = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
        left, top = max(0, int(x0 - margin_x)), max(0, int(y0 - margin_y))
        right, bottom = min(width, int(np.ceil(x1 + margin_x))), min(height, int(np.ceil(y1 + margin_y)))
        if right <= left or bottom <= top:
            continue
        big = cv2.resize(masks[i], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        py0, px0 = int(top * enc_scale), int(left * enc_scale)
        py1, px1 = max(py0 + 1, int(np.ceil(bottom * enc_scale))), max(px0 + 1, int(np.ceil(right * enc_scale)))
        mask = cv2.resize(big[py0:py1, px0:px1], (right - left, bottom - top), interpolation=cv2.INTER_LINEAR) > 0
        labels[top:bottom, left:right][mask] = i + 1
    return labels


WORK = 2048          # lado mayor del marco común donde se comparan las regiones


def _seeds(img: np.ndarray, pre: tuple, grid: int, max_seeds: int = 600) -> np.ndarray:
    """Puntos de prueba en el marco del codificador: primero los lugares que más se
    distinguen de su entorno (color Lab frente a un fondo muy suavizado: semillas, hojas,
    lesiones, terrones…), luego una cuadrícula gruesa para lo grande y lo poco contrastado."""
    lo_h, lo_w = int(round(pre[0] / 4)), int(round(pre[1] / 4))
    small = cv2.resize(img, (lo_w * 2, lo_h * 2), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = cv2.GaussianBlur(lab, (0, 0), max(lo_w, lo_h) / 8)
    contrast = cv2.GaussianBlur(np.sqrt(((lab - background) ** 2).sum(2)), (0, 0), 1.5)
    peak = (contrast == cv2.dilate(contrast, np.ones((7, 7), np.uint8))) & (contrast > max(6.0, float(np.percentile(contrast, 60))))
    peak_y, peak_x = np.nonzero(peak)
    order = np.argsort(-contrast[peak_y, peak_x])[:max_seeds]
    seeds = np.stack([peak_x[order] * 2 + 1, peak_y[order] * 2 + 1], 1).astype(np.float32).reshape(-1, 2)
    steps = (np.arange(grid) + 0.5) / grid
    grid_points = np.array([(x * pre[1], y * pre[0]) for y in steps for x in steps], np.float32)
    grid_points = grid_points[np.random.default_rng(134).permutation(len(grid_points))]
    return np.concatenate([seeds, grid_points])


def _proposals(img, box, frame, grid, min_iou, min_stability, batch, inner_edges):
    """Regiones propuestas en un recorte box = (x0, y0, x1, y1) de la foto, pasadas al marco
    común `frame` = (alto, ancho). Devuelve [(puntaje, (y0, x0, sub_máscara))]. Las que tocan
    un borde interior del recorte (objeto cortado) se descartan."""
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    emb, enc_scale, orig, pre = embed(crop)
    lo_h, lo_w = int(round(pre[0] / 4)), int(round(pre[1] / 4))
    height, width = img.shape[:2]
    scale_y, scale_x = frame[0] / height, frame[1] / width
    fx0, fy0 = int(round(x0 * scale_x)), int(round(y0 * scale_y))
    frame_width, frame_height = max(1, int(round(x1 * scale_x)) - fx0), max(1, int(round(y1 * scale_y)) - fy0)
    pts = _seeds(crop, pre, grid)
    covered = np.zeros((lo_h, lo_w), bool)
    out, found, cursor = [], [], 0
    while cursor < len(pts):
        batch_points = []
        while cursor < len(pts) and len(batch_points) < batch:       # se saltan los puntos que ya caen en una región
            x, y = pts[cursor]
            cursor += 1
            if not covered[min(int(y / 4), lo_h - 1), min(int(x / 4), lo_w - 1)]:
                batch_points.append((x, y))
        if not batch_points:
            break
        batch_points = np.array(batch_points, np.float32)
        coords = np.concatenate([batch_points[:, None, :], np.zeros((len(batch_points), 1, 2), np.float32)], axis=1)
        lab = np.tile(np.array([[1, -1]], np.float32), (len(batch_points), 1))
        masks, iou = _decode(emb, coords, lab)
        for i in range(len(batch_points)):
            for j in (1, 2, 3):
                if iou[i, j] < min_iou:
                    continue
                logits = masks[i, j][:lo_h, :lo_w]
                strict, loose = (logits > 1.0).sum(), (logits > -1.0).sum()
                if loose == 0 or strict / loose < min_stability:
                    continue
                mask = logits > 0
                if mask.sum() < 4:
                    continue
                if _touches(mask, inner_edges):
                    continue
                if mask.sum() < 0.25 * mask.size:           # las enormes (fondo) abarcan objetos: no tapan puntos
                    covered |= cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                score = float(iou[i, j]) * float(strict / loose)
                region = _box(mask)
                dup = next((index for index, entry in enumerate(found) if _iou(region, entry[1]) > 0.85), None)
                if dup is None:
                    found.append((score, region, logits.copy()))
                elif score > found[dup][0]:           # repetida: queda la mejor
                    found[dup] = (score, region, logits.copy())
    for score, _, logits in found:                         # al marco común
        big = cv2.resize(logits, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR) > 0
        region = _box(big)
        if region is not None:
            out.append((score, (fy0 + region[0], fx0 + region[1], region[2])))
    return out


def _box(mask: np.ndarray):
    pixel_y, pixel_x = np.nonzero(mask)
    if not len(pixel_y):
        return None
    return (int(pixel_y.min()), int(pixel_x.min()), mask[pixel_y.min():pixel_y.max() + 1, pixel_x.min():pixel_x.max() + 1])


def _touches(mask: np.ndarray, edges) -> bool:
    side = {"top": mask[0], "bottom": mask[-1], "left": mask[:, 0], "right": mask[:, -1]}
    return any(side[edge].any() for edge in edges)


def _iou(region_a, region_b) -> float:
    top_a, left_a, mask_a = region_a
    top_b, left_b, mask_b = region_b
    y0, x0 = max(top_a, top_b), max(left_a, left_b)
    y1, x1 = min(top_a + mask_a.shape[0], top_b + mask_b.shape[0]), min(left_a + mask_a.shape[1], left_b + mask_b.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0.0
    inter = (mask_a[y0 - top_a:y1 - top_a, x0 - left_a:x1 - left_a] & mask_b[y0 - top_b:y1 - top_b, x0 - left_b:x1 - left_b]).sum()
    return inter / (mask_a.sum() + mask_b.sum() - inter)


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
    height, width = img.shape[:2]
    factor = min(1.0, WORK / max(height, width))
    frame = (max(1, int(round(height * factor))), max(1, int(round(width * factor))))
    boxes = [((0, 0, width, height), ())]
    if detail >= 2:
        overlap_x, overlap_y = int(width * 0.1), int(height * 0.1)
        cuts_x, cuts_y = (0, width // 2 - overlap_x, width // 2 + overlap_x, width), (0, height // 2 - overlap_y, height // 2 + overlap_y, height)
        for row in range(2):
            for col in range(2):
                edges = (("bottom",) if row == 0 else ("top",)) + (("right",) if col == 0 else ("left",))
                boxes.append(((cuts_x[2 * col], cuts_y[2 * row], cuts_x[2 * col + 1], cuts_y[2 * row + 1]), edges))
    cands = []
    for box, edges in boxes:
        cands += _proposals(img, box, frame, grid, min_iou, min_stability, batch, edges)
    cands.sort(key=lambda candidate: -candidate[0])
    kept = []
    for _, region in cands:                           # sin repetidos (IoU de máscaras)
        if all(_iou(region, other) <= max_iou for other in kept):
            kept.append(region)
    labels = np.zeros((height, width), np.int32)
    for label_id, (y, x, mask) in enumerate(sorted(kept, key=lambda region: -region[2].sum()), 1):
        top, left = int(y / factor), int(x / factor)
        bottom, right = min(height, int((y + mask.shape[0]) / factor + 0.5)), min(width, int((x + mask.shape[1]) / factor + 0.5))
        big = cv2.resize(mask.astype(np.float32), (right - left, bottom - top), interpolation=cv2.INTER_LINEAR) > 0.5
        labels[top:bottom, left:right][big] = label_id
    return _relabel(_rest(labels))


def _rest(labels: np.ndarray) -> np.ndarray:
    """Lo que ninguna propuesta cubrió (p. ej. hojas muy finas) pasa a ser regiones
    también, una por zona continua; se descartan solo las franjas delgadas que quedan
    entre regiones vecinas (apertura de ~1/512 del lado mayor)."""
    radius = max(1, int(round(max(labels.shape) / 512)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    free = cv2.morphologyEx((labels == 0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n_free, free_labels, free_stats, _ = cv2.connectedComponentsWithStats(free)
    out, last_id = labels.copy(), int(labels.max())
    for i in range(1, n_free):
        if free_stats[i, 4] >= (4 * radius) ** 2:
            last_id += 1
            out[free_labels == i] = last_id
    return out


def categorize(labels: np.ndarray, default: int = 1, picks: list | None = None) -> np.ndarray:
    """Categoría de cada región (índice = etiqueta; 0 = sin usar). Todas empiezan en
    `default`; cada clic [x, y, categoría] (0-1) pone esa categoría a la región que está
    en ese punto (la de encima: la más pequeña). Si hay varios clics en la misma región,
    vale el último."""
    n_regions = int(labels.max())
    cats = np.full(n_regions + 1, int(default), np.int32)
    cats[0] = 0
    height, width = labels.shape[:2]
    for x, y, category in picks or []:
        i = int(labels[min(height - 1, max(0, int(y * height))), min(width - 1, max(0, int(x * width)))])
        if i:
            cats[i] = int(category)
    return cats


def _relabel(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(int(labels.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[labels]


# ── Pieza del pipeline ──────────────────────────────────────────────────────

# El decorador step registra MobileSAM como pieza; BACKGROUND_COLOR es el color del fondo en la
# vista de categorías.
from fenotit.core.pipeline.base import BACKGROUND_COLOR, step  # noqa: E402


@step("model_segment", "segmenter", provides=("mask",), params=[
    {"key": "model", "type": "str", "default": "mobilesam"},
    {"key": "mode", "type": "choice", "choices": ["auto", "points"], "default": "auto"},
    {"key": "grid", "type": "int", "default": 16, "min": 8, "max": 64},
    {"key": "points", "type": "list", "default": []},
    {"key": "categories", "type": "list", "default": []},
    {"key": "colors", "type": "list", "default": []},
    {"key": "default", "type": "int", "default": 1, "min": 0, "max": 99},
    {"key": "picks", "type": "list", "default": []},
    {"key": "own", "type": "bool", "default": True},
    {"key": "learn", "type": "dict", "default": {}},
])
def model_segment(ctx, params):
    """Segmentación con un modelo de IA (MobileSAM). Cada objeto sale ya separado: las
    etiquetas quedan en ctx.extra["instances"] y el paso de objetos las respeta.
    Automático: cada región tiene una categoría (default, o la que le dio el usuario con un
    clic, picks; o, en fotos sin clics, la que aprende `learn` de las otras); las de
    categoría 0 no se usan. ctx.classes = categoría de cada píxel."""
    from fenotit.core.segmentation import sam
    if params["mode"] == "points":
        labels = sam.from_points(ctx.image, list(params["points"] or []))
    else:
        labels = sam.automatic(ctx.image, grid=int(params["grid"]))
        learn = params.get("learn") or {}
        if not params.get("own", True) and learn.get("samples"):
            # foto sin clics: categorías aprendidas de las regiones que el usuario clasificó en otras
            from fenotit.core.segmentation import regions
            cats = regions.predict(regions.load(learn["samples"], learn["hash"]), ctx.image, labels)
        else:
            cats = sam.categorize(labels, int(params["default"]), list(params["picks"] or []))
        if params["categories"]:
            ctx.classes = cats[labels].astype(np.int32)
            ctx.extra["class_names"] = list(params["categories"])
            ctx.extra["class_colors"] = [BACKGROUND_COLOR] + list(params["colors"])
        labels = np.where(cats[labels] > 0, labels, 0)
    ctx.extra["instances"] = labels
    ctx.mask = (labels > 0).astype(np.uint8) * 255
    ctx.images["mask"] = ctx.mask
