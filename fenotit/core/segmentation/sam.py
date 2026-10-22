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
    while len(_EMB) > 4:
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


def automatic(img: np.ndarray, grid: int = 24, min_iou: float = 0.86, min_stability: float = 0.92,
              max_overlap: float = 0.7, max_area: float = 0.25, batch: int = 64) -> np.ndarray:
    """Todos los objetos: una propuesta por punto de una cuadrícula grid×grid; se quedan
    las de calidad estimada ≥ min_iou y estables (la máscara casi no cambia al mover el
    umbral), sin las que se repiten (solapan > max_overlap con una mejor) ni las que
    cubren más de max_area de la foto (suelen ser el fondo). Devuelve etiquetas."""
    emb, k, orig, pre = embed(img)
    step = 1.0 / grid
    xs = (np.arange(grid) + 0.5) * step
    pts = np.array([(x * pre[1], y * pre[0]) for y in xs for x in xs], np.float32)
    pts = pts[np.random.default_rng(134).permutation(len(pts))]   # los primeros lotes cubren toda la foto
    cands = []                                        # (puntaje, máscara bool en el marco de 256, logits)
    lo_h, lo_w = int(round(pre[0] / 4)), int(round(pre[1] / 4))
    # puntos dentro de un objeto ya propuesto se saltan (no el fondo: sus máscaras abarcan
    # también los objetos pequeños que hay encima)
    covered = np.zeros((lo_h, lo_w), bool)
    i0 = 0
    while i0 < len(pts):
        p = []
        while i0 < len(pts) and len(p) < batch:       # se saltan los puntos que ya caen en una máscara
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
                a = m.sum()
                if a < 4:
                    continue
                if a > max_area * m.size:
                    continue
                cands.append((float(iou[i, j]) * float(hi / lo), m, masks[i, j]))
                covered |= cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    cands.sort(key=lambda c: -c[0])
    kept = []
    for score, m, logits in cands:                    # sin repetidos: NMS por IoU de máscaras
        a = m.sum()
        if any((m & km).sum() / min(a, km.sum()) > max_overlap for km, _ in kept):
            continue
        kept.append((m, logits))
    labels = np.zeros(orig, np.int32)
    order = sorted(kept, key=lambda c: -c[0].sum())   # grandes primero; los chicos encima
    for n, (_, logits) in enumerate(order, 1):
        labels[_upscale(logits, pre, orig)] = n
    return _relabel(labels)


def _relabel(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(int(labels.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[labels]
