"""Afinar máscara: operaciones morfológicas sobre la máscara, en el orden que elige el
usuario (erosión, dilatación, apertura, cierre, apertura por reconstrucción, relleno de
huecos y suavizado). El tamaño es el diámetro del elemento circular en px."""
import cv2
import numpy as np

from fenotit.core.pipeline.base import step

# operaciones de "Afinar máscara" (tamaño en px del elemento circular; "veces" = iteraciones)
REFINE_OPS = {
    "erode": {"size": True},          # adelgaza: quita puntos y uniones finas
    "dilate": {"size": True},         # engrosa: une partes cercanas
    "open": {"size": True},           # erosión + dilatación: quita ruido sin encoger lo grande
    "close": {"size": True},          # dilatación + erosión: cierra grietas y huecos pequeños
    "open_rec": {"size": True},       # apertura por reconstrucción: quita lo pequeño sin tocar
                                      # el borde de lo que queda
    "fill_holes": {"size": False},    # rellena huecos dentro de los objetos
    "median": {"size": True},         # suaviza el borde (filtro de mediana)
}

DEFAULT_REFINE = [{"op": "fill_holes"}]


def _kernel(size: int):
    size = max(1, int(size)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def refine_op(mask: np.ndarray, op: str, size: int = 3, iterations: int = 1) -> np.ndarray:
    it = max(1, int(iterations))
    if op in ("erode", "dilate", "open", "close"):
        code = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE,
                "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}[op]
        return cv2.morphologyEx(mask, code, _kernel(size), iterations=it)
    if op == "open_rec":
        seed = cv2.erode(mask, _kernel(size), iterations=it)
        return reconstruct(seed, mask)
    if op == "fill_holes":
        from fenotit.core.objects.mask import _fill_small_holes
        return _fill_small_holes(mask)
    if op == "median":
        k = max(3, int(size) | 1)
        out = mask
        for _ in range(it):
            out = cv2.medianBlur(out, min(k, 255))
        return out
    return mask


def reconstruct(seed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reconstrucción morfológica por dilatación, rápida: se quedan las componentes de la
    máscara que tocan alguna semilla (equivale a dilatar hasta estabilizar)."""
    n, lab = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[seed > 0])] = True
    keep[0] = False
    return keep[lab].astype(np.uint8) * 255


def refine(mask: np.ndarray, ops: list[dict] | None, scale: float = 1.0) -> np.ndarray:
    """Aplica las operaciones en orden. `scale` < 1 en la vista previa reducida."""
    for o in ops or []:
        op = o.get("op")
        if op not in REFINE_OPS:
            continue
        size = max(1, round(int(o.get("size", 3)) * scale)) if REFINE_OPS[op]["size"] else 0
        mask = refine_op(mask, op, size, int(o.get("iter", 1)))
    return mask


@step("refine", "processor", requires=("mask",), provides=("mask",), params=[
    {"key": "ops", "type": "list", "default": []},
])
def refine_step(ctx, p):
    """Afinar máscara: operaciones morfológicas en el orden que eligió el usuario."""
    ctx.mask = refine(ctx.mask, p["ops"])
