"""Un objeto por región conectada de la máscara, o los objetos que separó el modelo (label)."""
import cv2
import numpy as np

from fenotit.core.pipeline.base import step


def _instances(ctx) -> bool:
    inst = ctx.extra.get("instances")
    if inst is None:
        return False
    labels = np.where(ctx.mask > 0, inst, 0).astype(np.int32)
    from fenotit.core.segmentation.sam import _relabel
    ctx.labels = _relabel(labels)
    return True


@step("label", "processor", requires=("mask",), provides=("labels",), params=[
    {"key": "single", "type": "bool", "default": False},
])
def label(ctx, p):
    """Un objeto por región conectada; single = todo lo que quedó en la máscara es un objeto.
    Si la segmentación ya separó los objetos (modelo de IA), se usan esos."""
    if _instances(ctx):
        return
    if p["single"]:
        ctx.labels = (ctx.mask > 0).astype(np.int32)
        return
    _, ctx.labels = cv2.connectedComponents((ctx.mask > 0).astype(np.uint8), connectivity=8)
