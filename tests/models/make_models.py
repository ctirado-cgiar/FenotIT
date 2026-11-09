"""Genera los detectores ONNX de prueba (salida fija, sin aprender nada) para
tests/test_detect.py: uno con la salida de YOLOv8, otro con la de YOLOv5 y otro con la de
RF-DETR, que encuentran los objetos de `scene()` en una foto de 800×600. Requiere el
paquete onnx (solo para generar los archivos).

    python tests/models/make_models.py
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W, H, S = 800, 600, 640
# objetos de la escena: (x0, y0, x1, y1, clase)
OBJECTS = [(100, 100, 220, 180, 0), (400, 300, 470, 420, 1), (600, 80, 700, 160, 0)]


def scene() -> np.ndarray:
    """Foto BGR de 800×600: fondo oscuro y una elipse clara dentro de cada caja."""
    import cv2
    img = np.full((H, W, 3), 40, np.uint8)
    for x0, y0, x1, y1, c in OBJECTS:
        color = (60, 200, 230) if c == 0 else (80, 120, 240)
        cv2.ellipse(img, ((x0 + x1) // 2, (y0 + y1) // 2), ((x1 - x0) // 2 - 4, (y1 - y0) // 2 - 4), 0, 0, 360,
                    color, -1)
    return img


def _graph(name, outputs, meta=None):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, S, S])
    nodes = [helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
             helper.make_node("Mul", ["m", "zero"], ["z"])]
    inits = [numpy_helper.from_array(np.array(0, np.float32), "zero")]
    outs = []
    for oname, arr in outputs:
        inits.append(numpy_helper.from_array(arr.astype(np.float32), f"{oname}_c"))
        nodes.append(helper.make_node("Add", [f"{oname}_c", "z"], [oname]))
        outs.append(helper.make_tensor_value_info(oname, TensorProto.FLOAT, list(arr.shape)))
    model = helper.make_model(helper.make_graph(nodes, name, [inp], outs, inits),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    for k, v in (meta or {}).items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.save(model, str(HERE / f"{name}.onnx"))


def main():
    r = min(S / W, S / H)
    py = (S - H * r) / 2
    rows = []
    for x0, y0, x1, y1, c in OBJECTS:
        cx, cy, bw, bh = (x0 + x1) / 2 * r, (y0 + y1) / 2 * r + py, (x1 - x0) * r, (y1 - y0) * r
        rows.append((cx, cy, bw, bh, c, 0.9))
    x0, y0, x1, y1, c = OBJECTS[0]                      # repetida, menos segura: la quita NMS
    rows.append(((x0 + x1) / 2 * r + 3, (y0 + y1) / 2 * r + py, (x1 - x0) * r, (y1 - y0) * r, c, 0.6))
    rows.append((300, 300, 50, 50, 1, 0.1))            # poco segura: la quita la confianza
    n = 10
    v8 = np.zeros((1, 6, n))
    v5 = np.zeros((1, n, 7))
    for i, (cx, cy, bw, bh, c, s) in enumerate(rows):
        v8[0, :4, i] = (cx, cy, bw, bh)
        v8[0, 4 + c, i] = s
        v5[0, i, :5] = (cx, cy, bw, bh, 1.0)
        v5[0, i, 5 + c] = s
    _graph("yolov8_test", [("output0", v8)], {"names": "{0: 'seed', 1: 'pod'}", "task": "detect",
                                              "description": "Test detector for seeds and pods",
                                              "author": "FenotIT tests", "license": "CC0-1.0",
                                              "docs": "https://github.com/ctirado-cgiar/FenotIT"})
    _graph("yolov5_test", [("output0", v5)])
    dets = np.zeros((1, n, 4))
    logits = np.full((1, n, 3), -10.0)
    for i, (x0, y0, x1, y1, c) in enumerate(OBJECTS):
        dets[0, i] = ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H)
        logits[0, i, 1 + c] = 4.0
    _graph("rfdetr_test", [("dets", dets), ("labels", logits)])
    (HERE / "data.yaml").write_text("names: ['seed', 'pod']\n", encoding="utf-8")


if __name__ == "__main__":
    main()
