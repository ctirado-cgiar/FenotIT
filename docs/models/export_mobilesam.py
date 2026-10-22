import sys, torch, torch.nn as nn
sys.path.insert(0, "MobileSAM")
from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel
sam = sam_model_registry["vit_t"](checkpoint="MobileSAM/weights/mobile_sam.pt").eval()

class Enc(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m.image_encoder(x)

class Dec(nn.Module):
    """Sin máscara de entrada; devuelve las 4 máscaras de baja resolución (256×256) y su
    calidad estimada (IoU) por cada juego de puntos."""
    def __init__(s, m):
        super().__init__(); s.m = m; s.o = SamOnnxModel(m, return_single_mask=False)
    def forward(s, emb, coords, labels):
        sparse = s.o._embed_points(coords, labels)
        dense = s.m.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            coords.shape[0], -1, emb.shape[2], emb.shape[3])
        masks, iou = s.m.mask_decoder.predict_masks(emb, s.m.prompt_encoder.get_dense_pe(), sparse, dense)
        return masks, iou

with torch.no_grad():
    x = torch.randn(1, 3, 1024, 1024)
    torch.onnx.export(Enc(sam), x, "mobilesam_encoder.onnx", input_names=["image"], output_names=["embeddings"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    emb = sam.image_encoder(x)
    c = torch.randint(0, 1024, (3, 2, 2)).float(); l = torch.tensor([[1, -1]] * 3).float()
    torch.onnx.export(Dec(sam), (emb, c, l), "mobilesam_decoder.onnx", input_names=["embeddings", "coords", "labels"],
                      output_names=["masks", "iou"], opset_version=17, do_constant_folding=True, dynamo=False,
                      dynamic_axes={"coords": {0: "b", 1: "n"}, "labels": {0: "b", 1: "n"}, "masks": {0: "b"}, "iou": {0: "b"}})
    print("torch masks", Dec(sam)(emb, c, l)[1][0])
import onnxruntime as ort, numpy as np
d = ort.InferenceSession("mobilesam_decoder.onnx")
print("onnx", d.run(None, {"embeddings": emb.numpy(), "coords": c.numpy(), "labels": l.numpy()})[1][0])
