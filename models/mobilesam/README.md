# MobileSAM (ONNX)

Segment Anything ligero (codificador TinyViT + decodificador de SAM) exportado a ONNX
para FenotIT con `docs/models/export_mobilesam.py`.

- Fuente: https://github.com/ChaoningZhang/MobileSAM (pesos `weights/mobile_sam.pt`)
- Licencia: Apache 2.0 (ver `LICENSE`)
- Cita: Zhang, C., Han, D., Qiao, Y., Kim, J. U., Bae, S.-H., Lee, S., Hong, C. S. (2023).
  Faster Segment Anything: Towards Lightweight SAM for Mobile Applications. arXiv:2306.14289.
- `mobilesam_encoder.onnx`: imagen RGB normalizada 1×3×1024×1024 → embeddings 1×256×64×64
- `mobilesam_decoder.onnx`: embeddings + puntos (B×N×2, en el marco de 1024) y etiquetas
  (B×N: 1 objeto, 0 fondo, -1 relleno) → 4 máscaras 256×256 (logits) y su IoU estimada por juego de puntos
