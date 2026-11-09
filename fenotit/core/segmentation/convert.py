"""Convierte un detector de PyTorch (.pt, .pth) a ONNX con las herramientas de su autor,
cuando están instaladas en el mismo Python que FenotIT:

  ultralytics  modelos YOLO (v5u, v8, v11…) entrenados con Ultralytics o descargados de
               Roboflow: `YOLO(pt).export(format="onnx")`; los nombres de las clases vienen
               dentro del .pt
  rfdetr       modelos RF-DETR (Roboflow): `RFDETR*(pretrain_weights=pt).export()`

La conversión corre en un proceso aparte (`python convert.py modelo.pt carpeta`, este archivo
solo usa la biblioteca estándar y la herramienta), que imprime en la última línea un JSON con
el .onnx y las clases. Las herramientas se instalan aparte (`pip install ultralytics` o
`pip install rfdetr`). El Python que convierte es el de FenotIT o el indicado en la
variable de entorno FENOTIT_CONVERT_PYTHON o en la configuración (`convert_python`)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

TOOLS = ("ultralytics", "rfdetr")


def python() -> str | None:
    """El Python que convierte: FENOTIT_CONVERT_PYTHON, `convert_python` de la configuración
    o el de FenotIT (None en la aplicación instalada sin otro Python indicado)."""
    own = os.environ.get("FENOTIT_CONVERT_PYTHON")
    if not own:
        try:
            from fenotit import settings
            own = settings.get("convert_python")
        except Exception:
            own = None
    if own:
        return own
    return None if getattr(sys, "frozen", False) else sys.executable


_FOUND: dict = {}


def tools() -> list[str]:
    """Herramientas de conversión instaladas en ese Python."""
    py = python()
    if not py:
        return []
    if py == sys.executable:
        return [name for name in TOOLS if importlib.util.find_spec(name) is not None]
    if py not in _FOUND:
        code = "import importlib.util as u;print(','.join(n for n in %r if u.find_spec(n)))" % (TOOLS,)
        try:
            out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out = ""
        _FOUND[py] = [n for n in out.split(",") if n]
    return _FOUND[py]


def to_onnx(pt: str, out_dir: str, timeout: int = 1800) -> dict:
    """Convierte `pt` y deja `model.onnx` en `out_dir`. Devuelve {"onnx", "classes", "input",
    "tool"}. RuntimeError con el motivo si ninguna herramienta pudo convertirlo."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    py = python()
    if not py:
        raise RuntimeError("no Python with ultralytics or rfdetr")
    proc = subprocess.run([py, str(Path(__file__).resolve()), str(pt), str(out_dir)],
                          capture_output=True, text=True, timeout=timeout, cwd=str(out_dir))
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise RuntimeError("\n".join(tail) or f"exit code {proc.returncode}")
    out = json.loads(lines[-1])
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _move(src: Path, out_dir: Path) -> str:
    dst = out_dir / "model.onnx"
    if src.resolve() != dst.resolve():
        dst.unlink(missing_ok=True)
        src.replace(dst)
    return str(dst)


def _ultralytics(pt: Path, out_dir: Path) -> dict:
    import shutil
    from ultralytics import YOLO
    work = out_dir / pt.name                          # la exportación queda junto a los pesos: en la copia
    if work.resolve() != pt.resolve():
        shutil.copy2(pt, work)
    model = YOLO(str(work))
    if model.task != "detect":
        raise ValueError(f"task: {model.task} (only detection models give boxes)")
    size = model.overrides.get("imgsz") or 640
    size = int(size[0] if isinstance(size, (list, tuple)) else size)
    path = model.export(format="onnx", imgsz=size, opset=17, dynamic=False, simplify=False)
    names = model.names
    classes = [str(names[k]) for k in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]
    onnx = _move(Path(path), out_dir)
    if work != pt:
        work.unlink(missing_ok=True)
    return {"onnx": onnx, "classes": classes, "input": size, "tool": "ultralytics"}


def _rfdetr(pt: Path, out_dir: Path) -> dict:
    import rfdetr
    import torch
    ckpt = torch.load(str(pt), map_location="cpu", weights_only=False)
    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    classes = list(getattr(args, "class_names", None) or [])
    errors = []
    for name in ("RFDETRBase", "RFDETRMedium", "RFDETRSmall", "RFDETRNano", "RFDETRLarge"):
        cls = getattr(rfdetr, name, None)
        if cls is None:
            continue
        try:
            model = cls(pretrain_weights=str(pt), **({"num_classes": len(classes)} if classes else {}))
            model.export(output_dir=str(out_dir))
        except Exception as e:                         # otra variante del modelo
            errors.append(f"{name}: {e}")
            continue
        found = sorted(out_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
        if found:
            size = int(getattr(model.model, "resolution", 0) or getattr(args, "resolution", 0) or 560)
            return {"onnx": _move(found[-1], out_dir), "classes": classes, "input": size, "tool": "rfdetr"}
    raise ValueError("; ".join(errors) or "no RF-DETR class could load the weights")


def _main(pt: str, out_dir: str):
    pt_path, out = Path(pt), Path(out_dir)
    errors = []
    for name, fn in (("ultralytics", _ultralytics), ("rfdetr", _rfdetr)):
        if importlib.util.find_spec(name) is None:
            continue
        try:
            print(json.dumps(fn(pt_path, out)), flush=True)
            return
        except Exception as e:
            errors.append(f"{name}: {e}")
    if not errors:
        errors.append("ultralytics and rfdetr are not installed")
    print(json.dumps({"error": " | ".join(errors)}), flush=True)


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2])
