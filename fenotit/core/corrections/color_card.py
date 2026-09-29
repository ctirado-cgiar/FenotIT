"""
corrections/color_card.py
Corrección colorimétrica usando tarjeta de colores (PlantCV).
Adaptado de scripts 03/04.
Salida: carpeta colorCorrejidas/ (no sobreescribe originales).
"""

import cv2
import numpy as np
import os
from pathlib import Path

from fenotit import log

_log = log.get("corrections.color_card")


def _plantcv_available() -> bool:
    try:
        import plantcv  # noqa
        return True
    except ImportError:
        return False


def detect_and_save_mask(
    image_path: str,
    radius: int = 6,
    output_mask_path: str | None = None,
) -> tuple[bool, str, np.ndarray | None]:
    """
    Detecta la tarjeta de color en una imagen y guarda la máscara.
    Retorna (success, message, mask).
    """
    if not _plantcv_available():
        return False, "PlantCV no está instalado.", None

    from plantcv import plantcv as pcv
    pcv.params.debug = None

    try:
        raw = Path(image_path).read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False, "No se pudo leer la imagen.", None

        mask = pcv.transform.detect_color_card(rgb_img=img, radius=radius)
        if mask is None:
            return False, "No se detectó la tarjeta de color.", None

        if output_mask_path:
            Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_mask_path, mask)

        return True, "Tarjeta detectada correctamente.", mask

    except Exception as e:
        _log.exception("Error al detectar tarjeta")
        return False, f"Error al detectar tarjeta: {e}", None


def correct_single(
    image_path: str,
    mask: np.ndarray,
    output_folder: str,
    pos: int = 3,
) -> tuple[bool, str]:
    """
    Aplica corrección colorimétrica a una imagen usando la máscara dada.
    pos: posición de la tarjeta estándar (1, 2, 3 o 4 — configurable).
    Guarda en output_folder. Retorna (success, message).
    """
    if not _plantcv_available():
        return False, "PlantCV no está instalado."

    from plantcv import plantcv as pcv
    pcv.params.debug = None

    try:
        raw = Path(image_path).read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False, "No se pudo leer la imagen."

        std_matrix = pcv.transform.std_color_matrix(pos=pos)
        headers, card_matrix = pcv.transform.get_color_matrix(
            rgb_img=img, mask=mask)

        if card_matrix.shape != std_matrix.shape:
            return False, (
                f"Matrices incompatibles: tarjeta {card_matrix.shape} "
                f"vs estándar {std_matrix.shape}. "
                f"Prueba con otra posición (pos=1,2,3,4).")

        corrected = pcv.transform.affine_color_correction(
            rgb_img=img,
            source_matrix=card_matrix,
            target_matrix=std_matrix)

        Path(output_folder).mkdir(parents=True, exist_ok=True)
        out_path = Path(output_folder) / Path(image_path).name
        ext = Path(image_path).suffix.lower()
        ok, buf = cv2.imencode(ext, corrected)
        if ok:
            out_path.write_bytes(buf.tobytes())
            return True, f"Guardado: {out_path.name}"
        return False, "Error al guardar imagen corregida."

    except Exception as e:
        _log.exception("Error corrigiendo color")
        return False, f"Error: {e}"


def correct_batch(
    input_folder: str,
    mask: np.ndarray,
    output_folder: str | None = None,
    pos: int = 3,
    progress_cb=None,
) -> dict:
    """
    Aplica corrección colorimétrica a todas las imágenes de input_folder.
    Guarda en output_folder (por defecto: input_folder/../colorCorrejidas/).
    """
    if output_folder is None:
        output_folder = str(
            Path(input_folder).parent / "colorCorrejidas")

    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in Path(input_folder).iterdir()
                    if p.suffix.lower() in exts)

    results = {"ok": 0, "error": 0,
               "errors": [], "output_folder": output_folder}

    for i, img_path in enumerate(images):
        if progress_cb:
            progress_cb(i + 1, len(images), img_path.name)
        ok, msg = correct_single(str(img_path), mask, output_folder, pos)
        if ok:
            results["ok"] += 1
        else:
            results["error"] += 1
            results["errors"].append(f"{img_path.name}: {msg}")

    return results
