"""
corrections/aruco.py
Corrección de perspectiva usando marcadores ArUco.
DICT_4X4_50, IDs 0-3.
Salida: imagen en tamaño y proporciones originales (sin redimensionamiento).
No sobreescribe originales.
"""

import cv2
import numpy as np
from pathlib import Path


# ── Detección de ArUcos ───────────────────────────────────────────────────────

def _get_aruco_dict():
    """Retorna el diccionario ArUco compatible con la versión de OpenCV."""
    try:
        # OpenCV 4.7+
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:
        return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


def _get_aruco_params():
    try:
        return cv2.aruco.DetectorParameters()
    except AttributeError:
        return cv2.aruco.DetectorParameters_create()


def detect_aruco_corners(image: np.ndarray) -> dict[int, np.ndarray]:
    """
    Detecta marcadores ArUco en la imagen.
    Retorna dict {id: center_point} para IDs 0-3.
    """
    aruco_dict   = _get_aruco_dict()
    aruco_params = _get_aruco_params()

    try:
        # OpenCV 4.7+
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(image)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image, aruco_dict, parameters=aruco_params)

    result = {}
    if ids is None:
        return result

    for i, marker_id in enumerate(ids.flatten()):
        if 0 <= marker_id <= 3:
            # Centro del marcador
            c = corners[i][0]
            result[int(marker_id)] = c.mean(axis=0)  # [x, y]

    return result


def correct_perspective(
    image: np.ndarray,
    margin: int = 20,
) -> tuple[np.ndarray | None, str]:
    """
    Corrige perspectiva usando los 4 marcadores ArUco (IDs 0-3).
    El orden esperado:
        ID 0 → esquina superior izquierda
        ID 1 → esquina superior derecha
        ID 2 → esquina inferior derecha
        ID 3 → esquina inferior izquierda

    La imagen de salida tiene el mismo tamaño que la entrada.
    margin: píxeles de margen interior desde los centros de los marcadores.

    Retorna (imagen_corregida, mensaje).
    """
    corners = detect_aruco_corners(image)

    missing = [i for i in range(4) if i not in corners]
    if missing:
        return None, f"No se detectaron marcadores: IDs {missing}"

    # Puntos de origen (en imagen original)
    src = np.float32([
        corners[0],  # TL
        corners[1],  # TR
        corners[2],  # BR
        corners[3],  # BL
    ])

    h, w = image.shape[:2]

    # Puntos de destino — misma resolución que la imagen original
    dst = np.float32([
        [margin,     margin    ],   # TL
        [w - margin, margin    ],   # TR
        [w - margin, h - margin],   # BR
        [margin,     h - margin],   # BL
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    corrected = cv2.warpPerspective(image, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))

    return corrected, "OK"


def correct_batch(
    input_folder: str,
    output_folder: str | None = None,
    margin: int = 20,
    progress_cb=None,
) -> dict:
    """
    Aplica corrección de perspectiva ArUco a todas las imágenes de una carpeta.
    Guarda en output_folder (default: input_folder/../perspectivaCorregida/).
    No sobreescribe originales.
    """
    if output_folder is None:
        output_folder = str(
            Path(input_folder).parent / "perspectivaCorregida")
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in Path(input_folder).iterdir()
                    if p.suffix.lower() in exts)

    results = {
        "ok": 0, "error": 0, "no_markers": 0,
        "errors": [], "output_folder": output_folder
    }

    for i, img_path in enumerate(images):
        if progress_cb:
            progress_cb(i + 1, len(images), img_path.name)

        try:
            raw = img_path.read_bytes()
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                results["error"] += 1
                results["errors"].append(f"{img_path.name}: no se pudo leer")
                continue

            corrected, msg = correct_perspective(img, margin=margin)

            if corrected is None:
                results["no_markers"] += 1
                results["errors"].append(f"{img_path.name}: {msg}")
                continue

            out_path = Path(output_folder) / img_path.name
            ext      = img_path.suffix.lower()
            ok, buf  = cv2.imencode(ext, corrected)
            if ok:
                out_path.write_bytes(buf.tobytes())
                results["ok"] += 1
            else:
                results["error"] += 1

        except Exception as e:
            results["error"] += 1
            results["errors"].append(f"{img_path.name}: {e}")

    return results


def preview_detection(image: np.ndarray) -> np.ndarray:
    """
    Genera imagen de preview mostrando los ArUcos detectados.
    Útil para verificar antes de aplicar la corrección.
    """
    preview = image.copy()
    corners_dict = detect_aruco_corners(image)

    colors = {
        0: (0, 200, 80),    # verde  — TL
        1: (0, 120, 255),   # azul   — TR
        2: (0, 60, 200),    # azul oscuro — BR
        3: (200, 160, 0),   # amarillo — BL
    }
    labels = {0: "ID0 TL", 1: "ID1 TR", 2: "ID2 BR", 3: "ID3 BL"}

    for mid, center in corners_dict.items():
        cx, cy = int(center[0]), int(center[1])
        color  = colors.get(mid, (128, 128, 128))
        cv2.circle(preview, (cx, cy), 10, color, -1)
        cv2.circle(preview, (cx, cy), 12, (255, 255, 255), 2)
        cv2.putText(preview, labels.get(mid, f"ID{mid}"),
                    (cx + 14, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (30, 30, 30), 2, cv2.LINE_AA)

    # Dibujar línea conectando los 4 si están todos presentes
    if len(corners_dict) == 4:
        pts = np.int32([corners_dict[i] for i in range(4)])
        cv2.polylines(preview, [pts], isClosed=True,
                      color=(100, 100, 100), thickness=1,
                      lineType=cv2.LINE_AA)

    detected = len(corners_dict)
    msg = (f"ArUcos detectados: {detected}/4"
           if detected < 4
           else "✓ 4 ArUcos detectados — listo para corregir")
    cv2.putText(preview, msg, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 150, 60) if detected == 4 else (0, 60, 200),
                2, cv2.LINE_AA)

    return preview
