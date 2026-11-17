"""Fotos de ejemplo de un montaje de captura completo y su preset.

Cada foto simula una bandeja de semillas (fotos de tests/data/images) sobre un tablero con cuatro
marcadores ArUco de separación conocida y una tarjeta de color, tomada con una cámara inclinada,
un lente con distorsión de barril y una luz distinta. El lente se calibra con vistas simuladas
de un tablero de ajedrez, con la misma calibración que usa la aplicación.

    python examples/capture_setup/make_photos.py

Escribe photo_1.jpg … photo_3.jpg, capture_setup.fntp y truth.csv (lo que se mide
en el tablero sin cámara, para comparar).
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from fenotit.core import preset                                    # noqa: E402
from fenotit.core.analysis.registry import ANALYSES                # noqa: E402
from fenotit.core.corrections import colorcard                     # noqa: E402
from fenotit.core.corrections.distortion import calibrate          # noqa: E402
from fenotit.core.project import Project                           # noqa: E402

MM_PER_BOARD_PX = 0.1
BOARD_WIDTH, BOARD_HEIGHT = 2400, 3400           # tablero: bandeja de 2400 px y dos franjas de 500 px
STRIP = 500
MARKER_SIZE, MARKER_PAD = 220, 24
MARKER_CENTERS = {0: (150, 150), 1: (2250, 150), 2: (2250, 3250), 3: (150, 3250)}   # TL, TR, BR, BL
CHIP, CHIP_GAP = 60, 12
CARD_TOP = 190                                   # la tarjeta queda dentro del rectángulo de los marcadores
BOARD_COLOR = (225, 225, 225)

PHOTO_WIDTH, PHOTO_HEIGHT = 2400, 3000
FOCAL = 2600.0
CAMERA = np.array([[FOCAL, 0, PHOTO_WIDTH / 2], [0, FOCAL, PHOTO_HEIGHT / 2], [0, 0, 1]])
LENS = np.array([-0.12, 0.03, 0.0, 0.0, 0.0])    # distorsión de barril
CAMERA_DISTANCE = 3700.0

# foto de la bandeja, giro de la cámara (rad, ejes x, y, z) y ganancias de la luz (R, G, B)
SHOTS = [("23379.jpg", (0.16, -0.05, 0.03), (1.10, 1.00, 0.80)),
         ("23505.jpg", (-0.12, 0.10, -0.04), (0.85, 1.00, 1.12)),
         ("01.jpg", (0.08, 0.14, 0.06), (1.00, 0.90, 1.05))]

LIGHT_MIX = np.array([[0.94, 0.04, 0.02], [0.04, 0.92, 0.04], [0.02, 0.06, 0.92]], np.float32)

ANALYSIS_AREA = [[0.03, 0.13], [0.97, 0.87]]      # la bandeja, sin las franjas de marcadores y tarjeta
MEASURE_ALL = {"touching": True, "exclude_border": True, "measure_count": True, "measure_size": True,
               "measure_shape": True, "measure_color": True, "color_mode": "object", "n_colors": 3}
SEGMENTATION = {"method": "threshold", "color_space": "LAB", "channel": 2, "auto": True,
                "refine": [{"op": "open", "size": 3, "iter": 1}, {"op": "fill_holes"}]}


def color_card() -> np.ndarray:
    """Tarjeta de 24 colores con los valores de referencia, sobre fondo negro."""
    rows, cols = colorcard.COLORCHECKER24.shape[:2]
    card = np.full((rows * CHIP + (rows + 1) * CHIP_GAP, cols * CHIP + (cols + 1) * CHIP_GAP, 3), 25, np.uint8)
    for row in range(rows):
        for col in range(cols):
            top, left = CHIP_GAP + row * (CHIP + CHIP_GAP), CHIP_GAP + col * (CHIP + CHIP_GAP)
            rgb = np.round(colorcard.COLORCHECKER24[row, col] * 255)
            card[top:top + CHIP, left:left + CHIP] = rgb[::-1]
    return card


def board(tray_photo: Path) -> np.ndarray:
    """Tablero plano: la bandeja en el centro, marcadores en las esquinas y la tarjeta arriba,
    por debajo de la línea de los marcadores (la corrección de perspectiva recorta ahí)."""
    tray = cv2.imread(str(tray_photo))
    height, width = tray.shape[:2]
    side = min(height, width, BOARD_WIDTH)
    top, left = (height - side) // 2, (width - side) // 2
    square = cv2.resize(tray[top:top + side, left:left + side], (BOARD_WIDTH, BOARD_WIDTH), interpolation=cv2.INTER_AREA)
    scene = np.full((BOARD_HEIGHT, BOARD_WIDTH, 3), BOARD_COLOR, np.uint8)
    scene[STRIP:STRIP + BOARD_WIDTH] = square
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    half = MARKER_SIZE // 2
    for marker_id, (cx, cy) in MARKER_CENTERS.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIZE)
        scene[cy - half - MARKER_PAD:cy + half + MARKER_PAD, cx - half - MARKER_PAD:cx + half + MARKER_PAD] = 255
        scene[cy - half:cy + half, cx - half:cx + half] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    card = color_card()
    card_top, card_left = CARD_TOP, (BOARD_WIDTH - card.shape[1]) // 2
    scene[card_top:card_top + card.shape[0], card_left:card_left + card.shape[1]] = card
    return scene


def lens_distort(ideal: np.ndarray, background) -> np.ndarray:
    """La imagen que registra el lente LENS de la escena ideal (sin distorsión)."""
    rows, columns = np.mgrid[0:PHOTO_HEIGHT, 0:PHOTO_WIDTH].astype(np.float32)
    pixels = np.stack([columns.ravel(), rows.ravel()], 1).reshape(-1, 1, 2)
    source = cv2.undistortPoints(pixels, CAMERA, LENS, P=CAMERA).reshape(PHOTO_HEIGHT, PHOTO_WIDTH, 2)
    return cv2.remap(ideal, source[..., 0], source[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=background)


def view_of(flat: np.ndarray, rotation, shift=(0.0, 0.0), distance=CAMERA_DISTANCE, background=(90, 90, 90)):
    """Foto de un plano (su centro frente a la cámara) con la cámara girada `rotation`."""
    height, width = flat.shape[:2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    points = np.hstack([corners - [width / 2, height / 2], np.zeros((4, 1), np.float32)]).astype(np.float64)
    position = np.array([shift[0], shift[1], distance], np.float64)
    projected, _ = cv2.projectPoints(points, np.asarray(rotation, np.float64), position, CAMERA, None)
    homography = cv2.getPerspectiveTransform(corners, projected.reshape(4, 2).astype(np.float32))
    ideal = cv2.warpPerspective(flat, homography, (PHOTO_WIDTH, PHOTO_HEIGHT), flags=cv2.INTER_AREA,
                                borderValue=background)
    return lens_distort(ideal, background)


def light(image: np.ndarray, gains, seed: int) -> np.ndarray:
    """Otra luz y otra cámara: ganancia y mezcla de canales en RGB lineal, otra curva de tono y ruido."""
    random = np.random.default_rng(seed)
    linear = (image[..., ::-1].astype(np.float32) / 255) ** 2.2
    linear = linear @ LIGHT_MIX.T * np.float32(gains)
    toned = np.clip(linear, 0, 1) ** (1 / 2.3) * 255 + random.normal(0, 1.5, linear.shape)
    return np.clip(toned, 0, 255).astype(np.uint8)[..., ::-1]


def calibrate_lens():
    """Calibración del lente con vistas simuladas de un tablero de ajedrez (9 x 6 esquinas interiores)."""
    square, columns, rows = 170, 10, 7
    chess = np.full(((rows + 2) * square, (columns + 2) * square), 255, np.uint8)
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2 == 0:
                chess[(row + 1) * square:(row + 2) * square, (column + 1) * square:(column + 2) * square] = 0
    chess = cv2.cvtColor(chess, cv2.COLOR_GRAY2BGR)
    random = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as folder:
        files = []
        for index in range(14):
            rotation = random.normal(0, 0.22, 3)
            shift = (random.uniform(-500, 500), random.uniform(-650, 650))
            view = view_of(chess, rotation, shift, random.uniform(2900, 3900), background=(200, 200, 200))
            files.append(str(Path(folder) / f"board_{index:02d}.jpg"))
            cv2.imwrite(files[-1], view)
        result = calibrate(files, inner_cols=columns - 1, inner_rows=rows - 1)
    if not result.success:
        raise SystemExit(f"calibración del lente: {result.message}")
    return result


def setup_project(calibration) -> Project:
    """Proyecto con todo lo que lleva el preset: correcciones, área, segmentación y mediciones."""
    project = Project()
    distortion = project.corrections.distortion
    distortion.enabled = True
    distortion.mtx, distortion.dist = calibration.mtx.tolist(), calibration.dist.tolist()
    distortion.rms = round(float(calibration.rms_error), 4)
    distortion.size = [int(calibration.size[0]), int(calibration.size[1])]
    distortion.source = "simulated chessboard, 9 x 6 inner corners"
    perspective = project.corrections.perspective
    perspective.enabled = True
    perspective.width_mm = (MARKER_CENTERS[1][0] - MARKER_CENTERS[0][0]) * MM_PER_BOARD_PX
    perspective.height_mm = (MARKER_CENTERS[3][1] - MARKER_CENTERS[0][1]) * MM_PER_BOARD_PX
    project.corrections.color.enabled = True
    project.max_mpx = None
    project.roi = [{"kind": "include", "type": "rect", "points": ANALYSIS_AREA}]
    for key, value in SEGMENTATION.items():
        setattr(project.segmentation, key, value)
    project.analysis = "objects"
    defaults = {item["key"]: item["default"] for item in ANALYSES["objects"].params_schema if "default" in item}
    project.params["objects"] = {**defaults, **MEASURE_ALL}
    return project


def analysis_params(project: Project, mm_per_pixel: float) -> dict:
    """Parámetros del análisis de objetos como los arma la aplicación a partir del proyecto."""
    from dataclasses import asdict
    seg = project.segmentation
    return {**project.params["objects"], "segmentation": asdict(seg), "color_space": seg.color_space,
            "channel_idx": seg.channel, "min_val": seg.min_val, "max_val": seg.max_val, "auto_threshold": seg.auto,
            "mm_per_pixel": mm_per_pixel, "length_unit": project.unit, "roi_shapes": project.roi, "area_unit": "px"}


def truth_rows(project: Project) -> list[dict]:
    """Lo que se mide en cada tablero plano, recortado al rectángulo de los marcadores."""
    rows = []
    (left, top), (right, bottom) = MARKER_CENTERS[0], MARKER_CENTERS[2]
    for index, (tray, _rotation, _gains) in enumerate(SHOTS, 1):
        flat = board(ROOT / "tests" / "data" / "images" / tray)[top:bottom, left:right]
        result = ANALYSES["objects"].func(flat, analysis_params(project, MM_PER_BOARD_PX))
        measured = [row for row in result.measurements if row.get("length_mm")]
        rows.append({"photo": f"photo_{index}.jpg", "tray": tray, "n_objects": result.stats["n_objects"],
                     "length_mm_mean": round(float(np.mean([row["length_mm"] for row in measured])), 3),
                     "width_mm_mean": round(float(np.mean([row["width_mm"] for row in measured])), 3),
                     "area_mm2_mean": round(float(np.mean([row["area_mm2"] for row in measured])), 3)})
    return rows


def main():
    for index, (tray, rotation, gains) in enumerate(SHOTS, 1):
        photo = light(view_of(board(ROOT / "tests" / "data" / "images" / tray), rotation), gains, seed=index)
        cv2.imwrite(str(HERE / f"photo_{index}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
    calibration = calibrate_lens()
    print(f"lente: error {calibration.rms_error:.3f} px con {calibration.n_images_used} vistas")
    project = setup_project(calibration)
    data = preset.from_project(project, "Capture setup example", preset.PARTS,
                               "Lens, ArUco perspective and scale, color card, tray area, Otsu on b*, all measurements")
    data.pop("scale"), data.pop("unit")           # la escala sale de los marcadores de cada foto
    preset.save(data, HERE / "capture_setup")
    rows = truth_rows(project)
    with open(HERE / "truth.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
