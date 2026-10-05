"""Calibración con tablero de ajedrez: fotos sintéticas grandes desde varios ángulos."""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fenotit.core.corrections import distortion as D

COLS, ROWS = 7, 6


def _board(sq=200):
    b = np.full(((ROWS + 3) * sq, (COLS + 3) * sq), 255, np.uint8)
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                y, x = (r + 1) * sq, (c + 1) * sq
                b[y:y + sq, x:x + sq] = 0
    return b


def _views(folder: Path, n=6, size=(4000, 3000)):
    """Vistas del tablero con una cámara real (K fija, poses distintas)."""
    board = _board()
    h, w = board.shape
    K = np.array([[3500, 0, size[0] / 2], [0, 3500, size[1] / 2], [0, 0, 1]], float)
    corners = np.float32([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]])
    rng = np.random.default_rng(1)
    paths = []
    for i in range(n):
        rvec = rng.uniform(-0.35, 0.35, 3)
        tvec = np.array([-w / 2 + rng.uniform(-150, 150), -h / 2 + rng.uniform(-150, 150), 4200 + rng.uniform(-400, 400)])
        R = cv2.Rodrigues(rvec)[0]
        center = np.array([w / 2, h / 2, 0])
        tvec = tvec + center - R @ center          # girar alrededor del centro del tablero
        dst, _ = cv2.projectPoints(corners, rvec, tvec, K, None)
        H = cv2.getPerspectiveTransform(np.ascontiguousarray(corners[:, :2]), dst.reshape(4, 2).astype(np.float32))
        img = cv2.warpPerspective(board, H, size, borderValue=200)
        p = folder / f"board_{i}.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        paths.append(str(p))
    blank = folder / "blank.jpg"
    cv2.imwrite(str(blank), np.full((3000, 4000, 3), 180, np.uint8))
    return paths + [str(blank)]


def test_calibrate_large_photos():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _views(Path(tmp))
        dets = [D.detect(p, COLS, ROWS) for p in paths]
        assert [d.found for d in dets] == [True] * 6 + [False]
        assert dets[0].preview is not None and max(dets[0].preview.shape[:2]) == 360
        res = D.solve(dets, COLS, ROWS)
        assert res.success and res.n_images_used == 6 and res.rms_error < 1.0, res.rms_error
        assert all(d.error_px is not None for d in dets[:6])


def test_spread():
    paths = [f"{i}.jpg" for i in range(266)]
    out = D.spread(paths, 40)
    assert len(out) == 40 and out[0] == "0.jpg" and out[-1] == "265.jpg"



def test_other_resolution():
    """La calibración se escala a otra resolución con la misma proporción; con otra
    proporción la foto no se corrige (y el lote la omite)."""
    from fenotit.core.corrections import pipeline as P
    K = [[3000, 0, 2000], [0, 3000, 1500], [0, 0, 1]]
    d = P.Distortion(True, K, [[-0.1, 0.01, 0, 0, 0]], size=[4000, 3000])
    assert np.allclose(P.camera_matrix(d, 8000, 6000), np.diag([2, 2, 1]) @ np.array(K))
    _, info = P.apply(np.zeros((300, 400, 3), np.uint8), P.Corrections(distortion=d))
    assert info.applied == ["distortion"]
    _, info = P.apply(np.zeros((400, 300, 3), np.uint8), P.Corrections(distortion=d))
    assert info.applied == [] and info.warnings

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
