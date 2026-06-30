"""
utils/image_io.py
Carga de imágenes Unicode-safe (OneDrive/CGIAR paths en Windows) + utilidades
de escalado responsivo para adaptar cualquier imagen al canvas disponible.
"""

import cv2
import numpy as np
from pathlib import Path


# ── Carga Unicode-safe ────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray | None:
    """
    Carga una imagen manejando rutas con espacios, tildes y caracteres Unicode
    (problema conocido con cv2.imread en OneDrive/Windows + OpenCV 4.10.0).
    Retorna array BGR o None si falla.
    """
    try:
        raw = Path(path).read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[image_io] Error cargando {path}: {e}")
        return None


def save_image(path: str, image: np.ndarray) -> bool:
    """
    Guarda imagen manejando rutas Unicode.
    Retorna True si fue exitoso.
    """
    try:
        ext = Path(path).suffix.lower()
        ok, buf = cv2.imencode(ext, image)
        if ok:
            Path(path).write_bytes(buf.tobytes())
            return True
        return False
    except Exception as e:
        print(f"[image_io] Error guardando {path}: {e}")
        return False


# ── Escalado responsivo ───────────────────────────────────────────────────────

class ImageScaler:
    """
    Maneja la relación entre la imagen original (full-res) y la imagen
    de display (escalada al canvas disponible).

    Todos los cálculos de medición/análisis deben usar coordenadas ORIGINALES.
    Esta clase provee los métodos de conversión necesarios.
    """

    def __init__(self):
        self.original: np.ndarray | None = None
        self.display: np.ndarray | None = None
        self.scale: float = 1.0          # factor display/original
        self.display_w: int = 0
        self.display_h: int = 0
        self.canvas_w: int = 400         # tamaño de canvas disponible
        self.canvas_h: int = 300

    def set_canvas_size(self, w: int, h: int):
        """Llamar cuando el canvas cambie de tamaño (resize de ventana)."""
        self.canvas_w = max(w, 100)
        self.canvas_h = max(h, 100)
        if self.original is not None:
            self._recompute_display()

    def load(self, path: str) -> bool:
        """Carga imagen y calcula display inicial."""
        img = load_image(path)
        if img is None:
            return False
        self.original = img
        self._recompute_display()
        return True

    def set_image(self, image: np.ndarray):
        """Asigna imagen directamente (p.ej. desde cámara)."""
        self.original = image.copy()
        self._recompute_display()

    def _recompute_display(self):
        """Recalcula la imagen de display para el canvas actual."""
        if self.original is None:
            return
        h, w = self.original.shape[:2]
        scale = min(self.canvas_w / w, self.canvas_h / h, 1.0)
        self.scale = scale
        self.display_w = int(w * scale)
        self.display_h = int(h * scale)
        if scale < 1.0:
            self.display = cv2.resize(
                self.original,
                (self.display_w, self.display_h),
                interpolation=cv2.INTER_AREA
            )
        else:
            self.display = self.original.copy()

    # ── Conversiones de coordenadas ──────────────────────────────────────────

    def display_to_original(self, x: int, y: int) -> tuple[int, int]:
        """Convierte punto de display → coordenadas originales."""
        if self.scale == 0:
            return x, y
        return int(x / self.scale), int(y / self.scale)

    def original_to_display(self, x: int, y: int) -> tuple[int, int]:
        """Convierte punto original → coordenadas de display."""
        return int(x * self.scale), int(y * self.scale)

    def scale_contours_to_display(self, contours: list) -> list:
        """Escala lista de contornos (coords originales) → display."""
        if self.scale == 1.0:
            return contours
        return [
            (cnt * self.scale).astype(np.int32)
            for cnt in contours
        ]

    def scale_contours_to_original(self, contours: list) -> list:
        """Escala lista de contornos (coords display) → originales."""
        if self.scale == 1.0:
            return contours
        return [
            (cnt / self.scale).astype(np.int32)
            for cnt in contours
        ]

    def scale_mask_to_original(self, mask_display: np.ndarray) -> np.ndarray:
        """Escala una máscara del tamaño display → tamaño original."""
        if self.original is None:
            return mask_display
        oh, ow = self.original.shape[:2]
        return cv2.resize(mask_display, (ow, oh), interpolation=cv2.INTER_NEAREST)

    def scale_mask_to_display(self, mask_original: np.ndarray) -> np.ndarray:
        """Escala una máscara del tamaño original → tamaño display."""
        return cv2.resize(
            mask_original,
            (self.display_w, self.display_h),
            interpolation=cv2.INTER_NEAREST
        )

    @property
    def has_image(self) -> bool:
        return self.original is not None

    @property
    def original_shape(self) -> tuple[int, int]:
        """Retorna (h, w) de la imagen original."""
        if self.original is None:
            return (0, 0)
        return self.original.shape[:2]

    @property
    def display_shape(self) -> tuple[int, int]:
        """Retorna (h, w) de la imagen de display."""
        if self.display is None:
            return (0, 0)
        return self.display.shape[:2]
