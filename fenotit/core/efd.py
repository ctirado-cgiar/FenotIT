"""Descriptores elípticos de Fourier (Kuhl & Giardina, 1982)."""
import numpy as np


def efd(contour: np.ndarray, order: int) -> np.ndarray:
    """Coeficientes (order × 4: a, b, c, d) de un contorno cerrado Nx2."""
    steps = np.diff(np.vstack([contour, contour[:1]]), axis=0).astype(float)
    step_length = np.hypot(steps[:, 0], steps[:, 1])
    keep = step_length > 0
    steps, step_length = steps[keep], step_length[keep]
    t = np.concatenate([[0.0], np.cumsum(step_length)])
    perimeter = t[-1]
    harmonic = np.arange(1, order + 1)[:, None]
    phi = 2 * np.pi * t / perimeter
    factor = perimeter / (2 * harmonic ** 2 * np.pi ** 2)
    dcos = np.cos(harmonic * phi[1:]) - np.cos(harmonic * phi[:-1])
    dsin = np.sin(harmonic * phi[1:]) - np.sin(harmonic * phi[:-1])
    unit_x, unit_y = steps[:, 0] / step_length, steps[:, 1] / step_length
    return np.column_stack([factor[:, 0] * (dcos @ unit_x), factor[:, 0] * (dsin @ unit_x),
                            factor[:, 0] * (dcos @ unit_y), factor[:, 0] * (dsin @ unit_y)])


def normalize(coeffs: np.ndarray) -> np.ndarray:
    """Invariante a tamaño, rotación y punto de inicio (primer armónico = elipse unitaria)."""
    a_first, b_first, c_first, d_first = coeffs[0]
    theta = 0.5 * np.arctan2(2 * (a_first * b_first + c_first * d_first), a_first ** 2 - b_first ** 2 + c_first ** 2 - d_first ** 2)
    out = coeffs.copy()
    for i in range(len(out)):
        harmonic = i + 1
        rot = np.array([[np.cos(harmonic * theta), -np.sin(harmonic * theta)], [np.sin(harmonic * theta), np.cos(harmonic * theta)]])
        out[i] = (np.array([[out[i, 0], out[i, 1]], [out[i, 2], out[i, 3]]]) @ rot).ravel()
    psi = np.arctan2(out[0, 2], out[0, 0])
    rot = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])
    for i in range(len(out)):
        out[i] = (rot @ np.array([[out[i, 0], out[i, 1]], [out[i, 2], out[i, 3]]])).ravel()
    return out / abs(out[0, 0])


def contour_points(coeffs: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Reconstruye el contorno (n puntos) desde los coeficientes."""
    t = np.linspace(0, 1, n_points, endpoint=False)
    harmonic = np.arange(1, len(coeffs) + 1)[:, None]
    cos, sin = np.cos(2 * np.pi * harmonic * t), np.sin(2 * np.pi * harmonic * t)
    x = (coeffs[:, 0:1] * cos + coeffs[:, 1:2] * sin).sum(0)
    y = (coeffs[:, 2:3] * cos + coeffs[:, 3:4] * sin).sum(0)
    return np.column_stack([x, y])


def columns(coeffs):
    return {f"efd_{letter}{i}": round(float(value), 6)
            for i, row in enumerate(coeffs, 1) for letter, value in zip("abcd", row)}


def from_row(row: dict) -> np.ndarray:
    n_harmonics = sum(1 for key in row if key.startswith("efd_a"))
    return np.array([[row[f"efd_{letter}{i}"] for letter in "abcd"] for i in range(1, n_harmonics + 1)], float)
