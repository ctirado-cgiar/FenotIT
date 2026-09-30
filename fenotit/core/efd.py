"""Descriptores elípticos de Fourier (Kuhl & Giardina, 1982)."""
import numpy as np


def efd(contour: np.ndarray, order: int) -> np.ndarray:
    """Coeficientes (order × 4: a, b, c, d) de un contorno cerrado Nx2."""
    d = np.diff(np.vstack([contour, contour[:1]]), axis=0).astype(float)
    dt = np.hypot(d[:, 0], d[:, 1])
    keep = dt > 0
    d, dt = d[keep], dt[keep]
    t = np.concatenate([[0.0], np.cumsum(dt)])
    T = t[-1]
    n = np.arange(1, order + 1)[:, None]
    phi = 2 * np.pi * t / T
    c = T / (2 * n ** 2 * np.pi ** 2)
    dcos = np.cos(n * phi[1:]) - np.cos(n * phi[:-1])
    dsin = np.sin(n * phi[1:]) - np.sin(n * phi[:-1])
    ux, uy = d[:, 0] / dt, d[:, 1] / dt
    return np.column_stack([c[:, 0] * (dcos @ ux), c[:, 0] * (dsin @ ux),
                            c[:, 0] * (dcos @ uy), c[:, 0] * (dsin @ uy)])


def normalize(coeffs: np.ndarray) -> np.ndarray:
    """Invariante a tamaño, rotación y punto de inicio (primer armónico = elipse unitaria)."""
    a1, b1, c1, d1 = coeffs[0]
    theta = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1 ** 2 - b1 ** 2 + c1 ** 2 - d1 ** 2)
    out = coeffs.copy()
    for i in range(len(out)):
        k = i + 1
        rot = np.array([[np.cos(k * theta), -np.sin(k * theta)], [np.sin(k * theta), np.cos(k * theta)]])
        out[i] = (np.array([[out[i, 0], out[i, 1]], [out[i, 2], out[i, 3]]]) @ rot).ravel()
    psi = np.arctan2(out[0, 2], out[0, 0])
    rot = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])
    for i in range(len(out)):
        out[i] = (rot @ np.array([[out[i, 0], out[i, 1]], [out[i, 2], out[i, 3]]])).ravel()
    return out / abs(out[0, 0])


def contour_points(coeffs: np.ndarray, n: int = 200) -> np.ndarray:
    """Reconstruye el contorno (n puntos) desde los coeficientes."""
    t = np.linspace(0, 1, n, endpoint=False)
    k = np.arange(1, len(coeffs) + 1)[:, None]
    cos, sin = np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)
    x = (coeffs[:, 0:1] * cos + coeffs[:, 1:2] * sin).sum(0)
    y = (coeffs[:, 2:3] * cos + coeffs[:, 3:4] * sin).sum(0)
    return np.column_stack([x, y])


def columns(coeffs):
    return {f"efd_{l}{i}": round(float(v), 6)
            for i, row in enumerate(coeffs, 1) for l, v in zip("abcd", row)}


def from_row(row: dict) -> np.ndarray:
    n = sum(1 for k in row if k.startswith("efd_a"))
    return np.array([[row[f"efd_{l}{i}"] for l in "abcd"] for i in range(1, n + 1)], float)
