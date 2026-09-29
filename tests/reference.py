"""
Resultados de referencia para detectar cambios en los números.

    python tests/reference.py --save   guarda la referencia en tests/reference/
    python tests/reference.py          compara contra la referencia guardada
"""
import argparse
import contextlib
import csv
import io
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with contextlib.redirect_stdout(io.StringIO()):
    from analysis.registry import ANALYSES
from utils.image_io import load_image

IMAGES = sorted((ROOT / "tests" / "images").glob("*.jpg"))
REF_DIR = ROOT / "tests" / "reference"
TOL = 1e-6

COMMON = {
    "color_space_code": cv2.COLOR_BGR2YCrCb,
    "channel_idx": 1,
    "min_val": 122,
    "max_val": 255,
    "mm_per_pixel": None,
    "roi_mask": None,
}

FILES = {
    "Morfometría": "morphometry.csv",
    "Color KMeans": "color_kmeans.csv",
    "Contador de semillas": "seed_counter.csv",
}


def _defaults(name):
    return {p["key"]: p["default"] for p in ANALYSES[name].params_schema}


def run_all():
    out = {}
    for name, fname in FILES.items():
        params = {**COMMON, **_defaults(name)}
        rows = []
        for img_path in IMAGES:
            res = ANALYSES[name].func(load_image(str(img_path)), params)
            if res.status != "ok":
                rows.append({"image": img_path.stem, "row": 0, "status": res.status or res.error})
                continue
            items = res.measurements or [res.stats]
            for i, m in enumerate(items, 1):
                rows.append({"image": img_path.stem, "row": i,
                             **{k: v for k, v in m.items() if not k.startswith("_")}})
        out[fname] = rows
    return out


def _write(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _same(a, b):
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return math.isclose(fa, fb, rel_tol=TOL, abs_tol=TOL)


def compare(current):
    n_diff = 0
    for fname, rows in current.items():
        ref_path = REF_DIR / fname
        if not ref_path.exists():
            print(f"[{fname}] sin referencia")
            n_diff += 1
            continue
        tmp = io.StringIO()
        cols = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(tmp, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
        tmp.seek(0)
        new = list(csv.DictReader(tmp))
        ref = _read(ref_path)

        issues = []
        if len(new) != len(ref):
            issues.append(f"filas: {len(ref)} -> {len(new)}")
        ref_cols, new_cols = set(ref[0]) if ref else set(), set(new[0]) if new else set()
        if ref_cols != new_cols:
            if ref_cols - new_cols:
                issues.append(f"columnas quitadas: {sorted(ref_cols - new_cols)}")
            if new_cols - ref_cols:
                issues.append(f"columnas nuevas: {sorted(new_cols - ref_cols)}")
        for r_old, r_new in zip(ref, new):
            for c in ref_cols & new_cols:
                if not _same(r_old[c], r_new[c]):
                    issues.append(f"{r_old['image']} fila {r_old['row']} {c}: {r_old[c]} -> {r_new[c]}")

        if issues:
            n_diff += len(issues)
            print(f"[{fname}] {len(issues)} diferencias")
            for s in issues[:20]:
                print("  " + s)
            if len(issues) > 20:
                print(f"  ... y {len(issues) - 20} más")
        else:
            print(f"[{fname}] igual ({len(ref)} filas)")
    return n_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    if not IMAGES:
        sys.exit("No hay imágenes en tests/images/")
    current = run_all()

    if args.save:
        REF_DIR.mkdir(exist_ok=True)
        for fname, rows in current.items():
            _write(REF_DIR / fname, rows)
            print(f"Guardado {fname}: {len(rows)} filas")
        return
    sys.exit(1 if compare(current) else 0)


if __name__ == "__main__":
    main()
