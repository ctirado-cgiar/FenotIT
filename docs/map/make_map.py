"""Mapa de conexiones del código: docs/map/index.html.

Lee los archivos de fenotit/ (sin importarlos) y arma una página navegable con cada carpeta,
cada archivo, lo que define, lo que importa y quién lo usa. En las partes de la ventana
(fenotit/gui/window/) muestra además qué estado crea cada parte y qué métodos de otras
partes llama.

Uso:  python docs/map/make_map.py
"""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "fenotit"
OUT = Path(__file__).with_name("index.html")
TEMPLATE = Path(__file__).with_name("template.html")
REPO = "https://github.com/ctirado-cgiar/fenotit/blob/main/"
WINDOW = ("fenotit.gui.window.", "fenotit.gui.layout", "fenotit.gui.main_window")


def module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def first_line(text: str | None) -> str:
    return (text or "").strip().split("\n\n")[0].replace("\n", " ").strip()


def signature(fn: ast.FunctionDef) -> str:
    args = [a.arg for a in fn.args.posonlyargs + fn.args.args if a.arg not in ("self", "cls")]
    if fn.args.vararg:
        args.append("*" + fn.args.vararg.arg)
    args += [a.arg for a in fn.args.kwonlyargs]
    if fn.args.kwarg:
        args.append("**" + fn.args.kwarg.arg)
    return f"({', '.join(args)})"


def resolve(mod: str, node: ast.ImportFrom, is_pkg: bool) -> str:
    if not node.level:
        return node.module or ""
    base = mod.split(".") if is_pkg else mod.split(".")[:-1]
    base = base[:len(base) - node.level + 1]
    return ".".join(base + ([node.module] if node.module else []))


def step_key(dec: ast.expr) -> str | None:
    if isinstance(dec, ast.Call) and getattr(dec.func, "id", getattr(dec.func, "attr", "")) == "step":
        if dec.args and isinstance(dec.args[0], ast.Constant):
            return str(dec.args[0].value)
    return None


def scan(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    mod = module_name(path)
    info = {"id": mod, "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "doc": first_line(ast.get_docstring(tree)), "lines": src.count("\n") + 1,
            "defs": [], "imports": defaultdict(set), "libs": set(), "steps": [],
            "writes": set(), "calls": set(), "methods": {}}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info["defs"].append({"kind": "def", "name": node.name, "sig": signature(node),
                                 "doc": first_line(ast.get_docstring(node)), "line": node.lineno})
            info["steps"] += [k for k in map(step_key, node.decorator_list) if k]
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            info["defs"].append({"kind": "class", "name": node.name, "sig": "",
                                 "doc": first_line(ast.get_docstring(node)), "line": node.lineno,
                                 "methods": [{"name": m.name, "sig": signature(m), "line": m.lineno,
                                              "doc": first_line(ast.get_docstring(m))} for m in methods]})
            for m in methods:
                info["methods"][m.name] = m.lineno
            info["steps"] += [k for k in map(step_key, node.decorator_list) if k]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = resolve(mod, node, path.name == "__init__.py")
            if target.startswith("fenotit"):
                for a in node.names:
                    sub = f"{target}.{a.name}"
                    info["imports"][sub if (PKG.parent / (sub.replace(".", "/") + ".py")).exists()
                                    or (PKG.parent / sub.replace(".", "/") / "__init__.py").exists()
                                    else target].add(a.name)
            else:
                info["libs"].add(target.split(".")[0])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("fenotit"):
                    info["imports"][a.name].add("")
                else:
                    info["libs"].add(a.name.split(".")[0])
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                info["writes"].add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            info["calls"].add(node.func.attr)
    return info


def build() -> dict:
    mods = {m["id"]: m for m in (scan(p) for p in sorted(PKG.rglob("*.py")) if "__pycache__" not in p.parts)}
    used_by = defaultdict(dict)
    for m in mods.values():
        m["imports"] = {k: sorted(n for n in v if n) for k, v in m["imports"].items() if k in mods and k != m["id"]}
        for target, names in m["imports"].items():
            used_by[target][m["id"]] = names
    window = {k: m for k, m in mods.items() if k.startswith(WINDOW)}
    owner = {name: k for k, m in window.items() for name in m["methods"]}
    calls_in = defaultdict(lambda: defaultdict(list))
    for k, m in window.items():
        out = defaultdict(list)
        for c in sorted(m["calls"]):
            dst = owner.get(c)
            if dst and dst != k:
                out[dst].append(c)
                calls_in[dst][k].append(c)
        m["calls_out"] = dict(out)
        m["state"] = sorted(a for a in m["writes"] if a not in owner)
    for k, m in mods.items():
        m["used_by"] = used_by.get(k, {})
        m["libs"] = sorted(lib for lib in m["libs"] if lib and lib != "__future__")
        if k in window:
            m["calls_in"] = {s: v for s, v in calls_in.get(k, {}).items()}
        m["folder"] = k.rsplit(".", 1)[0] if not m["path"].endswith("__init__.py") else k
        for key in ("writes", "calls", "methods"):
            m.pop(key)
    folders = defaultdict(list)
    for k, m in mods.items():
        folders[m["folder"]].append(k)
    folder_docs = {f: mods[f]["doc"] if f in mods else "" for f in folders}
    links = defaultdict(int)
    for m in mods.values():
        for target in m["imports"]:
            a, b = m["folder"], mods[target]["folder"]
            if a != b:
                links[(a, b)] += 1
    return {"modules": mods, "folders": {f: {"doc": folder_docs[f], "files": sorted(v)} for f, v in folders.items()},
            "links": [{"from": a, "to": b, "n": n} for (a, b), n in sorted(links.items())], "repo": REPO}


def main(bare: str | None = None):
    """bare: escribe solo el contenido de la página (sin <html>/<head>) en esa ruta."""
    data = build()
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    if bare:
        Path(bare).write_text(page, encoding="utf-8")
    html = ('<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n'
            f"{page}\n</body>\n</html>\n")
    OUT.write_text(html, encoding="utf-8")
    print(OUT, f"({len(data['modules'])} archivos, {len(data['folders'])} carpetas)")


if __name__ == "__main__":
    import sys
    main(sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--bare" else None)
