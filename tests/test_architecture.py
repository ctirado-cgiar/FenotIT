"""Reglas de la estructura del código. Cada prueba dice qué regla se rompió y dónde.

- El núcleo (fenotit/core) no importa la interfaz.
- Cada variable de la ventana principal se crea y se cambia en un solo archivo.
- Dos partes de la ventana no definen un método con el mismo nombre.
- Colores, letras y tamaños de la interfaz salen del tema, y el tema no tiene valores sin uso.
- Cada archivo empieza con una línea que dice qué hace, y no hay imports sin uso.
- El mapa de conexiones (docs/map/index.html) está al día.
"""
import ast
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "fenotit"
GUI = PKG / "gui"
WINDOW_FILES = sorted((GUI / "window").glob("*.py")) + [GUI / "layout.py", GUI / "main_window.py"]


def _py(folder: Path):
    return [p for p in sorted(folder.rglob("*.py")) if "__pycache__" not in p.parts]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def test_core_does_not_import_gui():
    bad = []
    for p in _py(PKG / "core"):
        for node in ast.walk(_tree(p)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            bad += [f"{_rel(p)}:{node.lineno} importa {n}" for n in names
                    if n.startswith(("fenotit.gui", "tkinter"))]
    assert not bad, "El núcleo no debe importar la interfaz:\n" + "\n".join(bad)


def _window_methods() -> dict[str, list[str]]:
    found = defaultdict(list)
    for p in WINDOW_FILES:
        for cls in (n for n in _tree(p).body if isinstance(n, ast.ClassDef)):
            for fn in cls.body:
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[fn.name].append(_rel(p))
    return found


def test_window_state_has_one_owner():
    methods = _window_methods()
    writers = defaultdict(set)
    for p in WINDOW_FILES:
        for node in ast.walk(_tree(p)):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" \
                    and isinstance(node.ctx, (ast.Store, ast.Del)):
                writers[node.attr].add(_rel(p))
            elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setattr" and len(node.args) >= 2 \
                    and getattr(node.args[0], "id", "") == "self" and isinstance(node.args[1], ast.Constant):
                writers[node.args[1].value].add(_rel(p))
    shared = {a: sorted(w) for a, w in writers.items() if len(w) > 1 and a not in methods}
    assert not shared, ("Cada variable de la ventana se crea y se cambia en un solo archivo; las demás partes "
                        "llaman a un método de ese archivo:\n"
                        + "\n".join(f"self.{a}: {', '.join(w)}" for a, w in sorted(shared.items())))


def test_window_methods_are_unique():
    dup = {m: sorted(set(f)) for m, f in _window_methods().items() if len(set(f)) > 1}
    assert not dup, ("Dos partes de la ventana definen el mismo método (solo se usaría uno):\n"
                     + "\n".join(f"{m}: {', '.join(f)}" for m, f in sorted(dup.items())))


def _theme() -> dict:
    with open(PKG / "assets" / "themes" / "classic.toml", "rb") as f:
        return tomllib.load(f)


def test_gui_uses_only_theme_values():
    theme = _theme()
    tables = {"COLORS": theme["colors"], "FONTS": theme["fonts"], "SIZES": theme["sizes"]}
    literal = re.compile(r"""["']#[0-9A-Fa-f]{6}["']""")
    bad = []
    for p in _py(GUI):
        if p.parent.name == "style":
            continue
        src = p.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            if literal.search(line) and "noqa" not in line:
                bad.append(f"{_rel(p)}:{i} color escrito a mano")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                    and node.value.id in tables and isinstance(node.slice, ast.Constant):
                if node.slice.value not in tables[node.value.id]:
                    bad.append(f"{_rel(p)}:{node.lineno} {node.value.id}[{node.slice.value!r}] no está en el tema")
    assert not bad, "Colores, letras y tamaños solo desde el tema (fenotit/assets/themes/classic.toml):\n" + "\n".join(bad)


def test_theme_has_no_unused_values():
    theme = _theme()
    code = "\n".join(p.read_text(encoding="utf-8") for p in _py(GUI))
    unused = [f"[{sec}] {k}" for sec in ("colors", "fonts", "sizes") for k in theme[sec]
              if f'"{k}"' not in code and f"'{k}'" not in code]
    assert not unused, "Valores del tema que nadie usa (quitarlos de classic.toml):\n" + "\n".join(unused)


def test_every_file_says_what_it_does():
    missing = [_rel(p) for p in _py(PKG) if not ast.get_docstring(_tree(p))]
    assert not missing, "Archivos sin una primera línea que diga qué hacen:\n" + "\n".join(missing)


def test_no_unused_imports():
    bad = []
    for p in _py(PKG):
        if p.name == "__init__.py":
            continue
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = {}
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and "noqa" not in src.splitlines()[node.lineno - 1]:
                for a in node.names:
                    if a.name != "annotations":
                        imported[(a.asname or a.name).split(".")[0]] = node.lineno
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {m for m in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(
            n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)))}
        bad += [f"{_rel(p)}:{line} {name}" for name, line in imported.items() if name not in used]
    assert not bad, "Imports sin uso:\n" + "\n".join(bad)


def test_map_is_up_to_date():
    sys.path.insert(0, str(ROOT / "docs" / "map"))
    import json
    import make_map
    page = (ROOT / "docs" / "map" / "index.html").read_text(encoding="utf-8")
    expected = json.dumps(make_map.build(), ensure_ascii=False)
    assert expected in page, "El mapa de conexiones cambió: correr  python docs/map/make_map.py"


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok ", name)
            except AssertionError as e:
                failed += 1
                print("FALLA", name, "\n" + str(e) + "\n")
    sys.exit(1 if failed else 0)
