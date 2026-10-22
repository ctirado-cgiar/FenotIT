"""Autoguardado para recuperar el trabajo si la app se cierra sin guardar (se traba, se va
la luz). No toca el archivo del proyecto: escribe una copia aparte en la carpeta de datos de
la app, con rutas absolutas, que se borra al guardar o al cerrar normalmente."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import os
from pathlib import Path

import yaml

from fenotit import settings
from fenotit.core.project import PROJECT_EXT, Project


def folder() -> Path:
    return settings.data_dir() / "recovery"


def path_for(project: Project, session: str) -> Path:
    key = str(project.folder) if project.folder else f"untitled-{session}"
    return folder() / f"{hashlib.sha1(key.encode()).hexdigest()[:12]}{PROJECT_EXT}"


def write(project: Project, session: str) -> Path:
    tmp = copy.copy(project)
    tmp.folder = None                                   # rutas absolutas: la copia vive en otra carpeta
    d = tmp.to_dict()
    d["recovery"] = {"of": str(project.folder) if project.folder else "",
                     "name": project.name, "saved": dt.datetime.now().isoformat(timespec="seconds")}
    dest = path_for(project, session)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    with open(part, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)
    os.replace(part, dest)                               # nunca queda un archivo a medias
    return dest


def pending() -> list[dict]:
    out = []
    for p in sorted(folder().glob(f"*{PROJECT_EXT}")) if folder().exists() else []:
        try:
            with open(p, encoding="utf-8") as f:
                info = (yaml.safe_load(f) or {}).get("recovery") or {}
        except (OSError, yaml.YAMLError):
            continue
        out.append({"file": p, "name": info.get("name", p.stem), "of": info.get("of", ""),
                    "saved": info.get("saved", ""), "n_images": _count(p)})
    return out


def _count(p: Path) -> int:
    try:
        with open(p, encoding="utf-8") as f:
            return len((yaml.safe_load(f) or {}).get("images") or [])
    except (OSError, yaml.YAMLError):
        return 0


def load(item: dict) -> Project:
    project = Project.load(item["file"])
    project.folder = Path(item["of"]) if item.get("of") else None
    if project.folder is None:
        project.name = item.get("name") or project.name
    return project


def discard(path: Path | None) -> None:
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
