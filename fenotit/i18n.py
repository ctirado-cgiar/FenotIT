"""Textos de la interfaz: t("clave") busca en lang/<idioma>.json con respaldo en inglés."""
import json
import locale
from pathlib import Path

from fenotit import APP_NAME, settings

LANG_DIR = Path(__file__).parent / "lang"
DEFAULT = "en"

_strings: dict[str, str] = {}
_current = DEFAULT


def available() -> dict[str, str]:
    """{código: nombre nativo} de los idiomas con archivo en lang/."""
    out = {}
    for f in sorted(LANG_DIR.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8")).get("_language", f.stem)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def system_language() -> str | None:
    try:
        import ctypes
        lcid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
        name = locale.windows_locale.get(lcid)
    except Exception:
        name = locale.getlocale()[0]
    return name.split("_")[0].lower() if name else None


def _read(code: str) -> dict:
    try:
        return json.loads((LANG_DIR / f"{code}.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load(code: str | None = None) -> str:
    """Carga el idioma pedido, el guardado o el del sistema (en ese orden)."""
    global _strings, _current
    langs = available()
    if code is None:
        code = settings.get("language")
        if code not in langs:
            sys_lang = system_language()
            code = sys_lang if sys_lang in langs else DEFAULT
            settings.set("language", code)
    if code not in langs:
        code = DEFAULT
    _strings = _read(DEFAULT)
    if code != DEFAULT:
        _strings.update(_read(code))
    _current = code
    return code


def current() -> str:
    return _current


def set_language(code: str) -> None:
    settings.set("language", code)


def t(key: str, default: str | None = None, **kw) -> str:
    text = _strings.get(key, default if default is not None else key)
    kw.setdefault("app", APP_NAME)
    try:
        return text.format(**kw)
    except (KeyError, IndexError, ValueError):
        return text
