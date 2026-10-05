import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANG = ROOT / "fenotit" / "lang"


def _keys_in_code():
    keys = set()
    for f in (ROOT / "fenotit").rglob("*.py"):
        keys |= set(re.findall(r'\bt\(\s*["\']([a-z_]+(?:\.[a-z_]+)+)["\']', f.read_text(encoding="utf-8")))
    return keys


def test_same_keys():
    en = json.loads((LANG / "en.json").read_text(encoding="utf-8"))
    for f in LANG.glob("*.json"):
        other = json.loads(f.read_text(encoding="utf-8"))
        missing = set(en) - set(other)
        extra = set(other) - set(en)
        assert not missing and not extra, f"{f.name}: faltan {sorted(missing)} sobran {sorted(extra)}"


def test_code_keys_exist():
    en = json.loads((LANG / "en.json").read_text(encoding="utf-8"))
    missing = sorted(k for k in _keys_in_code() if k not in en)
    assert not missing, f"Claves usadas sin traducción: {missing}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
