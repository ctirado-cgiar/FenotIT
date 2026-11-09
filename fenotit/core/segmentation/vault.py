"""Modelos que solo abre FenotIT: el .onnx se guarda cifrado (`model.fnt`) y se descifra en
memoria al cargarlo en ONNX Runtime, sin escribir el archivo descifrado en el disco.

Formato: b"FNTM1" + 16 bytes al azar (nonce) + datos XOR con un flujo SHAKE-256 de
(clave + nonce). La clave de la versión distribuida está en `fenotit/_model_key.py`, que no
está en el repositorio (se crea al armar el instalador); sin ese archivo se usa una clave de
desarrollo. Un modelo cifrado con una clave no se abre con la otra.

Alcance: impide abrir el modelo con otros programas de forma directa; quien analice el
ejecutable de FenotIT puede encontrar la clave."""
from __future__ import annotations

import hashlib
import os

import numpy as np

MAGIC = b"FNTM1"
_DEV_KEY = b"fenotit-development-key-not-for-release"


def _key() -> bytes:
    try:
        from fenotit._model_key import KEY          # clave de la versión distribuida
        return KEY if isinstance(KEY, bytes) else str(KEY).encode()
    except ImportError:
        return _DEV_KEY


def _xor(data: bytes, nonce: bytes) -> bytes:
    stream = hashlib.shake_256(_key() + nonce).digest(len(data))
    return (np.frombuffer(data, np.uint8) ^ np.frombuffer(stream, np.uint8)).tobytes()


def lock(data: bytes) -> bytes:
    nonce = os.urandom(16)
    return MAGIC + nonce + _xor(data, nonce)


def is_locked(data: bytes) -> bool:
    return data[:len(MAGIC)] == MAGIC


def unlock(data: bytes) -> bytes:
    """Los bytes del .onnx. ValueError si no es un modelo cifrado por FenotIT o si se cifró
    con otra clave (el resultado no es un ONNX)."""
    if not is_locked(data):
        raise ValueError("not a FenotIT model")
    nonce = data[len(MAGIC):len(MAGIC) + 16]
    plain = _xor(data[len(MAGIC) + 16:], nonce)
    if plain[:1] != b"\x08":                         # un ONNX (protobuf) empieza con el campo ir_version
        raise ValueError("wrong key")
    return plain
