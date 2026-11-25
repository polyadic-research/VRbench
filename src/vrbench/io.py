"""Portable JSON, content hashes and strict relative paths."""
import hashlib
import json
from pathlib import Path


def read(path):
    def reject(value):raise ValueError('Non-finite JSON value: '+value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),parse_constant=reject)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Expected a portable relative path")
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Path escapes dataset")
    root = Path(root).resolve()
    path = (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Resolved path escapes dataset")
    return path


def unchanged(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError("Input content changed: " + str(path))
