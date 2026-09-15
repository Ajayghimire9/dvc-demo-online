"""Integrity-checked loading for the trained CustomerFlow bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib


def sha256_file(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_bundle(model_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    model_file, manifest_file = Path(model_path), Path(manifest_path)
    if not model_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError("Model bundle or integrity manifest is missing")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected = manifest.get("artifact", {}).get("sha256")
    actual = sha256_file(model_file)
    if not expected or expected != actual:
        raise ValueError("Model checksum does not match its manifest")
    bundle = joblib.load(model_file)
    if not isinstance(bundle, dict) or not bundle.get("features") or "model" not in bundle:
        raise ValueError("Model bundle has an unsupported schema")
    return bundle
