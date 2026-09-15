"""Schema validation and privacy-aware preparation for customer data."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "avg_session_length",
    "time_on_app",
    "time_on_website",
    "length_of_membership",
]
TARGET = "yearly_amount_spent"
PII_COLUMNS = ["email", "address", "avatar"]


def _normalise_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def canonicalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common Ecommerce Customers headers to a stable internal schema."""

    renamed: dict[object, str] = {}
    seen: set[str] = set()
    for column in frame.columns:
        canonical = _normalise_header(column)
        if canonical in seen:
            raise ValueError(f"Duplicate column after normalisation: {canonical}")
        seen.add(canonical)
        renamed[column] = canonical
    return frame.rename(columns=renamed)


def load_source(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Customer source does not exist: {source}")
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("Customer source is empty")
    return canonicalise_columns(frame)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Customer source is missing columns: {', '.join(missing)}")


def prepare_frame(
    frame: pd.DataFrame,
    *,
    require_target: bool = True,
    minimum_membership: float = 0.0,
) -> pd.DataFrame:
    """Validate numeric fields, remove PII, and return model-ready columns."""

    canonical = canonicalise_columns(frame)
    required = FEATURES + ([TARGET] if require_target else [])
    _require_columns(canonical, required)
    selected = canonical[required].copy()
    for column in required:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Customer measures must be finite")
    if (selected[FEATURES] <= 0).any().any():
        raise ValueError("Customer features must be positive")
    if require_target and (selected[TARGET] < 0).any():
        raise ValueError("Yearly amount spent cannot be negative")
    selected = selected[selected["length_of_membership"] > minimum_membership]
    if selected.empty:
        raise ValueError("No customers remain after membership filter")
    return selected.reset_index(drop=True)


def source_sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_ingestion_manifest(
    source: str | Path, output: str | Path, frame: pd.DataFrame
) -> Path:
    """Write a small, reviewable record of the exact curation operation."""

    output_path = Path(output)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha256": source_sha256(source),
                "rows": len(frame),
                "columns": list(frame.columns),
                "pii_columns_removed": PII_COLUMNS,
                "membership_filter": "> 0 years",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path
