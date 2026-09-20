"""Production data-quality and drift monitoring for CustomerFlow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import FEATURES, load_source, prepare_frame


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute population stability index using reference quantile buckets."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, bins=edges)[0] / len(reference)
    cur = np.histogram(current, bins=edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_report(reference_path: str | Path, current_path: str | Path) -> dict[str, Any]:
    reference = prepare_frame(load_source(reference_path), require_target=False)
    current = prepare_frame(load_source(current_path), require_target=False)
    features: dict[str, Any] = {}
    for feature in FEATURES:
        score = _psi(reference[feature].to_numpy(), current[feature].to_numpy())
        features[feature] = {
            "psi": round(score, 6),
            "status": "critical" if score >= 0.25 else "warning" if score >= 0.10 else "stable",
            "reference_mean": round(float(reference[feature].mean()), 6),
            "current_mean": round(float(current[feature].mean()), 6),
        }
    worst = max((item["psi"] for item in features.values()), default=0.0)
    return {
        "schema_version": 1,
        "rows": {"reference": len(reference), "current": len(current)},
        "overall_status": "critical" if worst >= 0.25 else "warning" if worst >= 0.10 else "stable",
        "features": features,
    }


def write_report(reference_path: str | Path, current_path: str | Path, output: str | Path) -> dict[str, Any]:
    report = drift_report(reference_path, current_path)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
