"""Backward-compatible wrapper for the CustomerFlow ingestion command."""

import pandas as pd

from customerflow.data import (
    FEATURES,
    TARGET,
    load_source,
    prepare_frame,
    write_ingestion_manifest,
)
from customerflow.ingest import main, run

MODEL_COLUMNS = FEATURES + [TARGET]


def transform(frame: pd.DataFrame, minimum_membership: float = 0.0) -> pd.DataFrame:
    """Keep the historical import path while using the canonical validator."""

    return prepare_frame(frame, require_target=True, minimum_membership=minimum_membership)


__all__ = [
    "FEATURES",
    "MODEL_COLUMNS",
    "TARGET",
    "load_source",
    "main",
    "run",
    "transform",
    "write_ingestion_manifest",
]


if __name__ == "__main__":
    main()
