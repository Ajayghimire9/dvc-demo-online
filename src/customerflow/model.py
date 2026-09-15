"""Train, evaluate, and package a customer value model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, TARGET, load_source, prepare_frame
from .registry import sha256_file

RANDOM_STATE = 42


def _candidates() -> dict[str, Any]:
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "ridge": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.06,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 6),
        "r2": round(float(r2_score(actual, predicted)), 6),
    }


def _log_mlflow(metrics: dict[str, Any], selected_model: str) -> None:
    """Log an experiment only when a tracking URI is deliberately configured."""

    tracking_uri = os.getenv("CUSTOMERFLOW_MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - opt-in integration
        raise RuntimeError("MLflow tracking was requested but mlflow is not installed") from exc
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_name=f"customerflow-{selected_model}"):
        mlflow.log_param("selected_model", selected_model)
        mlflow.log_params({"features": ",".join(FEATURES), "random_state": RANDOM_STATE})
        mlflow.log_metrics(
            {f"test_{key}": value for key, value in metrics["test"].items()}
        )


def train_model(data_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Train candidates, select on validation RMSE, and write verifiable artifacts."""

    frame = prepare_frame(load_source(data_path), require_target=True)
    if len(frame) < 15:
        raise ValueError("At least 15 valid customers are required for a train/validation/test split")
    features = frame[FEATURES]
    target = frame[TARGET]
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features, target, test_size=0.4, random_state=RANDOM_STATE
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.5, random_state=RANDOM_STATE
    )

    validation: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    for name, candidate in _candidates().items():
        candidate.fit(x_train, y_train)
        fitted[name] = candidate
        validation[name] = _metrics(y_val, candidate.predict(x_val))
    selected_name = min(validation, key=lambda name: validation[name]["rmse"])

    final_model = fitted[selected_name]
    final_model.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
    test_metrics = _metrics(y_test, final_model.predict(x_test))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.joblib"
    bundle = {
        "schema_version": 1,
        "model_version": "customerflow-v1",
        "model_name": selected_name,
        "features": FEATURES,
        "target": TARGET,
        "model": final_model,
    }
    joblib.dump(bundle, model_path)
    metrics = {
        "schema_version": 1,
        "model_version": bundle["model_version"],
        "selected_model": selected_name,
        "features": FEATURES,
        "rows": {"total": len(frame), "train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "validation": validation,
        "test": test_metrics,
    }
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact": {"path": model_path.name, "bytes": model_path.stat().st_size, "sha256": sha256_file(model_path)},
        "model_version": bundle["model_version"],
        "features": FEATURES,
        "target": TARGET,
        "selected_model": selected_name,
    }
    (destination / "model.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log_mlflow(metrics, selected_name)
    return metrics


def predict_row(bundle: dict[str, Any], values: dict[str, float]) -> float:
    missing = sorted(set(bundle["features"]) - set(values))
    if missing:
        raise ValueError(f"Missing prediction features: {', '.join(missing)}")
    row = pd.DataFrame([{feature: values[feature] for feature in bundle["features"]}])
    if not np.isfinite(row.to_numpy(dtype=float)).all() or (row <= 0).any().any():
        raise ValueError("Prediction features must be finite and positive")
    return float(bundle["model"].predict(row)[0])
