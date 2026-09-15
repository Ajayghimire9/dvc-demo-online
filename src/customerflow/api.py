"""FastAPI inference service for the trained CustomerFlow bundle."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, Field

from .model import predict_row
from .registry import load_bundle

MODEL_PATH = Path(os.getenv("CUSTOMERFLOW_MODEL", "artifacts/model.joblib"))
MANIFEST_PATH = Path(os.getenv("CUSTOMERFLOW_MANIFEST", "artifacts/model.manifest.json"))
REQUESTS = Counter("customerflow_prediction_requests_total", "Prediction requests", ["status"])


class CustomerFeatures(BaseModel):
    avg_session_length: float = Field(gt=0, le=120)
    time_on_app: float = Field(gt=0, le=120)
    time_on_website: float = Field(gt=0, le=120)
    length_of_membership: float = Field(gt=0, le=100)


class Prediction(BaseModel):
    yearly_amount_spent: float
    model_version: str


app = FastAPI(title="CustomerFlow API", version="1.0.0")


@lru_cache(maxsize=1)
def _bundle() -> dict:
    return load_bundle(MODEL_PATH, MANIFEST_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "customerflow"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        bundle = _bundle()
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "model_version": bundle["model_version"]}


@app.post("/v1/predict", response_model=Prediction)
def predict(features: CustomerFeatures) -> Prediction:
    try:
        bundle = _bundle()
        values = features.model_dump() if hasattr(features, "model_dump") else features.dict()
        prediction = predict_row(bundle, values)
        REQUESTS.labels(status="success").inc()
        return Prediction(yearly_amount_spent=prediction, model_version=bundle["model_version"])
    except (FileNotFoundError, ValueError, OSError) as exc:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
