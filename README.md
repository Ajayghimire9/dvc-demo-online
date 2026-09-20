# CustomerFlow ML Platform

A production-oriented machine-learning system for customer lifetime-value regression. This repository started as an experiment in dataset versioning and has been rebuilt as an end-to-end ML engineering project: validated ingestion, reproducible training, model comparison, artifact integrity, experiment tracking, API serving, observability, drift detection, automated tests and containerized delivery.

## Why this project exists

A notebook can prove that a model works once. It does not show how that model should be reproduced, reviewed, served or monitored. CustomerFlow focuses on those engineering boundaries. The included dataset is synthetic so the entire system can be executed locally without credentials or hidden dependencies.

## Architecture

```text
Raw customer data
      |
      v
Schema validation + PII removal
      |
      v
DVC-versioned curated dataset
      |
      v
Train / validation / test pipeline
      |
      +--> baseline / Ridge / Random Forest / HistGradientBoosting
      |
      v
Model selection + MLflow (optional)
      |
      v
Versioned artifact + SHA-256 manifest
      |
      v
FastAPI inference service
      |
      +--> Prometheus metrics
      +--> readiness / health checks
      +--> population drift monitoring (PSI)
```

## Engineering features

- deterministic training and explicit train/validation/test boundaries
- DVC pipeline for reproducible data and model stages
- optional MLflow experiment tracking without requiring a hosted service
- candidate-model comparison against a naive baseline
- versioned model bundle with a SHA-256 integrity manifest
- typed FastAPI inference contract with Pydantic validation
- health and readiness endpoints designed for orchestration probes
- Prometheus-compatible request metrics
- PSI-based feature drift monitoring with stable/warning/critical thresholds
- pytest regression coverage and Ruff static checks
- non-root Docker runtime and GitHub Actions CI
- installable Python package and CLI entry points

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,ops]'

customerflow-ingest \
  --input examples/customers.csv \
  --output data/curated/customers.csv

customerflow-train \
  --data data/curated/customers.csv \
  --output artifacts

pytest -q
```

Run the service:

```bash
uvicorn customerflow.api:app --host 0.0.0.0 --port 8000
```

Example prediction:

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{"avg_session_length":34.5,"time_on_app":13.2,"time_on_website":37.1,"length_of_membership":4.0}'
```

## Reproducible pipeline

```bash
dvc repro
```

The pipeline separates curation from training so a data change invalidates the correct downstream artifacts. To record experiments in MLflow, set `CUSTOMERFLOW_MLFLOW_TRACKING_URI`; training remains completely local when the variable is absent.

## Production drift check

The monitoring command compares a reference feature population with a new batch and writes a machine-readable report:

```bash
customerflow-drift \
  --reference data/curated/customers.csv \
  --current path/to/production_batch.csv \
  --output artifacts/drift-report.json
```

Population Stability Index (PSI) is calculated independently for each serving feature. Scores below `0.10` are treated as stable, `0.10-0.25` as warning and `>=0.25` as critical. These thresholds are operational defaults, not universal statistical guarantees.

## Model lifecycle

Training evaluates a mean baseline, Ridge regression, Random Forest and histogram gradient boosting. Selection uses validation RMSE only. The winning estimator is retrained on train + validation data and evaluated once on the held-out test set. The serialized model is packaged with its feature contract, model version and checksum.

The API verifies the manifest before becoming ready. A modified or missing artifact therefore fails closed instead of silently serving an unknown model.

## Technology

**ML:** Python, pandas, NumPy, scikit-learn, joblib  
**MLOps:** DVC, MLflow, reproducible artifacts, model manifests  
**Serving:** FastAPI, Pydantic, Uvicorn  
**Observability:** Prometheus metrics, health/readiness probes, PSI drift detection  
**Engineering:** pytest, Ruff, Docker, GitHub Actions, Make

## Repository philosophy

The project intentionally avoids adding infrastructure only for appearance. Kubernetes, a feature store or a streaming platform would be reasonable when traffic, deployment topology or latency requirements justify them; they are not required to demonstrate the lifecycle implemented here. The goal is a system whose components have clear responsibilities and can actually be run by another engineer.

## Limitations

The bundled data is synthetic and small, so reported metrics demonstrate pipeline behavior rather than real-world business performance. PSI identifies distribution change but does not explain its cause or guarantee model degradation. A real deployment should add authenticated artifact storage, access control, production telemetry, alert routing, champion/challenger promotion rules and business-level outcome monitoring.
