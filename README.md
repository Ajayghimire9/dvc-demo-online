# CustomerFlow

CustomerFlow is a reproducible customer value modelling project. It takes an Ecommerce Customers style CSV, removes identifying fields, validates the analytical schema, compares several regression models, packages the selected model with a checksum, and exposes the result through a small FastAPI service.

The repository is local-first: the checked-in fixture is synthetic, so every reviewer can run the complete workflow without credentials or a data download. DVC and MLflow are optional integrations for teams that need remote data or experiment tracking.

## What the pipeline does

1. `customerflow.ingest` normalises source headers, validates finite positive measures, removes `Email`, `Address`, and `Avatar`, and writes a canonical CSV plus a source hash.
2. `customerflow.train` creates deterministic train/validation/test splits, evaluates a mean baseline, Ridge, Random Forest, and histogram gradient boosting, then selects the lowest validation RMSE.
3. The selected model is retrained on train plus validation data. `model.joblib`, `metrics.json`, and `model.manifest.json` are written together; the service refuses to load a bundle whose checksum has changed.
4. `customerflow.api` provides health, readiness, prediction, and Prometheus metrics endpoints.

## Run the complete workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,ops]'
python -m customerflow.ingest --input examples/customers.csv --output data/curated/customers.csv
python -m customerflow.train --data data/curated/customers.csv --output artifacts
python -m pytest tests -q
```

Start the API after training:

```bash
uvicorn customerflow.api:app --reload --port 8000
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{"avg_session_length": 34.5, "time_on_app": 13.2, "time_on_website": 37.1, "length_of_membership": 4.0}'
```

The same commands are available as `make curate`, `make train`, `make test`, and `make serve`. Build the non-root container with `docker build -t customerflow .`; mount a trained `artifacts/` directory at `/app/artifacts` when running it.

## DVC and MLflow

The two stages in `dvc.yaml` make the data boundary and model boundary explicit:

```bash
pip install 'dvc>=3.50,<4' 'mlflow>=2.15,<4'
dvc repro
```

Set `CUSTOMERFLOW_MLFLOW_TRACKING_URI` before training to log the selected model and test metrics to an existing MLflow server or local tracking directory. No tracking service is contacted by default.

## Technology and engineering choices

- Python, pandas, NumPy, scikit-learn, joblib
- DVC stages and metrics for reproducible data and model runs
- FastAPI and Pydantic for a typed inference contract
- SHA-256 model integrity manifest and fail-closed readiness checks
- Prometheus-compatible request metrics
- pytest, Ruff, GitHub Actions, and a non-root Docker image

The public API accepts only four validated numerical features. Identifiers and the target are never accepted as prediction inputs, which keeps the training and serving schemas separate and makes target leakage visible in code review.

## Scope and limitations

The example data is synthetic and deliberately small; its metrics demonstrate the workflow, not production performance. The checksum establishes integrity relative to the manifest, not authorship. A production deployment should add authenticated artifact storage, access control, drift monitoring, and a review policy for model promotion.
