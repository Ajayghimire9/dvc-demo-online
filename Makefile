.PHONY: install lint test curate train serve

install:
	python -m pip install -e '.[dev,ops]'

lint:
	ruff check src tests

test:
	python -m pytest tests -q

curate:
	python -m customerflow.ingest --input examples/customers.csv --output data/curated/customers.csv

train: curate
	python -m customerflow.train --data data/curated/customers.csv --output artifacts

serve:
	uvicorn customerflow.api:app --reload --port 8000
