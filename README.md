# Lightweight MLOps Prototype

A locally executable MLOps prototype for reproducible model training,
model versioning, API-based serving and batch drift monitoring.

## Motivation

Machine-learning models are often initially developed in notebooks.
Moving from an experiment to a reproducible and monitorable system
requires additional mechanisms for pipeline automation, traceability,
model management, deployment and monitoring.

This project demonstrates a lightweight implementation of this
lifecycle for a resource-constrained local development environment.

## Machine-Learning Use Case

The prototype uses the UCI Bank Marketing dataset.

The binary classification task predicts whether a bank customer will
subscribe to a term deposit.

## Architecture

````mermaid
flowchart LR
    A[Raw Data] --> B[Data Preparation]
    B --> C[Model Training]
    C --> D[Model Evaluation]
    D --> E[MLflow Tracking]
    E --> F[Model Registry]
    F --> G[FastAPI Prediction Service]
    B --> H[Simulated Current Data]
    C --> I[Prediction Probabilities]
    H --> J[Evidently Drift Monitoring]
    I --> J
````

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git
- Docker and Docker Compose (optional, for containerised serving)

## Setup

````bash
git clone https://github.com/andacguemues/lightweight-mlops-prototype.git
cd lightweight-mlops-prototype
uv sync --locked
````

`uv sync --locked` installs the exact package versions recorded in
`uv.lock`. This is a precondition for reproducible runs.

## Run the Pipeline

````bash
uv run dvc repro
````

This executes all seven stages: data download, preparation, training,
evaluation, model registration, monitoring data simulation and drift
detection.

After a successful run the following artefacts exist:

- `data/processed/train.csv`, `data/processed/test.csv`
- `models/model.joblib` — preprocessing and classifier in one artefact
- `reports/evaluation/metrics.json` — evaluation metrics
- `reports/evaluation/registry_info.json` — registered model version
- `reports/monitoring/*.html`, `reports/monitoring/monitoring_summary.json`
- an MLflow run containing parameters, metrics and provenance tags

## Start the Prediction Service

Local:

````bash
uv run uvicorn mlops_prototype.api.main:app --app-dir src \
  --host 127.0.0.1 --port 8000
````

Containerised:

````bash
docker compose up -d --build
````

The service exposes three endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service and model availability |
| `/model-info` | GET | Metadata of the served model version |
| `/predict` | POST | Single prediction |

## Example Request

````bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":32,"job":"management","marital":"divorced",
       "education":"university.degree","default":"no","housing":"no",
       "loan":"no","contact":"cellular","month":"jul","day_of_week":"tue",
       "duration":131,"campaign":5,"pdays":999,"previous":0,
       "poutcome":"nonexistent","emp_var_rate":1.4,
       "cons_price_idx":93.918,"cons_conf_idx":-42.7,
       "euribor3m":4.961,"nr_employed":5228.1}'
````

Expected response:

````json
{
  "prediction": "no",
  "probability_yes": 0.0509,
  "model_name": "bank-marketing-classifier",
  "model_alias": "candidate"
}
````

Note that the API schema uses underscores (`emp_var_rate`) where the
raw dataset uses dots (`emp.var.rate`). Unknown fields are rejected
with HTTP 422.

## Tests and Validation

````bash
uv run pytest -v
uv run python scripts/validate_artifact.py
make verify
````

## Model Source Configuration

The service loads the model either from the MLflow registry or
directly from a file:

````bash
MODEL_SOURCE=registry uv run uvicorn ...          # development
MODEL_SOURCE=file MODEL_PATH=models/model.joblib  # container default
````

## Project Structure

````text
src/mlops_prototype/   source code (data, training, registry, api, monitoring)
data/                  raw, processed and simulated data
models/                serialised model artefact
reports/               evaluation metrics and monitoring reports
docs/                  architecture, monitoring, data card, ADRs, research log
notebooks/             notebook baseline workflow
tests/                 automated tests
scripts/               artefact validation
````

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — components and flows
- [`docs/monitoring.md`](docs/monitoring.md) — monitoring design
- [`docs/data-card.md`](docs/data-card.md) — dataset origin and licence
- [`docs/evaluation-plan.md`](docs/evaluation-plan.md) — evaluation criteria
- [`docs/research-log.md`](docs/research-log.md) — development log
- [`docs/adr/`](docs/adr/) — architecture decision records

## Licence

MIT — see [LICENSE](LICENSE).