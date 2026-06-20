
### `docs/architecture.md`

```markdown
# System Architecture

## Purpose

The system demonstrates a lightweight, locally executable ML lifecycle
covering reproducible training, traceability, deployment and monitoring.

## Components

| Component | Responsibility |
|---|---|
| Notebook baseline | Represents the exploratory comparison workflow |
| DVC pipeline | Reproduces data preparation, training, evaluation and monitoring |
| Training component | Builds the classification pipeline |
| MLflow tracking | Stores parameters, metrics, tags and model artifacts |
| Model registry | Versions accepted models and assigns the candidate alias |
| Prediction service | Exposes the selected model over HTTP |
| Monitoring component | Detects feature and prediction drift |
| Docker image | Packages the selected model and prediction service |
| Validation script | Verifies the completeness of the artifact |

## Training Flow

```text
Raw data
→ train/test split
→ preprocessing
→ training
→ evaluation
→ MLflow run
→ model registration