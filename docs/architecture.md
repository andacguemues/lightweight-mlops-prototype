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
→ candidate alias
```

## Serving Flow

```text
Model registry (alias: candidate)
→ prediction service
→ validated request
→ prediction + model metadata
```

## Monitoring Flow

```text
Reference data (test set, target removed)
+ simulated current data
→ drift detection (feature and prediction distributions)
→ HTML report + machine-readable summary
→ expectation check (fails the pipeline on mismatch)
```

## Cross-cutting Components

| Component | Role |
|---|---|
| DVC | Dependency graph and execution control for the pipeline |
| MLflow | Run tracking and model registry, backed by a local SQLite file |

No cluster, no managed cloud services and no external runtime
dependencies are required.