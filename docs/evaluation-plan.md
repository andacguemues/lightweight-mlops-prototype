
### `docs/evaluation-plan.md`

```markdown
# Evaluation Plan

## Objective

The prototype is evaluated against the notebook-based baseline
workflow.

The evaluation examines measurable benefits as well as additional
implementation effort.

## Evaluation Criteria

| Criterion | Measurement |
|---|---|
| Reproducibility | Successful execution of the complete DVC pipeline |
| Traceability | Availability of code state, parameters, metrics, run ID and model version |
| Automation | Number of commands or manual interventions required |
| Deployment | Successful API health check and prediction request |
| Monitoring | Correct results for controlled no-drift and drift scenarios |
| Resource suitability | Successful local execution in the target environment |
| Transferability | Availability of setup, architecture and operating documentation |

## Baseline

The baseline is the notebook-based workflow created before the MLOps
components were introduced.

It includes:

- manual notebook execution,
- local model storage,
- no systematic model registry,
- no deployment service,
- no drift monitoring.

## Prototype

The prototype includes:

- reproducible pipeline execution,
- MLflow experiment tracking,
- model registration,
- API deployment,
- batch drift monitoring,
- automated tests and validation.

## Evidence

Evaluation evidence is generated from:

- `reports/evaluation/metrics.json`
- `reports/evaluation/mlflow_run.json`
- `reports/evaluation/registry_info.json`
- `reports/evaluation/project_validation.json`
- `reports/monitoring/monitoring_summary.json`
- automated test output
- Docker health and prediction requests