# Monitoring Design

## Objective

The monitoring component evaluates whether incoming feature
distributions and model prediction probabilities differ from the
reference data used during model evaluation.

## Reference Data

The feature portion of `data/processed/test.csv` is used as the
reference dataset.

The target column is excluded because production labels are not assumed
to be immediately available.

## Current Data Scenarios

### No Drift

An exact copy of the reference features is used.

Expected result:

- no dataset-level drift
- no false-positive monitoring alert

### Deliberate Drift

Multiple numerical distributions and categorical frequencies are
modified deterministically.

Expected result:

- dataset-level drift
- several changed columns identified as drifted

## Prediction Drift

The trained model is applied to reference and current data. Its positive
class probability is stored as `prediction_probability` and included in
the drift report.

## Dataset Drift Threshold

Dataset drift is declared when at least 20 percent of monitored columns
are classified as drifted.

This threshold is a prototype design decision and can be reassessed
during the Bachelor thesis evaluation.

## Outputs

The component generates:

- interactive HTML reports,
- machine-readable JSON reports,
- a compact monitoring summary,
- a manifest documenting the simulated changes.

## Limitation

The current implementation performs offline batch monitoring. It does
not yet collect live API requests or trigger automated retraining.  