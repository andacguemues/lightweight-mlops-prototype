import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_PATH = PROJECT_ROOT / "reports/evaluation/metrics.json"
REGISTRY_INFO_PATH = PROJECT_ROOT / "reports/evaluation/registry_info.json"
MONITORING_SUMMARY_PATH = PROJECT_ROOT / "reports/monitoring/monitoring_summary.json"
MODEL_PATH = PROJECT_ROOT / "models/model.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data/processed/test.csv"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_model_metrics_are_complete() -> None:
    metrics = load_json(METRICS_PATH)

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert expected_metrics.issubset(metrics)
    assert all(0.0 <= float(metrics[name]) <= 1.0 for name in expected_metrics)


def test_model_was_registered() -> None:
    registry_info = load_json(REGISTRY_INFO_PATH)

    assert registry_info["status"] == "registered"
    assert registry_info["registered_model_name"]
    assert registry_info["model_version"]
    assert registry_info["model_alias"] == "candidate"


def test_monitoring_expectations_were_met() -> None:
    summary = load_json(MONITORING_SUMMARY_PATH)

    assert summary["scenario_count"] == 2
    assert summary["successful_scenarios"] == 2
    assert summary["all_expectations_met"] is True


def test_model_artifact_can_predict() -> None:
    model = joblib.load(MODEL_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    features = test_data.drop(columns=["y"]).head(5)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    assert len(predictions) == 5
    assert probabilities.shape == (5, 2)
