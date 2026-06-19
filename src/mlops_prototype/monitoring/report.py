import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict:
    """Load the central project configuration."""

    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def add_predictions(
    data: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Add model prediction probabilities to monitoring data."""

    result = data.copy()

    result["prediction_probability"] = model.predict_proba(
        data
    )[:, 1]

    return result


def extract_drift_summary(
    snapshot_data: dict,
    drift_share_threshold: float,
) -> dict:
    """Extract dataset-level and column-level drift information."""

    metrics = snapshot_data.get("metrics", [])
    tests = snapshot_data.get("tests", [])

    drifted_count = None
    detected_drift_share = None

    for metric in metrics:
        config = metric.get("config", {})

        metric_type = str(
            config.get("type", "")
        ).rsplit(":", maxsplit=1)[-1]

        metric_name = str(
            metric.get("metric_name", "")
        )

        if (
            metric_type == "DriftedColumnsCount"
            or metric_name.startswith("DriftedColumnsCount(")
        ):
            value = metric.get("value") or {}

            drifted_count = value.get("count")
            detected_drift_share = value.get("share")
            break

    drifted_columns: list[str] = []

    for test in tests:
        raw_status = test.get("status", "")

        status = getattr(
            raw_status,
            "value",
            raw_status,
        )

        status = str(status).lower()

        metric_config = test.get("metric_config", {})
        metric_params = metric_config.get("params", {})

        metric_type = str(
            metric_params.get("type", "")
        ).rsplit(":", maxsplit=1)[-1]

        column = metric_params.get("column")

        if (
            metric_type == "ValueDrift"
            and status == "fail"
            and column is not None
        ):
            drifted_columns.append(str(column))

    dataset_drift = (
        detected_drift_share is not None
        and detected_drift_share >= drift_share_threshold
    )

    return {
        "dataset_drift": bool(dataset_drift),
        "drifted_column_count": drifted_count,
        "drifted_column_share": detected_drift_share,
        "drifted_columns": sorted(set(drifted_columns)),
        "drift_share_threshold": drift_share_threshold,
    }


def run_scenario(
    scenario_name: str,
    current_path: Path,
    reference_data: pd.DataFrame,
    model,
    report_dir: Path,
    drift_share: float,
    expected_dataset_drift: bool,
) -> dict:
    """Run an Evidently drift report for one scenario."""

    current_data = pd.read_csv(current_path)

    reference_with_predictions = add_predictions(
        reference_data,
        model,
    )
    current_with_predictions = add_predictions(
        current_data,
        model,
    )

    report = Report(
        [
            DataDriftPreset(
                drift_share=drift_share,
            ),
        ],
        include_tests=True,
        tags=["monitoring", scenario_name],
        metadata={
            "scenario": scenario_name,
            "model_output": "prediction_probability",
        },
    )

    snapshot = report.run(
        current_data=current_with_predictions,
        reference_data=reference_with_predictions,
        name=f"{scenario_name}-data-drift",
    )

    html_path = report_dir / f"{scenario_name}_report.html"
    json_path = report_dir / f"{scenario_name}_report.json"

    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    summary = extract_drift_summary(
        snapshot_data=snapshot.dict(),
        drift_share_threshold=drift_share,
    )

    summary.update(
        {
            "scenario": scenario_name,
            "current_path": str(
                current_path.relative_to(PROJECT_ROOT)
            ),
            "html_report": str(
                html_path.relative_to(PROJECT_ROOT)
            ),
            "json_report": str(
                json_path.relative_to(PROJECT_ROOT)
            ),
            "expected_dataset_drift": expected_dataset_drift,
        }
    )

    summary["expectation_met"] = (
        summary["dataset_drift"]
        == expected_dataset_drift
    )

    return summary


def main() -> None:
    params = load_params()

    reference_path = (
        PROJECT_ROOT
        / params["monitoring"]["reference_path"]
    )
    no_drift_path = (
        PROJECT_ROOT
        / params["monitoring"]["no_drift_path"]
    )
    drifted_path = (
        PROJECT_ROOT
        / params["monitoring"]["drifted_path"]
    )
    report_dir = (
        PROJECT_ROOT
        / params["monitoring"]["report_dir"]
    )
    summary_path = (
        PROJECT_ROOT
        / params["monitoring"]["summary_path"]
    )
    model_path = (
        PROJECT_ROOT
        / params["model"]["output_path"]
    )

    target = params["data"]["target"]
    drift_share = params["monitoring"]["drift_share"]

    reference_data = pd.read_csv(reference_path)

    if target in reference_data.columns:
        reference_features = reference_data.drop(
            columns=[target]
        )
    else:
        reference_features = reference_data.copy()

    model = joblib.load(model_path)

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenarios = [
        run_scenario(
            scenario_name="no_drift",
            current_path=no_drift_path,
            reference_data=reference_features,
            model=model,
            report_dir=report_dir,
            drift_share=drift_share,
            expected_dataset_drift=False,
        ),
        run_scenario(
            scenario_name="drifted",
            current_path=drifted_path,
            reference_data=reference_features,
            model=model,
            report_dir=report_dir,
            drift_share=drift_share,
            expected_dataset_drift=True,
        ),
    ]

    monitoring_summary = {
        "reference_path": str(
            reference_path.relative_to(PROJECT_ROOT)
        ),
        "model_path": str(
            model_path.relative_to(PROJECT_ROOT)
        ),
        "drift_share_threshold": drift_share,
        "scenario_count": len(scenarios),
        "successful_scenarios": sum(
            scenario["expectation_met"]
            for scenario in scenarios
        ),
        "all_expectations_met": all(
            scenario["expectation_met"]
            for scenario in scenarios
        ),
        "scenarios": scenarios,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            monitoring_summary,
            file,
            indent=2,
        )

    print(json.dumps(monitoring_summary, indent=2))

    if not monitoring_summary["all_expectations_met"]:
        raise RuntimeError(
            "At least one monitoring scenario did not "
            "produce the expected drift result."
        )


if __name__ == "__main__":
    main()