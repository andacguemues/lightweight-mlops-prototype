import json
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_params() -> dict:
    """Load the central project configuration."""
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def configure_mlflow() -> str:
    """Configure MLflow tracking and registry."""
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    return tracking_uri


def main() -> None:
    params = load_params()

    metrics_path = PROJECT_ROOT / params["evaluation"]["metrics_path"]
    run_info_path = PROJECT_ROOT / "reports/evaluation/mlflow_run.json"
    registry_info_path = (
        PROJECT_ROOT / "reports/evaluation/registry_info.json"
    )

    metrics = load_json(metrics_path)
    run_info = load_json(run_info_path)

    minimum_roc_auc = params["mlflow"]["minimum_roc_auc"]
    model_name = params["mlflow"]["registered_model_name"]
    model_alias = params["mlflow"]["model_alias"]

    tracking_uri = configure_mlflow()

    if metrics["roc_auc"] < minimum_roc_auc:
        registry_info = {
            "status": "not_registered",
            "reason": "roc_auc_below_threshold",
            "roc_auc": metrics["roc_auc"],
            "minimum_roc_auc": minimum_roc_auc,
            "run_id": run_info["run_id"],
        }

        registry_info_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with registry_info_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(registry_info, file, indent=2)

        print(json.dumps(registry_info, indent=2))
        return

    model_version = mlflow.register_model(
        model_uri=run_info["model_uri"],
        name=model_name,
    )

    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )

    client.set_registered_model_alias(
        name=model_name,
        alias=model_alias,
        version=model_version.version,
    )

    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="validation_status",
        value="passed",
    )

    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="roc_auc",
        value=str(metrics["roc_auc"]),
    )

    registry_info = {
        "status": "registered",
        "registered_model_name": model_name,
        "model_version": model_version.version,
        "model_alias": model_alias,
        "model_uri": f"models:/{model_name}/{model_version.version}",
        "alias_uri": f"models:/{model_name}@{model_alias}",
        "run_id": run_info["run_id"],
        "roc_auc": metrics["roc_auc"],
        "minimum_roc_auc": minimum_roc_auc,
    }

    registry_info_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with registry_info_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(registry_info, file, indent=2)

    print(json.dumps(registry_info, indent=2))


if __name__ == "__main__":
    main()  