import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"


def load_params() -> dict:
    """Load the central project configuration."""
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_sha256(path: Path) -> str:
    """Calculate a SHA-256 hash for a file."""
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_git_commit() -> str:
    """Return the current Git commit or 'unknown' if unavailable."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip()


def configure_mlflow(params: dict) -> str:
    """Configure MLflow to use the local SQLite backend."""
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    return tracking_uri


def main() -> None:
    params = load_params()

    test_path = PROJECT_ROOT / params["data"]["test_path"]
    model_path = PROJECT_ROOT / params["model"]["output_path"]
    metrics_path = PROJECT_ROOT / params["evaluation"]["metrics_path"]
    run_info_path = PROJECT_ROOT / "reports/evaluation/mlflow_run.json"
    target = params["data"]["target"]

    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=[target])
    y_true = test_df[target].map({"no": 0, "yes": 1})

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_proba)
        ),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    run_info_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    tracking_uri = configure_mlflow(params)

    input_example = X_test.head(5).copy()
    example_predictions = model.predict(input_example)
    signature = infer_signature(
        input_example,
        example_predictions,
    )

    with mlflow.start_run(
        run_name=params["mlflow"]["run_name"]
    ) as run:
        mlflow.log_params(
            {
                "model_type": params["model"]["type"],
                "max_iter": params["model"]["max_iter"],
                "class_weight": params["model"]["class_weight"],
                "random_state": params["data"]["random_state"],
                "test_size": params["data"]["test_size"],
                "test_rows": len(test_df),
                "feature_count": X_test.shape[1],
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.set_tags(
            {
                "workflow": "dvc",
                "dataset": "uci-bank-marketing",
                "git_commit": get_git_commit(),
                "test_data_sha256": calculate_sha256(test_path),
                "model_sha256": calculate_sha256(model_path),
            }
        )

        mlflow.log_artifact(
            str(PARAMS_PATH),
            artifact_path="project",
        )

        dvc_lock_path = PROJECT_ROOT / "dvc.lock"

        if dvc_lock_path.exists():
            mlflow.log_artifact(
                str(dvc_lock_path),
                artifact_path="project",
            )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        run_info = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "tracking_uri": tracking_uri,
        }

        with run_info_path.open("w", encoding="utf-8") as file:
            json.dump(run_info, file, indent=2)

    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"MLflow run ID: {run_info['run_id']}")
    print(f"MLflow model URI: {run_info['model_uri']}")


if __name__ == "__main__":
    main()