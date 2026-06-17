import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    params = load_params()

    test_path = PROJECT_ROOT / params["data"]["test_path"]
    model_path = PROJECT_ROOT / params["model"]["output_path"]
    metrics_path = PROJECT_ROOT / params["evaluation"]["metrics_path"]
    target = params["data"]["target"]

    df = pd.read_csv(test_path)

    X = df.drop(columns=[target])
    y_true = df[target].map({"no": 0, "yes": 1})

    model = joblib.load(model_path)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()