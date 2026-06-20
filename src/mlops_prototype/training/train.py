from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    params = load_params()

    train_path = PROJECT_ROOT / params["data"]["train_path"]
    model_path = PROJECT_ROOT / params["model"]["output_path"]
    target = params["data"]["target"]

    df = pd.read_csv(train_path)

    X = df.drop(columns=[target])
    y = df[target].map({"no": 0, "yes": 1})

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=params["model"]["max_iter"],
        class_weight=params["model"]["class_weight"],
        random_state=params["data"]["random_state"],
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
