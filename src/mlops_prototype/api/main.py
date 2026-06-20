import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from fastapi import Depends, FastAPI, HTTPException

from mlops_prototype.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

DEFAULT_REGISTRY_INFO_PATH = PROJECT_ROOT / "reports/evaluation/registry_info.json"


@lru_cache
def load_params() -> dict:
    """Load the central project configuration."""

    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(path_value: str) -> Path:
    """Resolve relative paths from the project root."""

    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_model_source() -> str:
    """Return and validate the configured model source."""

    model_source = (
        os.getenv(
            "MODEL_SOURCE",
            "registry",
        )
        .strip()
        .lower()
    )

    if model_source not in {"registry", "file"}:
        raise RuntimeError("MODEL_SOURCE must be either 'registry' or 'file'.")

    return model_source


def configure_mlflow() -> str:
    """Configure access to the local MLflow backend."""

    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    return tracking_uri


@lru_cache
def get_model() -> Any:
    """Load the configured model from MLflow or a local file."""

    model_source = get_model_source()

    if model_source == "file":
        configured_path = os.getenv(
            "MODEL_PATH",
            "models/model.joblib",
        )
        model_path = resolve_project_path(configured_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        return joblib.load(model_path)

    params = load_params()
    configure_mlflow()

    model_name = params["mlflow"]["registered_model_name"]
    model_alias = params["mlflow"]["model_alias"]
    model_uri = f"models:/{model_name}@{model_alias}"

    return mlflow.sklearn.load_model(model_uri)


def load_registry_info() -> dict:
    """Load the documented registry metadata."""

    configured_path = os.getenv(
        "REGISTRY_INFO_PATH",
        str(DEFAULT_REGISTRY_INFO_PATH),
    )
    registry_info_path = resolve_project_path(configured_path)

    if not registry_info_path.exists():
        return {}

    with registry_info_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


app = FastAPI(
    title="Bank Marketing Prediction API",
    description=("Prediction service for the lightweight MLOps prototype."),
    version="0.1.0",
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(
    model: Any = Depends(get_model),
) -> HealthResponse:
    """Check whether the application can load its model."""

    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info(
    model: Any = Depends(get_model),
) -> ModelInfoResponse:
    """Return information about the served model."""

    del model

    params = load_params()
    registry_info = load_registry_info()
    model_source = get_model_source()

    default_status = "registered" if model_source == "registry" else "loaded_from_file"

    return ModelInfoResponse(
        status=registry_info.get(
            "status",
            default_status,
        ),
        model_source=model_source,
        registered_model_name=params["mlflow"]["registered_model_name"],
        model_alias=params["mlflow"]["model_alias"],
        model_version=(
            str(registry_info["model_version"])
            if registry_info.get("model_version") is not None
            else None
        ),
        roc_auc=registry_info.get("roc_auc"),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
    model: Any = Depends(get_model),
) -> PredictionResponse:
    """Generate a term-deposit subscription prediction."""

    try:
        feature_frame = pd.DataFrame([request.to_feature_dict()])

        prediction = int(model.predict(feature_frame)[0])
        probabilities = model.predict_proba(feature_frame)[0]
        probability_yes = float(probabilities[1])

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {exc}",
        ) from exc

    params = load_params()

    return PredictionResponse(
        prediction="yes" if prediction == 1 else "no",
        probability_yes=probability_yes,
        model_name=params["mlflow"]["registered_model_name"],
        model_alias=params["mlflow"]["model_alias"],
    )
