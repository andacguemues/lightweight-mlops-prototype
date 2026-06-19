import json
from functools import lru_cache
from pathlib import Path
from typing import Any

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
REGISTRY_INFO_PATH = (
    PROJECT_ROOT / "reports/evaluation/registry_info.json"
)


@lru_cache
def load_params() -> dict:
    """Load the central project configuration."""

    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def configure_mlflow() -> str:
    """Configure MLflow tracking and registry access."""

    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    return tracking_uri


@lru_cache
def get_model() -> Any:
    """Load the model assigned to the configured registry alias."""

    params = load_params()
    configure_mlflow()

    model_name = params["mlflow"]["registered_model_name"]
    model_alias = params["mlflow"]["model_alias"]
    model_uri = f"models:/{model_name}@{model_alias}"

    return mlflow.sklearn.load_model(model_uri)


def load_registry_info() -> dict:
    """Load locally documented registry metadata."""

    if not REGISTRY_INFO_PATH.exists():
        return {}

    with REGISTRY_INFO_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


app = FastAPI(
    title="Bank Marketing Prediction API",
    description=(
        "Prediction service for the lightweight MLOps prototype."
    ),
    version="0.1.0",
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(
    model: Any = Depends(get_model),
) -> HealthResponse:
    """Check whether the application can load the registered model."""

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
    """Return information about the currently served model."""

    del model

    params = load_params()
    registry_info = load_registry_info()

    return ModelInfoResponse(
        status=registry_info.get("status", "unknown"),
        registered_model_name=params["mlflow"][
            "registered_model_name"
        ],
        model_alias=params["mlflow"]["model_alias"],
        model_version=str(registry_info.get("model_version")),
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
    """Generate a subscription prediction."""

    try:
        feature_frame = pd.DataFrame(
            [request.to_feature_dict()]
        )

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