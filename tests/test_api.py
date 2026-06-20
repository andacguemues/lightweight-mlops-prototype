import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops_prototype.api.main import app, get_model


class DummyModel:
    """Small deterministic test replacement for the real model."""

    def predict(self, data):
        assert len(data) == 1
        return np.array([1])

    def predict_proba(self, data):
        assert len(data) == 1
        return np.array([[0.2, 0.8]])


@pytest.fixture
def client():
    app.dependency_overrides[get_model] = lambda: DummyModel()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def valid_payload() -> dict:
    return {
        "age": 40,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 300,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": -1.8,
        "cons_price_idx": 92.893,
        "cons_conf_idx": -46.2,
        "euribor3m": 1.299,
        "nr_employed": 5099.1,
    }


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_predict(
    client: TestClient,
    valid_payload: dict,
) -> None:
    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == "yes"
    assert result["probability_yes"] == pytest.approx(0.8)
    assert result["model_alias"] == "candidate"


def test_predict_rejects_missing_fields(
    client: TestClient,
) -> None:
    response = client.post(
        "/predict",
        json={"age": 40},
    )

    assert response.status_code == 422
