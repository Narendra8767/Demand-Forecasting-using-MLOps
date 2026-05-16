"""
Unit tests for src/api/main.py — FastAPI endpoints
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient


# ------------------------------------------------------------------ #
# Fixtures and helpers
# ------------------------------------------------------------------ #

def make_mock_model():
    """Create a mock sklearn-like model that returns 350."""
    model = MagicMock()
    model.predict.return_value = np.array([350.0])
    model.feature_names_in_ = [
        "day_of_week", "day_of_month", "month", "quarter", "week_of_year",
        "year", "is_weekend", "is_month_start", "is_month_end",
        "month_sin", "month_cos", "dow_sin", "dow_cos",
        "is_holiday", "is_promotion",
        "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
        "Store_enc", "Product_enc", "CompetitionDistance", "Customers",
    ]
    return model


MOCK_BUNDLE = {
    "model": make_mock_model(),
    "name": "RandomForest",
    "metrics": {"rmse": 45.2, "mae": 32.1, "r2": 0.87, "mape": 12.5},
    "source": "local_pkl",
}


@pytest.fixture
def client():
    """Create TestClient with mocked model."""
    with patch("src.api.main.load_model_bundle", return_value=MOCK_BUNDLE):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


# ------------------------------------------------------------------ #
# /health tests
# ------------------------------------------------------------------ #

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_model_field(self, client):
        data = client.get("/health").json()
        assert "model" in data

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True


# ------------------------------------------------------------------ #
# /model-info tests
# ------------------------------------------------------------------ #

class TestModelInfoEndpoint:

    def test_model_info_returns_200(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_metrics(self, client):
        data = client.get("/model-info").json()
        assert "metrics" in data

    def test_model_info_has_model_name(self, client):
        data = client.get("/model-info").json()
        assert "model_name" in data


# ------------------------------------------------------------------ #
# /predict tests
# ------------------------------------------------------------------ #

class TestPredictEndpoint:

    VALID_PAYLOAD = {
        "product_id": "P001",
        "store_id": "S01",
        "date": "2024-06-15",
        "promotion": 1,
        "holiday": 0,
    }

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_has_predicted_demand(self, client):
        data = client.post("/predict", json=self.VALID_PAYLOAD).json()
        assert "predicted_demand" in data

    def test_predict_demand_is_non_negative(self, client):
        data = client.post("/predict", json=self.VALID_PAYLOAD).json()
        assert data["predicted_demand"] >= 0

    def test_predict_has_confidence_interval(self, client):
        data = client.post("/predict", json=self.VALID_PAYLOAD).json()
        assert "confidence_interval" in data
        ci = data["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] <= ci[1], "CI lower bound must be ≤ upper bound"

    def test_predict_has_unit(self, client):
        data = client.post("/predict", json=self.VALID_PAYLOAD).json()
        assert data["unit"] == "units"

    def test_predict_has_model_field(self, client):
        data = client.post("/predict", json=self.VALID_PAYLOAD).json()
        assert "model" in data

    def test_predict_has_timestamp(self, client):
        data = client.post("/predict", json=self.VALID_PAYLOAD).json()
        assert "timestamp" in data

    def test_predict_with_no_promotion(self, client):
        payload = {**self.VALID_PAYLOAD, "promotion": 0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_with_holiday(self, client):
        payload = {**self.VALID_PAYLOAD, "holiday": 1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_invalid_promotion_value(self, client):
        payload = {**self.VALID_PAYLOAD, "promotion": 5}  # out of 0-1 range
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_required_field(self, client):
        payload = {"product_id": "P001", "store_id": "S01"}  # missing date
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_different_stores(self, client):
        for store in ["S01", "S02", "S10"]:
            payload = {**self.VALID_PAYLOAD, "store_id": store}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200


# ------------------------------------------------------------------ #
# /predictions/history tests
# ------------------------------------------------------------------ #

class TestHistoryEndpoint:

    def test_history_returns_200(self, client):
        response = client.get("/predictions/history")
        assert response.status_code == 200

    def test_history_returns_list(self, client):
        data = client.get("/predictions/history").json()
        assert isinstance(data, list)

    def test_history_limit_parameter(self, client):
        response = client.get("/predictions/history?limit=5")
        assert response.status_code == 200


# ------------------------------------------------------------------ #
# /model-info when no model loaded
# ------------------------------------------------------------------ #

class TestNoModelLoaded:

    def test_model_info_503_when_no_model(self):
        with patch("src.api.main.load_model_bundle", return_value=None):
            from src.api.main import app
            with TestClient(app) as c:
                response = c.get("/model-info")
                assert response.status_code == 503
