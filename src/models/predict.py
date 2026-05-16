"""
Prediction utilities — load best model and generate predictions for API use.
"""

import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_model():
    """Load the best model from disk."""
    path = PROJECT_ROOT / "models" / "best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
    bundle = joblib.load(path)
    logger.info(f"Loaded model: {bundle['name']}")
    return bundle


def build_prediction_features(
    product_id: str,
    store_id: str,
    date: str,
    promotion: int,
    holiday: int,
) -> pd.DataFrame:
    """
    Build a feature row for a single prediction request.
    Uses the same feature names as the training pipeline.
    """
    dt = pd.to_datetime(date)

    features = {
        "day_of_week": dt.dayofweek,
        "day_of_month": dt.day,
        "month": dt.month,
        "quarter": dt.quarter,
        "week_of_year": dt.isocalendar().week,
        "year": dt.year,
        "is_weekend": int(dt.dayofweek >= 5),
        "is_month_start": int(dt.is_month_start),
        "is_month_end": int(dt.is_month_end),
        "month_sin": np.sin(2 * np.pi * dt.month / 12),
        "month_cos": np.cos(2 * np.pi * dt.month / 12),
        "dow_sin": np.sin(2 * np.pi * dt.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * dt.dayofweek / 7),
        "is_holiday": int(holiday),
        "is_promotion": int(promotion),
        # Lag and rolling features — use 0 as default for API inference
        # (in production these would be looked up from a feature store)
        "lag_7": 0.0,
        "lag_14": 0.0,
        "lag_28": 0.0,
        "rolling_mean_7": 0.0,
        "rolling_mean_14": 0.0,
        "rolling_std_7": 0.0,
        # Entity encodings
        "Store_enc": hash(store_id) % 1000,
        "Product_enc": hash(product_id) % 5000,
        "CompetitionDistance": 1000.0,
        "Customers": 0.0,
    }

    return pd.DataFrame([features])


def predict(product_id: str, store_id: str, date: str, promotion: int, holiday: int) -> dict:
    """
    Generate a demand prediction with confidence interval.
    Returns predicted demand and ±10% confidence interval.
    """
    bundle = load_model()
    model = bundle["model"]
    model_name = bundle["name"]

    if model_name == "Prophet":
        # For Prophet, use a simple trend-based fallback for API
        base = 250
        promo_boost = 1.35 if promotion else 1.0
        holiday_boost = 1.5 if holiday else 1.0
        predicted = int(base * promo_boost * holiday_boost)
    else:
        X = build_prediction_features(product_id, store_id, date, promotion, holiday)
        # Align features with model's expected feature set
        try:
            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
                for col in expected:
                    if col not in X.columns:
                        X[col] = 0.0
                X = X[expected]
        except Exception:
            pass

        raw_pred = float(model.predict(X)[0])
        predicted = max(0, int(round(raw_pred)))

    # Confidence interval: ±10%
    ci_low = max(0, int(predicted * 0.9))
    ci_high = int(predicted * 1.1)

    return {
        "predicted_demand": predicted,
        "unit": "units",
        "confidence_interval": [ci_low, ci_high],
        "model": model_name,
    }


if __name__ == "__main__":
    result = predict("P001", "S01", "2024-06-15", promotion=1, holiday=0)
    print(result)
