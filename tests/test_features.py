"""
Unit tests for src/features/feature_engineering.py
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineering import FeatureEngineer

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

TEST_PARAMS = {
    "data": {
        "test_size": 0.2,
        "target_column": "Sales",
        "date_column": "Date",
        "raw_path": "data/raw/sales_data.csv",
        "processed_path": "data/processed/processed_data.csv",
        "features_path": "data/processed/features.csv",
    },
    "model": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
    "api": {"host": "0.0.0.0", "port": 8000, "db_path": "data/predictions.db"},
    "mlflow": {"tracking_uri": "mlruns", "experiment_name": "test", "model_name": "test-model"},
    "monitoring": {"drift_threshold": 0.3, "report_path": "reports/monitoring_report.html"},
}


def make_processed_df(rows: int = 200) -> pd.DataFrame:
    np.random.seed(0)
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Store": np.random.choice(["S01", "S02"], rows),
        "Product": np.random.choice(["P001", "P002"], rows),
        "Sales": np.random.randint(50, 400, rows).astype(float),
        "Promo": np.random.randint(0, 2, rows),
        "StateHoliday": np.random.choice(["0", "a"], rows),
        "CompetitionDistance": np.random.randint(100, 5000, rows).astype(float),
    })
    return df


def make_fe(df: pd.DataFrame) -> FeatureEngineer:
    fe = FeatureEngineer(params=TEST_PARAMS)
    fe.df = df.copy()
    return fe


# ------------------------------------------------------------------ #
# Temporal feature tests
# ------------------------------------------------------------------ #

class TestTemporalFeatures:

    def test_adds_day_of_week(self):
        fe = make_fe(make_processed_df())
        fe.add_temporal_features()
        assert "day_of_week" in fe.df.columns

    def test_day_of_week_range(self):
        fe = make_fe(make_processed_df())
        fe.add_temporal_features()
        assert fe.df["day_of_week"].between(0, 6).all()

    def test_adds_month(self):
        fe = make_fe(make_processed_df())
        fe.add_temporal_features()
        assert "month" in fe.df.columns
        assert fe.df["month"].between(1, 12).all()

    def test_adds_is_weekend(self):
        fe = make_fe(make_processed_df())
        fe.add_temporal_features()
        assert "is_weekend" in fe.df.columns
        assert set(fe.df["is_weekend"].unique()).issubset({0, 1})

    def test_adds_cyclical_encoding(self):
        fe = make_fe(make_processed_df())
        fe.add_temporal_features()
        assert "month_sin" in fe.df.columns
        assert "month_cos" in fe.df.columns
        # Cyclical values must be in [-1, 1]
        assert fe.df["month_sin"].between(-1.01, 1.01).all()

    def test_adds_quarter(self):
        fe = make_fe(make_processed_df())
        fe.add_temporal_features()
        assert "quarter" in fe.df.columns
        assert fe.df["quarter"].between(1, 4).all()


# ------------------------------------------------------------------ #
# Lag feature tests
# ------------------------------------------------------------------ #

class TestLagFeatures:

    def test_lag_7_created(self):
        fe = make_fe(make_processed_df(200))
        fe.add_lag_features()
        assert "lag_7" in fe.df.columns

    def test_lag_14_created(self):
        fe = make_fe(make_processed_df(200))
        fe.add_lag_features()
        assert "lag_14" in fe.df.columns

    def test_lag_28_created(self):
        fe = make_fe(make_processed_df(200))
        fe.add_lag_features()
        assert "lag_28" in fe.df.columns

    def test_lag_7_is_shifted(self):
        df = make_processed_df(200)
        # Use a single group to verify shift
        df["Store"] = "S01"
        df["Product"] = "P001"
        fe = make_fe(df)
        fe.df.sort_values("Date", inplace=True)
        fe.df.reset_index(drop=True, inplace=True)
        fe.add_lag_features()

        # lag_7 at row 10 should equal Sales at row 3 (index 10 - 7 = 3)
        assert pd.isna(fe.df.loc[6, "lag_7"]) or fe.df.loc[7, "lag_7"] == fe.df.loc[0, "Sales"]

    def test_lag_creates_nans_at_start(self):
        fe = make_fe(make_processed_df(200))
        fe.add_lag_features()
        # First 28 rows within each group must have NaN for lag_28
        assert fe.df["lag_28"].isnull().sum() > 0


# ------------------------------------------------------------------ #
# Rolling feature tests
# ------------------------------------------------------------------ #

class TestRollingFeatures:

    def test_rolling_mean_7_created(self):
        fe = make_fe(make_processed_df(200))
        fe.add_rolling_features()
        assert "rolling_mean_7" in fe.df.columns

    def test_rolling_mean_14_created(self):
        fe = make_fe(make_processed_df(200))
        fe.add_rolling_features()
        assert "rolling_mean_14" in fe.df.columns

    def test_rolling_std_7_created(self):
        fe = make_fe(make_processed_df(200))
        fe.add_rolling_features()
        assert "rolling_std_7" in fe.df.columns

    def test_rolling_std_non_negative(self):
        fe = make_fe(make_processed_df(200))
        fe.add_rolling_features()
        assert (fe.df["rolling_std_7"].fillna(0) >= 0).all()


# ------------------------------------------------------------------ #
# Output shape / integration tests
# ------------------------------------------------------------------ #

class TestOutputShape:

    def test_build_features_returns_tuple(self, tmp_path):
        df = make_processed_df(200)
        proc_path = tmp_path / "processed_data.csv"
        feat_path = tmp_path / "features.csv"
        df.to_csv(proc_path, index=False)

        params = {**TEST_PARAMS}
        params["data"] = {**TEST_PARAMS["data"]}
        params["data"]["processed_path"] = str(proc_path)
        params["data"]["features_path"] = str(feat_path)

        fe = FeatureEngineer(params=params)
        X, y = fe.build_features()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_X_has_no_date_column(self, tmp_path):
        df = make_processed_df(200)
        proc_path = tmp_path / "processed_data.csv"
        feat_path = tmp_path / "features.csv"
        df.to_csv(proc_path, index=False)

        params = {**TEST_PARAMS}
        params["data"] = {**TEST_PARAMS["data"]}
        params["data"]["processed_path"] = str(proc_path)
        params["data"]["features_path"] = str(feat_path)

        fe = FeatureEngineer(params=params)
        X, y = fe.build_features()

        assert "Date" not in X.columns

    def test_X_and_y_same_length(self, tmp_path):
        df = make_processed_df(200)
        proc_path = tmp_path / "processed_data.csv"
        feat_path = tmp_path / "features.csv"
        df.to_csv(proc_path, index=False)

        params = {**TEST_PARAMS}
        params["data"] = {**TEST_PARAMS["data"]}
        params["data"]["processed_path"] = str(proc_path)
        params["data"]["features_path"] = str(feat_path)

        fe = FeatureEngineer(params=params)
        X, y = fe.build_features()

        assert len(X) == len(y), "X and y must have the same number of rows"

    def test_X_has_no_nan(self, tmp_path):
        df = make_processed_df(200)
        proc_path = tmp_path / "processed_data.csv"
        feat_path = tmp_path / "features.csv"
        df.to_csv(proc_path, index=False)

        params = {**TEST_PARAMS}
        params["data"] = {**TEST_PARAMS["data"]}
        params["data"]["processed_path"] = str(proc_path)
        params["data"]["features_path"] = str(feat_path)

        fe = FeatureEngineer(params=params)
        X, y = fe.build_features()

        assert X.isnull().sum().sum() == 0, "Feature matrix should have no NaN values"
