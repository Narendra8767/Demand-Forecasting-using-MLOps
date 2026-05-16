"""
Unit tests for src/data/preprocess.py
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import Preprocessor

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


def make_sample_df(rows=100, add_nans=False, add_dups=False) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Store": np.random.choice(["S01", "S02", "S03"], rows),
        "Product": np.random.choice(["P001", "P002"], rows),
        "Sales": np.random.randint(50, 500, rows).astype(float),
        "Customers": np.random.randint(10, 100, rows).astype(float),
        "Promo": np.random.randint(0, 2, rows),
        "StateHoliday": np.random.choice(["0", "a", "b"], rows),
        "SchoolHoliday": np.random.randint(0, 2, rows),
        "CompetitionDistance": np.random.randint(100, 5000, rows).astype(float),
        "StoreType": np.random.choice(["a", "b", "c"], rows),
    })
    if add_nans:
        df.loc[0:5, "Sales"] = np.nan
        df.loc[3:7, "Customers"] = np.nan
        df.loc[10:12, "StateHoliday"] = np.nan
    if add_dups:
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    return df


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestPreprocessor:

    def _make_preprocessor(self, df: pd.DataFrame) -> Preprocessor:
        p = Preprocessor(params=TEST_PARAMS)
        p.df = df.copy()
        return p

    def test_handle_missing_values_numeric(self):
        df = make_sample_df(100, add_nans=True)
        p = self._make_preprocessor(df)
        p.handle_missing_values()

        assert p.df["Sales"].isnull().sum() == 0, "Numeric NaNs should be filled"
        assert p.df["Customers"].isnull().sum() == 0

    def test_handle_missing_values_categorical(self):
        df = make_sample_df(100, add_nans=True)
        p = self._make_preprocessor(df)
        p.handle_missing_values()

        assert p.df["StateHoliday"].isnull().sum() == 0, "Categorical NaNs should be filled"

    def test_remove_duplicates(self):
        df = make_sample_df(50, add_dups=True)
        original_len = len(df)
        p = self._make_preprocessor(df)
        p.remove_duplicates()

        assert len(p.df) < original_len, "Duplicates should be removed"
        assert len(p.df) == 50

    def test_convert_dates(self):
        df = make_sample_df(50)
        p = self._make_preprocessor(df)
        p.convert_dates()

        assert pd.api.types.is_datetime64_any_dtype(p.df["Date"]), "Date column should be datetime"

    def test_convert_dates_sorted(self):
        df = make_sample_df(50)
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        p = self._make_preprocessor(df)
        p.convert_dates()

        dates = p.df["Date"].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1)), "Dates should be sorted"

    def test_treat_outliers_caps_values(self):
        df = make_sample_df(100)
        # Inject extreme outliers
        df.loc[0, "Sales"] = 999999.0
        df.loc[1, "CompetitionDistance"] = -50.0

        p = self._make_preprocessor(df)
        p.treat_outliers()

        # After IQR capping, Sales should not contain the extreme value
        assert p.df["Sales"].max() < 999999.0, "Outlier should be capped"

    def test_treat_outliers_no_data_loss(self):
        df = make_sample_df(100)
        p = self._make_preprocessor(df)
        original_len = len(p.df)
        p.treat_outliers()
        assert len(p.df) == original_len, "Outlier treatment should not drop rows"

    def test_encode_categoricals_removes_strings(self):
        df = make_sample_df(50)
        p = self._make_preprocessor(df)
        p.convert_dates()   # parse date so it's excluded from encoding
        p.encode_categoricals()

        remaining_obj = [
            c for c in p.df.columns
            if p.df[c].dtype == object and c != "Date"
        ]
        assert len(remaining_obj) == 0, f"All object columns should be encoded, remaining: {remaining_obj}"

    def test_pipeline_produces_valid_output(self, tmp_path):
        df = make_sample_df(100, add_nans=True, add_dups=True)

        # Write raw data to tmp
        raw_path = tmp_path / "sales_data.csv"
        df.to_csv(raw_path, index=False)

        processed_path = tmp_path / "processed_data.csv"

        params = {**TEST_PARAMS}
        params["data"] = {**TEST_PARAMS["data"]}
        params["data"]["raw_path"] = str(raw_path)
        params["data"]["processed_path"] = str(processed_path)

        p = Preprocessor(params=params)
        result = p.run_pipeline()

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "Sales" in result.columns
        assert result.isnull().sum().sum() == 0, "Processed data should have no NaNs"
