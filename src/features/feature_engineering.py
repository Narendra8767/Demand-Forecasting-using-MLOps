"""
Module 4 — Feature Engineering
Creates temporal, lag, rolling, and categorical features for demand forecasting.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


class FeatureEngineer:
    """
    Creates a rich feature set for demand forecasting from preprocessed sales data.
    """

    def __init__(self, params: dict = None):
        self.params = params or load_params()
        self.date_col = self.params["data"]["date_column"]
        self.target_col = self.params["data"]["target_column"]
        self.processed_path = PROJECT_ROOT / self.params["data"]["processed_path"]
        self.features_path = PROJECT_ROOT / self.params["data"]["features_path"]
        self.df: pd.DataFrame = None

    # ------------------------------------------------------------------
    def load(self) -> "FeatureEngineer":
        logger.info(f"Loading processed data from {self.processed_path}")
        self.df = pd.read_csv(self.processed_path, low_memory=False)
        # Ensure date column is datetime
        if self.date_col in self.df.columns:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")
            self.df.sort_values(self.date_col, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
        logger.info(f"Loaded {len(self.df):,} rows")
        return self

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------
    def add_temporal_features(self) -> "FeatureEngineer":
        logger.info("Adding temporal features...")
        d = self.df[self.date_col]
        self.df["day_of_week"] = d.dt.dayofweek          # 0=Mon, 6=Sun
        self.df["day_of_month"] = d.dt.day
        self.df["month"] = d.dt.month
        self.df["quarter"] = d.dt.quarter
        self.df["week_of_year"] = d.dt.isocalendar().week.astype(int)
        self.df["year"] = d.dt.year
        self.df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
        self.df["is_month_start"] = d.dt.is_month_start.astype(int)
        self.df["is_month_end"] = d.dt.is_month_end.astype(int)
        # Cyclical encoding to preserve periodicity
        self.df["month_sin"] = np.sin(2 * np.pi * self.df["month"] / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * self.df["month"] / 12)
        self.df["dow_sin"] = np.sin(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["dow_cos"] = np.cos(2 * np.pi * self.df["day_of_week"] / 7)
        return self

    # ------------------------------------------------------------------
    # Lag features — grouped by store + product to avoid data leakage
    # ------------------------------------------------------------------
    def add_lag_features(self) -> "FeatureEngineer":
        logger.info("Adding lag features (lag_7, lag_14, lag_28)...")
        group_cols = self._group_cols()

        for lag in [7, 14, 28]:
            col_name = f"lag_{lag}"
            if group_cols:
                self.df[col_name] = (
                    self.df.groupby(group_cols)[self.target_col]
                    .shift(lag)
                )
            else:
                self.df[col_name] = self.df[self.target_col].shift(lag)

        return self

    # ------------------------------------------------------------------
    # Rolling averages and std
    # ------------------------------------------------------------------
    def add_rolling_features(self) -> "FeatureEngineer":
        logger.info("Adding rolling mean/std features...")
        group_cols = self._group_cols()

        for window in [7, 14]:
            mean_col = f"rolling_mean_{window}"
            if group_cols:
                self.df[mean_col] = (
                    self.df.groupby(group_cols)[self.target_col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
            else:
                self.df[mean_col] = (
                    self.df[self.target_col].shift(1).rolling(window, min_periods=1).mean()
                )

        # Rolling std (7-day)
        std_col = "rolling_std_7"
        if group_cols:
            self.df[std_col] = (
                self.df.groupby(group_cols)[self.target_col]
                .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std().fillna(0))
            )
        else:
            self.df[std_col] = (
                self.df[self.target_col].shift(1).rolling(7, min_periods=1).std().fillna(0)
            )

        return self

    # ------------------------------------------------------------------
    # Boolean flags
    # ------------------------------------------------------------------
    def add_flag_features(self) -> "FeatureEngineer":
        logger.info("Adding flag features (holiday, promotion)...")

        # Holiday flag
        if "StateHoliday" in self.df.columns:
            self.df["is_holiday"] = (self.df["StateHoliday"] != 0).astype(int)
        elif "IsHoliday" in self.df.columns:
            self.df["is_holiday"] = self.df["IsHoliday"].astype(int)
        else:
            self.df["is_holiday"] = 0

        # Promotion flag
        if "Promo" in self.df.columns:
            self.df["is_promotion"] = self.df["Promo"].astype(int)
        else:
            self.df["is_promotion"] = 0

        return self

    # ------------------------------------------------------------------
    # Store/product encoding (if string IDs remain)
    # ------------------------------------------------------------------
    def add_entity_encoding(self) -> "FeatureEngineer":
        for col in ["Store", "Product"]:
            if col in self.df.columns:
                if self.df[col].dtype == object:
                    self.df[f"{col}_enc"] = self.df[col].astype("category").cat.codes
                    logger.info(f"Encoded '{col}' → '{col}_enc'")
                elif self.df[col].dtype in [np.int64, np.float64]:
                    self.df[f"{col}_enc"] = self.df[col]
        return self

    # ------------------------------------------------------------------
    # Build final feature matrix
    # ------------------------------------------------------------------
    def build_features(self) -> tuple[pd.DataFrame, pd.Series]:
        """Returns (X, y) ready for model training."""
        self.load()
        self.add_temporal_features()
        self.add_lag_features()
        self.add_rolling_features()
        self.add_flag_features()
        self.add_entity_encoding()

        # Drop rows with NaN from lag features
        before = len(self.df)
        self.df.dropna(subset=[f"lag_{l}" for l in [7, 14, 28]], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        logger.info(f"Dropped {before - len(self.df)} rows after lag NaN removal. Remaining: {len(self.df):,}")

        # Define feature columns — exclude date and target
        exclude = {self.date_col, self.target_col, "Store", "Product"}
        # Also exclude one-hot encoded store/product originals
        feature_cols = [
            c for c in self.df.columns
            if c not in exclude
            and not c.startswith("StateHoliday_")   # may be one-hot
            and self.df[c].dtype in [np.int64, np.float64, np.int32, np.float32, int, float]
        ]

        logger.info(f"Final feature columns ({len(feature_cols)}): {feature_cols}")

        X = self.df[feature_cols].copy()
        y = self.df[self.target_col].copy()

        # Save features to disk
        self.features_path.parent.mkdir(parents=True, exist_ok=True)
        full = self.df[[self.date_col] + feature_cols + [self.target_col]].copy()
        full.to_csv(self.features_path, index=False)
        logger.info(f"Feature matrix saved to {self.features_path}")

        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    # ------------------------------------------------------------------
    def _group_cols(self) -> list:
        """Determine grouping columns for lag/rolling calculations."""
        groups = []
        if "Store" in self.df.columns:
            groups.append("Store")
        if "Product" in self.df.columns:
            groups.append("Product")
        return groups


if __name__ == "__main__":
    fe = FeatureEngineer()
    X, y = fe.build_features()
    print(f"\nFeature Engineering done.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nFeature columns:\n{list(X.columns)}")
    print(f"\nSample features:\n{X.head(3)}")
