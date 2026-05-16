"""
Module 2 — Data Preprocessing
Cleans, transforms, and validates raw sales data.
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
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


class Preprocessor:
    """End-to-end preprocessing pipeline for raw sales data."""

    def __init__(self, params: dict = None):
        self.params = params or load_params()
        self.df: pd.DataFrame = None
        self.raw_path = PROJECT_ROOT / self.params["data"]["raw_path"]
        self.processed_path = PROJECT_ROOT / self.params["data"]["processed_path"]
        self.date_col = self.params["data"]["date_column"]
        self.target_col = self.params["data"]["target_column"]

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------
    def load_data(self) -> "Preprocessor":
        logger.info(f"Loading raw data from {self.raw_path}")
        self.df = pd.read_csv(self.raw_path, low_memory=False)
        logger.info(f"Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")
        return self

    # ------------------------------------------------------------------
    # Step 2 — Handle missing values
    # ------------------------------------------------------------------
    def handle_missing_values(self) -> "Preprocessor":
        before = self.df.isnull().sum().sum()
        logger.info(f"Missing values before: {before}")

        # Numeric columns — fill with median
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())

        # Categorical / object columns — fill with mode or 'Unknown'
        cat_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            if col == self.date_col:
                continue
            if self.df[col].isnull().any():
                mode_val = self.df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
                self.df[col] = self.df[col].fillna(fill_val)

        after = self.df.isnull().sum().sum()
        logger.info(f"Missing values after: {after}")
        return self

    # ------------------------------------------------------------------
    # Step 3 — Remove duplicates
    # ------------------------------------------------------------------
    def remove_duplicates(self) -> "Preprocessor":
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        logger.info(f"Removed {removed} duplicate rows. Remaining: {len(self.df):,}")
        return self

    # ------------------------------------------------------------------
    # Step 4 — Convert dates
    # ------------------------------------------------------------------
    def convert_dates(self) -> "Preprocessor":
        if self.date_col in self.df.columns:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")
            # Drop rows where date parsing failed
            bad = self.df[self.date_col].isnull().sum()
            if bad > 0:
                logger.warning(f"Dropping {bad} rows with unparseable dates")
                self.df.dropna(subset=[self.date_col], inplace=True)
            self.df.sort_values(self.date_col, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            logger.info(f"Date range: {self.df[self.date_col].min()} → {self.df[self.date_col].max()}")
        else:
            logger.warning(f"Date column '{self.date_col}' not found in dataset")
        return self

    # ------------------------------------------------------------------
    # Step 5 — Treat outliers (IQR capping)
    # ------------------------------------------------------------------
    def treat_outliers(self) -> "Preprocessor":
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Only cap the target and related numeric columns — not binary flags
        skip_cols = {"Promo", "SchoolHoliday", "Open"}
        cap_cols = [c for c in num_cols if c not in skip_cols]

        clipped = 0
        for col in cap_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            before = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            self.df[col] = self.df[col].clip(lower, upper)
            clipped += before

        logger.info(f"Outlier capping applied. Total values capped: {clipped}")
        return self

    # ------------------------------------------------------------------
    # Step 6 — Encode categoricals
    # ------------------------------------------------------------------
    def encode_categoricals(self) -> "Preprocessor":
        cat_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        # Exclude date column
        cat_cols = [c for c in cat_cols if c != self.date_col]

        for col in cat_cols:
            n_unique = self.df[col].nunique()
            if n_unique <= 10:
                # One-hot encode low cardinality
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=False, dtype=int)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(columns=[col], inplace=True)
                logger.info(f"One-hot encoded '{col}' ({n_unique} categories)")
            else:
                # Label encode high cardinality
                self.df[col] = self.df[col].astype("category").cat.codes
                logger.info(f"Label encoded '{col}' ({n_unique} categories)")

        return self

    # ------------------------------------------------------------------
    # Step 7 — Save processed data
    # ------------------------------------------------------------------
    def save_processed(self) -> "Preprocessor":
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.processed_path, index=False)
        logger.info(f"Processed data saved to {self.processed_path} ({len(self.df):,} rows)")
        return self

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------
    def run_pipeline(self) -> pd.DataFrame:
        logger.info("=== Starting Preprocessing Pipeline ===")
        (
            self
            .load_data()
            .handle_missing_values()
            .remove_duplicates()
            .convert_dates()
            .treat_outliers()
            .encode_categoricals()
            .save_processed()
        )
        logger.info("=== Preprocessing Complete ===")
        logger.info(f"Final shape: {self.df.shape}")
        return self.df


if __name__ == "__main__":
    preprocessor = Preprocessor()
    df = preprocessor.run_pipeline()
    print(f"\nPreprocessing done. Shape: {df.shape}")
    print(df.dtypes)
    print(df.head(3))
