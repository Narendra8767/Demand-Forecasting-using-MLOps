"""
Module 1 — Data Ingestion
Downloads Rossmann Store Sales dataset or generates synthetic fallback data.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def generate_synthetic_data(n_stores: int = 10, n_products: int = 50, days: int = 730) -> pd.DataFrame:
    """
    Generate realistic synthetic e-commerce sales data as a fallback.
    Creates ~730 days of daily sales across stores and products.
    """
    logger.info("Generating synthetic sales dataset...")

    np.random.seed(42)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    stores = [f"S{str(i).zfill(2)}" for i in range(1, n_stores + 1)]
    products = [f"P{str(i).zfill(3)}" for i in range(1, n_products + 1)]

    records = []
    for store in stores:
        store_factor = np.random.uniform(0.7, 1.5)
        for product in products:
            product_factor = np.random.uniform(0.5, 2.0)
            base_sales = np.random.randint(50, 300)

            for date in dates:
                day_of_week = date.weekday()
                month = date.month
                is_weekend = int(day_of_week >= 5)

                # Seasonal pattern — higher in Nov/Dec
                seasonal = 1.0 + 0.4 * np.sin(2 * np.pi * (month - 3) / 12)
                # Weekend bump
                weekend_bump = 1.2 if is_weekend else 1.0
                # Trend: slight upward over time
                day_idx = (date - start_date).days
                trend = 1.0 + 0.0003 * day_idx

                # Promotion: random ~20% of days
                promotion = int(np.random.random() < 0.2)
                promo_bump = 1.35 if promotion else 1.0

                # Holiday flag (simplified: Christmas week + New Year + summer peak)
                is_holiday = int(
                    (month == 12 and date.day >= 20)
                    or (month == 1 and date.day <= 3)
                    or (month == 7 and 10 <= date.day <= 20)
                )
                holiday_bump = 1.5 if is_holiday else 1.0

                # Noise
                noise = np.random.normal(1.0, 0.1)

                sales = max(
                    0,
                    int(
                        base_sales
                        * store_factor
                        * product_factor
                        * seasonal
                        * weekend_bump
                        * trend
                        * promo_bump
                        * holiday_bump
                        * noise
                    ),
                )

                records.append(
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        "Store": store,
                        "Product": product,
                        "Sales": sales,
                        "Customers": max(1, int(sales / np.random.uniform(2, 5))),
                        "Promo": promotion,
                        "StateHoliday": "a" if is_holiday else "0",
                        "SchoolHoliday": int(np.random.random() < 0.15),
                        "StoreType": np.random.choice(["a", "b", "c", "d"]),
                        "Assortment": np.random.choice(["a", "b", "c"]),
                        "CompetitionDistance": np.random.randint(100, 10000),
                    }
                )

    df = pd.DataFrame(records)
    logger.info(f"Synthetic dataset created: {len(df):,} rows, {df['Store'].nunique()} stores, {df['Product'].nunique()} products")
    return df


def try_kaggle_download() -> bool:
    """Attempt to download Rossmann dataset from Kaggle. Returns True on success."""
    try:
        import kaggle  # noqa: F401

        logger.info("Kaggle API found. Attempting download...")
        os.makedirs(RAW_DIR, exist_ok=True)
        os.system(
            f'kaggle competitions download -c rossmann-store-sales -p "{RAW_DIR}" --unzip'
        )

        expected = RAW_DIR / "train.csv"
        if expected.exists():
            logger.info(f"Rossmann dataset downloaded to {RAW_DIR}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Kaggle download skipped: {e}")
        return False


def ingest():
    """Main ingestion entry point."""
    os.makedirs(RAW_DIR, exist_ok=True)

    output_path = RAW_DIR / "sales_data.csv"

    if output_path.exists():
        logger.info(f"Raw data already exists at {output_path}. Skipping ingestion.")
        return str(output_path)

    # Try Kaggle first, fall back to synthetic
    kaggle_ok = try_kaggle_download()

    if kaggle_ok:
        train_path = RAW_DIR / "train.csv"
        df = pd.read_csv(train_path, low_memory=False)
        df = df.rename(columns={"Store": "Store", "Sales": "Sales", "Date": "Date"})
        # Add Product column (not in Rossmann — use Store as proxy)
        df["Product"] = "P001"
        df.to_csv(output_path, index=False)
        logger.info(f"Rossmann data saved to {output_path}")
    else:
        logger.info("Using synthetic data generator as fallback.")
        df = generate_synthetic_data(n_stores=10, n_products=20, days=730)
        df.to_csv(output_path, index=False)
        logger.info(f"Synthetic data saved to {output_path} ({len(df):,} rows)")

    # Print summary
    logger.info("\n=== Data Summary ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Total Sales: {df['Sales'].sum():,}")

    return str(output_path)


def init_dvc():
    """Initialize DVC tracking for the data directory."""
    try:
        ret = os.system(f'cd "{PROJECT_ROOT}" && dvc init --no-scm 2>/dev/null || dvc init 2>/dev/null')
        if ret == 0:
            os.system(f'cd "{PROJECT_ROOT}" && dvc add data/raw/sales_data.csv 2>/dev/null')
            logger.info("DVC tracking initialized for data/raw/")
        else:
            logger.warning("DVC init skipped (may already be initialized or git not available).")
    except Exception as e:
        logger.warning(f"DVC init skipped: {e}")


if __name__ == "__main__":
    path = ingest()
    init_dvc()
    print(f"\nData ingestion complete. Raw data at: {path}")
