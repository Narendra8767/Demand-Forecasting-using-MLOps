"""
Production startup script.
Runs the full ML pipeline if the model file is missing, then starts the API.
Used by Render / Railway / Docker as the entrypoint.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
MODEL_PKL = ROOT / "models" / "best_model.pkl"


def run(cmd: str):
    logger.info(f"▶  {cmd}")
    result = subprocess.run(cmd, shell=True, check=True)
    return result


def pipeline_needed() -> bool:
    """Return True if any required file is missing."""
    required = [
        ROOT / "data" / "raw"       / "sales_data.csv",
        ROOT / "data" / "processed" / "processed_data.csv",
        ROOT / "data" / "processed" / "features.csv",
        MODEL_PKL,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.info(f"Missing files detected: {missing}")
    return bool(missing)


def run_pipeline():
    logger.info("=" * 55)
    logger.info("  Running ML pipeline (first-time setup)…")
    logger.info("=" * 55)
    steps = [
        "python src/data/ingest.py",
        "python src/data/preprocess.py",
        "python src/features/feature_engineering.py",
        "python src/models/train.py",
    ]
    for step in steps:
        run(step)
    logger.info("Pipeline complete. Model ready.")


def start_api():
    port = os.environ.get("PORT", "8000")
    workers = os.environ.get("WEB_CONCURRENCY", "2")
    logger.info(f"Starting API on port {port} with {workers} worker(s)…")
    os.execvp("python", [
        "python", "-m", "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", port,
        "--workers", workers,
    ])


if __name__ == "__main__":
    if pipeline_needed():
        run_pipeline()
    else:
        logger.info("Model already trained. Skipping pipeline.")
    start_api()
