"""
Module 11 — Monitoring with Evidently AI
Generates data drift and model performance reports.
Triggers retraining if drift score exceeds threshold.
"""

import sys
import logging
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def load_reference_data(params: dict) -> pd.DataFrame:
    """Load training portion of feature data as reference."""
    features_path = PROJECT_ROOT / params["data"]["features_path"]
    date_col = params["data"]["date_column"]
    target_col = params["data"]["target_column"]
    test_size = params["data"]["test_size"]

    df = pd.read_csv(features_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    split = int(len(df) * (1 - test_size))
    return df.iloc[:split].copy()


def load_current_data(params: dict) -> pd.DataFrame:
    """Load test portion of feature data as current (new) data."""
    features_path = PROJECT_ROOT / params["data"]["features_path"]
    date_col = params["data"]["date_column"]
    test_size = params["data"]["test_size"]

    df = pd.read_csv(features_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    split = int(len(df) * (1 - test_size))
    return df.iloc[split:].copy()


def run_evidently_drift_report(reference: pd.DataFrame, current: pd.DataFrame, params: dict) -> dict:
    """
    Run Evidently data drift report.
    Returns dict with drift detected flag and report HTML path.
    """
    try:
        # Evidently 0.7+ uses different import paths
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset, DataQualityPreset
        except ImportError:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        date_col = params["data"]["date_column"]
        target_col = params["data"]["target_column"]

        # Select numeric feature columns only (drop date and target)
        exclude = {date_col, target_col}
        num_cols = [
            c for c in reference.columns
            if c not in exclude and reference[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ][:20]  # cap at 20 features for performance

        ref = reference[num_cols].fillna(0).reset_index(drop=True)
        cur = current[num_cols].fillna(0).reset_index(drop=True)

        # Sample if too large
        max_rows = 5000
        if len(ref) > max_rows:
            ref = ref.sample(max_rows, random_state=42)
        if len(cur) > max_rows:
            cur = cur.sample(max_rows, random_state=42)

        report = Report([DataDriftPreset()])
        # Evidently 0.7 — run() returns a Snapshot object
        snapshot = report.run(reference_data=ref, current_data=cur)

        report_path = REPORTS_DIR / "monitoring_report.html"
        snapshot.save_html(str(report_path))
        logger.info(f"Evidently report saved to {report_path}")

        # Extract drift score from snapshot dict (Evidently 0.7 structure)
        drift_detected = False
        drift_score = 0.0
        try:
            snap_dict = snapshot.dict() if hasattr(snapshot, "dict") else {}
            for metric_result in snap_dict.get("metrics", []):
                value = metric_result.get("value", {})
                # DriftedColumnsCount gives a "share" of drifted columns
                if isinstance(value, dict) and "share" in value:
                    drift_score = float(value["share"])
                    drift_detected = drift_score > params["monitoring"]["drift_threshold"]
                    break
        except Exception:
            pass

        return {
            "report_path": str(report_path),
            "drift_detected": drift_detected,
            "drift_score": drift_score,
        }

    except (ImportError, Exception) as e:
        logger.warning(f"Evidently report failed ({e}). Using fallback.")
        return _fallback_monitoring_report(reference, current, params)


def run_model_performance_report(reference: pd.DataFrame, current: pd.DataFrame, params: dict) -> dict:
    """Track model performance metrics over time."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import RegressionPreset

        target_col = params["data"]["target_column"]
        date_col = params["data"]["date_column"]

        bundle = joblib.load(PROJECT_ROOT / "models" / "best_model.pkl")
        model = bundle["model"]
        model_name = bundle["name"]

        exclude = {date_col, target_col}
        feature_cols = [c for c in reference.columns if c not in exclude]

        ref_copy = reference.copy()
        cur_copy = current.copy()

        if model_name != "Prophet":
            ref_X = ref_copy[feature_cols].fillna(0)
            cur_X = cur_copy[feature_cols].fillna(0)

            try:
                if hasattr(model, "feature_names_in_"):
                    expected = list(model.feature_names_in_)
                    for col in expected:
                        if col not in ref_X.columns:
                            ref_X[col] = 0.0
                        if col not in cur_X.columns:
                            cur_X[col] = 0.0
                    ref_X = ref_X[expected]
                    cur_X = cur_X[expected]
            except Exception:
                pass

            ref_copy["prediction"] = model.predict(ref_X[:1000])
            cur_copy["prediction"] = model.predict(cur_X[:1000])

            ref_perf = ref_copy[[target_col, "prediction"]].rename(columns={target_col: "target"}).head(1000)
            cur_perf = cur_copy[[target_col, "prediction"]].rename(columns={target_col: "target"}).head(1000)

            try:
                report = Report(metrics=[RegressionPreset()])
                report.run(reference_data=ref_perf, current_data=cur_perf)
                perf_path = REPORTS_DIR / "performance_report.html"
                report.save_html(str(perf_path))
                logger.info(f"Performance report saved to {perf_path}")
            except Exception as e:
                logger.warning(f"Evidently performance report failed: {e}")

    except Exception as e:
        logger.warning(f"Model performance report skipped: {e}")


def _fallback_monitoring_report(reference: pd.DataFrame, current: pd.DataFrame, params: dict) -> dict:
    """Generate a simple HTML monitoring report without Evidently."""
    target_col = params["data"]["target_column"]

    ref_stats = reference[target_col].describe() if target_col in reference.columns else pd.Series()
    cur_stats = current[target_col].describe() if target_col in current.columns else pd.Series()

    # Compute simple drift metric: normalized mean shift
    if len(ref_stats) > 0 and len(cur_stats) > 0:
        ref_mean = ref_stats.get("mean", 0)
        cur_mean = cur_stats.get("mean", 0)
        drift_score = abs(cur_mean - ref_mean) / (ref_mean + 1e-9)
    else:
        drift_score = 0.0

    drift_detected = drift_score > params["monitoring"]["drift_threshold"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Monitoring Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,.1); }}
  h1 {{ color: #1976D2; }} h2 {{ color: #424242; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
  th {{ background: #1976D2; color: white; }}
  .alert {{ background: #FFEBEE; border-left: 4px solid #F44336; padding: 12px; }}
  .ok {{ background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 12px; }}
</style>
</head>
<body>
<h1>Demand Forecasting — Monitoring Report</h1>
<p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>

<div class="card">
  <h2>Data Drift Summary</h2>
  <div class="{'alert' if drift_detected else 'ok'}">
    <b>Drift Detected: {'YES ⚠️' if drift_detected else 'NO ✅'}</b><br>
    Drift Score: {drift_score:.4f} (threshold: {params['monitoring']['drift_threshold']})
  </div>
</div>

<div class="card">
  <h2>Target Distribution Comparison ({target_col})</h2>
  <table>
    <tr><th>Statistic</th><th>Reference (Train)</th><th>Current (Test)</th></tr>
    {''.join(f"<tr><td>{k}</td><td>{ref_stats.get(k, 'N/A'):.2f}</td><td>{cur_stats.get(k, 'N/A'):.2f}</td></tr>" for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max'])}
  </table>
</div>

<div class="card">
  <h2>Reference Data Info</h2>
  <p>Rows: {len(reference):,} | Columns: {len(reference.columns)}</p>
  <h2>Current Data Info</h2>
  <p>Rows: {len(current):,} | Columns: {len(current.columns)}</p>
</div>
</body>
</html>"""

    report_path = REPORTS_DIR / "monitoring_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return {
        "report_path": str(report_path),
        "drift_detected": drift_detected,
        "drift_score": drift_score,
    }


def trigger_retraining():
    """Trigger model retraining pipeline."""
    logger.info("Drift threshold exceeded — triggering model retraining...")
    try:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "models" / "train.py")],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            logger.info("Retraining completed successfully.")
        else:
            logger.error(f"Retraining failed:\n{result.stderr}")
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")


def monitor():
    """Main monitoring entry point."""
    params = load_params()

    features_path = PROJECT_ROOT / params["data"]["features_path"]
    if not features_path.exists():
        logger.error("Feature data not found. Run the training pipeline first.")
        return

    reference = load_reference_data(params)
    current = load_current_data(params)

    logger.info(f"Reference data: {len(reference):,} rows")
    logger.info(f"Current data: {len(current):,} rows")

    # Run drift report
    drift_result = run_evidently_drift_report(reference, current, params)
    logger.info(f"Drift score: {drift_result['drift_score']:.4f}")
    logger.info(f"Drift detected: {drift_result['drift_detected']}")

    # Run performance report
    run_model_performance_report(reference, current, params)

    # Auto-retrain if drift exceeds threshold
    threshold = params["monitoring"]["drift_threshold"]
    if drift_result["drift_score"] > threshold:
        logger.warning(f"Drift score {drift_result['drift_score']:.4f} > threshold {threshold}")
        trigger_retraining()
    else:
        logger.info(f"Drift within acceptable range ({drift_result['drift_score']:.4f} ≤ {threshold})")

    return drift_result


if __name__ == "__main__":
    result = monitor()
    if result:
        print(f"\nMonitoring complete.")
        print(f"Drift score: {result['drift_score']:.4f}")
        print(f"Report: {result['report_path']}")
