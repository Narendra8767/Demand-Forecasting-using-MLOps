"""
Module 5 — Model Training + MLflow Experiment Tracking (Module 7)
Trains LinearRegression, RandomForest, XGBoost, and Prophet with MLflow logging.
"""

import sys
import os
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape_val = mape(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape_val}


def time_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Chronological train/test split (no shuffling)."""
    n = len(X)
    split = int(n * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ------------------------------------------------------------------
# Individual model trainers
# ------------------------------------------------------------------

def train_linear_regression(X_train, y_train, X_test, y_test, params: dict) -> dict:
    logger.info("Training LinearRegression (baseline)...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)
    return {"model": model, "name": "LinearRegression", "metrics": metrics, "preds": preds}


def train_random_forest(X_train, y_train, X_test, y_test, params: dict) -> dict:
    logger.info("Training RandomForestRegressor...")
    mp = params["model"]
    model = RandomForestRegressor(
        n_estimators=mp["n_estimators"],
        max_depth=mp["max_depth"],
        random_state=mp["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)
    return {"model": model, "name": "RandomForest", "metrics": metrics, "preds": preds}


def train_xgboost(X_train, y_train, X_test, y_test, params: dict) -> dict:
    logger.info("Training XGBRegressor...")
    mp = params["model"]
    model = XGBRegressor(
        n_estimators=mp["n_estimators"],
        max_depth=mp["max_depth"],
        learning_rate=mp["learning_rate"],
        random_state=mp["random_state"],
        tree_method="hist",
        eval_metric="rmse",
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)
    return {"model": model, "name": "XGBoost", "metrics": metrics, "preds": preds}


def train_prophet(df_train: pd.DataFrame, df_test: pd.DataFrame, date_col: str, target_col: str, params: dict) -> dict:
    """Train Prophet on aggregated daily sales (store-level aggregation)."""
    try:
        from prophet import Prophet

        logger.info("Training Prophet (time-series)...")

        # Prophet requires 'ds' and 'y' columns
        prophet_train = (
            df_train[[date_col, target_col]]
            .groupby(date_col)[target_col]
            .sum()
            .reset_index()
            .rename(columns={date_col: "ds", target_col: "y"})
        )
        prophet_test = (
            df_test[[date_col, target_col]]
            .groupby(date_col)[target_col]
            .sum()
            .reset_index()
            .rename(columns={date_col: "ds", target_col: "y"})
        )

        model = Prophet(
            changepoint_prior_scale=params["model"].get("prophet_changepoint_prior_scale", 0.05),
            seasonality_mode="multiplicative",
        )
        model.fit(prophet_train)

        future = model.make_future_dataframe(periods=len(prophet_test))
        forecast = model.predict(future)
        test_forecast = forecast.tail(len(prophet_test))["yhat"].values

        y_true = prophet_test["y"].values
        metrics = compute_metrics(y_true, test_forecast)
        return {"model": model, "name": "Prophet", "metrics": metrics, "preds": test_forecast}

    except Exception as e:
        logger.warning(f"Prophet training failed: {e}. Skipping.")
        return None


# ------------------------------------------------------------------
# MLflow experiment runner
# ------------------------------------------------------------------

def log_model_to_mlflow(result: dict, X_train, X_test, y_test, mlflow_params: dict, model_params: dict):
    """Log a model run to MLflow."""
    with mlflow.start_run(run_name=result["name"]) as run:
        # Log hyperparameters
        mlflow.log_params(model_params)
        mlflow.log_param("model_type", result["name"])

        # Log metrics
        mlflow.log_metrics(result["metrics"])

        # Log model artifact
        model = result["model"]
        if result["name"] == "Prophet":
            # Prophet uses its own serialization
            import tempfile, os
            tmp_dir = tempfile.gettempdir()
            prophet_pkl = os.path.join(tmp_dir, "prophet_model.pkl")
            joblib.dump(model, prophet_pkl)
            mlflow.log_artifact(prophet_pkl, "model")
        elif result["name"] in ["LinearRegression", "RandomForest"]:
            mlflow.sklearn.log_model(model, "model")
        elif result["name"] == "XGBoost":
            mlflow.xgboost.log_model(model, "model")

        run_id = run.info.run_id
        logger.info(
            f"[{result['name']}] RMSE={result['metrics']['rmse']:.2f} | "
            f"MAE={result['metrics']['mae']:.2f} | "
            f"R2={result['metrics']['r2']:.4f} | "
            f"MAPE={result['metrics']['mape']:.2f}%"
        )
        return run_id


def promote_best_model(results: list, mlflow_cfg: dict):
    """Register the best model (lowest RMSE) to MLflow Model Registry in Production stage."""
    best = min(results, key=lambda r: r["metrics"]["rmse"])
    model_name = mlflow_cfg["model_name"]
    experiment_name = mlflow_cfg["experiment_name"]

    logger.info(f"\nBest model: {best['name']} (RMSE={best['metrics']['rmse']:.2f})")

    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass  # Already exists

    # Get run_id for best model from experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.model_type = '{best['name']}'",
            order_by=["metrics.rmse ASC"],
            max_results=1,
        )
        if runs:
            run_id = runs[0].info.run_id
            model_uri = f"runs:/{run_id}/model"
            try:
                mv = client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    run_id=run_id,
                )
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Production",
                )
                logger.info(f"Model '{model_name}' v{mv.version} promoted to Production")
            except Exception as e:
                logger.warning(f"Model registry promotion failed: {e}")

    return best


def save_best_model(best_result: dict):
    """Save the best model as a pkl for direct loading."""
    path = MODELS_DIR / "best_model.pkl"
    joblib.dump(
        {"model": best_result["model"], "name": best_result["name"], "metrics": best_result["metrics"]},
        path,
    )
    logger.info(f"Best model saved to {path}")
    return str(path)


# ------------------------------------------------------------------
# Main training orchestrator
# ------------------------------------------------------------------

def train():
    params = load_params()
    mlflow_cfg = params["mlflow"]
    model_params = params["model"]
    data_params = params["data"]

    # Setup MLflow — support remote (DagsHub/hosted) or local filesystem
    # Priority: MLFLOW_TRACKING_URI env var > params.yaml > local mlruns/
    remote_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if remote_uri:
        # Remote tracking (DagsHub, hosted MLflow server, etc.)
        mlflow.set_tracking_uri(remote_uri)
        logger.info(f"MLflow remote tracking: {remote_uri}")
    else:
        # Local filesystem — use file:/// URI (required on Windows)
        tracking_path = PROJECT_ROOT / mlflow_cfg["tracking_uri"]
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = tracking_path.as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow local tracking: {tracking_uri}")

    # Set DagsHub credentials if provided
    dagshub_token = os.environ.get("DAGSHUB_TOKEN", "")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("DAGSHUB_USERNAME", "")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    # Enable autologging
    mlflow.sklearn.autolog(log_model_signatures=False, silent=True)
    mlflow.xgboost.autolog(log_model_signatures=False, silent=True)

    # Load feature matrix
    features_path = PROJECT_ROOT / data_params["features_path"]
    if not features_path.exists():
        logger.info("Feature file not found — running feature engineering...")
        from src.features.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(params)
        X_full, y_full = fe.build_features()
        df_full = pd.read_csv(features_path)
    else:
        df_full = pd.read_csv(features_path)

    date_col = data_params["date_column"]
    target_col = data_params["target_column"]
    test_size = data_params["test_size"]

    df_full[date_col] = pd.to_datetime(df_full[date_col])
    df_full.sort_values(date_col, inplace=True)
    df_full.reset_index(drop=True, inplace=True)

    # Feature/target split
    exclude = {date_col, target_col}
    feature_cols = [c for c in df_full.columns if c not in exclude]
    X = df_full[feature_cols]
    y = df_full[target_col]

    # Time-based train/test split
    X_train, X_test, y_train, y_test = time_split(X, y, test_size)
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Save split indices for Prophet
    split_idx = int(len(df_full) * (1 - test_size))
    df_train = df_full.iloc[:split_idx]
    df_test = df_full.iloc[split_idx:]

    results = []
    run_ids = []

    # Train all models
    for trainer, kwargs in [
        (train_linear_regression, {}),
        (train_random_forest, {}),
        (train_xgboost, {}),
    ]:
        result = trainer(X_train, y_train, X_test, y_test, params)
        results.append(result)
        rid = log_model_to_mlflow(result, X_train, X_test, y_test, mlflow_cfg, model_params)
        run_ids.append(rid)

    # Prophet
    prophet_result = train_prophet(df_train, df_test, date_col, target_col, params)
    if prophet_result:
        results.append(prophet_result)
        rid = log_model_to_mlflow(prophet_result, X_train, X_test, y_test, mlflow_cfg, model_params)
        run_ids.append(rid)

    # Print comparison table
    logger.info("\n=== Model Comparison ===")
    comparison = pd.DataFrame([
        {
            "Model": r["name"],
            "RMSE": round(r["metrics"]["rmse"], 2),
            "MAE": round(r["metrics"]["mae"], 2),
            "R2": round(r["metrics"]["r2"], 4),
            "MAPE(%)": round(r["metrics"]["mape"], 2),
        }
        for r in results
    ])
    comparison.sort_values("RMSE", inplace=True)
    print(comparison.to_string(index=False))

    # Promote best model to registry
    # Only register sklearn/xgboost results (Prophet has different artifacts)
    registry_results = [r for r in results if r["name"] != "Prophet"]
    best = promote_best_model(registry_results if registry_results else results, mlflow_cfg)

    # Save best model locally
    save_best_model(best)

    # Save comparison table
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    comparison.to_csv(PROJECT_ROOT / "reports" / "model_comparison.csv", index=False)
    logger.info("Model comparison saved to reports/model_comparison.csv")

    return best, results


if __name__ == "__main__":
    best, results = train()
    print(f"\nTraining complete. Best model: {best['name']}")
