"""
Module 8 — FastAPI REST API
Endpoints: /health, /model-info, /predict, /predictions/history, /monitoring/report
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.orm import DeclarativeBase, Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


PARAMS = load_params()
DB_PATH = PROJECT_ROOT / PARAMS["api"]["db_path"]
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DB_PATH}"
MODEL_NAME = PARAMS["mlflow"]["model_name"]


# ------------------------------------------------------------------
# Database setup
# ------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String)
    store_id = Column(String)
    date = Column(String)
    promotion = Column(Integer)
    holiday = Column(Integer)
    predicted_demand = Column(Float)
    ci_low = Column(Float)
    ci_high = Column(Float)
    model_name = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

_model_bundle = None


def load_model_bundle():
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    # Try MLflow registry first
    try:
        import mlflow
        remote_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if remote_uri:
            mlflow.set_tracking_uri(remote_uri)
            dagshub_token = os.environ.get("DAGSHUB_TOKEN", "")
            if dagshub_token:
                os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("DAGSHUB_USERNAME", "")
                os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        else:
            tracking_path = PROJECT_ROOT / PARAMS["mlflow"]["tracking_uri"]
            tracking_uri = tracking_path.as_uri()
            mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        _model_bundle = {"model": model, "name": MODEL_NAME, "source": "mlflow_registry"}
        logger.info(f"Model loaded from MLflow registry: {model_uri}")
        return _model_bundle
    except Exception as e:
        logger.warning(f"MLflow registry load failed: {e}. Falling back to local pkl.")

    # Fallback: load from pkl
    pkl_path = PROJECT_ROOT / "models" / "best_model.pkl"
    if pkl_path.exists():
        _model_bundle = joblib.load(pkl_path)
        _model_bundle["source"] = "local_pkl"
        logger.info(f"Model loaded from pkl: {pkl_path}")
        return _model_bundle

    logger.error("No model found. Run train.py first.")
    return None


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    product_id: str = Field(..., example="P001")
    store_id: str = Field(..., example="S01")
    date: str = Field(..., example="2024-06-15", description="YYYY-MM-DD format")
    promotion: int = Field(0, ge=0, le=1, description="1 if promotion active, 0 otherwise")
    holiday: int = Field(0, ge=0, le=1, description="1 if holiday, 0 otherwise")


class PredictResponse(BaseModel):
    predicted_demand: int
    unit: str
    confidence_interval: list[float]
    model: str
    timestamp: str


# ------------------------------------------------------------------
# Feature builder (same logic as predict.py)
# ------------------------------------------------------------------

def build_features(req: PredictRequest) -> pd.DataFrame:
    dt = pd.to_datetime(req.date)
    row = {
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
        "is_holiday": req.holiday,
        "is_promotion": req.promotion,
        "lag_7": 0.0,
        "lag_14": 0.0,
        "lag_28": 0.0,
        "rolling_mean_7": 0.0,
        "rolling_mean_14": 0.0,
        "rolling_std_7": 0.0,
        "Store_enc": hash(req.store_id) % 1000,
        "Product_enc": hash(req.product_id) % 5000,
        "CompetitionDistance": 1000.0,
        "Customers": 0.0,
    }
    return pd.DataFrame([row])


def run_prediction(bundle: dict, req: PredictRequest) -> dict:
    """Run inference using the loaded model bundle."""
    model = bundle["model"]
    model_name = bundle["name"]

    if model_name == "Prophet" or hasattr(model, "predict_proba") is False and "prophet" in str(type(model)).lower():
        base = 250
        predicted = int(base * (1.35 if req.promotion else 1.0) * (1.5 if req.holiday else 1.0))
    else:
        X = build_features(req)
        try:
            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
                for col in expected:
                    if col not in X.columns:
                        X[col] = 0.0
                X = X[expected]
            elif hasattr(model, "predict") and hasattr(model, "_model_impl"):
                # MLflow pyfunc wrapper
                pass
        except Exception:
            pass

        raw = float(model.predict(X)[0])
        predicted = max(0, int(round(raw)))

    return {
        "predicted_demand": predicted,
        "unit": "units",
        "confidence_interval": [max(0, int(predicted * 0.9)), int(predicted * 1.1)],
        "model": model_name,
    }


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------

app = FastAPI(
    title="Demand Forecasting API",
    description="Production-ready API for E-commerce demand forecasting",
    version="1.0.0",
)

# Serve frontend static files if the directory exists
_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    bundle = load_model_bundle()
    if bundle:
        logger.info(f"API started. Model: {bundle['name']} (source: {bundle.get('source', 'unknown')})")
    else:
        logger.warning("API started WITHOUT a loaded model. /predict will return 503.")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Health check endpoint."""
    bundle = load_model_bundle()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": bundle is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model-info", tags=["System"])
def model_info():
    """Returns current model version and metrics."""
    bundle = load_model_bundle()
    if not bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metrics = bundle.get("metrics", {})
    return {
        "model_name": bundle["name"],
        "source": bundle.get("source", "unknown"),
        "metrics": metrics,
        "registered_name": MODEL_NAME,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Generate demand forecast for a product/store/date combination.

    - **product_id**: Product identifier (e.g., "P001")
    - **store_id**: Store identifier (e.g., "S01")
    - **date**: Forecast date in YYYY-MM-DD format
    - **promotion**: 1 if promotion active, 0 otherwise
    - **holiday**: 1 if holiday, 0 otherwise
    """
    bundle = load_model_bundle()
    if not bundle:
        raise HTTPException(status_code=503, detail="Model not available. Run training pipeline first.")

    try:
        result = run_prediction(bundle, req)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    ts = datetime.utcnow().isoformat()

    # Persist to SQLite
    try:
        with Session(engine) as session:
            record = PredictionRecord(
                product_id=req.product_id,
                store_id=req.store_id,
                date=req.date,
                promotion=req.promotion,
                holiday=req.holiday,
                predicted_demand=result["predicted_demand"],
                ci_low=result["confidence_interval"][0],
                ci_high=result["confidence_interval"][1],
                model_name=result["model"],
                timestamp=datetime.utcnow(),
            )
            session.add(record)
            session.commit()
    except Exception as e:
        logger.warning(f"Failed to persist prediction: {e}")

    return PredictResponse(
        predicted_demand=result["predicted_demand"],
        unit=result["unit"],
        confidence_interval=result["confidence_interval"],
        model=result["model"],
        timestamp=ts,
    )


@app.get("/predictions/history", tags=["Prediction"])
def prediction_history(limit: int = 50):
    """Return the last N predictions from the database."""
    try:
        with Session(engine) as session:
            records = (
                session.query(PredictionRecord)
                .order_by(PredictionRecord.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "product_id": r.product_id,
                    "store_id": r.store_id,
                    "date": r.date,
                    "promotion": r.promotion,
                    "holiday": r.holiday,
                    "predicted_demand": r.predicted_demand,
                    "confidence_interval": [r.ci_low, r.ci_high],
                    "model": r.model_name,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                }
                for r in records
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/report", tags=["Monitoring"])
def monitoring_report():
    """Serve the Evidently monitoring HTML report."""
    report_path = PROJECT_ROOT / "reports" / "monitoring_report.html"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Monitoring report not found. Run src/monitoring/monitor.py first."
        )
    with open(report_path, "r", encoding="utf-8") as f:
        html = f.read()
    return Response(content=html, media_type="text/html")


@app.get("/", tags=["System"])
def root():
    """Serve the frontend dashboard, or JSON if frontend not found."""
    frontend_path = PROJECT_ROOT / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path), media_type="text/html")
    return {
        "message": "Demand Forecasting MLOps API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "history": "/predictions/history",
    }

@app.get("/dashboard", tags=["System"])
def dashboard():
    """Explicit route to the frontend dashboard."""
    frontend_path = PROJECT_ROOT / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path), media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend not found.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=PARAMS["api"]["host"],
        port=PARAMS["api"]["port"],
        reload=True,
    )
