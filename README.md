# Demand Forecasting MLOps Pipeline

Production-ready end-to-end MLOps pipeline for e-commerce demand forecasting using Python, MLflow, FastAPI, Docker, and GitHub Actions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEMAND FORECASTING MLOPS PIPELINE                 │
└─────────────────────────────────────────────────────────────────────┘

  ┌───────────┐    ┌─────────────┐    ┌──────────────────┐
  │  Data     │    │   Feature   │    │  Model Training  │
  │ Ingestion │───▶│ Engineering │───▶│  (4 Models)      │
  │ (DVC)     │    │             │    │  MLflow Tracking │
  └───────────┘    └─────────────┘    └────────┬─────────┘
        │                                       │
        ▼                                       ▼
  ┌───────────┐                       ┌──────────────────┐
  │   Data    │                       │  MLflow Model    │
  │ Preproc   │                       │  Registry        │
  └───────────┘                       └────────┬─────────┘
                                               │
                                               ▼
  ┌──────────────────────────────────────────────────────┐
  │               FastAPI REST API (Port 8000)            │
  │  GET  /health          → Health check                 │
  │  GET  /model-info      → Model version & metrics      │
  │  POST /predict         → Demand forecast              │
  │  GET  /predictions/history → Last 50 predictions      │
  │  GET  /monitoring/report   → Evidently AI report      │
  └──────────────────────────────────────────────────────┘
        │                         │
        ▼                         ▼
  ┌───────────┐           ┌──────────────┐
  │  SQLite   │           │  Evidently   │
  │  Storage  │           │  Monitoring  │
  └───────────┘           └──────────────┘
        │
        ▼
  ┌───────────────────────────────────────┐
  │  Docker Compose                        │
  │  ├─ api      (FastAPI, port 8000)      │
  │  ├─ mlflow   (Tracking, port 5000)     │
  │  └─ grafana  (Dashboard, port 3000)    │
  └───────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────┐
  │  GitHub Actions CI/CD                  │
  │  ├─ Job 1: test  (pytest + coverage)   │
  │  ├─ Job 2: train (DVC pipeline)        │
  │  └─ Job 3: deploy (Docker Hub)         │
  └───────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Models | LinearRegression, RandomForest, XGBoost, Prophet |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |
| Storage | SQLite |
| Visualization | Plotly, Seaborn, Matplotlib |

## Project Structure

```
demand-forecasting-mlops/
├── data/
│   ├── raw/              # Raw ingested data (DVC tracked)
│   └── processed/        # Preprocessed + feature data
├── notebooks/
│   └── eda.ipynb         # Exploratory Data Analysis
├── src/
│   ├── data/
│   │   ├── ingest.py          # Data ingestion
│   │   └── preprocess.py      # Data preprocessing
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py           # Model training + MLflow
│   │   ├── evaluate.py        # Model evaluation + reports
│   │   └── predict.py         # Prediction utilities
│   ├── api/
│   │   └── main.py            # FastAPI application
│   └── monitoring/
│       └── monitor.py         # Evidently AI monitoring
├── tests/
│   ├── test_preprocess.py
│   ├── test_features.py
│   └── test_api.py
├── .github/workflows/
│   └── ci_cd.yml              # GitHub Actions pipeline
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                   # DVC pipeline stages
├── params.yaml                # Hyperparameters + config
└── requirements.txt
```

## Setup Instructions

### 1. Clone and Install

```bash
git clone https://github.com/your-username/demand-forecasting-mlops.git
cd demand-forecasting-mlops
python -m venv venv
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Step 1: Ingest data (generates synthetic data if Kaggle not configured)
python src/data/ingest.py

# Step 2: Preprocess
python src/data/preprocess.py

# Step 3: Feature engineering
python src/features/feature_engineering.py

# Step 4: Train models (logs to MLflow)
python src/models/train.py

# Step 5: Evaluate best model
python src/models/evaluate.py

# Step 6: Start the API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Or use DVC (recommended)

```bash
dvc repro
```

## Running the API Locally

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

## API Endpoint Reference

### GET /health
```bash
curl http://localhost:8000/health
```
Response:
```json
{"status": "ok", "model": "demand-forecast-model", "model_loaded": true}
```

### GET /model-info
```bash
curl http://localhost:8000/model-info
```

### POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "P001",
    "store_id": "S01",
    "date": "2024-06-15",
    "promotion": 1,
    "holiday": 0
  }'
```
Response:
```json
{
  "predicted_demand": 350,
  "unit": "units",
  "confidence_interval": [315, 385],
  "model": "XGBoost",
  "timestamp": "2024-06-15T10:30:00"
}
```

### GET /predictions/history
```bash
curl http://localhost:8000/predictions/history?limit=10
```

### GET /monitoring/report
```bash
# Opens HTML report in browser
curl http://localhost:8000/monitoring/report > report.html
```

## Running with Docker

```bash
# Build and start all services
docker-compose up --build

# Services:
#   API:     http://localhost:8000
#   MLflow:  http://localhost:5000
#   Grafana: http://localhost:3000 (admin/admin)
```

## Viewing the MLflow UI

```bash
# Start MLflow UI locally
mlflow ui --backend-store-uri mlruns/ --port 5000
# Open: http://localhost:5000
```

You can view:
- All experiment runs
- Hyperparameters logged per run
- Metrics: RMSE, MAE, R2, MAPE
- Model artifacts
- Model registry with Production stage

## Running Tests

```bash
# All tests with coverage
pytest tests/ --cov=src --cov-report=term-missing -v

# Individual test files
pytest tests/test_preprocess.py -v
pytest tests/test_features.py -v
pytest tests/test_api.py -v
```

## Monitoring

```bash
# Generate drift + performance report
python src/monitoring/monitor.py

# View report
open reports/monitoring_report.html
# OR via API:
curl http://localhost:8000/monitoring/report > monitoring.html
```

Auto-retraining is triggered when drift score > 0.3 (configurable in `params.yaml`).

## CI/CD (GitHub Actions)

Push to `main` triggers:
1. **test** — pytest with coverage upload to Codecov
2. **train** — full DVC pipeline (ingest → preprocess → features → train → evaluate)
3. **deploy** — Docker build + push to Docker Hub

Required GitHub Secrets:
- `DOCKER_USERNAME` — Docker Hub username
- `DOCKER_PASSWORD` — Docker Hub password/token
- `CODECOV_TOKEN` — (optional) Codecov token
- `RENDER_WEBHOOK_URL` — (optional) Render deploy hook

## Configuration

Edit `params.yaml` to tune hyperparameters:

```yaml
model:
  n_estimators: 200     # Trees in RandomForest/XGBoost
  max_depth: 6          # Max tree depth
  learning_rate: 0.05   # XGBoost learning rate
  random_state: 42
```

## Models Trained

| Model | Description |
|-------|-------------|
| LinearRegression | Baseline — fast, interpretable |
| RandomForestRegressor | Ensemble — handles nonlinearity |
| XGBRegressor | Gradient boosting — usually best |
| Prophet | Facebook's time-series model |

Best model is automatically promoted to **Production** in the MLflow Model Registry.
