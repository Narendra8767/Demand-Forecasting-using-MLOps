# Deployment Guide

Deploy the Demand Forecasting MLOps project:
- **API + Frontend → Render.com** (free hosting)
- **MLflow Tracking UI → DagsHub.com** (free hosted MLflow server)

---

## PART 1 — MLflow on DagsHub (Hosted MLflow Website)

DagsHub gives you a **free hosted MLflow server** at a real URL like:
`https://dagshub.com/YOUR_USERNAME/demand-forecasting-mlops.mlflow`

### Step 1 — Create DagsHub account
1. Go to **https://dagshub.com** → Sign Up (free)
2. Use your GitHub account to sign up (easier)

### Step 2 — Create a DagsHub repository
1. Click **"New Repository"**
2. Name it: `demand-forecasting-mlops`
3. Choose **"Connect a GitHub repository"** and link your GitHub repo
   - OR choose "Create new DagsHub repo"

### Step 3 — Get your MLflow tracking URI
After creating the repo, go to:
```
https://dagshub.com/YOUR_USERNAME/demand-forecasting-mlops
```
Click **"Remote"** tab → **"MLflow"** section.

Your tracking URI will be:
```
https://dagshub.com/YOUR_USERNAME/demand-forecasting-mlops.mlflow
```

### Step 4 — Get your DagsHub token
1. Go to **https://dagshub.com/user/settings/tokens**
2. Click **"Generate New Token"**
3. Name it: `mlflow-token`
4. Copy the token — you'll need it below

### Step 5 — Run training with DagsHub tracking (Windows PowerShell)
```powershell
$env:MLFLOW_TRACKING_URI     = "https://dagshub.com/YOUR_USERNAME/demand-forecasting-mlops.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "YOUR_DAGSHUB_USERNAME"
$env:MLFLOW_TRACKING_PASSWORD = "YOUR_DAGSHUB_TOKEN"
$env:PYTHONPATH = "."

python src/models/train.py
```

### Step 6 — View experiments on DagsHub
Open: `https://dagshub.com/YOUR_USERNAME/demand-forecasting-mlops.mlflow`

You will see:
- All 4 model runs (LinearRegression, RandomForest, XGBoost, Prophet)
- Hyperparameters logged per run
- Metrics: RMSE, MAE, R², MAPE
- Model artifacts
- Model Registry with Production stage

---

## PART 2 — Deploy API to Render.com (Free)

### Step 1 — Push project to GitHub
```powershell
cd "C:\Users\narendra tekale\MLOPS Project\demand-forecasting-mlops"
git init
git add .
git commit -m "Initial MLOps pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/demand-forecasting-mlops.git
git push -u origin main
```

### Step 2 — Create Render account
1. Go to **https://render.com** → Sign Up (free)
2. Connect your **GitHub** account

### Step 3 — Create a new Web Service on Render
1. Click **"New +"** → **"Web Service"**
2. Select your **demand-forecasting-mlops** GitHub repo
3. Fill in settings:

| Setting | Value |
|---------|-------|
| Name | `demand-forecast-api` |
| Region | Oregon (US West) |
| Branch | `main` |
| Runtime | `Python 3` |
| Build Command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start Command | `python start.py` |
| Plan | **Free** |

4. Click **"Advanced"** → Add Environment Variables:

| Key | Value |
|-----|-------|
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/YOUR_USERNAME/demand-forecasting-mlops.mlflow` |
| `DAGSHUB_USERNAME` | `YOUR_DAGSHUB_USERNAME` |
| `DAGSHUB_TOKEN` | `YOUR_DAGSHUB_TOKEN` |
| `PYTHONPATH` | `.` |

5. Click **"Create Web Service"**

### Step 4 — Wait for first deploy (~5–10 minutes)
Render will:
1. Install requirements
2. Run `start.py` → auto-trains the model (ingest → preprocess → features → train)
3. Start the FastAPI server

### Step 5 — Your live URLs
After deploy succeeds:

| URL | What it is |
|-----|-----------|
| `https://demand-forecast-api.onrender.com/` | **Frontend Dashboard** |
| `https://demand-forecast-api.onrender.com/docs` | Swagger API docs |
| `https://demand-forecast-api.onrender.com/health` | Health check |
| `https://demand-forecast-api.onrender.com/predict` | POST predict endpoint |

> ⚠️ Free tier sleeps after 15 min of inactivity. First request after sleep takes ~30s to wake up.

---

## PART 3 — CI/CD with GitHub Actions + DagsHub + Render

### Step 1 — Add GitHub Secrets
Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/YOUR_USERNAME/demand-forecasting-mlops.mlflow` |
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | Your DagsHub access token |
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | Your Docker Hub password/token |
| `RENDER_DEPLOY_HOOK` | (Get from Render → Service → Settings → Deploy Hook URL) |
| `CODECOV_TOKEN` | (Optional — from codecov.io) |

### Step 2 — Get Render Deploy Hook URL
1. Render Dashboard → your service → **"Settings"** tab
2. Scroll to **"Deploy Hook"**
3. Copy the URL → paste as `RENDER_DEPLOY_HOOK` GitHub secret

### Step 3 — Push to trigger CI/CD
```powershell
git add .
git commit -m "Add deployment config"
git push origin main
```

This triggers:
1. **Test job** — pytest 51 tests
2. **Train job** — runs full pipeline, logs to DagsHub MLflow
3. **Deploy job** — builds Docker image, pushes to Docker Hub, triggers Render redeploy

---

## PART 4 — Docker Deployment (VPS / Self-hosted)

If you have a VPS (DigitalOcean, AWS EC2, etc.):

```bash
# Clone on server
git clone https://github.com/YOUR_USERNAME/demand-forecasting-mlops.git
cd demand-forecasting-mlops

# Set env vars
export MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_USERNAME/demand-forecasting-mlops.mlflow
export DAGSHUB_USERNAME=YOUR_USERNAME
export DAGSHUB_TOKEN=YOUR_TOKEN

# Run with Docker Compose
docker-compose up --build -d

# Services:
#   http://your-server:8000   → API + Frontend
#   http://your-server:5000   → Local MLflow UI
#   http://your-server:3000   → Grafana
```

---

## Quick Reference

```powershell
# Local development
python -m uvicorn src.api.main:app --reload --port 8000

# Train with DagsHub tracking
$env:MLFLOW_TRACKING_URI="https://dagshub.com/USER/REPO.mlflow"
$env:MLFLOW_TRACKING_USERNAME="USER"
$env:MLFLOW_TRACKING_PASSWORD="TOKEN"
python src/models/train.py

# View local MLflow UI
mlflow ui --backend-store-uri mlruns/ --port 5000
# Open http://localhost:5000

# Run all tests
python -m pytest tests/ -v

# Docker
docker-compose up --build
```

---

## Architecture After Deployment

```
GitHub (push) ──► GitHub Actions CI/CD
                      │
               ┌──────┴──────────────────┐
               │                         │
          pytest tests              Train Models
               │                         │
               │                   Log to DagsHub ──► dagshub.com/.../mlflow
               │                         │
               └──────────┬──────────────┘
                          │
                   Build Docker Image
                          │
                   Push to Docker Hub
                          │
                   Trigger Render Deploy
                          │
              https://your-app.onrender.com
                    ┌─────┴──────┐
                 Frontend    FastAPI
                (HTML UI)   /predict
```
