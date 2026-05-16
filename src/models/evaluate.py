"""
Module 6 — Model Evaluation
Generates evaluation reports, comparison tables, and plots.
"""

import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def load_best_model():
    path = PROJECT_ROOT / "models" / "best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Best model not found at {path}. Run train.py first.")
    return joblib.load(path)


def load_test_data(params: dict):
    """Load feature matrix and return test split."""
    features_path = PROJECT_ROOT / params["data"]["features_path"]
    date_col = params["data"]["date_column"]
    target_col = params["data"]["target_column"]
    test_size = params["data"]["test_size"]

    df = pd.read_csv(features_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    exclude = {date_col, target_col}
    feature_cols = [c for c in df.columns if c not in exclude]

    split = int(len(df) * (1 - test_size))
    X_test = df.iloc[split:][feature_cols]
    y_test = df.iloc[split:][target_col]
    dates_test = df.iloc[split:][date_col]

    return X_test, y_test, dates_test


def compute_all_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAPE(%)": round(mape(y_true, y_pred), 2),
    }


def plot_actual_vs_predicted(y_true, y_pred, dates, model_name: str) -> go.Figure:
    """Interactive Plotly chart — actual vs predicted demand."""
    # Sample for performance (max 2000 points)
    if len(y_true) > 2000:
        idx = np.linspace(0, len(y_true) - 1, 2000, dtype=int)
        dates_plot = dates.iloc[idx] if hasattr(dates, "iloc") else dates[idx]
        y_true_plot = np.array(y_true)[idx]
        y_pred_plot = np.array(y_pred)[idx]
    else:
        dates_plot, y_true_plot, y_pred_plot = dates, y_true, y_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_plot, y=y_true_plot,
        mode="lines", name="Actual",
        line=dict(color="#2196F3", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=dates_plot, y=y_pred_plot,
        mode="lines", name="Predicted",
        line=dict(color="#F44336", width=1.5, dash="dash")
    ))
    fig.update_layout(
        title=f"Actual vs Predicted Demand — {model_name}",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_residuals(y_true, y_pred, model_name: str) -> go.Figure:
    """Residuals scatter plot."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Residuals vs Fitted", "Residual Distribution"])

    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode="markers",
                   marker=dict(color="#9C27B0", opacity=0.4, size=4),
                   name="Residuals"),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=50,
                     marker_color="#4CAF50", opacity=0.7,
                     name="Distribution"),
        row=1, col=2
    )
    fig.update_layout(
        title=f"Residual Analysis — {model_name}",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def generate_html_report(model_name: str, metrics: dict, fig_avp: go.Figure, fig_res: go.Figure) -> str:
    """Build a full HTML evaluation report."""
    metrics_rows = "".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
        for k, v in metrics.items()
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Demand Forecast — Evaluation Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
  h1 {{ color: #1976D2; }} h2 {{ color: #424242; }}
  table {{ border-collapse: collapse; width: 400px; background: white; }}
  th, td {{ padding: 10px 16px; border: 1px solid #ddd; text-align: left; }}
  th {{ background: #1976D2; color: white; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,.1); }}
</style>
</head>
<body>
<h1>Demand Forecasting — Evaluation Report</h1>
<div class="card">
  <h2>Best Model: {model_name}</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {metrics_rows}
  </table>
</div>
<div class="card">
  <h2>Actual vs Predicted</h2>
  {fig_avp.to_html(full_html=False, include_plotlyjs='cdn')}
</div>
<div class="card">
  <h2>Residual Analysis</h2>
  {fig_res.to_html(full_html=False, include_plotlyjs=False)}
</div>
</body>
</html>"""
    return html


def evaluate():
    params = load_params()
    model_bundle = load_best_model()
    model = model_bundle["model"]
    model_name = model_bundle["name"]
    logger.info(f"Evaluating model: {model_name}")

    X_test, y_test, dates_test = load_test_data(params)

    # Prophet predictions are not row-wise — skip if loaded
    if model_name == "Prophet":
        logger.warning("Prophet evaluation via pkl not supported in row-wise mode. Using stored metrics.")
        metrics = model_bundle.get("metrics", {})
        preds = np.full(len(y_test), y_test.mean())
    else:
        preds = model.predict(X_test)
        metrics = compute_all_metrics(y_test, preds)

    logger.info("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    fig_avp = plot_actual_vs_predicted(y_test.values, preds, dates_test, model_name)
    fig_res = plot_residuals(y_test.values, preds, model_name)

    report_html = generate_html_report(model_name, metrics, fig_avp, fig_res)
    report_path = REPORTS_DIR / "evaluation_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    logger.info(f"Evaluation report saved to {report_path}")

    return metrics, preds


if __name__ == "__main__":
    metrics, preds = evaluate()
    print("\nEvaluation complete:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
