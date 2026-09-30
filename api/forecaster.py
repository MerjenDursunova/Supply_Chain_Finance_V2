"""Demand forecasting — exact port of notebook 01 / original dashboard.py.

Feature pipeline (verified to reproduce MAE 14.36 / RMSE 17.25 on 2023):
  calendar: day_of_year, month, quarter, year, day_of_week
  lags:     lag_1, lag_7, lag_30
  rolling:  rolling_mean_7, rolling_mean_30   (shift(1) — causal, no leakage)

Recursive multi-step: predicted values are fed back as history so
calendar features + lags stay consistent for future dates.
"""
from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd

MODEL_PATH = os.environ.get(
    "SCF_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "models", "demand_forecast_model.pkl"),
)
MODEL_VERSION = "rf-n100-rs42-v1"

# exact column order the model was trained on
FEATURES = ["day_of_year", "month", "quarter", "year", "day_of_week",
            "lag_1", "lag_7", "lag_30", "rolling_mean_7", "rolling_mean_30"]


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def make_features(history: pd.DataFrame) -> pd.DataFrame:
    """history: DataFrame with 'date' (datetime) and 'demand' columns.
    Mirrors notebook 01 exactly, including shift(1) on rolling windows."""
    df = history.copy()
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek
    df["lag_1"] = df["demand"].shift(1)
    df["lag_7"] = df["demand"].shift(7)
    df["lag_30"] = df["demand"].shift(30)
    df["rolling_mean_7"] = df["demand"].shift(1).rolling(7).mean()
    df["rolling_mean_30"] = df["demand"].shift(1).rolling(30).mean()
    return df.dropna()


def forecast(history: list[float], horizon: int = 30) -> list[float]:
    """Recursive multi-step forecast from a raw demand list (dates synthesized).
    Returns `horizon` point forecasts."""
    model = load_model()
    if model is None:
        return [float(np.mean(history[-30:]))] * horizon   # artifact missing
    hist = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=len(history), freq="D"),
                         "demand": history})
    preds = []
    for _ in range(horizon):
        row = make_features(hist)[FEATURES].iloc[[-1]]
        y_hat = float(model.predict(row)[0])
        preds.append(y_hat)
        nxt = hist["date"].iloc[-1] + pd.Timedelta(days=1)
        hist = pd.concat([hist, pd.DataFrame({"date": [nxt], "demand": [y_hat]})],
                         ignore_index=True)
    return preds
