"""FastAPI service — serves forecasting, risk estimation and the financing
recommendation engine, and exposes the SQL analytics layer.
"""
from __future__ import annotations
import os
from datetime import date, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import database as db
from .engine import full_recommendation
from .forecaster import forecast, MODEL_VERSION
from .report import render_report

app = FastAPI(title="Supply Chain Finance Advisor API", version="2.0.0")


class RecommendRequest(BaseModel):
    product: str
    production_cost: float = Field(gt=0)
    retail_price: float = Field(gt=0)


@app.on_event("startup")
def startup():
    db.init_db()


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION,
            "db": os.path.basename(db.current_db_path())}


@app.get("/products")
def products():
    return db.list_products()


@app.post("/forecast/{product_name}")
def make_forecast(product_name: str, horizon: int = 30, backtest: bool = True):
    """Backtest mode (default): forecast the LAST `horizon` historical days,
    so every prediction has a real actual to be scored against -> the
    analytics tables (MAE/VaR by region) work immediately. backtest=false
    logs true future-dated forecasts (unscorable until actuals arrive)."""
    history = db.demand_series(product_name)
    if len(history) < 35:
        raise HTTPException(404, f"insufficient history for '{product_name}' ({len(history)} rows)")
    if backtest:
        train, actuals = history[:-horizon], history[-horizon:]
    else:
        train, actuals = history, []
    preds = forecast(train, horizon)
    pid = next((p["id"] for p in db.list_products() if p["name"] == product_name), None)
    if pid is not None:
        target_dates = db.demand_dates(product_name)[-len(preds):] if backtest else None
        made_on = date.today().isoformat()
        for i, y in enumerate(preds):
            fd = target_dates[i] if target_dates else (date.today() + timedelta(days=i)).isoformat()
            db.log_forecast(pid, fd, made_on, y, MODEL_VERSION)
    out = {"product": product_name, "model_version": MODEL_VERSION,
           "horizon": horizon, "backtest": backtest,
           "forecast": [round(p, 2) for p in preds]}
    if backtest:
        out["actuals"] = [round(a, 2) for a in actuals]
        out["mae"] = round(sum(abs(p - a) for p, a in zip(preds, actuals)) / len(actuals), 2)
    return out


@app.post("/recommend")
def recommend(req: RecommendRequest):
    history = db.demand_series(req.product)
    if not history:
        raise HTTPException(404, f"no demand history for '{req.product}'")
    errors = db.forecast_errors(req.product)
    result = full_recommendation(req.production_cost, req.retail_price, history, errors)
    pid = next((p["id"] for p in db.list_products() if p["name"] == req.product), None)
    db.log_recommendation(pid, result["cost_ratio"], result["demand_cv"],
                          result["var_95"], result["risk_level"],
                          result["recommended_method"], MODEL_VERSION)
    return result


# ---------- analytics (SQL layer) ----------

@app.get("/analytics/mae-by-region")
def analytics_mae():
    return db.mae_by_region()


@app.get("/analytics/var-by-region")
def analytics_var():
    return db.var_by_region()


@app.get("/analytics/riskiest-products")
def analytics_risk():
    return db.riskiest_products()


@app.get("/analytics/recommendation-summary")
def analytics_rec_summary():
    return db.recommendation_summary()

@app.post("/report", response_class=HTMLResponse)
def report(req: RecommendRequest):
    """Full recommendation rendered as the downloadable HTML report."""
    history = db.demand_series(req.product)
    if not history:
        raise HTTPException(404, f"no demand history for '{req.product}'")
    errors = db.forecast_errors(req.product)
    result = full_recommendation(req.production_cost, req.retail_price, history, errors)
    return render_report(result)
