"""Seed the database with the real project data:
  - data/historical_demand.csv  (2021–2023 daily demand, from notebook 01)
  - data/forecast_results.csv   (logged 2023 predictions -> forecasts table,
                                 so MAE/VaR analytics work out of the box)

Usage: python db/seed.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.database import init_db, upsert_product, insert_demand, log_forecast  # noqa: E402

BASE = os.path.dirname(__file__)
FORECAST_MODEL_VERSION = "rf-n100-rs42-v1"


def run():
    db_path = os.environ.get("SCF_DB", os.path.join(BASE, "..", "data", "scf.db"))
    init_db(db_path)

    # one real product standing in for the manufacturer's line
    pid = upsert_product("main-line", "EMEA", unit_cost=8.0, retail_price=20.0)

    hist = pd.read_csv(os.path.join(BASE, "..", "data", "historical_demand.csv"))
    for _, r in hist.iterrows():
        insert_demand(pid, r["date"], float(r["demand"]))
    print(f"demand_history: {len(hist)} rows (2021-01-01 .. {hist['date'].iloc[-1]})")

    fc = pd.read_csv(os.path.join(BASE, "..", "data", "forecast_results.csv"))
    for _, r in fc.iterrows():
        log_forecast(pid, r["date"], made_on="2023-12-31",
                     y_pred=float(r["predicted_demand"]),
                     model_version=FORECAST_MODEL_VERSION)
    mae = (fc["demand"] - fc["predicted_demand"]).abs().mean()
    print(f"forecasts:      {len(fc)} logged 2023 predictions (MAE {mae:.2f})")
    print(f"database ready at {os.path.abspath(db_path)}")


if __name__ == "__main__":
    run()
