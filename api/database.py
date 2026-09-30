"""Database access layer — raw SQL via sqlite3 (stdlib), no ORM.

Every query an analyst would want lives here as a named function, so the
SQL is real, reviewable, and unit-tested.
"""
from __future__ import annotations
import os
import sqlite3
from contextlib import contextmanager

DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "data", "scf.db")


def current_db_path() -> str:
    """Resolved at call time so tests/containers can set SCF_DB late."""
    return os.environ.get("SCF_DB", DEFAULT_DB)
SCHEMA = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")


def init_db(path: str | None = None) -> None:
    path = path or current_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(SCHEMA) as f:
        sql = f.read()
    with sqlite3.connect(path) as con:
        con.executescript(sql)


@contextmanager
def connect(path: str | None = None):
    path = path or current_db_path()
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ---------- writes ----------

def upsert_region(name: str) -> int:
    with connect() as con:
        con.execute("INSERT OR IGNORE INTO regions(name) VALUES (?)", (name,))
        return con.execute("SELECT id FROM regions WHERE name=?", (name,)).fetchone()["id"]


def upsert_product(name: str, region: str, unit_cost: float, retail_price: float) -> int:
    rid = upsert_region(region)
    with connect() as con:
        con.execute(
            """INSERT INTO products(name, region_id, unit_cost, retail_price)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET region_id=excluded.region_id,
                   unit_cost=excluded.unit_cost, retail_price=excluded.retail_price""",
            (name, rid, unit_cost, retail_price),
        )
        return con.execute("SELECT id FROM products WHERE name=?", (name,)).fetchone()["id"]


def insert_demand(product_id: int, date: str, units: float) -> None:
    with connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO demand_history(product_id, date, units) VALUES (?, ?, ?)",
            (product_id, date, units),
        )


def log_forecast(product_id: int, forecast_date: str, made_on: str, y_pred: float, model_version: str) -> None:
    with connect() as con:
        con.execute(
            """INSERT INTO forecasts(product_id, forecast_date, made_on, y_pred, model_version)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(product_id, forecast_date)
               DO UPDATE SET made_on=excluded.made_on, y_pred=excluded.y_pred,
                             model_version=excluded.model_version""",
            (product_id, forecast_date, made_on, y_pred, model_version),
        )


def log_recommendation(product_id, cost_ratio, demand_cv, var95, risk_level, recommendation, model_version) -> None:
    from datetime import datetime, timezone
    with connect() as con:
        con.execute(
            """INSERT INTO predictions_log
               (created_at, product_id, cost_ratio, demand_cv, var_95, risk_level, recommendation, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), product_id, cost_ratio,
             demand_cv, var95, risk_level, recommendation, model_version),
        )


# ---------- analytical queries (the "SQL layer" deliverable) ----------

def mae_by_region() -> list[dict]:
    """Mean absolute forecast error per region — is forecasting equally good everywhere?"""
    with connect() as con:
        return [dict(r) for r in con.execute("""
            SELECT rg.name AS region,
                   COUNT(*)            AS n,
                   ROUND(AVG(ABS(f.y_pred - d.units)), 2) AS mae,
                   ROUND(AVG(d.units), 2)                 AS avg_demand
            FROM forecasts f
            JOIN demand_history d
              ON d.product_id = f.product_id AND d.date = f.forecast_date
            JOIN products p ON p.id = f.product_id
            JOIN regions rg ON rg.id = p.region_id
            GROUP BY rg.name
            ORDER BY mae DESC
        """)]


def var_by_region() -> list[dict]:
    """95th percentile of absolute errors per region (nearest-rank, in SQL)."""
    with connect() as con:
        return [dict(r) for r in con.execute("""
            WITH errs AS (
                SELECT rg.name AS region, ABS(f.y_pred - d.units) AS abs_err,
                       COUNT(*) OVER (PARTITION BY rg.name) AS n
                FROM forecasts f
                JOIN demand_history d
                  ON d.product_id = f.product_id AND d.date = f.forecast_date
                JOIN products p ON p.id = f.product_id
                JOIN regions rg ON rg.id = p.region_id
            ),
            ranked AS (
                SELECT region, abs_err,
                       ROW_NUMBER() OVER (PARTITION BY region ORDER BY abs_err) AS rn
                FROM errs
            )
            SELECT region, MAX(abs_err) AS var_95
            FROM ranked
            WHERE rn = CAST(CEIL(0.95 * (SELECT n FROM errs LIMIT 1)) AS INTEGER)
               OR rn = (SELECT MAX(rn) FROM ranked r2 WHERE r2.region = ranked.region)
            GROUP BY region
            ORDER BY var_95 DESC
        """)]


def riskiest_products() -> list[dict]:
    """Products ranked by demand volatility (coefficient of variation, computed in SQL)."""
    with connect() as con:
        return [dict(r) for r in con.execute("""
            SELECT p.name AS product, rg.name AS region,
                   ROUND(AVG(d.units), 2)  AS avg_demand,
                   ROUND(AVG(d.units*d.units) - AVG(d.units)*AVG(d.units), 2) AS variance,
                   ROUND(SQRT(AVG(d.units*d.units) - AVG(d.units)*AVG(d.units)) / AVG(d.units), 4) AS cv
            FROM demand_history d
            JOIN products p ON p.id = d.product_id
            JOIN regions rg ON rg.id = p.region_id
            GROUP BY p.id
            ORDER BY cv DESC
        """)]


def recommendation_summary() -> list[dict]:
    """What has the system recommended, to whom, at what risk level?"""
    with connect() as con:
        return [dict(r) for r in con.execute("""
            SELECT p.name AS product, rg.name AS region,
                   ROUND(AVG(l.cost_ratio), 3) AS avg_cost_ratio,
                   ROUND(AVG(l.demand_cv), 4)  AS avg_cv,
                   COUNT(*) AS n,
                   GROUP_CONCAT(DISTINCT l.recommendation) AS methods
            FROM predictions_log l
            LEFT JOIN products p ON p.id = l.product_id
            LEFT JOIN regions rg ON rg.id = p.region_id
            GROUP BY p.id
            ORDER BY n DESC
        """)]


def demand_series(product_name: str) -> list[float]:
    with connect() as con:
        rows = con.execute(
            """SELECT d.units FROM demand_history d
               JOIN products p ON p.id = d.product_id
               WHERE p.name = ? ORDER BY d.date""",
            (product_name,),
        ).fetchall()
        return [r["units"] for r in rows]


def demand_dates(product_name: str) -> list[str]:
    with connect() as con:
        rows = con.execute(
            """SELECT d.date FROM demand_history d
               JOIN products p ON p.id = d.product_id
               WHERE p.name = ? ORDER BY d.date""",
            (product_name,),
        ).fetchall()
        return [r["date"] for r in rows]


def forecast_errors(product_name: str) -> list[float]:
    """Signed forecast errors for scored (backtested) forecasts — feeds VaR in
    the recommendation engine, matching notebook 02's abs-error percentile."""
    with connect() as con:
        rows = con.execute("""
            SELECT f.y_pred - d.units AS err
            FROM forecasts f
            JOIN demand_history d
              ON d.product_id = f.product_id AND d.date = f.forecast_date
            JOIN products p ON p.id = f.product_id
            WHERE p.name = ?
        """, (product_name,)).fetchall()
        return [r["err"] for r in rows]


def list_products() -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute("""
            SELECT p.id, p.name, rg.name AS region, p.unit_cost, p.retail_price,
                   COUNT(d.id) AS n_obs
            FROM products p
            JOIN regions rg ON rg.id = p.region_id
            LEFT JOIN demand_history d ON d.product_id = p.id
            GROUP BY p.id ORDER BY p.id
        """)]
