# Supply Chain Finance Advisor — Technical Report

*How the system works, field by field. Written for the curious: visitors
who ask "but what actually happens when you click the button?", and for
future-you re-learning your own project.*

---

## 1. The mental model: two programs talking

When you run the project you start **two separate processes**:

| Process | Port | Role |
|---|---|---|
| `uvicorn api.main:app` | 8000 | the **server** (brain): model, decision logic, database |
| `streamlit run app/dashboard.py` | 8501 | the **client** (face): renders UI, sends requests |

The dashboard contains **no machine-learning code**. Clicking a button
sends an **HTTP request** (a formatted text message) to the server; the
server replies with **JSON** (structured text); the dashboard draws it.
This is *client-server architecture*, and it is how real ML systems are
deployed: one brain, many possible faces (dashboard today, another team's
script tomorrow).

The server, in turn, talks to two local resources:

- **`models/demand_forecast_model.pkl`** — the trained RandomForest,
  saved to disk ("model artifact"). Loading it rehydrates the model.
- **`data/scf.db`** — a **SQLite** database. SQLite is not a separate
  program; the database *is a file*, and **SQL** is the language used to
  ask it questions.

## 2. The data: what is stored where

| Table | Contents | Why it exists |
|---|---|---|
| `products` | one row per product (name, region, unit cost, retail price) | catalogue + costs used in recommendations |
| `demand_history` | one row per product per day: date, units sold | the raw material for forecasting and risk stats |
| `forecasts` | one row per prediction: product, target date, prediction, when it was made, model version | **every prediction is logged** → auditable ML + analytics |
| `predictions_log` | one row per recommendation: costs, CV, VaR, risk, method, timestamp | **every decision is logged** → decision-support audit trail |

Two design choices worth internalizing:

1. **Prediction logging.** The analytics endpoints cost almost nothing to
   build *because* predictions are logged. The database already contains
   everything needed to compute MAE and VaR — SQL just aggregates it.
2. **Idempotent logging.** `forecasts` has a UNIQUE index on
   `(product_id, forecast_date)` and writes use upsert
   (`ON CONFLICT DO UPDATE`). Re-running a forecast replaces rows instead
   of duplicating them. Without this, repeated calls silently inflate the
   analytics (this bug was found during development: the table grew from
   365 to 395 rows on a second call).

## 3. The life of a click

### 3.1 "Run forecast"

1. Dashboard sends `POST /forecast/main-line?horizon=22`.
2. `api/main.py:make_forecast` checks history exists (≥35 days).
3. In **backtest mode** (default), it holds out the last 22 days:
   `train = history[:-22]`, `actuals = history[-22:]`.
4. `api/forecaster.py:forecast` loads the pkl and predicts **recursively**:
   the model needs yesterday's demand (`lag_1`), last week's (`lag_7`),
   etc. Day 1 of the future has no yesterday, so the model's own
   prediction is appended to the history and used as input for day 2, and
   so on. This is why long-horizon lines flatten toward the recent mean —
   predictions feeding predictions regress to the average.
5. Each prediction is **logged** to `forecasts` (upsert).
6. Response JSON: `forecast` (22 numbers), `actuals`, and `mae` —
   the mean absolute error against the held-out days. A forecast you can
   score immediately.

### 3.2 "Get recommendation"

`POST /recommend` with `{product, production_cost, retail_price}` runs the
pipeline in `api/engine.py` (pure functions — no I/O, fully unit-tested):

| Step | Function | What it computes |
|---|---|---|
| 1 | — | mean, std of the product's demand history |
| 2 | — | **CV** = std / mean (volatility relative to level) |
| 3 | `classify_risk` | CV < 0.10 → Low Risk; < 0.20 → Medium Risk; else High Risk |
| 4 | `calculate_risk_adjusted_demand` | Low → mean + 0.5σ; Medium → mean; High → mean − 0.5σ (higher risk ⇒ more conservative demand estimate) |
| 5 | `var_95` | 95th percentile of **logged forecast errors** (from the DB), i.e. "on a bad day we're off by this much" |
| 6 | — | **cost ratio** = production cost / retail price |
| 7 | `recommend_method` | thresholds 0.40 / 0.70, shifted **up** by risk modifier (Low +0.00, Medium +0.05, High +0.10) |
| 8 | — | ratio ≤ low → Early Payment; ≤ high → In-House Factoring; else Bank Financing |

Example (real response, cost 8 / price 20):

```json
{
  "mean_demand": 125.33, "std_demand": 20.19, "demand_cv": 0.1611,
  "risk_level": "Medium Risk",
  "risk_adjusted_demand": 125.33,
  "var_95": 31.64, "var_source": "forecast_errors",
  "cost_ratio": 0.40,
  "low_threshold": 0.45, "high_threshold": 0.75,
  "recommended_method": "Early Payment",
  "rationale": "Production cost is low relative to retail price
                (cost ratio: 0.40). ..."
}
```

Read the decision out loud: 0.40 is *just under* the Medium-risk low
threshold of 0.45 (0.40 + 0.05), so Early Payment. Change cost to 10
(ratio 0.50) and it falls in [0.45, 0.75) → In-House Factoring. At 16
(ratio 0.80) → Bank Financing. These three scenarios are exactly the
supplier table in the reference paper.

The recommendation is then **logged** to `predictions_log`.

### 3.3 The analytics tabs

Each tab calls one `GET /analytics/...` endpoint, which runs one SQL
query from `api/database.py`. Example — MAE by region:

```sql
SELECT rg.name AS region,
       COUNT(*)                          AS n,
       ROUND(AVG(ABS(f.y_pred - d.units)), 2) AS mae,
       ROUND(AVG(d.units), 2)            AS avg_demand
FROM forecasts f
JOIN demand_history d
  ON d.product_id = f.product_id AND d.date = f.forecast_date
JOIN products p ON p.id = f.product_id
JOIN regions rg ON rg.id = p.region_id
GROUP BY rg.name
ORDER BY mae DESC
```

- **JOIN** pairs each prediction with the actual demand on its target date
  (predictions without matching actuals are excluded — future-dated
  forecasts wait for reality).
- **GROUP BY** collapses rows per region.
- The VaR endpoint computes a **nearest-rank 95th percentile in pure SQL**
  using `ROW_NUMBER() OVER (PARTITION BY region ORDER BY abs_err)` and
  selecting the row at `CEIL(0.95 * n)` — the same statistic as
  `np.percentile(..., 95)` with linear interpolation, by a different route.

## 4. A lesson hiding in the numbers

Open the dashboard and compare:

- Analytics tab: `avg_demand = 142.95` (over 2023 only — those are the 365
  logged forecasts).
- Recommendation: `mean_demand = 125.33` (over the **full 2021–2023**
  history).

Neither is wrong. They are **different windows on the same data**, and
they disagree because the series trends upward (100 → 150), so later
years average higher. The same window difference moves the CV (0.103 on
2023 vs 0.161 on all three years) — both still land in "Medium Risk", so
the verdict is robust, but **always know which data fed a number**. If an
interviewer asks why the dashboard's CV doesn't match notebook 02's
10.28%, this is the answer.

## 5. Reproducibility checklist

```bash
pip install -r requirements.txt
export SCF_DB=$PWD/data/scf.db
python db/seed.py                 # 1095 demand rows + 365 logged forecasts
python -m pytest tests/ -v        # engine + API tests
uvicorn api.main:app --port 8000  # then open /docs and try each endpoint
```

Expected anchors: analytics MAE ≈ 14.36; risk bands 0.10/0.20;
thresholds 0.40/0.70 with modifiers 0/0.05/0.10.

## 6. Glossary

- **API / endpoint** — a named function exposed over HTTP.
- **JSON** — the text format APIs speak.
- **Client / server** — asker / answerer; dashboard / FastAPI.
- **Model artifact** — a trained model saved to disk (`.pkl`).
- **Feature / lag feature** — model inputs; `lag_7` = demand 7 days before
  the target day. Models only know what you hand them.
- **Recursive forecasting** — multi-step prediction that feeds its own
  outputs back as inputs; flattens over long horizons.
- **Backtest** — scoring predictions against known past values.
- **MAE** — mean absolute error; average |prediction − actual|.
- **CV** — coefficient of variation; std / mean, scale-free volatility.
- **VaR 95** — 95th percentile of error magnitude; a "bad day" bound.
- **Cost ratio** — production cost / retail price; the decision variable.
- **SQL / JOIN / GROUP BY** — database language; combine tables, aggregate.
- **Upsert** — INSERT or UPDATE if the row exists; makes repeated calls safe.
