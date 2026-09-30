-- Supply Chain Finance Advisor — database schema
-- Standard SQL (SQLite by default; runs on PostgreSQL with minor type tweaks).

CREATE TABLE IF NOT EXISTS regions (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS products (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    region_id   INTEGER NOT NULL REFERENCES regions(id),
    unit_cost   REAL NOT NULL,      -- manufacturer production cost per unit ($)
    retail_price REAL NOT NULL      -- retail price per unit ($)
);

-- Actual observed demand (one row per product per day)
CREATE TABLE IF NOT EXISTS demand_history (
    id         INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    date       TEXT NOT NULL,       -- ISO YYYY-MM-DD
    units      REAL NOT NULL,
    UNIQUE (product_id, date)
);

-- Logged forecasts (every prediction the system makes is stored -> auditable ML)
CREATE TABLE IF NOT EXISTS forecasts (
    id           INTEGER PRIMARY KEY,
    product_id   INTEGER NOT NULL REFERENCES products(id),
    forecast_date TEXT NOT NULL,    -- date the forecast targets
    made_on      TEXT NOT NULL,     -- date the forecast was produced
    y_pred       REAL NOT NULL,
    model_version TEXT NOT NULL
);

-- Logged financing recommendations (decision support audit trail)
CREATE TABLE IF NOT EXISTS predictions_log (
    id            INTEGER PRIMARY KEY,
    created_at    TEXT NOT NULL,
    product_id    INTEGER REFERENCES products(id),
    cost_ratio    REAL NOT NULL,
    demand_cv     REAL NOT NULL,
    var_95        REAL NOT NULL,
    risk_level    TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    model_version TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_demand_product_date ON demand_history(product_id, date);
CREATE UNIQUE INDEX IF NOT EXISTS ux_forecasts_product_date ON forecasts(product_id, forecast_date);
CREATE INDEX IF NOT EXISTS idx_predictions_created  ON predictions_log(created_at);
