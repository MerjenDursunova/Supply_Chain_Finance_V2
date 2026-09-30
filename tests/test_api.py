import os
import tempfile

os.environ["SCF_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")

from fastapi.testclient import TestClient           
from api.main import app                           
from api import database as db                     

client = TestClient(app)


def setup_module():
    with client:
        pass  


def _seed_one():
    from datetime import date, timedelta
    pid = db.upsert_product("test-pump", "EMEA", 4.0, 20.0)
    base = date(2026, 1, 1)
    for i in range(60):
        db.insert_demand(pid, (base + timedelta(days=i)).isoformat(), 100 + i % 10)
    return pid


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_forecast_backtest_scores_and_feeds_analytics():
    _seed_one()
    r = client.post("/forecast/test-pump", params={"horizon": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["backtest"] is True and body["mae"] > 0
    assert len(body["actuals"]) == 7
    mae = client.get("/analytics/mae-by-region").json()
    assert any(row["region"] == "EMEA" for row in mae)
    var = client.get("/analytics/var-by-region").json()
    assert any("var_95" in row for row in var)


def test_recommend_uses_forecast_error_var():
    _seed_one()
    client.post("/forecast/test-pump", params={"horizon": 7})
    r = client.post("/recommend", json={
        "product": "test-pump", "production_cost": 4.0, "retail_price": 20.0})
    assert r.status_code == 200
    body = r.json()
    assert body["recommended_method"] == "Early Payment"   # ratio 0.20 <= 0.40
    assert body["var_source"] == "forecast_errors"
    assert "rationale" in body
    summary = client.get("/analytics/recommendation-summary").json()
    assert any(row["product"] == "test-pump" for row in summary)


def test_riskiest_products_ranked():
    _seed_one()
    rows = client.get("/analytics/riskiest-products").json()
    assert rows and all("cv" in row for row in rows)
