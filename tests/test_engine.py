"""Tests assert the exact values from notebooks 02 & 03."""
import numpy as np
import pytest
from api.engine import (classify_risk, calculate_risk_adjusted_demand, var_95,
                        recommend_method, full_recommendation)


# ---- notebook 02: classify_risk ----
def test_classify_risk_bands():
    assert classify_risk(0.02) == "Low Risk"
    assert classify_risk(0.1028) == "Medium Risk"   # README's 10.28% CV value
    assert classify_risk(0.3) == "High Risk"


# ---- notebook 02: risk-adjusted demand ----
def test_risk_adjusted_demand():
    assert calculate_risk_adjusted_demand(100, 10, "Low Risk") == 105.0
    assert calculate_risk_adjusted_demand(100, 10, "Medium Risk") == 100.0
    assert calculate_risk_adjusted_demand(100, 10, "High Risk") == 95.0


# ---- notebook 02: VaR is np.percentile(abs_error, 95), linear interpolation ----
def test_var_matches_notebook():
    assert var_95(list(range(1, 101))) == 95.05       # np.percentile(1..100, 95)
    assert var_95([5.0]) == 5.0


# ---- notebook 03: README scenario table (risk = Medium -> thresholds 0.45/0.75) ----
def test_readme_scenarios():
    adj = 140.0
    assert recommend_method(0.20, "Medium Risk", adj)["recommended_method"] == "Early Payment"
    assert recommend_method(0.50, "Medium Risk", adj)["recommended_method"] == "In-House Factoring"
    assert recommend_method(0.80, "Medium Risk", adj)["recommended_method"] == "Bank Financing"


# ---- notebook 03: risk modifier shifts thresholds UP ----
def test_risk_modifier_direction():
    low = recommend_method(0.45, "Low Risk", 0.0)
    high = recommend_method(0.45, "High Risk", 0.0)
    assert low["low_threshold"] == pytest.approx(0.40)
    assert low["high_threshold"] == pytest.approx(0.70)
    assert high["low_threshold"] == pytest.approx(0.50)
    assert high["high_threshold"] == pytest.approx(0.80)
    assert low["recommended_method"] == "In-House Factoring"   
    assert high["recommended_method"] == "Early Payment"       


# ---- notebook 03: rationale text present ----
def test_rationale_mentions_cost_ratio():
    out = recommend_method(0.50, "Medium Risk", 137.0, with_rationale=True)
    assert "cost ratio: 0.50" in out["rationale"]
    assert "Medium Risk" in out["rationale"]


# ---- assembled pipeline ----
def test_full_recommendation_shape():
    rng = np.random.default_rng(0)
    out = full_recommendation(10, 20, rng.normal(140, 15, 200))
    for k in ("mean_demand", "demand_cv", "risk_level", "var_95",
              "risk_adjusted_demand", "recommended_method", "rationale"):
        assert k in out
    assert out["cost_ratio"] == pytest.approx(0.5)


def test_full_recommendation_invalid_price():
    with pytest.raises(ValueError):
        full_recommendation(10, 0, [1, 2, 3])
