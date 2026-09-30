"""Core decision logic — EXACT port of notebooks 02 & 03.

Every threshold, formula and label matches the original notebooks:
  02_demand_risk.ipynb           -> classify_risk, risk_adjusted_demand, var_95
  03_recommendation_engine.ipynb -> get_financing_recommendation

Pure functions, no I/O, fully unit-tested against notebook values.
"""
from __future__ import annotations
import numpy as np

# ---------- from notebook 02: classify_risk ----------
def classify_risk(cv: float) -> str:
    """Notebook bands: <0.10 Low, <0.20 Medium, else High (labels verbatim)."""
    if cv < 0.10:
        return "Low Risk"
    elif cv < 0.20:
        return "Medium Risk"
    else:
        return "High Risk"


def calculate_risk_adjusted_demand(mean_demand: float, std_demand: float, risk_level: str) -> float:
    """Notebook 02: Low -> mean + 0.5*std, Medium -> mean, High -> mean - 0.5*std."""
    if risk_level == "Low Risk":
        return mean_demand + 0.5 * std_demand
    elif risk_level == "Medium Risk":
        return mean_demand
    else:
        return mean_demand - 0.5 * std_demand


def var_95(errors, method: str = "linear") -> float:
    """95th percentile of absolute errors. 'linear' = np.percentile default,
    matching notebook 02 exactly; 'nearest-rank' available for SQL comparison."""
    arr = np.asarray(errors, dtype=float)
    if arr.size == 0:
        return 0.0
    if method == "nearest-rank":
        rank = int(np.ceil(0.95 * arr.size))
        return float(arr[min(rank, arr.size) - 1])
    return float(np.percentile(arr, 95))


# ---------- from notebook 03: get_financing_recommendation ----------
RISK_MODIFIER = {"Low Risk": 0.0, "Medium Risk": 0.05, "High Risk": 0.10}
LOW_BASE, HIGH_BASE = 0.40, 0.70


def recommend_method(cost_ratio: float, risk_level: str,
                     risk_adjusted_demand: float = 0.0,
                     with_rationale: bool = False) -> dict:
    """Port of get_financing_recommendation(). Thresholds shift UP with risk
    (notebook logic, ported as-is)."""
    low_threshold = LOW_BASE + RISK_MODIFIER[risk_level]
    high_threshold = HIGH_BASE + RISK_MODIFIER[risk_level]

    if cost_ratio <= low_threshold:
        method = "Early Payment"
        rationale = (
            f"Production cost is low relative to retail price "
            f"(cost ratio: {cost_ratio:.2f}). "
            f"Benefit from increased production outweighs interest income. "
            f"Risk level is {risk_level}, demand forecast: "
            f"{risk_adjusted_demand:.0f} units."
        )
    elif cost_ratio <= high_threshold:
        method = "In-House Factoring"
        rationale = (
            f"Production cost is moderate relative to retail price "
            f"(cost ratio: {cost_ratio:.2f}). "
            f"Retailer can earn interest while maintaining production levels. "
            f"Risk level is {risk_level}, demand forecast: "
            f"{risk_adjusted_demand:.0f} units."
        )
    else:
        method = "Bank Financing"
        rationale = (
            f"Production cost is high relative to retail price "
            f"(cost ratio: {cost_ratio:.2f}). "
            f"Financial risk too high for retailer to absorb. "
            f"Risk level is {risk_level}, demand forecast: "
            f"{risk_adjusted_demand:.0f} units."
        )

    out = {
        "recommended_method": method,
        "cost_ratio": round(cost_ratio, 4),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "risk_level": risk_level,
        "forecast_demand": round(risk_adjusted_demand, 2),
    }
    if with_rationale:
        out["rationale"] = rationale
    return out


# ---------- assembled pipeline ----------
def full_recommendation(production_cost: float, retail_price: float,
                        demand_series, forecast_errors=None) -> dict:
    """Full pipeline: demand stats -> risk classification -> VaR -> recommendation.
    VaR uses forecast errors (notebook 02 style) when available, else falls
    back to deviations from the demand mean."""
    if retail_price <= 0:
        raise ValueError("retail_price must be positive")
    arr = np.asarray(demand_series, dtype=float)
    if arr.size == 0:
        raise ValueError("empty demand series")

    mean, std = float(arr.mean()), float(arr.std(ddof=1))
    cv = std / mean if mean else 0.0
    level = classify_risk(cv)
    adj = calculate_risk_adjusted_demand(mean, std, level)

    if forecast_errors is not None and len(forecast_errors) > 0:
        var_source = "forecast_errors"
        var = var_95(np.abs(np.asarray(forecast_errors, dtype=float)))
    else:
        var_source = "demand_deviations"
        var = var_95(np.abs(arr - mean))

    rec = recommend_method(production_cost / retail_price, level, adj, with_rationale=True)
    return {
        "mean_demand": round(mean, 2),
        "std_demand": round(std, 2),
        "demand_cv": round(cv, 4),
        "risk_level": level,
        "risk_adjusted_demand": round(adj, 2),
        "var_95": round(var, 2),
        "var_source": var_source,
        "production_cost": production_cost,
        "retail_price": retail_price,
        **rec,
    }
