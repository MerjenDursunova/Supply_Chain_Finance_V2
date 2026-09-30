"""Streamlit dashboard — now a thin client of the FastAPI service.
Run:  streamlit run app/dashboard.py
Env:  API_URL (default http://localhost:8000)
"""
import os
import requests
import pandas as pd
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Supply Chain Finance Advisor", layout="wide")
st.title("AI-Powered Supply Chain Finance Advisor")
st.caption("Extending Chen, Lu & Cai (2020) with ML demand forecasting — "
           "now served through a FastAPI backend with a SQL analytics layer.")

try:
    health = requests.get(f"{API}/health", timeout=5).json()
    st.sidebar.success(f"API online · model `{health['model_version']}`")
except Exception:
    st.sidebar.error(f"API unreachable at {API} — start it with `uvicorn api.main:app`")
    st.stop()

# ---------- Block 1: demand forecast ----------
st.header("1 · Demand Forecast")
products = requests.get(f"{API}/products", timeout=10).json()
if products:
    names = [p["name"] for p in products]
    col1, col2 = st.columns(2)
    sel = col1.selectbox("Product", names)
    horizon = col2.slider("Horizon (days)", 7, 90, 30)
    if st.button("Run forecast"):
        r = requests.post(f"{API}/forecast/{sel}", params={"horizon": horizon}, timeout=60).json()
        fc = pd.DataFrame({"day": range(1, len(r["forecast"]) + 1),
                           "forecast": r["forecast"]})
        st.line_chart(fc.set_index("day"))
        st.caption(f"Model: {r['model_version']} · every prediction is logged to the database")
else:
    st.info("No products in the database yet — run `python db/seed.py` first.")

# ---------- Block 2: risk + recommendation ----------
st.header("2 · Risk Estimation & Financing Recommendation")
if products:
    c1, c2, c3 = st.columns(3)
    prod = next(p for p in products if p["name"] == sel)
    p_cost = c1.number_input("Production cost ($/unit)", value=float(prod["unit_cost"]), min_value=0.01)
    r_price = c2.number_input("Retail price ($/unit)", value=float(prod["retail_price"]), min_value=0.01)
    p_sel = c3.selectbox("Recommend for product", names, key="rec_prod")
    if st.button("Get recommendation"):
        rec = requests.post(f"{API}/recommend", json={
            "product": p_sel, "production_cost": p_cost, "retail_price": r_price
        }, timeout=30).json()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cost ratio", f"{rec['cost_ratio']:.2f}")
        m2.metric("Demand CV", f"{rec['demand_cv']*100:.1f}%")
        m3.metric("VaR 95%", f"±{rec['var_95']:.0f} units")
        m4.metric("Risk level", rec["risk_level"])
        method = rec["recommended_method"]
        color = {"Early Payment": "green", "In-House Factoring": "orange",
                 "Bank Financing": "blue"}[method]
        st.markdown(f"### :{color}[{method}]")
        st.markdown(f"**Justification:** {rec['rationale']}")
        with st.expander("Full JSON response"):
            st.json(rec)
        html = requests.post(f"{API}/report", json={
            "product": p_sel, "production_cost": p_cost, "retail_price": r_price
        }, timeout=30).text
        st.download_button("Download Full Report (HTML)", data=html,
                           file_name="financing_recommendation.html",
                           mime="text/html")

# ---------- Block 3: SQL analytics ----------
st.header("3 · Portfolio Analytics (live SQL)")
tabs = st.tabs(["MAE by region", "VaR by region", "Riskiest products", "Recommendations"])
for tab, endpoint, title in zip(
    tabs,
    ["mae-by-region", "var-by-region", "riskiest-products", "recommendation-summary"],
    ["MAE by region", "VaR 95 by region", "Riskiest products by CV", "Recommendation log"],
):
    with tab:
        rows = requests.get(f"{API}/analytics/{endpoint}", timeout=30).json()
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No data yet — run a forecast and a recommendation first.")
