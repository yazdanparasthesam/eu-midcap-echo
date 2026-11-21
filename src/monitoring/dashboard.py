"""Streamlit dashboard for latest signals, equity curves, and model drift."""
from ..bootstrap import ensure
ensure({"streamlit": "streamlit"})
import streamlit as st
import pandas as pd, numpy as np
from pathlib import Path
from ..config import REPORTS, MODELS, DATA_PRO

st.set_page_config(page_title="eu-midcap-echo", layout="wide")
st.title("EU Midcap Echo — daily signal monitor")

@st.cache_data
def load():
    trades = pd.read_csv(REPORTS / "latest_trades.csv") if (REPORTS/"latest_trades.csv").exists() else pd.DataFrame()
    eq = pd.read_csv(REPORTS / "equity_curve.csv", parse_dates=["Date"]).set_index("Date") if (REPORTS/"equity_curve.csv").exists() else pd.DataFrame()
    bench = pd.read_csv(REPORTS / "benchmark_curve.csv", parse_dates=["Date"]).set_index("Date") if (REPORTS/"benchmark_curve.csv").exists() else pd.DataFrame()
    sims = pd.read_csv(REPORTS / "sim_summary.csv") if (REPORTS/"sim_summary.csv").exists() else pd.DataFrame()
    return trades, eq, bench, sims

trades, eq, bench, sims = load()

col1, col2, col3 = st.columns(3)
with col1: st.metric("Latest signals", len(trades))
with col2: st.metric("Strategies ran", len(sims))
with col3:
    if not sims.empty and "CAGR" in sims.columns:
        st.metric("Best CAGR", f"{sims.CAGR.max():.1%}")

if not eq.empty:
    st.subheader("Equity curve (risk-managed) vs benchmark")
    chart = pd.DataFrame({"strategy": eq.squeeze(), "benchmark": bench.squeeze() if not bench.empty else None})
    st.line_chart(chart)

st.subheader("Latest signals")
st.dataframe(trades.tail(20))

st.subheader("Strategy comparison")
st.dataframe(sims)
