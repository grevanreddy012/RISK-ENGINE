import streamlit as st
import pandas as pd
import numpy as np
import datetime

from src.data.fetcher import fetch_portfolio_data
from src.analytics.metrics import get_returns, calculate_all_metrics
from src.analytics.stress import run_stress_tests
from src.analytics.backtest import run_backtest
from src.ui.charts import (plot_price_history, plot_correlation_heatmap, 
                          plot_rolling_volatility, plot_return_distribution,
                          plot_stress_test_results, plot_backtest_results)

st.set_page_config(page_title="Portfolio Risk Engine", layout="wide", page_icon="📈")

st.title("📈 Portfolio Risk Engine")

# --- Sidebar Inputs ---
st.sidebar.header("Portfolio Parameters")
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL, MSFT, GOOGL, SPY")
weights_input = st.sidebar.text_input("Weights (comma-separated, optional)", "")
port_value = st.sidebar.number_input("Portfolio Value ($)", value=1000000.0, step=100000.0)

lookback_years = st.sidebar.slider("Lookback Window (Years)", min_value=1, max_value=20, value=5)
conf_level_pct = st.sidebar.radio("Confidence Level", [95, 99], index=1)
horizon_days = st.sidebar.radio("Horizon", [1, 10], index=0)

# Process inputs
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

# Weights
weights = []
if weights_input.strip():
    try:
        weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
        if len(weights) != len(tickers):
            st.error("Number of weights must match number of tickers.")
            st.stop()
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
    except Exception:
        st.error("Invalid weights format. Must be numbers separated by commas.")
        st.stop()
else:
    # Equal weight
    weights = np.ones(len(tickers)) / len(tickers)

conf_level = conf_level_pct / 100.0

# Calculate dates
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=int(lookback_years * 365.25))

# Fetch data
with st.spinner("Fetching market data..."):
    prices_df = fetch_portfolio_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

if prices_df.empty:
    st.error("Could not fetch data for any of the given tickers.")
    st.stop()

# Handle missing columns if any
valid_tickers = [t for t in tickers if t in prices_df.columns]
if len(valid_tickers) < len(tickers):
    missing = set(tickers) - set(valid_tickers)
    st.warning(f"Could not fetch data for: {', '.join(missing)}. They have been excluded.")

if not valid_tickers:
    st.error("No valid tickers remaining after fetching data.")
    st.stop()

# Re-adjust weights for valid tickers
if len(valid_tickers) < len(tickers):
    valid_indices = [i for i, t in enumerate(tickers) if t in valid_tickers]
    weights = np.array([weights[i] for i in valid_indices])
    weights = weights / np.sum(weights) # Re-normalize
    tickers = valid_tickers

returns_df = get_returns(prices_df)

if returns_df.empty:
    st.error("Not enough data to calculate returns.")
    st.stop()

# Calculate Risk Metrics
metrics = calculate_all_metrics(returns_df, weights, conf_level, horizon_days)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "VaR & CVaR", "Stress Tests", "Backtesting"])

with tab1:
    st.header("Portfolio Overview")
    st.markdown(f"**Assets:** {', '.join(tickers)}")
    st.markdown(f"**Weights:** {', '.join([f'{w:.2%}' for w in weights])}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_price_history(prices_df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_correlation_heatmap(returns_df), use_container_width=True)
        
    st.plotly_chart(plot_rolling_volatility(returns_df), use_container_width=True)

with tab2:
    st.header("Value at Risk (VaR) & Expected Shortfall (CVaR)")
    if horizon_days > 1:
        st.info("Note: Scaling VaR by sqrt(t) for >1 day horizon assumes i.i.d normal returns, which breaks down for fat-tailed empirical distributions.")
        
    metrics_data = []
    for method, vals in metrics.items():
        metrics_data.append({
            "Method": method,
            f"VaR ({conf_level_pct}%, {horizon_days}-Day)": f"{vals['VaR']:.2%}",
            f"CVaR ({conf_level_pct}%, {horizon_days}-Day)": f"{vals['CVaR']:.2%}",
            "VaR ($)": f"${vals['VaR'] * port_value:,.2f}",
            "CVaR ($)": f"${vals['CVaR'] * port_value:,.2f}"
        })
    st.table(pd.DataFrame(metrics_data).set_index("Method"))
    
    st.plotly_chart(plot_return_distribution(returns_df, weights, metrics), use_container_width=True)

with tab3:
    st.header("Historical Stress Tests")
    with st.spinner("Running stress scenarios..."):
        stress_df = run_stress_tests(tickers, weights, port_value)
        
    if not stress_df.empty:
        st.plotly_chart(plot_stress_test_results(stress_df), use_container_width=True)
        
        # Format table for display
        display_df = stress_df.copy()
        display_df['Portfolio Return (%)'] = display_df['Portfolio Return (%)'].map("{:.2f}%".format)
        display_df['$ Impact'] = display_df['$ Impact'].map("${:,.2f}".format)
        st.dataframe(display_df, use_container_width=True)

with tab4:
    st.header("Backtesting (Kupiec POF Test)")
    st.markdown("Comparing rolling 1-day Historical VaR against actual daily portfolio returns.")
    
    with st.spinner("Running backtest..."):
        # We run backtest for 1-day horizon, matching the specified conf_level
        # The window is standard 252 days (~1 year)
        backtest_res = run_backtest(returns_df, weights, conf_level=conf_level, window=252)
        
    if backtest_res:
        bt_df, bt_stats = backtest_res
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Days Tested", bt_stats['Total Observations'])
        col2.metric("Expected Breaches", bt_stats['Expected Breaches'])
        col3.metric("Actual Breaches", bt_stats['Actual Breaches'])
        col4.metric("Kupiec Test Result", "Pass" if bt_stats['Is Calibrated (95% sig)'] else "Fail", 
                   delta="Well Calibrated" if bt_stats['Is Calibrated (95% sig)'] else "Poorly Calibrated",
                   delta_color="normal" if bt_stats['Is Calibrated (95% sig)'] else "inverse")
                   
        st.markdown(f"**LR Statistic:** {bt_stats['LR Statistic']:.4f} | **p-value:** {bt_stats['p-value']:.4f}")
        
        st.plotly_chart(plot_backtest_results(bt_df), use_container_width=True)
    else:
        st.warning("Not enough history to run 1-year rolling backtest.")
