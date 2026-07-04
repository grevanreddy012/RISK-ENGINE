import pandas as pd
from ..data.fetcher import fetch_portfolio_data

SCENARIOS = {
    "Black Monday": {"start": "1987-10-14", "end": "1987-10-20"},
    "LTCM / Russian default": {"start": "1998-08-01", "end": "1998-10-02"},
    "Dot-com crash": {"start": "2000-03-01", "end": "2002-10-02"},
    "Global Financial Crisis": {"start": "2008-09-01", "end": "2008-12-01"},
    "Flash Crash": {"start": "2010-05-05", "end": "2010-05-07"},
    "China deval / global selloff": {"start": "2015-08-18", "end": "2015-08-26"},
    "COVID crash": {"start": "2020-02-19", "end": "2020-03-24"},
    "2022 rate-hike selloff": {"start": "2022-01-01", "end": "2022-10-02"},
}

def run_stress_tests(tickers, weights, portfolio_value):
    """
    Run historical stress tests on the given portfolio.
    """
    results = []
    
    for name, dates in SCENARIOS.items():
        start = dates["start"]
        end = dates["end"]
        
        # Fetch data for this specific window
        df = fetch_portfolio_data(tickers, start, end)
        
        scenario_return = 0.0
        scenario_warnings = []
        
        if df.empty:
            results.append({
                "Scenario": name,
                "Portfolio Return (%)": 0.0,
                "$ Impact": 0.0,
                "Warnings": "No data available for any ticker."
            })
            continue
            
        # Calculate return for each ticker
        asset_returns = {}
        for ticker in tickers:
            if ticker in df.columns and len(df[ticker].dropna()) >= 2:
                prices = df[ticker].dropna()
                # Return over the period
                ret = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                asset_returns[ticker] = ret
            else:
                asset_returns[ticker] = 0.0
                scenario_warnings.append(f"{ticker} missing data")
        
        # Calculate portfolio return
        for i, ticker in enumerate(tickers):
            scenario_return += asset_returns[ticker] * weights[i]
            
        dollar_loss = scenario_return * portfolio_value
        
        results.append({
            "Scenario": name,
            "Portfolio Return (%)": scenario_return * 100,
            "$ Impact": dollar_loss,
            "Warnings": ", ".join(scenario_warnings) if scenario_warnings else "None"
        })
        
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by="Portfolio Return (%)", ascending=True)
        
    return results_df
