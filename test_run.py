import pandas as pd
import numpy as np
import datetime
from src.data.fetcher import fetch_portfolio_data
from src.analytics.metrics import get_returns, calculate_all_metrics
from src.analytics.stress import run_stress_tests
from src.analytics.backtest import run_backtest

def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    port_value = 1000000.0
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=5*365)

    print("Fetching data...")
    prices_df = fetch_portfolio_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    print("Prices shape:", prices_df.shape)

    returns_df = get_returns(prices_df)
    print("Returns shape:", returns_df.shape)

    metrics = calculate_all_metrics(returns_df, weights, conf_level=0.99, horizon=1)
    print("Metrics:")
    for m, v in metrics.items():
        print(m, v)

    stress_df = run_stress_tests(tickers, weights, port_value)
    print("\nStress tests:\n", stress_df)

    backtest_res = run_backtest(returns_df, weights, conf_level=0.99, window=252)
    if backtest_res:
        bt_df, bt_stats = backtest_res
        print("\nBacktest stats:", bt_stats)

    print("\nAll tests passed successfully.")

if __name__ == "__main__":
    main()
