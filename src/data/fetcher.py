import yfinance as yf
import pandas as pd
import datetime
from .cache import init_db, get_cached_prices, get_max_cached_date, get_min_cached_date, save_prices

# Initialize database on module import
init_db()

def fetch_ticker_data(ticker, start_date, end_date):
    """
    Fetches data for a single ticker, utilizing the SQLite cache.
    Args:
        ticker (str): Ticker symbol
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
    Returns:
        pd.Series: Adjusted close prices, or empty Series if failed.
    """
    min_cached = get_min_cached_date(ticker)
    max_cached = get_max_cached_date(ticker)
    
    # We need to fetch from yfinance if we don't have the full range
    fetch_needed = False
    fetch_start = start_date
    fetch_end = end_date
    
    today = datetime.date.today().strftime('%Y-%m-%d')
    
    if not min_cached or not max_cached:
        fetch_needed = True
    else:
        if start_date < min_cached:
            fetch_needed = True
            fetch_start = start_date
        if end_date > max_cached and max_cached < today:
            fetch_needed = True
            fetch_end = today # Fetch up to today if we are extending the future

    if fetch_needed:
        try:
            # We add a buffer of a few days to start/end just in case of weekends/holidays
            # yfinance start is inclusive, end is exclusive
            dt_start = datetime.datetime.strptime(fetch_start, '%Y-%m-%d') - datetime.timedelta(days=7)
            dt_end = datetime.datetime.strptime(fetch_end, '%Y-%m-%d') + datetime.timedelta(days=1)
            
            yf_start = dt_start.strftime('%Y-%m-%d')
            yf_end = dt_end.strftime('%Y-%m-%d')
            
            df = yf.download(ticker, start=yf_start, end=yf_end, progress=False)
            
            if not df.empty and 'Adj Close' in df.columns:
                # Handle MultiIndex columns if yfinance returns them
                if isinstance(df.columns, pd.MultiIndex):
                    df = df['Adj Close']
                    df = pd.DataFrame(df)
                    df.columns = ['Adj Close']
                save_prices(ticker, df)
            elif not df.empty and 'Close' in df.columns:
                if isinstance(df.columns, pd.MultiIndex):
                    df = df['Close']
                    df = pd.DataFrame(df)
                    df.columns = ['Adj Close'] # Treat Close as Adj Close
                else:
                    df = df.rename(columns={'Close': 'Adj Close'})
                save_prices(ticker, df)
                
        except Exception as e:
            print(f"Warning: Failed to fetch {ticker} from yfinance: {e}")
            pass # Fall back to whatever is in the cache

    # Now get the requested range from cache
    cached_df = get_cached_prices(ticker, start_date, end_date)
    if cached_df.empty:
        return pd.Series(dtype=float, name=ticker)
        
    series = cached_df['Adj Close']
    series.name = ticker
    return series

def fetch_portfolio_data(tickers, start_date, end_date):
    """
    Fetches data for multiple tickers.
    Args:
        tickers (list): List of ticker strings
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
    Returns:
        pd.DataFrame: Prices with tickers as columns
    """
    series_list = []
    for ticker in tickers:
        s = fetch_ticker_data(ticker.strip().upper(), start_date, end_date)
        if not s.empty:
            series_list.append(s)
    
    if not series_list:
        return pd.DataFrame()
        
    df = pd.concat(series_list, axis=1)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True) # Drop initial NaNs if histories start at different times
    return df
