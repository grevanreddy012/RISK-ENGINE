import sqlite3
import pandas as pd
import os

# Database will be stored in risk-engine/data/market_data.db
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
DB_PATH = os.path.join(DB_DIR, 'market_data.db')

def get_connection():
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS historical_prices (
                ticker TEXT,
                date TEXT,
                adj_close REAL,
                PRIMARY KEY (ticker, date)
            )
        ''')
        conn.commit()

def get_cached_prices(ticker, start_date, end_date):
    """Retrieve prices from cache for a specific date range."""
    query = """
        SELECT date, adj_close as "Adj Close"
        FROM historical_prices
        WHERE ticker = ? AND date >= ? AND date <= ?
        ORDER BY date
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(ticker, start_date, end_date), parse_dates=['date'])
    
    if not df.empty:
        df.set_index('date', inplace=True)
    return df

def get_max_cached_date(ticker):
    """Get the most recent date we have data for a ticker."""
    query = "SELECT MAX(date) FROM historical_prices WHERE ticker = ?"
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (ticker,))
        result = cursor.fetchone()
    return result[0] if result and result[0] else None

def get_min_cached_date(ticker):
    """Get the earliest date we have data for a ticker."""
    query = "SELECT MIN(date) FROM historical_prices WHERE ticker = ?"
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (ticker,))
        result = cursor.fetchone()
    return result[0] if result and result[0] else None

def save_prices(ticker, df):
    """Save new prices to cache."""
    if df.empty:
        return
        
    # Expecting df to have Date index and 'Adj Close' column
    df_save = df[['Adj Close']].copy()
    df_save['ticker'] = ticker
    df_save = df_save.reset_index()
    # Ensure date is string YYYY-MM-DD
    df_save['date'] = df_save['Date'].dt.strftime('%Y-%m-%d')
    df_save = df_save[['ticker', 'date', 'Adj Close']]
    df_save.columns = ['ticker', 'date', 'adj_close']
    
    with get_connection() as conn:
        # Use executemany with REPLACE to handle conflicts gracefully
        records = df_save.to_records(index=False).tolist()
        conn.executemany('''
            INSERT OR REPLACE INTO historical_prices (ticker, date, adj_close)
            VALUES (?, ?, ?)
        ''', records)
        conn.commit()
