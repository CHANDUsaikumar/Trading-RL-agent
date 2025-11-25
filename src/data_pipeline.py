# src/data_pipeline.py
import argparse
import yfinance as yf
import pandas as pd
import ta
import os

def download_data(ticker='SPY', start='2010-01-01', end=None, interval='1d'):
    """
    Download OHLCV data using yfinance.
    """
    # ensure we keep non-adjusted columns so 'Adj Close' exists; some yfinance versions change defaults
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} {start} - {end}")
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    # some yfinance configs may not include 'Adj Close' if auto-adjust is on; fallback to Close
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']
    return df

def add_technical_indicators(df):
    """
    Add a small set of technical indicators and returns.
    Keeps 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume' + features.
    """
    ohlc = df.copy()
    ohlc['returns'] = ohlc['Adj Close'].pct_change().fillna(0)
    # Momentum
    close_series = ohlc['Adj Close'].squeeze()
    ohlc['rsi'] = ta.momentum.rsi(close_series, window=14).fillna(50)
    # Moving averages
    ohlc['ma10'] = ta.trend.sma_indicator(close_series, window=10).bfill()
    ohlc['ma50'] = ta.trend.sma_indicator(close_series, window=50).bfill()
    # Volatility
    high_series = ohlc['High'].squeeze()
    low_series = ohlc['Low'].squeeze()
    ohlc['atr'] = ta.volatility.average_true_range(high_series, low_series, ohlc['Close'].squeeze(), window=14).bfill()
    # Momentum: MACD diff
    macd = ta.trend.MACD(close_series)
    ohlc['macd_diff'] = (macd.macd_diff()).fillna(0)
    # Normalize / fill remaining NaNs
    ohlc = ohlc.ffill().bfill().dropna()
    return ohlc

def save_processed(df, path='data/processed/spy_features.parquet', output_format='parquet'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if output_format == 'csv':
        csv_path = path.replace('.parquet', '.csv')
        df.to_csv(csv_path, index=True)
        print(f"Saved processed data to {csv_path}")
        return

    # Try parquet first; if parquet engine is unavailable, fall back to CSV
    try:
        df.to_parquet(path)
        print(f"Saved processed data to {path}")
    except ImportError as e:
        csv_path = path.replace('.parquet', '.csv')
        df.to_csv(csv_path, index=True)
        print(f"parquet engine not available ({e}), saved processed data to {csv_path}")
    except Exception as e:
        # Pandas raises a generic Exception when no parquet engine is found.
        msg = str(e).lower()
        if 'parquet' in msg or 'pyarrow' in msg or 'fastparquet' in msg:
            csv_path = path.replace('.parquet', '.csv')
            df.to_csv(csv_path, index=True)
            print(f"Could not write parquet ({e}), saved processed data to {csv_path}")
        else:
            raise

def load_processed(path='data/processed/spy_features.parquet'):
    return pd.read_parquet(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='SPY')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--interval', default='1d')
    parser.add_argument('--output-format', choices=['parquet', 'csv'], default='parquet', help='output format to write processed data')
    parser.add_argument('--outpath', default=None, help='override output path')
    args = parser.parse_args()

    df = download_data(args.ticker, start=args.start, end=args.end, interval=args.interval)
    df = add_technical_indicators(df)
    if args.outpath:
        outpath = args.outpath
    else:
        outpath = f"data/processed/{args.ticker.lower()}_features.parquet" if args.output_format == 'parquet' else f"data/processed/{args.ticker.lower()}_features.csv"
    save_processed(df, path=outpath, output_format=args.output_format)
