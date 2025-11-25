import pandas as pd
import numpy as np

def sma_strategy(df, short=10, long=50, initial_capital=1.0, commission=0.0005):
    """
    Returns portfolio value series applying simple SMA crossover:
    - long entry when short_ma > long_ma -> long 1.0
    - exit/short when short_ma < long_ma -> long -1.0 (optional)
    Here we implement only long/flat for simplicity: positions are 1 or 0.
    """
    df = df.copy()
    df['ma_short'] = df['Adj Close'].rolling(short).mean()
    df['ma_long'] = df['Adj Close'].rolling(long).mean()
    df = df.dropna().reset_index(drop=True)
    position = 0.0
    pv = initial_capital
    pv_series = [pv]
    positions = [position]
    for i in range(1, len(df)):
        # Use iloc to ensure we get scalar values even if dataframe has odd indexing
        short_val = float(df['ma_short'].iloc[i])
        long_val = float(df['ma_long'].iloc[i])
        if short_val > long_val:
            target = 1.0
        else:
            target = 0.0
        trade = abs(target - position)
        cost = trade * commission
        # use today's returns
        r = df.loc[i, 'Adj Close'] / df.loc[i - 1, 'Adj Close'] - 1.0
        pnl = position * r
        pv = pv * (1 + pnl - cost)
        position = target
        pv_series.append(pv)
        positions.append(position)
    out = pd.DataFrame({'pv': pv_series, 'pos': positions}, index=df.index)
    return out


# Usage: import and run on processed df
