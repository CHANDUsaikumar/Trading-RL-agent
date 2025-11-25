# src/utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def build_feature_matrix(df, feature_cols, window_size=50):
    """
    Returns numpy array of shape (n_steps-window_size, window_size * n_features)
    and index mapping for each step.
    """
    X = []
    idxs = []
    arr = df[feature_cols].values
    for end in range(window_size, len(df)):
        start = end - window_size
        window = arr[start:end].flatten()
        X.append(window)
        idxs.append(df.index[end])
    X = np.array(X, dtype=np.float32)
    return X, idxs

def fit_scaler(train_df, feature_cols, scaler_path=None):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    return scaler

def load_scaler(scaler_path):
    return joblib.load(scaler_path)

def apply_scaler_df(df, feature_cols, scaler):
    arr = scaler.transform(df[feature_cols].values)
    out = df.copy()
    out[feature_cols] = arr
    return out

# Backtest metrics
def cumulative_return(portfolio_values):
    pv = np.asarray(portfolio_values, dtype=float)
    return pv[-1] / pv[0] - 1.0

def annualized_return(portfolio_values, days_per_year=252):
    pv = np.asarray(portfolio_values, dtype=float)
    total_return = pv[-1] / pv[0]
    periods = len(pv)
    return total_return ** (days_per_year / periods) - 1.0

def annualized_volatility(returns, days_per_year=252):
    arr = np.asarray(returns, dtype=float)
    return np.std(arr) * np.sqrt(days_per_year)

def sharpe_ratio(returns, risk_free=0.0, days_per_year=252):
    arr = np.asarray(returns, dtype=float)
    ann_ret = np.mean(arr) * days_per_year
    ann_vol = np.std(arr) * np.sqrt(days_per_year)
    if ann_vol == 0:
        return np.nan
    return (ann_ret - risk_free) / ann_vol

def sortino_ratio(returns, risk_free=0.0, days_per_year=252, mar=0.0):
    """Compute annualized Sortino ratio.

    returns: 1D array-like of periodic returns (e.g., daily)
    mar: minimum acceptable return per period (defaults to 0)
    """
    arr = np.asarray(returns, dtype=float)
    ann_ret = np.mean(arr) * days_per_year
    # downside deviation (per-period)
    downside = arr[arr < mar]
    if downside.size == 0:
        return np.nan
    downside_std = np.sqrt(np.mean((downside - mar) ** 2)) * np.sqrt(days_per_year)
    if downside_std == 0:
        return np.nan
    return (ann_ret - risk_free) / downside_std

def max_drawdown(portfolio_values):
    pv = np.array(portfolio_values)
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak
    return drawdown.min()

def calmar_ratio(portfolio_values, returns, days_per_year=252):
    """Annualized return divided by absolute max drawdown (Calmar ratio)."""
    ann = annualized_return(portfolio_values, days_per_year=days_per_year)
    mdd = max_drawdown(portfolio_values)
    if mdd == 0:
        return np.nan
    return ann / abs(mdd)

def returns_to_pv(returns, start_value=1.0):
    """Convert a series of periodic returns to a portfolio value series starting at start_value.

    Returns an array of length len(returns)+1 where pv[0] == start_value.
    """
    arr = np.asarray(returns, dtype=float)
    pv = start_value * np.concatenate([[1.0], np.cumprod(1.0 + arr)])
    return pv

def turnover_from_actions(actions):
    """Estimate turnover from an actions/position sequence.

    actions: iterable of positions or target allocations (scalars or arrays). Returns the mean absolute change.
    """
    # flatten nested arrays
    seq = []
    for a in actions:
        try:
            # if action is array-like
            val = np.array(a).ravel()[0]
        except Exception:
            val = float(a)
        seq.append(float(val))
    seq = np.asarray(seq, dtype=float)
    if len(seq) < 2:
        return 0.0
    changes = np.abs(np.diff(seq))
    return float(np.mean(changes))

def pretty_metrics(portfolio_values, returns):
    # Ensure numeric numpy arrays to avoid pandas scalar/series issues
    pv = np.asarray(portfolio_values, dtype=float)
    rets = np.asarray(returns, dtype=float)
    return {
        'cumulative_return': float(cumulative_return(pv)),
        'annualized_return': float(annualized_return(pv)),
        'annualized_volatility': float(annualized_volatility(rets)),
        'sharpe': float(sharpe_ratio(rets)),
        'max_drawdown': float(max_drawdown(pv))
    }
