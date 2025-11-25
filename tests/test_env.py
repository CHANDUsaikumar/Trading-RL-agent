import pandas as pd
from src.data_pipeline import download_data, add_technical_indicators
from src.env import TradingEnv
import numpy as np

def test_env_basic():
    df = download_data('SPY', start='2019-01-01', end='2020-01-01')
    df = add_technical_indicators(df)
    env = TradingEnv(df=df, window_size=20)
    obs = env.reset()
    assert obs.shape[0] == 20 * len(env.feature_cols) + 1
    action = np.array([0.5], dtype=float)
    obs, reward, done, info = env.step(action)
    assert isinstance(reward, float)
    assert 'portfolio_value' in info

if __name__ == "__main__":
    test_env_basic()
    print("Env sanity test passed.")
