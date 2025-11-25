"""backtest.py
Load trained model, run on test set, compute & plot metrics.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env import TradingEnv
from src.utils import pretty_metrics

MODEL_DIR = "results/models"

def load_model(model_path):
    if model_path.endswith('.zip') or os.path.isfile(model_path + ".zip"):
        # try PPO first
        try:
            return PPO.load(model_path)
        except Exception:
            try:
                return DQN.load(model_path)
            except Exception:
                raise
    else:
        raise ValueError("Model path should point to a saved SB3 model (without extension or with .zip)")

def run_backtest(model, test_df, window_size=50, render=False):
    env = DummyVecEnv([lambda: TradingEnv(df=test_df, window_size=window_size)])
    obs = env.reset()
    done = False
    pv_series = []
    actions = []
    # If model is SB3 model, we can use model.predict
    step = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        pv_series.append(info[0].get('portfolio_value', np.nan))
        actions.append(action)
        step += 1
        if done:
            break
    # align pv_series length to test_df days (may be shorter)
    pv = np.array(pv_series)
    # compute daily returns from pv
    returns = np.diff(pv) / pv[:-1]
    metrics = pretty_metrics(pv, returns)
    return pv, returns, metrics, actions

def plot_equity(pv, outpath=None):
    plt.figure(figsize=(10, 5))
    plt.plot(pv)
    plt.title("Equity Curve")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=200)
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    from src.data_pipeline import load_processed
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.path.join(MODEL_DIR, "ppo_spy"), help='model path (SB3)')
    parser.add_argument('--window', type=int, default=50)
    args = parser.parse_args()

    df = load_processed()
    split = int(0.8 * len(df))
    test_df = df.iloc[split:].reset_index(drop=True)
    model = load_model(args.model)
    pv, returns, metrics, actions = run_backtest(model, test_df, window_size=args.window)
    print("Backtest metrics:", metrics)
    plot_equity(pv)
