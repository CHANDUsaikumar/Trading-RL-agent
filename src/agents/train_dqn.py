# src/agents/train_dqn.py
import os
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from src.env import TradingEnv
from src.data_pipeline import load_processed

MODEL_DIR = "results/models"
os.makedirs(MODEL_DIR, exist_ok=True)

class DiscreteTradingEnv(TradingEnv):
    """
    Wrap continuous TradingEnv into discrete actions: [-1, 0, 1]
    """
    def __init__(self, df, *args, **kwargs):
        super().__init__(df, *args, **kwargs)
        self.action_space =  __import__('gym').spaces.Discrete(3)  # 0 -> -1, 1 -> 0, 2 -> 1

    def step(self, action):
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        return super().step(np.array([mapping[int(action)]]))

def make_env(df, window_size=50, commission=0.0005):
    return DiscreteTradingEnv(df=df, window_size=window_size, commission=commission)

def train_dqn(data_path='data/processed/spy_features.parquet',
              total_timesteps=200_000,
              window_size=50,
              model_path=os.path.join(MODEL_DIR, "dqn_spy")):
    df = pd.read_parquet(data_path)
    split = int(0.8 * len(df))
    train_df = df.iloc[:split].reset_index(drop=True)
    env = DummyVecEnv([lambda: make_env(train_df, window_size=window_size)])
    env = VecMonitor(env)
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-4, buffer_size=50_000, learning_starts=5_000)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    train_dqn()
