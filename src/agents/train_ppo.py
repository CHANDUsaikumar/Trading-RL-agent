# src/agents/train_ppo.py
import os
import argparse
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.env import TradingEnv
from src.data_pipeline import load_processed

MODEL_DIR = "results/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env(df, window_size=50, commission=0.0005):
    return TradingEnv(df=df, window_size=window_size, commission=commission)


def train_ppo(data_path='data/processed/spy_features.parquet',
              total_timesteps=200_000,
              window_size=50,
              model_path=os.path.join(MODEL_DIR, "ppo_spy")):
    df = pd.read_parquet(data_path)
    # split train/val: simple split, first 80% train
    split = int(0.8 * len(df))
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df = df.iloc[split - window_size:].reset_index(drop=True)  # include window overlap

    env = DummyVecEnv([lambda: make_env(train_df, window_size=window_size)])
    env = VecMonitor(env)

    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, policy_kwargs=policy_kwargs)

    # callbacks: make save frequency proportional to total_timesteps
    save_freq = max(1, int(total_timesteps // 4))
    checkpoint_callback = CheckpointCallback(save_freq=save_freq // max(1, env.num_envs), save_path=MODEL_DIR, name_prefix="ppo_checkpoint")

    # simple eval callback on validation env (reduced eval frequency for small runs)
    eval_env = DummyVecEnv([lambda: make_env(val_df, window_size=window_size)])
    eval_env = VecMonitor(eval_env)
    eval_freq = max(1, int(total_timesteps // 4))
    eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_DIR,
                                 log_path="results/", eval_freq=eval_freq, deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/processed/spy_features.parquet')
    parser.add_argument('--total-timesteps', type=int, default=200000)
    parser.add_argument('--window-size', type=int, default=50)
    parser.add_argument('--model-path', default=os.path.join(MODEL_DIR, 'ppo_spy'))
    args = parser.parse_args()

    train_ppo(data_path=args.data_path, total_timesteps=args.total_timesteps, window_size=args.window_size, model_path=args.model_path)
