## stock-trading-rl

A small research repo that implements a single-asset trading environment, data pipeline, simple baselines and RL agents (PPO, DQN scaffolds). The project is intentionally compact so you can iterate on features, agents, and experiments quickly.

This README was last updated to reflect the actions performed in this workspace: data preparation, environment fixes, short experiments, and a 3-seed PPO sweep (50k timesteps per seed) with backtests and comparison artifacts saved under `results/`.

## Quickstart

1) Create and activate the virtual environment (zsh):

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Prepare / update processed data (downloads with yfinance and computes indicators):

```bash
# writes to data/processed/spy_features.parquet (CSV fallback if parquet engine is missing)
python src/data_pipeline.py --ticker SPY --start 2010-01-01 --end 2024-12-31 --interval 1d
```

4) Train a single PPO or run the provided sweep (examples below).

## Useful commands

Train a single PPO run (default config in `src/agents/train_ppo.py`):

```bash
PYTHONPATH="$PWD" python src/agents/train_ppo.py --data_path data/processed/spy_features.parquet --total_timesteps 50000 --window_size 50 --model_path results/models/ppo_test
```

Run the 3-seed PPO sweep (what was executed during the last session):

```bash
PYTHONPATH="$PWD" python - <<'PY'
import random, numpy as np, torch
from src.agents.train_ppo import train_ppo
seeds = [42, 7, 123]
for i,s in enumerate(seeds, start=1):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    train_ppo(data_path='data/processed/spy_features.parquet', total_timesteps=50000, window_size=50, model_path=f'results/models/ppo_seed{i}')
PY
```

Run a backtest for a single saved model (loads model and evaluates on test split):

```bash
PYTHONPATH="$PWD" python - <<'PY'
from src.backtest import load_model, run_backtest, plot_equity
from src.data_pipeline import load_processed
df = load_processed()
split = int(0.8 * len(df))
test_df = df.iloc[split:].reset_index(drop=True)
model = load_model('results/models/ppo_seed1')
pv, returns, metrics, actions = run_backtest(model, test_df, window_size=50)
plot_equity(pv, outpath='results/plots/ppo_seed1_equity.png')
print(metrics)
PY
```

## What was run in this session (summary of artifacts created)

- Processed dataset: `data/processed/spy_features.parquet` (if your environment had pyarrow installed) or `data/processed/spy_features.csv` fallback.
- Trained models (3-seed sweep, 50k timesteps each):
  - `results/models/ppo_seed1`
  - `results/models/ppo_seed2`
  - `results/models/ppo_seed3`
- Per-model equity plots:
  - `results/plots/ppo_seed1_equity.png`
  - `results/plots/ppo_seed2_equity.png`
  - `results/plots/ppo_seed3_equity.png`
- Comparison artifacts:
  - `results/plots/ppo_sweep_metrics.csv` — metrics rows for each model and the SMA baseline
  - `results/plots/ppo_sweep_compare.png` — combined equity plot (PPO seeds vs SMA)

## Files of interest (high level)

- `src/data_pipeline.py` — downloads OHLCV via yfinance and computes simple technical indicators (RSI, MAs, ATR, MACD diff). Saves processed features.
- `src/env.py` — trading environment (windowed observations). Adjusted to be compatible with SB3 wrappers (Gym/Gymnasium compatibility layer).
- `src/agents/train_ppo.py` — SB3 PPO training entrypoint used for experiments.
- `src/backtest.py` — load a saved SB3 model and evaluate on the holdout/test split; produces PV series and metrics.
- `src/baselines/sma_baseline.py` — simple SMA crossover baseline used for comparison.
- `src/utils.py` — scaling helpers and metric computations (cumulative return, annualized return/volatility, sharpe, max drawdown).

## Notes, warnings & small fixes

- Stable-Baselines3 issues a warning about Gym → Gymnasium compatibility. The project currently relies on the compatibility layer (shimmy) and the environment's `reset()`/`step()` return signatures were adjusted to match SB3 expectations.
- There are a couple of non-fatal FutureWarnings about calling `float()` on single-element pandas Series in `src/utils.py`; these were noted and can be patched by using `.iloc[0]` — recommended to keep the code future-proof.
- Training in this environment runs on CPU (no CUDA detected). For longer experiments, consider running on a machine with GPU to speed things up.

## Next steps (suggestions)

- Patch the small FutureWarning in `src/utils.py` (I can apply this change if you want).
- Run longer training for the best seed (e.g., 200k timesteps) or run a hyperparameter sweep across learning rate and clip range.
- Add a small notebook that loads `results/plots/ppo_sweep_metrics.csv` and plots aggregated statistics and bootstrap confidence intervals across seeds.

## Reproduce exactly what I ran

From project root (zsh):

```bash
source .venv/bin/activate
PYTHONPATH="$PWD" python - <<'PY'
import random, numpy as np, torch
from src.agents.train_ppo import train_ppo
seeds = [42,7,123]
for i,s in enumerate(seeds, start=1):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    train_ppo(data_path='data/processed/spy_features.parquet', total_timesteps=50000, window_size=50, model_path=f'results/models/ppo_seed{i}')
PY
```

