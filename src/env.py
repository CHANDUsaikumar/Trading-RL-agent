# src/env.py
import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    """
    Simple single-asset trading environment.
    Observation: flattened window of features + current_position
    Action (continuous): target allocation in [-1, 1]
    Reward: log portfolio return after applying position and transaction cost
    Portfolio is represented as normalized value (start = 1.0)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, feature_cols=None, window_size=50, commission=0.0005, max_position=1.0, warmup=0):
        super().__init__()
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols if feature_cols is not None else ['Adj Close', 'returns', 'rsi', 'ma10', 'ma50', 'atr', 'macd_diff']
        for c in self.feature_cols:
            assert c in self.df.columns, f"missing column {c} in df"
        self.window_size = window_size
        self.commission = commission
        self.max_position = max_position
        self.warmup = warmup  # steps at the start to skip reward accumulation

        # observation: window_size * n_features (flattened) + 1 (position)
        n_features = len(self.feature_cols)
        obs_dim = window_size * n_features + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # continuous action: target position [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._reset_internal()

    def _reset_internal(self):
        # Start index must allow window + warmup
        self.pos = 0.0
        self.portfolio_value = 1.0
        self.current_step = self.window_size + self.warmup
        self.done = False
        self.history = {'pv': [self.portfolio_value], 'pos': [self.pos]}

    def reset(self, seed=None, options=None):
        self._reset_internal()
        # Gymnasium-compatible reset returns (obs, info). SB3's wrappers may expect this.
        return self._get_obs(), {}

    def _get_obs(self):
        start = self.current_step - self.window_size
        window = self.df.loc[start:self.current_step - 1, self.feature_cols].values.flatten()
        obs = np.concatenate([window, [self.pos]]).astype(np.float32)
        return obs

    def step(self, action):
        action = np.clip(action, -self.max_position, self.max_position)
        target_pos = float(action[0])
        # price movement realized next step: use next day's returns
        # current_step indexes the day where we decide action, returns at that row represent change from previous close to this close
        # We'll use df['returns'] at current_step to approximate P&L for the position we hold during the day.
        if self.current_step >= len(self.df) - 1:
            self.done = True
            # return Gymnasium-compatible 5-tuple even on terminal step
            return self._get_obs(), 0.0, True, False, {'portfolio_value': self.portfolio_value}

        # Use .iat to avoid creating a single-element Series and a future deprecation warning
        ret = float(self.df['returns'].iat[self.current_step])  # ~return for the day
        prev_pv = self.portfolio_value

        # transaction cost for changing position
        trade = abs(target_pos - self.pos)
        cost = trade * self.commission

        # apply new position (assume instantaneous rebalancing at start of day)
        self.pos = target_pos

        # portfolio update: simple linear exposure to asset returns
        pnl = self.pos * ret
        self.portfolio_value = self.portfolio_value * (1.0 + pnl - cost)

        # reward: log change in pv
        reward = 0.0
        if prev_pv > 0:
            reward = np.log(self.portfolio_value / prev_pv + 1e-12)

        self.current_step += 1

        self.history['pv'].append(self.portfolio_value)
        self.history['pos'].append(self.pos)

        obs = self._get_obs() if not self.done else np.zeros_like(self._get_obs())
        info = {'portfolio_value': self.portfolio_value}
        # Gymnasium expects (obs, reward, terminated, truncated, info)
        terminated = bool(self.done)
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self, mode='human'):
        step = self.current_step
        pv = self.portfolio_value
        pos = self.pos
        print(f"Step {step} | Pos {pos:.3f} | PV {pv:.6f}")

    def seed(self, seed=None):
        np.random.seed(seed)
