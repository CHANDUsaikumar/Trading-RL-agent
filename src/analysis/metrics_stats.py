"""Compute metrics and bootstrap significance tests for PPO seeds vs SMA baseline.

Usage: run from project root with PYTHONPATH. Example:

PYTHONPATH="$PWD" python src/analysis/metrics_stats.py

This script looks for models in `results/models/ppo_seedX`, runs backtests on the test split,
computes metrics (CAGR, Sharpe, Sortino, Volatility, Max Drawdown, Calmar, Turnover),
computes 95% bootstrap confidence intervals for each metric, and runs a paired bootstrap
significance test (same resample indices) comparing each PPO seed against the SMA baseline.
"""
import os
import numpy as np
import pandas as pd

from src.data_pipeline import load_processed
from src.backtest import load_model, run_backtest
from src.utils import (
    pretty_metrics,
    returns_to_pv,
    turnover_from_actions,
    sharpe_ratio,
    sortino_ratio,
    annualized_volatility,
    max_drawdown,
    calmar_ratio,
)


def metric_dict_from_pv(pv, returns, actions=None):
    # ensure numpy arrays
    pv = np.asarray(pv, dtype=float)
    returns = np.asarray(returns, dtype=float)
    metrics = {}
    metrics['cumulative_return'] = float(pv[-1] / pv[0] - 1.0)
    # annualized return (CAGR)
    periods = len(pv)
    metrics['annualized_return'] = float((pv[-1] / pv[0]) ** (252.0 / periods) - 1.0)
    metrics['volatility'] = float(annualized_volatility(returns))
    metrics['sharpe'] = float(sharpe_ratio(returns))
    metrics['sortino'] = float(sortino_ratio(returns))
    metrics['max_drawdown'] = float(max_drawdown(pv))
    metrics['calmar'] = float(calmar_ratio(pv, returns))
    if actions is not None:
        metrics['turnover'] = float(turnover_from_actions(actions))
    else:
        metrics['turnover'] = np.nan
    return metrics


def bootstrap_metric_ci(returns, metric_fn, n_boot=1000, alpha=0.05, random_state=0):
    """Bootstrap CI for a metric computed from returns.

    We resample indices with replacement and compute the metric on resampled returns.
    metric_fn should accept (pv, returns) or (returns,) depending on implementation. Here
    metric_fn will be a function that accepts (pv, returns) — we reconstruct pv from returns.
    """
    rng = np.random.default_rng(random_state)
    returns = np.asarray(returns, dtype=float)
    boot_stats = []
    n = len(returns)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled_returns = returns[idx]
        pv = returns_to_pv(resampled_returns, start_value=1.0)
        stat = metric_fn(pv, resampled_returns)
        boot_stats.append(stat)
    lo = np.percentile(boot_stats, 100 * (alpha / 2.0))
    hi = np.percentile(boot_stats, 100 * (1.0 - alpha / 2.0))
    return lo, hi, np.mean(boot_stats)


def paired_bootstrap_pvalue(returns_a, returns_b, metric_fn, n_boot=5000, random_state=0):
    """Paired bootstrap test comparing metric(A) - metric(B).

    We resample indices with replacement (same indices for both series) and compute metric
    differences. Returns p-value for two-sided test and 95% CI for the difference.
    """
    rng = np.random.default_rng(random_state)
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    n = min(len(a), len(b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ra = a[idx]
        rb = b[idx]
        pva = returns_to_pv(ra, start_value=1.0)
        pvb = returns_to_pv(rb, start_value=1.0)
        ma = metric_fn(pva, ra)
        mb = metric_fn(pvb, rb)
        diffs.append(ma - mb)
    diffs = np.asarray(diffs)
    # two-sided p-value: fraction of bootstrap diffs whose sign differs from observed
    obs_a = metric_fn(returns_to_pv(a, 1.0), a)
    obs_b = metric_fn(returns_to_pv(b, 1.0), b)
    obs_diff = obs_a - obs_b
    p_value = np.mean(np.abs(diffs) >= abs(obs_diff))
    ci_lo = np.percentile(diffs, 2.5)
    ci_hi = np.percentile(diffs, 97.5)
    return p_value, (ci_lo, ci_hi), obs_diff


def main():
    # find PPO seeds
    model_dir = 'results/models'
    seed_models = []
    for name in sorted(os.listdir(model_dir)):
        if name.startswith('ppo_seed'):
            seed_models.append(os.path.join(model_dir, name))
    if not seed_models:
        print('No ppo_seed models found in', model_dir)
        return

    df = load_processed()
    split = int(0.8 * len(df))
    test_df = df.iloc[split:].reset_index(drop=True)

    results = []
    pvs = {}

    for i, model_path in enumerate(seed_models, start=1):
        print('Backtesting', model_path)
        model = load_model(model_path)
        pv, returns, metrics, actions = run_backtest(model, test_df, window_size=50)
        md = metric_dict_from_pv(pv, returns, actions)
        md['model'] = os.path.basename(model_path)
        # bootstrap CIs for cumulative_return and sharpe as examples
        lo, hi, mean_boot = bootstrap_metric_ci(returns, lambda pv, r: float(pv[-1] / pv[0] - 1.0), n_boot=1000)
        md['cumulative_return_ci_lo'] = float(lo)
        md['cumulative_return_ci_hi'] = float(hi)
        slo, shi, _ = bootstrap_metric_ci(returns, lambda pv, r: float(sharpe_ratio(r)), n_boot=1000)
        md['sharpe_ci_lo'] = float(slo)
        md['sharpe_ci_hi'] = float(shi)
        results.append(md)
        pvs[md['model']] = (pv, returns)

    # SMA baseline
    from src.baselines.sma_baseline import sma_strategy
    baseline_out = sma_strategy(test_df, short=10, long=50)
    baseline_pv = baseline_out['pv'].values
    baseline_returns = np.diff(baseline_pv) / baseline_pv[:-1]
    bmd = metric_dict_from_pv(baseline_pv, baseline_returns, actions=None)
    bmd['model'] = 'sma_baseline'
    lo, hi, _ = bootstrap_metric_ci(baseline_returns, lambda pv, r: float(pv[-1] / pv[0] - 1.0), n_boot=1000)
    bmd['cumulative_return_ci_lo'] = float(lo)
    bmd['cumulative_return_ci_hi'] = float(hi)
    slo, shi, _ = bootstrap_metric_ci(baseline_returns, lambda pv, r: float(sharpe_ratio(r)), n_boot=1000)
    bmd['sharpe_ci_lo'] = float(slo)
    bmd['sharpe_ci_hi'] = float(shi)
    results.append(bmd)
    pvs[bmd['model']] = (baseline_pv, baseline_returns)

    # Paired bootstrap p-values comparing each PPO seed to SMA (on cumulative return)
    comparisons = []
    for md in results:
        if md['model'].startswith('ppo_seed'):
            pv_a, rets_a = pvs[md['model']]
            pv_b, rets_b = pvs['sma_baseline']
            pval, ci, obs_diff = paired_bootstrap_pvalue(rets_a, rets_b, lambda pv, r: float(pv[-1] / pv[0] - 1.0), n_boot=2000)
            comparisons.append({'model': md['model'], 'pvalue_cumret': float(pval), 'diff_obs': float(obs_diff), 'diff_ci_lo': float(ci[0]), 'diff_ci_hi': float(ci[1])})

    # Save results
    out_df = pd.DataFrame(results)
    out_csv = 'results/plots/ppo_sweep_metrics_with_ci.csv'
    out_df.to_csv(out_csv, index=False)
    print('Wrote metrics + CI to', out_csv)

    comp_df = pd.DataFrame(comparisons)
    comp_csv = 'results/plots/ppo_sweep_comparisons.csv'
    comp_df.to_csv(comp_csv, index=False)
    print('Wrote paired bootstrap comparisons to', comp_csv)


if __name__ == '__main__':
    main()
