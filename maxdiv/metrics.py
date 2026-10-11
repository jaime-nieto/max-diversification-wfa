"""
Summary statistics of the walk-forward results.
"""
import numpy as np
import pandas as pd

from . import config


def performance(returns, risk_free, benchmark):
    """Annualised statistics of a daily return series."""
    years = len(returns) / config.TRADING_DAYS
    wealth = (1 + returns).cumprod()
    excess = returns - risk_free
    return {
        "cagr": wealth.iloc[-1] ** (1 / years) - 1,
        "volatility": returns.std() * np.sqrt(config.TRADING_DAYS),
        "sharpe": excess.mean() / excess.std() * np.sqrt(config.TRADING_DAYS),
        "max_drawdown": (wealth / wealth.cummax() - 1).min(),
        "beta": returns.cov(benchmark) / benchmark.var(),
    }


def drawdown(returns):
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def walk_forward_summary(windows):
    """Average in-sample and out-of-sample diversification ratio per strategy.

    Only complete out-of-sample windows are used. Efficiency = out / in. The
    like-for-like version divides by the in-sample DR measured in blocks as
    long as the out-of-sample window.
    """
    w = windows.dropna(subset=["dr_out"])
    g = w.groupby("strategy", sort=False)
    out = pd.DataFrame({
        "windows": g.size(),
        "dr_in": g["dr_in"].mean(),
        "dr_in_blocks": g["dr_in_blocks"].mean(),
        "dr_out": g["dr_out"].mean(),
        "dr_out_min": g["dr_out"].min(),
        "dr_out_max": g["dr_out"].max(),
    })
    out["efficiency"] = out["dr_out"] / out["dr_in"]
    out["efficiency_like_for_like"] = out["dr_out"] / out["dr_in_blocks"]
    return out


def paired_comparison(windows, strategy, other):
    """Out-of-sample DR of `strategy` minus `other`, window by window."""
    dr = windows.pivot(index="optimised", columns="strategy", values="dr_out").dropna()
    diff = dr[strategy] - dr[other]
    return {
        "windows": len(diff),
        "mean_difference": diff.mean(),
        # Windows do not overlap, so a plain paired t-test is a fair first check.
        "t_stat": diff.mean() / (diff.std() / np.sqrt(len(diff))),
        "share_higher": (diff > 0).mean(),
    }


def turnover_per_year(windows, strategy, n_days):
    """One-way turnover per year: half of everything traded, excluding the initial purchase."""
    traded = windows.loc[windows.strategy == strategy, "traded"].iloc[1:]
    return traded.sum() / 2 / (n_days / config.TRADING_DAYS)


def sharpe_difference(returns, other, risk_free, draws=config.BOOTSTRAP_DRAWS,
                      block=config.BOOTSTRAP_BLOCK, seed=0):
    """Sharpe ratio of `returns` minus that of `other`, with a 95% interval.

    Two portfolios of the same stocks move together, so the difference between
    their Sharpe ratios is known much better than each ratio. The interval
    comes from a circular block bootstrap of the pairs of daily returns
    (blocks of `block` days keep the clustering of volatility): the days are
    redrawn many times and the difference is computed on each redraw. If the
    interval contains 0, the sample cannot tell the two portfolios apart.
    """
    frame = pd.DataFrame({"a": returns - risk_free, "b": other - risk_free}).dropna()
    a, b = frame["a"].to_numpy(), frame["b"].to_numpy()
    ratio = lambda x: x.mean(axis=-1) / x.std(axis=-1, ddof=1) * np.sqrt(config.TRADING_DAYS)
    rng = np.random.default_rng(seed)
    n = len(a)
    starts = rng.integers(0, n, (draws, n // block + 1))
    days = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    simulated = ratio(a[days]) - ratio(b[days])
    return {"difference": float(ratio(a) - ratio(b)), "low": float(np.quantile(simulated, 0.025)),
            "high": float(np.quantile(simulated, 0.975)), "days": n}
