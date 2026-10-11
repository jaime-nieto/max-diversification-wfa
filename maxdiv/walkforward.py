"""
Walk-forward analysis.

    |<------- in-sample: train_days ------->| lag |<- out-of-sample: test_days ->|
                                            t
            the weights are optimised here -^      and held here, untouched

Then both windows move forward by test_days and the process is repeated. No
out-of-sample day is ever used to choose the universe or the weights, and
every out-of-sample day belongs to exactly one window. The stocks that can be
chosen at t are the members of the S&P 500 on day t, not today's members.

For each window the diversification ratio is measured twice, with the same
formula: on the data the weights were fitted to (in-sample) and on the data
that came afterwards (out-of-sample). The ratio of the two is the
walk-forward efficiency: how much of the optimised quantity survives. The
in-sample figure is also computed in blocks as long as the out-of-sample
window, for a like-for-like comparison.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import config
from .optimizer import (STRATEGIES, blockwise_diversification_ratio, ledoit_wolf_shrinkage,
                        realised_diversification_ratio, standardise)

WINDOWS_PER_CHUNK = 16            # windows optimised at a time (bounds memory use)


@dataclass(frozen=True)
class Design:
    """The choices that define one walk-forward run."""
    train_days: int = config.TRAIN_DAYS
    test_days: int = config.TEST_DAYS
    max_weight: float | None = config.MAX_WEIGHT
    # False = the true index members of each date. True = only the companies
    # that are still listed today, i.e. what a "survivors only" data source
    # would give (used to measure the survivorship bias).
    survivors_only: bool = False

    def changed(self, **changes):
        return replace(self, **changes)


@dataclass
class Result:
    design: Design
    gross: pd.DataFrame            # daily out-of-sample returns before costs, one column per strategy
    windows: pd.DataFrame          # one row per window and strategy
    weights: dict                  # strategy -> target weights (optimisation dates x stocks)

    def net(self, cost_bps=config.COST_BPS):
        """Daily returns after trading costs, charged on the first day of each window."""
        net = self.gross.copy()
        for name in net.columns:
            w = self.windows[self.windows.strategy == name]
            days = pd.DatetimeIndex(w["first_day"])
            net.loc[days, name] = ((1 + net.loc[days, name])
                                   * (1 - w["traded"].to_numpy() * cost_bps / 1e4) - 1)
        return net


def optimisation_days(dates, design):
    """Row numbers of the days on which the weights are computed.

    The first one is the last trading day before config.OOS_START. A design
    whose in-sample window is longer than the history available on that day
    starts later, on the first rebalancing date that has a full window.
    """
    first = dates.searchsorted(pd.Timestamp(config.OOS_START)) - 1
    first += design.test_days * max(0, -(-(design.train_days - first) // design.test_days))
    last = len(dates) - 1 - config.EXECUTION_LAG
    return np.arange(first, last, design.test_days)


def eligibility_rules(returns, in_index):
    """One yes/no answer per stock and rule, judged on the in-sample window only."""
    valid = np.isfinite(returns)
    return {
        "member": np.asarray(in_index, dtype=bool),
        "history": (valid.mean(axis=0) >= config.MIN_COVERAGE) & valid[-1],
    }


def eligible_stocks(returns, in_index):
    """Which stocks may be held: index members with a full in-sample history."""
    return np.logical_and.reduce(list(eligibility_rules(returns, in_index).values()))


def index_members(panel, design):
    """Membership table used by a run (trading days x companies)."""
    member = panel.member.to_numpy()
    if design.survivors_only:
        member = member & panel.companies["listed"].to_numpy()
    return member


def hold(weights, returns):
    """Buy and hold. Returns the daily portfolio returns and the final weights."""
    growth = np.cumprod(1 + returns, axis=0)
    value = growth @ weights
    daily = value / np.concatenate([[1.0], value[:-1]]) - 1
    return daily, weights * growth[-1] / value[-1]


def _optimise(task):
    function, in_sample, max_weight = task
    return function(in_sample, max_weight)


def run(panel, design=Design(), strategies=("MDP", "MinVar", "EW"), pool=None):
    """One walk-forward run. `pool` is an optional concurrent.futures executor.

    `strategies` are names of optimizer.STRATEGIES, or a dictionary
    {name: function(in-sample returns, max_weight) -> weights}.
    """
    dates, tickers = panel.returns.index, panel.returns.columns
    R, member = panel.returns.to_numpy(), index_members(panel, design)
    lag = config.EXECUTION_LAG
    fit_days = optimisation_days(dates, design)
    parallel_map = pool.map if pool is not None else map
    if not isinstance(strategies, dict):
        strategies = {s: STRATEGIES[s] for s in strategies}

    gross = {s: [] for s in strategies}
    held = {s: np.zeros(len(tickers)) for s in strategies}       # weights before each trade
    targets = {s: [] for s in strategies}
    rows = []

    for chunk in np.array_split(fit_days, int(np.ceil(len(fit_days) / WINDOWS_PER_CHUNK))):
        # In-sample step: universe and weights, one independent task per window and strategy
        masks, samples = [], []
        for f in chunk:
            window = slice(f - design.train_days + 1, f + 1)
            mask = eligible_stocks(R[window], member[f])
            masks.append(mask)
            samples.append(np.nan_to_num(R[window][:, mask]))
            flat = tickers[mask][samples[-1].std(axis=0) == 0]
            if len(flat):                                     # a price that never moves is a data error
                raise ValueError(f"No price changes in the window that ends on {dates[f].date()}: "
                                 f"{list(flat)}")
        tasks = [(f, x, design.max_weight) for x in samples for f in strategies.values()]
        solutions = iter(list(parallel_map(_optimise, tasks)))

        # Out-of-sample step: trade at the close of day f + lag and hold until the next trade
        for f, mask, x in zip(chunk, masks, samples):
            start = f + lag
            stop = min(start + design.test_days, len(dates) - 1)
            oos = np.nan_to_num(R[start + 1:stop + 1][:, mask])   # no price = position frozen
            shrinkage = ledoit_wolf_shrinkage(standardise(x)[0])
            for s in strategies:
                w = next(solutions)
                target = np.zeros(len(tickers))
                target[mask] = w
                daily, drifted = hold(w, oos)
                complete = len(oos) == design.test_days
                rows.append({
                    "optimised": dates[f], "first_day": dates[start + 1], "last_day": dates[stop],
                    "strategy": s, "eligible": int(mask.sum()), "held": int((w > 0).sum()),
                    "effective_n": 1 / (w ** 2).sum(), "largest_weight": w.max(),
                    "shrinkage": shrinkage,
                    "dr_in": realised_diversification_ratio(w, x),
                    "dr_in_blocks": blockwise_diversification_ratio(w, x, design.test_days),
                    "dr_out": realised_diversification_ratio(w, oos) if complete else np.nan,
                    "traded": np.abs(target - held[s]).sum(),
                })
                gross[s].append(pd.Series(daily, index=dates[start + 1:stop + 1]))
                held[s] = np.zeros(len(tickers))
                held[s][mask] = drifted
                targets[s].append(target)

    return Result(
        design=design,
        gross=pd.DataFrame({s: pd.concat(parts) for s, parts in gross.items()}),
        windows=pd.DataFrame(rows),
        weights={s: pd.DataFrame(np.array(t), index=dates[fit_days], columns=tickers)
                 for s, t in targets.items()},
    )


def hold_targets(panel, design, targets, name):
    """Out-of-sample result of weights that were decided elsewhere.

    `targets` has one row of weights per optimisation date (dates x all
    stocks), as in Result.weights. They are traded and held exactly as in
    `run`: bought one day later, kept for design.test_days, costs on what is
    actually traded. Used for portfolios that combine or choose between the
    weights of several runs.
    """
    dates = panel.returns.index
    R, lag = panel.returns.to_numpy(), config.EXECUTION_LAG
    held = np.zeros(R.shape[1])
    gross, rows = [], []
    for day, target in zip(targets.index, targets.to_numpy()):
        f = dates.get_loc(day)
        start = f + lag
        stop = min(start + design.test_days, len(dates) - 1)
        mask = target > 0
        w = target[mask]
        oos = np.nan_to_num(R[start + 1:stop + 1][:, mask])
        daily, drifted = hold(w, oos)
        complete = len(oos) == design.test_days
        rows.append({
            "optimised": day, "first_day": dates[start + 1], "last_day": dates[stop],
            "strategy": name, "held": int(mask.sum()), "effective_n": 1 / (w ** 2).sum(),
            "largest_weight": w.max(),
            "dr_out": realised_diversification_ratio(w, oos) if complete else np.nan,
            "traded": np.abs(target - held).sum(),
        })
        gross.append(pd.Series(daily, index=dates[start + 1:stop + 1]))
        held = np.zeros(R.shape[1])
        held[mask] = drifted
    return Result(design, pd.DataFrame({name: pd.concat(gross)}), pd.DataFrame(rows), {name: targets})
