"""
Choosing between several versions of a portfolio without looking at the future.

If a design is picked because it had the best result over the whole sample,
that result is no longer out-of-sample: the choice used all of it. The honest
way to pick is to make the choice part of the strategy. On each optimisation
date every candidate has a track record (its own out-of-sample returns up to
that day); the candidate with the best record so far is the one that is held
for the next window. What this portfolio earns is out-of-sample twice: for the
weights and for the choice of design.

Two portfolios are built from a set of candidate runs:

  ensemble   the average of the weights of all candidates. Nothing is chosen.
  adaptive   the candidate with the highest Sharpe ratio so far.
"""
import numpy as np
import pandas as pd

from . import config
from . import walkforward as wf

def sharpe(excess):
    """Sharpe ratio per year of daily returns in excess of the risk-free rate."""
    return excess.mean() / excess.std() * np.sqrt(config.TRADING_DAYS)


def burn_in_windows(design):
    """Windows held with the default candidate before the first choice is made.

    Every candidate needs a track record of some length before it can be
    judged: config.SELECTION_BURN_IN_YEARS, whatever the holding period.
    """
    return int(np.ceil(config.SELECTION_BURN_IN_YEARS * config.TRADING_DAYS / design.test_days))


def _weights(result):
    """Target weights of a run with a single strategy (optimisation dates x stocks)."""
    return next(iter(result.weights.values()))


def choices(records, days, default, burn_in):
    """Which candidate is held from each optimisation date.

    `records`: daily excess returns after costs, one column per candidate
    (NaN before a candidate starts). `days`: the optimisation dates. On the
    first `burn_in` dates the choice is `default`. Afterwards it is the
    candidate with the highest Sharpe ratio on the returns up to that day,
    measured on the days for which every candidate has a return.
    """
    common = records.dropna()
    chosen = []
    for k, day in enumerate(days):
        past = common.loc[:day]
        chosen.append(default if k < burn_in or len(past) < 2 else sharpe(past).idxmax())
    return pd.Series(chosen, index=days)


def adaptive(panel, design, results, default, name="Adaptive", burn_in=None):
    """Hold, in each window, the candidate with the best Sharpe ratio so far.

    `results`: {candidate: Result of its walk-forward run}, each with one
    strategy and the same design.test_days, so that their optimisation dates
    coincide. Returns (Result, the candidate chosen on each date).
    """
    records = pd.DataFrame({c: r.net().iloc[:, 0] for c, r in results.items()})
    records = records.sub(panel.risk_free.reindex(records.index), axis=0)
    days = _weights(results[default]).index
    chosen = choices(records, days, default, burn_in_windows(design) if burn_in is None else burn_in)
    targets = pd.DataFrame([_weights(results[c]).loc[day] for day, c in chosen.items()], index=days)
    return wf.hold_targets(panel, design, targets, name), chosen


def ensemble(panel, design, results, name="Ensemble"):
    """Hold the average of the weights of all the candidates that exist on each date."""
    weights = [_weights(r) for r in results.values()]
    days = weights[0].index
    for w in weights[1:]:
        days = days.union(w.index)
    total = sum(w.reindex(days).fillna(0.0) for w in weights)
    running = sum(pd.Series(days.isin(w.index), index=days).astype(float) for w in weights)
    return wf.hold_targets(panel, design, total.div(running, axis=0), name)
