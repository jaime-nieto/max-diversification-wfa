"""
The diversification ratio and the portfolios that are built from it.

Diversification ratio of a portfolio with weights w (Choueifaty & Coignard, 2008):

    DR(w) = (weighted average of the stocks' volatilities) / (portfolio volatility)
          = (w' sigma) / sqrt(w' Sigma w)

DR = 1 for a single stock and grows as the holdings become less correlated.
The Most Diversified Portfolio (MDP) is the long-only portfolio with the
highest DR. It is also the portfolio with the highest Sharpe ratio if every
stock is expected to pay in proportion to its volatility; `risk_based` covers
other assumptions, with the MDP and minimum variance as its two ends.

Every function takes a T x N array of in-sample daily returns (no missing
values) and returns N weights that are non-negative and add up to 1.
"""
import cvxpy as cp
import numpy as np

# Weights below this share of the portfolio are numerical noise of the solver.
ZERO_WEIGHT = 1e-5


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def diversification_ratio(weights, cov):
    """DR from a covariance matrix (the textbook definition)."""
    vol = np.sqrt(np.diag(cov))
    return float(weights @ vol / np.sqrt(weights @ cov @ weights))


def realised_diversification_ratio(weights, returns):
    """DR of a constant-weight portfolio over a window of returns.

    Same number as `diversification_ratio(weights, sample covariance)`, but
    without building the N x N matrix.
    """
    stock_vol = returns.std(axis=0, ddof=1)
    portfolio_vol = (returns @ weights).std(ddof=1)
    return float(weights @ stock_vol / portfolio_vol)


def blockwise_diversification_ratio(weights, returns, block):
    """Average DR over consecutive blocks of `block` days.

    A DR measured on a short window is on average higher than one measured on
    a long window that mixes calm and turbulent periods. Cutting the in-sample
    window into blocks as long as the out-of-sample window makes the two
    comparable.
    """
    starts = range(0, len(returns) - block + 1, block)
    return float(np.mean([realised_diversification_ratio(weights, returns[i:i + block])
                          for i in starts]))


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------
def standardise(returns):
    """Demean each stock and scale it to unit volatility. Returns (x, volatilities)."""
    vol = returns.std(axis=0, ddof=1)
    return (returns - returns.mean(axis=0)) / vol, vol


def ledoit_wolf_shrinkage(x):
    """Ledoit & Wolf (2004) shrinkage intensity for a correlation matrix.

    With more stocks than days the sample correlation matrix R is singular and
    its small eigenvalues are noise. The estimator used here is

        C = (1 - delta) R + delta I

    and this function returns the delta that minimises the expected squared
    error of C. `x` is the T x N matrix of standardised returns. All the sums
    over the N x N matrix are computed from the T x T matrix x x'.
    """
    t, n = x.shape
    x = x / x.std(axis=0)                    # unit variance with the 1/T convention
    gram = x @ x.T
    s_norm2 = (gram ** 2).sum() / t ** 2     # ||S||^2, with S = x'x / T
    d2 = s_norm2 - n                         # ||S - I||^2  (the diagonal of S is 1)
    b2 = ((np.diag(gram) ** 2).sum() / t - s_norm2) / t
    return float(np.clip(b2 / d2, 0.0, 1.0)) if d2 > 0 else 1.0


def constant_correlation_shrinkage(x):
    """Ledoit & Wolf (2004, "Honey, I shrunk the sample covariance matrix") for a correlation matrix.

    The target is the matrix in which every pair of stocks has the same
    correlation, the average one (rho):

        C = (1 - delta) R + delta F,     F = (1 - rho) I + rho * ones

    For stocks this target is closer to the truth than the identity (no stock
    is unrelated to the others), so the optimal delta is larger. Returns
    (delta, rho). `x` is the T x N matrix of standardised returns; as in
    `ledoit_wolf_shrinkage`, every sum over the N x N matrix is computed from
    T x T or T x N arrays.
    """
    t, n = x.shape
    x = x / x.std(axis=0)                    # unit variance with the 1/T convention
    gram = x @ x.T
    s_norm2 = (gram ** 2).sum() / t ** 2     # sum of s_ij^2, with S = x.T x / T
    s_sum = (x.sum(axis=1) ** 2).sum() / t   # sum of s_ij
    rho = (s_sum - n) / (n * (n - 1))
    fourth = (x ** 4).sum() / t              # sum over i of mean(x_i^4)
    # pi: sum over i, j of the variance of the estimate s_ij
    pi = (np.diag(gram) ** 2).sum() / t - s_norm2
    # sum over i, j of the covariance between s_ij and the target f_ij
    theta = ((x ** 3).sum(axis=1) * x.sum(axis=1)).sum() / t - fourth - (s_sum - n)
    covariance_with_target = (fourth - n) + rho * theta
    # gamma: squared distance between the sample matrix and the target
    gamma = (s_norm2 - n) - (s_sum - n) ** 2 / (n * (n - 1))
    if gamma <= 0:
        return 1.0, float(rho)
    return float(np.clip((pi - covariance_with_target) / gamma / t, 0.0, 1.0)), float(rho)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
def _min_risk(x, vol, exponent, max_weight, target="identity"):
    """Shared quadratic programme behind every risk-based portfolio.

    Variables u_i = w_i * vol_i (up to scale), i.e. positions measured in units
    of risk. The risk of the portfolio is u.T C u with C the shrunk correlation
    matrix, written as a sum of squares so that the solver never sees an N x N
    matrix. With the identity as the shrinkage target:

        risk = (1 - delta) / (T - 1) * ||x u||^2 + delta * ||u||^2

    and with the constant-correlation target:

        risk = (1 - delta) / (T - 1) * ||x u||^2 + delta (1 - rho) ||u||^2 + delta rho (sum u)^2

    `exponent` says how the expected return of a stock grows with its
    volatility: return ~ vol ** exponent. The portfolio with the highest
    expected return per unit of risk minimises the risk subject to
    sum(u_i * vol_i ** (exponent - 1)) = 1. With exponent 1 that is sum(u) = 1
    (the MDP); with exponent 0 it is "the weights add up to 1" (minimum variance).
    """
    t, n = x.shape
    if max_weight is not None and max_weight * n < 1:
        raise ValueError(f"max_weight={max_weight} is infeasible with {n} stocks")
    scale = vol.mean() / vol                 # weights are proportional to scale * u

    u = cp.Variable(n, nonneg=True)
    s = cp.Variable(t)                       # s = x u: keeps the problem sparse
    if target == "identity":
        delta = ledoit_wolf_shrinkage(x)
        risk = (1 - delta) / (t - 1) * cp.sum_squares(s) + delta * cp.sum_squares(u)
    elif target == "constant correlation":
        delta, rho = constant_correlation_shrinkage(x)
        risk = ((1 - delta) / (t - 1) * cp.sum_squares(s) + delta * (1 - rho) * cp.sum_squares(u)
                + delta * rho * cp.square(cp.sum(u)))
    else:
        raise ValueError(f"Unknown shrinkage target: {target}")
    w = cp.multiply(scale, u)
    constraints = [s == x @ u]
    if exponent == 0:                        # minimum variance: weights add up to 1
        constraints.append(cp.sum(w) == 1)
        if max_weight is not None:
            constraints.append(w <= max_weight)
    else:
        if exponent == 1:                    # MDP: risk positions add up to 1
            constraints.append(cp.sum(u) == 1)
        else:
            constraints.append(cp.sum(cp.multiply(scale ** (1 - exponent), u)) == 1)
        if max_weight is not None:
            total = cp.Variable()            # sum of the unnormalised weights
            constraints += [total == cp.sum(w), w <= max_weight * total]
    problem = cp.Problem(cp.Minimize(risk), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Solver ended with status '{problem.status}'")

    weights = np.clip(scale * u.value, 0.0, None)
    weights[weights < ZERO_WEIGHT * weights.sum()] = 0.0
    return weights / weights.sum()


def max_diversification(returns, max_weight=None):
    """Long-only Most Diversified Portfolio.

    DR does not change if all weights are multiplied by a constant. Writing
    u_i = w_i * vol_i gives DR = sum(u) / sqrt(u' C u), so maximising DR is the
    same as minimising u' C u subject to sum(u) = 1: the MDP is the
    minimum-variance portfolio of the *correlation* matrix, with the result
    divided by each stock's volatility.
    """
    x, vol = standardise(returns)
    return _min_risk(x, vol, exponent=1, max_weight=max_weight)


def min_variance(returns, max_weight=None):
    """Long-only minimum-variance portfolio (same covariance estimate as the MDP)."""
    x, vol = standardise(returns)
    return _min_risk(x, vol, exponent=0, max_weight=max_weight)


def risk_based(returns, max_weight=None, exponent=1.0, target="identity"):
    """Long-only portfolio with the highest ratio (w . vol ** exponent) / portfolio volatility.

    It is the portfolio with the highest Sharpe ratio for an investor who
    expects each stock to pay in proportion to vol ** exponent. The MDP is
    exponent 1 (a stock twice as volatile pays twice as much) and minimum
    variance is exponent 0 (all stocks pay the same). `target` is the
    shrinkage target of the correlation matrix: "identity" or "constant
    correlation".
    """
    x, vol = standardise(returns)
    return _min_risk(x, vol, exponent=exponent, max_weight=max_weight, target=target)


def equal_weight(returns, max_weight=None):
    """1/N: the same weight in every eligible stock."""
    n = returns.shape[1]
    return np.full(n, 1.0 / n)


STRATEGIES = {"MDP": max_diversification, "MinVar": min_variance, "EW": equal_weight}
