"""
Tests of the optimiser, of the universe, of the price cleaning and of the
walk-forward engine. They use simulated data and need neither the network nor
the downloaded files:

    pytest
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maxdiv import bolsamania                                 # noqa: E402
from maxdiv import config                                     # noqa: E402
from maxdiv import data_download as dl                        # noqa: E402
from maxdiv import data_prep as dp                            # noqa: E402
from maxdiv import metrics                                    # noqa: E402
from maxdiv import optimizer as op                            # noqa: E402
from maxdiv import selection as sel                           # noqa: E402
from maxdiv import universe                                   # noqa: E402
from maxdiv import walkforward as wf                          # noqa: E402


def simulate(days, stocks, seed=0):
    """Returns with one market factor, a few sector factors and stock-specific noise."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.010, (days, 1)) * rng.uniform(0.5, 1.5, stocks)
    sectors = rng.normal(0, 0.008, (days, 4))[:, rng.integers(0, 4, stocks)]
    return market + sectors + rng.normal(0, 1, (days, stocks)) * rng.uniform(0.005, 0.03, stocks)


def shrunk_covariance(returns):
    """The N x N matrix that the optimiser works with implicitly."""
    x, vol = op.standardise(returns)
    delta = op.ledoit_wolf_shrinkage(x)
    corr = (1 - delta) * np.corrcoef(returns, rowvar=False) + delta * np.eye(len(vol))
    return corr * np.outer(vol, vol)


# ---------------------------------------------------------------------------
# Diversification ratio
# ---------------------------------------------------------------------------
def test_dr_textbook_cases():
    two = np.array([0.5, 0.5])
    assert op.diversification_ratio(np.array([1.0]), np.array([[0.04]])) == pytest.approx(1)
    assert op.diversification_ratio(two, np.diag([0.04, 0.04])) == pytest.approx(np.sqrt(2))
    assert op.diversification_ratio(two, np.full((2, 2), 0.04)) == pytest.approx(1)


def test_realised_dr_equals_definition():
    r = simulate(250, 12)
    w = np.random.default_rng(1).dirichlet(np.ones(12))
    assert op.realised_diversification_ratio(w, r) == pytest.approx(
        op.diversification_ratio(w, np.cov(r, rowvar=False)))
    halves = [op.realised_diversification_ratio(w, part) for part in (r[:125], r[125:])]
    assert op.blockwise_diversification_ratio(w, r, 125) == pytest.approx(np.mean(halves))


def test_ledoit_wolf_matches_the_formula_of_the_paper():
    x, _ = op.standardise(simulate(60, 15))
    t, n = x.shape
    z = x / x.std(axis=0)
    s = z.T @ z / t
    d2 = ((s - np.eye(n)) ** 2).sum()
    b2 = sum(((np.outer(row, row) - s) ** 2).sum() for row in z) / t ** 2
    assert op.ledoit_wolf_shrinkage(x) == pytest.approx(min(b2, d2) / d2)


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cap", [None, 0.08])
def test_mdp_matches_a_general_purpose_solver(cap):
    r = simulate(300, 30)
    cov = shrunk_covariance(r)
    w = op.max_diversification(r, cap)
    reference = minimize(lambda v: -op.diversification_ratio(v, cov), np.full(30, 1 / 30),
                         bounds=[(0, cap or 1)] * 30, method="SLSQP",
                         constraints={"type": "eq", "fun": lambda v: v.sum() - 1},
                         options={"ftol": 1e-12, "maxiter": 500}).x
    assert w.sum() == pytest.approx(1) and w.min() >= 0
    assert op.diversification_ratio(w, cov) >= op.diversification_ratio(reference, cov) - 1e-6
    assert np.abs(w - reference).max() < 1e-3


def test_mdp_has_the_highest_dr_and_minvar_the_lowest_variance():
    r = simulate(300, 30)
    cov = shrunk_covariance(r)
    mdp, minvar, ew = (f(r) for f in (op.max_diversification, op.min_variance, op.equal_weight))
    rng = np.random.default_rng(2)
    others = [minvar, ew] + [rng.dirichlet(np.ones(30)) for _ in range(200)]
    assert all(op.diversification_ratio(mdp, cov) > op.diversification_ratio(w, cov) for w in others)
    assert all(minvar @ cov @ minvar < w @ cov @ w for w in [mdp, ew] + others[2:])


def test_mdp_core_property():
    r = simulate(300, 30)
    w = op.max_diversification(r, max_weight=None)
    cov = shrunk_covariance(r)
    vols = np.sqrt(np.diag(cov))
    portfolio_vol = np.sqrt(w @ cov @ w)
    correlations = (cov @ w) / (vols * portfolio_vol)
    held = w > 1e-6
    assert np.ptp(correlations[held]) < 1e-5
    if (~held).any():
        assert (correlations[~held] >= correlations[held][0] - 1e-6).all()


@pytest.mark.parametrize("cap", [0.05, 0.10])
def test_caps_bind_at_the_right_level(cap):
    r = simulate(300, 50)
    for f in (op.max_diversification, op.min_variance):
        w = f(r, max_weight=cap)
        assert w.max() <= cap + 1e-8
        assert w.sum() == pytest.approx(1) and w.min() >= -1e-8


def test_optimizer_handles_redundant_assets():
    r = simulate(200, 10)
    r_redundant = np.column_stack([r, r[:, :3]])
    for f in (op.max_diversification, op.min_variance):
        w = f(r_redundant, max_weight=0.15)
        assert w.sum() == pytest.approx(1) and (w >= -1e-8).all()


def shrunk_covariance_constant_correlation(returns):
    """Covariance matrix behind the constant-correlation target, built explicitly."""
    x, vol = op.standardise(returns)
    delta, rho = op.constant_correlation_shrinkage(x)
    target = np.full((len(vol), len(vol)), rho)
    np.fill_diagonal(target, 1.0)
    corr = (1 - delta) * np.corrcoef(returns, rowvar=False) + delta * target
    return corr * np.outer(vol, vol)


def test_constant_correlation_shrinkage_matches_the_formula_of_the_paper():
    x, _ = op.standardise(simulate(80, 12))
    t, n = x.shape
    z = x / x.std(axis=0)
    s = z.T @ z / t
    rho = (s.sum() - n) / (n * (n - 1))
    target = np.full((n, n), rho)
    np.fill_diagonal(target, 1.0)
    pi, theta_ii, theta_jj = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for row in z:
        deviation = np.outer(row, row) - s
        pi += deviation ** 2 / t
        theta_ii += (row ** 2 - 1)[:, None] * deviation / t
        theta_jj += (row ** 2 - 1)[None, :] * deviation / t
    off_diagonal = ~np.eye(n, dtype=bool)
    rho_hat = np.trace(pi) + (rho / 2 * (theta_ii + theta_jj))[off_diagonal].sum()
    kappa = (pi.sum() - rho_hat) / ((target - s) ** 2).sum()
    delta, average = op.constant_correlation_shrinkage(x)
    assert delta == pytest.approx(min(max(kappa / t, 0), 1)) and 0 < delta < 1
    assert average == pytest.approx(rho)


def test_risk_based_family_has_the_mdp_and_minimum_variance_at_its_ends():
    r = simulate(300, 30)
    for cap in (None, 0.08):
        assert op.risk_based(r, cap, exponent=1) == pytest.approx(op.max_diversification(r, cap))
        assert op.risk_based(r, cap, exponent=0) == pytest.approx(op.min_variance(r, cap))


@pytest.mark.parametrize("target", ["identity", "constant correlation"])
@pytest.mark.parametrize("exponent", [1.0, 0.5, 0.0])
def test_risk_based_matches_a_general_purpose_solver(target, exponent):
    r = simulate(300, 30)
    vol = r.std(axis=0, ddof=1)
    cov = shrunk_covariance(r) if target == "identity" else shrunk_covariance_constant_correlation(r)
    ratio = lambda v: (v @ vol ** exponent) / np.sqrt(v @ cov @ v)
    reference = minimize(lambda v: -ratio(v), np.full(30, 1 / 30), bounds=[(0, 0.08)] * 30,
                         method="SLSQP", constraints={"type": "eq", "fun": lambda v: v.sum() - 1},
                         options={"ftol": 1e-13, "maxiter": 1000}).x
    w = op.risk_based(r, 0.08, exponent=exponent, target=target)
    assert w.sum() == pytest.approx(1) and w.min() >= 0 and w.max() <= 0.08 + 1e-6
    assert ratio(w) >= ratio(reference) - 1e-6
    assert np.abs(w - reference).max() < 1e-3


# ---------------------------------------------------------------------------
# Universe: who was in the index
# ---------------------------------------------------------------------------
ORIGINAL = """date,tickers
2018-01-02,"AAA,AGN,AGN-201503,PX,PX-201810,SCG,SCG-201812"
2018-06-01,"AAA,AGN,PX,PX-201810,SCG,SCG-201812"
2018-10-31,"AAA,AGN,SCG-201812"
2019-01-02,"AAA,AGN"
"""
UPDATED = """date,tickers
2018-10-31,"AAA,AGN,LIN,SCG"
2019-01-02,"AAA,AGN,LIN"
2019-06-03,"AAA,FB,LIN"
2022-06-09,"AAA,META,LIN"
"""


def joined_files():
    return universe.members_by_date(universe.read_file(ORIGINAL), universe.read_file(UPDATED))


def test_second_name_of_a_delisted_company_is_recognised():
    same = universe.duplicated_codes(universe.read_file(ORIGINAL))
    # SCG is never seen after December 2018: it is SCG-201812 written twice.
    # AGN is still a member in 2019: it is another company than AGN-201503.
    assert same == {"SCG": "SCG-201812", "PX": "PX-201810"}


def test_files_are_joined_without_duplicates():
    rows = joined_files()
    assert rows[pd.Timestamp("2018-01-02")] == {"AAA", "AGN", "AGN-201503", "PX-201810", "SCG-201812"}
    # the first file is used up to its last date; the patch adds Linde until then
    assert rows[pd.Timestamp("2018-10-31")] == {"AAA", "AGN", "SCG-201812", "LIN"}
    assert rows[pd.Timestamp("2019-06-03")] == {"AAA", "FB", "LIN"}
    assert all(len(codes) == len(set(codes)) for codes in rows.values())


def test_stays_in_the_index():
    stays = universe.membership_intervals(joined_files(), since="2018-01-01").set_index("code")
    assert stays.at["AGN-201503", "end"] == pd.Timestamp("2018-06-01")
    assert stays.at["AGN", "end"] == pd.Timestamp("2019-06-03")
    assert pd.isna(stays.at["AAA", "end"])
    # stays that ended before `since` are left out
    late = universe.membership_intervals(joined_files(), since="2018-07-01")
    assert "AGN-201503" not in set(late.code)


def test_codes_of_the_same_company_are_one_security():
    days = pd.bdate_range("2018-01-02", "2022-12-30")
    member = universe.membership(universe.membership_intervals(joined_files(), "2018-01-01"), days)
    assert "FB" not in member and "PX-201810" not in member
    assert member["META"].loc["2019-06-03":].all() and not member["META"].loc[:"2019-05-31"].any()
    # Praxair became Linde: one security without a gap
    assert member["LIN"].all()
    assert member.sum(axis=1).loc["2019-06-03":].eq(3).all()


def test_a_code_that_means_two_companies_is_cut(monkeypatch):
    monkeypatch.setattr(config, "SAME_COMPANY", [("IR", "TT", "2020-02-28")])
    stays = pd.DataFrame({"code": ["IR"], "start": [pd.Timestamp("2019-01-02")], "end": [pd.NaT]})
    days = pd.bdate_range("2019-01-02", "2020-12-31")
    member = universe.membership(stays, days)
    assert member["TT"].loc[:"2020-02-28"].all() and not member["TT"].loc["2020-03-02":].any()
    assert member["IR"].loc["2020-03-02":].all() and not member["IR"].loc[:"2020-02-28"].any()
    by_security = universe.security_stays(stays).set_index("security")
    assert by_security.at["TT", "end"] == pd.Timestamp("2020-02-29") and pd.isna(by_security.at["IR", "end"])


def test_codes_of_delisted_companies_and_yahoo_tickers():
    assert universe.is_delisted_code("LEHMQ-201203") and not universe.is_delisted_code("BRK.B")
    assert universe.yahoo_ticker("BRK.B") == "BRK-B"


HOLDINGS = """iShares Core S&P 500 ETF
Fund Holdings as of,"Oct 08, 2026"
\xa0
Ticker,Name,Sector,Asset Class,Market Value
"NVDA","NVIDIA","Information Technology","Equity","1.00"
"BRK B","BERKSHIRE HATHAWAY INC CLASS B","Financials","Equity","1.00"
"GOOGL","ALPHABET CLASS A","Communication","Equity","1.00"
"USD","USD CASH","Cash and/or Derivatives","Cash","1.00"
"""


def test_holdings_file_is_parsed():
    table, as_of = dl.parse_holdings(HOLDINGS)
    assert as_of == "Oct 08, 2026"
    assert table.security.tolist() == ["NVDA", "BRK.B", "GOOGL"]
    assert table.sector.tolist() == ["Information Technology", "Financials", "Communication Services"]


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
PAGE = """<html><head><title>Precios históricos de APPLE  - Bolsamania.com</title></head><body><table>
<tr><th>Fecha</th><th>Precio</th><th>Var.</th><th>Máximo</th><th>Mínimo</th><th>Apertura</th></tr>
<tr><td class="text-left">05-jun-17</td><td class="text-right">1.153,930$</td><td class="text-right"><span>-0,98%</span></td>
<td class="text-right">154,450$</td><td class="text-right">153,460$</td><td class="text-right">154,340$</td></tr>
<tr><td class="text-left">06-jun-17</td><td class="text-right">154,450$</td><td class="text-right"><span>0,34%</span></td>
<td class="text-right">155,980$</td><td class="text-right">153,340$</td><td class="text-right">-</td></tr>
</table></body></html>"""


def test_price_page_is_parsed():
    table = bolsamania.parse(PAGE)
    assert table.index.tolist() == [pd.Timestamp("2017-06-05"), pd.Timestamp("2017-06-06")]
    assert table.loc["2017-06-05"].tolist() == [1153.93, 154.45, 153.46, 154.34]
    assert np.isnan(table.at[pd.Timestamp("2017-06-06"), "open"])
    assert bolsamania.currency(PAGE) == "$"
    assert bolsamania.currency(PAGE.replace("$", "&euro;")) == "EUR"


def test_pages_of_one_company_are_joined_on_trading_days(monkeypatch):
    days = pd.DatetimeIndex(["2020-07-01", "2020-07-02", "2020-07-06", "2020-07-07", "2020-07-08"])
    pages = {
        # old exchange: ends on 2 July. 3 July is a US holiday: the page repeats the last close.
        "OLD": pd.Series([10.0, 11.0, 11.0], index=pd.to_datetime(["2020-07-01", "2020-07-02", "2020-07-03"])),
        # new exchange: starts on 2 July with another price for that day; 7 July is written on the 8th
        # of a week in which the 8th has its own price, and Friday 10 July is written on Saturday.
        "NEW": pd.Series([11.5, 12.0, 14.0], index=pd.to_datetime(["2020-07-02", "2020-07-06", "2020-07-08"])),
    }
    monkeypatch.setattr(bolsamania, "load", lambda page: pd.DataFrame({"close": pages[page]}))
    close = dp.traded_close(["OLD", "NEW"], days)
    assert close.tolist()[:3] == [10.0, 11.5, 12.0]          # the later page wins on the common day
    assert np.isnan(close["2020-07-07"]) and close["2020-07-08"] == 14.0
    # a price written on a day without trading goes to the previous trading day if it is empty
    pages["NEW"][pd.Timestamp("2020-07-11")] = 15.0
    longer = days.append(pd.DatetimeIndex(["2020-07-09", "2020-07-10", "2020-07-13"]))
    close = dp.traded_close(["OLD", "NEW"], longer)
    assert close["2020-07-10"] == 15.0 and np.isnan(close["2020-07-13"])
    assert dp.traded_close([], days).isna().all()


def test_holes_are_bridged_with_the_market():
    days = pd.bdate_range("2021-01-04", periods=6)
    market = pd.Series([100.0, 110.0, 99.0, 99.0, 108.9, 108.9], index=days)
    price = pd.DataFrame({"HOLE": [50.0, np.nan, np.nan, 49.5, 51.0, np.nan],
                          "HALT": [20.0, np.nan, 21.0, 21.0, np.nan, np.nan]}, index=days)
    filled = dp.bridge(price, market, ["HOLE"])
    # during the hole the stock moves with the market, and the return over the hole is the true one
    assert filled["HOLE"].tolist()[:4] == pytest.approx([50.0, 55.0, 49.5, 49.5])
    assert filled["HOLE"].iloc[3] / filled["HOLE"].iloc[0] == pytest.approx(49.5 / 50.0)
    # the other source repeats the last price, and nothing is added after the last price
    assert filled["HALT"].tolist()[:4] == [20.0, 20.0, 21.0, 21.0]
    assert np.isnan(filled["HOLE"].iloc[5]) and filled["HALT"].iloc[4:].isna().all()


def test_wrong_adjustments_of_yahoo_are_undone():
    days = pd.bdate_range("2016-06-27", periods=6)
    close = pd.DataFrame({"DIV": [100.0, 100.0, 70.0, 70.0, 70.0, 70.0],        # special dividend of 30
                          "ERR": [68.0, 68.0, 70.4, 70.4, 70.4, 70.4],          # +3.5%, wrongly adjusted
                          "CRASH": [100.0, 100.0, 60.0, 60.0, 60.0, 60.0]}, index=days)
    adjusted = close.copy()
    adjusted.loc[days[:2], "DIV"] = 70.0                     # no loss for the holder: adjustment is right
    adjusted.loc[days[:2], "ERR"] = 42.0                     # a gain of 68%: cannot be right
    adjusted.loc[days[:2], "CRASH"] = 99.0                   # ordinary dividend the day of a real fall
    checked, rejected = dp.checked_adjusted_close(adjusted, close)
    r = checked.pct_change().iloc[2]
    assert r["DIV"] == pytest.approx(0.0)
    assert r["ERR"] == pytest.approx(70.4 / 68.0 - 1)
    assert r["CRASH"] == pytest.approx(60.0 / 99.0 - 1)
    assert rejected.security.tolist() == ["ERR"] and rejected.date.tolist() == [days[2]]
    # prices from the wrong day on are not touched
    assert checked.iloc[2:].equals(adjusted.iloc[2:])


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
DESIGN = wf.Design(train_days=120, test_days=20, max_weight=0.2)


def make_panel(returns, member=None, listed=None):
    dates = pd.bdate_range("2015-01-01", periods=len(returns))
    names = [f"S{i:02d}" for i in range(returns.shape[1])]
    frame = pd.DataFrame(returns, index=dates, columns=names)
    if member is None:
        member = np.ones(returns.shape, dtype=bool)
    flat = pd.Series(0.0, index=dates)
    companies = pd.DataFrame({"name": names, "sector": "Industrials", "source": "Yahoo Finance",
                              "has_prices": True, "listed": True if listed is None else listed},
                             index=names)
    return dp.Panel(frame, pd.DataFrame(member, index=dates, columns=names), flat, flat, flat,
                    companies, pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(index=dates))


@pytest.fixture
def start(monkeypatch):
    """The backtest of the simulated panels starts when the first full window is available.

    The first row of a table of prices has no return, so a window of 120 returns ends on row 120.
    """
    dates = pd.bdate_range("2015-01-01", periods=400)
    monkeypatch.setattr(config, "OOS_START", str(dates[DESIGN.train_days + 1].date()))


def test_optimisation_days(start, monkeypatch):
    dates = pd.bdate_range("2015-01-01", periods=400)
    days = wf.optimisation_days(dates, DESIGN)
    assert days[0] == DESIGN.train_days and (np.diff(days) == DESIGN.test_days).all()
    assert days[-1] < len(dates) - 1 - config.EXECUTION_LAG
    # a longer window starts later, on a date of the same calendar of rebalances
    longer = wf.optimisation_days(dates, DESIGN.changed(train_days=150))
    assert longer[0] >= 150 and (longer[0] - days[0]) % DESIGN.test_days == 0
    assert set(longer) <= set(days)


def test_walk_forward_holds_the_optimised_weights(start):
    r = simulate(400, 15)
    panel = make_panel(r)
    result = wf.run(panel, DESIGN)
    fit_days = wf.optimisation_days(panel.returns.index, DESIGN)
    assert len(result.windows) == len(fit_days) * 3
    # every out-of-sample day belongs to exactly one window
    assert result.gross.index.is_unique
    assert result.gross.index[0] == panel.returns.index[fit_days[0] + config.EXECUTION_LAG + 1]
    # the first window reproduced by hand: weights from the in-sample data only, bought one
    # day later and held without rebalancing
    f = fit_days[0]
    w = op.max_diversification(r[f - DESIGN.train_days + 1:f + 1], DESIGN.max_weight)
    held = r[f + 2:f + 2 + DESIGN.test_days]
    value = np.cumprod(1 + held, axis=0) @ w
    by_hand = np.diff(np.concatenate([[1.0], value])) / np.concatenate([[1.0], value[:-1]])
    assert result.gross["MDP"].iloc[:DESIGN.test_days].to_numpy() == pytest.approx(by_hand)
    assert result.weights["MDP"].iloc[0].to_numpy() == pytest.approx(w)
    first = result.windows[result.windows.strategy == "MDP"].iloc[0]
    assert first.dr_in == pytest.approx(op.realised_diversification_ratio(w, r[f - DESIGN.train_days + 1:f + 1]))
    assert first.dr_out == pytest.approx(op.realised_diversification_ratio(w, held))


def test_future_returns_do_not_change_the_weights(start):
    r = simulate(400, 12)
    changed = r.copy()
    f = wf.optimisation_days(pd.bdate_range("2015-01-01", periods=400), DESIGN)[3]
    changed[f + 1:] = simulate(400, 12, seed=9)[f + 1:]
    a = wf.run(make_panel(r), DESIGN, ("MDP",)).weights["MDP"]
    b = wf.run(make_panel(changed), DESIGN, ("MDP",)).weights["MDP"]
    assert a.iloc[:4].to_numpy() == pytest.approx(b.iloc[:4].to_numpy())
    assert not np.allclose(a.iloc[4:].to_numpy(), b.iloc[4:].to_numpy())


def test_only_index_members_with_history_are_eligible(start):
    r = simulate(400, 10)
    member = np.ones(r.shape, dtype=bool)
    member[:, 0] = False                                     # never in the index
    member[250:, 1] = False                                  # leaves the index on day 250
    r[:200, 2] = np.nan                                      # listed on day 200: has to wait
    r[300:, 3] = np.nan                                      # stops trading on day 300
    panel = make_panel(r, member)
    result = wf.run(panel, DESIGN, ("EW",))
    w = result.weights["EW"]
    days = panel.returns.index
    assert (w["S00"] == 0).all()
    assert (w.loc[days[250]:, "S01"] == 0).all() and (w.loc[:days[249], "S01"] > 0).all()
    waiting = w.loc[:days[200 + int(DESIGN.train_days * config.MIN_COVERAGE) - 2], "S02"]
    assert (waiting == 0).all() and (w["S02"].iloc[-1] > 0)
    assert (w.loc[days[300]:, "S03"] == 0).all()
    # the position of the stock that stops trading is frozen at its last price
    assert np.isfinite(result.gross["EW"]).all()


def test_a_price_that_never_moves_is_reported(start):
    r = simulate(400, 8)
    r[:, 5] = 0.0
    with pytest.raises(ValueError, match="S05"):
        wf.run(make_panel(r), DESIGN, ("EW",))


def test_survivors_only_leaves_out_the_companies_that_disappeared(start):
    r = simulate(400, 10)
    listed = np.array([True] * 10)
    listed[[3, 7]] = False
    panel = make_panel(r, listed=listed)
    everyone = wf.run(panel, DESIGN, ("EW",)).weights["EW"]
    survivors = wf.run(panel, DESIGN.changed(survivors_only=True), ("EW",)).weights["EW"]
    assert (everyone[["S03", "S07"]] > 0).all().all()
    assert (survivors[["S03", "S07"]] == 0).all().all()
    assert survivors.sum(axis=1).to_numpy() == pytest.approx(1.0)


def test_trading_costs(start):
    result = wf.run(make_panel(simulate(400, 10)), DESIGN, ("MDP",))
    free, costly = result.net(cost_bps=0.0)["MDP"], result.net(cost_bps=10.0)["MDP"]
    w = result.windows
    first_days = pd.DatetimeIndex(w.first_day)
    # the first trade buys the whole portfolio: 100% of it is traded, at 10 basis points
    assert w.traded.iloc[0] == pytest.approx(1.0)
    assert (1 + costly[first_days[0]]) / (1 + free[first_days[0]]) == pytest.approx(1 - 0.001)
    assert (costly[first_days] < free[first_days]).all()
    others = free.index.difference(first_days)
    assert costly[others].equals(free[others])
    assert free.equals(result.gross["MDP"])


# ---------------------------------------------------------------------------
# Choosing a design without looking at the future
# ---------------------------------------------------------------------------
def test_holding_given_weights_reproduces_the_run(start):
    panel = make_panel(simulate(400, 12))
    result = wf.run(panel, DESIGN, ("MDP",))
    again = wf.hold_targets(panel, DESIGN, result.weights["MDP"], "MDP")
    assert again.gross["MDP"].to_numpy() == pytest.approx(result.gross["MDP"].to_numpy())
    assert again.gross.index.equals(result.gross.index)
    assert again.windows.traded.to_numpy() == pytest.approx(result.windows.traded.to_numpy())
    assert again.windows.dr_out.to_numpy() == pytest.approx(result.windows.dr_out.to_numpy(), nan_ok=True)
    assert again.net(10.0)["MDP"].to_numpy() == pytest.approx(result.net(10.0)["MDP"].to_numpy())


def test_the_choice_of_design_uses_only_the_past():
    days = pd.bdate_range("2015-01-01", periods=300)
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.01, 300)
    records = pd.DataFrame({"A": noise + 0.002, "B": noise - 0.002}, index=days)
    records.loc[days[150]:, ["A", "B"]] = records.loc[days[150]:, ["B", "A"]].to_numpy()   # B is better afterwards
    dates = days[20::20]
    chosen = sel.choices(records, dates, default="B", burn_in=2)
    assert chosen.iloc[:2].tolist() == ["B", "B"]                    # no record yet: the default
    assert (chosen.loc[dates[2]:days[150]] == "A").all()             # A has the best record so far
    assert chosen.iloc[-1] == "A" or (chosen.loc[days[250]:] == "B").any()
    # what happens after a date does not change the choice made on it
    changed = records.copy()
    changed.loc[days[150]:] = rng.normal(0, 0.05, (150, 2))
    assert sel.choices(changed, dates, "B", 2).loc[:days[149]].equals(chosen.loc[:days[149]])
    # a candidate that starts later is compared on the days that both have
    late = records.copy()
    late.loc[:days[59], "A"] = np.nan
    assert sel.choices(late, dates, "B", 2).loc[dates[2]] == "B"     # only the default has a record
    assert sel.choices(late, dates, "B", 2).loc[days[120]] == "A"


def test_the_first_choice_waits_the_same_time_whatever_the_holding_period(monkeypatch):
    monkeypatch.setattr(config, "SELECTION_BURN_IN_YEARS", 3)
    assert sel.burn_in_windows(wf.Design(test_days=252)) == 3
    assert sel.burn_in_windows(wf.Design(test_days=63)) == 12
    assert sel.burn_in_windows(wf.Design(test_days=100)) == 8      # rounded up: at least 3 years


def test_adaptive_and_ensemble_portfolios(start):
    from functools import partial
    panel = make_panel(simulate(400, 12))
    results = {name: wf.run(panel, DESIGN, {name: partial(op.risk_based, exponent=exponent)})
               for name, exponent in [("MDP", 1.0), ("MinVar", 0.0)]}
    adaptive, chosen = sel.adaptive(panel, DESIGN, results, default="MDP", burn_in=3)
    assert (chosen.iloc[:3] == "MDP").all() and set(chosen) <= {"MDP", "MinVar"}
    held = adaptive.weights["Adaptive"]
    for day, name in chosen.items():
        assert held.loc[day].to_numpy() == pytest.approx(results[name].weights[name].loc[day].to_numpy())
    # while the same candidate is held, the returns are those of that candidate
    same = adaptive.gross["Adaptive"].loc[:adaptive.windows.last_day.iloc[2]]
    assert same.to_numpy() == pytest.approx(results["MDP"].gross["MDP"].loc[same.index].to_numpy())
    mix = sel.ensemble(panel, DESIGN, results)
    average = (results["MDP"].weights["MDP"] + results["MinVar"].weights["MinVar"]) / 2
    assert mix.weights["Ensemble"].to_numpy() == pytest.approx(average.to_numpy())
    assert mix.weights["Ensemble"].sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert np.isfinite(mix.gross["Ensemble"]).all()


def test_interval_of_the_difference_between_two_sharpe_ratios():
    days = pd.bdate_range("2010-01-01", periods=2000)
    rng = np.random.default_rng(5)
    common = rng.normal(0.0004, 0.01, 2000)
    a = pd.Series(common + rng.normal(0, 0.002, 2000), index=days)
    flat = pd.Series(0.0, index=days)
    same = metrics.sharpe_difference(a, a, flat, draws=200)
    assert same["difference"] == 0 and same["low"] == 0 and same["high"] == 0
    better = metrics.sharpe_difference(a + 0.0005, a, flat, draws=500)      # 12.6% more per year
    assert better["difference"] > 0 and better["low"] > 0
    unclear = metrics.sharpe_difference(a, pd.Series(common + rng.normal(0, 0.002, 2000), index=days), flat, draws=500)
    assert unclear["low"] < 0 < unclear["high"]
    assert metrics.sharpe_difference(a + 0.0005, a, flat, draws=500) == better      # same draws every time
