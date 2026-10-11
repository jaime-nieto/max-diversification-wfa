"""
Walk-forward analysis of the Most Diversified Portfolio on the S&P 500,
with the index members of each date, including the ones that no longer trade.

    python run_all.py            # uses the cached data in data/raw (works offline)
    python run_all.py --refresh  # re-downloads everything first
    python run_all.py --quick    # skips the robustness runs and the study of the Sharpe ratio

Every table goes to results/tables/, every chart to results/figures/, and all
tables are collected in results/summary.md.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from maxdiv import config, metrics
from maxdiv import optimizer as op
from maxdiv import plotting as pl
from maxdiv import selection as sel
from maxdiv import walkforward as wf
from maxdiv.data_download import download_all
from maxdiv.data_prep import load_panel
from maxdiv.reporting import LABELS, num, pct, save_table, write_summary

OPTIMISED = ["MDP", "MinVar", "EW"]
BENCHMARKS = ["SPY", "RSP"]


# ---------------------------------------------------------------------------
# Step 0. Data checks: who is missing, where the prices come from, what was adjusted
# ---------------------------------------------------------------------------
def coverage_on(panel, design, day):
    """Index members on one trading day (row number): source of their prices and how many are eligible."""
    R = panel.returns.to_numpy()
    member = panel.member.to_numpy()[day]
    priced = member & np.isfinite(R[day])
    window = R[max(day - design.train_days + 1, 0):day + 1]
    eligible = wf.eligible_stocks(window, member) if day >= design.train_days else priced & False
    delisted = (panel.companies["source"] == "bolsamania.com").to_numpy()
    return {"members": int(member.sum()), "priced": int(priced.sum()),
            "delisted": int((priced & delisted).sum()), "eligible": int(eligible.sum()),
            "without_prices": sorted(panel.returns.columns[member & ~priced])}


def held_stock_returns(panel, result, strategy):
    """Every daily return of every stock while it was held: date, ticker, ret, weight."""
    w = result.windows[result.windows.strategy == strategy]
    parts = []
    for (_, target), first, last in zip(result.weights[strategy].iterrows(),
                                        w["first_day"], w["last_day"]):
        held = target[target > 0]
        r = panel.returns.loc[first:last, held.index].stack().dropna()
        r.index.names = ["date", "ticker"]
        frame = r.rename("ret").reset_index()
        frame["weight"] = frame["ticker"].map(held)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def step0_data_checks(panel, base):
    dates, design, info = panel.returns.index, base.design, panel.companies
    names = info["name"]
    fit_days = wf.optimisation_days(dates, design)
    first = fit_days[0]
    R, M = panel.returns.to_numpy(), panel.member.to_numpy()
    priced = M & np.isfinite(R)
    coverage = priced.sum(axis=1) / M.sum(axis=1)

    # Why the backtest starts where it starts
    earliest = design.train_days                     # first day with a full in-sample window
    low = np.flatnonzero(coverage < config.MIN_INDEX_COVERAGE)
    low = low[low >= earliest]
    if len(low) and low[-1] >= first:
        raise RuntimeError(f"Only {pct(coverage[low[-1]])} of the index members have prices on "
                           f"{dates[low[-1]].date()}, below the minimum of "
                           f"{pct(config.MIN_INDEX_COVERAGE, 0)}. config.OOS_START is too early.")
    rows = []
    for day in sorted(set(range(earliest, first, 63)) | {first}):
        c = coverage_on(panel, design, day)
        rows.append({"Date": dates[day].date(), "Index members": c["members"],
                     "With prices": c["priced"], "Without prices": c["members"] - c["priced"],
                     "Coverage": pct(c["priced"] / c["members"])})
    last_low = f"The last day below that level is {dates[low[-1]].date()}." if len(low) else ""
    save_table("t0a_start_date", f"Data check: why the backtest starts on {dates[first].date()}",
               pd.DataFrame(rows),
               note=f"Prices of delisted companies start on {dates[0].date()}, so the first date with a "
                    f"full in-sample window of {design.train_days} days is {dates[earliest].date()}. "
                    f"The backtest starts on the first date from which at least "
                    f"{pct(config.MIN_INDEX_COVERAGE, 0)} of the index members have prices on every "
                    f"trading day until the end of the sample. {last_low} The members without prices "
                    "are companies that no longer trade and for which no free price history was "
                    "found; most of the ones missing at the earliest dates stopped trading before "
                    "mid-2009.")

    # Where the prices come from at the optimisation dates
    rows = []
    shown = fit_days[::max(1, len(fit_days) // 9)]           # about ten dates
    for day in sorted(set(shown) | {fit_days[-1]}):
        c = coverage_on(panel, design, day)
        rows.append({"Optimisation date": dates[day].date(), "Index members": c["members"],
                     "Prices from Yahoo Finance": c["priced"] - c["delisted"],
                     "Prices from bolsamania.com": c["delisted"],
                     "Without prices": c["members"] - c["priced"],
                     "Coverage": pct(c["priced"] / c["members"]), "Eligible": c["eligible"]})
    after = coverage[first:]
    save_table("t0b_universe", f"Data check: index members at the optimisation dates ({len(shown)} of {len(fit_days)} shown)",
               pd.DataFrame(rows),
               note=f"Checked on all {len(after)} trading days from {dates[first].date()}: the coverage "
                    f"is never below {pct(after.min())} and is {pct(after.mean())} on average. "
                    "'Prices from bolsamania.com' are the members that have since left the stock "
                    "market. A member is eligible once it has returns for "
                    f"{pct(config.MIN_COVERAGE, 0)} of the {design.train_days} in-sample days, so a "
                    "recently listed company (and a delisted one whose page has only part of its "
                    "history) has to wait.")

    # The members that are missing
    rows = []
    for k in np.flatnonzero((M & ~np.isfinite(R))[first:].sum(axis=0) > 5):
        inside = M[first:, k]
        missing = inside & ~np.isfinite(R[first:, k])
        days_in = dates[first:][inside]
        security = panel.returns.columns[k]
        rows.append({"Company": names[security], "Code": security,
                     "In the index (within the backtest)": f"{days_in[0].date()} to {days_in[-1].date()}",
                     "Days in the index": int(inside.sum()), "Days without a price": int(missing.sum()),
                     "Why": "no price history found" if info.at[security, "source"] == "none"
                            else "the page found has only part of its history"})
    table = pd.DataFrame(rows).sort_values("Days without a price", ascending=False)
    save_table("t0c_missing", "Data check: index members without prices during the backtest", table,
               note="These companies cannot be held by any portfolio while their prices are missing. "
                    "All of them have left the stock market (takeovers, mostly), so leaving them "
                    "out is a survivorship bias of the kind this project tries to avoid; it "
                    f"affects at most {pct(1 - after.min())} of the index members on any day.")

    # The delisted members that do have prices, and the holes in their source
    delisted = info.index[info["source"] == "bolsamania.com"]
    in_index = panel.member[delisted] & panel.returns[delisted].notna()
    bridged = panel.bridged[delisted] & in_index
    by_year = lambda t: t.loc[dates[first]:].sum(axis=1).groupby(lambda d: d.year).sum()
    rows = [{"Year": year, "Delisted members with prices (average per day)":
             f"{in_index.loc[str(year)].loc[dates[first]:].sum(axis=1).mean():.0f}",
             "Days without a price in the source": pct(b / n) if n else "–"}
            for (year, n), b in zip(by_year(in_index).items(), by_year(bridged))]
    listing = pd.DataFrame({"security": delisted, "name": names[delisted].to_numpy(),
                            "sector": info.loc[delisted, "sector"].to_numpy(),
                            "first_price": [panel.returns[s].first_valid_index().date() for s in delisted],
                            "last_price": [panel.returns[s].last_valid_index().date() for s in delisted],
                            "days_in_index_in_backtest": in_index.loc[dates[first]:].sum().to_numpy(),
                            "share_of_them_bridged": (bridged.loc[dates[first]:].sum()
                                                      / in_index.loc[dates[first]:].sum().replace(0, np.nan)).round(4).to_numpy()})
    save_table("t0d_delisted", "Data check: the delisted members that have prices, and the holes in their source",
               pd.DataFrame(rows), raw=listing,
               note=f"{int((in_index.loc[dates[first]:].sum() > 0).sum())} companies that no longer trade "
                    "were in the index during the backtest and have prices from bolsamania.com "
                    "(listed in t0d_delisted.csv). That source has holes: days on which the stock "
                    "traded but the page has no price, mostly in 2022 and 2023. During a hole the "
                    "stock is moved with the S&P 500 fund and the day its price returns takes the "
                    "rest of the move, so the return over the hole is the true one "
                    "(see `bridge` in maxdiv/data_prep.py). Ordinary cash dividends of these "
                    "companies are not included because no free source has them, so their return "
                    "is understated.")

    rows = [{"Company": names[s], "Date": d, "Prices before that date multiplied by": f"{f:.4g}",
             "What happened": what} for s, d, f, what in config.ADJUSTMENTS]
    rows += [{"Company": names[s], "Date": d, "Prices before that date multiplied by": "–",
              "What happened": f"{what} (return of that day: {pct(r, 0)})"}
             for s, d, r, what in config.TERMINAL_RETURNS]
    save_table("t0e_corporate_actions", "Data check: corporate actions adjusted by hand in delisted companies",
               pd.DataFrame(rows),
               note="Found as jumps in the traded prices and matched to the corporate action that "
                    "explains each one. For spin-offs and special dividends the factor is the "
                    "opening price after the event divided by the previous close.")

    rows = [{"Date": "whole sample", "Company": names[s], "Return in Yahoo's adjusted prices": "–",
             "Return used": "–", "Why": f"Adjusted prices not used (close without dividends instead): {why}"}
            for s, why in config.YAHOO_WITHOUT_DIVIDENDS]
    rows += [{"Date": r.date.date(), "Company": names[r.security],
              "Return in Yahoo's adjusted prices": f"{100 * r.yahoo:+.0f}%",
              "Return used": f"{100 * r.used:+.0f}%", "Why": r.why}
             for r in panel.rejected.itertuples()]
    save_table("t0f_yahoo_corrections", "Data check: prices of Yahoo Finance that were corrected",
               pd.DataFrame(rows), raw=panel.rejected,
               note="Yahoo's adjusted prices are right on the day of an ordinary dividend or split, "
                    "but a few spin-offs and mergers are adjusted twice, with the wrong ratio or "
                    "not at all. They were found by comparing the adjusted price with the "
                    "unadjusted one and by looking at the largest one-day moves; others, smaller, "
                    "may remain.")

    if len(panel.removed):
        removed = panel.removed.sort_values(ascending=False)
        rows = [{"Date": d.date(), "Company": names[s], "One-day return": f"+{100 * v:,.0f}%"}
                for (d, s), v in removed.items()]
        save_table("t0g_removed_returns", "Data check: one-day gains set to missing", pd.DataFrame(rows),
                   note=f"Returns above +{pct(config.MAX_DAILY_RETURN, 0)} are share consolidations "
                        "for which no adjustment is available; the return of that day is set to 0.")

    # Largest single-stock moves while the stock was in the MDP, so that anyone can audit them
    moves = held_stock_returns(panel, base, "MDP")
    extremes = pd.concat([moves.nlargest(5, "ret"), moves.nsmallest(5, "ret")])
    rows = [{"Date": m.date.date(), "Company": names[m.ticker],
             "Prices from": info.at[m.ticker, "source"],
             "One-day return": f"{100 * m.ret:+.0f}%",
             "Weight at the last rebalance": pct(m.weight, 1),
             "Effect on the portfolio (approx.)": f"{100 * m.ret * m.weight:+.2f}%"}
            for m in extremes.itertuples()]
    save_table("t0h_largest_moves_held", "Data check: five largest one-day gains and losses of a stock held by the MDP",
               pd.DataFrame(rows), raw=extremes,
               note="Each of these should match a real company event. A data error or a "
                    "corporate action that was not adjusted would show up here.")


# ---------------------------------------------------------------------------
# Step 1. Walk-forward: does the optimised diversification ratio survive out-of-sample?
# ---------------------------------------------------------------------------
def step1_walk_forward(panel, base):
    summary = metrics.walk_forward_summary(base.windows)
    rows = [{"Portfolio": LABELS.get(s, s), "Windows": int(r.windows),
             "DR in-sample (average)": num(r.dr_in), "DR out-of-sample (average)": num(r.dr_out),
             "Walk-forward efficiency (out / in)": pct(r.efficiency, 0),
             "Lowest DR out-of-sample": num(r.dr_out_min),
             "Highest DR out-of-sample": num(r.dr_out_max),
             "DR in-sample, in blocks": num(r.dr_in_blocks),
             "Efficiency, like for like": pct(r.efficiency_like_for_like, 0)}
            for s, r in summary.iterrows()]
    design = base.design
    save_table("t1a_walk_forward", "Walk-forward: diversification ratio in-sample and out-of-sample",
               pd.DataFrame(rows), raw=summary.reset_index(),
               note=f"Each window: weights fitted on {design.train_days} trading days, then held "
                    f"for the next {design.test_days}. Both DR columns use the plain sample "
                    "covariance of the window they refer to and the weights set at the start of "
                    "the out-of-sample window. All three portfolios use the same eligible stocks. "
                    "The last, incomplete window is excluded. 'In blocks': the in-sample window "
                    f"is cut into consecutive blocks of {design.test_days} days and the DR of "
                    "each block is averaged, so that in-sample and out-of-sample are measured "
                    "over the same horizon; 'like for like' divides the out-of-sample DR by it.")

    rows, raw = [], []
    for other in ["EW", "MinVar"]:
        c = metrics.paired_comparison(base.windows, "MDP", other)
        raw.append({"other": other, **c})
        rows.append({"MDP compared with": LABELS.get(other, other), "Windows": c["windows"],
                     "Average difference in out-of-sample DR": f"{c['mean_difference']:+.2f}",
                     "t-statistic": num(c["t_stat"]),
                     "Windows in which the MDP is higher": pct(c["share_higher"], 0)})
    save_table("t1b_paired", "Walk-forward: out-of-sample DR of the MDP against alternatives, window by window",
               pd.DataFrame(rows), raw=pd.DataFrame(raw),
               note="Paired t-test on non-overlapping windows.")
    base.windows.to_csv(config.TABLE_DIR / "windows.csv", index=False)
    pl.plot_design(base.windows, panel.returns.index, base.design)
    pl.plot_walk_forward(base.windows)


# ---------------------------------------------------------------------------
# Step 2. Out-of-sample performance after costs
# ---------------------------------------------------------------------------
def returns_with_benchmark(panel, result, cost_bps=config.COST_BPS):
    net = result.net(cost_bps)
    net["SPY"] = panel.benchmark.reindex(net.index)
    net["RSP"] = panel.equal_weight.reindex(net.index)
    return net, panel.risk_free.reindex(net.index)


def idle_share(panel, result, strategy):
    """Average share of the portfolio in companies that no longer trade.

    When a company that is held is taken over, its position stays at the last
    price until the next rebalance: it is the cash received, and it earns
    nothing. This is the part of the portfolio in that state on an average day.
    """
    w = result.windows[result.windows.strategy == strategy]
    shares = []
    for (_, target), first, last in zip(result.weights[strategy].iterrows(), w["first_day"], w["last_day"]):
        held = target[target > 0]
        r = panel.returns.loc[first:last, held.index]
        value = (1 + r.fillna(0.0)).cumprod() * held
        shares.append((value * r.isna()).sum(axis=1) / value.sum(axis=1))
    return float(pd.concat(shares).mean())


def step2_performance(panel, base):
    net, rf = returns_with_benchmark(panel, base)
    rows, raw = [], []
    for s in OPTIMISED + BENCHMARKS:
        p = metrics.performance(net[s], rf, net["SPY"])
        row = {"Portfolio": LABELS.get(s, s), "Return per year": pct(p["cagr"]),
               "Before costs": "–", "Volatility": pct(p["volatility"]),
               "Sharpe ratio": num(p["sharpe"]), "Worst drawdown": pct(p["max_drawdown"]),
               "Beta to SPY": num(p["beta"]), "Turnover per year": "–", "Stocks held": "–",
               "Share that no longer trades": "–"}
        if s in OPTIMISED:
            w = base.windows[base.windows.strategy == s]
            gross = metrics.performance(base.gross[s], rf, net["SPY"])["cagr"]
            p.update(cagr_before_costs=gross, stocks_held=w["held"].mean(),
                     turnover=metrics.turnover_per_year(base.windows, s, len(net)),
                     idle_share=idle_share(panel, base, s))
            row.update({"Before costs": pct(gross), "Turnover per year": pct(p["turnover"], 0),
                        "Stocks held": f"{p['stocks_held']:.0f}",
                        "Share that no longer trades": pct(p["idle_share"])})
        rows.append(row)
        raw.append({"portfolio": s, **p})
    save_table("t2_performance", "Out-of-sample performance, after trading costs",
               pd.DataFrame(rows), raw=pd.DataFrame(raw),
               note=f"{net.index[0].date()} to {net.index[-1].date()}. Costs: {config.COST_BPS:g} bp "
                    "of every dollar bought or sold. Turnover is one-way (100% = the whole portfolio "
                    "replaced once a year). Sharpe ratio over 3-month Treasury bill (DTB3). "
                    "SPY is the S&P 500 cap-weighted ETF; RSP is the S&P 500 Equal Weight ETF. "
                    "'Share that no longer trades': part of the portfolio that, on an average day, is "
                    "in a company that has stopped trading (the cash of a takeover, mostly). It "
                    "stays at its last price, earning nothing, until the next rebalance.")
    net.to_csv(config.TABLE_DIR / "daily_returns.csv")
    pl.plot_performance(net)

    # Where the volatility of each portfolio comes from: the stocks it holds and how it combines them
    rows, raw = [], []
    for s in OPTIMISED:
        w = base.windows[(base.windows.strategy == s) & base.windows.dr_out.notna()]
        portfolio = np.array([base.gross.loc[a:b, s].std() * np.sqrt(config.TRADING_DAYS)
                              for a, b in zip(w.first_day, w.last_day)])
        stocks = portfolio * w.dr_out.to_numpy()             # DR = volatility of the stocks / of the portfolio
        raw.append({"portfolio": s, "stock_volatility": stocks.mean(), "dr_out": w.dr_out.mean(),
                    "portfolio_volatility": portfolio.mean()})
        rows.append({"Portfolio": LABELS[s], "Average volatility of the stocks held": pct(stocks.mean()),
                     "Diversification ratio": num(w.dr_out.mean()),
                     "Volatility of the portfolio": pct(portfolio.mean())})
    save_table("t2b_volatility", "Where the volatility comes from: the stocks held and how they are combined",
               pd.DataFrame(rows), raw=pd.DataFrame(raw),
               note="Averages of the out-of-sample windows (the volatility of a window is lower than "
                    "that of the whole period, which also has the changes from one window to the "
                    "next). The volatility of a portfolio is the average volatility of its stocks "
                    "divided by its diversification ratio. The MDP divides by the largest number, "
                    "but it starts from stocks that are more volatile than the average; minimum "
                    "variance starts from the calm ones.")
    return net


# ---------------------------------------------------------------------------
# Step 3. What the Most Diversified Portfolio holds
# ---------------------------------------------------------------------------
def step3_portfolio(panel, base):
    pl.plot_portfolio(base.windows)
    info = panel.companies
    weights = base.weights["MDP"]

    by_sector = weights.T.groupby(info["sector"]).sum().T
    members = panel.member.loc[weights.index].astype(float)
    index_share = members.T.groupby(info["sector"]).sum().T.div(members.sum(axis=1), axis=0)
    pl.plot_sectors(by_sector, index_share)
    table = pd.DataFrame({"mdp_average": by_sector.mean(), "mdp_latest": by_sector.iloc[-1],
                          "index_share": index_share.mean()}).sort_values("mdp_average", ascending=False)
    rows = [{"Sector": s, "MDP, average of all windows": pct(r.mdp_average),
             f"MDP on {weights.index[-1].date()}": pct(r.mdp_latest),
             "Share of index members (average)": pct(r.index_share)}
            for s, r in table.iterrows()]
    save_table("t3a_sectors", "Sector weights of the MDP (GICS classification)", pd.DataFrame(rows),
               raw=table.rename_axis("sector").reset_index(),
               note="11 official GICS sectors. The last column is the weight the sector has in the "
                    "equal-weight portfolio.")

    latest = weights.iloc[-1]
    latest = latest[latest > 0].sort_values(ascending=False)
    day = panel.returns.index.get_loc(weights.index[-1])
    in_sample = panel.returns.iloc[day - base.design.train_days + 1:day + 1][latest.index]
    vol = in_sample.std() * np.sqrt(config.TRADING_DAYS)
    rows = [{"Company": info.at[t, "name"], "Sector": info.at[t, "sector"],
             "Weight": pct(w), "In-sample volatility": pct(vol[t], 0)}
            for t, w in latest.items()]
    save_table("t3b_latest_holdings",
               f"The {len(latest)} holdings of the MDP optimised on {weights.index[-1].date()}",
               pd.DataFrame(rows),
               raw=pd.DataFrame({"ticker": latest.index, "weight": latest.values, "volatility": vol.values}))


# ---------------------------------------------------------------------------
# Step 4. Sub-periods
# ---------------------------------------------------------------------------
def step4_subperiods(base, net):
    rows = []
    for label, start, end in config.SUBPERIODS:
        w = base.windows[(base.windows.first_day >= start) & (base.windows.first_day <= end)]
        dr = w.groupby("strategy")["dr_out"].mean()
        r = net.loc[start:end]
        years = len(r) / config.TRADING_DAYS
        annual = (1 + r).prod() ** (1 / years) - 1
        worst = r.apply(lambda col: metrics.drawdown(col).min())
        rows.append({"Period": label,
                     **{f"DR: {s}": num(dr.get(s, np.nan)) for s in OPTIMISED},
                     **{f"Return: {s}": pct(annual[s]) for s in ["MDP", "EW", "SPY"]},
                     **{f"Drawdown: {s}": pct(worst[s], 0) for s in ["MDP", "EW", "SPY"]}})
    save_table("t4_subperiods", "Sub-periods: diversification ratio and performance",
               pd.DataFrame(rows),
               note="DR: out-of-sample, average over windows that start in the period. "
                    "Return: per year, after costs. Drawdown: worst fall from a previous peak.")


# ---------------------------------------------------------------------------
# Step 5. Survivorship bias: what a 'survivors only' data source would have shown
# ---------------------------------------------------------------------------
def step5_survivorship(panel, base, pool):
    survivors = wf.run(panel, base.design.changed(survivors_only=True), ("MDP", "EW"), pool)
    rf = panel.risk_free.reindex(base.gross.index)
    bench = panel.benchmark.reindex(base.gross.index)
    delisted = (panel.companies["has_prices"] & ~panel.companies["listed"]).to_numpy()

    curves, rows, raw = {}, [], []
    for s in ["MDP", "EW"]:
        runs = {"all members": base, "survivors only": survivors}
        perf = {k: metrics.performance(r.net()[s], rf, bench) for k, r in runs.items()}
        for k, r in runs.items():
            curves[(s, k)] = r.net()[s]
        weight = base.weights[s].to_numpy()[:, delisted].sum(axis=1) if delisted.any() else np.zeros(len(base.weights[s]))
        raw.append({"portfolio": s, **{f"{k}_{m}": perf[k][m] for k in perf for m in ("cagr", "volatility", "sharpe")},
                    "weight_in_delisted_mean": weight.mean(), "weight_in_delisted_max": weight.max()})
        rows.append({"Portfolio": LABELS.get(s, s),
                     "Return per year, all index members": pct(perf["all members"]["cagr"]),
                     "Return per year, survivors only": pct(perf["survivors only"]["cagr"]),
                     "Survivorship bias (points per year)":
                         f"{100 * (perf['survivors only']['cagr'] - perf['all members']['cagr']):+.1f}",
                     "Sharpe, all members": num(perf["all members"]["sharpe"]),
                     "Sharpe, survivors only": num(perf["survivors only"]["sharpe"]),
                     "Average weight in companies that later disappeared": pct(weight.mean()),
                     "Highest": pct(weight.max())})
    save_table("t5a_survivorship", "Survivorship bias: the same backtest without the companies that later disappeared",
               pd.DataFrame(rows), raw=pd.DataFrame(raw),
               note="'Survivors only' repeats the walk-forward run using, at each date, only the "
                    "index members that are still listed today. That is the universe available "
                    "from a source that deletes delisted companies, such as Yahoo Finance. The "
                    "difference between the two columns is the survivorship bias.")
    pl.plot_survivorship(pd.DataFrame(curves))

    # Sensitivity check for delisted companies dividends
    with_dividends = wf.run(load_panel(delisted_yield=config.ASSUMED_DELISTED_YIELD),
                            base.design, OPTIMISED, pool)
    rows = []
    for label, result in [("None (baseline: only recorded prices)", base),
                          (f"{pct(config.ASSUMED_DELISTED_YIELD, 0)} per year", with_dividends)]:
        perf = {s: metrics.performance(result.net()[s], rf, bench) for s in OPTIMISED}
        rows.append({"Dividend yield given to delisted companies": label,
                     **{f"{s}: return per year": pct(perf[s]["cagr"]) for s in OPTIMISED},
                     **{f"{s}: Sharpe": num(perf[s]["sharpe"]) for s in OPTIMISED}})
    save_table("t5b_missing_dividends", "Sensitivity: the dividends of delisted companies",
               pd.DataFrame(rows),
               note=f"The traded prices of delisted companies do not include ordinary cash dividends. "
                    f"The second row repeats the backtest assuming they paid an average {pct(config.ASSUMED_DELISTED_YIELD, 0)} dividend yield.")


# ---------------------------------------------------------------------------
# Step 6. Robustness: one-at-a-time changes to the baseline design
# ---------------------------------------------------------------------------
def alternative_designs(base):
    yield "Baseline", base
    for v in config.ROBUST_TRAIN_DAYS:
        yield f"In-sample window: {v} days", base.changed(train_days=v)
    for v in config.ROBUST_TEST_DAYS:
        yield f"Out-of-sample window: {v} days", base.changed(test_days=v)
    for v in config.ROBUST_MAX_WEIGHT:
        yield ("No weight cap" if v is None else f"Weight cap: {pct(v, 0)}"), base.changed(max_weight=v)


def step6_robustness(panel, base, pool):
    raw = []
    for label, design in alternative_designs(base.design):
        start = time.time()
        result = base if design == base.design else wf.run(panel, design, ("MDP", "EW"), pool)
        net, rf = returns_with_benchmark(panel, result)
        summary = metrics.walk_forward_summary(result.windows)
        perf = metrics.performance(net["MDP"], rf, net["SPY"])
        ew = metrics.performance(net["EW"], rf, net["SPY"])
        mdp = result.windows[result.windows.strategy == "MDP"]
        raw.append({
            "design": label, "windows": int(summary.at["MDP", "windows"]),
            "mdp_dr_in": summary.at["MDP", "dr_in"], "mdp_dr_out": summary.at["MDP", "dr_out"],
            "efficiency": summary.at["MDP", "efficiency"],
            "efficiency_like_for_like": summary.at["MDP", "efficiency_like_for_like"],
            "ew_dr_out": summary.at["EW", "dr_out"],
            "share_higher": metrics.paired_comparison(result.windows, "MDP", "EW")["share_higher"],
            "stocks_held": mdp["held"].mean(),
            "turnover": metrics.turnover_per_year(result.windows, "MDP", len(net)),
            "mdp_cagr": perf["cagr"], "mdp_volatility": perf["volatility"], "mdp_sharpe": perf["sharpe"],
            "ew_cagr": ew["cagr"], "ew_sharpe": ew["sharpe"],
        })
        print(f"    {label:<34} {time.time() - start:5.0f}s")
    raw = pd.DataFrame(raw)
    rows = [{"Design": r.design, "Windows": r.windows, "MDP: DR in-sample": num(r.mdp_dr_in),
             "MDP: DR out-of-sample": num(r.mdp_dr_out), "Efficiency": pct(r.efficiency, 0),
             "Efficiency, like for like": pct(r.efficiency_like_for_like, 0),
             "EW: DR out-of-sample": num(r.ew_dr_out),
             "Windows with MDP above EW": pct(r.share_higher, 0),
             "Stocks held": f"{r.stocks_held:.0f}", "Turnover per year": pct(r.turnover, 0),
             "MDP return per year": pct(r.mdp_cagr), "MDP volatility": pct(r.mdp_volatility),
             "MDP Sharpe": num(r.mdp_sharpe), "EW return per year": pct(r.ew_cagr),
             "EW Sharpe": num(r.ew_sharpe)} for r in raw.itertuples()]
    save_table("t6a_robustness", "Robustness: one change at a time to the baseline design",
               pd.DataFrame(rows), raw=raw,
               note="Each row changes one parameter and keeps the rest. DR is measured over the "
                    "out-of-sample window of that row. Returns after trading costs.")
    pl.plot_robustness(raw)

    rows = []
    for cost in sorted({config.COST_BPS, *config.ROBUST_COST_BPS}):
        net, rf = returns_with_benchmark(panel, base, cost)
        perf = {s: metrics.performance(net[s], rf, net["SPY"]) for s in net}
        rows.append({"Cost per unit traded": f"{cost:g} bp",
                     **{f"{s}: return per year": pct(perf[s]["cagr"]) for s in OPTIMISED},
                     "MDP: Sharpe ratio": num(perf["MDP"]["sharpe"]),
                     "SPY: return per year": pct(perf["SPY"]["cagr"])})
    save_table("t6b_costs", "Robustness: trading costs", pd.DataFrame(rows),
               note="Same baseline weights, different trading cost assumption.")


# ---------------------------------------------------------------------------
# Step 7. Can the Sharpe ratio be improved without choosing with hindsight?
# ---------------------------------------------------------------------------
def candidate_name(train_days, target, exponent):
    kind = {1.0: "MDP", 0.0: "Minimum variance"}.get(exponent, f"Exponent {exponent:g}")
    return f"{kind}, {train_days} days, {target} target"


def interval(d):
    return f"{d['difference']:+.2f} ({d['low']:+.2f} to {d['high']:+.2f})"


def step7_sharpe(panel, base, pool):
    design, rf, spy = base.design, panel.risk_free, panel.benchmark
    baseline = (design.train_days, "identity", 1.0)

    # Every candidate: the same walk-forward, one change or several
    results = {}
    for exponent in config.IMPROVE_EXPONENTS:
        for target in config.IMPROVE_TARGETS:
            for train_days in config.IMPROVE_TRAIN_DAYS:
                key = (train_days, target, exponent)
                if key == baseline:
                    results[key] = wf.Result(design, base.gross[["MDP"]],
                                             base.windows[base.windows.strategy == "MDP"],
                                             {"MDP": base.weights["MDP"]})
                    continue
                start = time.time()
                rule = partial(op.risk_based, exponent=exponent, target=target)
                results[key] = wf.run(panel, design.changed(train_days=train_days), {"candidate": rule}, pool)
                print(f"    {candidate_name(*key):<54} {time.time() - start:4.0f}s")
    net = {key: r.net().iloc[:, 0] for key, r in results.items()}
    reference = net[baseline]

    def describe(label, r, windows=None, kind="", reference=reference):
        p = metrics.performance(r, rf.reindex(r.index), spy.reindex(r.index))
        return {"portfolio": label, "kind": kind, "days": len(r), "cagr": p["cagr"],
                "volatility": p["volatility"], "sharpe": p["sharpe"], "max_drawdown": p["max_drawdown"],
                "dr_out": windows["dr_out"].mean() if windows is not None else np.nan,
                "turnover": windows["traded"].iloc[1:].sum() / 2 / (len(r) / config.TRADING_DAYS)
                            if windows is not None else np.nan,
                **metrics.sharpe_difference(r, reference, rf),
                **{f"spy_{k}": v for k, v in metrics.sharpe_difference(r, spy, rf).items()}}

    # (a) What each change does on its own. Informative, but choosing from this table is hindsight.
    one_change = [("Baseline MDP", baseline)]
    one_change += [(f"In-sample window: {d} days", (d, "identity", 1.0))
                   for d in config.IMPROVE_TRAIN_DAYS if d != design.train_days]
    one_change += [(f"Shrinkage towards {t}", (design.train_days, t, 1.0))
                   for t in config.IMPROVE_TARGETS if t != "identity"]
    one_change += [(f"Return grows with volatility ** {e:g}" + (" (minimum variance)" if e == 0 else ""),
                    (design.train_days, "identity", e)) for e in config.IMPROVE_EXPONENTS if e != 1.0]
    raw_a = pd.DataFrame([describe(label, net[key], results[key].windows, "one change")
                          for label, key in one_change])
    rows = [{"Change": r.portfolio, "DR out-of-sample": num(r.dr_out), "Return per year": pct(r.cagr),
             "Volatility": pct(r.volatility), "Sharpe ratio": num(r.sharpe),
             "Sharpe minus baseline (95% interval)": "–" if r.portfolio == "Baseline MDP" else interval(r._asdict())}
            for r in raw_a.itertuples()]
    save_table("t7a_one_change", "Sharpe ratio: one change at a time to the MDP", pd.DataFrame(rows), raw=raw_a,
               note="Each row changes one thing and keeps the rest. The interval of the difference comes "
                    "from redrawing the days in blocks (see `sharpe_difference` in maxdiv/metrics.py); "
                    "if it contains 0 the sample cannot tell the two apart. Picking the best row of "
                    "this table would be choosing with hindsight: the next table does not.")

    # (b) Portfolios in which nothing is chosen with hindsight
    mdp_only = {key: r for key, r in results.items() if key[2] == 1.0}
    mix = sel.ensemble(panel, design, mdp_only)
    adaptive_mdp, chosen_mdp = sel.adaptive(panel, design, mdp_only, baseline)
    adaptive_all, chosen_all = sel.adaptive(panel, design, results, baseline)
    portfolios = [("Baseline MDP", reference, results[baseline].windows),
                  (f"Average of the {len(mdp_only)} versions of the MDP", mix.net().iloc[:, 0], mix.windows),
                  (f"Best of the {len(mdp_only)} versions of the MDP so far",
                   adaptive_mdp.net().iloc[:, 0], adaptive_mdp.windows),
                  (f"Best of {len(results)} candidates so far (MDP to minimum variance)",
                   adaptive_all.net().iloc[:, 0], adaptive_all.windows),
                  (LABELS["SPY"], spy.reindex(reference.index), None)]
    raw_b = pd.DataFrame([describe(label, r, w, "no hindsight") for label, r, w in portfolios])

    # The same choice when the delisted companies are given the dividends that the data lack.
    # SPY has its dividends, so this is the fair comparison with it; the yield is an assumption.
    paid = load_panel(delisted_yield=config.ASSUMED_DELISTED_YIELD)
    with_dividends = {key: wf.run(paid, design.changed(train_days=key[0]),
                                  {"candidate": partial(op.risk_based, exponent=key[2], target=key[1])}, pool)
                      for key in mdp_only}
    adaptive_paid, _ = sel.adaptive(paid, design, with_dividends, baseline)
    paid_reference = with_dividends[baseline].net().iloc[:, 0]
    tag = f" + {pct(config.ASSUMED_DELISTED_YIELD, 0)} dividend for the delisted companies"
    raw_b = pd.concat([raw_b, pd.DataFrame([
        describe("Baseline MDP" + tag, paid_reference, with_dividends[baseline].windows, "assumed dividends",
                 reference=paid_reference),
        describe(f"Best of the {len(mdp_only)} versions of the MDP so far" + tag,
                 adaptive_paid.net().iloc[:, 0], adaptive_paid.windows,
                 "assumed dividends", reference=paid_reference)])], ignore_index=True)
    rows = [{"Portfolio": r.portfolio, "Return per year": pct(r.cagr), "Volatility": pct(r.volatility),
             "Sharpe ratio": num(r.sharpe), "Worst drawdown": pct(r.max_drawdown),
             "Turnover per year": "–" if np.isnan(r.turnover) else pct(r.turnover, 0),
             "Sharpe minus baseline (95% interval)": "–" if r.portfolio.startswith("Baseline MDP")
             else interval(r._asdict()),
             "Sharpe minus SPY (95% interval)": "–" if r.portfolio == LABELS["SPY"] else
             f"{r.spy_difference:+.2f} ({r.spy_low:+.2f} to {r.spy_high:+.2f})"}
            for r in raw_b.itertuples()]
    clear = [r.portfolio for r in raw_b.itertuples() if r.low > 0 and r.portfolio != LABELS["SPY"]]
    verdict = ("Interval above 0: " + "; ".join(clear) + ". Several portfolios are compared with the "
               "baseline, so an interval that only just leaves 0 out is weak evidence."
               if clear else "None of them is distinguishable from the baseline: every interval contains 0.")
    save_table("t7b_no_hindsight", "Sharpe ratio: portfolios that are not chosen with hindsight",
               pd.DataFrame(rows), raw=raw_b,
               note="'Average': the mean of the weights of the versions of the MDP (three in-sample "
                    "windows, two shrinkage targets); nothing is chosen. 'Best so far': on each "
                    "optimisation date, the candidate with the highest Sharpe ratio on its own "
                    f"out-of-sample returns up to that day; the first {config.SELECTION_BURN_IN_YEARS} years "
                    "hold the baseline. The last two rows repeat the baseline and the choice between "
                    "versions of the MDP giving the delisted companies the dividends that their prices "
                    "lack (an assumption, see t5b); their 'minus baseline' is against the baseline with "
                    f"the same dividends. {verdict}")

    # (c) What the choice was
    rows = []
    for label, chosen, pool_of in [("Among the versions of the MDP", chosen_mdp, mdp_only),
                                   ("From the MDP to minimum variance", chosen_all, results)]:
        after = chosen.iloc[sel.burn_in_windows(design):]
        for key, n in after.value_counts().items():
            rows.append({"Choice": label, "Candidate": candidate_name(*key),
                         "Windows in which it was held": n, "Share": pct(n / len(after), 0)})
    save_table("t7c_choices", "Sharpe ratio: what was chosen on each date", pd.DataFrame(rows),
               raw=pd.DataFrame({"mdp_versions": chosen_mdp.map(lambda k: candidate_name(*k)),
                                 "whole_family": chosen_all.map(lambda k: candidate_name(*k))}
                                ).rename_axis("optimised").reset_index(),
               note=f"Windows after the first {config.SELECTION_BURN_IN_YEARS} years, in which the baseline is held.")

    # (d) Every candidate over the whole sample, for the record
    raw_d = pd.DataFrame([{**describe(candidate_name(*key), net[key], results[key].windows, "candidate"),
                           "train_days": key[0], "target": key[1], "exponent": key[2]} for key in results])
    rows = [{"Candidate": r.portfolio, "DR out-of-sample": num(r.dr_out), "Return per year": pct(r.cagr),
             "Volatility": pct(r.volatility), "Sharpe ratio": num(r.sharpe),
             "Turnover per year": pct(r.turnover, 0)} for r in raw_d.itertuples()]
    save_table("t7d_candidates", f"Sharpe ratio: the {len(results)} candidates over the whole sample",
               pd.DataFrame(rows), raw=raw_d,
               note="For the record. The candidates with a 756-day window start one rebalance later, on "
                    "the first date with a full window. "
                    "This table must not be used to choose: the differences between its rows are "
                    "smaller than the uncertainty of each of them.")
    pl.plot_sharpe(pd.concat([raw_a.iloc[1:], raw_b[raw_b.kind == "no hindsight"].iloc[1:-1]],
                             ignore_index=True))


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-download every series")
    parser.add_argument("--quick", action="store_true",
                        help="skip the robustness runs and the study of the Sharpe ratio")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="processes used to optimise the windows in parallel")
    args = parser.parse_args()

    print("Data (downloads only what is missing from data/raw)")
    download_all(refresh=args.refresh)
    panel = load_panel()
    pl.setup_style()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        start = time.time()
        print(f"Baseline walk-forward on {args.workers} processes")
        base = wf.run(panel, wf.Design(), OPTIMISED, pool)
        print(f"    {len(base.weights['MDP'])} windows in {time.time() - start:.0f}s")

        print("Step 0: data checks")
        step0_data_checks(panel, base)
        print("Step 1: in-sample vs out-of-sample diversification ratio")
        step1_walk_forward(panel, base)
        print("Step 2: out-of-sample performance")
        net = step2_performance(panel, base)
        print("Step 3: portfolio composition")
        step3_portfolio(panel, base)
        print("Step 4: sub-periods")
        step4_subperiods(base, net)
        print("Step 5: survivorship bias")
        step5_survivorship(panel, base, pool)
        if not args.quick:
            print("Step 6: robustness")
            step6_robustness(panel, base, pool)
            print("Step 7: can the Sharpe ratio be improved?")
            step7_sharpe(panel, base, pool)
    write_summary()
    print(f"Done. Tables in {config.TABLE_DIR}, figures in {config.FIG_DIR}, "
          "summary in results/summary.md")


if __name__ == "__main__":
    main()
