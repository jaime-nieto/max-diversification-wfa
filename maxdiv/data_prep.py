"""
Turn the cached files into the panel used by the walk-forward analysis.

Each index member takes its prices from one of two sources:

  Yahoo Finance   adjusted close (splits and dividends included) of the
                  companies that are still listed. A few of its adjustments
                  are wrong; they are found by comparing it with the close
                  that is adjusted for splits only (`checked_adjusted_close`).
  bolsamania.com  the price at which the shares traded, for the companies that
                  no longer trade and that Yahoo has deleted. They are the rows
                  of data/reference/delisted.csv. These prices are not adjusted:
                  a split shows up as a jump (corrected with config.ADJUSTMENTS)
                  and ordinary dividends are missing.

A company of data/reference/delisted.csv never takes its prices from Yahoo,
even if Yahoo has data under its ticker: that ticker now belongs to another
company (S was Sprint until 2020 and is SentinelOne today).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import bolsamania, config, universe
from . import data_download as dl

MAX_GAP = 10         # trading days without a price that are bridged (halts, holes in the source)
STILL_LISTED = 10    # a company is still listed if it has a price in the last days of the sample
BELIEVABLE = 0.25    # largest one-day return accepted on a day with a large adjustment
LARGE = 0.10         # an adjustment is large if it changes the return of the day by more than this
UNKNOWN = "Unknown"


@dataclass
class Panel:
    returns: pd.DataFrame          # daily returns, trading days x securities (NaN = no price)
    member: pd.DataFrame           # True where the security was in the S&P 500 that day
    benchmark: pd.Series           # daily total return of the S&P 500 fund (SPY)
    equal_weight: pd.Series        # daily total return of the equal-weight S&P 500 fund (RSP)
    risk_free: pd.Series           # daily return of a 3-month Treasury bill
    companies: pd.DataFrame        # name, sector, source of the prices, listed, has_prices
    removed: pd.Series             # returns discarded by the cleaning rule (date, security)
    rejected: pd.DataFrame         # returns of Yahoo that were not used: date, security, yahoo, used, why
    bridged: pd.DataFrame          # delisted companies: True on the days whose price was missing


def _read_series(path):
    return pd.read_csv(path, parse_dates=["date"], index_col="date")["value"]


def traded_close(pages, days):
    """Traded close of a delisted company on the trading days, joining its pages.

    A company that moved from one exchange to another has one page for each
    (Juniper: Nasdaq until October 2009, NYSE afterwards). They are the same
    share, so the pages are put one after the other; where both have a price,
    the later page is used.

    Bolsamania repeats the last close on US holidays and sometimes writes a
    Friday on the Saturday. A price on a day without trading is moved to the
    previous trading day if that day has no price, and dropped otherwise.
    """
    if not pages:
        return pd.Series(np.nan, index=days)
    close = pd.concat([bolsamania.load(page)["close"] for page in pages])
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close[(close.index >= days[0]) & (close.index <= days[-1])]
    if close.empty:
        return pd.Series(np.nan, index=days)
    traded = close.index.isin(days)
    on_days, off_days = close[traded], close[~traded]
    previous = days[days.searchsorted(off_days.index) - 1]    # last trading day before each of them
    moved = pd.Series(off_days.to_numpy(), index=previous)
    moved = moved[~moved.index.duplicated(keep="first") & ~moved.index.isin(on_days.index)]
    return pd.concat([on_days, moved]).sort_index().reindex(days)


def checked_adjusted_close(adjusted, close):
    """Yahoo's adjusted close without the adjustments that cannot be right.

    The adjusted close and the close move together except on the day of a
    dividend or a spin-off, when the adjusted one falls less (the holder
    received something). A large adjustment (more than LARGE) that turns the
    return of the day into something unbelievable (more than BELIEVABLE in
    absolute value, and further from zero than the return of the close) is an
    error of the source: Danaher did not gain 61% on the day it spun off
    Fortive. On those days the return of the close is used.

    Returns the corrected prices and the list of days that were corrected.
    """
    with_dividends = adjusted.pct_change(fill_method=None)
    plain = close.pct_change(fill_method=None)
    wrong = ((with_dividends.abs() > BELIEVABLE) & ((with_dividends - plain).abs() > LARGE)
             & (with_dividends.abs() > plain.abs()))
    # Prices before a wrong day are rescaled so that its return is the return of the close.
    factor = ((1 + with_dividends) / (1 + plain)).where(wrong, 1.0)
    earlier = factor[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    days = wrong.stack()
    days = days[days].index
    rejected = pd.DataFrame({"date": days.get_level_values(0), "security": days.get_level_values(1),
                             "yahoo": [with_dividends.at[d, s] for d, s in days],
                             "used": [plain.at[d, s] for d, s in days],
                             "why": "adjustment of Yahoo not believable; return of the unadjusted close"})
    return adjusted * earlier, rejected


def bridge(price, market, with_market):
    """Fill the days without a price inside the life of each stock (at most MAX_GAP in a row).

    Yahoo: the last price is repeated (a trading halt). bolsamania.com: the
    source has holes, days on which the stock traded but the page has no price.
    Repeating the last price would give the stock a 0% return on a day on
    which the market moved, and make it look less correlated with the rest
    than it is. Instead, during the hole the stock moves with the market
    (`market`, the S&P 500 fund); when its price returns, that day takes the
    rest of the move, so the return over the whole hole is the true one.
    """
    last = price.apply(pd.Series.last_valid_index)
    filled = price.ffill(limit=MAX_GAP)
    relative = price[with_market].div(market, axis=0).ffill(limit=MAX_GAP)
    filled[with_market] = relative.mul(market, axis=0)
    for security, day in last.items():
        if day is not None:
            filled.loc[filled.index > day, security] = np.nan
    return filled


def company_profiles():
    """Name and sector of the companies that are still listed (fund holdings, then Yahoo)."""
    table = dl.read_holdings().set_index("security")[["name", "sector"]]
    if dl.PROFILES.exists():
        extra = pd.read_csv(dl.PROFILES, keep_default_na=False).set_index("security")[["name", "sector"]]
        table = pd.concat([table, extra[~extra.index.isin(table.index)]])
    return table


def load_panel(delisted_yield=0.0):
    """Read data/raw and align everything on the trading days of the US market.

    `delisted_yield` adds a dividend yield per year to the companies that are
    no longer listed (their dividends are not in the data). Only used for a
    sensitivity check.
    """
    for path in [dl.MEMBERSHIP, dl.ADJ_CLOSE, dl.CLOSE, dl.BENCHMARK, dl.EQUAL_WEIGHT, dl.RISK_FREE,
                 dl.HOLDINGS]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python run_all.py` to download it.")

    fund = _read_series(dl.BENCHMARK).loc[config.DOWNLOAD_START:config.SAMPLE_END]
    days = fund.index                                         # the fund trades every day the market is open
    member = universe.membership(dl.read_membership(), days)
    close = pd.read_parquet(dl.CLOSE)
    yahoo = pd.read_parquet(dl.ADJ_CLOSE)[close.columns]
    for security, _ in config.YAHOO_WITHOUT_DIVIDENDS:
        yahoo[universe.yahoo_ticker(security)] = close[universe.yahoo_ticker(security)]
    yahoo, rejected = checked_adjusted_close(yahoo, close)
    rejected["security"] = rejected["security"].map({universe.yahoo_ticker(s): s for s in member.columns})
    delisted, profiles = dl.read_delisted(), company_profiles()

    prices, rows = {}, {}
    for security in member.columns:
        if security in delisted.index:
            info = delisted.loc[security]
            prices[security] = traded_close(info["pages"], days)
            source = "bolsamania.com" if info["pages"] else "none"
            name, sector = info["name"], info["sector"]
        else:
            ticker = universe.yahoo_ticker(security)
            listed = ticker in yahoo.columns and not universe.is_delisted_code(security)
            prices[security] = yahoo[ticker].reindex(days) if listed else pd.Series(np.nan, index=days)
            source = "Yahoo Finance" if listed else "none"
            known = security in profiles.index
            name = profiles.at[security, "name"] if known else security
            sector = profiles.at[security, "sector"] if known else UNKNOWN
        rows[security] = {"name": name, "sector": sector or UNKNOWN, "source": source}
    price = pd.DataFrame(prices, index=days)
    for security, date, factor, _ in config.ADJUSTMENTS:
        price.loc[price.index < pd.Timestamp(date), security] *= factor
    price = price.where(price > 0)

    from_bolsamania = [s for s, row in rows.items() if row["source"] == "bolsamania.com"]
    quoted = price[from_bolsamania].notna()
    bridged = quoted.cummax() & quoted[::-1].cummax()[::-1] & ~quoted
    price = bridge(price, fund, from_bolsamania)

    returns = price.pct_change(fill_method=None)
    returns[from_bolsamania] += delisted_yield / config.TRADING_DAYS
    fixed = [{"date": pd.Timestamp(date), "security": security, "yahoo": returns.at[pd.Timestamp(date), security],
              "used": value, "why": why} for security, date, value, why in config.WRONG_RETURNS]
    in_yahoo = rejected.set_index(["date", "security"])["yahoo"].to_dict()
    for row in fixed:                                         # the return Yahoo had before any correction
        row["yahoo"] = in_yahoo.get((row["date"], row["security"]), row["yahoo"])
    rejected = (pd.concat([rejected, pd.DataFrame(fixed)], ignore_index=True)
                .drop_duplicates(["date", "security"], keep="last").sort_values(["date", "security"]))
    for security, date, value, _ in config.WRONG_RETURNS + config.TERMINAL_RETURNS:
        returns.loc[pd.Timestamp(date), security] = value
    jumps = returns > config.MAX_DAILY_RETURN                 # see config.MAX_DAILY_RETURN
    removed = returns.where(jumps).stack().dropna()
    returns = returns.mask(jumps)

    companies = pd.DataFrame.from_dict(rows, orient="index").loc[returns.columns]
    companies["has_prices"] = returns.notna().any()
    # "Listed" = what a source that deletes delisted companies still has today.
    companies["listed"] = ((companies["source"] == "Yahoo Finance")
                           & price.iloc[-STILL_LISTED:].notna().any())

    benchmark = fund.pct_change().fillna(0.0)
    equal_weight = (_read_series(dl.EQUAL_WEIGHT).reindex(days).ffill(limit=MAX_GAP)
                    .pct_change().fillna(0.0))
    # The rate is quoted in % per year; one trading day earns 1/252 of the
    # rate published the day before.
    rate = _read_series(dl.RISK_FREE).reindex(days, method="ffill")
    risk_free = (rate / 100 / config.TRADING_DAYS).shift(1).fillna(0.0)
    return Panel(returns, member[returns.columns], benchmark, equal_weight, risk_free,
                 companies, removed, rejected.reset_index(drop=True), bridged)
