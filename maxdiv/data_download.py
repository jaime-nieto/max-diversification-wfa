"""
Download the universe and every time series, and cache them in data/raw/.

    membership_*.csv    the two lists of S&P 500 members by date (github.com/fja05680/sp500)
    membership.csv      who was in the index and when, built from them
    holdings.csv        holdings of an S&P 500 fund: names and sectors of today's members
    profiles.csv        names and sectors of former members that are still listed (Yahoo)
    adj_close.parquet   adjusted close of every company that is still listed (Yahoo)
    close.parquet       their close adjusted for splits only, to check the adjustments (Yahoo)
    bolsamania/*.csv    traded prices of the companies that are no longer listed
    benchmark.csv       S&P 500 fund (SPY), adjusted close (Yahoo); its dates are the trading days
    equal_weight.csv    equal-weight S&P 500 fund (RSP), adjusted close (Yahoo)
    risk_free.csv       3-month Treasury bill rate (FRED)

manifest.json records where each file came from and when it was downloaded.
Once the cache exists the project runs fully offline.
"""
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests
import truststore

from . import bolsamania, config, universe

# Use the operating system's certificate store for HTTPS.
truststore.inject_into_ssl()

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
YAHOO_URL = "https://finance.yahoo.com/quote/{code}"   # documentation only
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

MANIFEST = config.RAW_DIR / "manifest.json"
MEMBERSHIP_FILES = {"": config.RAW_DIR / "membership_original.csv",
                    " (Updated)": config.RAW_DIR / "membership_updated.csv"}
MEMBERSHIP = config.RAW_DIR / "membership.csv"
HOLDINGS = config.RAW_DIR / "holdings.csv"
PROFILES = config.RAW_DIR / "profiles.csv"
ADJ_CLOSE = config.RAW_DIR / "adj_close.parquet"
CLOSE = config.RAW_DIR / "close.parquet"
BENCHMARK = config.RAW_DIR / "benchmark.csv"
EQUAL_WEIGHT = config.RAW_DIR / "equal_weight.csv"
RISK_FREE = config.RAW_DIR / "risk_free.csv"

# The fund and Yahoo name the sectors in their own way; these are the GICS names.
SECTOR_NAMES = {"Communication": "Communication Services", "Technology": "Information Technology",
                "Financial Services": "Financials", "Healthcare": "Health Care",
                "Consumer Cyclical": "Consumer Discretionary", "Consumer Defensive": "Consumer Staples",
                "Basic Materials": "Materials"}
# A company that leaves the index may still be held until the next rebalance,
# so its prices are needed for one holding period more (in calendar days).
AFTER_EXIT = pd.Timedelta(days=round(1.5 * max(config.TEST_DAYS, *config.ROBUST_TEST_DAYS)) + 15)


def _update_manifest(name, **entry):
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    entry["downloaded_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    manifest[name] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def read_membership():
    return pd.read_csv(MEMBERSHIP, parse_dates=["start", "end"])


def read_delisted():
    """data/reference/delisted.csv: name, sector and list of pages of each delisted company."""
    table = pd.read_csv(config.DELISTED_FILE, keep_default_na=False).set_index("security")
    table["pages"] = table["pages"].map(lambda text: [p for p in text.split("|") if p])
    return table


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
def download_membership():
    files = {}
    for suffix, path in MEMBERSHIP_FILES.items():
        url = config.MEMBERSHIP_URL.format(commit=config.MEMBERSHIP_COMMIT, suffix=suffix)
        response = requests.get(url, headers=BROWSER, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
        files[suffix] = universe.read_file(response.text)
    rows = universe.members_by_date(files[""], files[" (Updated)"])
    intervals = universe.membership_intervals(rows, config.DOWNLOAD_START)
    intervals.to_csv(MEMBERSHIP, index=False)
    _update_manifest("membership", source="github.com/fja05680/sp500 (MIT licence)",
                     url=config.MEMBERSHIP_URL.format(commit=config.MEMBERSHIP_COMMIT, suffix="[ (Updated)]"),
                     first_file_ends=str(max(files[""]).date()), last_change=str(max(rows).date()),
                     stays_since_download_start=len(intervals),
                     codes_since_download_start=int(intervals.code.nunique()))
    return intervals


def yahoo_securities(intervals):
    """Securities whose prices come from Yahoo: every one that is not in the list of delisted."""
    delisted = read_delisted().index
    securities = {s for code in intervals.code for s, _ in universe.securities_of(code)}
    return sorted(s for s in securities if s not in delisted and not universe.is_delisted_code(s))


def parse_holdings(text):
    """Holdings file of the fund as a table: security, name, sector, and the date of the file."""
    lines = text.splitlines()
    as_of = next(l for l in lines if l.startswith("Fund Holdings as of")).split(",", 1)[1].strip('"')
    header = next(i for i, l in enumerate(lines) if l.startswith("Ticker,"))
    raw = pd.read_csv(io.StringIO("\n".join(lines[header:])), dtype=str)
    stocks = raw[(raw["Asset Class"] == "Equity") & raw["Ticker"].str.fullmatch(r"[A-Z]+( [A-Z])?", na=False)]
    table = pd.DataFrame({"security": stocks["Ticker"].str.replace(" ", "."),      # BRK B -> BRK.B
                          "name": stocks["Name"].str.title(),
                          "sector": stocks["Sector"].replace(SECTOR_NAMES)})
    return table.drop_duplicates("security").reset_index(drop=True), as_of


def read_holdings():
    return parse_holdings(HOLDINGS.read_text(encoding="utf-8-sig"))[0]


def download_holdings():
    response = requests.get(config.HOLDINGS_URL, headers=BROWSER, timeout=60)
    response.raise_for_status()
    table, as_of = parse_holdings(response.content.decode("utf-8-sig"))
    if len(table) < 450:
        raise RuntimeError(f"Only {len(table)} holdings parsed from {config.HOLDINGS_URL}")
    HOLDINGS.write_bytes(response.content)
    _update_manifest("holdings", source="iShares Core S&P 500 ETF (IVV) holdings",
                     url=config.HOLDINGS_URL, holdings_as_of=as_of, stocks=len(table))


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------
def _yahoo_batch(tickers):
    """Daily prices for a list of Yahoo tickers: {ticker: table with `adjusted` and `close`}.

    `close` is adjusted for splits only; `adjusted` also for dividends.
    """
    import yfinance as yf
    end = pd.Timestamp(config.SAMPLE_END) + pd.Timedelta(days=1)    # `end` is exclusive
    df = yf.download(tickers, start=config.DOWNLOAD_START, end=end, auto_adjust=False,
                     actions=False, threads=True, progress=False, group_by="column")
    if df.empty:
        return {}
    if not isinstance(df.columns, pd.MultiIndex):                   # single ticker
        df.columns = pd.MultiIndex.from_product([df.columns, tickers])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return {t: pd.DataFrame({"adjusted": df[("Adj Close", t)], "close": df[("Close", t)]}).dropna()
            for t in tickers
            if ("Adj Close", t) in df.columns and df[("Adj Close", t)].notna().any()}


def download_yahoo(tickers):
    """Download all tickers in batches; retry the ones that come back empty."""
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    data, pending = {}, list(tickers)
    for _ in range(config.YAHOO_RETRIES):
        for i in range(0, len(pending), config.YAHOO_BATCH):
            try:
                data.update(_yahoo_batch(pending[i:i + config.YAHOO_BATCH]))
            except Exception as error:                               # network / rate limit
                print(f"    batch failed ({type(error).__name__}); will retry")
                time.sleep(30)
        pending = [t for t in tickers if t not in data]
        if not pending:
            break
        time.sleep(10)
    return data, pending


def _profile(security):
    """Name and sector of one company according to Yahoo."""
    import yfinance as yf
    try:
        info = yf.Ticker(universe.yahoo_ticker(security)).info
    except Exception:                                                # not found / rate limit
        info = {}
    sector = info.get("sector") or ""
    return {"security": security, "name": info.get("longName") or info.get("shortName") or security,
            "sector": SECTOR_NAMES.get(sector, sector)}


def download_profiles(securities):
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    with ThreadPoolExecutor(max_workers=4) as pool:
        table = pd.DataFrame(list(pool.map(_profile, securities)))
    table.to_csv(PROFILES, index=False)
    _update_manifest("profiles", source="Yahoo Finance", url=YAHOO_URL.format(code="<ticker>"),
                     field="longName, sector", companies=len(table),
                     without_sector=sorted(table.security[table.sector == ""]))


def _save_series(series, path):
    series.rename("value").rename_axis("date").to_csv(path)


# ---------------------------------------------------------------------------
# bolsamania.com
# ---------------------------------------------------------------------------
def price_pages(intervals):
    """What to download from bolsamania.com: (page, first day, last day).

    Each page is read from the start of the sample until one holding period
    after the company left the index (it may still be held then).
    """
    first, last = pd.Timestamp(config.DOWNLOAD_START), pd.Timestamp(config.SAMPLE_END)
    stays = universe.security_stays(intervals)
    exit_day = stays.assign(end=stays.end.fillna(last)).groupby("security").end.max()
    wanted = {}
    for security, info in read_delisted().iterrows():
        if security in exit_day.index:
            for page in info["pages"]:
                wanted[page] = max(wanted.get(page, first), min(last, exit_day[security] + AFTER_EXIT))
    return [(page, first, end) for page, end in sorted(wanted.items())]


def download_delisted(intervals, refresh=False):
    pages = price_pages(intervals)
    if refresh:
        for page, _, _ in pages:
            for old in bolsamania.DIR.glob(f"{page}.*"):
                old.unlink()
    pending = [(page, start, end) for page, start, end in pages if not bolsamania.is_complete(page, end)]
    if pending:
        print(f"  downloading {len(pending)} price histories of delisted companies from bolsamania.com")
        with ThreadPoolExecutor(max_workers=config.BOLSAMANIA_WORKERS) as pool:
            list(pool.map(lambda job: bolsamania.download(*job), pending))
    if pending or "delisted_prices" not in json.loads(MANIFEST.read_text()):
        tables = {page: bolsamania.load(page) for page, _, _ in pages}
        _update_manifest("delisted_prices", source="bolsamania.com",
                         url=bolsamania.URL.format(page="<page>"),
                         field="close, high, low, open as traded (not adjusted)",
                         pages={page: {"rows": len(t), "first_date": str(t.index.min().date()),
                                       "last_date": str(t.index.max().date())}
                                for page, t in tables.items() if len(t)},
                         pages_without_prices=sorted(page for page, t in tables.items() if t.empty))


# ---------------------------------------------------------------------------
def download_all(refresh=False):
    """Download whatever is missing from data/raw (or everything if refresh=True)."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if refresh or not MEMBERSHIP.exists():
        print("  downloading the history of the S&P 500 composition")
        download_membership()
    intervals = read_membership()
    securities = yahoo_securities(intervals)

    if refresh or not HOLDINGS.exists():
        print("  downloading the holdings of the S&P 500 fund (names and sectors)")
        download_holdings()

    if refresh or not ADJ_CLOSE.exists() or not CLOSE.exists():
        tickers =[universe.yahoo_ticker(s) for s in securities]
        print(f"  downloading the adjusted close of {len(tickers)} companies from Yahoo Finance")
        data, failed = download_yahoo(tickers)
        for field, path in [("adjusted", ADJ_CLOSE), ("close", CLOSE)]:
            prices = pd.DataFrame({t: table[field] for t, table in data.items()})
            prices.sort_index().rename_axis("date").to_parquet(path)
        _update_manifest("prices", source="Yahoo Finance", url=YAHOO_URL.format(code="<ticker>"),
                         field="Adj Close (splits and dividends) and Close (splits only)",
                         tickers_requested=len(tickers), tickers_downloaded=len(data),
                         tickers_without_data=failed,
                         first_date=str(prices.index.min().date()),
                         last_date=str(prices.index.max().date()))

    if refresh or not PROFILES.exists():
        have = set(pd.read_parquet(ADJ_CLOSE).columns)
        former = [s for s in securities
                  if s not in set(read_holdings().security) and universe.yahoo_ticker(s) in have]
        print(f"  downloading the name and sector of {len(former)} former members from Yahoo Finance")
        download_profiles(former)

    download_delisted(intervals, refresh)

    for name, path, code in [("benchmark", BENCHMARK, config.BENCHMARK),
                             ("equal_weight", EQUAL_WEIGHT, config.EQUAL_WEIGHT_FUND)]:
        if refresh or not path.exists():
            print(f"  downloading {code} from Yahoo Finance")
            series = _yahoo_batch([code])[code]["adjusted"]
            _save_series(series, path)
            _update_manifest(name, source="Yahoo Finance", code=code,
                             url=YAHOO_URL.format(code=code),
                             first_date=str(series.index.min().date()),
                             last_date=str(series.index.max().date()), n_obs=int(len(series)))

    if refresh or not RISK_FREE.exists():
        code = config.RISK_FREE_FRED
        print(f"  downloading risk-free rate {code} from FRED")
        # FRED rejects requests that send a fake browser User-Agent.
        response = requests.get(FRED_URL.format(code=code), timeout=60)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        rate = pd.Series(pd.to_numeric(df[code], errors="coerce").values,
                         index=pd.to_datetime(df["observation_date"])).dropna()
        _save_series(rate, RISK_FREE)
        _update_manifest("risk_free", source="FRED", code=code, url=FRED_URL.format(code=code),
                         first_date=str(rate.index.min().date()),
                         last_date=str(rate.index.max().date()), n_obs=int(len(rate)))


if __name__ == "__main__":
    download_all(refresh=False)
