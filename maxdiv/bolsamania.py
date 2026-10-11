"""
Daily prices of companies that are no longer listed, from bolsamania.com.

Yahoo Finance deletes a company when it stops trading. Bolsamania keeps a page
with its daily prices (close, high, low and open, from 30 November 2006). The
page answers a date range with an HTML table, and never returns more than the
last 120 days of the range, so the history is read in windows of that length.

What to know about these pages (see maxdiv/data_prep.py for how it is handled):
  - one page per company and exchange: a company that moved from Nasdaq to the
    NYSE has two pages, each with a part of its history;
  - the same company can also have pages for its listing in Germany (prices
    in euros) or Mexico. Only the US pages, in dollars, are used;
  - prices are the prices at which the shares traded, not adjusted for
    dividends and only sometimes for splits;
  - the last close is repeated on US holidays.

The files are cached in data/raw/bolsamania/<page>.csv and the download can be
interrupted and resumed.
"""
import re
import time

import pandas as pd
import requests
import truststore

from . import config

truststore.inject_into_ssl()

URL = "https://www.bolsamania.com/accion/{page}/historico-precios"
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
           "Accept-Language": "es-ES,es;q=0.9"}
DIR = config.RAW_DIR / "bolsamania"
WINDOW = pd.Timedelta(days=119)
PAUSE = 0.5                       # seconds between two requests for the same page
RETRY_WAITS = [2, 5, 10, 20, 40]
# A company that stopped trading has no more prices: the download of a page
# stops after this many empty windows in a row (about one year).
EMPTY_WINDOWS = 3
MONTHS = {m: k + 1 for k, m in enumerate("ene feb mar abr may jun jul ago sep oct nov dic".split())}
COLUMNS = ["close", "high", "low", "open"]


def _number(cell):
    """'1.234,560$' -> 1234.56"""
    digits = re.sub(r"[^0-9,.\-]", "", cell.replace("&euro;", ""))
    if not re.search(r"\d", digits):                     # an empty cell is shown as "-"
        return float("nan")
    return float(digits.replace(".", "").replace(",", "."))


def parse(html):
    """The price table of a page as a DataFrame indexed by date."""
    html = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
    rows = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S):
        cells = [re.sub(r"<[^>]+>|\s+", " ", c).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)]
        date = re.fullmatch(r"(\d{2})-([a-z]{3})-(\d{2})", cells[0]) if len(cells) >= 6 else None
        if date and date.group(2) in MONTHS:
            day = pd.Timestamp(2000 + int(date.group(3)), MONTHS[date.group(2)], int(date.group(1)))
            # columns of the site: date, price, change %, high, low, open
            rows[day] = [_number(cells[1]), _number(cells[3]), _number(cells[4]), _number(cells[5])]
    table = pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS).sort_index()
    table.index.name = "date"
    return table


def currency(html):
    """Currency in which a page quotes its prices: '$', 'EUR', ... ('' if the table is empty)."""
    units = re.findall(r'<td class="text-right">\s*[\d.,]+\s*([^<\s]*?)\s*</td>', html)
    unit = max(set(units), key=units.count) if units else ""
    return "EUR" if unit in ("&euro;", "€") else unit


def fetch(page, start, end):
    """One window of prices. Raises if the site keeps answering with a different page."""
    params = {"startDate": f"{start:%d-%m-%Y}", "endDate": f"{end:%d-%m-%Y}"}
    for wait in RETRY_WAITS:
        try:
            response = requests.get(URL.format(page=page), params=params, headers=BROWSER,
                                    timeout=30)
            if "Precios hist" in response.text[:3000]:        # the title of the right page
                return parse(response.text)
        except requests.RequestException:
            pass
        time.sleep(wait)                                      # slowed down: wait and retry
    raise RuntimeError(f"bolsamania.com does not return the price history of '{page}'")


def is_complete(page, end):
    """True if the page was already read up to `end` (or up to the end of its prices)."""
    marker = DIR / f"{page}.upto"
    return marker.exists() and pd.Timestamp(marker.read_text().strip()) >= pd.Timestamp(end)


def download(page, start, end):
    """Download the prices of one page between two dates, continuing a previous run."""
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / f"{page}.csv"
    marker = DIR / f"{page}.upto"                 # last date already asked for
    done = load(page) if path.exists() else None
    cursor, end = pd.Timestamp(start), pd.Timestamp(end)
    if marker.exists():
        cursor = max(cursor, pd.Timestamp(marker.read_text().strip()) + pd.Timedelta(days=1))
    empty = 0
    while cursor <= end:
        stop = min(cursor + WINDOW, end)
        window = fetch(page, cursor, stop)
        empty = empty + 1 if window.empty else 0
        if len(window):
            done = window if done is None else pd.concat([done, window])
            done = done[~done.index.duplicated(keep="last")].sort_index()
            done.to_csv(path)
        if empty >= EMPTY_WINDOWS and done is not None and len(done):
            stop = end                                        # the company no longer trades
        marker.write_text(f"{stop:%Y-%m-%d}")
        cursor = stop + pd.Timedelta(days=1)
        time.sleep(PAUSE)
    return done


def load(page):
    """Cached prices of one page (empty table if it has none)."""
    path = DIR / f"{page}.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"))
    table = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return table[table["close"] > 0]                     # days without a closing price are dropped
