"""
The S&P 500 through time: who was in the index on each date.

Source: the lists kept at github.com/fja05680/sp500, one row per date on which
the index changed, with every member of that date. Two files are used:

  the original one   ends in January 2019. A company that no longer trades has
                     the month of its delisting attached to its code
                     (LEHMQ-201203), so two companies that used the same ticker
                     at different times are kept apart.
  the updated one    reaches the present, with plain tickers.

The first file is used up to its last date and the second one afterwards.
The words used in this project:

  code       what the files contain: BRK.B, FB, LEHMQ-201203
  security   one share with one price history; the columns of every table.
             FB and META are the same security (config.SAME_COMPANY).
"""
import io
import re

import pandas as pd

from . import config

DELISTED = re.compile(r"(.+)-(\d{6})")


def read_file(text):
    """One file as {date: set of codes}."""
    table = pd.read_csv(io.StringIO(text), parse_dates=["date"])
    return {date: set(codes.split(",")) for date, codes in zip(table["date"], table["tickers"])}


def duplicated_codes(rows):
    """Codes of the original file that are a second name of a delisted company.

    For the companies delisted in 2018 the file has both SCG and SCG-201812 in
    the same rows. They are the same company if the plain code is never seen
    after the month of the delisting; otherwise they are two companies that
    shared a ticker (AGN-201503 is the old Allergan, AGN is Actavis).
    """
    last_seen, together = {}, set()
    for date in sorted(rows):
        for code in rows[date]:
            last_seen[code] = date
            found = DELISTED.fullmatch(code)
            if found and found.group(1) in rows[date]:
                together.add((found.group(1), code))
    same = {}
    for plain, delisted in together:
        month_end = pd.Timestamp(delisted[-6:] + "01") + pd.offsets.MonthEnd(0)
        if last_seen[plain] <= month_end:
            same[plain] = delisted
    return same


def members_by_date(original, updated):
    """The two files joined: {date: set of codes}, sorted by date."""
    same = duplicated_codes(original)
    switch = max(original)
    rows = {date: {same.get(code, code) for code in codes} for date, codes in original.items()}
    rows.update({date: codes for date, codes in updated.items() if date > switch})
    for code, start, end, _ in config.MEMBERSHIP_PATCHES:
        for date in rows:
            if pd.Timestamp(start) <= date < pd.Timestamp(end):
                rows[date] = rows[date] | {code}
    return dict(sorted(rows.items()))


def membership_intervals(rows, since):
    """One row per code and uninterrupted stay in the index: code, start, end.

    `end` is the date on which the code left (empty if it is still in the
    index). Stays that ended before `since` are left out.
    """
    inside, stays = {}, []
    for date, codes in rows.items():
        stays += [(code, inside.pop(code), date) for code in sorted(set(inside) - codes)]
        inside.update({code: date for code in codes - set(inside)})
    stays += [(code, start, pd.NaT) for code, start in inside.items()]
    table = pd.DataFrame(stays, columns=["code", "start", "end"])
    table = table[table["end"].isna() | (table["end"] > pd.Timestamp(since))]
    return table.sort_values(["start", "code"]).reset_index(drop=True)


def securities_of(code):
    """The securities behind a code: [(security, last day or None)], in order of time."""
    dated = sorted(((security, pd.Timestamp(until)) for c, security, until in config.SAME_COMPANY
                    if c == code and until is not None), key=lambda item: item[1])
    final = [security for c, security, until in config.SAME_COMPANY if c == code and until is None]
    return dated + [(final[0] if final else code, None)]


def security_stays(intervals):
    """The stays by security instead of by code: security, start, end (first day outside, or empty).

    A stay of a code that means two securities (IR was Ingersoll-Rand plc
    until February 2020 and another company afterwards) is cut in two.
    """
    rows = []
    for stay in intervals.itertuples():
        start = stay.start
        for security, until in securities_of(stay.code):
            end = stay.end
            if until is not None and (pd.isna(end) or end > until + pd.Timedelta(days=1)):
                end = until + pd.Timedelta(days=1)
            if pd.isna(end) or start < end:
                rows.append((security, start, end))
            if until is not None:
                start = max(start, until + pd.Timedelta(days=1))
    return pd.DataFrame(rows, columns=["security", "start", "end"])


def membership(intervals, days):
    """True/False table, trading days x securities: the security was in the index that day."""
    table = {}
    for stay in security_stays(intervals).itertuples():
        end = stay.end if pd.notna(stay.end) else pd.Timestamp.max
        inside = (days >= stay.start) & (days < end)
        if inside.any():
            table[stay.security] = table.get(stay.security, False) | inside
    return pd.DataFrame(table, index=days).sort_index(axis=1)


def is_delisted_code(security):
    """LEHMQ-201203: a company that stopped trading in March 2012."""
    return DELISTED.fullmatch(security) is not None


def yahoo_ticker(security):
    """Yahoo writes share classes with a dash: BRK.B -> BRK-B."""
    return security.replace(".", "-")
