"""
Central configuration.

Every choice that can change a result (dates, universe, data sources, window
lengths, constraints, costs) lives in this file, so that nothing is hidden
inside the analysis code.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
# One row per index member that is no longer listed: its name, its sector and
# the pages of bolsamania.com that keep its prices (empty = no page found).
DELISTED_FILE = ROOT / "data" / "reference" / "delisted.csv"

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
# First day for which prices of delisted companies are available.
DOWNLOAD_START = "2006-11-30"
# Share of the index members that must have prices on every optimisation date.
# No free source has every company that has left the stock market, so 100% is
# not reachable for the S&P 500; the run stops if the coverage falls below
# this level (see results/tables/t0b_universe.md).
MIN_INDEX_COVERAGE = 0.95
# First out-of-sample day: the earliest date from which the coverage stays
# above MIN_INDEX_COVERAGE (see results/tables/t0a_start_date.md). The first
# optimisation uses the index members of the last trading day before it.
OOS_START = "2009-09-08"
# The analysis stops at a fixed date. If you re-download the data later, the
# new observations are ignored.
SAMPLE_END = "2026-09-30"

# ---------------------------------------------------------------------------
# Universe: S&P 500 with the members of each date (point in time)
# ---------------------------------------------------------------------------
# Lists of the members on every date on which the index changed, kept at
# github.com/fja05680/sp500 (MIT licence). The commit is fixed so that a later
# download gives the same universe; it has the changes up to 18 August 2026,
# and the members of that day are kept until SAMPLE_END. See maxdiv/universe.py.
MEMBERSHIP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/{commit}/"
                  "S%26P%20500%20Historical%20Components%20%26%20Changes{suffix}.csv")
MEMBERSHIP_COMMIT = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"      # 7 September 2026
MEMBERSHIP_PATCHES = [
    # (code, first day, day it stops applying, why)
    ("LIN", "2018-10-31", "2019-01-12",
     "Linde replaced Praxair on 31 October 2018; the first file does not have it until it ends"),
]

# Codes of the membership files that are the same share as another code:
# changes of ticker, and companies that continued under a new name after a
# merger. (code in the file, security, last day or None). A code with a last
# day means one company up to that day and another one afterwards.
SAME_COMPANY = [
    ("ABC", "COR", None), ("ADS", "BFH", None), ("ANTM", "ELV", None),
    ("ARNC", "AA", "2016-10-31"), ("ARNC", "HWM", None),      # Alcoa split in two on 1 November 2016
    ("ATGE", "CVSA", None), ("BBT", "TFC", None), ("BHGE", "BKR", None), ("BK", "BNY", None),
    ("BLL", "BALL", None), ("CBS", "SKYD", None), ("VIAC", "SKYD", None), ("PARA", "SKYD", None),
    ("PSKY", "SKYD", None), ("CCE", "CCEP", None), ("CTL", "LUMN", None), ("DDR", "SITC", None),
    ("DWDP", "DD", None), ("FB", "META", None), ("FBHS", "FBIN", None), ("FI", "FISV", None),
    ("FII", "FHI", None), ("FLT", "CPAY", None), ("GPS", "GAP", None), ("HCP", "DOC", None),
    ("PEAK", "DOC", None), ("HFC", "DINO", None), ("HRS", "LHX", None), ("IAC", "MTCH", None),
    ("IR", "TT", "2020-02-28"),                               # Ingersoll-Rand plc became Trane Technologies
    ("JEC", "J", None), ("KORS", "CPRI", None), ("LB", "BBWI", None), ("MMC", "MRSH", None),
    ("MYL", "VTRS", None), ("NCR", "VYX", None), ("NLOK", "GEN", None), ("SYMC", "GEN", None),
    ("PKI", "RVTY", None), ("PX-201810", "LIN", None), ("RE", "EG", None), ("SATS", "ECHO", None),
    ("TMK", "GL", None), ("UTX", "RTX", None), ("WLTW", "WTW", None), ("WYND", "TNL", None),
    # companies that are no longer listed (their prices come from bolsamania.com)
    ("CDAY", "DAY", None),                                    # Ceridian became Dayforce
    ("COG", "CTRA", None),                                    # Cabot Oil & Gas became Coterra
    ("DISCA", "WBD", None),                                   # Discovery became Warner Bros. Discovery
    # Twenty-First Century Fox was bought by Disney in March 2019; the Fox
    # Corporation that trades today under the same tickers is a new company.
    ("FOXA", "FOXA-201903", "2019-03-18"), ("FOX", "FOX-201903", "2019-03-18"),
]

# Corporate actions of delisted companies adjusted by hand: prices before
# `date` are multiplied by `factor`. Bolsamania's prices are the prices at
# which the shares traded, so a split or a spin-off shows up as a jump that is
# not a gain or a loss. Every one-day move above 25% in these companies, and
# every overnight fall above 8% on a quiet day, was looked at (from three years
# before each company entered the index to 400 days after it left); the ones below
# are those that match a corporate action (the rest are takeover bids, results
# and the crashes of 2008 and 2020). Ordinary cash dividends of these companies
# are NOT included: no free source has them. Their returns are therefore
# understated.
ADJUSTMENTS = [
    # (security, date, factor, what happened)
    # --- splits: N new shares for each old one ---
    ("AGN-201503", "2007-06-25", 1 / 2, "Split 2 for 1"),
    ("ALXN", "2011-05-23", 1 / 2, "Split 2 for 1"),
    ("ANDV-201809", "2007-05-30", 1 / 2, "Split 2 for 1"),
    ("CELG", "2014-06-26", 1 / 2, "Split 2 for 1"),
    ("CERN", "2011-06-27", 1 / 2, "Split 2 for 1"),
    ("CERN", "2013-07-01", 1 / 2, "Split 2 for 1"),
    ("CTRA", "2007-04-02", 1 / 2, "Split 2 for 1"),
    ("CTRA", "2012-01-26", 1 / 2, "Split 2 for 1"),
    ("CTRA", "2013-08-15", 1 / 2, "Split 2 for 1"),
    ("DISCK", "2014-08-07", 1 / 2, "One new class C share given for each share held"),
    ("WBD", "2014-08-07", 1 / 2, "One new class C share given for each share held"),
    ("DNR", "2007-12-17", 1 / 2, "Split 2 for 1"),
    ("ESRX-201812", "2007-06-25", 1 / 2, "Split 2 for 1"),
    ("ESRX-201812", "2010-06-08", 1 / 2, "Split 2 for 1"),
    ("FLIR", "2007-12-11", 1 / 2, "Split 2 for 1"),
    ("HBI", "2015-03-04", 1 / 4, "Split 4 for 1"),
    ("LO-201506", "2013-01-16", 1 / 3, "Split 3 for 1"),
    ("MHS-201203", "2008-01-25", 1 / 2, "Split 2 for 1"),
    ("MRO", "2007-06-19", 1 / 2, "Split 2 for 1"),
    ("NBL", "2013-05-29", 1 / 2, "Split 2 for 1"),
    ("PBCT", "2007-04-16", 1 / 2.1, "Conversion to a listed holding company: 2.1 new shares for each old one"),
    ("RAI-201707", "2010-11-16", 1 / 2, "Split 2 for 1"),
    ("RAI-201707", "2015-09-01", 1 / 2, "Split 2 for 1"),
    ("SEE", "2007-03-20", 1 / 2, "Split 2 for 1"),
    ("SIAL-201511", "2007-01-03", 1 / 2, "Split 2 for 1"),
    ("SRCL", "2007-06-01", 1 / 2, "Split 2 for 1"),
    ("SWN", "2008-03-26", 1 / 2, "Split 2 for 1"),
    ("WFM-201708", "2013-05-30", 1 / 2, "Split 2 for 1"),
    ("XTO-201006", "2007-12-14", 4 / 5, "Split 5 for 4"),
    # --- share consolidations: 1 new share for N old ---
    ("BTUUQ-201704", "2015-10-01", 15.0, "Share consolidation 1 for 15"),
    ("DF", "2013-08-27", 2.0, "Share consolidation 1 for 2"),
    ("ETFC", "2010-06-02", 10.0, "Share consolidation 1 for 10"),
    ("HSH-201408", "2012-06-28", 5.0, "Share consolidation 1 for 5"),
    ("JAVA-201001", "2007-11-12", 4.0, "Share consolidation 1 for 4"),
    ("TWC-201605", "2009-03-13", 3.0, "Share consolidation 1 for 3"),
    # --- spin-offs and special dividends: factor = opening price / previous close ---
    ("BEAM-201404", "2011-10-04", 44.82 / 54.20, "Spin-off of Fortune Brands Home & Security"),
    ("COV-201501", "2013-07-01", 57.16 / 62.84, "Spin-off of Mallinckrodt"),
    ("CTXS", "2017-02-01", 74.27 / 91.19, "Spin-off of the GoTo business, merged with LogMeIn"),
    ("DF", "2007-04-03", 33.00 / 47.33, "Special dividend of 15 dollars per share"),
    ("DF", "2013-05-24", 9.82 / 20.41, "Spin-off of WhiteWave Foods"),
    ("FOXA-201903", "2013-07-01", 29.07 / 32.58, "Spin-off of the new News Corp"),
    ("FOX-201903", "2013-07-01", 29.38 / 32.82, "Spin-off of the new News Corp"),
    ("GGP-201808", "2010-11-10", 14.46 / 17.39, "New shares on leaving bankruptcy and spin-off of Howard Hughes"),
    ("HSH-201408", "2012-07-02", 28.94 / 92.50, "Spin-off of D.E Master Blenders 1753 and special dividend"),
    ("JCI-201609", "2016-10-31", 39.29 / 43.76, "Spin-off of Adient"),
    ("K", "2023-10-02", 55.70 / 59.20, "Spin-off of WK Kellogg"),
    ("MRO", "2011-07-01", 32.68 / 52.68, "Spin-off of Marathon Petroleum"),
    ("SUN-201210", "2012-01-18", 35.73 / 42.31, "Spin-off of SunCoke Energy"),
    ("SWY-201501", "2014-04-15", 34.16 / 38.01, "Spin-off of Blackhawk Network"),
    ("TGNA", "2017-06-01", 14.85 / 23.74, "Spin-off of Cars.com"),
    ("TLAB-201312", "2012-12-24", 2.58 / 3.49, "Special dividend of 1 dollar per share"),
    ("TWC-201605", "2009-03-09", 8.36 / 18.40, "Special dividend of 10.27 dollars per share before leaving Time Warner"),
    ("TWX-201806", "2009-03-30", 19.10 / 8.43, "Spin-off of Time Warner Cable and share consolidation 1 for 3"),
    ("VAR", "2017-01-30", 78.05 / 88.80, "Spin-off of Varex Imaging"),
    ("WIN", "2015-04-27", 11.79 / 7.83, "Spin-off of Communications Sales & Leasing and share consolidation 1 for 6"),
]

# What the holder received when a company stopped trading, if it was not the
# last price. The three banks that failed in 2023 while they were in the index
# were closed by their regulator between two trading days, and their shares
# lost practically all their value. In every other case (takeovers and
# mergers) the last price is close to what the holder got, and the position
# is kept at that value until the next rebalance.
TERMINAL_RETURNS = [
    # (security, date, return of that day, what happened)
    ("SIVB", "2023-03-13", -1.0, "Silicon Valley Bank was closed by its regulator on 10 March 2023"),
    ("SBNY", "2023-03-13", -1.0, "Signature Bank was closed by its regulator on 12 March 2023"),
    ("FRC", "2023-05-01", -1.0, "First Republic Bank was closed by its regulator on 1 May 2023"),
]

# Errors in Yahoo's adjusted prices that the automatic check of
# maxdiv/data_prep.py (`checked_adjusted_close`) cannot correct by itself.
# One company whose dividends are recorded several times too large, so that
# its adjusted price cannot be used: its close without dividends is used.
YAHOO_WITHOUT_DIVIDENDS = [
    # (security, what is wrong)
    ("SITC", "SITE Centers (formerly DDR): each quarterly dividend takes about 10% off the adjusted price"),
]
# Days on which a spin-off or a merger was not adjusted, or was adjusted with
# the wrong ratio, and the price shows a jump that the holder did not have.
# The return of the day is set to the value given (0 = the day is ignored).
WRONG_RETURNS = [
    # (security, date, return used, what happened)
    ("EXPE", "2011-12-21", 0.0, "Spin-off of TripAdvisor and share consolidation: -34% in Yahoo"),
    ("ITT", "2011-11-01", 0.0, "Spin-off of Xylem and Exelis and share consolidation: +24% in Yahoo"),
    ("TMUS", "2013-05-01", 0.0, "Merger of MetroPCS with T-Mobile, cash payment and share consolidation: -16% in Yahoo"),
    ("VTR", "2015-08-18", 0.0, "Spin-off of Care Capital Properties: -24% in Yahoo"),
    ("CHTR", "2016-05-18", 0.0, "Exchange of shares in the merger with Time Warner Cable: +12% in Yahoo"),
    ("XRX", "2017-01-03", 0.0, "Spin-off of Conduent: +38% in Yahoo"),
]

# Sensitivity check for the missing dividends of the delisted companies: the
# backtest is repeated giving them this yield per year (the dividend yield of
# the S&P 500 has been around 2%). It is not used in the baseline results.
ASSUMED_DELISTED_YIELD = 0.02

# Market references, both investable and with dividends included:
BENCHMARK = "SPY"                 # S&P 500 fund, weighted by market value
EQUAL_WEIGHT_FUND = "RSP"         # S&P 500 fund with the same weight in every member
RISK_FREE_FRED = "DTB3"           # 3-month Treasury bill, % per year, daily

# Names and sectors of the companies that are still listed: the holdings file
# of an S&P 500 fund for today's members, Yahoo Finance for the former ones.
HOLDINGS_URL = ("https://www.ishares.com/us/products/239726/"
                "ishares-core-sp-500-etf/latest-holdings.csv")

YAHOO_BATCH = 100                 # tickers per request
YAHOO_RETRIES = 3
BOLSAMANIA_WORKERS = 8            # pages downloaded at the same time

# Data cleaning. A one-day gain above +400% is a data error or an unadjusted
# share consolidation. Such returns are set to missing, i.e. the position
# earns 0% that day. The rule can remove gains but never losses.
MAX_DAILY_RETURN = 4.0

# ---------------------------------------------------------------------------
# Walk-forward design (baseline)
# ---------------------------------------------------------------------------
TRAIN_DAYS = 504                  # in-sample window: 2 years of trading days
TEST_DAYS = 252                   # out-of-sample window = holding period: 1 year
# Weights are computed with data up to the close of day t and traded at the
# close of day t + EXECUTION_LAG.
EXECUTION_LAG = 1
# Cost per unit of notional traded (one side), in basis points. Selling the
# whole portfolio and buying a new one costs 2 x COST_BPS.
COST_BPS = 10.0

# ---------------------------------------------------------------------------
# Eligibility at each optimisation date (uses in-sample information only)
# ---------------------------------------------------------------------------
# A stock is eligible if it is in the index that day and has a return on at
# least this share of the in-sample days. A company listed for less than the
# whole window (a recent IPO or spin-off) has to wait.
MIN_COVERAGE = 0.99

# ---------------------------------------------------------------------------
# Portfolio constraints
# ---------------------------------------------------------------------------
# Long-only, fully invested, at most MAX_WEIGHT per stock (None = no cap).
MAX_WEIGHT = 0.05

# ---------------------------------------------------------------------------
# Robustness: one-at-a-time changes to the baseline
# ---------------------------------------------------------------------------
ROBUST_TRAIN_DAYS = [252, 756]
ROBUST_TEST_DAYS = [63, 126]      # a quarter (the baseline of the first version) and half a year
ROBUST_MAX_WEIGHT = [None, 0.10, 0.02]
ROBUST_COST_BPS = [0.0, 25.0]

# ---------------------------------------------------------------------------
# Can the Sharpe ratio of the MDP be improved? (step 7)
# ---------------------------------------------------------------------------
# Three ways of changing the MDP, each with a reason that does not come from
# this sample (see the README). They were fixed before running them. Every
# combination is a candidate; none is picked by its result over the whole
# sample (maxdiv/selection.py picks on each date with the returns known then).
IMPROVE_TRAIN_DAYS = [252, 504, 756]                  # more history: less noise, older information
IMPROVE_TARGETS = ["identity", "constant correlation"]  # what the correlations are shrunk towards
# Expected return of a stock ~ volatility ** exponent. 1 is the MDP, 0 is
# minimum variance; 0.5 is half-way.
IMPROVE_EXPONENTS = [1.0, 0.5, 0.0]
SELECTION_BURN_IN_YEARS = 3       # years held with the baseline before the first choice
BOOTSTRAP_DRAWS = 5000            # redraws for the interval of a difference of Sharpe ratios
BOOTSTRAP_BLOCK = 21              # days per block (keeps the clustering of volatility)

TRADING_DAYS = 252                # annualisation factor

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
SUBPERIODS = [
    ("2009-2012 (recovery, euro crisis)", "2009-09-01", "2012-12-31"),
    ("2013-2016", "2013-01-01", "2016-12-31"),
    ("2017-2019", "2017-01-01", "2019-12-31"),
    ("2020-2022 (COVID, inflation)", "2020-01-01", "2022-12-31"),
    ("2023-2026", "2023-01-01", SAMPLE_END),
]
