### Data check: the delisted members that have prices, and the holes in their source

| Year | Delisted members with prices (average per day) | Days without a price in the source |
|---|---|---|
| 2009 | 143 | 1.3% |
| 2010 | 137 | 0.9% |
| 2011 | 130 | 0.3% |
| 2012 | 117 | 0.7% |
| 2013 | 109 | 0.1% |
| 2014 | 103 | 0.1% |
| 2015 | 96 | 0.1% |
| 2016 | 85 | 0.0% |
| 2017 | 73 | 1.1% |
| 2018 | 63 | 0.0% |
| 2019 | 53 | 1.4% |
| 2020 | 43 | 0.0% |
| 2021 | 38 | 3.5% |
| 2022 | 30 | 13.1% |
| 2023 | 22 | 9.6% |
| 2024 | 17 | 4.6% |
| 2025 | 10 | 0.7% |
| 2026 | 3 | 0.0% |

195 companies that no longer trade were in the index during the backtest and have prices from bolsamania.com (listed in t0d_delisted.csv). That source has holes: days on which the stock traded but the page has no price, mostly in 2022 and 2023. During a hole the stock is moved with the S&P 500 fund and the day its price returns takes the rest of the move, so the return over the hole is the true one (see `bridge` in maxdiv/data_prep.py). Ordinary cash dividends of these companies are not included because no free source has them, so their return is understated.
