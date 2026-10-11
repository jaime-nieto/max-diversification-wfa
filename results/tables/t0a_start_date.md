### Data check: why the backtest starts on 2009-09-04

| Date | Index members | With prices | Without prices | Coverage |
|---|---|---|---|---|
| 2008-12-02 | 500 | 462 | 38 | 92.4% |
| 2009-03-05 | 500 | 471 | 29 | 94.2% |
| 2009-06-04 | 499 | 473 | 26 | 94.8% |
| 2009-09-02 | 500 | 474 | 26 | 94.8% |
| 2009-09-04 | 500 | 475 | 25 | 95.0% |

Prices of delisted companies start on 2006-11-30, so the first date with a full in-sample window of 504 days is 2008-12-02. The backtest starts on the first date from which at least 95% of the index members have prices on every trading day until the end of the sample. The last day below that level is 2009-09-03. The members without prices are companies that no longer trade and for which no free price history was found; most of the ones missing at the earliest dates stopped trading before mid-2009.
