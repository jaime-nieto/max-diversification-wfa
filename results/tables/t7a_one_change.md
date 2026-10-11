### Sharpe ratio: one change at a time to the MDP

| Change | DR out-of-sample | Return per year | Volatility | Sharpe ratio | Sharpe minus baseline (95% interval) |
|---|---|---|---|---|---|
| Baseline MDP | 2.36 | 10.8% | 14.7% | 0.68 | – |
| In-sample window: 252 days | 2.33 | 10.9% | 14.0% | 0.71 | +0.03 (-0.09 to +0.15) |
| In-sample window: 756 days | 2.38 | 12.1% | 14.6% | 0.75 | +0.10 (+0.00 to +0.19) |
| Shrinkage towards constant correlation | 2.36 | 10.9% | 14.6% | 0.68 | +0.01 (-0.00 to +0.02) |
| Return grows with volatility ** 0.5 | 2.25 | 9.9% | 13.2% | 0.67 | -0.00 (-0.15 to +0.17) |
| Return grows with volatility ** 0 (minimum variance) | 2.01 | 11.0% | 12.7% | 0.77 | +0.09 (-0.13 to +0.35) |

Each row changes one thing and keeps the rest. The interval of the difference comes from redrawing the days in blocks (see `sharpe_difference` in maxdiv/metrics.py); if it contains 0 the sample cannot tell the two apart. Picking the best row of this table would be choosing with hindsight: the next table does not.
