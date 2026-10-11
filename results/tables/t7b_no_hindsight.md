### Sharpe ratio: portfolios that are not chosen with hindsight

| Portfolio | Return per year | Volatility | Sharpe ratio | Worst drawdown | Turnover per year | Sharpe minus baseline (95% interval) | Sharpe minus SPY (95% interval) |
|---|---|---|---|---|---|---|---|
| Baseline MDP | 10.8% | 14.7% | 0.68 | -33.5% | 56% | – | -0.12 (-0.37 to +0.13) |
| Average of the 6 versions of the MDP | 11.5% | 14.3% | 0.73 | -31.7% | 52% | +0.05 (+0.00 to +0.11) | -0.07 (-0.31 to +0.17) |
| Best of the 6 versions of the MDP so far | 12.2% | 14.6% | 0.76 | -32.2% | 50% | +0.08 (+0.01 to +0.15) | -0.04 (-0.30 to +0.22) |
| Best of 18 candidates so far (MDP to minimum variance) | 10.2% | 14.4% | 0.64 | -34.2% | 60% | -0.03 (-0.19 to +0.13) | -0.15 (-0.42 to +0.12) |
| S&P 500 ETF (SPY) | 14.5% | 17.0% | 0.80 | -33.7% | – | +0.12 (-0.13 to +0.37) | – |
| Baseline MDP + 2% dividend for the delisted companies | 11.3% | 14.7% | 0.70 | -33.5% | 56% | – | -0.09 (-0.35 to +0.16) |
| Best of the 6 versions of the MDP so far + 2% dividend for the delisted companies | 12.3% | 14.6% | 0.77 | -32.2% | 51% | +0.07 (+0.00 to +0.14) | -0.03 (-0.29 to +0.23) |

'Average': the mean of the weights of the versions of the MDP (three in-sample windows, two shrinkage targets); nothing is chosen. 'Best so far': on each optimisation date, the candidate with the highest Sharpe ratio on its own out-of-sample returns up to that day; the first 3 years hold the baseline. The last two rows repeat the baseline and the choice between versions of the MDP giving the delisted companies the dividends that their prices lack (an assumption, see t5b); their 'minus baseline' is against the baseline with the same dividends. Interval above 0: Average of the 6 versions of the MDP; Best of the 6 versions of the MDP so far; Best of the 6 versions of the MDP so far + 2% dividend for the delisted companies. Several portfolios are compared with the baseline, so an interval that only just leaves 0 out is weak evidence.
