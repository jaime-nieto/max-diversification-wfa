### Robustness: one change at a time to the baseline design

| Design | Windows | MDP: DR in-sample | MDP: DR out-of-sample | Efficiency | Efficiency, like for like | EW: DR out-of-sample | Windows with MDP above EW | Stocks held | Turnover per year | MDP return per year | MDP volatility | MDP Sharpe | EW return per year | EW Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 17 | 2.62 | 2.36 | 90% | 90% | 1.86 | 100% | 45 | 56% | 10.8% | 14.7% | 0.68 | 12.8% | 0.69 |
| In-sample window: 252 days | 17 | 2.92 | 2.33 | 80% | 80% | 1.86 | 100% | 46 | 78% | 10.9% | 14.0% | 0.71 | 12.8% | 0.69 |
| In-sample window: 756 days | 16 | 2.50 | 2.38 | 95% | 94% | 1.89 | 100% | 45 | 50% | 12.1% | 14.6% | 0.75 | 12.7% | 0.69 |
| Out-of-sample window: 63 days | 68 | 2.66 | 2.51 | 94% | 92% | 1.97 | 99% | 45 | 106% | 11.4% | 14.1% | 0.73 | 12.7% | 0.68 |
| Out-of-sample window: 126 days | 34 | 2.64 | 2.43 | 92% | 91% | 1.89 | 100% | 45 | 79% | 11.1% | 14.1% | 0.72 | 12.8% | 0.68 |
| No weight cap | 17 | 2.65 | 2.34 | 88% | 88% | 1.86 | 100% | 41 | 58% | 10.9% | 14.8% | 0.67 | 12.8% | 0.69 |
| Weight cap: 10% | 17 | 2.65 | 2.35 | 89% | 88% | 1.86 | 100% | 41 | 58% | 10.8% | 14.8% | 0.67 | 12.8% | 0.69 |
| Weight cap: 2% | 17 | 2.48 | 2.34 | 94% | 93% | 1.86 | 100% | 66 | 51% | 10.8% | 14.6% | 0.67 | 12.8% | 0.69 |

Each row changes one parameter and keeps the rest. DR is measured over the out-of-sample window of that row. Returns after trading costs.
