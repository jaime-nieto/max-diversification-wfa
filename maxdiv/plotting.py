"""
All charts. Each strategy keeps the same colour in every figure, and no chart
uses two y-axes: series with different units go in separate panels.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter

from . import config
from .metrics import drawdown
from .reporting import LABELS

# Colour = identity of the strategy (fixed order of a colour-blind-checked palette);
# the market reference is drawn in neutral ink.
COLORS = {"MDP": "#2a78d6", "MinVar": "#eb6834", "EW": "#1baf7a", "SPY": "#52514e", "RSP": "#8b5cf6"}
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8985"
SURFACE, GRID, SHADE = "#fcfcfb", "#e4e3df", "#ebeae6"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.titlesize": 11, "axes.titlelocation": "left", "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": INK_3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 1.6,
    })


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _date_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_design(windows, dates, design):
    """How the walk-forward works: one window in detail, and every window of the run on the calendar.

    `dates` is the trading calendar, needed to find where each in-sample window starts.
    """
    w = windows.drop_duplicates("optimised")
    train, test, lag = design.train_days, design.test_days, config.EXECUTION_LAG
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7.6), gridspec_kw={"height_ratios": [1, 3.3]})

    # One window. The lag is drawn wider than it is so that it can be seen.
    gap = 0.025 * (train + test)
    a.barh(0, train, left=0, height=0.6, color=SHADE, edgecolor=INK_3, lw=0.8)
    a.barh(0, test, left=train + gap, height=0.6, color=COLORS["MDP"])
    a.text(train / 2, 0, f"In-sample: {train} trading days", ha="center", va="center")
    a.text(train + gap + test / 2, 0, f"Out-of-sample: {test} trading days", ha="center",
           va="center", color="white")
    for x, align, note in [
            (train, "right", "Day t: the weights are computed,\nwith these returns only"),
            (train + gap, "left", f"Day t + {lag}: the portfolio is bought at the close\n"
                                  "and held untouched for these days")]:
        a.plot([x, x], [-0.3, -0.52], color=INK_3, lw=0.8)
        a.text(x, -0.6, note, ha=align, va="top", color=INK_2, fontsize=9)
    a.set_xlim(0, train + gap + test)
    a.set_ylim(-1.2, 0.4)
    a.set_title("One walk-forward window")
    a.axis("off")
    a.title.set_visible(True)

    # Every window of the run, on the calendar
    fitted = dates.get_indexer(w["optimised"])
    in_sample_start = mdates.date2num(dates[fitted - train + 1])
    in_sample_end = mdates.date2num(w["optimised"])
    first, last = mdates.date2num(w["first_day"]), mdates.date2num(w["last_day"])
    rows = np.arange(len(w))
    joined = len(w) + 0.5
    b.barh(rows, in_sample_end - in_sample_start, left=in_sample_start, height=0.62,
           color=SHADE, edgecolor=INK_3, lw=0.6)
    b.barh(rows, last - first, left=first, height=0.62, color=COLORS["MDP"])
    b.barh(joined, last - first, left=first, height=0.62, color=COLORS["MDP"], edgecolor=SURFACE, lw=0.8)
    b.set_yticks(list(rows) + [joined], [str(k + 1) for k in rows] + ["Backtest"], fontsize=8.5)
    b.invert_yaxis()
    b.set_ylabel("Window")
    b.grid(axis="y", visible=False)
    b.grid(axis="x")
    b.set_axisbelow(True)
    b.set_xlim(in_sample_start.min() - 150, last.max() + 150)
    b.xaxis_date()
    _date_axis(b)
    incomplete = " (the last one is not complete)" if w["dr_out"].isna().any() else ""
    b.set_title(f"The {len(w)} windows of this study{incomplete}. Each one starts {test} trading days "
                "after the\nprevious one; the out-of-sample periods, one after another, are the backtest")
    fig.tight_layout()
    _save(fig, "fig00_walk_forward_design.png")


def plot_walk_forward(windows):
    """In-sample vs out-of-sample DR of the MDP, and out-of-sample DR of every strategy."""
    w = windows.dropna(subset=["dr_out"])
    dr_in = w.pivot(index="first_day", columns="strategy", values="dr_in")
    dr_out = w.pivot(index="first_day", columns="strategy", values="dr_out")
    x = dr_out.index

    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True, sharey=True)
    a.fill_between(x, dr_out["MDP"], dr_in["MDP"], color=SHADE, lw=0,
                   label="Gap between the two")
    a.plot(x, dr_in["MDP"], color=INK_3, ls="--", lw=1.4,
            label="In-sample: on the data the weights were fitted to")
    a.plot(x, dr_out["MDP"], color=COLORS["MDP"], lw=2,
           label="Out-of-sample: on the following window")
    a.set_title("Most Diversified Portfolio: diversification ratio in each walk-forward window")
    a.legend(loc="upper left", ncol=1)

    for s in ["MDP", "MinVar", "EW"]:
        b.plot(x, dr_out[s], color=COLORS[s], lw=2 if s == "MDP" else 1.6, label=LABELS[s])
    b.set_title("Out-of-sample diversification ratio: MDP vs two alternatives built from the same stocks")
    b.legend(loc="upper left", ncol=3)
    for ax in (a, b):
        ax.set_ylabel("Diversification ratio")
        ax.set_ylim(bottom=1)
    b.set_xlabel("First day of the out-of-sample window")
    _date_axis(b)
    fig.tight_layout()
    _save(fig, "fig01_walk_forward_dr.png")


def plot_performance(returns):
    """Growth of 1 dollar (log scale) and drawdowns, after costs."""
    order = [c for c in ["MDP", "MinVar", "EW", "SPY", "RSP"] if c in returns]
    wealth = (1 + returns[order]).cumprod()

    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    for s in order:
        a.plot(wealth.index, wealth[s], color=COLORS.get(s, INK_3), lw=2 if s == "MDP" else 1.5,
               label=f"{LABELS.get(s, s)}: {wealth[s].iloc[-1]:.2f} $")
        b.plot(wealth.index, drawdown(returns[s]), color=COLORS.get(s, INK_3), lw=1.6 if s == "MDP" else 1.1)
    a.set_yscale("log")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} $"))
    a.yaxis.set_minor_locator(NullLocator())
    a.set_title(f"Out-of-sample growth of 1 $ invested on {wealth.index[0]:%d %b %Y} "
                "(after trading costs, log scale)")
    a.legend(loc="upper left", title="Value at the end of the sample", alignment="left")
    b.set_title("Drawdown from the previous peak")
    b.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _date_axis(b)
    fig.tight_layout()
    _save(fig, "fig02_performance.png")


def plot_portfolio(windows):
    """Size of the eligible universe and of the MDP through time."""
    mdp = windows[windows.strategy == "MDP"].set_index("optimised")

    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    a.plot(mdp.index, mdp["eligible"], color=INK_2, lw=2)
    a.set_ylim(bottom=0)
    a.set_title("Stocks the optimiser can choose from at each optimisation date")
    b.plot(mdp.index, mdp["held"], color=COLORS["MDP"], lw=2, label="Stocks held")
    b.plot(mdp.index, mdp["effective_n"], color=COLORS["MDP"], lw=1.4, ls="--",
           label="Effective number of stocks (1 / sum of squared weights)")
    b.set_ylim(bottom=0)
    b.set_title("Most Diversified Portfolio: number of holdings")
    b.legend(loc="upper left")
    _date_axis(b)
    fig.tight_layout()
    _save(fig, "fig03_portfolio.png")


def plot_sectors(sector_weights, index_share):
    """One small panel per sector: MDP weight through time vs the sector's share of the index."""
    order = sector_weights.mean().sort_values(ascending=False).index[:11]
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, sector in zip(axes, order):
        if sector in index_share:
            ax.plot(index_share.index, index_share[sector], color=INK_3, lw=1.1, ls=":")
        ax.plot(sector_weights.index, sector_weights[sector], color=COLORS["MDP"], lw=1.6)
        ax.set_title(sector, fontsize=9.5)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    key = axes[-1]
    key.plot([], [], color=COLORS["MDP"], lw=1.6, label="MDP weight")
    key.plot([], [], color=INK_3, lw=1.1, ls=":", label="Share of the index members")
    key.legend(loc="center")
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.suptitle("Most Diversified Portfolio: sector weights at each optimisation date",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig04_sectors.png")


def plot_robustness(table):
    """Out-of-sample DR of the MDP and of equal weight under each alternative design."""
    t = table.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 1.6))
    ax.hlines(t.index, t["ew_dr_out"], t["mdp_dr_out"], color=GRID, lw=2, zorder=1)
    ax.scatter(t["mdp_dr_in"], t.index, s=60, facecolor=SURFACE, edgecolor=INK_3, lw=1.2,
               zorder=2, label="MDP, in-sample")
    ax.scatter(t["ew_dr_out"], t.index, s=60, color=COLORS["EW"], zorder=3,
               label="Equal weight, out-of-sample")
    ax.scatter(t["mdp_dr_out"], t.index, s=60, color=COLORS["MDP"], zorder=3,
               label="MDP, out-of-sample")
    for i, row in t.iterrows():
        ax.annotate(f"{row['mdp_dr_out']:.2f}", (row["mdp_dr_out"], i), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(t.index, t["design"])
    ax.set_xlim(left=1)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average diversification ratio across walk-forward windows")
    ax.set_title("Robustness: the result under one-at-a-time changes to the design")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    fig.tight_layout()
    _save(fig, "fig05_robustness.png")


def plot_survivorship(curves):
    """Growth of 1 dollar with the true index members vs with today's survivors only."""
    wealth = (1 + curves).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    styles = {"all members": "-", "survivors only": "--"}
    for column in wealth:
        strategy, u = column
        ax.plot(wealth.index, wealth[column], color=COLORS.get(strategy, INK_2), ls=styles[u],
                lw=2 if u == "all members" else 1.5,
                label=f"{LABELS.get(strategy, strategy)}, {u}: {wealth[column].iloc[-1]:.2f} $")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} $"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_title("Survivorship bias: the same portfolios without the companies that later disappeared")
    ax.legend(loc="upper left", title="Value at the end of the sample", alignment="left")
    _date_axis(ax)
    fig.tight_layout()
    _save(fig, "fig06_survivorship.png")


def plot_sharpe(table):
    """Sharpe ratio of each alternative minus that of the baseline MDP, with its 95% interval."""
    t = table.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 1.6))
    ax.axvline(0, color=INK_3, lw=1)
    hindsight = t["kind"] == "one change"
    ax.hlines(t.index, t["low"], t["high"], color=GRID, lw=3, zorder=1)
    ax.scatter(t["difference"][hindsight], t.index[hindsight], s=60, facecolor=SURFACE,
               edgecolor=INK_2, lw=1.3, zorder=3, label="One change to the baseline, whole sample")
    ax.scatter(t["difference"][~hindsight], t.index[~hindsight], s=60, color=COLORS["MDP"], zorder=3,
               label="No choice made with hindsight")
    for i, row in t.iterrows():
        ax.annotate(f"{row['difference']:+.2f}", (row["difference"], i), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(t.index, t["portfolio"])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Sharpe ratio minus the Sharpe ratio of the baseline MDP (bar: 95% interval)")
    ax.set_title("Can the Sharpe ratio be improved? Difference with the baseline MDP")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    _save(fig, "fig07_sharpe.png")
