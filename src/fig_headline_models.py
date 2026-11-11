"""TRI overlay and rolling tracking error of the four headline models.

Thesis figures:
  tri_overlay_house.png         (fig:tri_overlay) the headline models against the peer-average benchmark,
                                total return indices rebased to 100 on 28 January 2009
  rolling_te_overlay_house.png  (fig:rolling_te) rolling one-year annualised tracking error

This module also holds what the other figure scripts share: the four headline models
with their labels and colours, the house plotting style, and the dashed line that marks
the bandit's training/test split (the bandit's settings were chosen on 2009-2016 and
tested on 2017-2024).

Run:    python src/fig_headline_models.py
Input:  paths.FINAL/timeseries/<run>.csv
Output: paths.FIGURES/tri_overlay_house.png, paths.FIGURES/rolling_te_overlay_house.png
"""
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")   # write files only, never open windows
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402

REBASE_DATE = pd.Timestamp("2009-01-28")   # first possible bandit decision, common start of the runs
SPLIT_DATE = pd.Timestamp("2017-01-01")    # start of the test period

# The four headline models: (run, legend label, colour, line style)
HEADLINE_MODELS = [
    ("equal_weight__ma__reb21__tc0",           "Equal weight (monthly)",          "#7f7f7f", "-"),
    ("ridge_returns__ma__w252__reb21__tc0",    "Return-space ridge (monthly)",    "#2ca02c", "-"),
    ("window_fixedbest__ma__w252__reb63__tc0", "Fixed 1-year window (quarterly)", "#ff7f0e", "-"),
    ("rl_tuned__ma__reb63__tc0",               "Tuned bandit (quarterly)",        "#6a0dad", "-"),
]
EMPHASIS = {"rl_tuned__ma__reb63__tc0"}    # drawn thicker and on top

# House style of the thesis figures, applied on top of Matplotlib's defaults
HOUSE_STYLE = {
    "figure.figsize": (10, 5), "figure.dpi": 120,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "-", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False, "legend.fontsize": 9,
}


@contextmanager
def figure_style(rc=None):
    """Draw with Matplotlib's defaults plus `rc` (the house style by default), whatever
    the caller's settings, and restore the caller's settings afterwards. Pass rc={} for
    plain Matplotlib defaults."""
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update(HOUSE_STYLE if rc is None else rc)
        yield


def load_timeseries(run):
    """A run's daily timeseries (portfolio_tri, benchmark_tri, rolling_te_1y, ...)."""
    df = pd.read_csv(paths.FINAL / "timeseries" / f"{run}.csv", parse_dates=["datestamp"])
    return df.set_index("datestamp").sort_index()


def save(fig, name, **kwargs):
    """Save a figure to paths.FIGURES and close it."""
    out = paths.ensure(paths.FIGURES) / name
    fig.savefig(out, **kwargs)
    plt.close(fig)
    print(f"wrote {out}")


def mark_split(ax, x=SPLIT_DATE, zorder=1.5, fontsize=8, gap=0.015):
    """Dashed vertical line at x (the split date, or a number on a numeric axis),
    labelled "bandit training | test" near the top of the axes with the "|" on the line.
    Call it after ax.legend(): if the legend covers the top of the line, the label goes
    just below the legend and the line stops at the label. Works on a log y-axis."""
    fig = ax.figure
    fig.canvas.draw()   # fix the axis limits and the legend position
    renderer = fig.canvas.get_renderer()
    widths = {}
    for key, text in (("head", "bandit training |"), ("bar", "|"),
                      ("full", "bandit training | test")):
        probe = ax.text(0, 0, text, fontsize=fontsize, transform=ax.transAxes)
        ext = probe.get_window_extent(renderer)
        widths[key], height = ext.width, ext.height
        probe.remove()
    # Shift the label left by the width of "bandit training " plus half the "|"
    shift_px = widths["head"] - widths["bar"] / 2
    xn = mdates.date2num(x) if isinstance(x, pd.Timestamp) else x
    x_px = ax.get_xaxis_transform().transform((xn, 0))[0]
    left_px, right_px = x_px - shift_px, x_px - shift_px + widths["full"]
    y = 1 - gap
    top_px = ax.transAxes.transform((0, y))[1]
    legend = ax.get_legend()
    if legend is not None:
        lb = legend.get_window_extent(renderer)
        if lb.x0 < right_px and left_px < lb.x1 and lb.y0 < top_px and top_px - height < lb.y1:
            y = ax.transAxes.inverted().transform((0, lb.y0))[1] - gap
    ax.axvline(x, ymax=y, color="0.25", lw=0.9, linestyle=(0, (4, 3)), zorder=zorder)
    ax.annotate("bandit training | test",
                xy=(xn, y), xycoords=ax.get_xaxis_transform(),
                xytext=(-shift_px * 72.0 / fig.dpi, 0), textcoords="offset points",
                ha="left", va="top", fontsize=fontsize, color="0.25", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def _rebased(values):
    return values / values.iloc[0] * 100.0


def figure_tri_overlay(series):
    fig, ax = plt.subplots(figsize=(11, 6))
    # The benchmark is the same in every run file: take it from the first model
    bench = _rebased(series[HEADLINE_MODELS[0][0]].loc[REBASE_DATE:, "benchmark_tri"])
    ax.plot(bench.index, bench.values, lw=1.6, color="black", linestyle=":",
            label="Benchmark (peer average)", zorder=4)
    for run, label, colour, ls in HEADLINE_MODELS:
        tri = _rebased(series[run].loc[REBASE_DATE:, "portfolio_tri"])
        emph = run in EMPHASIS
        ax.plot(tri.index, tri.values, lw=1.5 if emph else 1.1, color=colour,
                linestyle=ls, label=label, zorder=3 if emph else 2)
    ax.set_title("Main Replication Portfolios vs Benchmark TRI (rebased to 100 on 28 January 2009)")
    ax.set_ylabel("Total return index (rebased to 100)")
    ax.legend(loc="upper left", ncol=1)
    mark_split(ax)
    save(fig, "tri_overlay_house.png")


def figure_rolling_te(series):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for run, label, colour, ls in HEADLINE_MODELS:
        te = series[run].loc[REBASE_DATE:, "rolling_te_1y"].dropna() * 100.0
        emph = run in EMPHASIS
        ax.plot(te.index, te.values, lw=1.5 if emph else 1.1, color=colour,
                linestyle=ls, label=label, zorder=3 if emph else 2)
    ax.set_title("Rolling One-Year Annualised Tracking Error")
    ax.set_ylabel("Rolling 1Y tracking error (%)")
    # Legend below the axes so it covers no lines
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=9)
    mark_split(ax)
    save(fig, "rolling_te_overlay_house.png")


def main():
    series = {run: load_timeseries(run) for run, *_ in HEADLINE_MODELS}
    common_start = max(s.index.min() for s in series.values())
    if common_start != REBASE_DATE:
        raise SystemExit(f"Common start {common_start.date()} is not the rebase date {REBASE_DATE.date()}")
    with figure_style():
        figure_tri_overlay(series)
        figure_rolling_te(series)


if __name__ == "__main__":
    main()
