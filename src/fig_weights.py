"""Effective asset allocation of the headline models through time (stacked areas).

Thesis figures:
  weights1.png  (fig:weights_static) the two monthly headline models: equal weight and
                the return-space ridge
  weights2.png  (fig:weights_dyn) the two quarterly headline models, which share the
                inner tracker: the fixed 1-year window and the tuned bandit, with the
                bandit's training/test split marked

The two thesis figures differ slightly in style: weights1.png uses plain Matplotlib
defaults (a box around each panel), weights2.png the house style (no top or right
axis line). Both are kept as they are so each reproduces the thesis file.

Run:    python src/fig_weights.py
Input:  paths.FINAL/weights/<run>.csv (effective daily weights)
Output: paths.FIGURES/weights1.png, paths.FIGURES/weights2.png
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # write files only, never open windows
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402

import paths  # noqa: E402
from fig_headline_models import figure_style, mark_split, save  # noqa: E402

# One figure per pair so each fits on a page
PANEL_1 = [  # the two monthly headline models: a static and a high-turnover fit
    ("equal_weight__ma__reb21__tc0",           "Equal weight (monthly)"),
    ("ridge_returns__ma__w252__reb21__tc0",    "Return-space ridge (monthly)"),
]
PANEL_2 = [  # the two quarterly headline models, which share the inner tracker
    ("window_fixedbest__ma__w252__reb63__tc0", "Fixed 1-year window (quarterly)"),
    ("rl_tuned__ma__reb63__tc0",               "Tuned bandit (quarterly)"),
]


def render(runs, name, suptitle, split=False):
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.4 * n), sharex=False)
    handles = labels = None
    for ax, (run, label) in zip(axes, runs):
        w = pd.read_csv(paths.FINAL / "weights" / f"{run}.csv",
                        index_col=0, parse_dates=True).dropna(how="all")
        ax.stackplot(w.index, w.T.values, labels=w.columns,
                     alpha=0.85, colors=plt.cm.tab20.colors[:len(w.columns)])
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Weight")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        if split:
            mark_split(ax, zorder=5)
        if handles is None:   # every weights file has the same 11 assets in the same order
            handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    # One shared legend below the panels, so no legend covers the top of a stack
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=6, fontsize=9, frameon=False)
    save(fig, name, dpi=140, bbox_inches="tight")


def main():
    with figure_style({}):   # plain Matplotlib defaults, as in the thesis figure
        render(PANEL_1, "weights1.png", "Effective Asset Allocation: Monthly Headline Models")
    with figure_style():
        render(PANEL_2, "weights2.png", "Effective Asset Allocation: Quarterly Headline Models",
               split=True)


if __name__ == "__main__":
    main()
