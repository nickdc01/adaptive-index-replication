"""Window choices of the tuned bandit (quarterly).

Left: the window chosen at each of its 58 quarterly decisions, with the trailing 63-day
benchmark volatility the bandit sees in its context. Right: how often each window is
chosen in the training (2009-2016) and test (2017-2023) periods. Also prints the
correlation between the window and the volatility.

Thesis figure: tuned_windows.png (fig:tuned_windows)
Run:    python src/fig_tuned_windows.py
Inputs: paths.FINAL/rl_diagnostics/rl_tuned__ma__reb63__tc0.csv, paths.DATA (sheet Y,
        through run_models.load_data)
Output: paths.FIGURES/tuned_windows.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # write files only, never open windows
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy import stats  # noqa: E402

import paths  # noqa: E402
from fig_headline_models import HOUSE_STYLE, figure_style, save  # noqa: E402
from run_models import load_data  # noqa: E402

RUN = "rl_tuned__ma__reb63__tc0"
SPLIT = pd.Timestamp("2017-01-01")
WINDOWS = [126, 252, 378, 504, 756]
LABELS = ["0.5", "1", "1.5", "2", "3"]
STYLE = {**HOUSE_STYLE, "axes.titlesize": 11}


def bench_vol_asof(Y, date):
    """Std of the (up to) 63 benchmark returns strictly before the decision date, as in
    the bandit's context."""
    prior = Y.loc[Y.index < date]
    return float(prior.iloc[-63:].std()) if len(prior) >= 10 else np.nan


def plot(d):
    pos = {w: k for k, w in enumerate(WINDOWS)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [2.3, 1]})

    # Left: window chosen at each decision, with the trailing volatility on a second axis
    y = d["chosen_window"].map(pos)
    ax1.step(d["datestamp"], y, where="post", color="#1f3a68", lw=1.6, label="Window chosen (left axis)")
    ax1.plot(d["datestamp"], y, "o", color="#1f3a68", ms=3)
    ax1.axvline(SPLIT, color="grey", ls="--", lw=1)
    ax1.text(SPLIT, len(WINDOWS) - 0.55, "training | test", ha="center", va="bottom", fontsize=9,
             color="#444444", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax1.set_yticks(range(len(WINDOWS)), LABELS)
    ax1.set_ylim(-0.5, len(WINDOWS) - 0.2)
    ax1.set_ylabel("Lookback window chosen (years)")
    ax1.set_title("Window chosen at each quarterly decision")
    ax1b = ax1.twinx()
    ax1b.plot(d["datestamp"], d["vol"], color="#c0504d", lw=1.1, alpha=0.8,
              label="Trailing 63-day benchmark volatility (right axis)")
    ax1b.set_ylabel("Benchmark volatility (% a year)", color="#c0504d")
    ax1b.tick_params(axis="y", colors="#c0504d")
    ax1b.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Right: frequency by period
    train = d[d["datestamp"] < SPLIT]["chosen_window"].value_counts().reindex(WINDOWS, fill_value=0)
    test = d[d["datestamp"] >= SPLIT]["chosen_window"].value_counts().reindex(WINDOWS, fill_value=0)
    x = np.arange(len(WINDOWS))
    ax2.bar(x - 0.2, train.values, 0.4, color="#8fa8c8", label=f"Training period ({int(train.sum())} decisions)")
    ax2.bar(x + 0.2, test.values, 0.4, color="#1f3a68", label=f"Test period ({int(test.sum())} decisions)")
    ax2.set_xticks(x, LABELS)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim(0, 27)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlabel("Lookback window (years)")
    ax2.set_ylabel("Number of decisions")
    ax2.set_title("How often each window is chosen")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    save(fig, "tuned_windows.png")
    return train, test


def main():
    d = pd.read_csv(paths.FINAL / "rl_diagnostics" / f"{RUN}.csv", parse_dates=["datestamp"])
    assert len(d) == 58, len(d)
    _, Y = load_data("ma")
    d["vol"] = [bench_vol_asof(Y, t) * np.sqrt(252) * 100 for t in d["datestamp"]]

    with figure_style(STYLE):
        train, test = plot(d)

    r, p = stats.pearsonr(d["vol"], d["chosen_window"])
    med = d["vol"].median()
    hi = d.loc[d["vol"] > med, "chosen_window"].mean()
    lo = d.loc[d["vol"] <= med, "chosen_window"].mean()
    print("training", train.to_dict(), "test", test.to_dict())
    print(f"pearson r={r:.3f} p={p:.3f}; mean window high-vol {hi:.0f}d, calm {lo:.0f}d; "
          f"mean window all {d.chosen_window.mean():.0f}d, training {d[d.datestamp < SPLIT].chosen_window.mean():.0f}d, "
          f"test {d[d.datestamp >= SPLIT].chosen_window.mean():.0f}d")


if __name__ == "__main__":
    main()
