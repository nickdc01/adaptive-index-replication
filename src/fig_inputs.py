"""Figures of the inputs and of the tuned bandit's fit to the benchmark.

Thesis figures:
  Assets_house.png            (fig:asset_tri) total return indices of the 11 asset
                              classes, rebased to 1.0
  benchmark_select_house.png  (fig:benchmark_select) the peer-category average against
                              the rolling top-10 average
  monthly_r2_house.png        (fig:monthly_r2) 21-day non-overlapping returns of the tuned
                              bandit against the benchmark, with the OLS fit (beta, alpha, R^2)

Run:    python src/fig_inputs.py
Inputs: paths.DATA (sheets X, Y and "top10 av"),
        paths.FINAL/timeseries/rl_tuned__ma__reb63__tc0.csv
Output: the three PNGs in paths.FIGURES; paths.FIGURE_DATA/monthly_r2_house.csv holds
        the OLS beta, alpha and R^2 behind monthly_r2_house.png.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # write files only, never open windows
import matplotlib.pyplot as plt  # noqa: E402

import paths  # noqa: E402
from fig_headline_models import figure_style, load_timeseries, save  # noqa: E402

warnings.filterwarnings("ignore")

HEADLINE = "rl_tuned__ma__reb63__tc0"   # tuned bandit (quarterly)
# The 11th asset would repeat the first colour of Matplotlib's 10-colour cycle; the
# thesis figure draws it in black.
ASSET_COLOURS = {"Hedged EM Equities": "black"}


def returns_to_tri(df_returns, value_col):
    """Daily returns indexed by datestamp -> TRI starting at 1.0."""
    s = df_returns.set_index("datestamp")[value_col].sort_index()
    return (1.0 + s).cumprod()


def figure_assets():
    X = pd.read_excel(paths.DATA, sheet_name="X").set_index("datestamp").sort_index()
    X = X / X.iloc[0]   # rebase to 1 on the first date
    fig, ax = plt.subplots()
    for col in X.columns:
        ax.plot(X.index, X[col], lw=1.0, label=col, color=ASSET_COLOURS.get(col))
    ax.set_title("Total Return Indices: Constituent Asset Classes (rebased to 1.0)")
    ax.set_ylabel("TRI")
    ax.set_xlabel("")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    save(fig, "Assets_house.png")


def figure_benchmark_select():
    Y = pd.read_excel(paths.DATA, sheet_name="Y")
    top10 = pd.read_excel(paths.DATA, sheet_name="top10 av").dropna()
    avg_tri = returns_to_tri(Y, "value")
    top10_tri = returns_to_tri(top10.rename(columns={"top10": "value"}), "value")
    # Rebase both to 1.0 on the first common date from 2006 (the thesis sample period)
    common = avg_tri.index.intersection(top10_tri.index)
    common = common[common >= pd.Timestamp("2006-01-01")]
    avg_tri = avg_tri.loc[common] / avg_tri.loc[common].iloc[0]
    top10_tri = top10_tri.loc[common] / top10_tri.loc[common].iloc[0]

    fig, ax = plt.subplots()
    ax.plot(avg_tri.index, avg_tri.values, lw=1.2, label="Peer category average", color="#1f4e79")
    ax.plot(top10_tri.index, top10_tri.values, lw=1.2, label="Rolling top-10 average", color="#c55a11")
    ax.set_title("Peer Category Average vs Rolling Top-10 Average")
    ax.set_ylabel("Total Return Index (rebased to 1.0)")
    ax.legend(loc="upper left")
    save(fig, "benchmark_select_house.png")


def figure_monthly_r2():
    """Scatter of 21-day non-overlapping tuned-bandit vs benchmark returns, OLS fit."""
    df = load_timeseries(HEADLINE)
    pr = df["portfolio_tri"].iloc[::21].pct_change().dropna()
    br = df["benchmark_tri"].iloc[::21].pct_change().dropna()
    common = pr.index.intersection(br.index)
    x, y = br.loc[common].values, pr.loc[common].values
    beta, alpha = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(x, y, s=12, alpha=0.5, color="#1f4e79", edgecolor="none", label="21-day returns")
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, alpha + beta * xr, color="#c0392b", lw=1.6,
            label=f"OLS fit: $\\beta$ = {beta:.3f}, $\\alpha$ = {alpha:.4f}, $R^2$ = {r2:.3f}")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("21-day benchmark return")
    ax.set_ylabel("21-day tuned bandit return")
    ax.set_title("Tuned bandit (quarterly) vs benchmark: 21-day non-overlapping returns")
    ax.legend(loc="lower right")
    save(fig, "monthly_r2_house.png")
    out = paths.ensure(paths.FIGURE_DATA) / "monthly_r2_house.csv"
    pd.DataFrame([dict(run_name=HEADLINE, beta=beta, alpha=alpha, r2=r2, n_obs=int(len(x)),
                       first_date=common.min().date(), last_date=common.max().date())]
                 ).to_csv(out, index=False)
    print(f"wrote {out} (beta={beta:.3f}, alpha={alpha:.4f}, R2={r2:.3f})")


def main():
    with figure_style():
        figure_assets()
        figure_benchmark_select()
        figure_monthly_r2()


if __name__ == "__main__":
    main()
