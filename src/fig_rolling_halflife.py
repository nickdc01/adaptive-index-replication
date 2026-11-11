"""Rolling one-year half-life of the log spread, four headline models.

Definition
  S_t = log(portfolio_tri) - log(benchmark_tri) from each run's tc0 timeseries,
  restricted to the full period 2009-01-28 to 2024-05-15.
  For every trading day t from the first day with WINDOW = 252 prior observations,
  take the trailing 252 daily changes of S (the 253 levels S_{t-252..t}) and run
  exactly the regression metrics.half_life uses:
      Delta S_k = mu + beta * k + phi * S_{k-1} + e_k,   k = 0..251,
  then H_t = -ln 2 / ln(1 + phi) in trading days. When phi >= 0 or 1 + phi <= 0
  the half-life is undefined (no mean reversion in that year) and the line has a
  gap. This uses the same 252 daily changes as the rolling one-year tracking
  error, so both series start on the same date (12 February 2010).
  Unlike metrics.half_life, no upper cap (max_days = 1e4) is applied, so every
  window with -1 < phi < 0 has a value; the number of values above that cap is
  reported in the summary.

Each value is estimated within its own year, so it is in-sample for that window
and noisy. A shorter half-life means the spread returns to its trend faster.

Reference level. In a one-year window the regression is biased towards mean
reversion (the Dickey-Fuller small-sample bias, made larger by the trend term):
a spread with NO mean reversion at all (a pure random walk) still gives a defined
half-life in almost every window, typically around 19 trading days. The script
simulates that reference with the same regression (NULL_REPS random walks of
WINDOW + 1 levels) and draws its median on the chart as a dotted grey line.

Robustness (summary CSV only, prefix aug5_). Daily changes of S are negatively
autocorrelated (part of each day's gap reverses the next day), which also pulls
the one-year half-life down. The aug5_ columns repeat the calculation with five
lags of the change in S added to the regression (an ADF-style augmentation) and
compare with a random-walk reference simulated the same way.

Thesis figure: rolling_halflife_house.png (fig:rolling_halflife)
Run:    python src/fig_rolling_halflife.py        (about 20 s: 100,000 simulated random walks)
Inputs: paths.FINAL/timeseries/<run>.csv and paths.FINAL/period_tables.csv
Output: paths.FIGURES/rolling_halflife_house.png; in paths.FIGURE_DATA
        rolling_halflife_daily.csv (daily phi and half-life of each model) and
        rolling_halflife_summary.csv (the numbers quoted in the text, plus the period
        half-lives and ADF/ZA p-values from period_tables.csv)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # write files only, never open windows
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter  # noqa: E402

import metrics  # noqa: E402
import paths  # noqa: E402
from fig_headline_models import (  # noqa: E402
    EMPHASIS, HEADLINE_MODELS, figure_style, load_timeseries, mark_split, save)

FULL = (pd.Timestamp("2009-01-28"), pd.Timestamp("2024-05-15"))
TRAIN = (pd.Timestamp("2009-01-28"), pd.Timestamp("2016-12-30"))
TEST = (pd.Timestamp("2017-01-01"), pd.Timestamp("2024-05-15"))
# COVID-19: the crash month through the month the crash leaves a 252-day window
COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-04-30"))
PRE_COVID = pd.Timestamp("2020-02-28")

WINDOW = 252            # daily changes of S in each rolling regression
HL_CAP = 1e4            # metrics.half_life's max_days, reported only
AUG_LAGS = 5            # robustness: lagged changes added to the regression
NULL_REPS = 50000       # random walks behind the reference level
NULL_SEED = 20261022


def log_spread(df):
    return (np.log(df["portfolio_tri"]) - np.log(df["benchmark_tri"])).rename("S")


def to_half_life(phi):
    """H = -ln 2 / ln(1 + phi); NaN when phi >= 0 or 1 + phi <= 0."""
    phi = np.asarray(phi, dtype=float)
    ok = (phi < 0) & (1.0 + phi > 0)
    hl = np.full(phi.shape, np.nan)
    hl[ok] = -np.log(2.0) / np.log1p(phi[ok])
    return hl


def _phi_one(s, e, window, lags):
    """phi from the regression of the `window` changes of s ending at position e on
    a constant, the lagged level, a trend and `lags` lagged changes."""
    lev = s[e - window:e + 1]                          # window + 1 levels
    ds = np.diff(lev)                                  # window changes
    cols = [np.ones(window), lev[:-1], np.arange(window, dtype=float)]
    for j in range(1, lags + 1):                       # change j days before each ds
        cols.append(s[e - window + 1 - j:e + 1 - j] - s[e - window - j:e - j])
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), ds, rcond=None)
    return coef[1]


def rolling_half_life(S, window=WINDOW, lags=0):
    """Daily phi and half-life on the trailing `window` daily changes of S. With
    lags=0 this is exactly the metrics.half_life regression. Returns a DataFrame
    indexed like S (NaN before the first full window and wherever the half-life
    is undefined)."""
    s = S.to_numpy(dtype=float)
    n = len(s)
    phi = np.full(n, np.nan)
    for e in range(window + lags, n):
        phi[e] = _phi_one(s, e, window, lags)
    return pd.DataFrame({"phi": phi, "hl": to_half_life(phi)}, index=S.index)


def null_half_life(window=WINDOW, lags=0, reps=NULL_REPS, seed=NULL_SEED):
    """Half-life the same regression gives on pure random walks (no mean reversion).
    The distribution of phi does not depend on the step size, so standard normal
    steps are enough. Returns (share defined, 5th, 10th, 25th, 50th percentile)."""
    rng = np.random.default_rng(seed)
    phi = np.empty(reps)
    for r in range(reps):
        s = np.concatenate([[0.0], np.cumsum(rng.standard_normal(window + lags))])
        phi[r] = _phi_one(s, len(s) - 1, window, lags)
    hl = to_half_life(phi)
    ok = ~np.isnan(hl)
    return (float(ok.mean()), *[float(np.percentile(hl[ok], p)) for p in (5, 10, 25, 50)])


def check_against_metrics(S, roll, n_checks=40, seed=0):
    """metrics.half_life on the same slices must reproduce the rolling values
    (below its 1e4 cap). Returns the max absolute difference."""
    rng = np.random.default_rng(seed)
    pos = rng.choice(np.arange(WINDOW, len(S)), size=n_checks, replace=False)
    worst = 0.0
    for e in pos:
        ref = metrics.half_life(S.iloc[e - WINDOW:e + 1].to_numpy())
        mine = roll["hl"].iloc[e]
        if np.isnan(ref):
            assert np.isnan(mine) or mine > HL_CAP, (S.index[e], ref, mine)
        else:
            worst = max(worst, abs(ref - mine))
    return worst


def q(x, p):
    x = x.dropna()
    return float(np.percentile(x, p)) if len(x) else float("nan")


def share_defined(hl, lo, hi):
    x = hl.loc[lo:hi]
    return float(x.notna().mean()), int(x.notna().sum()), int(len(x))


def gap_runs(hl):
    """Contiguous runs of undefined half-life: list of (start, end, n_days)."""
    miss = hl.isna().to_numpy()
    idx = hl.index
    runs, start = [], None
    for i, m in enumerate(miss):
        if m and start is None:
            start = i
        if (not m or i == len(miss) - 1) and start is not None:
            end = i - 1 if not m else i
            runs.append((idx[start], idx[end], end - start + 1))
            start = None
    return runs


def plot(daily, summ, rw_median):
    """Same layout as rolling_te_overlay_house.png, on a log scale."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for run, label, colour, ls in HEADLINE_MODELS:
        h = daily[f"{run}__hl"].loc[summ.loc[run, "first_date"]:]
        emph = run in EMPHASIS
        # NaNs left in place so undefined years show as gaps
        ax.plot(h.index, h.values, lw=1.5 if emph else 1.1, color=colour,
                linestyle=ls, label=label, zorder=3 if emph else 2)
    ax.set_yscale("log")
    hl_cols = [f"{run}__hl" for run, *_ in HEADLINE_MODELS]
    ymax = float(np.nanmax(daily[hl_cols].to_numpy()))
    ymin = float(np.nanmin(daily[hl_cols].to_numpy()))
    ax.set_ylim(ymin * 0.8, ymax * 1.4)
    lo, hi = ax.get_ylim()
    # 1 day, 1 week, 1 month, 1 quarter, 1 year, 5 years (trading days)
    ticks = [1, 2, 5, 10, 21, 63, 252, 1260]
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("Rolling One-Year Half-Life of the Log Spread")
    ax.set_ylabel("Rolling 1Y half-life (trading days, log scale)")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=9)
    # Random-walk reference: the median half-life of a spread with no mean reversion
    ref = ax.axhline(rw_median, color="0.35", lw=1.0, linestyle=(0, (1, 2)), zorder=1.6)
    ax.add_artist(leg)
    leg2 = ax.legend([ref], [f"No mean reversion (random walk): median {rw_median:.0f} days"],
                     loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=1, fontsize=9)
    mark_split(ax)
    # both legends sit below the axes; name them so the tight bounding box keeps them
    save(fig, "rolling_halflife_house.png", bbox_extra_artists=[leg, leg2])


def main():
    pt = pd.read_csv(paths.FINAL / "period_tables.csv")

    def ptv(run, window, col):
        v = pt.loc[(pt.run == run) & (pt.window == window), col]
        assert len(v) == 1, (run, window, col)
        return float(v.iloc[0])

    # Random-walk reference for the plain regression and the augmented one
    null0 = null_half_life(lags=0)
    nullA = null_half_life(lags=AUG_LAGS)
    print(f"random walk, plain: share defined {null0[0]:.4f}, half-life 5/10/25/50% "
          f"{null0[1]:.1f} {null0[2]:.1f} {null0[3]:.1f} {null0[4]:.1f}")
    print(f"random walk, {AUG_LAGS} lags: share defined {nullA[0]:.4f}, half-life 5/10/25/50% "
          f"{nullA[1]:.1f} {nullA[2]:.1f} {nullA[3]:.1f} {nullA[4]:.1f}")

    daily, rows = {}, []
    for run, label, *_ in HEADLINE_MODELS:
        df = load_timeseries(run).loc[FULL[0]:FULL[1]]
        S = log_spread(df)
        assert S.index[0] == FULL[0] and S.index[-1] == FULL[1], run

        # The full-period half-life from this S must match period_tables.csv
        hl_full_check = metrics.half_life(S.to_numpy())
        assert abs(hl_full_check - ptv(run, "full", "hl")) < 1e-6, (run, hl_full_check)

        roll = rolling_half_life(S)
        worst = check_against_metrics(S, roll)
        assert worst < 1e-6, (run, worst)
        first = roll["phi"].first_valid_index()
        te_first = df["rolling_te_1y"].first_valid_index()
        assert first == te_first, (run, first, te_first)

        hl = roll["hl"].loc[first:]
        daily[f"{run}__phi"] = roll["phi"]
        daily[f"{run}__hl"] = roll["hl"]

        # Robustness: AUG_LAGS lagged changes added (starts AUG_LAGS days later)
        aug = rolling_half_life(S, lags=AUG_LAGS)
        daily[f"{run}__hl_aug{AUG_LAGS}"] = aug["hl"]
        ha = aug["hl"].loc[first:]
        ha_test = ha.loc[TEST[0]:TEST[1]]
        ha_covid = ha.loc[COVID[0]:COVID[1]]
        dS = S.diff()

        test = hl.loc[TEST[0]:TEST[1]]
        train = hl.loc[:TRAIN[1]]
        covid = hl.loc[COVID[0]:COVID[1]]
        runs = gap_runs(hl)
        longest = max(runs, key=lambda r: r[2]) if runs else (None, None, 0)
        covid_runs = [r for r in runs if r[1] >= COVID[0] and r[0] <= COVID[1]]

        sh_full = share_defined(hl, first, FULL[1])
        sh_train = share_defined(hl, first, TRAIN[1])
        sh_test = share_defined(hl, *TEST)
        sh_covid = share_defined(hl, *COVID)

        rows.append({
            "run": run, "model": label,
            "first_date": first.date().isoformat(),
            "n_days_full": sh_full[2], "n_days_train": sh_train[2], "n_days_test": sh_test[2],
            "share_defined_full": sh_full[0], "share_defined_train": sh_train[0],
            "share_defined_test": sh_test[0],
            "n_defined_full": sh_full[1], "n_defined_test": sh_test[1],
            "median_full": q(hl, 50), "q25_full": q(hl, 25), "q75_full": q(hl, 75),
            "median_train": q(train, 50), "q25_train": q(train, 25), "q75_train": q(train, 75),
            "median_test": q(test, 50), "q25_test": q(test, 25), "q75_test": q(test, 75),
            "share_le_63_of_all_full": float((hl <= 63).mean()),
            "share_le_63_of_all_test": float((test <= 63).mean()),
            "share_gt_252_of_defined_full": float((hl.dropna() > 252).mean()),
            "share_gt_252_of_defined_test": float((test.dropna() > 252).mean()),
            "n_above_metrics_cap_1e4": int((hl > HL_CAP).sum()),
            "min_hl": float(hl.min()), "min_hl_date": hl.idxmin().date().isoformat(),
            "max_hl": float(hl.max()), "max_hl_date": hl.idxmax().date().isoformat(),
            "n_gap_runs": len(runs),
            "longest_gap_days": int(longest[2]),
            "longest_gap_start": longest[0].date().isoformat() if longest[0] is not None else "",
            "longest_gap_end": longest[1].date().isoformat() if longest[1] is not None else "",
            "hl_pre_covid_2020_02_28": float(hl.loc[PRE_COVID]),
            "hl_2020_03_31": float(hl.loc[:"2020-03-31"].iloc[-1]),
            "hl_2020_06_30": float(hl.loc[:"2020-06-30"].iloc[-1]),
            "hl_2020_12_31": float(hl.loc[:"2020-12-31"].iloc[-1]),
            "hl_2021_04_30": float(hl.loc[:"2021-04-30"].iloc[-1]),
            "share_defined_covid": sh_covid[0],
            "median_covid": q(covid, 50), "q25_covid": q(covid, 25), "q75_covid": q(covid, 75),
            "min_covid": float(covid.min()), "max_covid": float(covid.max()),
            "covid_gaps": "; ".join(f"{a.date()}..{b.date()} ({n}d)" for a, b, n in covid_runs),
            "hl_end_2024_05_15": float(hl.iloc[-1]),
            "metrics_check_max_abs_diff": worst,
            "hl_full_pt": ptv(run, "full", "hl"), "hl_train_pt": ptv(run, "train", "hl"),
            "hl_test_pt": ptv(run, "test", "hl"),
            "adf_full": ptv(run, "full", "adf"), "za_full": ptv(run, "full", "za"),
            "adf_train": ptv(run, "train", "adf"), "za_train": ptv(run, "train", "za"),
            "adf_test": ptv(run, "test", "adf"), "za_test": ptv(run, "test", "za"),
            "te_test_pt": ptv(run, "test", "te") * 100.0,
            # random-walk reference (plain regression)
            "rw_share_defined": null0[0], "rw_hl_p05": null0[1], "rw_hl_p10": null0[2],
            "rw_hl_p25": null0[3], "rw_hl_median": null0[4],
            "share_below_rw_median_full": float((hl < null0[4]).mean()),
            "share_below_rw_median_test": float((test < null0[4]).mean()),
            # day-to-day reversal in the spread changes
            "ac1_dS_full": float(dS.autocorr(1)), "ac1_dS_test": float(dS.loc[TEST[0]:].autocorr(1)),
            "ac2_dS_full": float(dS.autocorr(2)),
            # robustness: AUG_LAGS lagged changes in the regression
            f"aug{AUG_LAGS}_share_defined_full": float(ha.loc[ha.index >= aug['phi'].first_valid_index()].notna().mean()),
            f"aug{AUG_LAGS}_share_defined_test": float(ha_test.notna().mean()),
            f"aug{AUG_LAGS}_median_full": q(ha, 50), f"aug{AUG_LAGS}_q25_full": q(ha, 25),
            f"aug{AUG_LAGS}_q75_full": q(ha, 75),
            f"aug{AUG_LAGS}_median_test": q(ha_test, 50), f"aug{AUG_LAGS}_q25_test": q(ha_test, 25),
            f"aug{AUG_LAGS}_q75_test": q(ha_test, 75),
            f"aug{AUG_LAGS}_median_covid": q(ha_covid, 50),
            f"aug{AUG_LAGS}_rw_hl_p05": nullA[1], f"aug{AUG_LAGS}_rw_hl_median": nullA[4],
            f"aug{AUG_LAGS}_share_below_rw_p05_full": float((ha < nullA[1]).mean()),
            f"aug{AUG_LAGS}_share_below_rw_p05_test": float((ha_test < nullA[1]).mean()),
            f"aug{AUG_LAGS}_share_below_rw_median_full": float((ha < nullA[4]).mean()),
            f"aug{AUG_LAGS}_share_below_rw_median_test": float((ha_test < nullA[4]).mean()),
        })

    summ = pd.DataFrame(rows).set_index("run")
    for col in ["median_full", "median_test", "share_defined_full", "share_defined_test",
                f"aug{AUG_LAGS}_median_full", f"aug{AUG_LAGS}_median_test",
                "hl_full_pt", "hl_test_pt", "adf_test", "za_test"]:
        summ[f"rank_{col}"] = summ[col].rank().astype(int)
    daily = pd.DataFrame(daily)
    out_dir = paths.ensure(paths.FIGURE_DATA)
    summ.to_csv(out_dir / "rolling_halflife_summary.csv")
    daily.to_csv(out_dir / "rolling_halflife_daily.csv", index_label="datestamp")
    print(f"wrote rolling_halflife_summary.csv and rolling_halflife_daily.csv to {out_dir}")

    with figure_style():
        plot(daily, summ, null0[4])

    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 200)
    print(summ.T.to_string())


if __name__ == "__main__":
    main()
