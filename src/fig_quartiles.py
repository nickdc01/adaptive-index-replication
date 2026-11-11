"""Rolling 12-month return quartiles of the headline models against random portfolios.

For every month-end with a full 12 months of model history (the models start on
2009-01-28), compare the trailing 12-month return of each headline model and of
the peer average with the trailing 12-month returns of the 5,000 random
portfolios over the same window, rank each series in that distribution, and
count how often it falls in each quartile (Q1 = top 25%).

- Models: gross of fees and at zero trading cost (the tc0 runs), portfolio_tri.
- Peer average: net of fees, from the same files' benchmark_tri column.
- Random portfolios: weights set at the window start and left to drift for the
  12 months (buy and hold inside the window), so the window return is
  prod(1 + R_window) @ W - 1 with R the daily asset returns from
  run_models.load_data("ma").

Windows run between consecutive month-ends (last trading day of each completed
calendar month). 2024-05-15 is the data end and not a month-end, so the last
window ends on 2024-04-30. Period classes:
  full  = every window
  train = window ends on or before 2016-12-30 (all returns in the training period)
  test  = window starts on or after 2016-12-30 (all returns in the test period)
  test_end = window ends in the test period (a looser check, overlaps training)
Windows overlap by 11 months, so consecutive ranks are highly dependent.

Two diagnostics quoted in the text: the ranks against the equity-matched random
portfolios (50% to 65% equity at the window start) and the models' average equity,
offshore and property exposure.

Thesis figure: quartile_vs_random_house.png (fig:quartile_random)
Run:    python src/fig_quartiles.py
Inputs: paths.DATA (through run_models.load_data), paths.RESULTS/random_portfolios.csv
        (python src/random_portfolios.py), paths.FINAL/timeseries and weights
Output: paths.FIGURES/quartile_vs_random_house.png; in paths.FIGURE_DATA
        quartile_vs_random_months.csv (per-window returns, ranks and quartiles),
        quartile_vs_random_summary.csv (quartile shares and percentile summary) and
        quartile_vs_random_exposure.csv (exposure diagnostic of the models)
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # write files only, never open windows
import matplotlib.pyplot as plt  # noqa: E402

import paths  # noqa: E402
from fig_headline_models import HEADLINE_MODELS, figure_style, load_timeseries, save  # noqa: E402
from run_models import load_data  # noqa: E402

warnings.filterwarnings("ignore")

START = pd.Timestamp("2009-01-28")
END = pd.Timestamp("2024-05-15")
TRAIN_END = pd.Timestamp("2016-12-30")     # last trading day of the training period
TEST_START = pd.Timestamp("2017-01-01")

MODELS = [(run, label) for run, label, *_ in HEADLINE_MODELS]
PEER = "Peer average (net of fees)"
SERIES = [lab for _, lab in MODELS] + [PEER]

# One-hue ordinal ramp on the house navy, Q1 darkest
Q_COLOURS = {"Q1": "#13324f", "Q2": "#1f4e79", "Q3": "#4a78a8", "Q4": "#8eb0d4"}
Q_INK = {"Q1": "white", "Q2": "white", "Q3": "white", "Q4": "#1a1a1a"}
Q_LABELS = {"Q1": "Q1 (top quartile)", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4 (bottom quartile)"}

EQUITY = ["SA Equities", "Unhedged Equities", "Unhedged EM Equities",
          "Hedged Equities", "Hedged EM Equities"]
OFFSHORE = ["Unhedged Bonds", "Unhedged Equities", "Unhedged Cash",
            "Unhedged EM Equities", "Hedged Bonds", "Hedged Equities", "Hedged EM Equities"]
PROPERTY = ["SA Property"]
# Equity-matched sensitivity: the random portfolios average about 39% equity, the
# fixed 1-year window and the tuned bandit about 56%. Re-rank against the random
# portfolios whose starting equity weight is between 50% and 65%, so a rank cannot
# come from holding more equity than the typical random portfolio.
EQ_LO, EQ_HI = 0.50, 0.65


def quartile(pct):
    return "Q1" if pct >= 75 else "Q2" if pct >= 50 else "Q3" if pct >= 25 else "Q4"


def load_inputs():
    """Asset returns, benchmark returns, random-portfolio weights and the TRIs."""
    asset_ret, Y = load_data("ma")
    wdf = pd.read_csv(paths.RESULTS / "random_portfolios.csv")
    assert set(wdf.columns) == set(asset_ret.columns), "random-portfolio assets differ from the loader's"
    asset_ret = asset_ret[list(wdf.columns)]
    row_sums = wdf.sum(axis=1).values
    assert np.allclose(row_sums, 1.0, atol=1e-9), "random-portfolio rows do not sum to one"
    assert (wdf.values >= 0).all(), "negative random-portfolio weight"
    print(f"random portfolios: {len(wdf)} rows, row sums in [{row_sums.min():.12f}, {row_sums.max():.12f}]")

    tri, bench = {}, None
    for run, lab in MODELS:
        df = load_timeseries(run).loc[START:END]
        tri[lab] = df["portfolio_tri"]
        if bench is None:
            bench = df["benchmark_tri"]
        else:
            assert df.index.equals(bench.index), f"{run}: dates differ"
            assert np.allclose(df["benchmark_tri"].values, bench.values, rtol=0, atol=1e-12), \
                f"{run}: benchmark_tri differs"
    tri[PEER] = bench
    assert bench.index.isin(asset_ret.index).all(), "timeseries dates not on the asset calendar"
    print(f"timeseries: {bench.index[0].date()} to {bench.index[-1].date()}, {len(bench)} days")
    return asset_ret, Y, wdf, tri


def month_ends(dates):
    """Last trading day of each completed calendar month in the sample."""
    me = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).last().values
    me = pd.DatetimeIndex(me)
    me = me[me < pd.Timestamp("2024-05-01")]         # May 2024 is incomplete (data end 15 May)
    print(f"month-ends: {me[0].date()} to {me[-1].date()}, {len(me)}")
    return me


def rank_windows(asset_ret, Y, wdf, tri, me):
    """Per-window returns, percentiles and quartiles of every series."""
    W = wdf.values.T                                   # (K, N)
    eq_start = wdf[EQUITY].sum(axis=1).values
    eq_mask = (eq_start >= EQ_LO) & (eq_start <= EQ_HI)
    print(f"equity-matched subset ({EQ_LO:.0%} to {EQ_HI:.0%} equity): {eq_mask.sum()} of {len(eq_mask)}")
    R_all = asset_ret.values
    r_idx = asset_ret.index
    Y_al = Y.reindex(r_idx)
    rows = []
    max_peer_gap = 0.0
    for i in range(12, len(me)):
        s, e = me[i - 12], me[i]
        m = (r_idx > s) & (r_idx <= e)
        growth = np.prod(1.0 + R_all[m], axis=0)       # (K,) asset growth over the window
        rp = growth @ W - 1.0                          # (N,) buy-and-hold random returns
        q = np.percentile(rp, [5, 25, 50, 75, 95])
        peer_check = float(np.prod(1.0 + Y_al.values[m]) - 1.0)
        if e <= TRAIN_END:
            cls = "train"
        elif s >= TRAIN_END:
            cls = "test"
        else:
            cls = "straddle"
        for lab in SERIES:
            t = tri[lab]
            r = float(t.loc[e] / t.loc[s] - 1.0)
            if lab == PEER:
                max_peer_gap = max(max_peer_gap, abs(r - peer_check))
            pct = float((rp < r).mean() * 100.0)
            pct_eq = float((rp[eq_mask] < r).mean() * 100.0)
            rows.append(dict(start_date=s.date(), end_date=e.date(), window_class=cls,
                             end_in_test=bool(e >= TEST_START), series=lab,
                             ret_pct=r * 100, rand_p5_pct=q[0] * 100, rand_p25_pct=q[1] * 100,
                             rand_median_pct=q[2] * 100, rand_p75_pct=q[3] * 100,
                             rand_p95_pct=q[4] * 100, excess_over_rand_median_pp=(r - q[2]) * 100,
                             percentile=pct, quartile=quartile(pct),
                             rand_eqmatched_median_pct=float(np.median(rp[eq_mask])) * 100,
                             percentile_eqmatched=pct_eq, quartile_eqmatched=quartile(pct_eq)))
    print(f"peer return from benchmark_tri vs load_data compounding: max abs gap {max_peer_gap:.2e}")
    assert max_peer_gap < 1e-9
    return pd.DataFrame(rows)


def summarise(months):
    periods = {
        "full": months["window_class"].notna(),
        "train": months["window_class"] == "train",
        "test": months["window_class"] == "test",
        "test_end": months["end_in_test"],
    }
    summ = []
    for per, mask in periods.items():
        sub = months[mask]
        for lab in SERIES:
            x = sub[sub["series"] == lab]
            d = dict(period=per, series=lab, n_windows=len(x),
                     first_end=x["end_date"].min(), last_end=x["end_date"].max())
            for qq in ["Q1", "Q2", "Q3", "Q4"]:
                d[f"share_{qq}_pct"] = 100.0 * (x["quartile"] == qq).mean()
            d["share_above_median_pct"] = 100.0 * (x["percentile"] >= 50).mean()
            d["percentile_median"] = float(x["percentile"].median())
            d["percentile_mean"] = float(x["percentile"].mean())
            d["percentile_p25"] = float(x["percentile"].quantile(0.25))
            d["percentile_p75"] = float(x["percentile"].quantile(0.75))
            d["excess_over_rand_median_pp_median"] = float(x["excess_over_rand_median_pp"].median())
            d["excess_over_rand_median_pp_mean"] = float(x["excess_over_rand_median_pp"].mean())
            for qq in ["Q1", "Q2", "Q3", "Q4"]:
                d[f"eqmatched_share_{qq}_pct"] = 100.0 * (x["quartile_eqmatched"] == qq).mean()
            d["eqmatched_share_above_median_pct"] = 100.0 * (x["percentile_eqmatched"] >= 50).mean()
            d["eqmatched_percentile_median"] = float(x["percentile_eqmatched"].median())
            summ.append(d)
    summary = pd.DataFrame(summ)

    # The models' percentile lead over the peer average, window by window
    wide = months.pivot_table(index="end_date", columns="series", values="percentile")
    cls_by_end = months.drop_duplicates("end_date").set_index("end_date")["window_class"]
    lead_rows = []
    for per in ["full", "train", "test"]:
        idx = wide.index if per == "full" else cls_by_end[cls_by_end == per].index
        for lab in SERIES[:-1]:
            diff = wide.loc[idx, lab] - wide.loc[idx, PEER]
            lead_rows.append(dict(period=per, series=lab,
                                  pct_points_above_peer_median=float(diff.median()),
                                  share_windows_above_peer_pct=100.0 * float((diff > 0).mean())))
    return summary, pd.DataFrame(lead_rows), list(periods)


def exposure_table(wdf, me):
    """Regulation 28 exposure diagnostic: the models were not held to the caps."""
    cols = list(wdf.columns)
    rows = []
    for run, lab in MODELS:
        w = pd.read_csv(paths.FINAL / "weights" / f"{run}.csv", parse_dates=["datestamp"])
        w = w.set_index("datestamp").sort_index().loc[START:END, cols]
        w_me = w.loc[w.index.isin(me)]
        for per, sl in [("full", slice(None)), ("test", slice(TEST_START, None))]:
            ww = w_me.loc[sl]
            eq, off, prop = ww[EQUITY].sum(1), ww[OFFSHORE].sum(1), ww[PROPERTY].sum(1)
            rows.append(dict(period=per, series=lab,
                             equity_mean_pct=100 * eq.mean(), offshore_mean_pct=100 * off.mean(),
                             property_mean_pct=100 * prop.mean(),
                             share_offshore_above_45_pct=100 * (off > 0.45 + 1e-9).mean(),
                             share_equity_above_75_pct=100 * (eq > 0.75 + 1e-9).mean()))
    eqr, offr, propr = wdf[EQUITY].sum(1), wdf[OFFSHORE].sum(1), wdf[PROPERTY].sum(1)
    rows.append(dict(period="weights at window start", series="Random portfolios (mean of 5,000)",
                     equity_mean_pct=100 * eqr.mean(), offshore_mean_pct=100 * offr.mean(),
                     property_mean_pct=100 * propr.mean(),
                     share_offshore_above_45_pct=100 * (offr > 0.45 + 1e-9).mean(),
                     share_equity_above_75_pct=100 * (eqr > 0.75 + 1e-9).mean()))
    return pd.DataFrame(rows)


def plot(summary):
    """Full period, one stacked bar per series."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    sub = summary[summary["period"] == "full"].set_index("series").loc[SERIES]
    ypos = np.arange(len(SERIES))[::-1].astype(float)
    ypos[-1] -= 0.35                                # small gap before the peer average
    left = np.zeros(len(SERIES))
    for qq in ["Q1", "Q2", "Q3", "Q4"]:
        v = sub[f"share_{qq}_pct"].values
        ax.barh(ypos, v, left=left, height=0.62, color=Q_COLOURS[qq],
                edgecolor="white", linewidth=1.5, label=Q_LABELS[qq], zorder=3)
        for y_, l_, v_ in zip(ypos, left, v):
            if v_ >= 6:
                ax.text(l_ + v_ / 2, y_, f"{v_:.0f}%", ha="center", va="center",
                        fontsize=8.5, color=Q_INK[qq], zorder=4)
        left += v
    ax.set_yticks(ypos)
    ax.set_yticklabels(SERIES)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of month-ends (%)")
    ax.grid(axis="y", visible=False)
    n = int(sub["n_windows"].iloc[0])
    ax.set_title(f"Rolling 12-Month Return Quartiles against 5,000 Random Portfolios ({n} month-ends)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    save(fig, "quartile_vs_random_house.png")


def main():
    asset_ret, Y, wdf, tri = load_inputs()
    me = month_ends(tri[PEER].index)
    months = rank_windows(asset_ret, Y, wdf, tri, me)
    summary, lead, periods = summarise(months)
    exposure = exposure_table(wdf, me)

    with figure_style():
        plot(summary)

    out_dir = paths.ensure(paths.FIGURE_DATA)
    months.to_csv(out_dir / "quartile_vs_random_months.csv", index=False)
    summary.merge(lead, on=["period", "series"], how="left").to_csv(
        out_dir / "quartile_vs_random_summary.csv", index=False)
    exposure.to_csv(out_dir / "quartile_vs_random_exposure.csv", index=False)
    print(f"wrote quartile_vs_random_months.csv, _summary.csv and _exposure.csv to {out_dir}")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for per in periods:
        sub = summary[summary["period"] == per]
        print(f"\n== {per}: {int(sub['n_windows'].iloc[0])} windows, ends "
              f"{sub['first_end'].iloc[0]} to {sub['last_end'].iloc[0]}")
        print(sub[["series", "share_Q1_pct", "share_Q2_pct", "share_Q3_pct", "share_Q4_pct",
                   "share_above_median_pct", "percentile_median", "percentile_mean",
                   "excess_over_rand_median_pp_median"]].round(1).to_string(index=False))
    print("\n== equity-matched random set (50% to 65% equity at the window start)")
    for per in ["full", "train", "test"]:
        sub = summary[summary["period"] == per]
        print(f"-- {per}")
        print(sub[["series", "eqmatched_share_Q1_pct", "eqmatched_share_Q2_pct", "eqmatched_share_Q3_pct",
                   "eqmatched_share_Q4_pct", "eqmatched_share_above_median_pct",
                   "eqmatched_percentile_median"]].round(1).to_string(index=False))
    print("\n== models' percentile lead over the peer average")
    print(lead.round(1).to_string(index=False))
    print("\n== exposure (month-end weights)")
    print(exposure.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
