"""
Peer-category average against the random-portfolio return distribution (thesis
Figure fig:peer_vs_random, peer_av_vs_rand_house.png).

The difficulty question is a returns comparison: each calendar year, the peer
average's realised (net-of-fee) return is set against the annual returns of the
5,000 random Regulation 28 portfolios. No replicator weights are used, so the
comparison does not depend on any tracker. The random portfolios are gross
asset-class combinations, so the comparison is conservative (the peer average
carries fees). Each random portfolio is set at the start of the calendar year
and left to drift within the year (annual rebalance). Years with fewer than 60
trading days are skipped.

Run:     python src/fig_peer_vs_random.py
Inputs:  data/data.xlsx (sheets X and Y, via run_models.load_data("ma"))
         <results>/random_portfolios.csv (from random_portfolios.py)
Outputs: <results>/figures/peer_av_vs_rand_house.png
         <results>/figure_data/peer_av_vs_rand_years.csv    per-year returns and percentile bands
         <results>/figure_data/peer_av_vs_rand_summary.csv  the peer's percentile across years
"""
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402
from run_models import load_data  # noqa: E402

warnings.filterwarnings("ignore")

RP = paths.RESULTS / "random_portfolios.csv"
OUT = paths.FIGURES / "peer_av_vs_rand_house.png"
YEARS_CSV = paths.FIGURE_DATA / "peer_av_vs_rand_years.csv"
SUMMARY_CSV = paths.FIGURE_DATA / "peer_av_vs_rand_summary.csv"

STYLE = {
    "figure.figsize": (10, 5), "figure.dpi": 120, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.35,
    "grid.linestyle": "-", "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    "legend.fontsize": 9,
}


def main():
    if not RP.exists():
        raise SystemExit(f"{RP} not found: run python src/random_portfolios.py first")

    # load_data returns assets and the peer average on one (asset) calendar.
    asset_ret, Y = load_data("ma")
    weights = pd.read_csv(RP)

    cols = [c for c in weights.columns if c in asset_ret.columns]
    W = weights[cols].values.T          # (K, N) random portfolio weights
    asset_ret = asset_ret[cols]

    # common dates between assets and the peer average
    common = asset_ret.index.intersection(Y.index)
    asset_ret = asset_ret.loc[common]
    peer = Y.loc[common]

    # ── per-year comparison (annual rebalance) ──────────────────────────────
    years = sorted({d.year for d in common})
    rows = []
    band = {q: [] for q in [5, 25, 50, 75, 95]}
    peer_ann, peer_pctl, yr_labels = [], [], []

    for y in years:
        mask = np.array([d.year == y for d in asset_ret.index])
        n_days = int(mask.sum())
        if n_days < 60:                  # skip stub years with <~3 months of data
            continue
        R_y = asset_ret.values[mask]                         # (T_y, K)
        # Weights are set at the start of the calendar year and drift with the
        # assets within the year (buy and hold inside the year), so the year's
        # return is the weighted sum of each asset's compounded return.
        # (Compounding R_y @ W day by day would reset the weights every day.)
        asset_growth = np.prod(1.0 + R_y, axis=0)            # (K,) asset growth over the year
        rp_ann = asset_growth @ W - 1.0                      # (N,) random annual rets
        p_ann = float(np.prod(1.0 + peer.values[mask]) - 1.0)
        pct = float((rp_ann < p_ann).mean() * 100.0)

        yr_labels.append(y)
        peer_ann.append(p_ann * 100)
        peer_pctl.append(pct)
        for q in band:
            band[q].append(np.percentile(rp_ann, q) * 100)
        rows.append((y, n_days, p_ann * 100, np.percentile(rp_ann, 50) * 100, pct,
                     *[band[q][-1] for q in [5, 25, 75, 95]]))

    # ── figure ──────────────────────────────────────────────────────────────
    with plt.style.context(STYLE, after_reset=True):
        fig, ax = plt.subplots()
        yrs = np.array(yr_labels)
        ax.fill_between(yrs, band[5], band[95], color="#1f4e79", alpha=0.15,
                        label="Random portfolios, 5th to 95th percentile")
        ax.fill_between(yrs, band[25], band[75], color="#1f4e79", alpha=0.30,
                        label="Random portfolios, 25th to 75th percentile")
        ax.plot(yrs, band[50], color="#1f4e79", lw=1.2, label="Random-portfolio median")
        ax.plot(yrs, peer_ann, color="#c0392b", lw=1.6, marker="o", ms=4,
                label="Peer-category average (net of fees)")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("")
        ax.set_xticks(yrs[::2])                          # integer calendar-year ticks
        ax.set_xticklabels([str(int(y)) for y in yrs[::2]])
        ax.set_ylabel("Calendar-year return (%)")
        ax.set_title("Peer-Category Average vs Feasible Random-Portfolio Return Distribution")
        # legend below the axes, so it hides no data point
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
        paths.ensure(OUT.parent)
        fig.savefig(OUT)
        plt.close(fig)

    # ── report (percentile summary only) and per-year table ─────────────────
    pcts = np.array(peer_pctl)
    print(f"peer percentile across {len(pcts)} years: min={pcts.min():.0f}  median={np.median(pcts):.0f}  "
          f"max={pcts.max():.0f}  mean={pcts.mean():.0f}")
    print(f"years peer in top quartile (>75th): {(pcts>75).sum()} / {len(pcts)}")
    print(f"years peer below median (<50th):    {(pcts<50).sum()} / {len(pcts)}")

    paths.ensure(paths.FIGURE_DATA)
    pd.DataFrame(rows, columns=["year", "n_days", "peer_ret_pct", "rand_median_pct",
                                "peer_percentile", "rand_p5_pct", "rand_p25_pct",
                                "rand_p75_pct", "rand_p95_pct"]
                 ).to_csv(YEARS_CSV, index=False)
    pd.DataFrame([dict(
        n_years=int(len(pcts)),
        first_year=int(yrs.min()), last_year=int(yrs.max()),
        pctl_min=float(pcts.min()), pctl_median=float(np.median(pcts)),
        pctl_max=float(pcts.max()), pctl_mean=float(pcts.mean()),
        years_above_75th=int((pcts > 75).sum()),
        years_below_50th=int((pcts < 50).sum()),
        years_below_25th=int((pcts < 25).sum()),
    )]).to_csv(SUMMARY_CSV, index=False)
    print(f"wrote {OUT}, {YEARS_CSV.name}, {SUMMARY_CSV.name}")


if __name__ == "__main__":
    main()
