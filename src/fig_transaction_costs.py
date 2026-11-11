"""Trading costs per calendar year of the four headline models at 10 bps one-way.

The thesis charges a one-way cost at each rebalance (metrics.simulate_with_rebased_tri):

    cost_i = (c_bps / 10,000) x turnover_i x V_i

where turnover_i = sum_j |w_new - w_drifted| on rebalance date i (turnover/<run>.csv)
and V_i is the portfolio value at the close before the rebalance. As a share of
portfolio value each rebalance therefore costs c_bps / 10,000 x turnover_i, and the
yearly cost below is the sum of that over the rebalances in each calendar year, in
percent of portfolio value. The start-up purchase on 28 January 2009 is not charged
(the simulator starts fully invested), so the turnover files begin at the first
rebalance after the start.

The figure costs the zero-cost (tc0) runs' trades. The weights of equal weight and the
return-space ridge do not depend on the cost, so costing their tc0 turnover at 10 bps
is exactly what a 10 bps run charges; a self-check confirms this against the tc10 runs
when final_runs has made them.

2009 (from 28 January) and 2024 (to 15 May) are partial years and are hatched.

Thesis figure: transaction_costs_house.png (fig:transaction_costs)
Run:    python src/fig_transaction_costs.py
Inputs: paths.FINAL/turnover, timeseries, weights and rl_diagnostics of the headline runs
Output: paths.FIGURES/transaction_costs_house.png; in paths.FIGURE_DATA
        transaction_costs_by_year.csv, transaction_costs_top_rebalances.csv and
        transaction_costs_summary.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # write files only, never open windows
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import paths  # noqa: E402
from fig_headline_models import HEADLINE_MODELS, figure_style, load_timeseries, mark_split, save  # noqa: E402

C_BPS = 10.0
START, END = pd.Timestamp("2009-01-28"), pd.Timestamp("2024-05-15")
PARTIAL = {2009, 2024}
SPLIT_YEAR = 2017   # start of the test period
BANDIT = "rl_tuned__ma__reb63__tc0"


def load_turnover(run):
    s = pd.read_csv(paths.FINAL / "turnover" / f"{run}.csv", parse_dates=["datestamp"])
    return s.set_index("datestamp")["turnover"].sort_index()


def yearly_cost_pct(turn, c_bps=C_BPS):
    """Sum over each calendar year of c x turnover, in % of portfolio value.
    Years with no rebalance count as zero."""
    per_reb = (c_bps / 1e4) * turn * 100.0
    years = range(START.year, END.year + 1)
    return per_reb.groupby(per_reb.index.year).sum().reindex(years, fill_value=0.0)


def yearly_cost_value_weighted(run, turn, c_bps=C_BPS):
    """Cross-check: rand cost (c x turnover x V before the trade) summed per year,
    divided by the mean portfolio value over the year, in %."""
    tri = load_timeseries(run)["portfolio_tri"]
    v_before = tri.shift(1).reindex(turn.index)          # close before the rebalance
    cost = (c_bps / 1e4) * turn * v_before
    years = range(START.year, END.year + 1)
    by_year = cost.groupby(cost.index.year).sum().reindex(years, fill_value=0.0)
    mean_v = tri.groupby(tri.index.year).mean().reindex(years)
    return by_year / mean_v * 100.0


def biggest_moves(run, date, n=3):
    """Assets with the largest weight change at a rebalance (post minus pre)."""
    w = pd.read_csv(paths.FINAL / "weights" / f"{run}.csv", parse_dates=["datestamp"]).set_index("datestamp")
    pos = w.index.get_loc(date)
    dw = (w.iloc[pos] - w.iloc[pos - 1]).sort_values(key=np.abs, ascending=False)
    return ", ".join(f"{a} {v * 100:+.0f}pp" for a, v in dw.head(n).items())


def check_cost_runs():
    """Self-check: equal weight and the return-space ridge trade the same at 0 and
    10 bps, and the 10 bps run's realised cost matches the cost implied by the tc0
    turnover."""
    for run in ("equal_weight__ma__reb21", "ridge_returns__ma__w252__reb21"):
        if not (paths.FINAL / "turnover" / f"{run}__tc10.csv").exists():
            print(f"{run}: no tc10 run in {paths.FINAL}, cost cross-check skipped")
            continue
        t0, t10 = load_turnover(f"{run}__tc0"), load_turnover(f"{run}__tc10")
        same = t0.index.equals(t10.index) and np.allclose(t0.values, t10.values)
        r0 = load_timeseries(f"{run}__tc0")["portfolio_tri"]
        r10 = load_timeseries(f"{run}__tc10")["portfolio_tri"]
        realised = 1 - r10.iloc[-1] / r0.iloc[-1]
        implied = 1 - np.prod(1 - C_BPS / 1e4 * t0.values)
        print(f"{run}: tc0 and tc10 turnover identical = {same}; "
              f"cumulative cost realised {realised * 100:.3f}% vs implied {implied * 100:.3f}%")


def plot(table):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    years = np.array(table.index)
    k = len(HEADLINE_MODELS)
    width = 0.8 / k
    full = np.array([y not in PARTIAL for y in years])
    for i, (run, label, colour, _) in enumerate(HEADLINE_MODELS):
        x = years + (i - (k - 1) / 2) * width
        vals = table[label].values
        ax.bar(x[full], vals[full], width, color=colour, edgecolor="white", linewidth=0.6,
               label=label, zorder=3)
        ax.bar(x[~full], vals[~full], width, color=colour, alpha=0.45, hatch="////",
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks(years)
    ax.set_xticklabels([f"{y}*" if y in PARTIAL else str(y) for y in years])
    ax.set_xlim(years[0] - 0.6, years[-1] + 0.6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Cost (% of portfolio value)")
    ax.set_title("Transaction Costs per Calendar Year at 10 Basis Points One-Way")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.6", alpha=0.45, hatch="////", edgecolor="white"))
    labels.append("* partial year (2009 from 28 January, 2024 to 15 May)")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=9)
    mark_split(ax, SPLIT_YEAR - 0.5, zorder=2)
    save(fig, "transaction_costs_house.png")


def main():
    turns = {run: load_turnover(run) for run, *_ in HEADLINE_MODELS}
    table = pd.DataFrame({label: yearly_cost_pct(turns[run]) for run, label, *_ in HEADLINE_MODELS})
    table.index.name = "year"

    # Self-checks
    vw = pd.DataFrame({label: yearly_cost_value_weighted(run, turns[run])
                       for run, label, *_ in HEADLINE_MODELS})
    print(f"value-weighted vs sum-of-shares, max abs diff: {(vw - table).abs().max().max() * 100:.3f} bp")
    check_cost_runs()

    # Summary
    complete = [y for y in table.index if y not in PARTIAL]
    n_years = (END - START).days / 365.25
    summary = pd.DataFrame({
        "complete years 2010-2023": table.loc[complete].mean(),
        "training complete years 2010-2016": table.loc[2010:2016].mean(),
        "test complete years 2017-2023": table.loc[2017:2023].mean(),
        "whole period, total / years": table.sum() / n_years,
        "highest complete year": table.loc[complete].max(),
        "year of highest": table.loc[complete].idxmax(),
        "lowest complete year": table.loc[complete].min(),
        "year of lowest": table.loc[complete].idxmin(),
    })
    pd.set_option("display.width", 200)
    print("\nYearly cost at 10 bps one-way, % of portfolio value (2009 and 2024 partial):")
    print(table.round(3).to_string())
    print("\nSummary (%):")
    print(summary.T.to_string())

    # What drives the spikes: the largest single rebalances
    rows = []
    rl = pd.read_csv(paths.FINAL / "rl_diagnostics" / f"{BANDIT}.csv",
                     parse_dates=["datestamp"]).set_index("datestamp")["chosen_window"]
    for run, label, *_ in HEADLINE_MODELS:
        t = turns[run]
        print(f"\n{label}: {len(t)} rebalances, median turnover {t.median() * 100:.1f}%, "
              f"mean {t.mean() * 100:.1f}%")
        for d, v in t.sort_values(ascending=False).head(5).items():
            extra = ""
            if run == BANDIT:
                prev = rl.loc[:d - pd.Timedelta(days=1)]
                prev_w = int(prev.iloc[-1]) if len(prev) else None
                extra = f"; window {prev_w} -> {int(rl.asof(d))} days"
            moves = biggest_moves(run, d)
            print(f"  {d.date()} turnover {v * 100:5.1f}%  cost {C_BPS / 1e4 * v * 100:.3f}%  [{moves}]{extra}")
            rows.append(dict(model=label, date=d.date(), turnover_pct=v * 100,
                             cost_pct=C_BPS / 1e4 * v * 100, biggest_moves=moves, note=extra.strip("; ")))

    out_dir = paths.ensure(paths.FIGURE_DATA)
    out = table.copy()
    out["partial_year"] = [y in PARTIAL for y in out.index]
    out.to_csv(out_dir / "transaction_costs_by_year.csv", float_format="%.5f")
    pd.DataFrame(rows).to_csv(out_dir / "transaction_costs_top_rebalances.csv",
                              index=False, float_format="%.4f")
    summary.to_csv(out_dir / "transaction_costs_summary.csv", float_format="%.5f")
    print(f"\nwrote transaction_costs_*.csv to {out_dir}")

    with figure_style():
        plot(table)


if __name__ == "__main__":
    main()
