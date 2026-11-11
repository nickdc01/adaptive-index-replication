"""Numbers quoted in the Results text and appendix that are not in any table.

Reads the saved runs of stage final_runs (results/final/timeseries, weights, turnover and
rl_diagnostics) and writes <results>/final/prose_numbers.csv with one row per number:
end values of the total return indices, how often the tuned bandit's rolling 1-year tracking
error is below the other models', the weights of the return-space ridge, the equity and
offshore shares, trading costs at 10 basis points, the fit of 21-day returns, the
autocorrelation of the daily change in the log spread, and the bandit's window choices.
Takes seconds.

    python src/prose_numbers.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import paths

RUNS = {"equal_weight": "equal_weight__ma__reb21__tc0",
        "return_space_ridge": "ridge_returns__ma__w252__reb21__tc0",
        "fixed_1y_quarterly": "window_fixedbest__ma__w252__reb63__tc0",
        "tuned_bandit": "rl_tuned__ma__reb63__tc0"}
TEST_START = pd.Timestamp("2017-01-03")


def read(kind, run):
    return pd.read_csv(paths.FINAL / kind / f"{run}.csv", parse_dates=["datestamp"]).set_index("datestamp")


def main():
    rows = []

    def add(item, value, unit=""):
        rows.append({"item": item, "value": float(value), "unit": unit})
        print(f"  {item}: {value:.4g} {unit}")

    ts = {k: read("timeseries", r) for k, r in RUNS.items()}
    w = {k: read("weights", r) for k, r in RUNS.items()}
    to = {k: read("turnover", r)["turnover"] for k, r in RUNS.items()}

    # end values of the total return indices, rebased to 100 at the first date
    b = ts["tuned_bandit"].benchmark_tri
    add("tri_end_peer_average", 100 * b.iloc[-1] / b.iloc[0], "index, start = 100")
    for k in RUNS:
        p = ts[k].portfolio_tri
        add(f"tri_end_{k}", 100 * p.iloc[-1] / p.iloc[0], "index, start = 100")

    # rolling 1-year tracking error
    te = pd.DataFrame({k: ts[k].rolling_te_1y for k in RUNS}).dropna()
    add("share_days_bandit_te_below_equal_weight", (te.tuned_bandit < te.equal_weight).mean(), "fraction")
    add("share_days_bandit_te_below_fixed_1y", (te.tuned_bandit < te.fixed_1y_quarterly).mean(), "fraction")
    add("share_days_ridge_te_lowest", (te.return_space_ridge == te.min(axis=1)).mean(), "fraction")
    sell_off = te.loc["2020-02-01":"2021-06-30"]
    add("rolling_te_peak_2020_tuned_bandit", sell_off.tuned_bandit.max(), "fraction a year")
    add("rolling_te_peak_2020_fixed_1y", sell_off.fixed_1y_quarterly.max(), "fraction a year")

    # weights
    r = w["return_space_ridge"]
    add("ridge_mean_sa_cash", r["SA Cash & Income"].mean(), "fraction")
    add("ridge_max_sa_cash", r["SA Cash & Income"].max(), "fraction")
    add("ridge_max_sa_cash_year", r["SA Cash & Income"].idxmax().year)
    add("ridge_mean_sa_equities", r["SA Equities"].mean(), "fraction")
    add("ridge_share_days_some_asset_at_zero", (r.min(axis=1) < 1e-6).mean(), "fraction")
    cols = list(w["equal_weight"].columns)
    equity = [c for c in cols if "Equities" in c]
    offshore = [c for c in cols if c.startswith("Unhedged") or c.startswith("Hedged")]
    for k in RUNS:
        add(f"mean_equity_share_{k}", w[k][equity].sum(axis=1).mean(), "fraction")
        add(f"mean_offshore_share_{k}", w[k][offshore].sum(axis=1).mean(), "fraction")
    d = (w["tuned_bandit"] - w["fixed_1y_quarterly"]).abs().sum(axis=1) / 2
    add("mean_weight_difference_bandit_vs_fixed_1y_test", d[d.index >= TEST_START].mean(), "fraction (half sum of |differences|)")

    # trading costs at 10 basis points one way, complete calendar years 2010-2023
    for k in RUNS:
        y = to[k].groupby(to[k].index.year).sum().reindex(range(2010, 2024), fill_value=0.0)
        add(f"trading_cost_10bp_{k}", 10 * y.mean(), "basis points a year")

    # non-overlapping 21-day returns of the tuned bandit on the peer average
    s = ts["tuned_bandit"]
    idx = np.arange(0, len(s), 21)
    rp = s.portfolio_tri.iloc[idx].pct_change().dropna().values
    rb = s.benchmark_tri.iloc[idx].pct_change().dropna().values
    add("monthly_fit_n", len(rp))
    add("monthly_fit_slope", np.polyfit(rb, rp, 1)[0])
    add("monthly_fit_r2", np.corrcoef(rb, rp)[0, 1] ** 2)

    # first-order autocorrelation of the daily change in the log spread
    for k in RUNS:
        dd = (np.log(ts[k].portfolio_tri) - np.log(ts[k].benchmark_tri)).diff().dropna()
        add(f"spread_change_autocorr_{k}", dd.autocorr(1))

    # window choices of the tuned bandit and of the context-free bandit
    a = read("rl_diagnostics", "rl_tuned__ma__reb63__tc0")
    c = read("rl_diagnostics", "rl_tuned_ctxfree__ma__reb63__tc0")
    test = a.index >= TEST_START
    add("bandit_decisions", len(a))
    add("bandit_decisions_training", (~test).sum())
    add("bandit_decisions_test", test.sum())
    add("bandit_test_decisions_1y_or_2y", a.chosen_window[test].isin([252, 504]).sum())
    add("context_free_same_choice", (a.chosen_window.values == c.chosen_window.values).sum())

    out = paths.ensure(paths.FINAL) / "prose_numbers.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
