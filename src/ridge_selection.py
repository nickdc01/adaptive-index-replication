"""
Where the turnover-penalised ridge (monthly, one-year window, delta 1) puts its
cross-validated ridge strength r under the final specification (constant and trend outside
the log, delta * n / 252): the median r and the share of rebalances at which r sits on a
boundary of the grid (0 or the largest value).

Uses final_runs._cv_trend_out, the cross-validation of the final turnover-penalised ridge,
and always recomputes the picks. Inside every fold the penalty is measured against equal
weight, and the previous weights (drifted to the rebalance date in the final runs) enter
only the final fit at the selected r, which this script does not run; so the picks do not
depend on the penalty's reference. Run it after final_runs.py runs, which clears
results/final/.

    python src/ridge_selection.py    # about 6 minutes

Input: data/data.xlsx. Output: results/final/tpr_ridge_picks.npy (the r picked at each
rebalance) and the summary line on screen.
"""
import numpy as np

import final_runs as fr
import paths
import run_models as rm


def main():
    A, B = rm.load_data("ma")
    asset_tri, bench_tri = (1 + A).cumprod(), (1 + B).cumprod()
    grid = np.asarray(rm._resolve_ridge_grid({}), dtype=float)
    window, reb, delta = 252, 21, 1.0
    picks = []
    for i in range(window, len(asset_tri), reb):
        ta = asset_tri.iloc[i - window:i].values / asset_tri.iloc[i - window].values * 100
        tb = bench_tri.iloc[i - window:i].values / bench_tri.iloc[i - window] * 100
        X = np.hstack([ta, np.arange(ta.shape[0]).reshape(-1, 1)])
        d = delta * X.shape[0] / fr.NREF
        cvs = [fr._cv_trend_out(X, tb, r, A.shape[1], delta=d) for r in grid]
        picks.append(float(grid[int(np.argmin(cvs))]))
    picks = np.array(picks)
    lo, hi = grid.min(), grid.max()
    print(f"rebalances {len(picks)}; median r {np.median(picks):.4g}; at 0 {np.mean(picks == lo):.1%}; "
          f"at the top {np.mean(picks == hi):.1%}; on a boundary {np.mean((picks == lo) | (picks == hi)):.1%}")
    np.save(paths.ensure(paths.FINAL) / "tpr_ridge_picks.npy", picks)


if __name__ == "__main__":
    main()
