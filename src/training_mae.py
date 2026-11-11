"""
Training MAE and the hyperparameter selection of the final specification.

The thesis chooses the bandit's hyperparameters by the training MAE (methodology,
"Hyperparameter selection"): the mean absolute residual |u_t| of the fee-drag regression
S_t = c + g t + u_t, fitted by least squares to a setting's log spread
S_t = log(portfolio TRI) - log(peer-average TRI) over the training period (28 January 2009
to 30 December 2016, t = 0, 1, ... in trading days). The rule: among the settings of the
trend-outside grid (grid_final) with training turnover below 150% a year
(select_tuning.TO_LINE on train_to, the mean over complete calendar years), the one with
the lowest training MAE, exact ties to the name. Settings whose path (decision dates,
windows and weights) is identical to the selected one are the same portfolio under another
name and are marked. The test and full-period MAE are the same regression fitted on those
periods; they are reported, not used for selection.

The trend-inside grid (grid_final_in) is reported in the appendix only: its eligible bandit
with the lowest training MAE is the row "Bandit, trend inside the logarithm" of
tab:original, and the same rule pooled over both grids is reported beside it.

Feeds: the selected setting (tab:models and every final run, through selection_outside.csv
and final_bandit.selected_setting), the MAE columns of tab:tuning_key and
tab:tuning_neighbours (grid_final_with_mae.csv, through thesis_tables.py), and the training
MAEs of the methodology and of tab:original and its caption (selection_outside.csv,
grid_final_in_train_mae_all.csv, selection_pooled.csv).

    python src/training_mae.py           # about 2 minutes
    python src/training_mae.py select    # the selections only, from the saved
                                         # grid_final*_train_mae_all.csv (seconds)

Inputs (results/grids/, from `python src/final_bandit.py final`): grid_final.csv,
grid_final_paths.npz, grid_final_in.csv, grid_final_in_paths.npz; data/data.xlsx.
Outputs (results/grids/):
  grid_final_with_mae.csv          grid_final.csv plus train_mae, test_mae and full_mae
  grid_final_train_mae_all.csv     grid_final.csv plus the training MAE of every setting
                                   (train_mae_all)
  grid_final_in_train_mae_all.csv  grid_final_in.csv plus train_mae_all
  selection_outside.csv            the trend-outside settings ranked by training MAE, with
                                   eligible, identical_path_to_best, mae_tie_with_best and
                                   selected (the rule; read by final_bandit.selected_setting)
  selection_pooled.csv             both grids ranked by training MAE (trend 'outside' or
                                   'inside'), with eligible and selected (appendix only)
  grid_summary.csv                 rows 'final' and 'final_in': each grid's pick by the
                                   training-MAE rule, its scores, its window shares on the
                                   test period and a DM test against the best fixed window
"""
import sys

import numpy as np
import pandas as pd

import final_bandit  # noqa: F401  (below-normal priority; N_JOBS workers if a ridge cache is missing)
import paths
import robustness_grid as rg
import run_tuning as rt
from select_tuning import TO_LINE

GRIDS = paths.GRIDS
PERIODS = (("train", rt.TRAIN), ("test", rt.TEST), ("full", rt.FULL))
SEL_COLS = ["name", "family", "train_mae_all", "train_to", "train_to_ay", "train_te",
            "test_te", "test_to", "test_drag", "test_adf", "test_za", "full_te", "full_to", "full_adf", "full_za"]


def spread(data, idx, W):
    """Log spread log(portfolio TRI) - log(peer-average TRI) of a weight path."""
    p_ret, p_tri, _, _ = rt.simulate(data, idx, W)
    bench_tri = (1 + data.B.reindex(p_ret.index).fillna(0.0)).cumprod()
    return np.log(p_tri) - np.log(bench_tri)


def fee_drag_mae(s):
    """Mean absolute residual of the least-squares fit S_t = c + g t + u_t, t = 0, 1, ..."""
    t = np.arange(len(s), dtype=float)
    X = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(X, s.values, rcond=None)
    return float(np.mean(np.abs(s.values - X @ coef)))


def period_mae(data, idx, W, periods=PERIODS):
    """{period: MAE} of a weight path, from one simulation."""
    s = spread(data, idx, W)
    return {tag: fee_drag_mae(s.loc[start:end].dropna()) for tag, (start, end, _) in periods}


def grid_mae(data, grid_csv, paths_npz, periods):
    """The grid with one '<period>_mae' column per period, and its paths."""
    g = pd.read_csv(GRIDS / grid_csv)
    z = np.load(GRIDS / paths_npz)
    rows = [period_mae(data, z[f"{n}__idx"], z[f"{n}__W"], periods) for n in g["name"]]
    for tag, _ in periods:
        g[f"{tag}_mae"] = [r[tag] for r in rows]
    return g, z


def select_by_mae(df):
    """The rule on one grid (a table with train_mae_all): lowest training MAE among the
    settings with training turnover below TO_LINE, ties to the name. Returns (selected
    row, eligible rows), as robustness_grid.summarise expects."""
    elig = df[(df["train_to"] < TO_LINE) & df["train_mae_all"].notna()]
    return elig.sort_values(["train_mae_all", "name"]).iloc[0], elig


def outside_selection(o, z):
    """The rule on the trend-outside grid: (table, pick). o is grid_final.csv with
    train_mae_all and z its paths."""
    s = o[SEL_COLS].assign(trend="outside")
    s["eligible"] = (s["train_to"] < TO_LINE) & s["train_mae_all"].notna()
    s = s.sort_values(["train_mae_all", "name"]).reset_index(drop=True)
    pick = s[s["eligible"]].iloc[0]["name"]

    def same_path(a, b):
        return all(f"{a}__{k}" in z and np.array_equal(z[f"{a}__{k}"], z[f"{b}__{k}"]) for k in ("idx", "win", "W"))
    s["identical_path_to_best"] = [bool(e) and same_path(n, pick) for n, e in zip(s["name"], s["eligible"])]
    s["mae_tie_with_best"] = s["eligible"] & (s["train_mae_all"] == s.loc[s["name"] == pick, "train_mae_all"].iloc[0])
    odd = s[s["mae_tie_with_best"] & ~s["identical_path_to_best"]]
    if len(odd):
        print("WARNING: equal training MAE but a different path from the selected setting:",
              odd["name"].tolist(), flush=True)
    s["selected"] = s["name"] == pick
    return s, pick


def pooled_selection(o, i):
    """The same rule over both grids together (appendix only)."""
    p = pd.concat([o[SEL_COLS].assign(trend="outside"), i[SEL_COLS].assign(trend="inside")], ignore_index=True)
    p["eligible"] = p["train_to"] < TO_LINE
    p = p.sort_values(["train_mae_all", "trend", "name"]).reset_index(drop=True)
    p["selected"] = False
    p.loc[p[p["eligible"]].index[0], "selected"] = True
    return p


def select(o, i, z_out):
    """Write selection_pooled.csv and selection_outside.csv, print both picks and return
    the selected setting."""
    p = pooled_selection(o, i)
    p.to_csv(GRIDS / "selection_pooled.csv", index=False)
    sel = p[p["selected"]].iloc[0]
    other = p[p.eligible & (p.trend != sel["trend"])].head(1)
    print(f"pooled rule (appendix only): {sel['name']} (trend {sel['trend']}), training MAE "
          f"{100 * sel['train_mae_all']:.3f}%, training turnover {100 * sel['train_to']:.0f}%; best eligible with "
          f"the trend {'inside' if sel['trend'] == 'outside' else 'outside'}: "
          + ", ".join(f"{r['name']} {100 * r['train_mae_all']:.3f}%" for _, r in other.iterrows()), flush=True)
    s, pick = outside_selection(o, z_out)
    s.to_csv(GRIDS / "selection_outside.csv", index=False)
    r = s[s["selected"]].iloc[0]
    ties = s[s["identical_path_to_best"]]["name"].tolist()
    print(f"SELECTED (trend outside, training turnover < {TO_LINE:.0%}): {pick}, training MAE "
          f"{100 * r['train_mae_all']:.3f}%, training turnover {100 * r['train_to']:.0f}%; settings with the same "
          f"path: {ties}; lowest training MAE without the turnover filter: {s.iloc[0]['name']}", flush=True)
    ins = p[(p.trend == "inside") & (p.family == "bandit") & p.eligible].iloc[0]
    print(f"trend-inside bandit with the lowest training MAE (tab:original): {ins['name']}, "
          f"{100 * ins['train_mae_all']:.3f}%, test TE {100 * ins['test_te']:.2f}%", flush=True)
    return pick


def main(stage="all"):
    for f in ("grid_final.csv", "grid_final_paths.npz", "grid_final_in.csv", "grid_final_in_paths.npz"):
        if not (GRIDS / f).exists():
            sys.exit(f"{GRIDS / f} not found: run `python src/final_bandit.py final` first")
    if stage == "select":
        o = pd.read_csv(GRIDS / "grid_final_train_mae_all.csv")
        i = pd.read_csv(GRIDS / "grid_final_in_train_mae_all.csv")
        select(o, i, np.load(GRIDS / "grid_final_paths.npz"))
        return
    if stage != "all":
        sys.exit(f"unknown stage {stage!r}: use select, or nothing for everything")
    data = rt.load()

    # trend outside the log: train, test and full MAE; trend inside: training MAE
    out, z_out = grid_mae(data, "grid_final.csv", "grid_final_paths.npz", PERIODS)
    out.to_csv(GRIDS / "grid_final_with_mae.csv", index=False)
    inn, z_in = grid_mae(data, "grid_final_in.csv", "grid_final_in_paths.npz", PERIODS[:1])
    o = out.drop(columns=["test_mae", "full_mae"]).rename(columns={"train_mae": "train_mae_all"})
    i = inn.rename(columns={"train_mae": "train_mae_all"})
    o.to_csv(GRIDS / "grid_final_train_mae_all.csv", index=False)
    i.to_csv(GRIDS / "grid_final_in_train_mae_all.csv", index=False)
    for tag, d in (("outside", o), ("inside", i)):
        el = d["train_to"] < TO_LINE
        best, excl = d[el].nsmallest(1, "train_mae_all").iloc[0], d[~el].nsmallest(1, "train_mae_all")
        print(f"trend {tag}: {int(el.sum())} eligible; lowest training MAE {best['name']} {best['train_mae_all']:.3%}"
              + (f"; lowest excluded (turnover >= {TO_LINE:.0%}) {excl.iloc[0]['name']} "
                 f"{excl.iloc[0]['train_mae_all']:.3%} at {excl.iloc[0]['train_to']:.0%}" if len(excl) else ""),
              flush=True)
    pick = select(o, i, z_out)

    # grid_summary.csv: each grid's pick by the same rule
    summary = []
    for spec, df, z in (("final", o, z_out), ("final_in", i, z_in)):
        s = rg.summarise(data, spec, df, z, select=select_by_mae)
        s["grid_stuck_total"] = int(df["stuck"].sum())
        s["grid_fallbacks_total"] = int(df["fallbacks"].sum())
        summary.append(s)
    assert summary[0]["selected"] == pick, (summary[0]["selected"], pick)
    pd.DataFrame(summary).to_csv(GRIDS / "grid_summary.csv", index=False)
    print(pd.DataFrame(summary).set_index("variant").T.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
