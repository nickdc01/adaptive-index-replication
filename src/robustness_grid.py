"""
The original 910-setting tuning grid (bandit_tuning.grid) run through the copied loops of
original_spec.py with the original selection rule, and the summary row of a grid.

  baseline  the original specification itself. Its summary row reproduces the original
            selection (lowest training joint ADF/ZA p among settings with training turnover
            below 150%, ties within 0.001 to the lower training TE), which chose the first
            version's setting in Appendix "The Original Specification"
            (bandit_k0.6_e0.5_g1_d1_reb63_ce252). If results/original_grid/grid_scores.csv
            (`python src/run_tuning.py grid`) exists, the baseline grid is compared with it.
            Optional: no table or figure reads its outputs.

summarise() writes the summary row of a grid; training_mae.py uses it for the two final
grids (final_bandit.py), there with the training-MAE rule.

    python src/robustness_grid.py    # optional (about 11 min on original_spec.N_JOBS workers)

Inputs: data/data.xlsx; ridge caches in results/cache.
Outputs (results/robustness/):
  grid_<variant>.csv        train / test / full scores of every setting
  grid_<variant>_paths.npz  each setting's weights (__W), windows (__win) and decision indices (__idx)
  grid_summary.csv          one row per variant: the selected setting and its scores
"""
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bandit_tuning as bt  # noqa: E402
import original_spec as orig  # noqa: E402
import paths  # noqa: E402
import run_tuning as rt  # noqa: E402
import select_tuning as st  # noqa: E402
from dm_test import dm_test  # noqa: E402

OUT = paths.ROBUSTNESS
VARIANTS = {
    "baseline": dict(),
}


def evaluate_v(data, cfg, variant):
    fit = orig.Fit(data, **VARIANTS[variant])
    fam = cfg["family"]
    if fam == "fixed":
        idx, chosen, W = orig.run_fixed_v(data, int(cfg["window"]), fit, delta=float(cfg["delta"]), reb=int(cfg["reb"]))
    elif fam == "bandit":
        idx, chosen, W = orig.run_bandit_v(data, cfg, fit)
    else:
        # error-correcting family: the bandit_tuning loop with the variant's tracker fit swapped
        # in (the penalty reference is the target weights, so prev_i is not needed)
        old = bt.fitted_weights
        bt.fitted_weights = lambda _d, i, window, delta, w_prev, prev_i=None: fit(i, window, delta, w_prev)
        try:
            idx, chosen, W = bt.run_bandit(data, cfg, fixes=True)
        finally:
            bt.fitted_weights = old
    p_ret, p_tri, _, turn = rt.simulate(data, idx, W)
    row = dict(name=bt.config_name(cfg), variant=variant, **cfg)
    for tag, seg in (("train", rt.TRAIN), ("test", rt.TEST), ("full", rt.FULL)):
        for k, v in rt.segment(data, p_ret, p_tri, turn, seg).items():
            row[f"{tag}_{k}"] = v
    row["n_decisions"] = len(idx)
    row["fallbacks"] = fit.fallbacks
    row["stuck"] = fit.stuck
    return row, np.asarray(W), np.asarray(chosen), np.asarray(idx)


def active_returns(data, idx, W):
    p_ret, p_tri, _, _ = rt.simulate(data, idx, W)
    bench_tri = (1 + data.B.reindex(p_ret.index).fillna(0.0)).cumprod()
    return (np.log(p_tri).diff() - np.log(bench_tri).diff()).dropna()


def summarise(data, variant, df, paths_, select=st.select):
    """Summary row of a grid: the setting chosen by `select` (df -> (row, eligible rows);
    by default the original rule, select_tuning.select), its scores, its window shares on
    the test period, and a DM test against the best fixed window (by test TE) at the same
    rebalancing and delta. paths_ maps '<name>__W' / '__win' / '__idx' to arrays."""
    sel, elig = select(df)
    name = sel["name"]
    # fixed windows at the selected configuration's rebalancing and delta, same variant
    fx = df[(df.family == "fixed") & (df.reb == sel["reb"]) & (df.delta == sel["delta"])] if sel["family"] != "fixed" else df.iloc[0:0]
    best_fx = fx.sort_values("test_te").iloc[0] if len(fx) else None
    row = dict(variant=variant, selected=name, n_eligible=len(elig),
               anchors_test=int(((df.test_adf < 0.05) & (df.test_za < 0.05)).sum()),
               anchors_test_bandit=int(((df.test_adf < 0.05) & (df.test_za < 0.05) & (df.family == "bandit")).sum()))
    for k in ("train_joint", "train_te", "train_to", "test_te", "test_to", "test_drag", "test_adf", "test_za",
              "test_hl", "full_te", "full_to", "full_drag", "full_adf", "full_za", "fallbacks", "stuck"):
        row[k] = sel[k]
    chosen = paths_[f"{name}__win"]
    idx = paths_[f"{name}__idx"]
    dates = data.A.index[idx]
    test_mask = dates >= pd.Timestamp("2017-01-01")
    for w in bt.WINDOWS:
        row[f"test_share_w{w}"] = float((chosen[test_mask] == w).mean()) if test_mask.any() else np.nan
    if best_fx is not None:
        row["best_fixed_test"] = best_fx["name"]
        row["best_fixed_test_te"] = best_fx["test_te"]
        a_sel = active_returns(data, idx, paths_[f"{name}__W"])
        a_fx = active_returns(data, paths_[f"{best_fx['name']}__idx"], paths_[f"{best_fx['name']}__W"])
        for tag, (start, end, _) in (("test", rt.TEST), ("full", rt.FULL)):
            a, f = a_sel.loc[start:end], a_fx.loc[start:end]
            c = a.index.intersection(f.index)
            r = dm_test(a.loc[c].values, f.loc[c].values)
            row[f"dm_{tag}_vs_best_fixed_p"] = r["pvalue_two_sided"]
            row[f"dm_{tag}_vs_best_fixed_stat"] = r["dm_stat_hln"]
    return row


def compare_with_original_grid(df):
    """The baseline grid must reproduce the original grid's scores, when that grid exists."""
    ref_path = paths.ORIGINAL_GRID / "grid_scores.csv"
    if not ref_path.exists():
        print(f"skipped: comparison with the original grid ({ref_path} not found; "
              "`python src/run_tuning.py grid` writes it)", flush=True)
        return
    ref = pd.read_csv(ref_path).set_index("name")
    cols = [c for c in ref.columns if c.startswith(("train_", "test_", "full_")) and ref[c].dtype != object]
    mine = df.set_index("name").loc[ref.index, cols]
    diff = float(np.nanmax(np.abs(mine.values.astype(float) - ref[cols].values.astype(float))))
    print(f"baseline grid reproduces {ref_path.name}: max abs diff {diff:.2e}", flush=True)


def main(which):
    from joblib import Parallel, delayed
    t0 = time.time()
    data = rt.load()
    paths.ensure(OUT)
    cfgs = bt.grid()
    sp = OUT / "grid_summary.csv"
    summary = [r for r in pd.read_csv(sp).to_dict("records") if r["variant"] not in which] if sp.exists() else []
    for variant in which:
        t1 = time.time()
        res = Parallel(n_jobs=orig.N_JOBS, verbose=0)(delayed(evaluate_v)(data, c, variant) for c in cfgs)
        df = pd.DataFrame([r[0] for r in res])
        df["train_joint"] = df[["train_adf", "train_za"]].max(axis=1, skipna=False)
        df.to_csv(OUT / f"grid_{variant}.csv", index=False)
        grid_paths = {}
        for r in res:
            n = r[0]["name"]
            grid_paths[f"{n}__W"], grid_paths[f"{n}__win"], grid_paths[f"{n}__idx"] = r[1], r[2], r[3]
        np.savez_compressed(OUT / f"grid_{variant}_paths.npz", **grid_paths)
        if variant == "baseline":
            compare_with_original_grid(df)
        s = summarise(data, variant, df, grid_paths)
        s["grid_stuck_total"] = int(df["stuck"].sum())
        s["grid_fallbacks_total"] = int(df["fallbacks"].sum())
        summary.append(s)
        pd.DataFrame(summary).to_csv(sp, index=False)
        print(f"{variant}: {len(df)} configs in {(time.time() - t1) / 60:.1f} min; selected {s['selected']}; "
              f"test TE {s['test_te']:.4f} vs best fixed {s.get('best_fixed_test', '-')} "
              f"{s.get('best_fixed_test_te', float('nan')):.4f}; test ADF {s['test_adf']:.3f} ZA {s['test_za']:.3f}; "
              f"anchors on test {s['anchors_test']}", flush=True)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary).T.to_string())
    print(f"DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["baseline"]
    unknown = [v for v in which if v not in VARIANTS]
    if unknown:
        sys.exit(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")
    main(which)
