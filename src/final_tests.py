"""Significance and break tests on the final runs (stages dm, features and breaks of run_all.py).

Stages:

  dm        Diebold-Mariano tests of squared daily tracking error, HLN-corrected:
            the 30-pair family of inference.py on the full and test periods
            (inference_dm.csv, inference_dm_test.csv), the 8 static ridges against
            equal weight (dm_h1.csv), then dm_pooled.pool(), which adjusts all 38 pairs
            per period in one Benjamini-Hochberg family (inference_dm_pooled38.csv,
            column p_bh38: tab:dm, tab:dm_variants, tab:dm_h1). Also recomputes the
            moving-block bootstrap and HAC confidence intervals of every run
            (bootstrap_ci.csv: tab:ci). About a minute.
  breaks    Lumsdaine-Papell and Lee-Strazicich two-break unit-root tests
            (two_break_unit_root.py) on all 11 spreads of tab:two_breaks, each
            computed here from its run in final/timeseries: break dates searched over
            the middle 70% of the sample, at least 10% apart, on a monthly grid
            refined to the day; the lag from the AIC choice of the ADF test; the
            199-draw sieve wild bootstrap; 12 and 20 lags; large daily moves capped
            (two_break.csv). Simulates the critical values first if
            two_break_critical_values.csv is missing (600 random walks). Every draw
            has its own seed, so a rerun gives the same numbers. Then writes the table
            (stage `table`). Slow: about 5 minutes a spread on 4 workers, plus the
            critical values.
  table     tab_two_breaks.csv: two_break.csv with the Gregory-Hansen C/S statistics
            (test and full periods) of gregory_hansen.csv, so run gregory_hansen.py
            run first. Seconds.
  features  Reruns the selected bandit with each context feature zeroed in turn and
            counts the decisions that change (feature_zeroing.csv). Under a minute
            once the ridge caches exist.

    python src/final_tests.py dm
    python src/final_tests.py features
    python src/final_tests.py breaks
    python src/final_tests.py table

Set the environment variable N_JOBS to change the number of worker processes of
`breaks` (default: all cores but one); the results do not depend on it.

Inputs: final/timeseries/*.csv (final_runs.py); for features also data/data.xlsx,
grids/selection_outside.csv (training_mae.py), grids/grid_final.csv and
grids/grid_final_paths.npz (final_bandit.py); for table also final/gregory_hansen.csv.
Outputs: in final/: inference_dm.csv, inference_dm_test.csv, dm_h1.csv,
inference_dm_pooled38.csv, bootstrap_ci.csv, two_break.csv,
two_break_critical_values.csv, tab_two_breaks.csv, feature_zeroing.csv.
"""
import os
import sys
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402

BOOT = 199
WORKERS = int(os.environ.get("N_JOBS", max(1, (os.cpu_count() or 2) - 1)))
EW = "equal_weight__ma__reb21__tc0"
# Hypothesis H1 (tracking part): each static ridge against equal weight.
H1 = ["ridge_trend_to__ma__w252__reb21__d1__tc0", "ridge_no_trend__ma__w252__reb21__tc0",
      "ridge_trend__ma__w252__reb21__tc0"] + [f"window_fixedbest__ma__w{w}__reb63__tc0" for w in (126, 252, 378, 504, 756)]


def _low_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def dm():
    import dm_pooled
    import dm_test
    import inference
    paths.ensure(paths.FINAL)
    dm_test.TS_DIR = paths.FINAL / "timeseries"
    # inference.run_inference_panel computes bootstrap_ci.csv only when it is missing,
    # so remove an old copy to keep the intervals in step with the runs.
    (paths.FINAL / "bootstrap_ci.csv").unlink(missing_ok=True)
    for start, name in ((None, "inference_dm.csv"), ("2017-01-01", "inference_dm_test.csv")):
        inference.run_inference_panel(results_dir=str(paths.FINAL), start=start, out_name=name)
    rows = []
    for per, start in (("test", "2017-01-01"), ("full", None)):
        res = []
        for run in H1:
            e1, e2 = dm_test.load_active_returns(run), dm_test.load_active_returns(EW)
            idx = e1.index.intersection(e2.index)
            if start:
                idx = idx[idx >= pd.Timestamp(start)]
            r = dm_test.dm_test(e1.loc[idx].to_numpy(), e2.loc[idx].to_numpy(), h=1)
            res.append((run, r["dm_stat_hln"], r["pvalue_two_sided"]))
        bh = inference.benjamini_hochberg(np.array([x[2] for x in res]))
        rows += [dict(period=per, run=a, dm=b, p_raw=c, p_bh=d) for (a, b, c), d in zip(res, bh)]
    pd.DataFrame(rows).to_csv(paths.FINAL / "dm_h1.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    dm_pooled.pool(paths.FINAL)


def breaks(only=None, out_name="two_break.csv", boot=BOOT):
    """Two-break tests on the 11 spreads of tab:two_breaks, written to final/<out_name>,
    then the table (tab_two_breaks.csv) when all 11 are done.

    only: a subset of the runs (for testing). Each run's bootstrap seeds are tied to its
    position in two_break_unit_root.MODELS, so a subset gives the same rows."""
    import two_break_unit_root as tb
    break_runs = [run for run, _ in tb.MODELS]
    todo = [r for r in break_runs if not only or r in only]
    spreads = {run: tb.load_spread(run) for run in todo}
    T = len(next(iter(spreads.values())))
    assert all(len(s) == T for s in spreads.values()), "spreads differ in length"
    cv_path = paths.ensure(paths.FINAL) / "two_break_critical_values.csv"
    rows = []
    with Pool(WORKERS, initializer=_low_priority) as pool:
        if not cv_path.exists():
            t0 = time.time()
            print(f"{cv_path.name} missing: simulating {tb.CV_REPS} random walks of length {T}", flush=True)
            pd.DataFrame([tb.critical_values(T, pool)]).to_csv(cv_path, index=False)
            print(f"wrote {cv_path} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        cv = pd.read_csv(cv_path).iloc[0]
        assert int(cv["T"]) == T, f"critical values are for T = {int(cv['T'])}, spreads have T = {T}"
        print(f"LP cv 1/5/10%: {cv['lp_cv_0.01']:.2f} {cv['lp_cv_0.05']:.2f} {cv['lp_cv_0.1']:.2f}; "
              f"LS cv: {cv['ls_cv_0.01']:.2f} {cv['ls_cv_0.05']:.2f} {cv['ls_cv_0.1']:.2f}", flush=True)
        for run in todo:
            n_m = break_runs.index(run)
            t1 = time.time()
            s = spreads[run]
            y = s.to_numpy()
            k = tb.aic_lag(y)
            lp, lb1, lb2 = tb.lp_stat(y, k)
            ls, sb1, sb2 = tb.ls_stat(y, k)
            phi, resid = tb.null_ar(y, k)
            boots = np.array(pool.map(tb._boot, [(7_000_000 + 10_000 * n_m + r, phi, resid, T) for r in range(boot)]))
            yw = tb.winsorise_level(y)
            kw = tb.aic_lag(yw)
            row = dict(run=run, T=T, lags=k, lp_stat=lp, lp_p_boot=tb.pval(boots[:, 0], lp),
                       lp_break1=s.index[lb1].date().isoformat(), lp_break2=s.index[lb2].date().isoformat(),
                       ls_stat=ls, ls_p_boot=tb.pval(boots[:, 1], ls),
                       ls_break1=s.index[sb1].date().isoformat(), ls_break2=s.index[sb2].date().isoformat(),
                       lp_k12=tb.lp_stat(y, 12)[0], lp_k20=tb.lp_stat(y, 20)[0],
                       ls_k12=tb.ls_stat(y, 12)[0], ls_k20=tb.ls_stat(y, 20)[0],
                       lp_capped=tb.lp_stat(yw, kw)[0], ls_capped=tb.ls_stat(yw, kw)[0])
            for q in (0.01, 0.05, 0.1):
                row[f"lp_rej_sim_{q:g}"] = bool(lp <= cv[f"lp_cv_{q:g}"])
                row[f"ls_rej_sim_{q:g}"] = bool(ls <= cv[f"ls_cv_{q:g}"])
            rows.append(row)
            pd.DataFrame(rows).to_csv(paths.FINAL / out_name, index=False)
            print(f"{run:45s} LP {lp:6.2f} p {row['lp_p_boot']:.3f} {row['lp_break1']}/{row['lp_break2']} | "
                  f"LS {ls:6.2f} p {row['ls_p_boot']:.3f} {row['ls_break1']}/{row['ls_break2']} | "
                  f"capped {row['lp_capped']:.2f}/{row['ls_capped']:.2f} ({(time.time() - t1) / 60:.1f} min)", flush=True)
    print("DONE", flush=True)
    if not only and out_name == "two_break.csv":
        table()


def table():
    """tab:two_breaks, all 11 rows in the thesis's order -> final/tab_two_breaks.csv:
    two_break.csv plus the Gregory-Hansen C/S statistic and its 5% verdict for the test
    and full periods (from gregory_hansen.csv, when it exists)."""
    import two_break_unit_root as tb
    tab = pd.read_csv(paths.FINAL / "two_break.csv")
    runs = [run for run, _ in tb.MODELS]
    assert list(tab.run) == runs, f"two_break.csv must hold the {len(runs)} runs of the table, has {list(tab.run)}"
    gh_path = paths.FINAL / "gregory_hansen.csv"
    if gh_path.exists():
        gh = pd.read_csv(gh_path)
        for per in ("test", "full"):
            g = gh[gh.period == per].set_index("run")
            tab[f"gh_cs_{per}"] = tab["run"].map(g["CS_stat"])
            tab[f"gh_cs_{per}_rej5"] = tab["run"].map(g["CS_rej5"])
        tab.insert(1, "model", tab["run"].map(gh.drop_duplicates("run").set_index("run")["model"]))
    else:
        print(f"{gh_path} missing: run gregory_hansen.py run first for the Gregory-Hansen columns", flush=True)
    tab.to_csv(paths.FINAL / "tab_two_breaks.csv", index=False)
    print(f"wrote {paths.FINAL / 'tab_two_breaks.csv'} ({len(tab)} rows)", flush=True)


FEATURES = ["intercept", "benchmark volatility", "active volatility", "momentum", "autocorrelation",
            "drawdown", "volatility of volatility", "cross-sectional dispersion", "recent drift"]


def features():
    """Which context features the selected bandit's choices depend on: rerun it with each
    feature zeroed in turn (with a ridge prior, a feature fixed at zero is the same as
    leaving it out) and count the decisions that pick another window. The rerun with all
    features must give the windows the final grid stored for the setting."""
    import bandit_tuning as bt
    import final_bandit as fb
    import run_tuning as rt
    data = rt.load()
    ro = fb.build_ridge_cache_out(data, bt.schedule(data.n, 21))
    name, cfg = fb.selected_config()
    spec = fb.SPECS["final"]

    def run(extra):
        fit = fb.FitB(data, trend_out=True, n_ref=spec["n_ref"], ridge_out=ro, to_ref=spec["to_ref"])
        return fb.run_bandit_m(data, dict(cfg, **extra), fit, drift=True, reward="mae")[1]
    base = run({})
    z = np.load(paths.GRIDS / "grid_final_paths.npz")
    assert list(base) == list(z[f"{name}__win"]), "the rerun bandit's windows differ from the grid's"
    rows = []
    for j in range(1, 9):
        ch = run(dict(features=[k for k in range(9) if k != j]))
        rows.append(dict(feature=FEATURES[j], decisions_changed=int(sum(a != b for a, b in zip(base, ch))),
                         of=len(base)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(paths.ensure(paths.FINAL) / "feature_zeroing.csv", index=False)
    print(f"features [{name}] -> {paths.FINAL / 'feature_zeroing.csv'}", flush=True)


STAGES = {"dm": dm, "features": features, "breaks": breaks, "table": table}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        sys.exit("usage: python src/final_tests.py dm|features|breaks|table")
    _low_priority()
    STAGES[sys.argv[1]]()
