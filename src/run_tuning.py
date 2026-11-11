"""
Score the 910-configuration tuning grid of the first version of the bandit
(bandit_tuning.grid) on the training, test and full periods.

    python src/run_tuning.py cache   # the ridge cache only (paths.CACHE)
    python src/run_tuning.py grid    # all 910 configurations (step 0, optional)

grid writes paths.ORIGINAL_GRID / grid_scores.csv (one row per configuration:
tracking error, turnover, fee drag, ADF, Zivot-Andrews, Engle-Granger and
half-life on each period) and grid_paths.npz (decision indices, windows and
weights). original_spec.py and robustness_grid.py check their baseline against
grid_scores.csv when it exists, and select_tuning.py prints the configuration
the selection rule picks from it (the first version in Table tab:original).

The module is also a library: load, simulate and segment are the shared
scoring code of the tuning scripts, and TRAIN / TEST / FULL define the periods.
Input: data/data.xlsx (via run_models.load_data).
"""
import os
import sys
import time
import warnings

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.stattools import adfuller, coint  # noqa: E402

import bandit_tuning as bt  # noqa: E402
import paths  # noqa: E402
from breakpoint_coint import za_break_test  # noqa: E402
from metrics import compute_metrics, half_life, simulate_with_rebased_tri  # noqa: E402

OUT = paths.ORIGINAL_GRID
# (start, end, complete calendar years over which turnover is averaged)
TRAIN = (None, "2016-12-30", range(2010, 2017))
TEST = ("2017-01-01", None, range(2017, 2024))
FULL = (None, None, range(2010, 2024))
# Parallel workers; the results do not depend on it.
N_JOBS = int(os.environ.get("N_JOBS", max(1, (os.cpu_count() or 2) - 1)))


def load():
    """The high-equity data as bandit_tuning.Data, with the ridge cache and features for the monthly schedule."""
    import run_models as rm
    data = bt.Data(*rm.load_data("ma"))
    bt.build_ridge_cache(data, bt.schedule(data.n, 21), n_jobs=N_JOBS)
    for i in bt.schedule(data.n, 21):
        data.features(i)
    return data


def simulate(data, idx, W):
    """Simulate weights W set at decision indices idx (no costs)."""
    weights = pd.DataFrame(W, index=data.A.index[idx], columns=data.A.columns)
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(data.A, weights)
    return p_ret, p_tri, eff_w, turn


def segment(data, p_ret, p_tri, turn, seg):
    """Scores of a simulated run on one period (TRAIN, TEST or FULL)."""
    start, end, years = seg
    m = compute_metrics(p_ret, p_tri, data.B, turn, eval_start=start, eval_end=end)
    bench_tri = (1 + data.B.reindex(p_ret.index).fillna(0.0)).cumprod()
    df = pd.concat([bench_tri.rename("B"), p_tri.rename("P")], axis=1).dropna().loc[start:end]
    s = np.log(df["P"] / df["B"]).values
    adf = float(adfuller(s, regression="ct", autolag="AIC")[1])
    za = float(za_break_test(pd.Series(s)).get("za_pvalue", np.nan))
    eg = float(coint(np.log(df["B"].values), np.log(df["P"].values), trend="ct", autolag="AIC")[1])
    joint = np.nan if np.isnan(adf) or np.isnan(za) else max(adf, za)
    # mean one-way turnover over the period's complete calendar years (years
    # with no rebalance count as zero); avg_yearly_turnover kept alongside
    to_fy = float(turn.groupby(turn.index.year).sum().reindex(list(years), fill_value=0.0).mean())
    return dict(te=float(m["annualised_te"]), to=to_fy, to_ay=float(m["avg_yearly_turnover"]),
                drag=float(m["fee_drag_pa"]), adf=adf, za=za, eg=eg, joint=joint,
                hl=float(half_life(s)), start=str(df.index[0].date()), end=str(df.index[-1].date()))


def evaluate(data, cfg):
    """Run one configuration and score it on the three periods."""
    idx, chosen, W = bt.run_config(data, cfg)
    p_ret, p_tri, _, turn = simulate(data, idx, W)
    row = dict(name=bt.config_name(cfg), **cfg)
    for tag, seg in (("train", TRAIN), ("test", TEST), ("full", FULL)):
        for k, v in segment(data, p_ret, p_tri, turn, seg).items():
            row[f"{tag}_{k}"] = v
    row["n_decisions"] = len(idx)
    return row, np.asarray(W), np.asarray(chosen), np.asarray(idx)


def stage_grid(data):
    from joblib import Parallel, delayed
    paths.ensure(OUT)
    cfgs = bt.grid()
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(evaluate)(data, c) for c in cfgs)
    rows = [r[0] for r in res]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "grid_scores.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "grid_paths.npz"),
                        **{f"{r[0]['name']}__W": r[1] for r in res},
                        **{f"{r[0]['name']}__win": r[2] for r in res},
                        **{f"{r[0]['name']}__idx": r[3] for r in res})
    print(f"GRID DONE {len(rows)} configurations in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if stage not in ("cache", "grid"):
        sys.exit("usage: python src/run_tuning.py [cache|grid]")
    t0 = time.time()
    data = load()
    print(f"loaded + cache in {(time.time() - t0) / 60:.1f} min; n = {data.n}, "
          f"decisions (reb 21) = {len(bt.schedule(data.n, 21))}", flush=True)
    if stage == "grid":
        stage_grid(data)
