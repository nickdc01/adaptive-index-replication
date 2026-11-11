"""
The first version of the bandit (the original specification), the effect of each change,
and Table tab:original (appendix "The Original Specification").

Every run here keeps the first version's setting (LinUCB, kappa 0.6, eta 0.5, lambda 2,
c_e 252, gamma 1, delta 1, quarterly), which the original selection rule chose on the
training period from the original 910-setting grid (robustness_grid.py baseline
re-runs that grid and its selection), and changes one thing:

  A  trend outside the log. The tracker fits
         log y_i = log(sum_j X_ij w_j) + beta * (i - 1)
     (a proportional drift, like the fee-drag regression) instead of
         log y_i = log(sum_j X_ij w_j + beta * (i - 1)).
     Both the cross-validation pass and the refit change, so the ridge strength is
     chosen again for every (decision, window). Unlike Equation 3.4 of the final
     specification (final_bandit.py), this fit has no constant.

  B  turnover penalty scaled with window length: delta_eff = delta * n / n_ref.
     The fitting loss is a sum over the n days of the window, so a fixed delta
     weighs less on long windows; scaling it by n is the same as normalising the
     loss by n. n_ref = 504 leaves the 2-year window unchanged, n_ref = 252 the
     1-year window.

  C  reward on drifting weights. The reward portfolio buys and holds the decision's
     weights over the c_e days, as the reported portfolio drifts between
     rebalances, instead of being reset to them every day.

  D  turnover penalty measured against the drifted weights: the previous target
     weights after drifting with asset returns to the close before the rebalance
     (final_bandit.drifted_reference), instead of the previous target weights themselves.

  A+B504+C combines the first three ("All three, delta by n/504").

The other three rows of tab:original come from the final grids (final_bandit.py and
training_mae.py): "All five, delta by n/252" is the first version's setting with eta = 0
under the final specification (bandit_k0.6_e0_g1_d1_reb63_ce252 of grid_final: A with the
constant, B with n/252, C, D and the MAE reward), "Tuned bandit (trend outside)" is the
selected setting, and "Bandit, trend inside the logarithm" is the trend-inside bandit with
the lowest training MAE among those with training turnover below 150% (grid_final_in).

The module also holds the tracker fit of these changes (the Fit class, with the
trend-outside fit refit_out and its ridge cache) and the copied bandit and fixed-window
loops, which robustness_grid.py and the self-checks of final_bandit.py use.

Built-in checks (run first; a failure stops the script): the copied bandit and
fixed-window loops reproduce bandit_tuning.run_bandit / run_fixed exactly with no change
switched on, the buy-and-hold reward portfolio is consistent, and the drifted reference
equals the simulator's weights on the day before each rebalance. If the original grid's
scores (results/original_grid/grid_scores.csv, from `python src/run_tuning.py grid`)
exist, the first version's scores must also match its row there; otherwise that
comparison is skipped.

    python src/original_spec.py          # checks, the seven runs, then the table
    python src/original_spec.py check    # checks only (fast)
    python src/original_spec.py runs     # checks and the seven runs (about 2 min; building the
                                         # trend-outside ridge cache adds about 3 min the first time)
    python src/original_spec.py table    # the table only, from scores.csv and the final grids
    python src/original_spec.py breaks   # two-break tests on the first version's spread (10 to
                                         # 20 minutes on 2 workers; quoted in the appendix text)

Inputs: data/data.xlsx; ridge caches in results/cache (built if missing). The table also
reads results/grids/grid_final.csv, grid_final_train_mae_all.csv,
grid_final_in_train_mae_all.csv and selection_outside.csv (`python src/final_bandit.py
final`, then `python src/training_mae.py`); without them the default run stops after the
seven runs.
Outputs:
  results/robustness/scores.csv        TE, turnover, fee drag, ADF / ZA / EG p and half-life
                                       by run and period (train, test, full), with the number
                                       of refits that fell back to equal weight (fallbacks)
                                       or stopped at the equal-weight start point (stuck)
  results/robustness/tab_original.csv  the ten rows of tab:original (fractions and p-values;
                                       the setting and its training MAE for the grid rows)
  results/robustness/two_break_first_version.csv   Lumsdaine-Papell and Lee-Strazicich
                                       statistics, break dates and bootstrap p-values of the
                                       first version's full-period spread (stage breaks)
  results/cache/ridge_trendout_tyears_<key>.npz   ridge strengths of the trend-outside fit
                                                  (rows A and A+B504+C)
"""
import os
import sys
import time
import warnings
from collections import deque

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import bandit_tuning as bt  # noqa: E402
import paths  # noqa: E402
import rl_window_selection as rws  # noqa: E402
import run_tuning as rt  # noqa: E402
import select_tuning as st  # noqa: E402
from final_bandit import TO_REFS, drifted_reference, selected_setting  # noqa: E402

OUT = paths.ROBUSTNESS
REB = 63
TUNED = dict(family="bandit", kappa=0.6, eta=0.5, lambda_coint=2.0, ce=252, gamma=1.0, delta=1.0, reb=REB)
TUNED_NAME = "bandit_k0.6_e0.5_g1_d1_reb63_ce252"
N_JOBS = int(os.environ.get("N_JOBS", 4))   # workers for a missing ridge cache, the break bootstrap and robustness_grid.py


# -----------------------------
# Change A: trend outside the log
# -----------------------------
# The trend regressor is measured in years (t / 252), so beta is a log return a year.
# This is a pure change of units (beta is not penalised), but with t in days SLSQP
# sometimes stopped after one step at the equal-weight start point.
TSCALE = 252.0


def _log_fit_out(X_aug, w, n_assets):
    port = X_aug[:, :n_assets] @ w[:n_assets]
    if np.any(port <= 0):
        return None
    return np.log(port) + w[n_assets] * X_aug[:, n_assets] / TSCALE


def cv_loss_out(X_aug, y_tri, ridge, n_assets, n_splits=3):
    """rws.cross_val_loss_drift_tri with the trend moved outside the log."""
    T = X_aug.shape[0]
    min_train = T // 3
    fold_size = (T - min_train) // n_splits
    val_losses = []
    for j in range(n_splits):
        train_end = min_train + j * fold_size
        val_end = min(train_end + fold_size, T)
        X_train, y_train = X_aug[:train_end], y_tri[:train_end]
        X_val, y_val = X_aug[train_end:val_end], y_tri[train_end:val_end]
        if len(X_val) < 5:
            continue

        def penalized_train_loss(w):
            f = _log_fit_out(X_train, w, n_assets)
            if f is None or np.any(y_train <= 0):
                return np.inf
            return np.sum((f - np.log(y_train)) ** 2) + ridge * np.sum(w[:n_assets] ** 2)

        def fold_loss(w):
            f = _log_fit_out(X_val, w, n_assets)
            if f is None or np.any(y_val <= 0):
                return np.inf
            return np.sum((f - np.log(y_val)) ** 2)

        bounds = [(0, 1)] * n_assets + [(None, None)]
        w0 = np.concatenate([np.ones(n_assets) / n_assets, np.array([0.0])])
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w[:n_assets]) - 1}
        res = minimize(penalized_train_loss, w0, bounds=bounds, constraints=cons, method='SLSQP')
        if not res.success:
            continue
        val_losses.append(fold_loss(res.x))
    return float(np.mean(val_losses)) if val_losses else np.inf


def best_ridge_out(X_aug, yw, n_assets):
    best_cv, best = np.inf, None
    for ridge in rws._default_ridge_grid():
        cv = cv_loss_out(X_aug, yw, ridge, n_assets, n_splits=3)
        if cv < best_cv:
            best_cv, best = cv, ridge
    return best


def refit_out(X_aug, yw, n_assets, ridge, delta_turnover, w_prev):
    """bandit_tuning.refit with the trend moved outside the log."""
    if ridge is None:
        return None
    bounds = [(0, 1)] * n_assets + [(None, None)]
    w0 = np.concatenate([np.ones(n_assets) / n_assets, np.array([0.0])])
    _w_prev = w_prev if w_prev is not None else np.ones(n_assets) / n_assets
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w[:n_assets]) - 1}

    def full_loss(w):
        f = _log_fit_out(X_aug, w, n_assets)
        if f is None or np.any(yw <= 0):
            return np.inf
        return (np.sum((f - np.log(yw)) ** 2) + ridge * np.sum(w[:n_assets] ** 2)
                + delta_turnover * np.sum((w[:n_assets] - _w_prev) ** 2))

    res = minimize(full_loss, w0, bounds=bounds, constraints=cons, method='SLSQP')
    if not res.success:
        return None
    return res.x[:n_assets].astype(float)


def _ridge_out_one(data, i, window):
    X_aug, yw = data.window(i, window)
    return best_ridge_out(X_aug, yw, data.n_assets)


def build_ridge_cache_out(data, indices):
    """Ridge strength of the trend-outside fit for every (decision index, window), cached
    on disk in results/cache and extended with any pairs not yet in it."""
    path = paths.CACHE / f"ridge_trendout_tyears_{data.key()}.npz"
    cache = {}
    if path.exists():
        z = np.load(path)
        for i, w, r in zip(z["i"], z["w"], z["r"]):
            cache[(int(i), int(w))] = None if np.isnan(r) else float(r)
    pairs = [(int(i), w) for i in indices for w in bt.WINDOWS if (int(i), w) not in cache]
    if pairs:
        from joblib import Parallel, delayed
        t0 = time.time()
        vals = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_ridge_out_one)(data, i, w) for i, w in pairs)
        for (i, w), r in zip(pairs, vals):
            cache[(i, w)] = None if r is None else float(r)
        keys = sorted(cache)
        paths.ensure(paths.CACHE)
        np.savez(path, i=np.array([k[0] for k in keys]), w=np.array([k[1] for k in keys]),
                 r=np.array([np.nan if cache[k] is None else cache[k] for k in keys]))
        print(f"trend-outside-log ridge cache: {len(pairs)} fits in {(time.time() - t0) / 60:.1f} min", flush=True)
    return cache


# -----------------------------
# The tracker fit, with changes A, B and D as switches
# -----------------------------
class Fit:
    """fitted_weights(data, i, window, delta, w_prev) with optional trend outside
    the log (A) and delta scaled by window / n_ref (B). Counts equal-weight fallbacks
    (failed refits) and stuck refits (reported as converged but left at the
    equal-weight start point, which an optimum essentially never is).

    to_ref sets the reference of the turnover penalty (D): 'target' (the previous
    rebalance's target weights w_prev) or 'drifted' (w_prev drifted to the close before
    this rebalance, final_bandit.drifted_reference; the caller passes prev_i, the
    decision index at which w_prev was set). Equal weight at the first rebalance either
    way; the cross-validation never sees w_prev."""

    def __init__(self, data, trend_out=False, n_ref=None, ridge_out=None, to_ref="target"):
        if to_ref not in TO_REFS:
            raise ValueError(f"to_ref must be one of {TO_REFS}, not {to_ref!r}")
        self.data, self.trend_out, self.n_ref, self.ridge_out = data, trend_out, n_ref, ridge_out
        self.to_ref = to_ref
        self.fallbacks = 0
        self.stuck = 0
        self.calls = 0
        self.ref_log = []      # (fit_end, prev_i, reference) of every fit under to_ref='drifted'

    def reference(self, i, w_prev, prev_i):
        """The penalty's reference for a fit ending at i (w_prev itself under 'target')."""
        if w_prev is None or self.to_ref == "target":
            return w_prev
        if prev_i is None:
            raise ValueError("to_ref='drifted' needs prev_i, the decision index of w_prev")
        ref = drifted_reference(self.data.asset_tri, w_prev, prev_i, i)
        self.ref_log.append((int(i), int(prev_i), ref))
        return ref

    def __call__(self, i, window, delta, w_prev, prev_i=None):
        self.calls += 1
        w_prev = self.reference(i, w_prev, prev_i)
        d = delta * window / self.n_ref if self.n_ref else delta
        X_aug, yw = self.data.window(i, window)
        if self.trend_out:
            w = refit_out(X_aug, yw, self.data.n_assets, self.ridge_out[(i, window)], d, w_prev)
        else:
            w = bt.refit(X_aug, yw, self.data.n_assets, self.data.ridge[(i, window)], d, w_prev)
        if w is None:
            self.fallbacks += 1
            w = np.ones(self.data.n_assets) / self.data.n_assets
        elif np.allclose(w, 1.0 / self.data.n_assets, atol=1e-6, rtol=0):
            self.stuck += 1
        return w


# -----------------------------
# Change C: the reward portfolio
# -----------------------------
def reward_port_returns(data, i0, i1, w, drift):
    R = data.Ar[i0:i1]
    if not drift:
        return R @ w                       # reset to w every day (the first version)
    V = np.cumprod(1.0 + R, axis=0) @ w    # buy and hold w from i0
    return V / np.concatenate([[1.0], V[:-1]]) - 1.0


# -----------------------------
# Copies of bandit_tuning.run_bandit (fixes=True, no lag or context subset) and run_fixed
# -----------------------------
def run_bandit_v(data, cfg, fit, drift=False):
    kappa, eta, lam = float(cfg["kappa"]), float(cfg["eta"]), float(cfg.get("lambda_coint", 2.0))
    tc = float(cfg.get("tc_bps", 0.0))
    gamma, delta, reb = float(cfg["gamma"]), float(cfg["delta"]), int(cfg["reb"])
    ce = int(cfg["ce"])
    horizon = max(bt.FORWARD_DAYS, ce)
    idx = bt.schedule(data.n, reb)
    bandit = bt._make_bandit(True, 9, kappa, gamma)
    pending, hist = deque(), []
    prev_w, prev_i = None, None
    chosen, weights = [], []
    ones = np.ones(data.n_assets) / data.n_assets
    for k, i in enumerate(idx):
        while pending and i >= pending[0]["i"] + horizon:
            p = pending.popleft()
            turn = float(np.abs(p["w"] - p["prev"]).sum())
            ce_end = min(p["i"] + ce, data.n)
            c = rws.cointegration_reward(reward_port_returns(data, p["i"], ce_end, p["w"], drift),
                                         data.Br[p["i"]:ce_end])
            reward = lam * c - tc * turn - eta * turn
            hist.append(reward)
            rn = (reward - np.mean(hist)) / (np.std(hist) + 1e-8) if len(hist) > 1 else 0.0
            bandit.update(p["a"], p["x"], rn)
        x = data.features(i)
        alpha_t = kappa * max(1.0, 3.0 / np.sqrt(1 + k))
        a = bandit.select(x, alpha_override=alpha_t)
        bandit.register(a, x)
        # prev_i: the decision at which prev_w was set, for a drifted penalty reference
        w = fit(i, bt.WINDOWS[a], delta, prev_w, prev_i=prev_i)
        pending.append(dict(a=a, x=x, w=w, i=i, prev=prev_w if prev_w is not None else ones))
        chosen.append(bt.WINDOWS[a])
        weights.append(w)
        prev_w, prev_i = w, i
    return idx, chosen, np.array(weights)


def run_fixed_v(data, window, fit, delta=1.0, reb=REB):
    idx = bt.schedule(data.n, reb)
    prev_w, prev_i, weights = None, None, []
    for i in idx:
        w = fit(i, window, delta, prev_w, prev_i=prev_i)
        weights.append(w)
        prev_w, prev_i = w, i
    return idx, [window] * len(idx), np.array(weights)


# -----------------------------
# Scoring
# -----------------------------
SEGS = (("train", rt.TRAIN), ("test", rt.TEST), ("full", rt.FULL))


def score(data, run, model, idx, W, fallbacks, calls, stuck=0):
    """One row per period (train, test, full) with run_tuning.segment's metrics."""
    p_ret, p_tri, _, turn = rt.simulate(data, idx, W)
    rows = []
    for tag, seg in SEGS:
        m = rt.segment(data, p_ret, p_tri, turn, seg)
        rows.append(dict(run=run, model=model, period=tag, fallbacks=fallbacks, stuck=stuck, refits=calls, **m))
    return rows


# -----------------------------
# Main
# -----------------------------
def check(data):
    """The copied loops must reproduce bandit_tuning exactly, and the first version must
    reproduce its row of the original grid when that grid has been run."""
    ok = True
    idx0, ch0, W0 = bt.run_config(data, TUNED)
    b_idx, b_ch, b_W = run_bandit_v(data, TUNED, Fit(data))
    same = list(idx0) == list(b_idx) and list(ch0) == list(b_ch) and np.array_equal(W0, b_W)
    print(f"bandit copy reproduces bandit_tuning.run_bandit: {same}", flush=True)
    ok &= same
    for w in bt.WINDOWS:
        f0_idx, _, f0_W = bt.run_config(data, dict(family="fixed", window=w, delta=1.0, reb=REB))
        f1_idx, _, f1_W = run_fixed_v(data, w, Fit(data))
        same = list(f0_idx) == list(f1_idx) and np.array_equal(f0_W, f1_W)
        print(f"fixed {w} copy reproduces bandit_tuning.run_fixed: {same}", flush=True)
        ok &= same
    # Change C with buy-and-hold must give the same return as the daily-reset
    # portfolio on day 1 and differ afterwards (sanity of reward_port_returns).
    r_c = reward_port_returns(data, 1000, 1252, b_W[5], False)
    r_d = reward_port_returns(data, 1000, 1252, b_W[5], True)
    bh = float(np.prod(1 + r_d))
    bh_direct = float((np.prod(1 + data.Ar[1000:1252], axis=0) * b_W[5]).sum())
    print(f"drift reward: day-1 equal {np.isclose(r_c[0], r_d[0])}, "
          f"buy-and-hold growth matches direct {np.isclose(bh, bh_direct)}, "
          f"mean |daily diff| {np.mean(np.abs(r_c - r_d)):.2e}", flush=True)
    ok &= bool(np.isclose(r_c[0], r_d[0]) and np.isclose(bh, bh_direct))
    # Change D: each drifted reference must be the simulator's effective weights at the
    # close before its rebalance, drifted from the previous decision's target weights.
    fit = Fit(data, to_ref="drifted")
    d_idx, _, d_W = run_fixed_v(data, 504, fit)
    _, _, eff_w, _ = rt.simulate(data, d_idx, d_W)
    dates = data.A.index
    gap = max(float(np.abs(ref - eff_w.loc[dates[i - 1]].to_numpy(float)).max()) for i, _, ref in fit.ref_log)
    order = ([i for i, _, _ in fit.ref_log] == [int(i) for i in d_idx[1:]]
             and [p for _, p, _ in fit.ref_log] == [int(i) for i in d_idx[:-1]])
    print(f"drifted reference equals the simulator's pre-rebalance weights: max gap {gap:.2e}, "
          f"previous decision found at every rebalance {order}", flush=True)
    ok &= bool(gap < 1e-12 and order)
    ref_path = paths.ORIGINAL_GRID / "grid_scores.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path).set_index("name").loc[TUNED_NAME]
        match = True
        for r in score(data, "baseline", "bandit", b_idx, b_W, 0, 0):
            for k in ("te", "to", "adf", "za"):
                d = abs(r[k] - ref[f"{r['period']}_{k}"])
                if d > 1e-10:
                    print(f"  MISMATCH {r['period']} {k}: {r[k]} vs {ref[r['period'] + '_' + k]}")
                    match = False
        print(f"first version reproduces its row of {ref_path.name}: {match}", flush=True)
        ok &= match
    else:
        print(f"skipped: comparison with the original grid ({ref_path} not found; "
              "`python src/run_tuning.py grid` writes it)", flush=True)
    print("CHECK", "PASSED" if ok else "FAILED", flush=True)
    return ok


# (run name, tracker-fit switches, reward on drifting weights?). Every row except D
# measures the penalty against the previous target weights.
RUNS = [
    ("baseline", dict(), False),                                # first version
    ("A_trend_outside_log", dict(trend_out=True), False),
    ("B_delta_scaled_n504", dict(n_ref=504), False),
    ("B_delta_scaled_n252", dict(n_ref=252), False),
    ("C_reward_drifting_weights", dict(), True),
    ("D_drifted_reference", dict(to_ref="drifted"), False),
    ("A+B504+C", dict(trend_out=True, n_ref=504), True),        # all three
]


def runs(data):
    """The seven runs of the first version's setting, written to results/robustness/scores.csv."""
    t0 = time.time()
    ridge_out = build_ridge_cache_out(data, bt.schedule(data.n, REB))
    paths.ensure(OUT)
    score_rows = []
    for run, kw, drift in RUNS:
        t1 = time.time()
        fit = Fit(data, ridge_out=ridge_out, **kw)
        b_idx, _, b_W = run_bandit_v(data, TUNED, fit, drift=drift)
        score_rows += score(data, run, "bandit", b_idx, b_W, fit.fallbacks, fit.calls, fit.stuck)
        print(f"{run}: {(time.time() - t1):.0f}s", flush=True)
    sdf = pd.DataFrame(score_rows)
    sdf.to_csv(OUT / "scores.csv", index=False)
    pd.set_option("display.width", 200)
    show = sdf[sdf.period.isin(["test", "full"])][["run", "period", "te", "to", "drag", "adf", "za", "hl", "fallbacks", "stuck"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"runs: {(time.time() - t0) / 60:.1f} min", flush=True)


# -----------------------------
# Table tab:original
# -----------------------------
COLS = ["test_te", "test_drag", "test_to", "test_adf", "test_za", "full_te", "full_to", "full_adf", "full_za"]
# (label, run in scores.csv, penalty reference), in the order of the table
TABLE_RUNS = [
    ("First version", "baseline", "target"),
    ("Trend outside the logarithm", "A_trend_outside_log", "target"),
    ("delta scaled by n/252", "B_delta_scaled_n252", "target"),
    ("delta scaled by n/504", "B_delta_scaled_n504", "target"),
    ("Reward weights drift", "C_reward_drifting_weights", "target"),
    ("Penalty against the drifted weights", "D_drifted_reference", "drifted"),
    ("All three, delta by n/504", "A+B504+C", "target"),
]
ALL_FIVE = "bandit_k0.6_e0_g1_d1_reb63_ce252"    # the first version's setting with eta = 0, in grid_final
# tab:original as printed in the thesis: TE, fee drag and turnover in %, then p-values
PRINTED = {
    "First version": (4.85, -1.28, 34, 0.696, 0.001, 4.90, 37, 0.816, 0.165),
    "Trend outside the logarithm": (5.17, -1.76, 41, 0.939, 0.014, 5.04, 39, 0.922, 0.841),
    "delta scaled by n/252": (4.96, -1.25, 27, 0.717, 0.001, 5.01, 34, 0.867, 0.933),
    "delta scaled by n/504": (5.18, -1.90, 46, 0.933, 0.001, 5.06, 50, 0.851, 0.992),
    "Reward weights drift": (4.96, -1.33, 33, 0.678, 0.001, 5.02, 41, 0.747, 0.910),
    "Penalty against the drifted weights": (5.21, -1.61, 35, 0.918, 0.023, 5.07, 34, 0.929, 0.852),
    "All three, delta by n/504": (5.00, -1.19, 33, 0.708, 0.001, 4.98, 37, 0.871, 0.918),
    "All five, delta by n/252": (5.25, -1.72, 23, 0.921, 0.001, 5.13, 24, 0.965, 0.247),
    "Tuned bandit (trend outside)": (5.11, -1.29, 19, 0.817, 0.001, 5.07, 22, 0.958, 0.192),
    "Bandit, trend inside the logarithm": (5.01, -1.08, 22, 0.709, 0.001, 4.98, 26, 0.916, 0.726),
}
TABLE_INPUTS = [OUT / "scores.csv"] + [paths.GRIDS / f for f in (
    "grid_final.csv", "grid_final_train_mae_all.csv", "grid_final_in_train_mae_all.csv", "selection_outside.csv")]


def printed(v):
    """A row rounded to the thesis's printed precision."""
    return (round(100 * v["test_te"], 2), round(100 * v["test_drag"], 2), round(100 * v["test_to"]),
            round(v["test_adf"], 3), round(v["test_za"], 3), round(100 * v["full_te"], 2),
            round(100 * v["full_to"]), round(v["full_adf"], 3), round(v["full_za"], 3))


def table():
    """The ten rows of tab:original: the seven runs of scores.csv, then from the final grids
    the first version's setting under the final specification (ALL_FIVE), the selected
    setting (final_bandit.selected_setting) and the trend-inside bandit with the lowest
    training MAE among those with training turnover below 150%. Written to
    results/robustness/tab_original.csv, with each row compared with the printed table."""
    missing = [str(f) for f in TABLE_INPUTS if not f.exists()]
    if missing:
        sys.exit(f"missing {missing}: run `python src/original_spec.py runs`, `python src/final_bandit.py final` "
                 "and `python src/training_mae.py` first")
    sc = pd.read_csv(OUT / "scores.csv")
    sc = sc[sc.model == "bandit"].set_index(["run", "period"])
    rows = []
    for label, run, ref in TABLE_RUNS:
        v = {f"{p}_{k}": float(sc.loc[(run, p), k]) for p in ("test", "full") for k in ("te", "drag", "to", "adf", "za")
             if f"{p}_{k}" in COLS}
        rows.append(dict(label=label, setting=TUNED_NAME, source=f"scores.csv {run}", reference=ref,
                         train_mae=np.nan, **v))
    g = pd.read_csv(paths.GRIDS / "grid_final.csv").set_index("name")
    g_mae = pd.read_csv(paths.GRIDS / "grid_final_train_mae_all.csv").set_index("name")["train_mae_all"]
    selected = selected_setting()
    gi = pd.read_csv(paths.GRIDS / "grid_final_in_train_mae_all.csv")
    inside = gi[(gi.family == "bandit") & (gi.train_to < st.TO_LINE) & gi.train_mae_all.notna()].sort_values(
        ["train_mae_all", "name"], kind="stable").iloc[0]
    for label, name, src, r, mae in (
            ("All five, delta by n/252", ALL_FIVE, "grid_final.csv", g.loc[ALL_FIVE], g_mae.loc[ALL_FIVE]),
            ("Tuned bandit (trend outside)", selected, "grid_final.csv", g.loc[selected], g_mae.loc[selected]),
            ("Bandit, trend inside the logarithm", inside["name"], "grid_final_in.csv", inside, inside["train_mae_all"])):
        rows.append(dict(label=label, setting=name, source=f"{src} {name}", reference="drifted",
                         train_mae=float(mae), **{c: float(r[c]) for c in COLS}))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "tab_original.csv", index=False)

    n_match = 0
    for r in tab.to_dict("records"):
        p = printed(r)
        same = p == PRINTED[r["label"]]
        n_match += same
        print(f"{r['label']:<38s} {p}  {'as printed' if same else 'printed ' + str(PRINTED[r['label']])}", flush=True)
    print(f"{n_match} of {len(tab)} rows match the printed table", flush=True)
    for r in tab[tab.train_mae.notna()].to_dict("records"):
        print(f"{r['label']}: {r['setting']}, training MAE {100 * r['train_mae']:.3f}%", flush=True)
    print(f"wrote {OUT / 'tab_original.csv'}", flush=True)
    return tab


# -----------------------------
# Two-break tests on the first version's spread (appendix text)
# -----------------------------
def first_version_spread(data):
    """Full-period log spread log(portfolio TRI) - log(peer-average TRI) of the first version,
    up to the last date of the two-break tests (two_break_unit_root.END)."""
    import two_break_unit_root as tb
    idx, _, W = run_bandit_v(data, TUNED, Fit(data))
    p_ret, p_tri, _, _ = rt.simulate(data, idx, W)
    bench_tri = (1 + data.B.reindex(p_ret.index).fillna(0.0)).cumprod()
    df = pd.concat([bench_tri.rename("B"), p_tri.rename("P")], axis=1).dropna().loc[:tb.END]
    return np.log(df["P"] / df["B"])


def breaks(data, boot=199):
    """Lumsdaine-Papell and Lee-Strazicich two-break tests on the first version's full-period
    spread, quoted in the text of Appendix "The Original Specification" beside
    tab:two_breaks. The settings are those of final_tests.breaks (AIC lag, break search,
    199-draw sieve wild bootstrap); the draws use seeds 7,000,000 + r, the seed block of the
    first row of tab:two_breaks. Writes results/robustness/two_break_first_version.csv."""
    from multiprocessing import Pool

    import two_break_unit_root as tb
    t0 = time.time()
    s = first_version_spread(data)
    y = s.to_numpy()
    T = len(y)
    k = tb.aic_lag(y)
    lp, lb1, lb2 = tb.lp_stat(y, k)
    ls, sb1, sb2 = tb.ls_stat(y, k)
    phi, resid = tb.null_ar(y, k)
    with Pool(N_JOBS) as pool:
        boots = np.array(pool.map(tb._boot, [(7_000_000 + r, phi, resid, T) for r in range(boot)]))
    row = dict(run="baseline", setting=TUNED_NAME, T=T, lags=k,
               lp_stat=lp, lp_p_boot=tb.pval(boots[:, 0], lp),
               lp_break1=s.index[lb1].date().isoformat(), lp_break2=s.index[lb2].date().isoformat(),
               ls_stat=ls, ls_p_boot=tb.pval(boots[:, 1], ls),
               ls_break1=s.index[sb1].date().isoformat(), ls_break2=s.index[sb2].date().isoformat())
    paths.ensure(OUT)
    pd.DataFrame([row]).to_csv(OUT / "two_break_first_version.csv", index=False)
    print(f"first version, full period (T = {T}, {k} lags): LP {lp:.2f} (bootstrap p {row['lp_p_boot']:.3f}), "
          f"LS {ls:.2f} (bootstrap p {row['ls_p_boot']:.3f}); {(time.time() - t0) / 60:.1f} min -> "
          f"{OUT / 'two_break_first_version.csv'}", flush=True)
    return row


def main(stage):
    if stage not in ("all", "check", "runs", "table", "breaks"):
        sys.exit("usage: python src/original_spec.py [all|check|runs|table|breaks]")
    if stage == "table":
        table()
        return
    t0 = time.time()
    data = rt.load()
    print(f"loaded in {(time.time() - t0) / 60:.1f} min; n = {data.n}", flush=True)
    if stage == "breaks":
        breaks(data)
        return
    if not check(data):
        sys.exit(1)
    if stage == "check":
        return
    runs(data)
    if stage == "all":
        if all(f.exists() for f in TABLE_INPUTS):
            table()
        else:
            print("table not built: it needs the final grids and the selection (`python src/final_bandit.py final`, "
                  "`python src/training_mae.py`); then run `python src/original_spec.py table`",
                  flush=True)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
