"""
Final runs: every high-equity model reported in the thesis, under the final specification,
written to results/final/ (timeseries/, turnover/, weights/, rl_diagnostics/, runs/ and
period_tables.csv).

The final specification (Chapter 3): the tracker fits
    log y_i = log(sum_j X_ij w_j) + alpha + beta * (i - 1) / 252
with the constant alpha and the trend beta outside the log and unpenalised (Equation 3.4);
the turnover penalty in the fit is delta * n / 252 for a window of n days and is measured
against the weights held just before the rebalance, the previous target weights drifted
with asset returns to the rebalance date (TO_REF = "drifted"); and the bandit is rewarded on
the forward mean absolute tracking error alone (final_bandit.SPECS["final"]). Inside the
cross-validation folds the penalty is measured against equal weight, so the drifted
weights enter only the final fit at the selected ridge strength.

Feeds every table built from results/final/period_tables.csv by thesis_tables.py
(tab:core_results, tab:core_results_all, tab:data_latency, tab:subperiod,
tab:window_comparison, tab:variants), the tests in final_tests.py (tab:dm, tab:dm_variants,
tab:dm_h1, tab:ci, tab:two_breaks) and the figures drawn from results/final/.

How each model is produced:
  0. Models the final specification does not change (equal weight, the return-space ridge,
     the log-TRI ridge without trend, the 60/40 and Reg28 passives): run_models.run_and_save
     with their original parameters. Equal weight and the return-space ridge are also run at
     5 and 10 bps a trade for the fee and transaction-cost figures.
  1. The selected bandit, the earlier bandit settings and the ten fixed windows (quarterly
     and monthly): taken from the paths stored by the final grid (final_bandit.py final).
  2. The bandit with a 5- or 21-day reporting lag and the context-free bandit:
     final_bandit.run_bandit_m with the selected settings.
  3. The window rules (20 random-window draws, the two look-ahead oracles, the volatility
     heuristic) and the two earlier bandits with a tracking reward.
  4. The monthly log-TRI ridge with trend and the turnover-penalised ridge (with its delta,
     rebalancing and window sensitivities); run_models' to_ref="drifted" measures the
     penalised ridge's final fit against the drifted weights.

Why steps 3 and 4 swap functions at run time. The window rules and the earlier bandits live
in run_models, window_baselines and rl_window_selection. They all call one inner tracker fit,
rl_window_selection.fit_drift_tri_elasticnet, which has the original specification (trend
inside the log, no delta scaling). Rather than copy their window-choice code, patch()
replaces that inner fit, inside this process only, with the final tracker
(final_bandit.FitB with the constant and trend outside the log, delta * n / 252, the drifted
penalty reference and the cached ridge strengths). Each rule still chooses its windows, and
each earlier bandit computes its reward, with its own unchanged code. In the same way the
monthly log-TRI ridges get copies of run_models' fit and cross-validation functions with the
constant and trend outside the log (_fit_date_trend, _fit_date_trend_to). The swap is made
only after steps 0 to 2 have run. `check` confirms that the swap itself changes nothing: with
the original tracker swapped in, the window rules' fit returns the same weights as their own
inner fit.

    python src/final_runs.py check   # self-checks, a few minutes; writes nothing but ridge caches
    python src/final_runs.py runs    # all runs, about 90 minutes with warm ridge caches

`runs` first deletes results/final/, so rerun every later step (sibling_runs, final_tests,
ridge_selection, gregory_hansen, thesis_tables, thesis_figures) afterwards.

Inputs: data/data.xlsx; results/grids/selection_outside.csv (the selected setting, read
through final_bandit.selected_config: the lowest training MAE among the trend-outside
settings with training turnover below 150% a year, from training_mae.py), grid_final.csv
and grid_final_paths.npz; the ridge caches in results/cache/ (built if missing).
"""
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("RIDGE_INNER_JOBS", "4")

import final_bandit as fb  # noqa: E402  (sets below-normal priority)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import bandit_tuning as bt  # noqa: E402
import paths  # noqa: E402
import period_tables as pt  # noqa: E402
import rl_window_selection as rws  # noqa: E402
import run_models as rm  # noqa: E402
import run_tuning as rt  # noqa: E402
import window_baselines as wb  # noqa: E402
from metrics import build_timeseries, save_timeseries, save_weights_to_csv, simulate_with_rebased_tri  # noqa: E402

SPEC = fb.SPECS["final"]
NREF = SPEC["n_ref"]
TO_REF = fb.TO_REF     # "drifted": the turnover penalty's reference in every final fit
WINDOWS = bt.WINDOWS
COST_BPS = (0, 5, 10)


# -----------------------------
# Writer for runs computed outside run_models
# -----------------------------
def save_run(data, name, idx, W, results_dir, chosen=None, tc=0.0):
    """Simulate the weights W set at decision rows idx and write timeseries, turnover and
    weights (and the bandit's chosen windows, if given) under results_dir, as
    run_models.run_and_save does."""
    results_dir = str(results_dir)
    weights = pd.DataFrame(W, index=data.A.index[idx], columns=data.A.columns)
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(data.A, weights, tc_bps=tc)
    p_ret, p_tri, eff_w, turn = rm.clip_to_common_start(p_ret, p_tri, eff_w, turn, data.A.index)
    save_timeseries(name, build_timeseries(p_ret, p_tri, data.B), results_dir)
    rm.save_turnover(name, turn, results_dir)
    save_weights_to_csv(name, eff_w, results_dir)
    if chosen is not None:
        d = paths.ensure(Path(results_dir) / "rl_diagnostics")
        pd.DataFrame({"datestamp": data.A.index[idx], "chosen_window": list(chosen)}).to_csv(
            d / f"{name}.csv", index=False)


def run_rm(run_name, model, params, results_dir=None):
    """run_models.run_and_save on the high-equity category, into results/final by default."""
    rm.run_and_save(run_name=run_name, model=model, index="ma", model_params=dict(params, _run_name=run_name),
                    results_dir=str(results_dir or paths.FINAL))


# -----------------------------
# The final tracker behind the window rules' interfaces
# -----------------------------
def fitb_interface(data, fit):
    """A drop-in for rl_window_selection.fit_drift_tri_elasticnet: finds the decision
    row from the window's last row and calls `fit` (a final_bandit.FitB) there.

    With fit.to_ref == 'drifted' the fit also needs prev_i, the decision at which w_prev
    was set. The callers (window_baselines, rl_window_selection, anchor_oracle) pass as
    w_prev the weights this interface returned at their previous decision, so it is
    found from a record of what was returned where: the latest decision before i that
    returned exactly w_prev. The record restarts when the decision row goes back (a new
    run). These callers run with no reporting lag, so the decision row is the fit end i;
    a w_prev that was never returned raises an error instead of being guessed."""
    end_of = {data.asset_tri[k].tobytes(): k for k in range(data.n)}
    made = {}          # weights bytes -> decision rows at which they were returned
    last = [-1]

    def f(asset_tri_window, bench_tri_window, ridge_grid=None, delta_turnover=0.0, w_prev=None, **_kw):
        i = end_of[np.ascontiguousarray(asset_tri_window[-1], dtype=float).tobytes()] + 1
        prev_i = None
        if fit.to_ref == "drifted":
            if i < last[0]:
                made.clear()
            last[0] = i
            if w_prev is not None:
                key = np.ascontiguousarray(w_prev, dtype=float).tobytes()
                cands = [k for k in made.get(key, ()) if k < i]
                if not cands:
                    raise RuntimeError(f"fitb_interface: w_prev at decision {i} was not returned at an earlier decision")
                prev_i = max(cands)
        w = fit(i, asset_tri_window.shape[0], float(delta_turnover), w_prev, prev_i=prev_i)
        if fit.to_ref == "drifted":
            made.setdefault(np.ascontiguousarray(w, dtype=float).tobytes(), []).append(i)
        return w, fit.beta
    return f


def _logfit_out(X, w, n):
    """Equation 3.4's fitted log index for w = (weights, beta, alpha)."""
    return fb._log_fit_out(X, w, n)


def _cv_trend_out(X_aug, y_tri, ridge, n_assets, k=3, delta=0.0):
    """run_models._cv_with_trend / _cv_with_trend_to with the constant and the trend outside
    the log. Inside the folds the penalty is measured against equal weight."""
    T = X_aug.shape[0]
    seg = T // (k + 1)
    if seg == 0:
        return np.inf
    bounds = [(0, 1)] * n_assets + [(None, None), (None, None)]
    cons = {"type": "eq", "fun": rm._sum_to_one_trend(n_assets)}
    w0 = np.concatenate([np.ones(n_assets) / n_assets, [0.0, 0.0]])
    w_prev = np.ones(n_assets) / n_assets
    val_losses = []
    for j in range(k):
        train_end = (j + 1) * seg
        val_end = min((j + 2) * seg, T)
        X_tr, y_tr = X_aug[:train_end], y_tri[:train_end]
        X_val, y_val = X_aug[train_end:val_end], y_tri[train_end:val_end]

        def train_loss(w, _ridge=ridge, _d=delta):
            f = _logfit_out(X_tr, w, n_assets)
            if f is None or np.any(y_tr <= 0):
                return np.inf
            loss = np.sum((f - np.log(y_tr)) ** 2) + _ridge * np.sum(w[:n_assets] ** 2)
            if _d:
                loss += _d * np.sum((w[:n_assets] - w_prev) ** 2)
            return loss

        res = minimize(train_loss, w0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            return np.inf
        f = _logfit_out(X_val, res.x, n_assets)
        if f is None or np.any(y_val <= 0):
            return np.inf
        val_losses.append(float(np.sum((f - np.log(y_val)) ** 2)))
    return float(np.mean(val_losses)) if val_losses else np.inf


def _fit_date_trend_out(X_aug, tri_b, n_assets, ridge_grid):
    """run_models._fit_date_trend with the constant and the trend outside the log."""
    bounds = [(0, 1)] * n_assets + [(None, None), (None, None)]
    cons = {"type": "eq", "fun": rm._sum_to_one_trend(n_assets)}
    w0 = np.concatenate([np.ones(n_assets) / n_assets, [0.0, 0.0]])
    best_cv, best_ridge = np.inf, None
    for ridge in ridge_grid:
        cv = _cv_trend_out(X_aug, tri_b, ridge, n_assets)
        if cv < best_cv:
            best_cv, best_ridge = cv, ridge
    if best_ridge is None:
        return None

    def full_loss(w):
        f = _logfit_out(X_aug, w, n_assets)
        if f is None or np.any(tri_b <= 0):
            return np.inf
        return np.sum((f - np.log(tri_b)) ** 2) + best_ridge * np.sum(w[:n_assets] ** 2)
    res = minimize(full_loss, w0, bounds=bounds, constraints=cons, method="SLSQP")
    return res.x if res.success else None


def _fit_date_trend_to_out(X_aug, tri_b, n_assets, ridge_grid, delta, w_prev):
    """run_models._fit_date_trend_to with the constant and the trend outside the log and
    delta * n / 252. w_prev is the reference run_models passes: with to_ref="drifted" the
    previous target weights drifted to the rebalance date."""
    delta = delta * X_aug.shape[0] / NREF   # delta per year of data
    bounds = [(0, 1)] * n_assets + [(None, None), (None, None)]
    cons = {"type": "eq", "fun": rm._sum_to_one_trend(n_assets)}
    w0 = np.concatenate([np.ones(n_assets) / n_assets, [0.0, 0.0]])
    best_cv, best_ridge = np.inf, None
    for ridge in ridge_grid:
        cv = _cv_trend_out(X_aug, tri_b, ridge, n_assets, delta=delta)
        if cv < best_cv:
            best_cv, best_ridge = cv, ridge
    if best_ridge is None:
        return None

    def full_loss(w):
        f = _logfit_out(X_aug, w, n_assets)
        if f is None or np.any(tri_b <= 0):
            return np.inf
        return (np.sum((f - np.log(tri_b)) ** 2) + best_ridge * np.sum(w[:n_assets] ** 2)
                + delta * np.sum((w[:n_assets] - w_prev) ** 2))
    res = minimize(full_loss, w0, bounds=bounds, constraints=cons, method="SLSQP")
    return res.x if res.success else None


def patch(data, fit):
    """Swap the final tracker into window_baselines and rl_window_selection, and the fits
    with the constant and trend outside the log into run_models (see the module docstring)."""
    f = fitb_interface(data, fit)
    wb.fit_drift_tri_elasticnet = f
    rws.fit_drift_tri_elasticnet = f
    rm._fit_date_trend = _fit_date_trend_out
    rm._fit_date_trend_to = _fit_date_trend_to_out


# -----------------------------
# Self-checks
# -----------------------------
def check_swap(data, n_decisions=4):
    """Largest weight gap between rl_window_selection.fit_drift_tri_elasticnet (the inner fit
    of the window rules, original specification) and the same specification put in through
    fitb_interface (final_bandit.FitB(data, to_ref="target"): trend inside the log, no delta
    scaling, penalty against the previous target weights, cached ridge strengths). Compared
    at n_decisions quarterly decisions spread over the sample, for every window, with delta 1
    and the previous decision's weights as w_prev, exactly as the window rules call the fit."""
    asset_tri = (1 + data.A).cumprod().values
    bench_tri = (1 + data.B).cumprod().values
    rows = list(wb._rebalance_indices(data.n, max(WINDOWS), 63, 63))
    rows = [rows[k] for k in np.linspace(0, len(rows) - 1, n_decisions).astype(int)]
    swapped = fitb_interface(data, fb.FitB(data, to_ref="target"))
    gap, prev = 0.0, None
    for i in rows:
        for window in WINDOWS:
            args = (asset_tri[i - window:i], bench_tri[i - window:i])
            w_own, _ = rws.fit_drift_tri_elasticnet(*args, delta_turnover=1.0, w_prev=prev)
            w_swap, _ = swapped(*args, delta_turnover=1.0, w_prev=prev)
            if w_own is None:
                return np.inf
            gap = max(gap, float(np.abs(np.asarray(w_own) - np.asarray(w_swap)).max()))
        prev = w_own
    return gap


def check(data):
    """1. The swap: the original tracker put in through fitb_interface must give the same
    weights as the window rules' own inner fit, to 1e-9 (check_swap).
    2. final_bandit.run_bandit_m with the original switches (and the penalty against the
    previous target weights) must equal bandit_tuning.run_bandit exactly, with a 5-day
    reporting lag and with the context-free features."""
    ok = True
    g = check_swap(data)
    ok &= g < 1e-9
    print(f"CHECK swapped-in original tracker vs the window rules' own fit: max weight gap {g:.3g}", flush=True)
    for extra in (dict(lag=5), dict(features=[0])):
        cfg = dict(fb.TUNED, **extra)
        lag = int(extra.get("lag", 0))
        if lag:
            bt.build_ridge_cache(data, [i - lag for i in bt.schedule(data.n, 63)], n_jobs=fb.N_JOBS)
        i0, c0, W0 = bt.run_bandit(data, cfg, fixes=True)
        i1, c1, W1 = fb.run_bandit_m(data, cfg, fb.FitB(data, to_ref="target"))
        same = list(i0) == list(i1) and list(c0) == list(c1) and np.array_equal(W0, W1)
        ok &= same
        print(f"CHECK final_bandit.run_bandit_m == bandit_tuning.run_bandit with {extra}: {same}", flush=True)
    print("CHECK", "PASSED" if ok else "FAILED", flush=True)
    return ok


# -----------------------------
# The runs
# -----------------------------
def unchanged_specs():
    """(run, model, params) of the models the final specification does not change, with
    the parameters described in Chapter 3 of the thesis."""
    specs = []
    for tc in COST_BPS:
        specs.append((f"equal_weight__ma__reb21__tc{tc}", "equal_weight", dict(rebalance_freq=21, tc_bps=tc)))
        specs.append((f"ridge_returns__ma__w252__reb21__tc{tc}", "ridge_returns",
                      dict(window=252, rebalance_freq=21, tc_bps=tc)))
    specs.append(("ridge_no_trend__ma__w252__reb21__tc0", "ridge_no_trend",
                  dict(window=252, rebalance_freq=21, tc_bps=0)))
    for tag in ("60_40", "reg28"):
        specs.append((f"passive_{tag}__ma__reb21__tc0", "passive_static",
                      dict(target_weights=rm.PASSIVE_TARGETS[tag], rebalance_freq=21, tc_bps=0)))
    return specs


def unchanged_runs():
    for run, model, params in unchanged_specs():
        run_rm(run, model, params)


def grid_runs(data, selected):
    """Step 1: the selected bandit, the earlier bandit settings and the fixed windows,
    from the paths stored by the final grid."""
    z = np.load(paths.GRIDS / "grid_final_paths.npz")
    names = {selected: "rl_tuned__ma__reb63__tc0",
             "bandit_k1.2_e0_g1_d1_reb21_ce504": "rl_tuned__ma__prior_reb21__tc0"}
    for w in WINDOWS:
        names[f"fixed_w{w}_d1_reb63"] = f"window_fixedbest__ma__w{w}__reb63__tc0"
        names[f"fixed_w{w}_d1_reb21"] = f"window_fixedbest__ma__w{w}__tc0"
    for name, run in names.items():
        chosen = z[f"{name}__win"] if name.startswith("bandit") else None
        save_run(data, run, z[f"{name}__idx"], z[f"{name}__W"], paths.FINAL, chosen=chosen)
    print(f"grid runs saved ({len(names)}), bandit {selected}", flush=True)


def lag_and_context_runs(data, ro, cfg):
    """Step 2: the selected bandit with a reporting lag, and without its context features.
    With a lag the shifted decision row can fall short of the longest windows at the first
    decisions; the ridge cache stores None there (final_bandit._ridge_out_one), as the
    original-specification cache (bandit_tuning.build_ridge_cache) does. With a lag the
    drifted reference stops at the last close the fit sees (final_bandit.drifted_reference)."""
    for lag in (5, 21):
        ro.update(fb.build_ridge_cache_out(data, [i - lag for i in bt.schedule(data.n, 63)]))
        fit = fb.FitB(data, trend_out=True, n_ref=NREF, ridge_out=ro, to_ref=TO_REF)
        idx, ch, W = fb.run_bandit_m(data, dict(cfg, lag=lag), fit, drift=True, reward="mae")
        save_run(data, f"rl_tuned__ma__reb63__lag{lag}__tc0", idx, W, paths.FINAL, chosen=ch)
        print(f"lag {lag}: done ({fit.fallbacks} fallbacks)", flush=True)
    fit = fb.FitB(data, trend_out=True, n_ref=NREF, ridge_out=ro, to_ref=TO_REF)
    idx, ch, W = fb.run_bandit_m(data, dict(cfg, features=[0]), fit, drift=True, reward="mae")
    save_run(data, "rl_tuned_ctxfree__ma__reb63__tc0", idx, W, paths.FINAL, chosen=ch)
    print("context-free: done", flush=True)


def window_rule_specs():
    """(run, model, params) of step 3, run through run_models with the final tracker swapped in."""
    base_m = dict(window_choices=WINDOWS, rebalance_freq=21, forward_days=63, delta_turnover=1.0, tc_bps=0)
    base_q = dict(base_m, rebalance_freq=63)
    specs = [(f"window_random__ma__seed{seed}__reb63__tc0", "window_random", dict(seed=seed, **base_q))
             for seed in range(20)]
    specs += [("window_oracle__ma__reb63__tc0", "window_oracle", dict(base_q)),
              ("window_oracle__ma__tc0", "window_oracle", dict(base_m)),
              ("window_volheuristic__ma__tc0", "window_volheuristic", dict(vol_lookback=63, **base_m)),
              ("window_volheuristic__ma__reb63__tc0", "window_volheuristic", dict(vol_lookback=63, **base_q))]
    return specs


def earlier_bandit_specs():
    """(run, model, params) of the two earlier bandits with a tracking reward (step 3)."""
    return [(run, "rl_bandit", dict(window_choices=WINDOWS, rebalance_freq=21, forward_days=63, alpha_ucb=1.2,
                                    ridge=1.0, eta_turnover=0.25, lambda_coint=0.05, delta_turnover=d, tc_bps=0))
            for run, d in (("rl_bandit_to__ma__tc0", 1.0), ("rl_bandit__ma__tc0", 0.0))]


def window_rule_runs(data, fit):
    """Step 3, with patch(data, fit) already applied."""
    for run, model, params in window_rule_specs():
        run_rm(run, model, params)
    anchor_oracle(data)
    for run, model, params in earlier_bandit_specs():
        run_rm(run, model, params)
    print(f"window rules and earlier bandits: done, {fit.fallbacks} fallbacks, {fit.stuck} stuck", flush=True)


def ridge_trend_specs():
    """(run, model, params) of step 4: the monthly log-TRI ridges with trend. The
    turnover-penalised runs measure the final fit's penalty against the drifted weights
    (run_models' to_ref)."""
    specs = [("ridge_trend__ma__w252__reb21__tc0", "ridge_trend", dict(window=252, rebalance_freq=21, tc_bps=0))]
    to = dict(tc_bps=0, to_ref=TO_REF)
    for tag, d in (("d1", 1.0), ("d0p01", 0.01), ("d0p1", 0.1), ("d0p5", 0.5), ("d2p0", 2.0)):
        specs.append((f"ridge_trend_to__ma__w252__reb21__{tag}__tc0", "ridge_trend_to",
                      dict(window=252, rebalance_freq=21, delta_turnover=d, **to)))
    for reb in (5, 63):
        specs.append((f"ridge_trend_to__ma__w252__reb{reb}__d1__tc0", "ridge_trend_to",
                      dict(window=252, rebalance_freq=reb, delta_turnover=1.0, **to)))
    for w in (126, 504):
        specs.append((f"ridge_trend_to__ma__w{w}__reb21__d1__tc0", "ridge_trend_to",
                      dict(window=w, rebalance_freq=21, delta_turnover=1.0, **to)))
    return specs


def ridge_trend_runs():
    """Step 4, with patch() already applied."""
    for run, model, params in ridge_trend_specs():
        t1 = time.time()
        run_rm(run, model, params)
        print(f"{run}: {(time.time() - t1) / 60:.1f} min", flush=True)


def anchor_oracle(data):
    """The look-ahead oracle (drift): at each monthly decision, the window whose weights give
    the smallest squared cumulative active return over the next 63 days. Uses the swapped-in
    tracker, so call it after patch()."""
    A, B = data.A, data.B
    n, FD, DELTA, REB = data.n, 63, 1.0, 21
    A_tri = (1 + A).cumprod()
    B_tri = (1 + B).cumprod()

    def fwd_anchor_loss(i, w):
        end = min(i + FD, n)
        act = (A.iloc[i:end].values @ w) - B.iloc[i:end].values
        return float(np.sum(act) ** 2)

    dates, weights, prev, chosen = [], [], None, []
    for i in wb._rebalance_indices(n, max(WINDOWS), FD, REB):
        best = None
        for window in WINDOWS:
            w, _ = wb.fit_drift_tri_elasticnet(A_tri.iloc[i - window:i].values, B_tri.iloc[i - window:i].values,
                                               delta_turnover=DELTA, w_prev=prev)
            if w is None:
                continue
            al = fwd_anchor_loss(i, w)
            if best is None or al < best[0]:
                best = (al, window, w)
        dates.append(i)
        weights.append(best[2])
        chosen.append(best[1])
        prev = best[2]
    save_run(data, "window_oracle_anchor__ma__tc0", np.array(dates), np.array(weights), paths.FINAL, chosen=chosen)


def period_table():
    """Step 5: period metrics of every zero-cost run in results/final (the 5 and 10 bps runs
    are left out; the figures that use them score them themselves)."""
    runs_all = sorted(f[:-4] for f in os.listdir(paths.FINAL / "timeseries") if f.endswith("__tc0.csv"))
    ptab = pt.build(runs_all, results_dir=str(paths.FINAL))
    ptab.to_csv(paths.FINAL / "period_tables.csv", index=False)
    return runs_all


def runs(data):
    t0 = time.time()
    shutil.rmtree(paths.FINAL, ignore_errors=True)
    for sub in ("timeseries", "turnover", "weights", "rl_diagnostics"):
        paths.ensure(paths.FINAL / sub)

    unchanged_runs()                       # 0. before any function is swapped
    print(f"unchanged models: done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    ro = fb.build_ridge_cache_out(data, bt.schedule(data.n, 21))
    selected, cfg = fb.selected_config()
    grid_runs(data, selected)              # 1
    lag_and_context_runs(data, ro, cfg)    # 2

    fit = fb.FitB(data, trend_out=True, n_ref=NREF, ridge_out=ro, to_ref=TO_REF)
    patch(data, fit)
    window_rule_runs(data, fit)            # 3
    ridge_trend_runs()                     # 4

    runs_all = period_table()              # 5
    print(f"period tables: {len(runs_all)} runs; DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


def main(stage):
    if stage not in ("check", "runs"):
        sys.exit("usage: python src/final_runs.py check|runs")
    data = rt.load()
    if stage == "check":
        sys.exit(0 if check(data) else 1)
    runs(data)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "check")
