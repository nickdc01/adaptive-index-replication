"""
Model runners: load the data, fit a tracking model through time, simulate the
portfolio and save its results.

This is a library used by final_runs.py and sibling_runs.py (and by several
figure scripts for load_data). It is not run on its own. Example:

    from run_models import run_and_save
    import paths
    run_and_save("equal_weight__ma__reb21__tc0", "equal_weight", "ma",
                 dict(rebalance_freq=21, tc_bps=0), paths.FINAL)

Input: data/data.xlsx (paths.DATA), sheets X (asset total return indices) and
Y, Y_le, Y_me (daily returns of the high-, low- and medium-equity peer averages).

Output of run_and_save, all under results_dir (an absolute path):
    runs/<run>.csv            parameters and full-sample metrics
    timeseries/<run>.csv      benchmark and portfolio TRI, rolling 1-year TE
    weights/<run>.csv         daily effective (drifting) weights
    turnover/<run>.csv        one-way turnover at each rebalance
    rl_diagnostics/<run>.csv  chosen window at each rebalance (window-choosing models only)

Models (keys of MODELS):
    equal_weight     equal weight in every asset class
    passive_static   fixed target weights (the 60/40 and Regulation 28 allocations)
    ridge_returns    return-space ridge: non-negative ridge on daily returns
    ridge_no_trend   log-TRI ridge without trend
    ridge_trend      log-TRI ridge with trend
    ridge_trend_to   turnover-penalised ridge (log-TRI ridge with trend and a
                     turnover penalty, measured against the previous target
                     weights or, with to_ref="drifted", against those weights
                     drifted to the rebalance date)
    rl_bandit        LinUCB window-choosing bandit with a tracking reward
                     (rl_window_selection.py)
    window_random, window_oracle, window_volheuristic
                     window-choosing controls (window_baselines.py)

The log-TRI ridges with trend here keep the trend inside the log, with no
constant. final_runs.py swaps in the final specification's fits (trend and
constant outside the log, delta scaled by n / 252) for _fit_date_trend and
_fit_date_trend_to, which the runners look up at call time.

Why the ridge models are non-negative ridge: the weights are long-only and sum
to one, so sum_j |w_j| = 1 for every feasible portfolio. An elastic-net L1 term
is then a constant and has no effect on the solution, which leaves a single
ridge coefficient r = lambda * (1 - alpha) / 2 (methodology chapter). The ridge
grid is the set of those coefficients over lambda in logspace(-5, 2, 20) and
alpha in {0, 0.25, 0.5, 0.75, 1}: 81 values from 0 to 50.
"""

import os

# Pin BLAS threads before numpy is imported, so SLSQP results are deterministic.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import random  # noqa: E402
import warnings  # noqa: E402

import cvxpy as cp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from tqdm import tqdm  # noqa: E402

import paths  # noqa: E402
from metrics import (  # noqa: E402
    build_timeseries,
    compute_metrics,
    save_run,
    save_timeseries,
    save_weights_to_csv,
    simulate_with_rebased_tri,
    solver_failures,
)
from rl_window_selection import _default_ridge_grid  # noqa: E402

DEFAULT_WINDOWS = [126, 252, 378, 504, 756]


def _resolve_ridge_grid(params: dict):
    """The ridge grid: params['ridge_grid'] if given, else the default 81-point grid."""
    if params.get("ridge_grid") is not None:
        return np.asarray(params["ridge_grid"], dtype=float)
    return _default_ridge_grid()


# Number of rebalance rows dropped (failed fits) by the latest _warn_dropped
# call, so run_and_save can log it with the metrics.
_LAST_N_DROPPED = {"n": 0}


def _warn_dropped(weights_df, model_tag):
    """Drop rebalance rows with a failed (NaN) fit and warn how many were dropped."""
    cleaned = weights_df.dropna()
    n_dropped = len(weights_df) - len(cleaned)
    _LAST_N_DROPPED["n"] = n_dropped
    if n_dropped > 0:
        warnings.warn(
            f"[{model_tag}] dropped {n_dropped} of {len(weights_df)} rebalance "
            f"date(s) with failed/NaN weights before simulation.",
            stacklevel=2,
        )
    return cleaned


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

_BENCH = "__benchmark__"
_BENCH_SHEETS = {"ma": "Y", "ma_le": "Y_le", "ma_me": "Y_me"}


def load_data(index: str = "ma"):
    """
    Daily asset returns and peer-average (benchmark) returns.

    index: "ma" (high-equity peer average, the main study) or the sibling
    categories "ma_le" / "ma_me" (low- and medium-equity peer averages; same
    11 asset classes).

    Returns (asset_returns DataFrame, bench_returns Series named "value").
    """
    if index not in _BENCH_SHEETS:
        raise ValueError(f"index must be one of {list(_BENCH_SHEETS)}, got {index!r}")
    if not paths.DATA.exists():
        raise FileNotFoundError(f"{paths.DATA} not found; see data/README.md for the expected file.")
    X = pd.read_excel(paths.DATA, sheet_name="X")
    Y = pd.read_excel(paths.DATA, sheet_name=_BENCH_SHEETS[index])
    X["datestamp"] = pd.to_datetime(X["datestamp"])
    Y["datestamp"] = pd.to_datetime(Y["datestamp"])
    levels = X.set_index("datestamp").sort_index()
    bench_tri = (1 + Y.set_index("datestamp")["value"].sort_index()).cumprod()
    # Align levels on the asset calendar before differencing. The peer series is
    # priced on some days the asset sheet lacks (mostly US/UK holidays);
    # reindexing the peer TRI onto the asset dates folds the return of each such
    # day into the next shared date instead of discarding it.
    levels[_BENCH] = bench_tri.reindex(levels.index).ffill()
    rets = levels.dropna().pct_change(fill_method=None).iloc[1:]
    return rets.drop(columns=_BENCH), rets[_BENCH].rename("value")


# ─────────────────────────────────────────────────────────────
# Per-date fits (top-level functions so joblib can pickle them)
# ─────────────────────────────────────────────────────────────

def _sum_to_one(w):
    return np.sum(w) - 1


def _sum_to_one_trend(n_assets):
    def _f(w):
        return np.sum(w[:n_assets]) - 1
    return _f


def _fit_date_returns(ra, rb, ridge_grid):
    """
    Return-space ridge at one rebalance date (CVXPY with OSQP).
    Pass 1: walk-forward CV over the ridge grid. Pass 2: one full fit at the
    selected ridge coefficient.
    """
    n = ra.shape[1]

    best_cv, best_ridge = np.inf, None
    for ridge in ridge_grid:
        cv = _cv_returns(ra, rb, ridge)
        if cv < best_cv:
            best_cv, best_ridge = cv, ridge

    if best_ridge is None:
        return None

    w_var = cp.Variable(n, nonneg=True)
    obj = (cp.sum_squares(ra @ w_var - rb)
           + best_ridge * cp.sum_squares(w_var))
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(w_var) == 1, w_var <= 1])
    try:
        prob.solve(solver=cp.OSQP, warm_start=True,
                   eps_abs=1e-6, eps_rel=1e-6, verbose=False)
    except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
        solver_failures[f"osqp_{type(e).__name__}"] += 1
        return None
    return w_var.value if w_var.value is not None else None


def _fit_date_no_trend(tri_a, tri_b, ridge_grid):
    """Log-TRI ridge without trend at one rebalance date (CV pass, then one full SLSQP fit)."""
    n = tri_a.shape[1]
    bounds = [(0, 1)] * n
    cons   = {"type": "eq", "fun": _sum_to_one}

    best_cv, best_ridge = np.inf, None
    for ridge in ridge_grid:
        cv = _cv_no_trend(tri_a, tri_b, ridge, bounds, cons)
        if cv < best_cv:
            best_cv, best_ridge = cv, ridge

    if best_ridge is None:
        return None
    def full_loss(w):
        p = tri_a @ w
        if np.any(p <= 0) or np.any(tri_b <= 0):
            return np.inf
        mse = np.sum((np.log(p) - np.log(tri_b)) ** 2)
        return mse + best_ridge * np.sum(w ** 2)
    res = minimize(full_loss, np.ones(n) / n, bounds=bounds, constraints=cons, method="SLSQP")
    return res.x if res.success else None


def _fit_date_trend(X_aug, tri_b, n_assets, ridge_grid):
    """Log-TRI ridge with trend at one rebalance date (CV pass, then one full SLSQP fit)."""
    bounds = [(0, 1)] * n_assets + [(None, None)]
    cons   = {"type": "eq", "fun": _sum_to_one_trend(n_assets)}
    w0     = np.concatenate([np.ones(n_assets) / n_assets, [0.0]])

    best_cv, best_ridge = np.inf, None
    for ridge in ridge_grid:
        cv = _cv_with_trend(X_aug, tri_b, ridge, n_assets)
        if cv < best_cv:
            best_cv, best_ridge = cv, ridge

    if best_ridge is None:
        return None
    def full_loss(w):
        p = X_aug @ w
        if np.any(p <= 0) or np.any(tri_b <= 0):
            return np.inf
        mse = np.sum((np.log(p) - np.log(tri_b)) ** 2)
        return mse + best_ridge * np.sum(w[:n_assets] ** 2)
    res = minimize(full_loss, w0, bounds=bounds, constraints=cons, method="SLSQP")
    return res.x if res.success else None


# ─────────────────────────────────────────────────────────────
# Return-space ridge
# ─────────────────────────────────────────────────────────────

def _cv_returns(ret_a, ret_b, ridge, k=3):
    """
    Walk-forward CV loss of the return-space ridge (CVXPY with OSQP).
    The window is cut into k + 1 equal segments; fold j trains on segments
    0..j and validates on segment j + 1.
    """
    T, n = ret_a.shape
    seg = T // (k + 1)
    if seg == 0:
        return np.inf
    val_losses = []

    for j in range(k):
        train_end = (j + 1) * seg
        val_end   = min((j + 2) * seg, T)
        X_tr, y_tr   = ret_a[:train_end], ret_b[:train_end]
        X_val, y_val = ret_a[train_end:val_end], ret_b[train_end:val_end]

        w_var = cp.Variable(n, nonneg=True)
        obj = (cp.sum_squares(X_tr @ w_var - y_tr)
               + ridge * cp.sum_squares(w_var))
        prob = cp.Problem(cp.Minimize(obj), [cp.sum(w_var) == 1, w_var <= 1])
        try:
            prob.solve(solver=cp.OSQP, warm_start=True,
                       eps_abs=1e-5, eps_rel=1e-5, verbose=False)
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
            solver_failures[f"osqp_cv_{type(e).__name__}"] += 1
            return np.inf
        if w_var.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            return np.inf
        w_sol = w_var.value

        val_losses.append(float(np.sum((X_val @ w_sol - y_val) ** 2)))

    return float(np.mean(val_losses)) if val_losses else np.inf


def run_ridge_returns(asset_returns, bench_returns, params):
    """
    Return-space ridge, refitted on a rolling window.

    params: window (default 252), rebalance_freq (21), ridge_grid (default grid),
    tc_bps (0), data_lag (0: weights apply data_lag days after the fit).
    """
    window = params.get("window", 252)
    reb    = params.get("rebalance_freq", 21)
    ridge_grid = _resolve_ridge_grid(params)
    tc_bps     = params.get("tc_bps", 0.0)
    data_lag   = int(params.get("data_lag", 0))

    ret_a = asset_returns.values
    ret_b = bench_returns.values
    n = ret_a.shape[1]

    indices = list(range(window, len(asset_returns), reb))
    rebalance_dates = [asset_returns.index[min(i + data_lag, len(asset_returns) - 1)]
                       for i in indices]

    results = Parallel(n_jobs=int(os.environ.get("RIDGE_INNER_JOBS", "4")), backend="loky")(
        delayed(_fit_date_returns)(
            ret_a[i - window:i], ret_b[i - window:i], ridge_grid
        )
        for i in tqdm(indices, desc="Ridge (returns)")
    )

    tracking_weights = [w if w is not None else [np.nan] * n for w in results]
    weights_df = pd.DataFrame(tracking_weights, index=rebalance_dates, columns=asset_returns.columns)
    weights_clean = _warn_dropped(weights_df, "ridge_returns")
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(
        asset_returns, weights_clean, tc_bps=tc_bps
    )
    return p_ret, p_tri, eff_w, turn, None


# ─────────────────────────────────────────────────────────────
# Log-TRI ridge without trend
# ─────────────────────────────────────────────────────────────

def _cv_no_trend(tri_a, tri_b, ridge, bounds, cons, k=3):
    """Walk-forward CV loss of the log-TRI ridge without trend (folds as _cv_returns)."""
    T = tri_a.shape[0]
    seg = T // (k + 1)
    if seg == 0:
        return np.inf
    val_losses = []

    for j in range(k):
        train_end = (j + 1) * seg
        val_end   = min((j + 2) * seg, T)
        X_tr, y_tr   = tri_a[:train_end], tri_b[:train_end]
        X_val, y_val = tri_a[train_end:val_end], tri_b[train_end:val_end]

        def train_loss(w, _ridge=ridge):
            p = X_tr @ w
            if np.any(p <= 0) or np.any(y_tr <= 0):
                return np.inf
            mse = np.sum((np.log(p) - np.log(y_tr)) ** 2)
            return mse + _ridge * np.sum(w ** 2)

        res = minimize(train_loss, np.ones(X_tr.shape[1]) / X_tr.shape[1],
                       bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            return np.inf
        p_val = X_val @ res.x
        if np.any(p_val <= 0) or np.any(y_val <= 0):
            return np.inf
        val_losses.append(float(np.sum((np.log(p_val) - np.log(y_val)) ** 2)))

    return float(np.mean(val_losses)) if val_losses else np.inf


def run_ridge_no_trend(asset_returns, bench_returns, params):
    """
    Log-TRI ridge without trend, refitted on a rolling window. The TRIs are
    rebased to 100 at the start of each window.

    params: window (default 252), rebalance_freq (63), ridge_grid, tc_bps (0), data_lag (0).
    """
    window = params.get("window", 252)
    reb = params.get("rebalance_freq", 63)
    ridge_grid = _resolve_ridge_grid(params)
    tc_bps = params.get("tc_bps", 0.0)
    data_lag = int(params.get("data_lag", 0))

    asset_tri = (1 + asset_returns).cumprod()
    bench_tri = (1 + bench_returns).cumprod()
    n = asset_returns.shape[1]

    def _prep_no_trend(i):
        ta = asset_tri.iloc[i - window:i].values
        tb = bench_tri.iloc[i - window:i].values
        ta = ta / ta[0] * 100
        tb = tb / tb[0] * 100
        return ta, tb

    indices = list(range(window, len(asset_tri), reb))
    rebalance_dates = [asset_tri.index[min(i + data_lag, len(asset_tri) - 1)]
                       for i in indices]

    results = Parallel(n_jobs=int(os.environ.get("RIDGE_INNER_JOBS", "4")), backend="loky")(
        delayed(_fit_date_no_trend)(*_prep_no_trend(i), ridge_grid)
        for i in tqdm(indices, desc="Ridge (no trend)")
    )

    tracking_weights = [w if w is not None else [np.nan] * n for w in results]
    weights_df = pd.DataFrame(tracking_weights, index=rebalance_dates, columns=asset_returns.columns)
    weights_clean = _warn_dropped(weights_df, "ridge_no_trend")
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(
        asset_returns, weights_clean, tc_bps=tc_bps
    )
    return p_ret, p_tri, eff_w, turn, None


# ─────────────────────────────────────────────────────────────
# Log-TRI ridge with trend, and the turnover-penalised ridge
# ─────────────────────────────────────────────────────────────

def _cv_with_trend_to(X_aug, y_tri, ridge, delta, n_assets, k=3):
    """
    Walk-forward CV loss of the turnover-penalised ridge. Inside the folds the
    turnover penalty is measured against equal weight, so the CV does not
    depend on the previous rebalance.

    Loss = log-TRI SSE + ridge * ||w||^2 + delta * ||w - w_equal||^2
    """
    T = X_aug.shape[0]
    seg = T // (k + 1)
    if seg == 0:
        return np.inf

    bounds = [(0, 1)] * n_assets + [(None, None)]
    cons   = {"type": "eq", "fun": _sum_to_one_trend(n_assets)}
    w0     = np.concatenate([np.ones(n_assets) / n_assets, [0.0]])
    w_prev = np.ones(n_assets) / n_assets   # equal-weight CV reference
    val_losses = []

    for j in range(k):
        train_end = (j + 1) * seg
        val_end   = min((j + 2) * seg, T)
        X_tr, y_tr   = X_aug[:train_end], y_tri[:train_end]
        X_val, y_val = X_aug[train_end:val_end], y_tri[train_end:val_end]

        def train_loss(w, _ridge=ridge, _d=delta):
            p = X_tr @ w
            if np.any(p <= 0) or np.any(y_tr <= 0):
                return np.inf
            mse   = np.sum((np.log(p) - np.log(y_tr)) ** 2)
            ridge_pen = _ridge * np.sum(w[:n_assets] ** 2)
            to_pen = _d * np.sum((w[:n_assets] - w_prev) ** 2)
            return mse + ridge_pen + to_pen

        res = minimize(train_loss, w0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            return np.inf
        p_val = X_val @ res.x
        if np.any(p_val <= 0) or np.any(y_val <= 0):
            return np.inf
        val_losses.append(float(np.sum((np.log(p_val) - np.log(y_val)) ** 2)))

    return float(np.mean(val_losses)) if val_losses else np.inf


def _fit_date_trend_to(X_aug, tri_b, n_assets, ridge_grid, delta, w_prev):
    """
    Turnover-penalised ridge at one rebalance date. The CV pass uses equal
    weight as the turnover reference; the full fit uses w_prev, the weights of
    the previous rebalance. delta is fixed, not cross-validated.
    """
    bounds = [(0, 1)] * n_assets + [(None, None)]
    cons   = {"type": "eq", "fun": _sum_to_one_trend(n_assets)}
    w0     = np.concatenate([np.ones(n_assets) / n_assets, [0.0]])

    best_cv, best_ridge = np.inf, None
    for ridge in ridge_grid:
        cv = _cv_with_trend_to(X_aug, tri_b, ridge, delta, n_assets)
        if cv < best_cv:
            best_cv, best_ridge = cv, ridge

    if best_ridge is None:
        return None

    def full_loss(w):
        p = X_aug @ w
        if np.any(p <= 0) or np.any(tri_b <= 0):
            return np.inf
        mse       = np.sum((np.log(p) - np.log(tri_b)) ** 2)
        ridge_pen = best_ridge * np.sum(w[:n_assets] ** 2)
        to_pen    = delta * np.sum((w[:n_assets] - w_prev) ** 2)
        return mse + ridge_pen + to_pen

    res = minimize(full_loss, w0, bounds=bounds, constraints=cons, method="SLSQP")
    return res.x if res.success else None


def _cv_with_trend(X_aug, y_tri, ridge, n_assets, k=3):
    """Walk-forward CV loss of the log-TRI ridge with trend (folds as _cv_returns)."""
    T = X_aug.shape[0]
    seg = T // (k + 1)
    if seg == 0:
        return np.inf

    bounds = [(0, 1)] * n_assets + [(None, None)]
    cons   = {"type": "eq", "fun": _sum_to_one_trend(n_assets)}
    w0     = np.concatenate([np.ones(n_assets) / n_assets, [0.0]])
    val_losses = []

    for j in range(k):
        train_end = (j + 1) * seg
        val_end   = min((j + 2) * seg, T)
        X_tr, y_tr   = X_aug[:train_end], y_tri[:train_end]
        X_val, y_val = X_aug[train_end:val_end], y_tri[train_end:val_end]

        def train_loss(w, _ridge=ridge):
            p = X_tr @ w
            if np.any(p <= 0) or np.any(y_tr <= 0):
                return np.inf
            mse = np.sum((np.log(p) - np.log(y_tr)) ** 2)
            return mse + _ridge * np.sum(w[:n_assets] ** 2)

        res = minimize(train_loss, w0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            return np.inf
        p_val = X_val @ res.x
        if np.any(p_val <= 0) or np.any(y_val <= 0):
            return np.inf
        val_losses.append(float(np.sum((np.log(p_val) - np.log(y_val)) ** 2)))

    return float(np.mean(val_losses)) if val_losses else np.inf


def run_ridge_trend(asset_returns, bench_returns, params):
    """
    Log-TRI ridge with a linear time trend, refitted on a rolling window.

    params: window (default 252), rebalance_freq (21), ridge_grid, tc_bps (0), data_lag (0).
    """
    window = params.get("window", 252)
    reb = params.get("rebalance_freq", 21)
    ridge_grid = _resolve_ridge_grid(params)
    tc_bps = params.get("tc_bps", 0.0)
    data_lag = int(params.get("data_lag", 0))

    asset_tri = (1 + asset_returns).cumprod()
    bench_tri = (1 + bench_returns).cumprod()
    n_assets = asset_returns.shape[1]

    def _prep_trend(i):
        ta = asset_tri.iloc[i - window:i].values / asset_tri.iloc[i - window].values * 100
        tb = bench_tri.iloc[i - window:i].values / bench_tri.iloc[i - window] * 100
        t = np.arange(ta.shape[0]).reshape(-1, 1)
        return np.hstack([ta, t]), tb

    indices = list(range(window, len(asset_tri), reb))
    rebalance_dates = [asset_tri.index[min(i + data_lag, len(asset_tri) - 1)]
                       for i in indices]

    # _fit_date_trend is looked up at call time, so a caller can swap in another fit.
    results = Parallel(n_jobs=int(os.environ.get("RIDGE_INNER_JOBS", "4")), backend="loky")(
        delayed(_fit_date_trend)(*_prep_trend(i), n_assets, ridge_grid)
        for i in tqdm(indices, desc="Ridge (with trend)")
    )

    tracking_weights = [w[:n_assets] if w is not None else [np.nan] * n_assets for w in results]
    weights_df = pd.DataFrame(tracking_weights, index=rebalance_dates, columns=asset_returns.columns)
    weights_clean = _warn_dropped(weights_df, "ridge_trend")
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(
        asset_returns, weights_clean, tc_bps=tc_bps
    )
    return p_ret, p_tri, eff_w, turn, None


def _drifted_reference(asset_tri, w_prev, prev_pos, i):
    """
    The drifted turnover reference at a rebalance whose fit ends at row i - 1:
    the target weights w_prev, set on row prev_pos, drifted with the asset TRIs
    to the last close the fit sees and rescaled to sum to one, i.e. the
    simulator's pre-rebalance effective weights. Weights set on row prev_pos
    earn that day's return, so they were bought at the close of prev_pos - 1.
    """
    end = max(i - 1, prev_pos - 1)
    g = w_prev * (asset_tri.iloc[end].values / asset_tri.iloc[prev_pos - 1].values)
    return g / g.sum()


def run_ridge_trend_to(asset_returns, bench_returns, params):
    """
    Turnover-penalised ridge: the log-TRI ridge with trend plus
    delta * ||w - w_ref||^2. Runs sequentially, as each fit needs the previous
    weights.

    params: window (default 252), rebalance_freq (21), delta_turnover (1.0),
    ridge_grid, tc_bps (0), data_lag (0), and to_ref, the turnover reference of
    the full fit (the CV keeps equal weight either way):
      "target"   (default) w_ref = the previous rebalance's target weights
      "drifted"  w_ref = those weights drifted to the last close the fit sees
                 (_drifted_reference), the final specification
    Equal weight is the reference at the first rebalance in both cases. A failed
    fit is dropped before simulation, so the simulator keeps holding (and
    drifting) the last successful weights; the reference follows the same rule.
    """
    window = params.get("window", 252)
    reb    = params.get("rebalance_freq", 21)
    delta  = params.get("delta_turnover", 1.0)
    ridge_grid = _resolve_ridge_grid(params)
    tc_bps     = params.get("tc_bps", 0.0)
    data_lag   = int(params.get("data_lag", 0))
    to_ref     = params.get("to_ref", "target")
    if to_ref not in ("target", "drifted"):
        raise ValueError(f"to_ref must be 'target' or 'drifted', not {to_ref!r}")

    asset_tri = (1 + asset_returns).cumprod()
    bench_tri = (1 + bench_returns).cumprod()
    n_assets  = asset_returns.shape[1]

    def _prep(i):
        ta = asset_tri.iloc[i - window:i].values / asset_tri.iloc[i - window].values * 100
        tb = bench_tri.iloc[i - window:i].values / bench_tri.iloc[i - window] * 100
        t  = np.arange(ta.shape[0]).reshape(-1, 1)
        return np.hstack([ta, t]), tb

    indices         = list(range(window, len(asset_tri), reb))
    rebalance_dates = [asset_tri.index[min(i + data_lag, len(asset_tri) - 1)]
                       for i in indices]

    w_prev   = np.ones(n_assets) / n_assets
    prev_pos = None     # row of the rebalance date from which w_prev is held
    results  = []
    for i in tqdm(indices, desc="Ridge (trend+TO)"):
        X_aug, tb = _prep(i)
        ref = w_prev
        if to_ref == "drifted" and prev_pos is not None:
            ref = _drifted_reference(asset_tri, w_prev, prev_pos, i)
        # _fit_date_trend_to is looked up at call time, so a caller can swap in another fit.
        w = _fit_date_trend_to(X_aug, tb, n_assets, ridge_grid, delta, ref)
        results.append(w)
        if w is not None:
            w_prev = w[:n_assets]
            prev_pos = min(i + data_lag, len(asset_tri) - 1)

    tracking_weights = [w[:n_assets] if w is not None else [np.nan] * n_assets for w in results]
    weights_df = pd.DataFrame(tracking_weights, index=rebalance_dates,
                               columns=asset_returns.columns)
    weights_clean = _warn_dropped(weights_df, "ridge_trend_to")
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(
        asset_returns, weights_clean, tc_bps=tc_bps
    )
    return p_ret, p_tri, eff_w, turn, None


# ─────────────────────────────────────────────────────────────
# Equal weight and passive allocations
# ─────────────────────────────────────────────────────────────

def run_equal_weight(asset_returns, _bench_returns, params):
    """
    Equal weight in every asset, rebalanced every rebalance_freq days.

    params: rebalance_freq (default 21), tc_bps (0).
    """
    reb    = params.get("rebalance_freq", 21)
    tc_bps = params.get("tc_bps", 0.0)
    n = asset_returns.shape[1]
    w = np.ones(n) / n

    rebalance_dates = asset_returns.index[::reb]
    weights_df = pd.DataFrame(
        np.tile(w, (len(rebalance_dates), 1)),
        index=rebalance_dates,
        columns=asset_returns.columns,
    )
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(
        asset_returns, weights_df, tc_bps=tc_bps
    )
    return p_ret, p_tri, eff_w, turn, None


def run_passive_static(asset_returns, _bench_returns, params):
    """
    Fixed target weights (a passive allocation), rebalanced every
    rebalance_freq days.

    params: target_weights ({asset column: weight}, summing to one; see
    PASSIVE_TARGETS), rebalance_freq (default 63), tc_bps (0).
    """
    reb    = params.get("rebalance_freq", 63)
    tc_bps = params.get("tc_bps", 0.0)
    target = params["target_weights"]

    missing = [c for c in target if c not in asset_returns.columns]
    if missing:
        raise ValueError(
            f"target_weights references columns not in asset universe: {missing}. "
            f"Available: {list(asset_returns.columns)}"
        )
    s = float(sum(target.values()))
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"target_weights must sum to 1; got {s:.6f}")

    w = np.zeros(asset_returns.shape[1])
    cols = list(asset_returns.columns)
    for c, v in target.items():
        w[cols.index(c)] = v

    rebalance_dates = asset_returns.index[::reb]
    weights_df = pd.DataFrame(
        np.tile(w, (len(rebalance_dates), 1)),
        index=rebalance_dates,
        columns=asset_returns.columns,
    )
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(
        asset_returns, weights_df, tc_bps=tc_bps
    )
    return p_ret, p_tri, eff_w, turn, None


# The two passive allocations (methodology chapter, Table "Models compared").
PASSIVE_TARGETS = {
    "60_40": {
        "SA Equities": 0.60,
        "SA Bonds":    0.40,
    },
    "reg28": {
        "SA Equities":         0.45,
        "Unhedged Equities":   0.25,
        "SA Bonds":            0.15,
        "SA Property":         0.10,
        "SA Cash & Income":    0.05,
    },
}


# ─────────────────────────────────────────────────────────────
# Window-choosing models (rl_window_selection.py, window_baselines.py)
# ─────────────────────────────────────────────────────────────

def run_rl_bandit(asset_returns, bench_returns, params):
    """
    LinUCB bandit that chooses the lookback window at each rebalance, with the
    tracking reward of rl_window_selection.run_bandit_window_selection.

    params: window_choices (default 126..756), rebalance_freq (21),
    forward_days (63), alpha_ucb (1.2), ridge (1.0), eta_turnover (0.25),
    lambda_coint (0.05), delta_turnover (0.0), tc_bps (0), data_lag (0),
    mse_weight (1.0), coint_eval_days (None = forward_days),
    feature_subset (None = all nine context features).
    """
    from rl_window_selection import run_bandit_window_selection

    weights_reb, chosen_windows, diagnostics = run_bandit_window_selection(
        asset_returns=asset_returns,
        bench_returns=bench_returns,
        window_choices=params.get("window_choices", DEFAULT_WINDOWS),
        rebalance_freq=params.get("rebalance_freq", 21),
        forward_days=params.get("forward_days", 63),
        alpha_ucb=params.get("alpha_ucb", 1.2),
        ridge=params.get("ridge", 1.0),
        eta_turnover=params.get("eta_turnover", 0.25),
        lambda_coint=params.get("lambda_coint", 0.05),
        delta_turnover=params.get("delta_turnover", 0.0),
        tc_bps=params.get("tc_bps", 0.0),
        data_lag=params.get("data_lag", 0),
        mse_weight=params.get("mse_weight", 1.0),
        coint_eval_days=params.get("coint_eval_days", None),
        feature_subset=params.get("feature_subset", None),
    )
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(
        asset_returns, weights_reb, tc_bps=params.get("tc_bps", 0.0)
    )
    return p_ret, p_tri, eff_w, turn, pd.concat([chosen_windows, diagnostics], axis=1)


def run_window_oracle(asset_returns, bench_returns, params):
    """
    Look-ahead oracle: picks, with hindsight, the window with the lowest forward
    tracking error (window_baselines.oracle_window). A ceiling, not an
    implementable strategy.

    params: window_choices, rebalance_freq, forward_days, delta_turnover, data_lag, tc_bps.
    """
    from window_baselines import oracle_window

    weights_reb, chosen_windows, diagnostics = oracle_window(
        asset_returns=asset_returns,
        bench_returns=bench_returns,
        window_choices=params.get("window_choices", DEFAULT_WINDOWS),
        rebalance_freq=params.get("rebalance_freq", 21),
        forward_days=params.get("forward_days", 63),
        delta_turnover=params.get("delta_turnover", 0.0),
        data_lag=params.get("data_lag", 0),
    )
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(
        asset_returns, weights_reb, tc_bps=params.get("tc_bps", 0.0)
    )
    return p_ret, p_tri, eff_w, turn, pd.concat([chosen_windows, diagnostics], axis=1)


def run_window_random(asset_returns, bench_returns, params):
    """
    Random window at each rebalance (window_baselines.random_window).

    params: window_choices, rebalance_freq, forward_days, delta_turnover,
    data_lag, tc_bps, seed (default 0).
    """
    from window_baselines import random_window

    weights_reb, chosen_windows, diagnostics = random_window(
        asset_returns=asset_returns,
        bench_returns=bench_returns,
        window_choices=params.get("window_choices", DEFAULT_WINDOWS),
        rebalance_freq=params.get("rebalance_freq", 21),
        forward_days=params.get("forward_days", 63),
        delta_turnover=params.get("delta_turnover", 0.0),
        data_lag=params.get("data_lag", 0),
        seed=params.get("seed", 0),
    )
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(
        asset_returns, weights_reb, tc_bps=params.get("tc_bps", 0.0)
    )
    return p_ret, p_tri, eff_w, turn, pd.concat([chosen_windows, diagnostics], axis=1)


def run_window_volheuristic(asset_returns, bench_returns, params):
    """
    Volatility rule with no learning: shorter windows when trailing benchmark
    volatility is high (window_baselines.vol_heuristic_window).

    params: window_choices, rebalance_freq, forward_days, delta_turnover,
    data_lag, tc_bps, vol_lookback (default 63).
    """
    from window_baselines import vol_heuristic_window

    weights_reb, chosen_windows, diagnostics = vol_heuristic_window(
        asset_returns=asset_returns,
        bench_returns=bench_returns,
        window_choices=params.get("window_choices", DEFAULT_WINDOWS),
        rebalance_freq=params.get("rebalance_freq", 21),
        forward_days=params.get("forward_days", 63),
        delta_turnover=params.get("delta_turnover", 0.0),
        data_lag=params.get("data_lag", 0),
        vol_lookback=params.get("vol_lookback", 63),
    )
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(
        asset_returns, weights_reb, tc_bps=params.get("tc_bps", 0.0)
    )
    return p_ret, p_tri, eff_w, turn, pd.concat([chosen_windows, diagnostics], axis=1)


# Each runner takes (asset_returns, bench_returns, params) and returns
# (portfolio returns, portfolio TRI, effective weights, turnover, diagnostics),
# where diagnostics is the per-rebalance window choice or None.
MODELS = {
    "equal_weight":        run_equal_weight,
    "passive_static":      run_passive_static,
    "ridge_returns":       run_ridge_returns,
    "ridge_no_trend":      run_ridge_no_trend,
    "ridge_trend":         run_ridge_trend,
    "ridge_trend_to":      run_ridge_trend_to,
    "rl_bandit":           run_rl_bandit,
    "window_oracle":       run_window_oracle,
    "window_random":       run_window_random,
    "window_volheuristic": run_window_volheuristic,
}


# ─────────────────────────────────────────────────────────────
# Saving and the common evaluation start
# ─────────────────────────────────────────────────────────────

def save_turnover(run_name: str, turn: pd.Series, results_dir=paths.FINAL):
    """
    Write the one-way turnover at each rebalance (the non-zero entries of
    simulate_with_rebased_tri's turnover) to <results_dir>/turnover/<run>.csv,
    columns datestamp, turnover.
    """
    t_dir = os.path.join(results_dir, "turnover")
    os.makedirs(t_dir, exist_ok=True)
    safe_name = run_name.replace("/", "_").replace("\\", "_")
    path = os.path.join(t_dir, f"{safe_name}.csv")
    t = pd.Series(turn, dtype=float)
    t = t[t != 0]
    pd.DataFrame({"datestamp": pd.DatetimeIndex(t.index), "turnover": t.values}).to_csv(
        path, index=False, date_format="%Y-%m-%d")
    print(f"Saved turnover for '{run_name}' -> {path}")


def save_diagnostics(run_name: str, diag: pd.DataFrame, results_dir=paths.FINAL):
    """Write a window-choosing model's per-rebalance choices to <results_dir>/rl_diagnostics/<run>.csv."""
    diag_dir = os.path.join(results_dir, "rl_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, f"{run_name}.csv")
    diag.to_csv(path)
    print(f"Saved window choices for '{run_name}' -> {path}")


# Every model is evaluated from the same day: index 756 of its sample, the
# window-choosing models' first possible decision (756 = the longest lookback).
# For the full sample that is 2009-01-28. Models that start earlier (the
# ridges from index 252, the passives from day 0) keep their weights and are
# scored from that day; the first day's return is set to zero and its
# rebalance trade dropped, the simulator's convention for a first day.
EVAL_START_INDEX = 756


def clip_to_common_start(p_ret, p_tri, eff_w, turn, sample_index, start_index=EVAL_START_INDEX):
    """Cut a simulated run to start at sample_index[start_index] (see EVAL_START_INDEX)."""
    if len(sample_index) <= start_index:
        return p_ret, p_tri, eff_w, turn
    start = sample_index[start_index]
    if p_ret.index[0] > start:
        print(f"WARNING: model output starts {p_ret.index[0].date()}, after the common start {start.date()}")
        return p_ret, p_tri, eff_w, turn
    p_ret = p_ret.loc[start:].copy()
    p_ret.iloc[0] = 0.0
    p_tri = (1.0 + p_ret).cumprod()
    eff_w = eff_w.loc[start:]
    turn = turn[turn.index > start]
    return p_ret, p_tri, eff_w, turn


def run_and_save(
    run_name: str,
    model: str,
    index: str,
    model_params: dict,
    results_dir=paths.FINAL,
) -> dict:
    """
    Run one model, clip it to the common evaluation start, compute its metrics
    and save everything under results_dir (see the module docstring).

    run_name     : file name for the outputs, e.g. "equal_weight__ma__reb21__tc0"
    model        : a key of MODELS
    index        : "ma", "ma_le" or "ma_me" (see load_data)
    model_params : the runner's parameters; optional start_date / end_date
                   restrict the sample first
    results_dir  : absolute output folder (default paths.FINAL)

    Returns the metrics dict of metrics.compute_metrics, plus n_dropped_rebalances.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(MODELS)}")

    # Seed before every run so results do not depend on what ran before.
    np.random.seed(42)
    random.seed(42)

    print(f"\n{'='*60}")
    print(f"  Run      : {run_name}")
    print(f"  Model    : {model}")
    print(f"  Index    : {index}")
    print(f"  Params   : {model_params}")
    print(f"{'='*60}")

    asset_returns, bench_returns = load_data(index)

    start_date = model_params.get("start_date")
    end_date = model_params.get("end_date")
    if start_date:
        start_ts = pd.Timestamp(start_date)
        asset_returns = asset_returns[asset_returns.index >= start_ts]
        bench_returns = bench_returns[bench_returns.index >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date)
        asset_returns = asset_returns[asset_returns.index <= end_ts]
        bench_returns = bench_returns[bench_returns.index <= end_ts]

    _LAST_N_DROPPED["n"] = 0
    p_ret, p_tri, eff_w, turn, diag = MODELS[model](asset_returns, bench_returns, model_params)
    p_ret, p_tri, eff_w, turn = clip_to_common_start(p_ret, p_tri, eff_w, turn, asset_returns.index)

    metrics = compute_metrics(p_ret, p_tri, bench_returns, turn)
    n_dropped = int(_LAST_N_DROPPED["n"])
    metrics["n_dropped_rebalances"] = n_dropped
    if n_dropped > 0:
        print(f"\nWARNING: dropped {n_dropped} rebalance date(s) with failed/NaN "
              f"weights before simulation (model={model}).")

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:<28} {v:.6f}")

    params_to_save = {"model": model, "index": index, **model_params}
    save_run(run_name, params_to_save, metrics, results_dir)
    save_timeseries(run_name, build_timeseries(p_ret, p_tri, bench_returns), results_dir)
    save_weights_to_csv(run_name, eff_w, results_dir=results_dir)
    save_turnover(run_name, turn, results_dir=results_dir)
    if diag is not None:
        save_diagnostics(run_name, diag, results_dir=results_dir)

    if solver_failures:
        print(f"\nSolver failures this run: {dict(solver_failures)}")

    return metrics
