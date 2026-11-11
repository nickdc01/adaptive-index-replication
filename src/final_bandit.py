"""
The final specification of the bandit (Chapter 3) and its two tuning grids.

Tracker (Eq. 3.4). On an estimation window of n days ending the day before decision i,
        min over w, alpha, beta of
            sum_t [log y_t - log(sum_j X_tj w_j) - alpha - beta * t / 252]^2
            + r * sum_j w_j^2 + delta_n * sum_j (w_j - w_drift,j)^2
        subject to w >= 0, sum_j w_j = 1,
    where y and X are the peer-average and asset total-return indices rebased to 100 at
    the window start and t = 0, ..., n - 1. The constant alpha and the trend beta (a log
    drift a year) sit outside the log and are not penalised. The ridge strength r is
    chosen for every (decision, window) by 3-fold expanding-window cross-validation of
    the fit without the turnover term (build_ridge_cache_out). The turnover penalty grows
    with the window length, delta_n = delta * n / 252 (the same as normalising the fitting
    loss by n, with delta unchanged on the 1-year window), and is measured against
    w_drift, the previous rebalance's target weights drifted with the asset returns to
    the close before the decision (drifted_reference); at the first decision it is
    measured against equal weight.

Reward. Minus the forward mean absolute tracking error of the decision's portfolio,
        e_h = log(B_{i+h-1} / B_{i-1}) - log(V_{i+h-1} / V_{i-1}) - beta * h / 252,
              h = 1, ..., c_e
        MAE = mean_h |e_h|,
    where V is the buy-and-hold value of the decision's weights from decision day i (so
    the reward portfolio drifts as the reported one does), B the peer average and beta
    the drift fitted on the estimation window. The expected path carries the fitted drift
    forward and only the deviation from it is penalised; both paths start at the decision
    date, so the constant and the in-sample fit do not enter the reward. The reward is
    standardised by its running z-score before the bandit update. There is no turnover
    term (eta = 0).

Specifications (SPECS), both with the drifted penalty reference:
  final     the thesis specification above.
  final_in  the same with the trend inside the log and no constant,
            log(sum_j X_tj w_j + beta * t) with beta in index points, as in the original
            specification; the reward then carries the fitted drift forward in index
            points (forward_errors_in). Appendix tab:original, row "Bandit, trend inside
            the logarithm".

Each grid has 262 settings, all run here with the spec's tracker: the 216 bandit settings
with eta = 0 (kappa in {0, 0.6, 1.2, 2.4}, c_e in {252, 504, 756}, gamma in {1, 0.95, 0.9},
delta in {0, 1, 4}, monthly or quarterly rebalancing) and the 30 fixed-window and 16
error-correcting settings of bandit_tuning.grid. training_mae.py then applies the
selection rule: the lowest training MAE among the trend-outside settings with training
turnover below 150% a year.

Feeds: the selected setting (tab:models and every final run, through training_mae.py and
selected_setting), tab:tuning_key and tab:tuning_neighbours (grid_final, through
training_mae.py and thesis_tables.py), and the final-specification rows of tab:original
(grid_final and grid_final_in). FitB, run_bandit_m, run_fixed_v, build_ridge_cache_out and
selected_setting are also used by final_runs.py, sibling_runs.py, final_tests.py and the
figure scripts.

    python src/final_bandit.py cache            # the two ridge caches only (monthly schedule)
    python src/final_bandit.py check            # self-checks (minutes once the ridge caches exist)
    python src/final_bandit.py final            # both grids, final and final_in
    python src/final_bandit.py final final      # one grid

The trend-outside grid takes about 20 minutes on 2 workers. The first run also builds the
Eq. 3.4 ridge cache (1,160 decision-window pairs at about 10 seconds each, so several
CPU-hours). Runs at below-normal priority on N_JOBS workers (environment variable N_JOBS,
default 4) so the machine stays usable; the results do not depend on it.

Inputs: data/data.xlsx; ridge caches in results/cache (built from the data if missing:
ridge_<key>.npz for the trend inside the log, ridge_trendout_intercept_<key>.npz for Eq. 3.4).
Outputs (results/grids/): grid_<spec>.csv (train / test / full scores of every setting) and
grid_<spec>_paths.npz (each setting's weights __W, windows __win and decision indices __idx).
"""
import os
import sys
import time
import warnings
from collections import deque

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")


def _low_priority():
    """Below-normal priority; on Windows child processes inherit it."""
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        return
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)
        except Exception:
            pass


_low_priority()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import bandit_tuning as bt  # noqa: E402
import paths  # noqa: E402
import rl_window_selection as rws  # noqa: E402
import run_tuning as rt  # noqa: E402

OUT = paths.GRIDS
N_JOBS = int(os.environ.get("N_JOBS", 4))
rt.N_JOBS = N_JOBS     # rt.load() builds a missing ridge cache on at most N_JOBS workers
REB = 63
# The first version's setting (Table tab:original), used by the self-checks
TUNED = dict(family="bandit", kappa=0.6, eta=0.5, lambda_coint=2.0, ce=252, gamma=1.0, delta=1.0, reb=REB)
# Reference of the turnover penalty: the previous rebalance's target weights ('target', the
# first version) or those weights drifted to the close before the rebalance ('drifted')
TO_REFS = ("target", "drifted")
SPECS = {
    # the thesis specification: Eq. 3.4 (constant and trend outside the log), delta * n / 252
    # against the drifted weights, MAE reward only, standardised once by the running
    # z-score (so lambda drops out); run on the eta = 0 bandit settings
    "final": dict(trend_out=True, n_ref=252, drift=True, reward="mae", annualise=False, standardise=False,
                  to_ref="drifted"),
    # the same with the trend inside the log (additive drift in index points, no constant);
    # the MAE reward then carries that drift forward in index points (forward_errors_in)
    "final_in": dict(trend_out=False, n_ref=252, drift=True, reward="mae", annualise=False, standardise=False,
                     to_ref="drifted"),
}
TO_REF = SPECS["final"]["to_ref"]    # the final specification's penalty reference


def _z(x, hist):
    """x standardised by the mean and SD of hist (which already holds x), as the
    total reward is standardised; 0 until there are two values."""
    return (x - np.mean(hist)) / (np.std(hist) + 1e-8) if len(hist) > 1 else 0.0


# -----------------------------
# The tracker of Eq. 3.4: constant and trend outside the log
# -----------------------------
# The trend regressor is measured in years (t / 252), so beta is a log return a year.
# This is a change of units only (beta is not penalised), but with t in days SLSQP
# sometimes stopped after one step at the equal-weight start point.
TSCALE = 252.0


def _log_fit_out(X_aug, w, n_assets):
    """Fitted log peer average, log(X w) + beta * t / 252 + alpha, for the parameter vector
    w = (weights, beta, alpha); X_aug holds the rebased asset indices and the trend column
    t (bandit_tuning.Data.window). None where the portfolio is not positive."""
    port = X_aug[:, :n_assets] @ w[:n_assets]
    if np.any(port <= 0):
        return None
    return np.log(port) + w[n_assets] * X_aug[:, n_assets] / TSCALE + w[n_assets + 1]


def cv_loss_out(X_aug, y_tri, ridge, n_assets, n_splits=3):
    """Mean validation loss of the Eq. 3.4 fit without the turnover term over n_splits
    expanding-window folds (the first third of the window is the first training set), as
    rl_window_selection.cross_val_loss_drift_tri does for the trend inside the log."""
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

        bounds = [(0, 1)] * n_assets + [(None, None), (None, None)]
        w0 = np.concatenate([np.ones(n_assets) / n_assets, np.array([0.0, 0.0])])
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w[:n_assets]) - 1}
        res = minimize(penalized_train_loss, w0, bounds=bounds, constraints=cons, method='SLSQP')
        if not res.success:
            continue
        val_losses.append(fold_loss(res.x))
    return float(np.mean(val_losses)) if val_losses else np.inf


def best_ridge_out(X_aug, yw, n_assets):
    """The ridge strength with the lowest cross-validation loss on the default grid
    (rl_window_selection._default_ridge_grid); None if every value fails."""
    best_cv, best = np.inf, None
    for ridge in rws._default_ridge_grid():
        cv = cv_loss_out(X_aug, yw, ridge, n_assets, n_splits=3)
        if cv < best_cv:
            best_cv, best = cv, ridge
    return best


def refit_out_params(X_aug, yw, n_assets, ridge, delta_turnover, w_prev):
    """The Eq. 3.4 fit with the turnover penalty against w_prev (equal weight if None).
    Returns the parameter vector (weights, beta, alpha), or None when there is no ridge
    strength or the solver fails."""
    if ridge is None:
        return None
    bounds = [(0, 1)] * n_assets + [(None, None), (None, None)]
    w0 = np.concatenate([np.ones(n_assets) / n_assets, np.array([0.0, 0.0])])
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
    return res.x


def refit_out_full(X_aug, yw, n_assets, ridge, delta_turnover, w_prev):
    """refit_out_params as (weights, beta), beta a log drift a year; None on failure."""
    x = refit_out_params(X_aug, yw, n_assets, ridge, delta_turnover, w_prev)
    if x is None:
        return None
    return x[:n_assets].astype(float), float(x[n_assets])


def _ridge_out_one(data, i, window):
    """Ridge strength of the Eq. 3.4 fit on the window of `window` days ending before i;
    None when there are fewer than `window` days before i (reporting-lag runs)."""
    if i < window:
        return None
    X_aug, yw = data.window(i, window)
    return best_ridge_out(X_aug, yw, data.n_assets)


def build_ridge_cache_out(data, indices):
    """Ridge strength of the Eq. 3.4 fit for every (index in indices, window), as a dict
    {(i, window): ridge or None}. Cached in results/cache and extended with any pairs not
    yet in it, on N_JOBS workers."""
    path = paths.CACHE / f"ridge_trendout_intercept_{data.key()}.npz"
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
        print(f"Eq. 3.4 ridge cache: {len(pairs)} fits in {(time.time() - t0) / 60:.1f} min", flush=True)
    return cache


# The trend inside the log with no constant (the original specification's tracker; final_in)
def refit_in_full(X_aug, yw, n_assets, ridge, delta_turnover, w_prev):
    """bandit_tuning.refit (trend inside the log), returning (weights, beta, level):
    beta is the drift in index points a day on the window's base of 100, and level
    the fitted portfolio level X_n w on the last day of the window. The optimisation
    is bandit_tuning.refit's exactly, so the weights are identical."""
    if ridge is None:
        return None
    bounds = [(0, 1)] * n_assets + [(None, None)]
    w0 = np.concatenate([np.ones(n_assets) / n_assets, np.array([0.0])])
    _w_prev = w_prev if w_prev is not None else np.ones(n_assets) / n_assets

    def sum_to_one_constraint(w):
        return np.sum(w[:n_assets]) - 1
    cons = {'type': 'eq', 'fun': sum_to_one_constraint}

    def full_loss(w):
        port_tri = X_aug @ w
        if np.any(port_tri <= 0) or np.any(yw <= 0):
            return np.inf
        mse = np.sum((np.log(port_tri) - np.log(yw)) ** 2)
        ridge_pen = ridge * np.sum(w[:n_assets] ** 2)
        to_pen = delta_turnover * np.sum((w[:n_assets] - _w_prev) ** 2)
        return mse + ridge_pen + to_pen

    res = minimize(full_loss, w0, bounds=bounds, constraints=cons, method='SLSQP')
    if not res.success:
        return None
    w = res.x[:n_assets].astype(float)
    return w, float(res.x[n_assets]), float(X_aug[-1, :n_assets] @ w)


# -----------------------------
# Reference weights of the turnover penalty
# -----------------------------
def drifted_reference(asset_tri, w_prev, prev_i, fit_end):
    """The weights actually held just before a rebalance: the previous rebalance's target
    weights w_prev, set at decision index prev_i, drifted with each asset's total-return
    index to the last close the fit may see (fit_end - 1) and rescaled to sum to one.

    In the simulator (metrics.simulate_with_rebased_tri) weights set at index prev_i earn
    prev_i's close-to-close return, so they are bought at the close of prev_i - 1 and the
    weights held on day j >= prev_i are w_prev * tri[j] / tri[prev_i - 1], normalised. The
    pre-rebalance weights at the next decision i are those of day i - 1, so with no
    reporting lag (fit_end = i)
        g = w_prev * tri[i - 1] / tri[prev_i - 1],  reference = g / sum(g).
    With a reporting lag the fit ends at i - lag and the drift stops at i - lag - 1 (never
    before the purchase close prev_i - 1)."""
    end = max(int(fit_end) - 1, int(prev_i) - 1)
    g = np.asarray(w_prev, float) * (asset_tri[end] / asset_tri[int(prev_i) - 1])
    return g / g.sum()


# -----------------------------
# The tracker as a callable, keeping the fitted drift
# -----------------------------
class FitB:
    """fit(i, window, delta, w_prev, prev_i=None) -> weights: the tracker fitted on the
    `window` days before index i, with the turnover penalty delta * window / n_ref (delta
    itself if n_ref is None) against the reference of w_prev.

    trend_out=True is Eq. 3.4 (constant and trend outside the log; ridge strengths from
    ridge_out, build_ridge_cache_out); trend_out=False the trend inside the log with no
    constant (bandit_tuning.refit; ridge strengths from data.ridge). to_ref sets the
    penalty's reference: 'target' (w_prev itself) or 'drifted' (w_prev drifted to the close
    before the fit's end, drifted_reference; the caller passes prev_i, the decision index
    at which w_prev was set). Equal weight at the first rebalance either way; the
    cross-validation never sees w_prev. The defaults reproduce the first version's tracker
    (bandit_tuning.fitted_weights).

    After each call self.beta is the fitted drift (a log drift a year outside the log,
    index points a day inside, with self.level the fitted portfolio level on the window's
    last day), 0 for failed fits. Counts equal-weight fallbacks (failed fits) and stuck
    fits (reported as converged but left at the equal-weight start point, which an optimum
    essentially never is); ref_log keeps (fit_end, prev_i, reference) of every drifted
    reference."""

    def __init__(self, data, trend_out=False, n_ref=None, ridge_out=None, to_ref="target"):
        if to_ref not in TO_REFS:
            raise ValueError(f"to_ref must be one of {TO_REFS}, not {to_ref!r}")
        self.data, self.trend_out, self.n_ref, self.ridge_out = data, trend_out, n_ref, ridge_out
        self.to_ref = to_ref
        self.fallbacks = 0
        self.stuck = 0
        self.calls = 0
        self.ref_log = []
        self.beta = 0.0
        self.level = 100.0
        self.n_window = None

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
        self.beta = 0.0
        self.level = 100.0
        self.n_window = window
        d = delta * window / self.n_ref if self.n_ref else delta
        X_aug, yw = self.data.window(i, window)
        w = None
        if self.trend_out:
            r = refit_out_full(X_aug, yw, self.data.n_assets, self.ridge_out[(i, window)], d, w_prev)
            if r is not None:
                w, self.beta = r
        else:
            r = refit_in_full(X_aug, yw, self.data.n_assets, self.data.ridge[(i, window)], d, w_prev)
            if r is not None:
                w, self.beta, self.level = r
        if w is None:
            self.fallbacks += 1
            self.beta = 0.0
            self.level = 100.0
            w = np.ones(self.data.n_assets) / self.data.n_assets
        elif np.allclose(w, 1.0 / self.data.n_assets, atol=1e-6, rtol=0):
            self.stuck += 1
        return w


# -----------------------------
# The rewards
# -----------------------------
def reward_port_returns(data, i0, i1, w, drift):
    """Daily returns of the reward portfolio over [i0, i1): reset to w every day (the first
    version) or, with drift, w bought at decision day i0 and held."""
    R = data.Ar[i0:i1]
    if not drift:
        return R @ w
    V = np.cumprod(1.0 + R, axis=0) @ w
    return V / np.concatenate([[1.0], V[:-1]]) - 1.0


def forward_errors(data, i0, i1, w, beta):
    """e_h for h = 1 .. i1 - i0: peer average minus buy-and-hold portfolio, in logs,
    both from decision day i0, minus the fitted drift carried forward."""
    V = np.cumprod(1.0 + data.Ar[i0:i1], axis=0) @ w
    B = np.cumprod(1.0 + data.Br[i0:i1])
    h = np.arange(1, i1 - i0 + 1)
    return np.log(B) - np.log(V) - beta * h / TSCALE


def forward_mae(data, i0, i1, w, beta):
    return float(np.mean(np.abs(forward_errors(data, i0, i1, w, beta))))


def forward_errors_in(data, i0, i1, w, beta, level, n):
    """The same with the trend inside the log. The fit says the peer average's level is
    the portfolio's level plus beta * t index points (t = 0 at the window start, n - 1
    on its last day, base 100), so the expected log gap at window day t is
    log(1 + beta * t / M_t). Carried forward from the decision day, with the portfolio
    level M growing as the held portfolio V:
        d_h = log(1 + beta * (n - 1 + h) / (level * V_h)) - log(1 + beta * (n - 1) / level)
        e_h = log(B_h) - log(V_h) - d_h."""
    V = np.cumprod(1.0 + data.Ar[i0:i1], axis=0) @ w
    B = np.cumprod(1.0 + data.Br[i0:i1])
    h = np.arange(1, i1 - i0 + 1)
    a = np.maximum(1.0 + beta * (n - 1 + h) / (level * V), 1e-6)
    a0 = max(1.0 + beta * (n - 1) / level, 1e-6)
    return np.log(B) - np.log(V) - (np.log(a) - np.log(a0))


def forward_mae_in(data, i0, i1, w, beta, level, n):
    return float(np.mean(np.abs(forward_errors_in(data, i0, i1, w, beta, level, n))))


# -----------------------------
# Bandit and fixed-window loops
# -----------------------------
def run_bandit_m(data, cfg, fit, drift=False, reward="adf", annualise=False, standardise=False):
    """The bandit with tracker `fit` (a FitB); returns (decision indices, chosen windows,
    weights). reward "mae" is the forward MAE above; "adf" the original specification's
    ADF reward (on a buy-and-hold reward portfolio if drift). annualise scales the turnover
    term to a yearly rate and standardise puts the tracking and turnover terms on their
    own z-scores (both off in SPECS). cfg may carry "features" (a context subset; [0]
    keeps only the intercept, the context-free bandit) and "lag" (a reporting lag:
    context, fit and matured rewards use data up to lag days before each decision, and
    the weights apply from the decision). Each fit gets prev_i, the decision at which the
    previous weights were set, so a drifted reference runs from their purchase close to
    the last close the fit sees, i - lag - 1."""
    kappa, eta, lam = float(cfg["kappa"]), float(cfg["eta"]), float(cfg.get("lambda_coint", 2.0))
    tc = float(cfg.get("tc_bps", 0.0))
    gamma, delta, reb = float(cfg["gamma"]), float(cfg["delta"]), int(cfg["reb"])
    ce = int(cfg["ce"])
    horizon = max(bt.FORWARD_DAYS, ce)
    idx = bt.schedule(data.n, reb)
    subset = cfg.get("features")
    subset = None if subset is None else [int(j) for j in subset]
    lag = int(cfg.get("lag", 0))
    bandit = bt._make_bandit(True, 9 if subset is None else len(subset), kappa, gamma)
    pending, hist = deque(), []
    h_term, h_turn = [], []
    prev_w, prev_i = None, None
    chosen, weights = [], []
    ones = np.ones(data.n_assets) / data.n_assets
    for k, i in enumerate(idx):
        while pending and i >= pending[0]["i"] + horizon + lag:
            p = pending.popleft()
            turn = float(np.abs(p["w"] - p["prev"]).sum())
            if annualise:
                turn *= 252.0 / reb
            ce_end = min(p["i"] + ce, data.n)
            if reward == "adf":
                # -ADF statistic: larger = more stationary
                term = rws.cointegration_reward(reward_port_returns(data, p["i"], ce_end, p["w"], drift),
                                                data.Br[p["i"]:ce_end])
                good = term
            elif fit.trend_out:
                term = forward_mae(data, p["i"], ce_end, p["w"], p["beta"])
                good = -term
            else:
                term = forward_mae_in(data, p["i"], ce_end, p["w"], p["beta"], p["level"], p["n"])
                good = -term
            if standardise:
                # each term on its own scale (matured decisions only), so eta weighs
                # turnover against tracking in standard deviations; tc is in raw units
                h_term.append(good)
                h_turn.append(turn)
                r = lam * _z(good, h_term) - eta * _z(turn, h_turn) - tc * turn
            else:
                r = lam * good - tc * turn - eta * turn
            hist.append(r)
            rn = (r - np.mean(hist)) / (np.std(hist) + 1e-8) if len(hist) > 1 else 0.0
            bandit.update(p["a"], p["x"], rn)
        x = data.features(i - lag)
        if subset is not None:
            x = np.asarray(x, float)[subset]
        alpha_t = kappa * max(1.0, 3.0 / np.sqrt(1 + k))
        a = bandit.select(x, alpha_override=alpha_t)
        bandit.register(a, x)
        if i - lag < bt.WINDOWS[a]:
            raise ValueError(f"lag {lag} leaves too little data for window {bt.WINDOWS[a]} at index {i}")
        # the reward's turnover term keeps the target weights (it has zero weight in SPECS)
        w = fit(i - lag, bt.WINDOWS[a], delta, prev_w, prev_i=prev_i)
        pending.append(dict(a=a, x=x, w=w, i=i, beta=fit.beta, level=getattr(fit, "level", 100.0),
                            n=bt.WINDOWS[a], prev=prev_w if prev_w is not None else ones))
        chosen.append(bt.WINDOWS[a])
        weights.append(w)
        prev_w, prev_i = w, i
    return idx, chosen, np.array(weights)


def run_fixed_v(data, window, fit, delta=1.0, reb=REB):
    """One fixed window with tracker `fit` on the decision schedule; returns (decision
    indices, windows, weights)."""
    idx = bt.schedule(data.n, reb)
    prev_w, prev_i, weights = None, None, []
    for i in idx:
        w = fit(i, window, delta, prev_w, prev_i=prev_i)
        weights.append(w)
        prev_w, prev_i = w, i
    return idx, [window] * len(idx), np.array(weights)


def run_ec(data, cfg, fit):
    """The error-correcting family (bandit_tuning.run_bandit: its own reward, always
    monthly) with tracker `fit`. bandit_tuning calls its tracker once per decision, in
    order and with no reporting lag, so the decision at which the previous weights were
    set (prev_i, for a drifted reference) is the index of the previous call."""
    last = {"i": None}

    def fitted_weights(_data, i, window, delta, w_prev, prev_i=None):
        if prev_i is None and w_prev is not None:
            prev_i = last["i"]
        w = fit(i, window, delta, w_prev, prev_i=prev_i)
        last["i"] = i
        return w

    old = bt.fitted_weights
    bt.fitted_weights = fitted_weights
    try:
        return bt.run_bandit(data, dict(cfg), fixes=True)
    finally:
        bt.fitted_weights = old


# -----------------------------
# Grid evaluation
# -----------------------------
def evaluate_m(data, cfg, spec, ridge_out, to_ref=None):
    """One setting of bandit_tuning.grid under SPECS[spec] (penalty reference to_ref, by
    default the spec's): returns (score row, weights, chosen windows, decision indices).
    The fixed windows and the error-correcting family use the spec's tracker but not the
    bandit's reward."""
    s = SPECS[spec]
    fit = FitB(data, trend_out=s["trend_out"], n_ref=s["n_ref"], ridge_out=ridge_out, to_ref=to_ref or s["to_ref"])
    fam = cfg["family"]
    if fam == "fixed":
        idx, chosen, W = run_fixed_v(data, int(cfg["window"]), fit, delta=float(cfg["delta"]), reb=int(cfg["reb"]))
    elif fam == "bandit":
        idx, chosen, W = run_bandit_m(data, cfg, fit, drift=s["drift"], reward=s["reward"],
                                      annualise=s["annualise"], standardise=s["standardise"])
    else:
        idx, chosen, W = run_ec(data, cfg, fit)
    p_ret, p_tri, _, turn = rt.simulate(data, idx, W)
    row = dict(name=bt.config_name(cfg), variant=spec, **cfg)
    for tag, seg in (("train", rt.TRAIN), ("test", rt.TEST), ("full", rt.FULL)):
        for k, v in rt.segment(data, p_ret, p_tri, turn, seg).items():
            row[f"{tag}_{k}"] = v
    row["n_decisions"] = len(idx)
    row["fallbacks"] = fit.fallbacks
    row["stuck"] = fit.stuck
    return row, np.asarray(W), np.asarray(chosen), np.asarray(idx)


# -----------------------------
# Stages
# -----------------------------
def check(data, ro):
    """Self-checks; returns True if all pass."""
    import original_spec as osp   # the copied bandit loop of the original specification
    ok = True
    # 1. switches off (trend inside, no scaling, target reference, ADF reward) -> the
    #    first version, bandit_tuning.run_bandit, exactly
    idx0, ch0, W0 = bt.run_config(data, TUNED)
    idx1, ch1, W1 = run_bandit_m(data, TUNED, FitB(data))
    same = list(idx0) == list(idx1) and list(ch0) == list(ch1) and np.array_equal(W0, W1)
    print(f"switches off reproduces bandit_tuning.run_bandit: {same}", flush=True)
    ok &= same
    # 2. the Eq. 3.4 tracker, delta * n / 504 and the ADF reward on drifting weights: the
    #    loop reproduces original_spec.run_bandit_v with the same tracker exactly
    idx0, ch0, W0 = osp.run_bandit_v(data, TUNED, FitB(data, trend_out=True, n_ref=504, ridge_out=ro), drift=True)
    idx1, ch1, W1 = run_bandit_m(data, TUNED, FitB(data, trend_out=True, n_ref=504, ridge_out=ro), drift=True)
    same = list(idx0) == list(idx1) and list(ch0) == list(ch1) and np.array_equal(W0, W1)
    print(f"the loop reproduces original_spec.run_bandit_v with the Eq. 3.4 tracker: {same}", flush=True)
    ok &= same
    # 3. forward errors: vectorised = day-by-day loop, and h = 1 by hand
    i0, w = bt.START_I + 21 * 36, W1[10]   # a decision index on the monthly schedule
    X_aug, yw = data.window(i0, 504)
    r = refit_out_full(X_aug, yw, data.n_assets, ro[(i0, 504)], 1.0, None)
    beta = r[1] if r is not None else 0.0
    e = forward_errors(data, i0, i0 + 252, w, beta)
    hold = np.ones(data.n_assets)
    bench = 1.0
    loop = []
    for h in range(1, 253):
        hold = hold * (1 + data.Ar[i0 + h - 1])
        bench *= 1 + data.Br[i0 + h - 1]
        loop.append(np.log(bench) - np.log(hold @ w) - beta * h / TSCALE)
    e1 = np.log(1 + data.Br[i0]) - np.log((1 + data.Ar[i0]) @ w) - beta / TSCALE
    same = np.allclose(e, loop, atol=1e-12) and np.isclose(e[0], e1)
    print(f"forward errors match a day-by-day loop: {same} (beta {beta:.4f} a year, MAE {np.mean(np.abs(e)):.4f})", flush=True)
    ok &= same
    # 4. the fitted constant and drift explain the in-sample gap (printed only)
    x = refit_out_params(X_aug, yw, data.n_assets, ro[(i0, 504)], 1.0, None)
    if x is not None:
        f = _log_fit_out(X_aug, x, data.n_assets)
        print(f"  in-sample check: alpha {x[-1]:.4f}, beta {x[-2]:.4f} a year, "
              f"RMSE of the fit {np.sqrt(np.mean((np.log(yw) - f) ** 2)):.4f}", flush=True)
    # 5. drifted reference: every reference a fixed window uses equals the simulator's
    #    weights at the close before the rebalance, and the first fit has none
    fit = FitB(data, trend_out=True, n_ref=252, ridge_out=ro, to_ref="drifted")
    idx, _, W = run_fixed_v(data, 504, fit, delta=1.0, reb=REB)
    _, _, eff_w, _ = rt.simulate(data, idx, W)
    dates = data.A.index
    worst = max(float(np.abs(ref - eff_w.loc[dates[fe - 1]].to_numpy(float)).max()) for fe, _, ref in fit.ref_log)
    prev_ok = [(fe, p) for fe, p, _ in fit.ref_log] == [(int(idx[k]), int(idx[k - 1])) for k in range(1, len(idx))]
    same = prev_ok and worst < 1e-10
    print(f"drifted references equal the simulator's pre-rebalance weights: {same} "
          f"({len(fit.ref_log)} references, max |diff| {worst:.2e})", flush=True)
    ok &= same
    # 6. with no turnover penalty (delta = 0) the reference does not matter
    paths_ = [run_fixed_v(data, 252, FitB(data, trend_out=True, n_ref=252, ridge_out=ro, to_ref=t), delta=0.0, reb=REB)[2]
              for t in TO_REFS]
    same = np.array_equal(paths_[0], paths_[1])
    print(f"delta = 0: target and drifted references give the same weights: {same}", flush=True)
    ok &= same
    print("CHECK", "PASSED" if ok else "FAILED", flush=True)
    return ok


def grid_final(data, ro, spec="final"):
    """The final grid under SPECS[spec]: the 216 bandit settings with eta = 0 and the 30
    fixed-window and 16 error-correcting settings of bandit_tuning.grid, 262 in all, each
    with the spec's tracker and penalty reference. Writes grid_<spec>.csv and
    grid_<spec>_paths.npz to results/grids."""
    from joblib import Parallel, delayed
    t1 = time.time()
    cfgs = [c for c in bt.grid() if c["family"] == "bandit" and float(c["eta"]) == 0.0]
    others = [c for c in bt.grid() if c["family"] != "bandit"]
    assert len(cfgs) == 216 and len(others) == 46, (len(cfgs), len(others))
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(evaluate_m)(data, c, spec, ro) for c in cfgs + others)
    df = pd.DataFrame([r[0] for r in res])
    df["train_joint"] = df[["train_adf", "train_za"]].max(axis=1, skipna=False)
    assert len(df) == 262
    df.to_csv(OUT / f"grid_{spec}.csv", index=False)
    grid_paths = {}
    for r in res:
        n = r[0]["name"]
        grid_paths[f"{n}__W"], grid_paths[f"{n}__win"], grid_paths[f"{n}__idx"] = r[1], r[2], r[3]
    np.savez_compressed(OUT / f"grid_{spec}_paths.npz", **grid_paths)
    print(f"{spec}: {len(df)} settings in {(time.time() - t1) / 60:.1f} min (penalty reference "
          f"{SPECS[spec]['to_ref']}); {int(df['fallbacks'].sum())} fallbacks, {int(df['stuck'].sum())} stuck fits. "
          "Next: python src/training_mae.py (selection)", flush=True)


# -----------------------------
# The selected setting
# -----------------------------
def selected_setting():
    """Name of the headline bandit setting: the row marked selected in
    results/grids/selection_outside.csv, written by training_mae.py (lowest training MAE
    among the trend-outside settings with training turnover below 150% a year). Read by
    every script that refits or reports the tuned bandit."""
    path = OUT / "selection_outside.csv"
    if not path.exists():
        sys.exit(f"{path} not found: run `python src/training_mae.py` first")
    sel = pd.read_csv(path)
    sel = sel[sel["selected"]]
    if len(sel) != 1:
        sys.exit(f"{path}: {len(sel)} rows marked selected, expected 1")
    sel = sel.iloc[0]
    if sel["trend"] != "outside" or sel["family"] != "bandit":
        sys.exit(f"the selection rule picks {sel['name']} ({sel['family']}, trend {sel['trend']}), not a "
                 "trend-outside bandit setting, which the final runs assume")
    return sel["name"]


def selected_config():
    """(name, cfg) of the headline bandit setting, cfg as in bandit_tuning.grid (read from
    grid_final.csv)."""
    name = selected_setting()
    g = pd.read_csv(OUT / "grid_final.csv").set_index("name").loc[name]
    cfg = dict(family="bandit", kappa=float(g.kappa), eta=float(g.eta), ce=int(g.ce), gamma=float(g.gamma),
               delta=float(g.delta), reb=int(g.reb))
    assert bt.config_name(cfg) == name, (bt.config_name(cfg), name)
    return name, cfg


def main(stage, args):
    t0 = time.time()
    if stage not in ("cache", "check", "final"):
        sys.exit(f"unknown stage {stage!r}: use cache, check or final")
    specs = args or ["final", "final_in"]
    unknown = [s for s in specs if s not in SPECS]
    if stage == "final" and unknown:
        sys.exit(f"unknown specification(s) {unknown}; choose from {list(SPECS)}")
    paths.ensure(OUT)
    # rt.load() builds the ridge cache of the trend inside the log (bandit_tuning) if missing
    data = rt.load()
    need_ro = stage in ("cache", "check") or any(SPECS[s]["trend_out"] for s in specs)
    ro = build_ridge_cache_out(data, bt.schedule(data.n, 21)) if need_ro else {}
    print(f"loaded in {(time.time() - t0) / 60:.1f} min; n = {data.n}", flush=True)
    if stage == "check":
        if not check(data, ro):
            sys.exit(1)
    elif stage == "final":
        for spec in specs:
            grid_final(data, ro, spec)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "check", sys.argv[2:])
