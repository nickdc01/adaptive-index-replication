"""
The window-choosing bandit and its inner tracker.

Library module (no command line), used by run_models.py (model "rl_bandit"),
window_baselines.py, bandit_tuning.py and the scripts built on them.

    fit_drift_tri_elasticnet   the inner tracker: non-negative ridge on log
                               total-return-index levels with a linear trend
                               and an optional turnover penalty, ridge
                               coefficient chosen by expanding-window CV
    compute_state_features     the nine context features, from data strictly
                               before the decision
    cointegration_reward       minus the ADF statistic of a forward log spread
    LinUCB                     the contextual bandit
    run_bandit_window_selection
                               a monthly window-choosing bandit with a tracking
                               (MSE) reward, run as rl_bandit__ma__tc0 and
                               rl_bandit_to__ma__tc0 (pairs in the
                               Diebold-Mariano family of dm_test.py)

The tracker is named "elasticnet", although the L1 term has no effect: the weights are
long-only and sum to one, so the L1 term of an elastic net is constant on the
feasible set and has no effect; the fit is a non-negative ridge with
coefficient r = lambda * (1 - alpha) / 2. The ridge grid is the set of those
coefficients over lambda in logspace(-5, 2, 20) and alpha in
{0, 0.25, 0.5, 0.75, 1} (81 values from 0 to 50; see Jagannathan and Ma, 2003,
for the same argument on no-short-sale constraints).
"""

from collections import deque

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

_LAM_GRID = np.logspace(-5, 2, 20)
_ALPHA_GRID = np.linspace(0, 1, 5)


def _default_ridge_grid():
    """The 81-point ridge grid { lambda * (1 - alpha) * 0.5 } (see the module docstring)."""
    return np.array(sorted(set(
        float(lam) * (1.0 - float(alpha)) * 0.5
        for lam in _LAM_GRID
        for alpha in _ALPHA_GRID
    )))


# -----------------------------
# 1) Inner tracker: log-TRI ridge with trend, expanding-window CV
# -----------------------------
def cross_val_loss_drift_tri(X_aug, y_tri, ridge, n_assets, n_splits=3):
    """
    Expanding-window CV loss of the log-TRI ridge with trend: the first third
    of the window is the first training set, and the rest is cut into
    n_splits validation folds, each trained on everything before it.
    """
    T = X_aug.shape[0]
    min_train = T // 3
    fold_size = (T - min_train) // n_splits
    val_losses = []

    for j in range(n_splits):
        train_end = min_train + j * fold_size
        val_end = min(train_end + fold_size, T)

        X_train = X_aug[:train_end]
        y_train = y_tri[:train_end]
        X_val = X_aug[train_end:val_end]
        y_val = y_tri[train_end:val_end]

        if len(X_val) < 5:
            continue

        def penalized_train_loss(w):
            port_tri = X_train @ w
            if np.any(port_tri <= 0) or np.any(y_train <= 0):
                return np.inf
            mse = np.sum((np.log(port_tri) - np.log(y_train)) ** 2)
            ridge_pen = ridge * np.sum(w[:n_assets] ** 2)
            return mse + ridge_pen

        def fold_loss(w):
            port_tri = X_val @ w
            if np.any(port_tri <= 0) or np.any(y_val <= 0):
                return np.inf
            return np.sum((np.log(port_tri) - np.log(y_val)) ** 2)

        def sum_to_one_constraint(w):
            return np.sum(w[:n_assets]) - 1

        bounds = [(0, 1)] * n_assets + [(None, None)]
        w0 = np.concatenate([np.ones(n_assets) / n_assets, np.array([0.0])])
        cons = {'type': 'eq', 'fun': sum_to_one_constraint}

        res = minimize(penalized_train_loss, w0, bounds=bounds,
                       constraints=cons, method='SLSQP')
        if not res.success:
            continue

        val_losses.append(fold_loss(res.x))

    return float(np.mean(val_losses)) if val_losses else np.inf


def fit_drift_tri_elasticnet(asset_tri_window, bench_tri_window,
                             ridge_grid=None, delta_turnover=0.0, w_prev=None):
    """
    The inner tracker: non-negative ridge on log TRI levels with a linear trend
    (inside the logarithm) and the turnover penalty delta * ||w - w_prev||^2.

    asset_tri_window : (n, p) asset TRIs on the estimation window
    bench_tri_window : (n,) benchmark TRI on the same days
    ridge_grid       : ridge coefficients to search (default: the 81-point grid)
    delta_turnover   : turnover penalty (0 = none)
    w_prev           : previous rebalance weights (equal weight if None)

    Returns (weights, trend slope), or (None, None) when the fit fails.
    """
    if ridge_grid is None:
        ridge_grid = _default_ridge_grid()
    else:
        ridge_grid = np.asarray(ridge_grid, dtype=float)

    # rebase to 100
    Xw = asset_tri_window / asset_tri_window[0, :] * 100.0
    yw = bench_tri_window / bench_tri_window[0] * 100.0

    # add time trend
    t = np.arange(Xw.shape[0]).reshape(-1, 1)
    X_aug = np.hstack([Xw, t])

    n_assets = Xw.shape[1]
    bounds = [(0, 1)] * n_assets + [(None, None)]
    w0 = np.concatenate([np.ones(n_assets) / n_assets, np.array([0.0])])
    _w_prev = w_prev if w_prev is not None else np.ones(n_assets) / n_assets

    def sum_to_one_constraint(w):
        return np.sum(w[:n_assets]) - 1
    cons = {'type': 'eq', 'fun': sum_to_one_constraint}

    # Pass 1: choose the ridge coefficient by CV (no turnover penalty in the folds).
    best_cv, best_ridge = np.inf, None
    for ridge in ridge_grid:
        cv = cross_val_loss_drift_tri(X_aug, yw, ridge, n_assets, n_splits=3)
        if cv < best_cv:
            best_cv, best_ridge = cv, ridge

    if best_ridge is None:
        return None, None

    # Pass 2: one full SLSQP fit at that coefficient, with the turnover penalty.
    def full_loss(w):
        port_tri = X_aug @ w
        if np.any(port_tri <= 0) or np.any(yw <= 0):
            return np.inf
        mse       = np.sum((np.log(port_tri) - np.log(yw)) ** 2)
        ridge_pen = best_ridge * np.sum(w[:n_assets] ** 2)
        to_pen    = delta_turnover * np.sum((w[:n_assets] - _w_prev) ** 2)
        return mse + ridge_pen + to_pen

    res = minimize(full_loss, w0, bounds=bounds, constraints=cons, method='SLSQP')
    if not res.success:
        return None, None
    return res.x[:n_assets].astype(float), float(res.x[-1])


# -----------------------------
# 2) State features
# -----------------------------
def compute_state_features(asset_returns, bench_returns, t_idx, lookback=63):
    """
    The nine context features at decision index t_idx, from data strictly
    before t_idx: [0] constant, [1] benchmark volatility, [2] active volatility
    of equal weight, [3] benchmark momentum, [4] return autocorrelation,
    [5] benchmark drawdown over the last year, [6] volatility of volatility,
    [7] cross-sectional dispersion, [8] recent drift of the equal-weight spread.
    """
    start = max(0, t_idx - lookback)
    br = bench_returns.iloc[start:t_idx]
    ar = asset_returns.iloc[start:t_idx].values

    if len(br) < 10:
        return np.zeros(9, dtype=float)

    bench_vol = float(br.std())
    bench_mom = float((1 + br).prod() - 1.0)

    # equal-weight active vol proxy
    ew = np.ones(ar.shape[1]) / ar.shape[1]
    port = ar @ ew
    active = port - br.values
    active_vol = float(np.std(active))

    # short-term autocorr proxy (regime-ish)
    ac = float(pd.Series(br.values).autocorr(lag=1))
    if np.isnan(ac):
        ac = 0.0

    # last 1y drawdown in benchmark TRI
    tri = (1 + bench_returns).cumprod()
    dd_start = max(0, t_idx - 252)
    tri_slice = tri.iloc[dd_start:t_idx]
    peak = tri_slice.cummax()
    bench_dd = float((tri_slice / peak - 1.0).min()) if len(tri_slice) else 0.0

    # vol-of-vol (captures regime transitions)
    if len(br) >= 21:
        rolling_vol = pd.Series(br.values).rolling(21).std().dropna()
        vol_of_vol = float(rolling_vol.std()) if len(rolling_vol) > 1 else 0.0
    else:
        vol_of_vol = 0.0

    # cross-sectional dispersion of asset returns
    cross_disp = float(np.std(ar.mean(axis=0))) if ar.shape[0] > 1 else 0.0

    # log-spread trend (recent tracking quality under equal weights)
    bench_cum = (1 + br).cumprod().values
    port_cum = (1 + pd.Series(port)).cumprod().values
    if len(bench_cum) > 5:
        spread = np.log(port_cum) - np.log(bench_cum)
        recent_drift = float(spread[-1] - spread[0])
    else:
        recent_drift = 0.0

    return np.array([1.0, bench_vol, active_vol, bench_mom, ac,
                     bench_dd, vol_of_vol, cross_disp, recent_drift], dtype=float)


# -----------------------------
# 3) Reward: cointegration signal
# -----------------------------
def cointegration_reward(port_rets, bench_rets):
    """
    Minus the ADF statistic (constant and trend, 5 lags) of the log spread of
    the given daily returns. The reward adds lambda_coint times this signal, so
    a more stationary spread (a more negative ADF statistic) earns more.
    Returns 0 for fewer than 15 days or a failed test.
    """
    tri_p = (1 + pd.Series(port_rets)).cumprod().values
    tri_b = (1 + pd.Series(bench_rets)).cumprod().values
    spread = np.log(tri_p) - np.log(tri_b)
    if len(spread) < 15:
        return 0.0
    try:
        # constant + trend, as the ADF reported in metrics.compute_metrics
        adf_stat = adfuller(spread, maxlag=5, regression="ct", autolag=None)[0]
        return -float(adf_stat)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        return 0.0


# -----------------------------
# 4) LinUCB contextual bandit
# -----------------------------
class LinUCB:
    """
    LinUCB (Li et al., 2010). For each action a:
      A_a = dxd matrix, b_a = dx1 vector
      theta_a = inv(A_a) b_a
    Choose argmax_a [ x^T theta_a + alpha * sqrt(x^T inv(A_a) x) ]
    Update with observed reward r: A_a += x x^T, b_a += r x
    """
    def __init__(self, n_actions, d, alpha=1.0, ridge=1.0):
        self.n_actions = n_actions
        self.d = d
        self.alpha = float(alpha)
        self.As = [ridge * np.eye(d) for _ in range(n_actions)]
        self.bs = [np.zeros(d) for _ in range(n_actions)]

    def select(self, x, alpha_override=None):
        alpha = alpha_override if alpha_override is not None else self.alpha
        x = x.reshape(-1, 1)
        best_a, best_p = 0, -1e18
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.As[a])
            theta = A_inv @ self.bs[a].reshape(-1, 1)
            mu = float((x.T @ theta).item())
            s = float(np.sqrt((x.T @ A_inv @ x).item()))
            p = mu + alpha * s
            if p > best_p:
                best_p = p
                best_a = a
        return best_a

    def update(self, a, x, r):
        x = x.reshape(-1, 1)
        self.As[a] = self.As[a] + (x @ x.T)
        self.bs[a] = self.bs[a] + float(r) * x.flatten()


# -----------------------------
# 5) Walk-forward backtest with bandit window selection
# -----------------------------
def run_bandit_window_selection(
    asset_returns: pd.DataFrame,
    bench_returns: pd.Series,
    window_choices,
    rebalance_freq=21,
    forward_days=63,
    lookback_state=63,
    alpha_ucb=1.2,
    ridge=1.0,
    eta_turnover=0.25,
    lambda_coint=0.05,
    tc_bps=0.0,
    delta_turnover=0.0,
    data_lag=0,
    mse_weight=1.0,
    coint_eval_days=None,
    feature_subset=None,
):
    """
    Walk-forward window choice with LinUCB:
      - at each rebalance, the context and the fit use only past data
      - the reward is measured over the following forward_days
      - the bandit is updated only once that reward has been observed

    Exploration comes from a UCB width that decays with the step number.

    Reward shaping
    --------------
    reward = -mse_weight*mse_active + lambda_coint*coint_signal
             - tc_cost - eta_turnover*turn

    mse_weight     : float  weight on the active-tracking MSE term (default 1.0;
                            0.0 drops tracking entirely so the reward is purely
                            cointegration-seeking).
    coint_eval_days: int or None  days after the decision over which the ADF
                            signal is measured. None (default) uses forward_days,
                            the MSE window; otherwise the ADF uses the longer
                            slice (clamped to the data) and the MSE stays on
                            forward_days.

    A decision's reward is used only once max(forward_days, coint_eval_days)
    days have passed, so no choice is informed by data after its own date.

    feature_subset : list[int] or None
        Indices into the fixed 9-element state vector from compute_state_features
        (order: [0]=intercept, [1]=benchmark vol, [2]=active vol,
        [3]=benchmark momentum, [4]=return autocorrelation, [5]=drawdown,
        [6]=vol-of-vol, [7]=cross-sectional dispersion, [8]=recent drift).
        None (default) uses all nine. A list restricts the context to those
        columns; the vector is sliced once, so selection and update see the
        same reduced vector.
    """

    # align
    idx = asset_returns.index.intersection(bench_returns.index)
    asset_returns = asset_returns.loc[idx].sort_index()
    bench_returns = bench_returns.loc[idx].sort_index()

    asset_tri = (1 + asset_returns).cumprod()
    bench_tri = (1 + bench_returns).cumprod()

    max_w = int(max(window_choices))
    n_actions = len(window_choices)

    # The reward horizon is the longer of the MSE window and the ADF window.
    # Decisions stop this many days before the end of the data.
    reward_horizon = forward_days if coint_eval_days is None else max(forward_days, int(coint_eval_days))

    # determine rebalance indices
    start_i = max_w
    end_i = len(asset_returns) - reward_horizon - 1
    rebalance_is = list(range(start_i, end_i, rebalance_freq))

    d = 9 if feature_subset is None else len(feature_subset)
    bandit = LinUCB(n_actions=n_actions, d=d, alpha=alpha_ucb, ridge=ridge)

    # logs
    chosen_windows = []
    chosen_actions = []
    chosen_dates = []
    chosen_weights = []
    reward_history = []

    prev_w = None

    # Pending decisions (action, x, w, i, previous weights), used for an
    # update once their reward horizon has passed.
    pending = deque()

    for step_k, i in enumerate(tqdm(rebalance_is, desc="Bandit Window Selection")):
        date_t = asset_returns.index[i]

        # ── Update with the decisions whose reward horizon has passed ──
        while pending and i >= pending[0][3] + reward_horizon:
            p_action, p_x, p_w, p_i, p_prev_w = pending.popleft()
            # MSE stays on the forward_days window.
            X_test = asset_returns.iloc[p_i:p_i + forward_days].values
            y_test = bench_returns.iloc[p_i:p_i + forward_days].values
            port_ret = X_test @ p_w
            mse_active = float(np.mean((port_ret - y_test) ** 2))

            # ADF signal over the (optionally longer) coint_eval_days window.
            if coint_eval_days is None:
                coint_port, coint_bench = port_ret, y_test
            else:
                ce_end = min(p_i + int(coint_eval_days), len(asset_returns))
                X_ce = asset_returns.iloc[p_i:ce_end].values
                y_ce = bench_returns.iloc[p_i:ce_end].values
                coint_port, coint_bench = X_ce @ p_w, y_ce
            coint_signal = cointegration_reward(coint_port, coint_bench)

            # Turnover vs the weights held before this decision
            turn = float(np.abs(p_w - p_prev_w).sum())
            # tc_bps enters in basis points (not divided by 1e4); eta_turnover
            # is a separate turnover penalty.
            tc_cost = tc_bps * turn
            reward = -mse_weight * mse_active + lambda_coint * coint_signal - tc_cost - eta_turnover * turn

            reward_history.append(reward)
            if len(reward_history) > 1:
                reward_norm = (reward - np.mean(reward_history)) / (np.std(reward_history) + 1e-8)
            else:
                reward_norm = 0.0

            bandit.update(p_action, p_x, reward_norm)

        # ── Context and action ─────────────────────────────────
        # With data_lag > 0 the context and the fit use data only up to
        # i - data_lag (a reporting lag).
        fit_end = i - data_lag
        x = compute_state_features(asset_returns, bench_returns, fit_end, lookback=lookback_state)
        if feature_subset is not None:
            x = np.asarray(x, float)[feature_subset]
        alpha_t = alpha_ucb * max(1.0, 3.0 / np.sqrt(1 + step_k))
        action  = bandit.select(x, alpha_override=alpha_t)
        window  = int(window_choices[action])

        # ── Fit weights on [fit_end-window, fit_end) ───────────
        Xw_tri = asset_tri.iloc[fit_end - window:fit_end].values
        yw_tri = bench_tri.iloc[fit_end - window:fit_end].values
        w, trend = fit_drift_tri_elasticnet(
            Xw_tri, yw_tri,
            delta_turnover=delta_turnover,
            w_prev=prev_w,
        )
        if w is None:
            w = np.ones(asset_returns.shape[1]) / asset_returns.shape[1]

        # ── Queue this decision for deferred update ────────────
        # Keep the previous weights so the reward's turnover uses them
        p_prev = prev_w if prev_w is not None else np.ones(asset_returns.shape[1]) / asset_returns.shape[1]
        pending.append((action, x, w, i, p_prev))

        # store for output
        chosen_actions.append(action)
        chosen_windows.append(window)
        chosen_dates.append(date_t)
        chosen_weights.append(w)
        prev_w = w

    # Update with the remaining decisions on the data available
    while pending:
        p_action, p_x, p_w, p_i, p_prev_w = pending.popleft()
        avail_end = min(p_i + forward_days, len(asset_returns))
        if avail_end <= p_i:
            continue
        # MSE stays on the forward_days window (clamped to available data).
        X_test = asset_returns.iloc[p_i:avail_end].values
        y_test = bench_returns.iloc[p_i:avail_end].values
        port_ret = X_test @ p_w
        mse_active = float(np.mean((port_ret - y_test) ** 2))

        # ADF signal over the (optionally longer) coint_eval_days window.
        if coint_eval_days is None:
            coint_port, coint_bench = port_ret, y_test
        else:
            ce_end = min(p_i + int(coint_eval_days), len(asset_returns))
            X_ce = asset_returns.iloc[p_i:ce_end].values
            y_ce = bench_returns.iloc[p_i:ce_end].values
            coint_port, coint_bench = X_ce @ p_w, y_ce
        coint_signal = cointegration_reward(coint_port, coint_bench)

        turn     = float(np.abs(p_w - p_prev_w).sum())
        tc_cost  = tc_bps * turn
        reward   = -mse_weight * mse_active + lambda_coint * coint_signal - tc_cost - eta_turnover * turn
        reward_history.append(reward)
        if len(reward_history) > 1:
            reward_norm = (reward - np.mean(reward_history)) / (np.std(reward_history) + 1e-8)
        else:
            reward_norm = 0.0
        bandit.update(p_action, p_x, reward_norm)

    weights_reb = pd.DataFrame(
        chosen_weights,
        index=pd.DatetimeIndex(chosen_dates),
        columns=asset_returns.columns
    )
    chosen_windows = pd.Series(
        chosen_windows,
        index=pd.DatetimeIndex(chosen_dates),
        name="chosen_window"
    )
    diagnostics = pd.DataFrame(
        {"reward": reward_history[:len(chosen_dates)]},
        index=pd.DatetimeIndex(chosen_dates[:len(reward_history)]),
    )
    return weights_reb, chosen_windows, diagnostics
