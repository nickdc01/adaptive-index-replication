"""
The window-choosing bandit (first version), its fixed-window comparators and
the 910-configuration tuning grid.

Library module (no command line), used by run_tuning.py, original_spec.py,
robustness_grid.py, final_bandit.py, final_runs.py, sibling_runs.py and
several figure scripts.

Three families share one decision schedule and one cached ridge choice:

- "bandit" : the window-choosing bandit (LinUCB, or D-LinUCB for gamma < 1),
             rewarded with lambda * (-ADF of the forward c_e-day spread of the
             chosen weights) - eta * turnover. This is the first version of the
             bandit (Table tab:original); final_bandit.py changes the tracker
             and the reward for the revised specification.
- "fixed"  : one lookback window, no bandit.
- "ec"     : the error-correcting bandit: the same arms, a context that also
             holds the realised spread's gap to its trailing trend line, and a
             reward equal to the reduction in that gap over the next 63 days.

The bandit updates its exploration width when an arm is chosen
(DelayedDLinUCB) and decides up to n - 64, the same dates as the fixed windows.

The inner tracker is rl_window_selection.fit_drift_tri_elasticnet. Its CV
choice of ridge coefficient does not depend on the previous weights, so it is
computed once per (decision index, window) and cached in paths.CACHE; only the
turnover-penalised refit (its second pass, refit below) runs per configuration.

grid() lists the 910 configurations of the tuning grid (864 bandit, 30 fixed,
16 error-correcting).
"""
import hashlib
import os
from collections import deque

import numpy as np
from scipy.optimize import minimize

import paths
import rl_window_selection as rws

WINDOWS = (126, 252, 378, 504, 756)
START_I = 756
FORWARD_DAYS = 63
LOOKBACK_STATE = 63
EC_TREND_DAYS = 504
EC_MIN_DAYS = 63
EC_HOLD = 63
EC_SIGMA0 = 0.01
CACHE_DIR = paths.CACHE


def schedule(n_obs, rebalance_freq=21, start_i=START_I):
    """Decision indices: from START_I every rebalance_freq days up to n_obs - FORWARD_DAYS - 1."""
    return list(range(start_i, n_obs - FORWARD_DAYS - 1, rebalance_freq))


# -----------------------------
# Data, features and the ridge cache
# -----------------------------
class Data:
    """Aligned returns and TRIs as arrays, with caches for features and ridge choices."""

    def __init__(self, asset_returns, bench_returns):
        idx = asset_returns.index.intersection(bench_returns.index)
        self.A = asset_returns.loc[idx].sort_index()
        self.B = bench_returns.loc[idx].sort_index()
        self.Ar = self.A.values
        self.Br = self.B.values
        self.asset_tri = (1 + self.A).cumprod().values
        self.bench_tri = (1 + self.B).cumprod().values
        self.n, self.n_assets = self.A.shape
        self._x = {}
        self.ridge = {}

    def key(self):
        """Cache key: a hash of the data, the ridge grid and the windows."""
        h = hashlib.sha1(np.ascontiguousarray(self.Ar).tobytes()
                         + np.ascontiguousarray(self.Br).tobytes()).hexdigest()
        g = hashlib.sha1(np.ascontiguousarray(rws._default_ridge_grid()).tobytes()).hexdigest()
        return f"{h[:16]}_{g[:8]}_{'-'.join(map(str, WINDOWS))}"

    def features(self, i):
        """The nine context features (rws.compute_state_features), strictly before i."""
        if i not in self._x:
            self._x[i] = rws.compute_state_features(self.A, self.B, i, lookback=LOOKBACK_STATE)
        return self._x[i]

    def window(self, fit_end, window):
        """Rebased TRIs plus a trend column on [fit_end - window, fit_end), as fit_drift_tri_elasticnet."""
        Xw_tri = self.asset_tri[fit_end - window:fit_end]
        yw_tri = self.bench_tri[fit_end - window:fit_end]
        Xw = Xw_tri / Xw_tri[0, :] * 100.0
        yw = yw_tri / yw_tri[0] * 100.0
        t = np.arange(Xw.shape[0]).reshape(-1, 1)
        return np.hstack([Xw, t]), yw


def best_ridge(X_aug, yw, n_assets):
    """fit_drift_tri_elasticnet's first pass: the CV-best ridge coefficient on the default grid."""
    best_cv, best = np.inf, None
    for ridge in rws._default_ridge_grid():
        cv = rws.cross_val_loss_drift_tri(X_aug, yw, ridge, n_assets, n_splits=3)
        if cv < best_cv:
            best_cv, best = cv, ridge
    return best


def refit(X_aug, yw, n_assets, ridge, delta_turnover, w_prev):
    """fit_drift_tri_elasticnet's second pass (the same code). None where that returns None."""
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
    return res.x[:n_assets].astype(float)


def _ridge_one(data, i, window):
    if i < window:   # only with a reporting lag, at the first decisions
        return None
    X_aug, yw = data.window(i, window)
    return best_ridge(X_aug, yw, data.n_assets)


def build_ridge_cache(data, indices, n_jobs=1, use_disk=True):
    """
    The CV-best ridge coefficient for every (decision index, window), read from
    and saved to CACHE_DIR (None where the CV fails).
    """
    path = os.path.join(CACHE_DIR, f"ridge_{data.key()}.npz")
    todo = sorted(set(int(i) for i in indices))
    if use_disk and os.path.exists(path):
        z = np.load(path)
        for i, w, r in zip(z["i"], z["w"], z["r"]):
            data.ridge[(int(i), int(w))] = None if np.isnan(r) else float(r)
    pairs = [(i, w) for i in todo for w in WINDOWS if (i, w) not in data.ridge]
    if pairs:
        if n_jobs == 1:
            vals = [_ridge_one(data, i, w) for i, w in pairs]
        else:
            from joblib import Parallel, delayed
            vals = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_ridge_one)(data, i, w) for i, w in pairs)
        for (i, w), r in zip(pairs, vals):
            data.ridge[(i, w)] = None if r is None else float(r)
        if use_disk:
            os.makedirs(CACHE_DIR, exist_ok=True)
            keys = sorted(data.ridge)
            np.savez(path, i=np.array([k[0] for k in keys]), w=np.array([k[1] for k in keys]),
                     r=np.array([np.nan if data.ridge[k] is None else data.ridge[k] for k in keys]))
    return data.ridge


def fitted_weights(data, i, window, delta_turnover, w_prev, prev_i=None):
    """Tracker weights at decision i on the given window (equal weight if the fit fails),
    with the turnover penalty measured against w_prev. prev_i, the decision index at
    which w_prev was set, is unused here; run_bandit and run_fixed pass it so that a
    replacement tracker (final_bandit.py) can drift w_prev to the rebalance date."""
    X_aug, yw = data.window(i, window)
    w = refit(X_aug, yw, data.n_assets, data.ridge[(i, window)], delta_turnover, w_prev)
    if w is None:
        w = np.ones(data.n_assets) / data.n_assets
    return w


# -----------------------------
# Bandit with the exploration width updated at selection
# -----------------------------
class DelayedDLinUCB:
    """D-LinUCB with delayed rewards. The point estimate uses matured rewards
    only (discounted V_a, b_a); the exploration width sqrt(x' W_a^-1 x) uses
    W_a, updated when arm a is chosen: W_a <- gamma W_a + x x' + (1 - gamma) ridge I.
    gamma = 1 gives LinUCB with delayed rewards."""

    def __init__(self, n_actions, d, alpha=1.0, ridge=1.0, gamma=1.0):
        self.n_actions, self.d = n_actions, d
        self.alpha, self.gamma = float(alpha), float(gamma)
        self._rI = ridge * np.eye(d)
        self.Vs = [ridge * np.eye(d) for _ in range(n_actions)]
        self.Ws = [ridge * np.eye(d) for _ in range(n_actions)]
        self.bs = [np.zeros(d) for _ in range(n_actions)]

    def select(self, x, alpha_override=None):
        alpha = self.alpha if alpha_override is None else alpha_override
        x = x.reshape(-1, 1)
        best_a, best_p = 0, -np.inf
        for a in range(self.n_actions):
            theta = np.linalg.solve(self.Vs[a], self.bs[a].reshape(-1, 1))
            mu = float((x.T @ theta).item())
            s = float(np.sqrt((x.T @ np.linalg.solve(self.Ws[a], x)).item()))
            p = mu + alpha * s
            if p > best_p:
                best_a, best_p = a, p
        return best_a

    def register(self, a, x):
        x = x.reshape(-1, 1)
        g = self.gamma
        self.Ws[a] = g * self.Ws[a] + x @ x.T + (1.0 - g) * self._rI

    def update(self, a, x, r):
        x = x.reshape(-1, 1)
        g = self.gamma
        self.Vs[a] = g * self.Vs[a] + x @ x.T + (1.0 - g) * self._rI
        self.bs[a] = g * self.bs[a] + float(r) * x.flatten()


def _make_bandit(fixes, d, kappa, gamma):
    """The bandit over WINDOWS. `fixes` must be True (see run_bandit)."""
    if not fixes:
        raise ValueError("only fixes=True (exploration width updated at selection) is implemented")
    return DelayedDLinUCB(len(WINDOWS), d, alpha=kappa, ridge=1.0, gamma=gamma)


# -----------------------------
# Error-correcting state and reward
# -----------------------------
class RealisedPath:
    """log V - log B of the portfolio actually held, with the simulator's
    conventions (weights set at decision i earn day i's return and drift
    between decisions; no costs). Only days before the current decision are read."""

    def __init__(self, data):
        self.d = data
        self.logV = {}
        self.base_logV = 0.0
        self.last = None       # (i, w) of the decision being held

    def advance_to(self, i):
        """Fill log V for days up to i - 1 from the held weights."""
        if self.last is None:
            return
        i0, w = self.last
        tri = self.d.asset_tri
        base = tri[i0 - 1] if i0 > 0 else tri[i0]
        days = np.arange(i0, i)
        mult = (tri[i0:i] / base) @ w
        for j, m in zip(days, mult):
            self.logV[int(j)] = self.base_logV + np.log(m)

    def rebalance(self, i, w):
        self.advance_to(i)
        if self.last is not None:
            self.base_logV = self.logV[i - 1]
        self.last = (i, w)

    def state(self, i):
        """(gap, slope, sigma) from the last EC_TREND_DAYS realised days before i."""
        days = [j for j in range(max(0, i - EC_TREND_DAYS), i) if j in self.logV]
        if len(days) < EC_MIN_DAYS:
            return 0.0, 0.0, EC_SIGMA0
        t = np.array(days, float)
        s = np.array([self.logV[j] for j in days]) - np.log(self.d.bench_tri[days])
        b, a = np.polyfit(t, s, 1)
        resid = s - (a + b * t)
        sigma = max(float(np.std(resid)), 1e-6)
        return float(s[-1] - (a + b * t[-1])), float(b), sigma


def ec_hold_gap_change(data, i, w):
    """63-day log return of buying and holding w from decision i, minus the benchmark's."""
    M = np.cumprod(1 + data.Ar[i:i + EC_HOLD], axis=0)
    port = float((M[-1] * w).sum())
    bench = float(np.prod(1 + data.Br[i:i + EC_HOLD]))
    return np.log(port) - np.log(bench)


# -----------------------------
# Runs
# -----------------------------
def run_bandit(data, cfg, fixes=True, log=None):
    """
    Run the bandit ("bandit" or "ec" family) on the decision schedule.

    cfg: family, kappa, eta, lambda_coint (default 2), ce, gamma, delta, reb, and
      tc_bps    optional reward cost term - tc_bps * turnover (default 0)
      features  bandit family only: indices into the nine context features
                (None keeps all nine; [0] is the constant alone, a context-free bandit)
      lag       bandit family only: a reporting lag in trading days (default 0);
                the context, the fit and the rewards then use data only up to
                lag days before each decision, while the weights still apply
                from the decision date. The ridge cache must hold the indices i - lag.
    fixes: must be True (kept so callers can state it).
    log:   if a list, (decision index, raw reward) pairs are appended to it.

    Returns (decision indices, chosen windows, weights array).
    """
    family = cfg["family"]
    kappa, eta, lam = float(cfg["kappa"]), float(cfg["eta"]), float(cfg.get("lambda_coint", 2.0))
    tc = float(cfg.get("tc_bps", 0.0))
    gamma, delta, reb = float(cfg["gamma"]), float(cfg["delta"]), int(cfg["reb"])
    subset = cfg.get("features")
    lag = int(cfg.get("lag", 0))
    if lag and family != "bandit":
        raise ValueError("cfg['lag'] applies to the bandit family only")
    if subset is not None:
        if family != "bandit":
            raise ValueError("cfg['features'] applies to the bandit family only")
        subset = [int(j) for j in subset]
    if family == "bandit":
        ce = int(cfg["ce"])
        horizon = max(FORWARD_DAYS, ce)
    else:
        horizon = EC_HOLD
    d = (9 if subset is None else len(subset)) if family == "bandit" else 10
    bandit = _make_bandit(fixes, d, kappa, gamma)
    idx = schedule(data.n, reb)
    path = RealisedPath(data) if family == "ec" else None

    pending, hist = deque(), []
    prev_w, prev_i = None, None
    chosen, weights = [], []
    ones = np.ones(data.n_assets) / data.n_assets
    for k, i in enumerate(idx):
        while pending and i >= pending[0]["i"] + horizon + lag:
            p = pending.popleft()
            turn = float(np.abs(p["w"] - p["prev"]).sum())
            if family == "bandit":
                ce_end = min(p["i"] + ce, data.n)
                c = rws.cointegration_reward(data.Ar[p["i"]:ce_end] @ p["w"], data.Br[p["i"]:ce_end])
                reward = lam * c - tc * turn - eta * turn
            else:
                D = ec_hold_gap_change(data, p["i"], p["w"])
                after = p["g"] + D - EC_HOLD * p["b"]
                reward = lam * (abs(p["g"]) - abs(after)) / p["sigma"] - tc * turn - eta * turn
            hist.append(reward)
            if log is not None:
                log.append((p["i"], reward))   # raw reward of the decision made at index p['i']
            rn = (reward - np.mean(hist)) / (np.std(hist) + 1e-8) if len(hist) > 1 else 0.0
            bandit.update(p["a"], p["x"], rn)

        x = data.features(i - lag)
        if subset is not None:
            # sliced once, so select, the pending entry and update all see it
            x = np.asarray(x, float)[subset]
        g = b = 0.0
        sigma = EC_SIGMA0
        if family == "ec":
            path.advance_to(i)
            g, b, sigma = path.state(i)
            x = np.append(x, np.tanh(g / sigma))
        alpha_t = kappa * max(1.0, 3.0 / np.sqrt(1 + k))
        a = bandit.select(x, alpha_override=alpha_t)
        bandit.register(a, x)
        if i - lag < WINDOWS[a]:
            raise ValueError(f"lag {lag} leaves too little data for window {WINDOWS[a]} at index {i}")
        # prev_i: the decision at which prev_w was set (for a drifted turnover reference)
        w = fitted_weights(data, i - lag, WINDOWS[a], delta, prev_w, prev_i=prev_i)
        pending.append(dict(a=a, x=x, w=w, i=i, prev=prev_w if prev_w is not None else ones,
                            g=g, b=b, sigma=sigma))
        if family == "ec":
            path.rebalance(i, w)
        chosen.append(WINDOWS[a])
        weights.append(w)
        prev_w, prev_i = w, i
    return idx, chosen, np.array(weights)


def run_fixed(data, cfg):
    """One fixed window (cfg: window, delta, reb) on the decision schedule."""
    idx = schedule(data.n, int(cfg["reb"]))
    prev_w, prev_i, weights = None, None, []
    for i in idx:
        w = fitted_weights(data, i, int(cfg["window"]), float(cfg["delta"]), prev_w, prev_i=prev_i)
        weights.append(w)
        prev_w, prev_i = w, i
    return idx, [int(cfg["window"])] * len(idx), np.array(weights)


def run_config(data, cfg):
    """Run any configuration of grid()."""
    if cfg["family"] == "fixed":
        return run_fixed(data, cfg)
    return run_bandit(data, cfg, fixes=True)


def config_name(cfg):
    """The configuration's name, e.g. bandit_k0.6_e0.5_g1_d1_reb63_ce252 or fixed_w252_d1_reb63."""
    f = cfg["family"]
    if f == "fixed":
        return "fixed_w%d_d%g_reb%d" % (cfg["window"], cfg["delta"], cfg["reb"])
    s = "%s_k%g_e%g_g%g_d%g_reb%d" % (f, cfg["kappa"], cfg["eta"], cfg["gamma"], cfg["delta"], cfg["reb"])
    return s + ("_ce%d" % cfg["ce"] if f == "bandit" else "")


def grid():
    """The 910 configurations of the tuning grid."""
    out = []
    for kappa in (0.0, 0.6, 1.2, 2.4):
        for eta in (0.0, 0.125, 0.25, 0.5):
            for ce in (252, 504, 756):
                for gamma in (1.0, 0.95, 0.9):
                    for delta in (0.0, 1.0, 4.0):
                        for reb in (21, 63):
                            out.append(dict(family="bandit", kappa=kappa, eta=eta, ce=ce,
                                            gamma=gamma, delta=delta, reb=reb))
    for window in WINDOWS:
        for delta in (0.0, 1.0, 4.0):
            for reb in (21, 63):
                out.append(dict(family="fixed", window=window, delta=delta, reb=reb))
    for kappa in (0.6, 1.2):
        for eta in (0.0, 0.25):
            for gamma in (1.0, 0.95):
                for delta in (0.0, 1.0):
                    out.append(dict(family="ec", kappa=kappa, eta=eta, gamma=gamma,
                                    delta=delta, reb=21))
    return out
