"""Two-break unit-root tests on a replication spread: Lumsdaine-Papell and Lee-Strazicich.

Feeds the Lumsdaine-Papell (LP), Lee-Strazicich (LS) and Capped columns of the
two-break table (tab:two_breaks) and its simulated critical values. This module holds
the tests; final_tests.py breaks runs them on the final spreads.

Spread: log(portfolio TRI) - log(benchmark TRI) over 28 January 2009 to 15 May 2024.

1. Lumsdaine-Papell (1997), model CC: breaks in intercept and trend at two unknown
   dates, no breaks under the null.

       dy_t = mu + b*t + th1*DU1 + g1*DT1 + th2*DU2 + g2*DT2 + alpha*y_{t-1}
              + sum_i c_i*dy_{t-i} + e_t,   DUj = 1(t > TBj), DTj = (t - TBj)*DUj

2. Lee-Strazicich (2003) minimum LM test, model C: the same two breaks in level and
   trend, but allowed under the null as well, so a rejection is not an artefact of a
   unit-root process that itself has breaks.

       first stage   dy_t on dZ_t = [1, B1, B2, DU1, DU2]  (Bj = 1(t = TBj + 1))
       S_t = y_t - y_0 - Z_t'delta    (detrended level; S_0 = 0)
       test          dy_t on [dZ_t, S_{t-1}, dS_{t-1..t-k}],  statistic = t-ratio on S_{t-1}

For both, the statistic is the minimum t-ratio over break pairs (the dates least
favourable to the null, not formal break estimates). The reported date TB is the
last trading day before each break.

Search: break dates in the middle 70% of the sample (15% trimming, as in the
Zivot-Andrews test), breaks at least 10% of the sample apart, a 21-day grid refined
day by day within one grid step of the best pair, with the same rules in both stages.
The lag length k is the AIC choice of the plain ADF regression with constant and
trend (statsmodels default maximum), applied identically to data, simulations and
bootstrap draws.

p-values and critical values:
  critical values  quantiles of the statistics on 600 Gaussian driftless random walks
                   of the same length (seeds 20261001 + r; for LS this no-break null
                   with a full search is conservative).
  p_boot           sieve wild bootstrap under the unit-root null (AR(k) fitted to dy,
                   Rademacher-signed residuals), which keeps each spread's own
                   volatility path, including the COVID-19 spike. (count + 1) / (reps + 1).

Robustness: k fixed at 12 and at 20; daily changes capped (winsorised) at 4 standard
deviations, rebuilding the level, and the search rerun.

The LP (1997) tabulated critical values for model CC (-7.34, -6.82, -6.49 at 1/5/10%)
come from their finite-sample simulation and are a reference only.

    python src/two_break_unit_root.py check    # fast statistics equal plain OLS (seconds)

Inputs: <results>/final/timeseries/<run>.csv (through load_spread).
Outputs: none here; final_tests.py breaks writes final/two_break.csv (all 11 rows of
MODELS), final/two_break_critical_values.csv and final/tab_two_breaks.csv.
"""

import os
import sys

# One BLAS thread per process: the work is spread over a process pool, and letting
# every worker also start a full set of BLAS threads oversubscribes the CPU badly.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import lfilter  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

import paths  # noqa: E402

END = "2024-05-15"
TRIM, STEP, GAP_FRAC = 0.15, 21, 0.10
CV_REPS, CV_SEED = 600, 20261001     # random walks for the simulated critical values

# The rows of the thesis table, in order. The bootstrap seed of each run is tied to
# its position here (7_000_000 + 10_000 * position + draw).
MODELS = [
    ("rl_tuned__ma__reb63__tc0", "Tuned bandit (quarterly)"),
    ("window_fixedbest__ma__w126__reb63__tc0", "Fixed 0.5-year window (quarterly)"),
    ("window_fixedbest__ma__w252__reb63__tc0", "Fixed 1-year window (quarterly)"),
    ("window_fixedbest__ma__w378__reb63__tc0", "Fixed 1.5-year window (quarterly)"),
    ("window_fixedbest__ma__w504__reb63__tc0", "Fixed 2-year window (quarterly)"),
    ("window_fixedbest__ma__w756__reb63__tc0", "Fixed 3-year window (quarterly)"),
    ("ridge_trend_to__ma__w252__reb21__d1__tc0", "Turnover-penalised ridge (monthly)"),
    ("ridge_returns__ma__w252__reb21__tc0", "Return-space ridge (monthly)"),
    ("equal_weight__ma__reb21__tc0", "Equal weight (monthly)"),
    ("passive_reg28__ma__reb21__tc0", "Regulation 28 allocation"),
    ("passive_60_40__ma__reb21__tc0", "60/40 allocation"),
]


def aic_lag(y):
    return int(adfuller(y, regression="ct", autolag="AIC")[2])


def _bounds(T):
    return int(np.ceil(TRIM * T)), int(np.floor((1 - TRIM) * T)), int(np.ceil(GAP_FRAC * T))


def _search(T, score):
    """Grid search then local refinement over break pairs; score(b1s, b2s) -> t-ratios."""
    lo, hi, gap = _bounds(T)
    grid = np.arange(lo, hi + 1, STEP)
    g1, g2 = np.meshgrid(grid, grid, indexing="ij")
    keep = g2 - g1 >= gap
    b1s, b2s = g1[keep], g2[keep]
    t = score(b1s, b2s)
    i = int(np.argmin(t))
    c1, c2 = int(b1s[i]), int(b2s[i])
    r1 = np.arange(max(lo, c1 - STEP), min(hi, c1 + STEP) + 1)
    r2 = np.arange(max(lo, c2 - STEP), min(hi, c2 + STEP) + 1)
    g1, g2 = np.meshgrid(r1, r2, indexing="ij")
    keep = g2 - g1 >= gap
    t2 = score(g1[keep], g2[keep])
    cand_t = np.r_[t[i], t2]
    cand_1 = np.r_[c1, g1[keep]]
    cand_2 = np.r_[c2, g2[keep]]
    j = int(np.argmin(cand_t))
    return float(cand_t[j]), int(cand_1[j]), int(cand_2[j])


# ---------------------------------------------------------------- Lumsdaine-Papell
def _lp_design(y, k):
    y = np.asarray(y, dtype=float)
    T = len(y)
    dy = np.diff(y)
    idx = np.arange(k + 1, T)
    Y = dy[idx - 1]
    base = np.column_stack([np.ones(len(idx)), idx / T, y[idx - 1]] + [dy[idx - 1 - i] for i in range(1, k + 1)])
    return T, idx, Y, base


def _lp_pairs(Y, base, idx, T, b1s, b2s):
    """LP t-ratio on y_{t-1} for many pairs (b1 < b2) at once, from suffix sums."""
    n, q = base.shape
    rev = lambda a: np.cumsum(a[::-1], axis=0)[::-1]
    pad = lambda a: np.concatenate([a, np.zeros((1,) + a.shape[1:])])
    f = idx.astype(float)
    S_b, S_bi = pad(rev(base)), pad(rev(base * f[:, None]))
    S_y, S_yi = pad(rev(Y)), pad(rev(Y * f))
    S_1, S_i, S_ii = pad(rev(np.ones(n))), pad(rev(f)), pad(rev(f * f))
    b1s, b2s = np.asarray(b1s, float), np.asarray(b2s, float)
    j1 = np.searchsorted(idx, b1s, side="right")
    j2 = np.searchsorted(idx, b2s, side="right")
    m, p = len(b1s), q + 4
    XtX = np.zeros((m, p, p))
    XtX[:, :q, :q] = base.T @ base
    cols = [S_b[j1], (S_bi[j1] - b1s[:, None] * S_b[j1]) / T, S_b[j2], (S_bi[j2] - b2s[:, None] * S_b[j2]) / T]
    for c, v in enumerate(cols):
        XtX[:, :q, q + c] = v
        XtX[:, q + c, :q] = v
    c1, c2 = S_1[j1], S_1[j2]
    i1, i2, ii1, ii2 = S_i[j1], S_i[j2], S_ii[j1], S_ii[j2]
    B = np.zeros((m, 4, 4))
    B[:, 0, 0] = c1
    B[:, 0, 1] = (i1 - b1s * c1) / T
    B[:, 1, 1] = (ii1 - 2 * b1s * i1 + b1s ** 2 * c1) / T ** 2
    B[:, 2, 2] = c2
    B[:, 2, 3] = (i2 - b2s * c2) / T
    B[:, 3, 3] = (ii2 - 2 * b2s * i2 + b2s ** 2 * c2) / T ** 2
    B[:, 0, 2] = c2                                      # du1*du2 = du2 since b1 < b2
    B[:, 0, 3] = (i2 - b2s * c2) / T                     # du1*dt2 = dt2
    B[:, 1, 2] = (i2 - b1s * c2) / T                     # dt1*du2
    B[:, 1, 3] = (ii2 - (b1s + b2s) * i2 + b1s * b2s * c2) / T ** 2
    B = np.triu(B) + np.transpose(np.triu(B, 1), (0, 2, 1))
    XtX[:, q:, q:] = B
    XtY = np.zeros((m, p))
    XtY[:, :q] = base.T @ Y
    XtY[:, q:] = np.column_stack([S_y[j1], (S_yi[j1] - b1s * S_y[j1]) / T, S_y[j2], (S_yi[j2] - b2s * S_y[j2]) / T])
    rhs = np.stack([XtY, np.broadcast_to(np.eye(p)[2], (m, p))], axis=2)
    sol = np.linalg.solve(XtX, rhs)
    coef, inv22 = sol[:, :, 0], sol[:, 2, 1]
    rss = Y @ Y - np.einsum("ij,ij->i", coef, XtY)
    return coef[:, 2] / np.sqrt(rss / (n - p) * inv22)


def lp_stat(y, k):
    T, idx, Y, base = _lp_design(y, k)
    return _search(T, lambda b1s, b2s: _lp_pairs(Y, base, idx, T, b1s, b2s))


def lp_t_direct(y, k, b1, b2):
    """Plain OLS LP t-ratio for one pair (reference for checking _lp_pairs)."""
    T, idx, Y, base = _lp_design(y, k)
    du1, du2 = (idx > b1).astype(float), (idx > b2).astype(float)
    X = np.column_stack([base, du1, (idx - b1) * du1, du2, (idx - b2) * du2])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ coef
    inv = np.linalg.inv(X.T @ X)
    return coef[2] / np.sqrt(e @ e / (len(Y) - X.shape[1]) * inv[2, 2])


# ---------------------------------------------------------------- Lee-Strazicich
def ls_t(y, k, b1, b2):
    """Lee-Strazicich (2003) model C LM t-ratio for one break pair."""
    y = np.asarray(y, dtype=float)
    T = len(y)
    dy = np.diff(y)                                   # dy[t-1] = y_t - y_{t-1}, t = 1..T-1
    t = np.arange(1, T)
    D1, D2 = (t > b1).astype(float), (t > b2).astype(float)
    P1, P2 = (t == b1 + 1).astype(float), (t == b2 + 1).astype(float)
    dZ = np.column_stack([np.ones(T - 1), P1, P2, D1, D2])
    d = np.linalg.lstsq(dZ, dy, rcond=None)[0]
    r = dy - dZ @ d                                   # dS_t
    S = np.r_[0.0, np.cumsum(r)]                      # S_0 = 0, S_t for t = 0..T-1
    rows = np.arange(k + 1, T)                        # regression dates t
    Y = dy[rows - 1]
    X = np.column_stack([dZ[rows - 1], S[rows - 1]] + [r[rows - 1 - i] for i in range(1, k + 1)])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ coef
    inv = np.linalg.inv(X.T @ X)
    j = dZ.shape[1]
    return coef[j] / np.sqrt(e @ e / (len(Y) - X.shape[1]) * inv[j, j])


def ls_t_fast(y, dy, C, k, b1, b2):
    """ls_t by Frisch-Waugh: dZ spans indicators of the five regimes (before b1, the
    pulse at b1+1, between, the pulse at b2+1, after), so partialling it out is
    demeaning within the three regimes and dropping the two pulse rows."""
    T = len(y)
    mA = C[b1] / b1
    mB = (C[b2] - C[b1 + 1]) / (b2 - b1 - 1)
    mC = (C[T - 1] - C[b2 + 1]) / (T - 2 - b2)
    r = np.empty(T - 1)
    r[:b1] = dy[:b1] - mA
    r[b1] = 0.0
    r[b1 + 1:b2] = dy[b1 + 1:b2] - mB
    r[b2] = 0.0
    r[b2 + 1:] = dy[b2 + 1:] - mC
    S = np.r_[0.0, np.cumsum(r)]
    N = T - 1 - k
    M = np.empty((N, k + 2))
    M[:, 0] = dy[k:T - 1]
    M[:, 1] = S[k:T - 1]
    for i in range(1, k + 1):
        M[:, 1 + i] = r[k - i:T - 1 - i]
    segs = [(0, b1 - k), (b1 - k + 1, b2 - k), (b2 - k + 1, N)]
    W = np.concatenate([M[a:b] - M[a:b].mean(axis=0) for a, b in segs])
    Yt, Xt = W[:, 0], W[:, 1:]
    XtX = Xt.T @ Xt
    inv = np.linalg.inv(XtX)
    coef = inv @ (Xt.T @ Yt)
    e = Yt - Xt @ coef
    return coef[0] / np.sqrt(e @ e / (N - 5 - (k + 1)) * inv[0, 0])


def ls_stat(y, k):
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    C = np.r_[0.0, np.cumsum(dy)]
    return _search(len(y), lambda b1s, b2s: np.array([ls_t_fast(y, dy, C, k, int(a), int(b)) for a, b in zip(b1s, b2s)]))


# ---------------------------------------------------------------- simulation, bootstrap
def _both(y):
    k = aic_lag(y)
    return lp_stat(y, k)[0], ls_stat(y, k)[0]


def _sim(args):
    seed, T = args
    return _both(np.cumsum(np.random.default_rng(seed).standard_normal(T)))


def null_ar(y, k):
    """AR(k) with a constant fitted to dy under the unit-root null: (phi, residuals)."""
    dy = np.diff(np.asarray(y, dtype=float))
    Y = dy[k:]
    X = np.column_stack([np.ones(len(Y))] + [dy[k - i:len(dy) - i] for i in range(1, k + 1)])
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    return coef[1:], Y - X @ coef


def _boot(args):
    """Sieve wild bootstrap draw under the null; keeps the spread's volatility path."""
    seed, phi, resid, T = args
    e = resid * np.random.default_rng(seed).choice([-1.0, 1.0], size=len(resid))
    dy = lfilter([1.0], np.r_[1.0, -phi], np.r_[np.zeros(T - 1 - len(e)), e])
    return _both(np.r_[0.0, np.cumsum(dy)])


def winsorise_level(y, z=4.0):
    dy = np.diff(y)
    m, s = dy.mean(), dy.std()
    return np.r_[y[0], y[0] + np.cumsum(np.clip(dy, m - z * s, m + z * s))]


def load_spread(run, ts_dir=None):
    """Log spread of a saved run up to END (default folder: final/timeseries)."""
    ts_dir = paths.FINAL / "timeseries" if ts_dir is None else ts_dir
    df = pd.read_csv(os.path.join(ts_dir, f"{run}.csv"), parse_dates=["datestamp"]).set_index("datestamp").sort_index()
    df = df.loc[:END]
    return np.log(df["portfolio_tri"]) - np.log(df["benchmark_tri"])


def pval(draws, stat):
    return float(((draws <= stat).sum() + 1) / (len(draws) + 1))


def critical_values(T, pool, reps=CV_REPS):
    """Simulated 1, 5 and 10% critical values of LP and LS for spreads of length T.
    One row in the format of two_break_critical_values.csv."""
    sims = np.array(pool.map(_sim, [(CV_SEED + r, T) for r in range(reps)]))
    lp_sims, ls_sims = sims[:, 0], sims[:, 1]
    cv = {f"{name}_cv_{q:g}": float(np.quantile(d, q)) for name, d in (("lp", lp_sims), ("ls", ls_sims))
          for q in (0.01, 0.05, 0.10)}
    return {"T": T, "reps": reps, **cv}


def check():
    """The vectorised LP t-ratios and the Frisch-Waugh LS t-ratios equal plain OLS."""
    y = np.cumsum(np.random.default_rng(0).standard_normal(500))
    k = 3
    b1s, b2s = np.array([100, 150, 200]), np.array([300, 260, 400])
    T, idx, Y, base = _lp_design(y, k)
    lp_gap = float(np.max(np.abs(_lp_pairs(Y, base, idx, T, b1s, b2s)
                                 - np.array([lp_t_direct(y, k, a, b) for a, b in zip(b1s, b2s)]))))
    dy = np.diff(y)
    C = np.r_[0.0, np.cumsum(dy)]
    ls_gap = max(abs(ls_t_fast(y, dy, C, k, a, b) - ls_t(y, k, a, b)) for a, b in zip(b1s, b2s))
    ok = max(lp_gap, ls_gap) < 1e-8
    print(f"LP fast vs OLS max |diff| {lp_gap:.1e}; LS fast vs OLS max |diff| {ls_gap:.1e}: "
          f"{'CHECK PASSED' if ok else 'CHECK FAILED'}")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] != ["check"]:
        sys.exit("usage: python src/two_break_unit_root.py check")
    sys.exit(0 if check() else 1)
