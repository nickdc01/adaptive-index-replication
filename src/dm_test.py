"""
Diebold-Mariano test for pairwise tracking-error comparisons, and the family
of model pairs it is applied to.

Library module (no command line), used by inference.py and final_tests.py
for Tables tab:dm, tab:dm_variants and tab:dm_h1.

For two models with daily active returns e^(1)_t = r_p1_t - r_b_t and
e^(2)_t = r_p2_t - r_b_t, tests H0: E[d_t] = 0 against H1: E[d_t] != 0,
where d_t = (e^(1)_t)^2 - (e^(2)_t)^2.

A negative DM statistic means model 1 has the *smaller* squared tracking
error on average, i.e. it tracks more tightly.

HAC variance via Newey-West with bandwidth = floor(T^(1/3)).
Small-sample correction per Harvey, Leybourne, Newbold (1997) with
h=1 (one-step "forecasts": daily returns).

Runs are read from TS_DIR (default paths.FINAL / "timeseries"); a caller may
set dm_test.TS_DIR or pass ts_dir to load_active_returns.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import paths

TS_DIR = paths.FINAL / "timeseries"


def load_active_returns(run_name: str, ts_dir=None) -> pd.Series:
    """Daily active log returns of a run, from <ts_dir or TS_DIR>/<run>.csv."""
    folder = Path(TS_DIR if ts_dir is None else ts_dir)
    df = pd.read_csv(folder / f"{run_name}.csv", parse_dates=["datestamp"])
    df = df.set_index("datestamp").sort_index()
    log_p = np.log(df["portfolio_tri"])
    log_b = np.log(df["benchmark_tri"])
    active = log_p.diff() - log_b.diff()
    return active.dropna()


def newey_west_var(d: np.ndarray, lag: int) -> float:
    """Newey-West HAC long-run variance of d_t."""
    T = len(d)
    d_centered = d - d.mean()
    gamma0 = np.dot(d_centered, d_centered) / T
    lrv = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        gamma_k = np.dot(d_centered[k:], d_centered[:-k]) / T
        lrv += 2.0 * w * gamma_k
    return lrv


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1):
    """
    Squared-loss Diebold-Mariano with HLN small-sample correction.

    Returns
    -------
    dict with mean_diff, dm_stat, dm_stat_hln, pvalue_two_sided, T, lag
    """
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    lag = max(1, int(np.floor(T ** (1.0 / 3.0))))
    lrv = newey_west_var(d, lag)
    if lrv <= 0:
        return dict(
            mean_diff=float(d.mean()), dm_stat=np.nan, dm_stat_hln=np.nan,
            pvalue_two_sided=np.nan, T=T, lag=lag,
        )
    dm = d.mean() / np.sqrt(lrv / T)
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * hln_factor
    pval = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_hln), df=T - 1))
    return dict(
        mean_diff=float(d.mean()), dm_stat=float(dm), dm_stat_hln=float(dm_hln),
        pvalue_two_sided=float(pval), T=int(T), lag=int(lag),
    )


# The pairs tested, as (model_1, model_2, label); a negative DM statistic
# means model_1 tracks more tightly. Together with the two extra pairs in
# inference.DEFAULT_PAIRS they form the 30-pair family over which the
# Benjamini-Hochberg correction is applied (inference.run_inference_panel).
# Changing this list changes the adjusted p-values.
_TUNED = "rl_tuned__ma__reb63__tc0"          # the tuned bandit (quarterly)
_PRIOR = "rl_tuned__ma__prior_reb21__tc0"    # earlier bandit settings (kappa 1.2, gamma 1, c_e 504, monthly)
PAIRS = [
    # tuned bandit against each comparator
    (_TUNED, "ridge_trend_to__ma__w252__reb21__d1__tc0", "tuned bandit vs turnover-penalised ridge"),
    (_TUNED, "ridge_returns__ma__w252__reb21__tc0",      "tuned bandit vs return-space ridge"),
    (_TUNED, "equal_weight__ma__reb21__tc0",             "tuned bandit vs equal weight"),
    (_TUNED, "passive_reg28__ma__reb21__tc0",            "tuned bandit vs Regulation 28 allocation"),
    (_TUNED, "passive_60_40__ma__reb21__tc0",            "tuned bandit vs 60/40 allocation"),
    (_TUNED, _PRIOR,                                     "tuned bandit vs earlier bandit settings (monthly)"),
    (_TUNED, "rl_bandit_to__ma__tc0",                    "tuned bandit vs tracking-reward bandit"),
    (_TUNED, "window_volheuristic__ma__tc0",             "tuned bandit vs volatility heuristic"),
    (_TUNED, "window_volheuristic__ma__reb63__tc0",      "tuned bandit vs volatility heuristic (quarterly)"),
    # tuned bandit against the fixed windows and the oracle, monthly
    (_TUNED, "window_fixedbest__ma__w252__tc0",          "tuned bandit vs fixed 1-year window"),
    (_TUNED, "window_fixedbest__ma__w378__tc0",          "tuned bandit vs fixed 1.5-year window"),
    (_TUNED, "window_fixedbest__ma__w504__tc0",          "tuned bandit vs fixed 2-year window"),
    (_TUNED, "window_fixedbest__ma__w756__tc0",          "tuned bandit vs fixed 3-year window"),
    (_TUNED, "window_oracle__ma__tc0",                   "tuned bandit vs look-ahead oracle"),
    # the same, quarterly
    (_TUNED, "window_fixedbest__ma__w126__reb63__tc0",   "tuned bandit vs fixed 0.5-year window (quarterly)"),
    (_TUNED, "window_fixedbest__ma__w252__reb63__tc0",   "tuned bandit vs fixed 1-year window (quarterly)"),
    (_TUNED, "window_fixedbest__ma__w378__reb63__tc0",   "tuned bandit vs fixed 1.5-year window (quarterly)"),
    (_TUNED, "window_fixedbest__ma__w504__reb63__tc0",   "tuned bandit vs fixed 2-year window (quarterly)"),
    (_TUNED, "window_fixedbest__ma__w756__reb63__tc0",   "tuned bandit vs fixed 3-year window (quarterly)"),
    (_TUNED, "window_oracle__ma__reb63__tc0",            "tuned bandit vs look-ahead oracle (quarterly)"),
    # other contrasts
    ("rl_bandit_to__ma__tc0",         "ridge_trend_to__ma__w252__reb21__d1__tc0", "tracking-reward bandit vs turnover-penalised ridge"),
    ("rl_bandit_to__ma__tc0",         "ridge_returns__ma__w252__reb21__tc0",      "tracking-reward bandit vs return-space ridge"),
    ("rl_bandit_to__ma__tc0",         "equal_weight__ma__reb21__tc0",             "tracking-reward bandit vs equal weight"),
    ("rl_bandit_to__ma__tc0",         "rl_bandit__ma__tc0",                       "tracking-reward bandit with vs without turnover penalty"),
    ("ridge_trend_to__ma__w252__reb21__d1__tc0", "ridge_trend__ma__w252__reb21__tc0", "turnover penalty vs none (log-TRI ridge with trend)"),
    ("ridge_returns__ma__w252__reb21__tc0",      "equal_weight__ma__reb21__tc0",      "return-space ridge vs equal weight"),
    (_PRIOR, "rl_bandit_to__ma__tc0",                    "earlier bandit settings vs tracking-reward bandit"),
    (_PRIOR, "equal_weight__ma__reb21__tc0",             "earlier bandit settings vs equal weight"),
]
