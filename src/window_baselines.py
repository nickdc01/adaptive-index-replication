"""
Window-choosing controls that do not learn: a random window, a volatility
rule and a look-ahead oracle.

Library module (no command line), used through run_models.py (models
"window_random", "window_volheuristic", "window_oracle") by final_runs.py.
They feed the random-window and look-ahead-oracle rows of Table tab:variants
and pairs in the Diebold-Mariano family (dm_test.py).

Each control uses the bandit's inner tracker
(rl_window_selection.fit_drift_tri_elasticnet) and the same walk-forward
schedule as rl_window_selection.run_bandit_window_selection, so it differs from
the bandit only in how the window is chosen. Each returns the same
(weights_reb, chosen_windows, diagnostics) triple.

    random_window         a window drawn at random at each rebalance (seeded)
    vol_heuristic_window  a volatility heuristic: shorter windows when trailing
                          benchmark volatility is high
    oracle_window         the window with the lowest forward tracking error, chosen
                          with hindsight: a ceiling, not an implementable strategy
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

# Imported by name so a caller can replace the inner fit (final_runs.py does).
from rl_window_selection import fit_drift_tri_elasticnet


# -----------------------------
# Shared walk-forward harness
# -----------------------------
def _rebalance_indices(n_obs, max_w, forward_days, rebalance_freq):
    """The rebalance schedule of run_bandit_window_selection (forward_days horizon)."""
    start_i = int(max_w)
    end_i = n_obs - forward_days - 1
    return list(range(start_i, end_i, rebalance_freq))


def _forward_mse(asset_returns, bench_returns, i, w, forward_days):
    """
    Mean squared active daily return of weights w over [i, i + forward_days),
    clamped to the sample: the MSE term of the bandit's reward.
    """
    avail_end = min(i + forward_days, len(asset_returns))
    if avail_end <= i:
        return np.inf
    X_test = asset_returns.iloc[i:avail_end].values
    y_test = bench_returns.iloc[i:avail_end].values
    port_ret = X_test @ w
    return float(np.mean((port_ret - y_test) ** 2))


def _run_window_selector(
    asset_returns: pd.DataFrame,
    bench_returns: pd.Series,
    window_choices,
    select_window,
    rebalance_freq=21,
    forward_days=63,
    delta_turnover=0.0,
    data_lag=0,
    desc="Window Baseline",
):
    """
    Walk-forward runner for a window rule that does not learn.

    As run_bandit_window_selection except for the rule that chooses the
    window: same schedule, fit window [fit_end - window, fit_end), data_lag
    convention, equal-weight fallback when the fit fails, and output shapes.

    Parameters
    ----------
    select_window : callable(step_k, i, fit_end) -> int
        The lookback window (days) for the rebalance at index i. fit_end =
        i - data_lag is the exclusive end of the data the rule and the fit may
        use.

    Returns
    -------
    weights_reb    : pd.DataFrame  (rebalance dates x assets)
    chosen_windows : pd.Series     (rebalance dates -> chosen window), name "chosen_window"
    diagnostics    : pd.DataFrame  (rebalance dates -> "forward_mse")
    """
    idx = asset_returns.index.intersection(bench_returns.index)
    asset_returns = asset_returns.loc[idx].sort_index()
    bench_returns = bench_returns.loc[idx].sort_index()

    asset_tri = (1 + asset_returns).cumprod()
    bench_tri = (1 + bench_returns).cumprod()

    max_w = int(max(window_choices))
    rebalance_is = _rebalance_indices(len(asset_returns), max_w, forward_days, rebalance_freq)

    n_assets = asset_returns.shape[1]
    chosen_windows = []
    chosen_dates = []
    chosen_weights = []
    fwd_mses = []

    prev_w = None
    for step_k, i in enumerate(tqdm(rebalance_is, desc=desc)):
        date_t = asset_returns.index[i]
        fit_end = i - data_lag

        window = int(select_window(step_k, i, fit_end))

        Xw_tri = asset_tri.iloc[fit_end - window:fit_end].values
        yw_tri = bench_tri.iloc[fit_end - window:fit_end].values
        w, _trend = fit_drift_tri_elasticnet(
            Xw_tri, yw_tri,
            delta_turnover=delta_turnover,
            w_prev=prev_w,
        )
        if w is None:
            w = np.ones(n_assets) / n_assets

        fwd_mses.append(_forward_mse(asset_returns, bench_returns, i, w, forward_days))
        chosen_windows.append(window)
        chosen_dates.append(date_t)
        chosen_weights.append(w)
        prev_w = w

    weights_reb = pd.DataFrame(
        chosen_weights,
        index=pd.DatetimeIndex(chosen_dates),
        columns=asset_returns.columns,
    )
    chosen_windows = pd.Series(
        chosen_windows,
        index=pd.DatetimeIndex(chosen_dates),
        name="chosen_window",
    )
    diagnostics = pd.DataFrame(
        {"forward_mse": fwd_mses},
        index=pd.DatetimeIndex(chosen_dates),
    )
    return weights_reb, chosen_windows, diagnostics


# -----------------------------
# 1) Random window
# -----------------------------
def random_window(
    asset_returns: pd.DataFrame,
    bench_returns: pd.Series,
    window_choices=(126, 252, 378, 504, 756),
    rebalance_freq=21,
    forward_days=63,
    delta_turnover=0.0,
    data_lag=0,
    seed=0,
):
    """
    A window drawn uniformly from window_choices at each rebalance, from a
    local generator seeded with `seed`, so a given seed always gives the same
    sequence. The thesis averages seeds 0-19 (quarterly).
    """
    rng = np.random.default_rng(seed)
    choices = np.asarray(window_choices, dtype=int)

    def select(step_k, i, fit_end):
        return int(rng.choice(choices))

    return _run_window_selector(
        asset_returns, bench_returns, window_choices, select,
        rebalance_freq=rebalance_freq, forward_days=forward_days,
        delta_turnover=delta_turnover, data_lag=data_lag,
        desc="Random Window",
    )


# -----------------------------
# 2) Volatility rule (shorter windows when volatility is high, no learning)
# -----------------------------
def vol_heuristic_window(
    asset_returns: pd.DataFrame,
    bench_returns: pd.Series,
    window_choices=(126, 252, 378, 504, 756),
    rebalance_freq=21,
    forward_days=63,
    delta_turnover=0.0,
    data_lag=0,
    vol_lookback=63,
):
    """
    A fixed rule with no learning: shorter windows when volatility is high.

    At each rebalance the trailing benchmark volatility over the previous
    vol_lookback (default 63) days, using only data before fit_end = i -
    data_lag (the volatility the bandit sees as a context feature), is ranked
    as a percentile of all trailing volatilities observed up to that date. The
    percentile is mapped to a window:

        vol percentile in (0.8, 1.0]  (top quintile, most turbulent) -> shortest
        vol percentile in (0.6, 0.8]                                  -> 2nd shortest
        vol percentile in (0.4, 0.6]  (middle)                        -> middle
        vol percentile in (0.2, 0.4]                                  -> 2nd longest
        vol percentile in [0.0, 0.2]  (bottom quintile, calmest)      -> longest

    Nothing is tuned: vol_lookback, the quintile cut points and the map are fixed.
    """
    # Windows from shortest to longest; index 0 is used in the top volatility quintile.
    ranked = [int(w) for w in sorted(window_choices)]
    n_w = len(ranked)

    # Align as the runner does, so positions match its index.
    idx = asset_returns.index.intersection(bench_returns.index)
    bench_al = bench_returns.loc[idx].sort_index()

    # Trailing benchmark volatility; the value at position p uses returns up to
    # and including p, and only positions before fit_end are read below.
    trail_vol = bench_al.rolling(int(vol_lookback)).std()
    vol_vals = trail_vol.to_numpy()

    def select(step_k, i, fit_end):
        # The latest trailing volatility known at the decision ends at fit_end - 1.
        cur_pos = fit_end - 1
        if cur_pos < 0:
            return ranked[n_w // 2]  # not enough history -> middle window
        cur_vol = vol_vals[cur_pos]
        # Percentile of cur_vol among all trailing volatilities up to cur_pos.
        hist = vol_vals[: cur_pos + 1]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(cur_vol) or hist.size < 2:
            return ranked[n_w // 2]
        # Fraction of past observations <= current vol => percentile in [0, 1].
        pct = float(np.mean(hist <= cur_vol))
        # Map percentile -> rank: high vol (pct near 1) -> shortest (rank 0).
        # bucket b = floor((1 - pct) * n_w), clamped to [0, n_w - 1].
        b = int(np.floor((1.0 - pct) * n_w))
        b = max(0, min(n_w - 1, b))
        return ranked[b]

    return _run_window_selector(
        asset_returns, bench_returns, window_choices, select,
        rebalance_freq=rebalance_freq, forward_days=forward_days,
        delta_turnover=delta_turnover, data_lag=data_lag,
        desc="Vol-Heuristic Window",
    )


# -----------------------------
# 3) Look-ahead oracle (not implementable)
# -----------------------------
def oracle_window(
    asset_returns: pd.DataFrame,
    bench_returns: pd.Series,
    window_choices=(126, 252, 378, 504, 756),
    rebalance_freq=21,
    forward_days=63,
    delta_turnover=0.0,
    data_lag=0,
):
    """
    Look-ahead oracle (tracking error). Not an implementable strategy.

    At each rebalance it fits weights for every candidate window and keeps the
    window whose weights have the lowest MSE of active returns over the next
    forward_days, which uses future returns. It is a ceiling for window
    choice. The diagnostics column "forward_mse" is that lowest forward MSE.
    """
    idx = asset_returns.index.intersection(bench_returns.index)
    asset_returns = asset_returns.loc[idx].sort_index()
    bench_returns = bench_returns.loc[idx].sort_index()

    asset_tri = (1 + asset_returns).cumprod()
    bench_tri = (1 + bench_returns).cumprod()
    choices = [int(w) for w in window_choices]

    # The choice needs a fit per candidate window, so this repeats the loop of
    # _run_window_selector and keeps the winning fit's weights.
    max_w = int(max(window_choices))
    rebalance_is = _rebalance_indices(len(asset_returns), max_w, forward_days, rebalance_freq)
    n_assets = asset_returns.shape[1]

    chosen_windows, chosen_dates, chosen_weights, fwd_mses = [], [], [], []
    prev_w = None
    for step_k, i in enumerate(tqdm(rebalance_is, desc="Oracle Window (look-ahead)")):
        date_t = asset_returns.index[i]
        fit_end = i - data_lag

        best_mse = np.inf
        best_w = None
        best_window = choices[0]
        for window in choices:
            Xw_tri = asset_tri.iloc[fit_end - window:fit_end].values
            yw_tri = bench_tri.iloc[fit_end - window:fit_end].values
            w, _trend = fit_drift_tri_elasticnet(
                Xw_tri, yw_tri,
                delta_turnover=delta_turnover,
                w_prev=prev_w,
            )
            if w is None:
                continue
            # Look-ahead: scored on the next forward_days.
            mse = _forward_mse(asset_returns, bench_returns, i, w, forward_days)
            if mse < best_mse:
                best_mse = mse
                best_w = w
                best_window = window

        if best_w is None:
            best_w = np.ones(n_assets) / n_assets
            best_mse = _forward_mse(asset_returns, bench_returns, i, best_w, forward_days)

        chosen_windows.append(int(best_window))
        chosen_dates.append(date_t)
        chosen_weights.append(best_w)
        fwd_mses.append(best_mse)
        prev_w = best_w

    weights_reb = pd.DataFrame(
        chosen_weights,
        index=pd.DatetimeIndex(chosen_dates),
        columns=asset_returns.columns,
    )
    chosen_windows = pd.Series(
        chosen_windows,
        index=pd.DatetimeIndex(chosen_dates),
        name="chosen_window",
    )
    diagnostics = pd.DataFrame(
        {"forward_mse": fwd_mses},
        index=pd.DatetimeIndex(chosen_dates),
    )
    return weights_reb, chosen_windows, diagnostics
