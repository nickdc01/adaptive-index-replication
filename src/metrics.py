"""
Portfolio simulation and the tracking metrics used throughout the thesis.

Library module (no command line). Every model's weights go through
simulate_with_rebased_tri, and every reported tracking error, fee drag,
turnover, ADF p-value and half-life is computed here:

    simulate_with_rebased_tri  rebalanced portfolio returns, TRI, drifting weights, turnover
    compute_metrics            full-sample metrics of a run (runs/<run>.csv)
    _metrics_from_compare      the same metrics on any window (period_tables.py, run_tuning.py)
    complete_years, turnover_complete_years
                               turnover as the mean over complete calendar years
    half_life                  mean-reversion half-life of the log spread
    bootstrap_ci, hac_slope_ci, run_bootstrap_all
                               confidence intervals for tracking error and fee drag
                               (bootstrap_ci.csv, Table tab:ci)
    build_timeseries, save_run, save_timeseries, save_weights_to_csv
                               the per-run files under a results folder

Outputs are written under the results_dir passed in (default paths.FINAL).
"""

import collections
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller

import paths

# Count of solver failures, by kind; incremented in the except blocks of the model code.
solver_failures = collections.Counter()


# ─────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────

def simulate_with_rebased_tri(
    asset_returns: pd.DataFrame,
    weights_reb: pd.DataFrame,
    initial_value: float = 1.0,
    tc_bps: float = 0.0,
):
    """
    Simulate a rebalanced portfolio.

    weights_reb holds the target weights, one row per rebalance date. Between
    rebalances the weights drift with returns. Turnover is one-way,
    sum_j |w_new,j - w_end,j|, and a cost of tc_bps basis points is charged on
    it at each rebalance.

    Returns
    -------
    portfolio_returns : pd.Series
    portfolio_tri    : pd.Series
    eff_w            : pd.DataFrame  (drifting effective weights)
    turnover         : pd.Series     (non-zero entries only, one-way)
    rebalanced_on    : pd.Series[bool]
    """
    r = asset_returns.sort_index().astype(float)
    tri = (1.0 + r).cumprod()

    W = weights_reb.sort_index().astype(float)
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    assets = [a for a in W.columns if a in tri.columns]
    if not assets:
        raise ValueError("No overlapping assets between returns and weights.")
    tri = tri[assets]
    W = W[assets]

    seg_starts = [d for d in W.index if d in tri.index]
    if not seg_starts:
        raise ValueError("No overlap between rebalance dates and return index.")

    full_idx = tri.index[tri.index >= seg_starts[0]]
    V = pd.Series(index=full_idx, dtype=float)
    eff_w = pd.DataFrame(index=full_idx, columns=assets, dtype=float)
    turnover = pd.Series(0.0, index=full_idx)
    rebalanced_on = pd.Series(False, index=full_idx)

    baseV = initial_value

    for i, d0 in enumerate(seg_starts):
        d1 = seg_starts[i + 1] if i + 1 < len(seg_starts) else None

        if d1 is None:
            seg_idx = tri.index[tri.index >= d0]
        else:
            seg_idx = tri.index[(tri.index >= d0) & (tri.index < d1)]
        if len(seg_idx) == 0:
            continue

        # Weights decided at the close of the previous trading day earn the
        # close-to-close return on the rebalance day itself. The first segment
        # has no prior close, so it starts at d0 (zero return on day zero).
        pos_d0 = tri.index.get_loc(d0)
        base_tri = tri.iloc[pos_d0 - 1, :] if pos_d0 > 0 else tri.loc[d0, :]

        M = tri.loc[seg_idx, :].div(base_tri)
        w0 = W.loc[d0, :].values
        port_mult = (M * w0).sum(axis=1)

        V.loc[seg_idx] = baseV * port_mult.values
        eff = (M.mul(w0, axis=1)).div(port_mult, axis=0)
        eff_w.loc[seg_idx, :] = eff.values

        if d1 is not None:
            d_end = seg_idx[-1]
            w_end = eff_w.loc[d_end, :].values
            w_next = W.loc[d1, :].values
            l1 = float(np.abs(w_next - w_end).sum())
            turnover.loc[d1] = l1                          # one-way
            cost = (tc_bps / 1e4) * l1 * float(V.loc[d_end])
            baseV = float(V.loc[d_end]) - cost
            rebalanced_on.loc[d1] = True
        else:
            rebalanced_on.loc[d0] = True

    portfolio_returns = V.pct_change().fillna(0.0)
    portfolio_tri = (1.0 + portfolio_returns).cumprod()

    return (
        portfolio_returns,
        portfolio_tri,
        eff_w.fillna(0.0),
        turnover[turnover != 0],
        rebalanced_on,
    )


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    p_ret: pd.Series,
    p_tri: pd.Series,
    bench_returns: pd.Series,
    turn: pd.Series,
    eval_start=None,
    eval_end=None,
) -> dict:
    """
    Tracking metrics of a simulated portfolio (outputs of simulate_with_rebased_tri)
    against the benchmark.

    eval_start, eval_end : optional dates. If given, the benchmark and portfolio
        TRIs are cut to [eval_start, eval_end] before any metric is computed
        (run_tuning.segment uses this for the training, test and full periods).
        By default the whole sample is used.

    Returns
    -------
    dict with keys:
        annualised_te          annualised tracking error (std of active daily returns)
        r2_monthly             R² of rolling-21-day returns vs benchmark
        beta_monthly           OLS beta of rolling-21-day returns vs benchmark
        mae_tri                mean absolute error on TRI levels
        mse_tri                mean squared error on TRI levels
        correlation_daily      Pearson correlation of daily returns
        directional_accuracy   fraction of days portfolio direction == benchmark direction
        avg_yearly_turnover    average one-way turnover per calendar year
        avg_turn_per_reb       average one-way turnover per rebalance event
        ann_turnover           total one-way turnover / years in sample
        adf_stat               ADF statistic on log(portfolio TRI / benchmark TRI)
        adf_pval               ADF p-value (above)
        fee_drag_pa            implied annual fee drag from log-spread time trend
    """
    # Align benchmark to portfolio period
    bench_al = bench_returns.reindex(p_ret.index).fillna(0.0)
    bench_tri = (1 + bench_al).cumprod()

    df_compare = pd.concat(
        [bench_tri.rename("Benchmark"), p_tri.rename("Tracking Portfolio")],
        axis=1,
    ).dropna()

    # Optional window: only changes the sample the metrics are measured on.
    if eval_start is not None or eval_end is not None:
        df_compare = df_compare.loc[eval_start:eval_end]
        if len(turn) > 0:
            turn = turn.loc[
                (turn.index >= df_compare.index[0])
                & (turn.index <= df_compare.index[-1])
            ] if len(df_compare) else turn.iloc[0:0]

    return _metrics_from_compare(df_compare, turn)


def _metrics_from_compare(df_compare: pd.DataFrame, turn: pd.Series) -> dict:
    """
    The metrics of compute_metrics from a TRI frame with columns 'Benchmark'
    and 'Tracking Portfolio' and a one-way turnover Series. Shared by
    compute_metrics, period_tables.py and run_tuning.py, so every table uses
    the same formulas.
    """
    daily = df_compare.pct_change().dropna()
    active = daily["Tracking Portfolio"] - daily["Benchmark"]

    # ── Tracking error ──────────────────────────────────────
    te_ann = float(active.std() * np.sqrt(252))

    # ── Monthly R² and beta ─────────────────────────────────
    monthly = df_compare.pct_change(21).dropna()
    X_reg = sm.add_constant(monthly["Benchmark"])
    ols_m = sm.OLS(monthly["Tracking Portfolio"], X_reg).fit()
    beta_m = float(ols_m.params.iloc[1])
    r2_m = float(r2_score(monthly["Benchmark"].values, monthly["Tracking Portfolio"].values))

    # ── MAE / MSE on TRI levels ─────────────────────────────
    mae_tri = float(mean_absolute_error(
        df_compare["Benchmark"].values, df_compare["Tracking Portfolio"].values
    ))
    mse_tri = float(mean_squared_error(
        df_compare["Benchmark"].values, df_compare["Tracking Portfolio"].values
    ))

    # ── Daily correlation and directional accuracy ──────────
    corr = float(np.corrcoef(
        daily["Tracking Portfolio"].values, daily["Benchmark"].values
    )[0, 1])
    dir_acc = float((
        (daily["Tracking Portfolio"] > 0).astype(int)
        == (daily["Benchmark"] > 0).astype(int)
    ).mean())

    # ── ADF on log spread (constant + trend) ────────────────
    log_spread = (
        np.log(df_compare["Tracking Portfolio"])
        - np.log(df_compare["Benchmark"])
    ).dropna()
    adf = adfuller(log_spread, regression="ct")
    adf_stat = float(adf[0])
    adf_pval = float(adf[1])

    # ── Fee drag (annualised drift of log spread) ────────────
    df_fee = pd.DataFrame({"log_diff": log_spread})
    df_fee["t"] = np.arange(len(df_fee))
    X_fee = sm.add_constant(df_fee["t"])
    ols_fee = sm.OLS(df_fee["log_diff"], X_fee).fit()
    beta_t = float(ols_fee.params.iloc[1])
    fee_drag = float(1 - np.exp(252 * beta_t))

    # ── Turnover ─────────────────────────────────────────────
    n_years = len(df_compare) / 252
    if len(turn) > 0:
        avg_yearly_turn = float(turn.resample("Y").sum().mean())
        avg_turn_per_reb = float(turn.mean())
        ann_turnover = float(turn.sum() / n_years) if n_years > 0 else 0.0
    else:
        avg_yearly_turn = avg_turn_per_reb = ann_turnover = 0.0

    return {
        "annualised_te": te_ann,
        "r2_monthly": r2_m,
        "beta_monthly": beta_m,
        "mae_tri": mae_tri,
        "mse_tri": mse_tri,
        "correlation_daily": corr,
        "directional_accuracy": dir_acc,
        "avg_yearly_turnover": avg_yearly_turn,
        "avg_turn_per_reb": avg_turn_per_reb,
        "ann_turnover": ann_turnover,
        "adf_stat": adf_stat,
        "adf_pval": adf_pval,
        "fee_drag_pa": fee_drag,
    }


# ─────────────────────────────────────────────────────────────
# Saved runs and turnover over complete calendar years
# ─────────────────────────────────────────────────────────────

def _load_compare_from_timeseries(ts_path: str) -> pd.DataFrame:
    """
    A saved timeseries CSV (save_timeseries format: datestamp index,
    benchmark_tri, portfolio_tri) as a TRI frame with the column names of
    compute_metrics ('Benchmark', 'Tracking Portfolio').
    """
    df = pd.read_csv(ts_path, index_col=0, parse_dates=True).sort_index()
    out = pd.DataFrame(
        {
            "Benchmark": df["benchmark_tri"],
            "Tracking Portfolio": df["portfolio_tri"],
        }
    ).dropna()
    return out


def complete_years(start, end) -> list:
    """
    Calendar years wholly inside [start, end]. A year counts when the window
    opens on or before its first weekday and closes on or after its last
    weekday, so 2016-12-30 (a Friday) completes 2016 and 2017-01-01 opens
    2017. The full period 2009-01-28 -> 2024-05-15 gives 2010-2023.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    years = []
    for y in range(start.year, end.year + 1):
        days = pd.bdate_range(f"{y}-01-02", f"{y}-12-31")   # 1 January is never a trading day
        if start <= days[0] and end >= days[-1]:
            years.append(y)
    return years


def load_run_turnover(run_name: str, results_dir=paths.FINAL):
    """
    A run's saved one-way turnover (<results_dir>/turnover/<run>.csv, columns
    datestamp, turnover) as a Series, or None when the file does not exist.
    """
    path = os.path.join(results_dir, "turnover", f"{run_name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "turnover" not in df.columns or "datestamp" not in df.columns:
        return None
    return pd.Series(df["turnover"].astype(float).values,
                     index=pd.DatetimeIndex(pd.to_datetime(df["datestamp"])),
                     name="turnover").sort_index()


def turnover_complete_years(turn: pd.Series, years) -> float:
    """
    Mean one-way turnover over the given calendar years, years with no
    rebalance counting as zero. This is the turnover reported in the thesis.
    """
    years = list(years)
    if turn is None or not years:
        return float("nan")
    if len(turn) == 0:
        return 0.0
    by_year = turn.groupby(turn.index.year).sum()
    return float(by_year.reindex(years, fill_value=0.0).mean())


# ─────────────────────────────────────────────────────────────
# Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    ts_path: str,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Confidence intervals for tracking error and fee drag from a saved
    timeseries CSV (save_timeseries format).

    Tracking error: moving-block bootstrap of the daily active returns,
    reporting the (1-ci)/2 and (1+ci)/2 percentiles. Fee drag: HAC interval
    on the slope of the log spread (hac_slope_ci).

    Parameters
    ----------
    ts_path  : path to a timeseries CSV with benchmark_tri, portfolio_tri
    n_boot   : number of bootstrap resamples (default 1000)
    ci       : confidence level (default 0.95 -> 95% CI)
    seed     : random seed for reproducibility

    Returns
    -------
    dict with keys:
        te_point, te_lo, te_hi,
        fee_drag_point, fee_drag_lo, fee_drag_hi
    """
    df = pd.read_csv(ts_path, index_col=0, parse_dates=True)
    daily = df[["benchmark_tri", "portfolio_tri"]].pct_change().dropna()
    active = (daily["portfolio_tri"] - daily["benchmark_tri"]).values
    n = len(active)

    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2

    # ── Moving-block bootstrap for TE ────────────────────────
    # Iid resampling destroys serial correlation in daily active returns and
    # biases the TE standard error downward. Block length ~ n^(1/3) preserves
    # autocorrelation structure. See Politis & Romano (1994).
    block = int(max(5, round(n ** (1 / 3))))
    n_blocks = int(np.ceil(n / block))

    te_boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        sample = active[idx]
        try:
            te_boots[b] = float(np.std(sample, ddof=1) * np.sqrt(252))
        except (ValueError, FloatingPointError):
            solver_failures["bootstrap_te"] += 1
            te_boots[b] = np.nan

    te_point = float(np.std(active, ddof=1) * np.sqrt(252))

    # ── Fee-drag CI via HAC (Newey-West) on the OLS time slope ───
    # Resampling returns would destroy the drift being measured, so the
    # interval comes from the OLS slope of the log spread on time with a
    # Newey-West HAC variance (bandwidth in hac_slope_ci).
    bench_tri = df["benchmark_tri"].dropna()
    port_tri  = df["portfolio_tri"].dropna()
    fd_point, fd_lo, fd_hi = hac_slope_ci(
        np.log(port_tri.values) - np.log(bench_tri.values),
        ci=ci,
    )

    te_boots_clean = te_boots[~np.isnan(te_boots)]
    if len(te_boots_clean) == 0:
        te_lo = te_hi = te_point
    else:
        te_lo = float(np.percentile(te_boots_clean, alpha * 100))
        te_hi = float(np.percentile(te_boots_clean, (1 - alpha) * 100))

    return {
        "te_point":      te_point,
        "te_lo":         te_lo,
        "te_hi":         te_hi,
        "fee_drag_point": fd_point,
        "fee_drag_lo":   fd_lo,
        "fee_drag_hi":   fd_hi,
    }


def hac_slope_ci(log_spread, ci: float = 0.95):
    """
    OLS of log_spread on time t with HAC (Newey-West) covariance.
    Returns (fee_drag_point, fee_drag_lo, fee_drag_hi) where fee_drag = 1 - exp(252*slope).
    maxlags is the Andrews (1991) AR(1) plug-in Bartlett bandwidth. The log spread
    is very persistent (AR(1) rho ~ 0.98), and a fixed-lag rule such as Greene's
    4*(n/100)**(2/9) would understate the HAC standard error and give intervals
    that are too narrow.
    """
    log_spread = np.asarray(log_spread, dtype=float)
    n = len(log_spread)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    t = np.arange(n, dtype=float)
    X = sm.add_constant(t)
    # Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel, computed from the
    # detrended-residual first-order autocorrelation.
    _resid = sm.OLS(log_spread, X).fit().resid
    _rho = float(np.clip(sm.OLS(_resid[1:], _resid[:-1]).fit().params[0], -0.999, 0.999))
    _a1 = 4 * _rho ** 2 / ((1 - _rho) ** 2 * (1 + _rho) ** 2)
    maxlags = int(max(1, round(1.1447 * (_a1 * n) ** (1 / 3))))
    ols = sm.OLS(log_spread, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    beta_t = float(ols.params[1])
    se = float(ols.bse[1])
    z = 1.959963984540054 if abs(ci - 0.95) < 1e-9 else float(
        sm.distributions.norm.ppf(0.5 + ci / 2)
    )
    lo_slope = beta_t - z * se
    hi_slope = beta_t + z * se
    fd_point = float(1 - np.exp(252 * beta_t))
    # fee drag is a decreasing function of slope, so swap CI bounds
    fd_lo = float(1 - np.exp(252 * hi_slope))
    fd_hi = float(1 - np.exp(252 * lo_slope))
    return fd_point, fd_lo, fd_hi


def half_life(log_spread, max_days: float = 1e4) -> float:
    """
    Mean-reversion half-life (trading days) of the log-spread, from the lagged-
    level coefficient phi of the ADF constant+trend regression:
        Delta s_t = mu + beta*t + phi*s_{t-1} + eps.
    Half-life H = -ln(2) / ln(1 + phi). A stationary, mean-reverting spread has
    phi < 0 (and 1 + phi in (0, 1)), giving a finite positive half-life.
    Returns nan if phi >= 0 (no mean reversion) or the regression is degenerate.

    This is a plain OLS on [const, lagged level, trend], without the lagged
    differences of the ADF regression.
    """
    s = pd.Series(np.asarray(log_spread, dtype=float)).dropna().reset_index(drop=True)
    if len(s) < 10:
        return float("nan")
    ds = s.diff().dropna()
    s_lag = s.shift(1).reindex(ds.index)
    t = np.arange(len(ds), dtype=float)
    X = sm.add_constant(np.column_stack([s_lag.values, t]))
    try:
        phi = float(sm.OLS(ds.values, X).fit().params[1])
    except Exception:
        return float("nan")
    if phi >= 0 or (1.0 + phi) <= 0:
        return float("nan")
    hl = -np.log(2) / np.log(1.0 + phi)
    if not np.isfinite(hl) or hl <= 0 or hl > max_days:
        return float("nan")
    return float(hl)


def run_bootstrap_all(results_dir=paths.FINAL, n_boot: int = 1000):
    """
    bootstrap_ci for every run in <results_dir>/timeseries, written to
    <results_dir>/bootstrap_ci.csv (Table tab:ci).
    """
    ts_dir = os.path.join(results_dir, "timeseries")
    if not os.path.exists(ts_dir):
        print("No timeseries directory found.")
        return

    rows = []
    files = sorted(f for f in os.listdir(ts_dir) if f.endswith(".csv"))
    for fname in files:
        run_name = fname[:-4]
        path = os.path.join(ts_dir, fname)
        try:
            ci = bootstrap_ci(path, n_boot=n_boot)
            rows.append({"run_name": run_name, **ci})
            print(f"  {run_name}: TE={ci['te_point']:.2%} [{ci['te_lo']:.2%}, {ci['te_hi']:.2%}]"
                  f"  Fee drag={ci['fee_drag_point']:.2%} [{ci['fee_drag_lo']:.2%}, {ci['fee_drag_hi']:.2%}]")
        except (ValueError, KeyError, RuntimeError, np.linalg.LinAlgError) as e:
            solver_failures["bootstrap_file"] += 1
            print(f"  {run_name}: FAILED ({type(e).__name__}: {e})")

    if rows:
        out = pd.DataFrame(rows)
        out_path = os.path.join(results_dir, "bootstrap_ci.csv")
        out.to_csv(out_path, index=False)
        print(f"\nSaved bootstrap CIs -> {out_path}  ({len(rows)} runs)")


# ─────────────────────────────────────────────────────────────
# Per-run files
# ─────────────────────────────────────────────────────────────

def build_timeseries(p_ret, p_tri, bench_returns, rolling_window=252):
    """
    Build a DataFrame of TRI levels and rolling 1Y tracking error for one run.

    Returns
    -------
    pd.DataFrame with columns:
        benchmark_tri, portfolio_tri, rolling_te_1y
    """
    bench_al = bench_returns.reindex(p_ret.index).fillna(0.0)
    bench_tri = (1 + bench_al).cumprod().rename("benchmark_tri")
    port_tri  = p_tri.rename("portfolio_tri")

    df = pd.concat([bench_tri, port_tri], axis=1).dropna()

    daily  = df.pct_change().dropna()
    active = daily["portfolio_tri"] - daily["benchmark_tri"]
    rolling_te = (active.rolling(rolling_window).std() * np.sqrt(252)).rename("rolling_te_1y")

    return pd.concat([df, rolling_te], axis=1)


def save_run(
    run_name: str,
    params: dict,
    metrics: dict,
    results_dir=paths.FINAL,
):
    """
    Write a run's parameters and metrics to <results_dir>/runs/<run>.csv,
    one file per run (so parallel runs never write to the same file).
    """
    safe_params = {}
    for k, v in params.items():
        if isinstance(v, (list, np.ndarray)):
            safe_params[k] = str(list(v))
        else:
            safe_params[k] = v

    row = {"run_name": run_name, "timestamp": pd.Timestamp.now().isoformat(timespec="seconds")}
    row.update(safe_params)
    row.update(metrics)
    df_new = pd.DataFrame([row])

    per_run_dir = os.path.join(results_dir, "runs")
    os.makedirs(per_run_dir, exist_ok=True)
    safe_name = (run_name.replace("/", "_").replace("\\", "_")
                          .replace(" ", "_").replace(":", "_"))
    path = os.path.join(per_run_dir, f"{safe_name}.csv")
    df_new.to_csv(path, index=False)
    print(f"Saved run '{run_name}' -> {path}")



def save_timeseries(
    run_name: str,
    ts: pd.DataFrame,
    results_dir=paths.FINAL,
):
    """
    Write a run's benchmark and portfolio TRI and rolling TE (build_timeseries)
    to <results_dir>/timeseries/<run>.csv.
    """
    ts_dir = os.path.join(results_dir, "timeseries")
    os.makedirs(ts_dir, exist_ok=True)
    safe_name = run_name.replace("/", "_").replace("\\", "_")
    path = os.path.join(ts_dir, f"{safe_name}.csv")
    ts.to_csv(path)
    print(f"Saved timeseries for '{run_name}' -> {path}")


def save_weights_to_csv(
    run_name: str,
    eff_w: pd.DataFrame,
    results_dir=paths.FINAL,
):
    """
    Write the daily effective weights of simulate_with_rebased_tri (drifting
    between rebalances) to <results_dir>/weights/<run>.csv.
    """
    weights_dir = os.path.join(results_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    safe_name = run_name.replace("/", "_").replace("\\", "_")
    path = os.path.join(weights_dir, f"{safe_name}.csv")
    eff_w.to_csv(path)
    print(f"Saved weights for '{run_name}' -> {path}")