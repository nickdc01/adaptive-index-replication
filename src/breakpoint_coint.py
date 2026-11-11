"""
Zivot-Andrews unit-root test with one structural break, on the log spread
log(portfolio TRI) - log(benchmark TRI).

Library module (no command line). za_break_test gives the ZA p-values ("ZA p")
of every results table, through period_tables.py and run_tuning.segment.

The spread is a difference with a unit coefficient, not an estimated
cointegrating regression, so the question is whether this one series is
stationary around a trend that may shift once at an unknown date. A
full-sample ADF test can fail to reject a unit root when the equilibrium
re-levels at a regime change; the Zivot and Andrews (1992) test searches for
that break. Gregory-Hansen tests a break in an estimated cointegrating
relationship and is run separately (gregory_hansen.py).
"""

import numpy as np
from arch.unitroot import ZivotAndrews

# Zivot-Andrews (1992) Table 4, model C (break in both intercept and trend),
# asymptotic critical values of the minimum t-statistic.
ZA_MODEL_C_CRIT = {"1%": -5.57, "5%": -5.08, "10%": -4.82}


def za_break_test(spread, trim: float = 0.15) -> dict:
    """
    Zivot-Andrews test (model C: break in intercept and trend) on a log spread,
    with arch.unitroot.ZivotAndrews.

    The null is a unit root; rejection means the spread is stationary around a
    trend that breaks once. The break is searched for in [trim, 1 - trim] of
    the sample.

    Returns a dict:
        stat             minimum t-statistic on the lagged level
        crit_5pct        5% critical value
        break_index      position (0-based) of the estimated break, -1 if unknown
        break_frac       break_index / (n - 1)
        stationary_5pct  True if the unit root is rejected at 5%
        method           "arch"
        za_pvalue        the p-value (when arch provides one)
    """
    y = np.asarray(spread, dtype=float)
    y = y[~np.isnan(y)]
    n = y.size
    if n < 20:
        raise ValueError(f"spread too short for ZA test (n={n}); need >= 20 obs.")

    za = ZivotAndrews(y, trend="ct", trim=trim)
    stat = float(za.stat)

    # Break location: arch keeps the statistic of every candidate break in
    # _all_stats (NaN in the trimmed ends); the break is its argmin. Older
    # versions exposed the index directly.
    break_index = None
    for attr in ("_break_idx", "break_idx", "_break_index"):
        if hasattr(za, attr):
            try:
                break_index = int(getattr(za, attr))
            except (TypeError, ValueError):
                break_index = None
            if break_index is not None:
                break
    if break_index is None and hasattr(za, "_all_stats"):
        try:
            all_stats = np.asarray(za._all_stats, dtype=float)
            if np.isfinite(all_stats).any():
                break_index = int(np.nanargmin(all_stats))
        except (TypeError, ValueError):
            break_index = None
    if break_index is None:
        break_index = -1

    crit_5pct = ZA_MODEL_C_CRIT["5%"]
    try:
        cv = za.critical_values
        if isinstance(cv, dict) and "5%" in cv:
            crit_5pct = float(cv["5%"])
    except (AttributeError, KeyError, TypeError):
        pass

    # Reject on the p-value when there is one, else on the 5% critical value
    # (a left-tailed test).
    pvalue = None
    try:
        pvalue = float(za.pvalue)
    except (AttributeError, ValueError, TypeError):
        pvalue = None

    if pvalue is not None and np.isfinite(pvalue):
        stationary_5pct = bool(pvalue < 0.05)
    else:
        stationary_5pct = bool(stat < crit_5pct)

    break_frac = float(break_index / (n - 1)) if break_index >= 0 else float("nan")
    out = {
        "stat": stat,
        "crit_5pct": float(crit_5pct),
        "break_index": int(break_index),
        "break_frac": break_frac,
        "stationary_5pct": stationary_5pct,
        "method": "arch",
    }
    if pvalue is not None:
        out["za_pvalue"] = pvalue
    return out
