"""Hindsight check of the long-only constraint (Appendix A, other specifications).

1. Over all the data, find the fixed long-only mix of the 11 asset classes (w >= 0, sum w = 1,
   rebalanced daily) whose log spread against the peer average reverts fastest: the most
   negative coefficient phi in the regression of the daily change in the spread on a constant,
   the lagged spread and a trend (SLSQP from equal weight and 7 seeded random starts). Its
   spread is then tested with the ADF (constant and trend, AIC lags), Zivot-Andrews and
   Engle-Granger (both directions, the smaller p-value) tests.
2. Rerun the same search every 126 trading days on the data available at the time (the first
   origin after 756 days), hold the mix for the next 504 trading days and test that spread with
   the ADF and Engle-Granger tests. The appendix counts the origins where both reject at 5%.

The objective never uses a test statistic. Reads data/data.xlsx. Writes
<results>/robustness/longonly_hindsight.csv (the full-sample mix and its tests) and
<results>/robustness/longonly_rolling.csv (one row per origin). About 5 minutes on one core.

    python src/longonly_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, coint

import paths
from breakpoint_coint import za_break_test
from metrics import half_life
from run_models import load_data

TEST_LEN = 504      # days the mix is held after each origin
STEP = 126          # days between origins
MIN_TRAIN = 756     # days of data before the first origin


def spread_of(w, R, b):
    """Log spread of a daily-rebalanced fixed mix against the benchmark, both rebased to 1."""
    port = np.cumprod(1.0 + R @ w)
    bench = np.cumprod(1.0 + b)
    return np.log(port / port[0]) - np.log(bench / bench[0])


def phi(w, R, b):
    """Coefficient on the lagged spread in d(spread) ~ 1 + spread_lag + t (more negative = faster reversion)."""
    s = spread_of(w, R, b)
    if not np.all(np.isfinite(s)):
        return 1.0
    ds = np.diff(s)
    X = np.column_stack([np.ones(ds.size), s[:-1], np.arange(ds.size, dtype=float)])
    beta, *_ = np.linalg.lstsq(X, ds, rcond=None)
    return float(beta[1])


def max_reversion_mix(R, b, starts=8, seed=0, maxiter=500):
    n = R.shape[1]
    rng = np.random.default_rng(seed)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    best = None
    for k in range(starts):
        w0 = np.ones(n) / n if k == 0 else rng.dirichlet(np.ones(n))
        res = minimize(lambda w: phi(w, R, b), w0, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                       constraints=cons, options={"maxiter": maxiter, "ftol": 1e-10})
        w = np.clip(res.x, 0, None)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        f = phi(w, R, b)
        if best is None or f < best[0]:
            best = (f, w)
    return best[1], best[0]


def eg_min_p(R, b, w):
    lp = np.log(np.cumprod(1 + R @ w))
    lb = np.log(np.cumprod(1 + b))
    return min(float(coint(lb, lp, trend="ct", autolag="AIC")[1]),
               float(coint(lp, lb, trend="ct", autolag="AIC")[1]))


def main():
    assets, bench = load_data("ma")
    cols = list(assets.columns)
    R, b, dates = assets.values, bench.values, assets.index

    # 1. full-sample mix (the single run uses maxiter 500; the rolling fits below use 400)
    w, f = max_reversion_mix(R, b)
    s = spread_of(w, R, b)
    row = {"start": str(dates[0].date()), "end": str(dates[-1].date()), "phi": f,
           "adf_ct_p": float(adfuller(s, regression="ct", autolag="AIC")[1]),
           "za_p": float(za_break_test(pd.Series(s)).get("za_pvalue", np.nan)),
           "eg_min_p": eg_min_p(R, b, w), "half_life_days": float(half_life(s))}
    row.update({f"w[{c}]": float(x) for c, x in zip(cols, w)})
    out = paths.ensure(paths.ROBUSTNESS)
    pd.DataFrame([row]).to_csv(out / "longonly_hindsight.csv", index=False)
    print(f"hindsight mix ({row['start']} to {row['end']}): ADF p {row['adf_ct_p']:.4f}, ZA p {row['za_p']:.4f}, "
          f"EG p {row['eg_min_p']:.4f}")
    print("  " + ", ".join(f"{c} {100 * x:.0f}%" for c, x in sorted(zip(cols, w), key=lambda t: -t[1]) if x > 0.005))

    # 2. rolling origins
    rows = []
    for o in range(MIN_TRAIN, len(dates) - TEST_LEN, STEP):
        w, _ = max_reversion_mix(R[:o], b[:o], maxiter=400)
        Rte, bte = R[o:o + TEST_LEN], b[o:o + TEST_LEN]
        s_te = spread_of(w, Rte, bte)
        adf_te = float(adfuller(s_te, regression="ct", autolag="AIC")[1])
        eg_te = eg_min_p(Rte, bte, w)
        r = {"origin_date": str(dates[o].date()),
             "train_adf_p": float(adfuller(spread_of(w, R[:o], b[:o]), regression="ct", autolag="AIC")[1]),
             "test_adf_p": adf_te, "test_eg_min_p": eg_te, "rejects": bool(adf_te < 0.05 and eg_te < 0.05)}
        r.update({f"w[{c}]": float(x) for c, x in zip(cols, w)})
        rows.append(r)
        print(f"  {r['origin_date']}: held-out ADF p {adf_te:.3f}, EG p {eg_te:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "longonly_rolling.csv", index=False)
    print(f"rolling: the held-out spread rejects on both tests at {int(df['rejects'].sum())} of {len(df)} origins")


if __name__ == "__main__":
    main()
