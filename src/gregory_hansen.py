"""Gregory-Hansen (1996) cointegration tests with one regime shift at an unknown date.

Feeds the Gregory-Hansen columns of tab:two_breaks (C/S statistic, test and full
periods; the C and C/T statistics back the text) and draws fig:gh_fit
(figures/gh_regime_shift_house.png).

Gregory, A.W. and Hansen, B.E. (1996), Residual-based tests for cointegration in models
with regime shifts, Journal of Econometrics 70(1), 99-126.

For each candidate break b in [0.15n, 0.85n] (phi_t = 1 for t > b), fit by OLS

    C    y = mu1 + mu2 phi + a x + e                    (level shift, eq. 2.2)
    C/T  y = mu1 + mu2 phi + beta t + a x + e           (level shift with trend)
    C/S  y = mu1 + mu2 phi + a1 x + a2 x phi + e        (regime shift, eq. 2.3)

with y = log benchmark TRI and x = log portfolio TRI (the direction of the thesis's
Engle-Granger test), run an ADF regression on the residuals (no deterministic terms,
as statsmodels' coint does), and keep ADF* = min_b ADF(b). Critical values: the paper's
Table 1, m = 1 (approximate asymptotic).

Lag K: the AIC lag that statsmodels' coint(trend="ct") picks for the no-break
Engle-Granger residual, held fixed across break dates (as the two-break tests hold
the ADF lag fixed). Check columns *_aic: K re-chosen by AIC at every break date on a
21-day grid.

    python src/gregory_hansen.py run          # final/gregory_hansen.csv (about 10 minutes)
    python src/gregory_hansen.py plot [out]   # figures/gh_regime_shift_house.png, or the file out
    python src/gregory_hansen.py check        # reproduces statsmodels' EG statistic; size/power simulations

Run before final_tests.py table (or breaks), which adds the C/S columns to tab_two_breaks.csv.

Paths are relative to results/.
Inputs: final/timeseries/<run>.csv (final_runs.py).
Outputs: final/gregory_hansen.csv, figures/gh_regime_shift_house.png.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.stattools import adfuller, coint  # noqa: E402

import paths  # noqa: E402

START, TEST, END = "2009-01-28", "2017-01-03", "2024-05-15"
TRIM = 0.15

# Gregory and Hansen (1996), Table 1, m = 1, ADF* / Zt*.
CV = {"C": {"1%": -5.13, "5%": -4.61, "10%": -4.34},
      "C/T": {"1%": -5.45, "5%": -4.99, "10%": -4.72},
      "C/S": {"1%": -5.47, "5%": -4.95, "10%": -4.68}}

RUNS = {
    "Tuned bandit (quarterly)": "rl_tuned__ma__reb63__tc0",
    **{f"Fixed {w / 252:g}-year window (quarterly)": f"window_fixedbest__ma__w{w}__reb63__tc0"
       for w in (126, 252, 378, 504, 756)},
    "Turnover-penalised ridge (monthly)": "ridge_trend_to__ma__w252__reb21__d1__tc0",
    "Return-space ridge (monthly)": "ridge_returns__ma__w252__reb21__tc0",
    "Equal weight (monthly)": "equal_weight__ma__reb21__tc0",
    "Regulation 28 allocation (monthly)": "passive_reg28__ma__reb21__tc0",
    "60/40 allocation (monthly)": "passive_60_40__ma__reb21__tc0",
}


def load(run, start):
    df = pd.read_csv(paths.FINAL / "timeseries" / f"{run}.csv", parse_dates=["datestamp"])
    df = df.set_index("datestamp").sort_index().loc[start:END]
    return np.log(df["benchmark_tri"]), np.log(df["portfolio_tri"])


def design(x, b, model):
    n = len(x)
    phi = (np.arange(n) > b).astype(float)
    cols = [np.ones(n), phi]
    if model == "C/T":
        cols.append(np.arange(1, n + 1, dtype=float))
    cols.append(x)
    if model == "C/S":
        cols.append(x * phi)
    return np.column_stack(cols)


def adf_t(e, k):
    """t-statistic on e_{t-1} in  de_t = rho e_{t-1} + sum_{i<=k} c_i de_{t-i} + u_t  (no constant)."""
    de = np.diff(e)
    yv = de[k:]
    X = [e[k:-1]]
    for i in range(1, k + 1):
        X.append(de[k - i:-i])
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    u = yv - X @ beta
    s2 = u @ u / (len(yv) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    return beta[0] / se


def eg_lag(y, x):
    """AIC lag statsmodels' coint(trend='ct') uses for the no-break EG residual."""
    X = np.column_stack([np.ones(len(x)), np.arange(1, len(x) + 1, dtype=float), x])
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    res = adfuller(e, regression="n", autolag="aic")
    return int(res[2]), e


def gh(y, x, model, k=None, step=1):
    """ADF* and the first date of the new regime. k=None -> AIC lag re-chosen at each break."""
    n = len(y)
    lo, hi = int(np.ceil(TRIM * n)), int(np.floor((1 - TRIM) * n))
    best, bbest, kbest = np.inf, None, None
    for b in range(lo, hi + 1, step):
        X = design(x, b, model)
        e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        if k is None:
            kk = int(adfuller(e, regression="n", autolag="aic")[2])
        else:
            kk = k
        t = adf_t(e, kk)
        if t < best:
            best, bbest, kbest = t, b, kk
    return best, bbest + 1, kbest


def check():
    rng = np.random.default_rng(0)
    # 1. no-break statistic reproduces statsmodels' coint (trend="ct")
    y, x = load(RUNS["Tuned bandit (quarterly)"], TEST)
    k, e = eg_lag(y.to_numpy(), x.to_numpy())
    sm = coint(y.to_numpy(), x.to_numpy(), trend="ct")
    print(f"EG test period: statsmodels {sm[0]:.4f} (p {sm[1]:.3f}), own {adf_t(e, k):.4f}, lag {k}")
    # 2. size: independent random walks, n = 400, fixed lag 0
    for model in CV:
        rej = 0
        for _ in range(300):
            yy, xx = rng.standard_normal(400).cumsum(), rng.standard_normal(400).cumsum()
            rej += gh(yy, xx, model, k=0, step=2)[0] <= CV[model]["5%"]
        print(f"size {model:4s}: {rej / 300:.3f} rejected at 5% (should be near 0.05)")
    # 3. power: cointegrated with a level shift at 0.5n
    rej = 0
    for _ in range(100):
        xx = rng.standard_normal(400).cumsum()
        ee = np.zeros(400)
        for t in range(1, 400):
            ee[t] = 0.5 * ee[t - 1] + rng.standard_normal()
        yy = xx + 3.0 * (np.arange(400) > 200) + ee
        rej += gh(yy, xx, "C", k=0, step=2)[0] <= CV["C"]["5%"]
    print(f"power C, level shift 3 sd: {rej / 100:.2f}")


def run():
    rows = []
    for name, run_id in RUNS.items():
        for per, start in (("test", TEST), ("full", START)):
            ys, xs = load(run_id, start)
            y, x = ys.to_numpy(), xs.to_numpy()
            k, _ = eg_lag(y, x)
            eg_p = float(coint(y, x, trend="ct")[1])
            row = dict(model=name, run=run_id, period=per, n=len(y), lag=k, eg_p=eg_p)
            for model in CV:
                stat, b, _ = gh(y, x, model, k=k)
                stat_a, b_a, k_a = gh(y, x, model, k=None, step=21)
                tag = model.replace("/", "")
                row.update({f"{tag}_stat": stat, f"{tag}_break": ys.index[b].date().isoformat(),
                            f"{tag}_rej5": bool(stat <= CV[model]["5%"]),
                            f"{tag}_rej1": bool(stat <= CV[model]["1%"]),
                            f"{tag}_stat_aic": stat_a, f"{tag}_break_aic": ys.index[b_a].date().isoformat(),
                            f"{tag}_lag_aic": k_a})
            rows.append(row)
            print(f"{name:40s} {per:4s} k={k:2d} EG p {eg_p:.3f} | "
                  + " | ".join(f"{m} {row[m.replace('/', '') + '_stat']:6.2f} {row[m.replace('/', '') + '_break']}"
                               f" (aic {row[m.replace('/', '') + '_stat_aic']:6.2f})" for m in CV), flush=True)
            pd.DataFrame(rows).to_csv(paths.ensure(paths.FINAL) / "gregory_hansen.csv", index=False)
    print("DONE")


def plot(out=None):
    """Thesis figure: the tuned bandit's C/S fit on the test period (spread and fitted value; gap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    run_id = RUNS["Tuned bandit (quarterly)"]
    ys, xs = load(run_id, TEST)
    y, x = ys.to_numpy(), xs.to_numpy()
    k, _ = eg_lag(y, x)
    stat, first_new, _ = gh(y, x, "C/S", k=k)
    X = design(x, first_new - 1, "C/S")
    yhat = X @ np.linalg.lstsq(X, y, rcond=None)[0]
    dates, i0 = ys.index, first_new
    spread, fitted = (x - y) * 100, (x - yhat) * 100

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.titlesize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})
    gap = spread - fitted
    fig, ax = plt.subplots(3, 1, figsize=(10, 8.8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1]})
    ax[0].plot(dates, spread, color="#555555", lw=1, label="Spread")
    ax[0].plot(dates[:i0], fitted[:i0], color="#6a0dad", lw=2, label="Fitted, before the break")
    ax[0].plot(dates[i0:], fitted[i0:], color="#ff7f0e", lw=2, label="Fitted, after the break")
    ax[0].set_title("Spread and fitted value")
    ax[0].set_ylabel("Log %")
    ax[0].legend(loc="upper left", frameon=False)
    ax[1].plot(dates, gap, color="#555555", lw=1)
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_title("Gap to the fitted value")
    ax[1].set_ylabel("Log %")
    ax[2].plot(dates[1:], np.diff(gap), color="#555555", lw=0.8)
    ax[2].axhline(0, color="black", lw=0.8)
    ax[2].set_title("Daily change in the gap")
    ax[2].set_ylabel("Log % per day")
    ax[2].xaxis.set_major_locator(mdates.YearLocator())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a in ax:
        a.axvline(dates[i0], color="#c0392b", lw=1.2, ls="--", zorder=1)
    fig.tight_layout()
    out = out or paths.ensure(paths.FIGURES) / "gh_regime_shift_house.png"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"C/S ADF* {stat:.2f}, first date of new regime {dates[i0].date()}, lag {k} -> {out}")


def _low_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


STAGES = {"run": run, "plot": plot, "check": check}

if __name__ == "__main__":
    _args = sys.argv[1:]
    if not _args or _args[0] not in STAGES or len(_args) > (2 if _args[0] == "plot" else 1):
        sys.exit("usage: python src/gregory_hansen.py run|plot [out]|check")
    _low_priority()
    STAGES[_args[0]](*_args[1:])
