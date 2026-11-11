"""Per-period scores of saved runs: every model scored on the same calendar windows.

The scores behind the results tables (through thesis_tables.py): each run's
saved timeseries (timeseries/<run>.csv) and turnover (turnover/<run>.csv) are
scored on three fixed windows,

    full   2009-01-28 -> 2024-05-15   (the bandit's first decision to the data end)
    train  2009-01-28 -> 2016-12-30   (the training period of the tuning)
    test   2017-01-01 -> 2024-05-15   (the test period)

with each window's TRIs rebased to 1 on its first day. Columns:

    te    annualised std of daily active returns        (metrics._metrics_from_compare)
    drag  fee drag, 1 - exp(252 * log-spread trend)     (metrics._metrics_from_compare)
    to    mean one-way turnover over the window's complete calendar years
          (2010-2023, 2010-2016, 2017-2023), years with no rebalance counting
          as zero (metrics.turnover_complete_years)
    to_ay calendar-year average of the turnover inside the window, partial
          years included (compute_metrics' avg_yearly_turnover)
    adf   ADF p of the log spread, constant + trend, AIC lags
    za    Zivot-Andrews p (breakpoint_coint.za_break_test)
    eg    Engle-Granger p, coint(log B, log P, trend="ct", autolag="AIC")
    hl    half-life of the log spread in trading days (metrics.half_life)

The train / test / full rows of a grid configuration match its row in the
tuning grids, which score the same windows the same way (run_tuning.segment).

final_runs.py and sibling_runs.py call build() and write period_tables.csv in
paths.FINAL and paths.FINAL_SIBLINGS. From the command line:

    python src/period_tables.py                       # every run in paths.FINAL
    python src/period_tables.py RUN [RUN ...] [--results-dir DIR] [--out FILE]

Output: <results dir>/period_tables.csv in long format (one row per run and
window). Runs without a timeseries are skipped with a note; a run without
saved turnover gets NaN turnover.
"""
import argparse
import os
import warnings

# Single-threaded BLAS before numpy loads, so the test statistics are reproducible
# to the last digit (multi-threaded BLAS changes them at about 1e-14).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.stattools import coint  # noqa: E402

import paths  # noqa: E402
from breakpoint_coint import za_break_test  # noqa: E402
from metrics import (  # noqa: E402
    _load_compare_from_timeseries,
    _metrics_from_compare,
    complete_years,
    half_life,
    load_run_turnover,
    turnover_complete_years,
)

RESULTS = paths.FINAL

# (label, start, end); complete calendar years for turnover come from
# metrics.complete_years: 2010-2023, 2010-2016 and 2017-2023.
PERIODS = [
    ("full", "2009-01-28", "2024-05-15"),
    ("train", "2009-01-28", "2016-12-30"),
    ("test", "2017-01-01", "2024-05-15"),
]
# A run whose data starts later / ends earlier than a window by more than this
# many calendar days does not cover it; its row is kept but flagged.
COVER_SLACK_DAYS = 7


def window_frame(run, start, end, results_dir=RESULTS):
    """The run's Benchmark / Tracking Portfolio TRI frame on [start, end], rebased
    to 1 on the first day; None when the run has no timeseries or the window is
    (almost) empty. start / end of None leave that side open."""
    path = os.path.join(results_dir, "timeseries", f"{run}.csv")
    if not os.path.exists(path):
        return None
    cmp = _load_compare_from_timeseries(path).loc[start:end]
    if len(cmp) < 30:
        return None
    return cmp / cmp.iloc[0]


def frame_metrics(cmp, turn, years):
    """te, drag, to, to_ay, adf, za, eg, hl on a (rebased) TRI frame. turn is the
    run's full one-way turnover Series (None when not saved); years the complete
    calendar years that the `to` column averages over."""
    if turn is not None and len(turn):
        turn_w = turn.loc[(turn.index >= cmp.index[0]) & (turn.index <= cmp.index[-1])]
    else:
        turn_w = pd.Series(dtype=float)
    m = _metrics_from_compare(cmp, turn_w)
    lp = np.log(cmp["Tracking Portfolio"].values)
    lb = np.log(cmp["Benchmark"].values)
    s = lp - lb
    try:
        za = float(za_break_test(pd.Series(s)).get("za_pvalue", np.nan))
    except Exception:
        za = np.nan
    eg = float(coint(lb, lp, trend="ct", autolag="AIC")[1])
    return dict(
        te=float(m["annualised_te"]),
        drag=float(m["fee_drag_pa"]),
        to=turnover_complete_years(turn, years),
        to_ay=float(m["avg_yearly_turnover"]) if turn is not None else np.nan,
        adf=float(m["adf_pval"]),
        za=za,
        eg=eg,
        hl=float(half_life(s)),
    )


def build(runs, results_dir=RESULTS, periods=PERIODS):
    """Scores of each run on each period, as a DataFrame (one row per run and window)."""
    rows = []
    for run in runs:
        if not os.path.exists(os.path.join(results_dir, "timeseries", f"{run}.csv")):
            print(f"[skip] {run}: no timeseries in {os.path.join(results_dir, 'timeseries')}")
            continue
        turn = load_run_turnover(run, results_dir)
        if turn is None:
            print(f"[note] {run}: no turnover file; turnover left NaN")
        for label, start, end in periods:
            cmp = window_frame(run, start, end, results_dir)
            if cmp is None:
                print(f"[skip] {run} / {label}: fewer than 30 days in {start} -> {end}")
                continue
            covers = bool(cmp.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=COVER_SLACK_DAYS)
                          and cmp.index[-1] >= pd.Timestamp(end) - pd.Timedelta(days=COVER_SLACK_DAYS))
            if not covers:
                print(f"[note] {run} / {label}: data {cmp.index[0].date()} -> "
                      f"{cmp.index[-1].date()} does not cover {start} -> {end}")
            years = complete_years(start, end)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = frame_metrics(cmp, turn, years)
            rows.append(dict(
                run=run, window=label, window_start=start, window_end=end,
                first=str(cmp.index[0].date()), last=str(cmp.index[-1].date()),
                n_obs=int(len(cmp)), covers_window=covers,
                turnover_years=f"{years[0]}-{years[-1]}" if years else "",
                **m,
            ))
    return pd.DataFrame(rows)


def all_runs(results_dir=RESULTS):
    """Every run with a saved timeseries in results_dir, sorted by name."""
    ts_dir = os.path.join(results_dir, "timeseries")
    if not os.path.isdir(ts_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(ts_dir) if f.endswith(".csv"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", help="run names (default: every run in the results folder)")
    ap.add_argument("--results-dir", default=str(RESULTS), help="folder with timeseries/ and turnover/")
    ap.add_argument("--out", default=None, help="output CSV (default: <results dir>/period_tables.csv)")
    args = ap.parse_args(argv)
    out_path = args.out or os.path.join(args.results_dir, "period_tables.csv")
    out = build(args.runs or all_runs(args.results_dir), results_dir=args.results_dir)
    if not len(out):
        print("No run had a timeseries; nothing written.")
        return out
    out.to_csv(out_path, index=False)
    show = out.assign(
        te=lambda t: (100 * t["te"]).round(2), drag=lambda t: (100 * t["drag"]).round(2),
        to=lambda t: (100 * t["to"]).round(1), to_ay=lambda t: (100 * t["to_ay"]).round(1),
        adf=lambda t: t["adf"].round(3), za=lambda t: t["za"].round(3),
        eg=lambda t: t["eg"].round(3), hl=lambda t: t["hl"].round(0))
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(show[["run", "window", "te", "drag", "to", "to_ay", "adf", "za", "eg", "hl"]]
              .to_string(index=False))
    print(f"\nWritten -> {out_path} ({out['run'].nunique()} runs, {len(out)} rows; "
          "te/drag/to in the CSV are decimals)")
    return out


if __name__ == "__main__":
    main()
