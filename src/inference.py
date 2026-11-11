"""
Diebold-Mariano tests over the 30-pair family with a Benjamini-Hochberg
correction, plus the bootstrap confidence intervals.

Used by final_tests.py dm, which runs it on paths.FINAL for the full period
(inference_dm.csv) and the test period from 2017 (inference_dm_test.csv).
dm_pooled.py then pools these 30 pairs with the 8 pairs of Table tab:dm_h1
into the 38-pair family of Tables tab:dm, tab:dm_variants and tab:dm_h1.

    python src/inference.py   # full period, paths.FINAL

The family is the union of dm_test.PAIRS and DEFAULT_PAIRS below, with an
unordered pair counted once (30 pairs). The Benjamini-Hochberg adjustment is
applied over the whole family in one pass, and the 5% flag is read from the
adjusted p-value. The DM test itself is dm_test.dm_test.

If <results_dir>/bootstrap_ci.csv does not exist, metrics.run_bootstrap_all
writes it (Table tab:ci).
"""

import os

import numpy as np
import pandas as pd

import paths
from dm_test import PAIRS as DM_TEST_PAIRS
from dm_test import dm_test, load_active_returns
from metrics import run_bootstrap_all

# Pairs (model_1, model_2) of the tracking-reward bandit against the static
# models; all at zero transaction cost. Three of them are already in
# dm_test.PAIRS, so they add two pairs to the family.
DEFAULT_PAIRS = [
    ("rl_bandit_to__ma__tc0", "ridge_trend_to__ma__w252__reb21__d1__tc0"),
    ("rl_bandit_to__ma__tc0", "ridge_trend__ma__w252__reb21__tc0"),
    ("rl_bandit_to__ma__tc0", "ridge_returns__ma__w252__reb21__tc0"),
    ("rl_bandit_to__ma__tc0", "equal_weight__ma__reb21__tc0"),
    ("rl_bandit_to__ma__tc0", "passive_reg28__ma__reb21__tc0"),
]


def build_reported_family(default_pairs=None, dm_test_pairs=None):
    """
    The family of pairs the correction is applied over: dm_test.PAIRS followed
    by DEFAULT_PAIRS, with (a, b) and (b, a) counted once and the first-seen
    orientation kept. Returns a list of (model_1, model_2).
    """
    if default_pairs is None:
        default_pairs = DEFAULT_PAIRS
    if dm_test_pairs is None:
        dm_test_pairs = DM_TEST_PAIRS

    # dm_test.PAIRS entries carry a label as a third element
    dm_pairs_2 = [(m1, m2) for (m1, m2, *_rest) in dm_test_pairs]

    family = []
    seen = set()
    for m1, m2 in list(dm_pairs_2) + list(default_pairs):
        key = frozenset((m1, m2))  # unordered: (a, b) == (b, a)
        if key in seen:
            continue
        seen.add(key)
        family.append((m1, m2))
    return family


def benjamini_hochberg(pvals):
    """
    Benjamini-Hochberg (1995) FDR-adjusted p-values.

    Returns an array of BH-adjusted p-values aligned to the input order.
    NaN p-values are propagated as NaN and excluded from the ranking.

    The adjustment is the standard step-up procedure: rank the m valid
    p-values ascending, scale each by m / rank, then enforce monotonicity by
    taking the cumulative minimum from the largest rank downward. Adjusted
    values are clipped to [0, 1]. This is the same quantity statsmodels
    returns from multipletests(method='fdr_bh')[1].
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)

    valid = ~np.isnan(p)
    pv = p[valid]
    m = pv.size
    if m == 0:
        return out

    order = np.argsort(pv, kind="mergesort")          # ascending
    ranks = np.arange(1, m + 1)
    adj_sorted = pv[order] * m / ranks
    # Enforce monotonicity from the top down, then clip.
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    out[valid] = adj
    return out


def run_inference_panel(results_dir=paths.FINAL, pairs=None, start=None,
                        out_name="inference_dm.csv"):
    """
    Diebold-Mariano tests on every pair, with the Benjamini-Hochberg adjustment
    over the family.

    For each (model_1, model_2), the daily active returns of the two runs are
    read from <results_dir>/timeseries, restricted to their common dates (and
    to dates from `start`, if given), and tested with dm_test.dm_test (squared
    loss, HLN correction).

    Writes <results_dir>/<out_name> with columns pair ("model_1 vs model_2"),
    dm_stat (HLN-corrected), p_raw, p_bh (adjusted), significant_5pct_bh
    (p_bh < 0.05) and n_pairs_in_family. A pair whose run is missing is kept
    with NaN values. Returns the same DataFrame.

    results_dir : folder holding timeseries/ and receiving the outputs
    pairs       : list of (model_1, model_2); default build_reported_family()
    start       : first date of the test period, e.g. "2017-01-01" (default: all dates)
    out_name    : output file name
    """
    if pairs is None:
        pairs = build_reported_family()

    ts_dir = os.path.join(results_dir, "timeseries")
    rows = []
    for m1, m2 in pairs:
        pair_label = f"{m1} vs {m2}"
        try:
            e1 = load_active_returns(m1, ts_dir)
            e2 = load_active_returns(m2, ts_dir)
        except FileNotFoundError as exc:
            print(f"SKIP {pair_label}: {exc}")
            rows.append({
                "pair": pair_label,
                "dm_stat": np.nan,
                "p_raw": np.nan,
            })
            continue

        idx = e1.index.intersection(e2.index)
        if start is not None:
            idx = idx[idx >= pd.Timestamp(start)]
        a1 = e1.loc[idx].to_numpy()
        a2 = e2.loc[idx].to_numpy()
        res = dm_test(a1, a2, h=1)

        rows.append({
            "pair": pair_label,
            "dm_stat": res["dm_stat_hln"],
            "p_raw": res["pvalue_two_sided"],
        })

    df = pd.DataFrame(rows)

    # Benjamini-Hochberg over the whole family in one pass
    df["p_bh"] = benjamini_hochberg(df["p_raw"].to_numpy())
    df["significant_5pct_bh"] = df["p_bh"] < 0.05
    df["n_pairs_in_family"] = len(pairs)

    df = df[["pair", "dm_stat", "p_raw", "p_bh", "significant_5pct_bh",
             "n_pairs_in_family"]]

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, out_name)
    df.to_csv(out_path, index=False)
    print(f"Saved DM inference panel -> {out_path}  ({len(df)} pairs)")

    # Bootstrap confidence intervals (Table tab:ci), if not already there
    boot_path = os.path.join(results_dir, "bootstrap_ci.csv")
    if not os.path.exists(boot_path):
        print(f"{boot_path} missing; running run_bootstrap_all ...")
        run_bootstrap_all(results_dir=results_dir)
    else:
        print(f"{boot_path} already exists; skipping bootstrap.")

    # Pretty print
    print()
    print("Diebold-Mariano panel with Benjamini-Hochberg FDR correction")
    print("Negative DM => model_1 (left) tracks more tightly than model_2 (right)")
    print()
    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(df.to_string(
            index=False,
            formatters={
                "dm_stat": lambda v: f"{v:+.3f}" if pd.notna(v) else "nan",
                "p_raw": lambda v: f"{v:.4f}" if pd.notna(v) else "nan",
                "p_bh": lambda v: f"{v:.4f}" if pd.notna(v) else "nan",
            },
        ))

    return df


if __name__ == "__main__":
    run_inference_panel()
