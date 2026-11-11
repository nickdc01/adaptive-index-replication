"""Pool the Diebold-Mariano tests into one Benjamini-Hochberg family of 38 pairs per period.

The thesis adjusts its Diebold-Mariano p-values once per period, over all 38 pairs it
tests: the 30 pairs of inference.py (inference_dm.csv for the full period,
inference_dm_test.csv for the test period) and the 8 static ridges against equal weight
(dm_h1.csv). The adjusted value p_bh38 is the p_BH printed in tab:dm, tab:dm_variants and
tab:dm_h1. The p_bh column carried over from the source files is the adjustment within
the smaller family (30 or 8 pairs) and is not used in the thesis.

final_tests.py dm calls pool() after writing the three source files. To rerun it alone:

    python src/dm_pooled.py

Inputs:  final/inference_dm.csv, final/inference_dm_test.csv, final/dm_h1.csv
Output:  final/inference_dm_pooled38.csv (pair, dm_stat, p_raw, p_bh, family, p_bh38, period)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import paths

EW = "equal_weight__ma__reb21__tc0"


def bh(p):
    """Benjamini-Hochberg adjusted p-values (step-up, capped at 1)."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 1.0
    for r in range(m - 1, -1, -1):
        i = o[r]
        run = min(run, p[i] * m / (r + 1))
        adj[i] = run
    return adj


def pool(final_dir=None):
    """Write inference_dm_pooled38.csv in final_dir (default paths.FINAL) and return it."""
    final_dir = Path(paths.FINAL if final_dir is None else final_dir)
    h1 = pd.read_csv(final_dir / "dm_h1.csv")
    out = []
    for period, f in (("test", "inference_dm_test.csv"), ("full", "inference_dm.csv")):
        a = pd.read_csv(final_dir / f)[["pair", "dm_stat", "p_raw", "p_bh"]].assign(family="30")
        b = h1[h1.period == period].rename(columns={"dm": "dm_stat"})
        b = b.assign(pair=b["run"] + f" vs {EW}", family="H1")[["pair", "dm_stat", "p_raw", "p_bh", "family"]]
        x = pd.concat([a, b], ignore_index=True)
        assert len(x) == 38 and x.pair.is_unique, (period, len(x))
        x["p_bh38"] = bh(x.p_raw)
        x["period"] = period
        out.append(x)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(final_dir / "inference_dm_pooled38.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 90, "display.max_rows", 100):
        print(res[["period", "family", "pair", "dm_stat", "p_raw", "p_bh38"]].round(4).to_string(index=False))
    print("wrote", final_dir / "inference_dm_pooled38.csv")
    return res


if __name__ == "__main__":
    pool()
