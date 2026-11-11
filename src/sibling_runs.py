"""
Sibling categories: the low- and medium-equity peer averages (data sheets Y_le and Y_me)
under the final specification, written to results/final_siblings/.

For each category: the bandit with the settings selected on the high-equity category (not
re-tuned), the fixed 1.5- and 3-year windows at quarterly and monthly rebalancing, and equal
weight (monthly). The tracker is final_bandit.FitB with the constant and the trend outside
the log, delta * n / 252 and the penalty measured against the drifted weights
(final_runs.TO_REF); the bandit is rewarded on the forward mean absolute tracking error
alone. Equal weight does not depend on the tracker and is run by run_models.run_and_save.

Feeds tab:siblings_models (results/final_siblings/period_tables.csv) and tab:siblings
(through thesis_tables.py).

    python src/sibling_runs.py    # about 35 minutes with cold ridge caches, a few with warm ones

Inputs: data/data.xlsx; results/grids/selection_outside.csv and grid_final.csv (read through
final_bandit.selected_config); the Equation 3.4 ridge caches in results/cache/ (built if
missing).
Outputs (deletes results/final_siblings/ first): timeseries/, turnover/, weights/,
rl_diagnostics/, runs/ (equal weight only) and period_tables.csv.
"""
import os
import shutil
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import final_bandit as fb  # noqa: E402  (sets below-normal priority)
import pandas as pd  # noqa: E402

import bandit_tuning as bt  # noqa: E402
import paths  # noqa: E402
import period_tables as pt  # noqa: E402
import run_models as rm  # noqa: E402
from final_runs import TO_REF, save_run  # noqa: E402

SIBS = ("ma_le", "ma_me")
NREF = fb.SPECS["final"]["n_ref"]


def equal_weight_run(idx):
    """Monthly equal weight on category idx, with the parameters described in Chapter 3 of the thesis."""
    run = f"equal_weight__{idx}__reb21__tc0"
    rm.run_and_save(run_name=run, model="equal_weight", index=idx, model_params=dict(rebalance_freq=21, tc_bps=0),
                    results_dir=str(paths.FINAL_SIBLINGS))
    return run


def main():
    t0 = time.time()
    shutil.rmtree(paths.FINAL_SIBLINGS, ignore_errors=True)
    for sub in ("timeseries", "turnover", "weights", "rl_diagnostics"):
        paths.ensure(paths.FINAL_SIBLINGS / sub)
    _, cfg = fb.selected_config()
    runs = []
    for idx in SIBS:
        data = bt.Data(*rm.load_data(idx))
        ro = fb.build_ridge_cache_out(data, bt.schedule(data.n, 21))
        fit = fb.FitB(data, trend_out=True, n_ref=NREF, ridge_out=ro, to_ref=TO_REF)
        i_, ch, W = fb.run_bandit_m(data, cfg, fit, drift=True, reward="mae")
        run = f"rl_tuned__{idx}__reb63__tc0"
        save_run(data, run, i_, W, paths.FINAL_SIBLINGS, chosen=ch)
        runs.append(run)
        for w in (378, 756):
            for reb, tag in ((63, "__reb63"), (21, "")):
                fit = fb.FitB(data, trend_out=True, n_ref=NREF, ridge_out=ro, to_ref=TO_REF)
                i_, _, W = fb.run_fixed_v(data, w, fit, delta=1.0, reb=reb)
                run = f"window_fixedbest__{idx}__w{w}{tag}__tc0"
                save_run(data, run, i_, W, paths.FINAL_SIBLINGS)
                runs.append(run)
        runs.append(equal_weight_run(idx))
        print(f"{idx}: done ({(time.time() - t0) / 60:.1f} min); bandit windows "
              f"{pd.Series(ch).value_counts().sort_index().to_dict()}", flush=True)
    ptab = pt.build(runs, results_dir=str(paths.FINAL_SIBLINGS))
    ptab.to_csv(paths.FINAL_SIBLINGS / "period_tables.csv", index=False)
    show = ptab[ptab.window.isin(["test", "full"])][["run", "window", "te", "drag", "to", "adf", "za"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
