"""Rebuild every table and figure of the thesis from data/data.xlsx.

    python run_all.py                     # every stage, in order
    python run_all.py --list              # the stages, their outputs and estimated runtimes
    python run_all.py --from final_runs   # resume at a stage (by name or number)
    python run_all.py --only tables figures
    python run_all.py --fresh             # first delete everything earlier runs wrote, caches included
    python run_all.py --jobs 4            # worker processes (default: environment N_JOBS, else 2)
    python run_all.py --optional          # also run the optional stages (self-checks and the
                                          # original 910-setting grid)

Each stage runs its scripts in child Python processes, writes their output to
results/logs/<nn>_<stage>.log and stops the pipeline at the first failure. The pipeline
lowers its own priority (below normal on Windows, nice 10 elsewhere) and the child processes
inherit it, so the machine stays usable. Every parallel step uses at most --jobs worker
processes; the results do not depend on the number of workers.

The estimated runtimes are for 2 workers on a four-core laptop from a clean start (no ridge
caches). The ridge caches in results/cache are keyed by a hash of the data, so later runs
reuse them; delete results/cache (or use --fresh) to rebuild them.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
import paths  # noqa: E402

Stage = namedtuple("Stage", "name minutes commands outputs about optional")


def stage(name, minutes, commands, outputs, about, optional=False):
    return Stage(name, minutes, commands, outputs, about, optional)


FIGURES_FAST = ["fig_inputs", "fig_peer_vs_random", "fig_peer_average", "fig_headline_models", "fig_weights",
                "fig_quartiles", "fig_transaction_costs", "fig_rolling_halflife", "fig_tuned_windows",
                "fig_fee_sensitivity"]

# (name, estimated minutes on 2 workers, [script and arguments, ...], outputs, what it does)
STAGES = [
    stage("random_portfolios", 0.2, [["random_portfolios.py"]],
          "random_portfolios.csv",
          "5,000 seeded random Regulation 28 portfolios (no data read)"),
    stage("ridge_cache", 60, [["final_bandit.py", "cache"]],
          "cache/ridge_<key>.npz, cache/ridge_trendout_intercept_<key>.npz",
          "cross-validated ridge strengths of both tracker forms for every monthly decision and window"),
    stage("grids", 45, [["final_bandit.py", "final", "final", "final_in"]],
          "grids/grid_final*.csv, grids/grid_final*_paths.npz",
          "the two 262-setting tuning grids (trend outside and inside the log)"),
    stage("selection", 2, [["training_mae.py"]],
          "grids/selection_outside.csv, grid_final_with_mae.csv, grid_summary.csv, ...",
          "training MAE of every setting and the selection rule"),
    stage("final_runs", 80, [["final_runs.py", "runs"]],
          "final/{timeseries,turnover,weights,rl_diagnostics,runs}/, final/period_tables.csv",
          "every high-equity model of the thesis (clears final/ first)"),
    stage("siblings", 55, [["sibling_runs.py"]],
          "final_siblings/",
          "the low- and medium-equity categories"),
    stage("dm", 1, [["final_tests.py", "dm"]],
          "final/inference_dm*.csv, dm_h1.csv, bootstrap_ci.csv",
          "Diebold-Mariano tests and bootstrap confidence intervals"),
    stage("features", 1, [["final_tests.py", "features"]],
          "final/feature_zeroing.csv",
          "which context features the tuned bandit's choices depend on (a diagnostic; no table or figure reads it)"),
    stage("gh_run", 10, [["gregory_hansen.py", "run"]],
          "final/gregory_hansen.csv",
          "Gregory-Hansen regime-shift cointegration tests"),
    stage("gh_plot", 0.2, [["gregory_hansen.py", "plot"]],
          "figures/gh_regime_shift_house.png",
          "Gregory-Hansen figure"),
    stage("breaks", 180, [["final_tests.py", "breaks"]],
          "final/two_break.csv, two_break_critical_values.csv, tab_two_breaks.csv",
          "two-break unit-root tests on all 11 spreads (bootstrap and simulated critical values)"),
    stage("ridge_selection", 6, [["ridge_selection.py"]],
          "final/tpr_ridge_picks.npy",
          "where the turnover-penalised ridge puts its ridge strength (recomputed; a diagnostic)"),
    stage("original_spec", 20, [["original_spec.py", "all"]],
          "robustness/scores.csv, robustness/tab_original.csv",
          "the first version of the bandit, the effect of each change and tab:original"),
    stage("first_version_breaks", 15, [["original_spec.py", "breaks"]],
          "robustness/two_break_first_version.csv",
          "two-break tests on the first version's spread (appendix text)"),
    stage("longonly", 5, [["longonly_check.py"]],
          "robustness/longonly_hindsight.csv, robustness/longonly_rolling.csv",
          "hindsight check of the long-only constraint (appendix text)"),
    stage("tables", 0.5, [["thesis_tables.py"]],
          "final/tables.json",
          "the numbers of every numeric thesis table"),
    stage("prose_numbers", 0.2, [["prose_numbers.py"]],
          "final/prose_numbers.csv",
          "numbers quoted in the text that are in no table"),
    stage("figures", 2, [["thesis_figures.py"] + FIGURES_FAST],
          "figures/*.png, figure_data/",
          "the figures drawn from saved results"),
    stage("fig_delta", 5, [["thesis_figures.py", "fig_delta"]],
          "figures/delta_headline_house.png, figure_data/delta_headline.csv",
          "turnover-penalty sensitivity (reruns two models at seven values of delta)"),
    stage("fig_regpath", 30, [["thesis_figures.py", "fig_regularisation_path"]],
          "figures/lambdapaths.png, figure_data/regularisation_paths.npz",
          "two-panel regularisation path (global search over supports in panel b)"),
    # optional: not needed for any table or figure
    stage("checks", 25, [["final_bandit.py", "check"], ["final_runs.py", "check"], ["two_break_unit_root.py", "check"]],
          "screen output only (and ridge caches)",
          "self-checks of the final tracker, the swapped-in fits and the fast two-break statistics",
          optional=True),
    stage("original_grid", 60, [["run_tuning.py", "grid"], ["select_tuning.py"], ["original_spec.py", "check"]],
          "original_grid/grid_scores.csv, grid_paths.npz",
          "the original 910-setting grid, its selection rule and the first version's check against it",
          optional=True),
]
# everything the stages write directly under the results folder (removed by --fresh)
OUTPUT_ENTRIES = ["cache", "grids", "final", "final_siblings", "robustness", "original_grid", "figure_data",
                  "figures", "logs", "random_portfolios.csv"]


def lower_priority():
    """Below-normal priority on Windows, nice 10 on POSIX; child processes inherit it."""
    if os.name == "nt":
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return
        except Exception:
            pass
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:
            pass
    else:
        try:
            os.nice(10)
        except Exception:
            pass


def child_env(jobs):
    env = os.environ.copy()
    env["N_JOBS"] = str(jobs)                 # worker processes of every parallel step
    env["RIDGE_INNER_JOBS"] = str(jobs)       # workers inside run_models' log-TRI ridges
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = "1"                          # one BLAS thread per process
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def fmt_minutes(m):
    if m < 1:
        return "<1 min"
    if m < 90:
        return f"{m:.0f} min"
    return f"{m / 60:.1f} h"


def show_list(stages):
    print(f"{'#':>2}  {'stage':<21} {'est.':>7}  what it does / outputs (under {paths.RESULTS})")
    for n, s in stages:
        tag = "  [optional]" if s.optional else ""
        print(f"{n:>2}  {s.name:<21} {fmt_minutes(s.minutes):>7}  {s.about}{tag}")
        print(f"{'':>2}  {'':<21} {'':>7}    -> {s.outputs}")
        for c in s.commands:
            print(f"{'':>2}  {'':<21} {'':>7}    $ python src/{' '.join(c)}")
    main_total = sum(s.minutes for _, s in stages if not s.optional)
    opt_total = sum(s.minutes for _, s in stages if s.optional)
    print(f"\nestimated total: {fmt_minutes(main_total)} on 2 workers from a clean start "
          f"(optional stages add {fmt_minutes(opt_total)})")


def run_stage(n, s, env):
    log = paths.ensure(paths.LOGS) / f"{n:02d}_{s.name}.log"
    t0 = time.time()
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    with open(log, "w", encoding="utf-8") as f:
        for c in s.commands:
            cmd = [sys.executable, str(SRC / c[0])] + c[1:]
            print(f"{time.strftime('%H:%M:%S')} start {n:02d} {s.name}: python src/{' '.join(c)}", flush=True)
            f.write(f"$ python src/{' '.join(c)}\n")
            f.flush()
            rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=flags)
            if rc != 0:
                f.flush()
                print(f"FAILED {n:02d} {s.name} (exit code {rc}); last lines of {log}:", flush=True)
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                print("\n".join("    " + line for line in lines[-25:]), flush=True)
                return False
    print(f"{time.strftime('%H:%M:%S')} done  {n:02d} {s.name} in {fmt_minutes((time.time() - t0) / 60)}"
          f" (log {log})", flush=True)
    return True


def find(stages, key):
    for n, s in stages:
        if key == s.name or (key.isdigit() and int(key) == n):
            return n
    sys.exit(f"unknown stage {key!r}; see python run_all.py --list")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rebuild every table and figure of the thesis from data/data.xlsx.")
    ap.add_argument("--list", action="store_true", help="list the stages with estimated runtimes and exit")
    ap.add_argument("--from", dest="start", metavar="STAGE", help="resume at this stage (name or number)")
    ap.add_argument("--only", nargs="+", metavar="STAGE", help="run only these stages (names or numbers)")
    ap.add_argument("--optional", action="store_true", help="also run the optional stages")
    ap.add_argument("--fresh", action="store_true",
                    help="delete everything earlier runs wrote under the results folder (caches included) first")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("N_JOBS", 2)),
                    help="worker processes for every parallel step (default: environment N_JOBS, else 2)")
    args = ap.parse_args(argv)
    stages = list(enumerate(STAGES, start=1))
    if args.list:
        show_list(stages)
        return 0
    if args.fresh and (args.start or args.only):
        sys.exit("--fresh rebuilds everything, so it cannot be combined with --from or --only")
    if args.jobs < 1:
        sys.exit("--jobs must be at least 1")
    if args.only:
        keep = {find(stages, k) for k in args.only}
        todo = [(n, s) for n, s in stages if n in keep]
    else:
        first = find(stages, args.start) if args.start else 1
        todo = [(n, s) for n, s in stages if n >= first and (args.optional or not s.optional)]
    if not paths.DATA.exists():
        sys.exit(f"{paths.DATA} not found: the input data are licensed and not included; see data/README.md")

    lower_priority()
    if args.fresh:
        for name in OUTPUT_ENTRIES:
            p = paths.RESULTS / name
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        print(f"--fresh: removed earlier outputs under {paths.RESULTS}", flush=True)
    env = child_env(args.jobs)
    t0 = time.time()
    est = sum(s.minutes for _, s in todo)
    print(f"{len(todo)} stage(s), {args.jobs} worker(s), results in {paths.RESULTS}; "
          f"estimated {fmt_minutes(est)} from a clean start", flush=True)
    for n, s in todo:
        if not run_stage(n, s, env):
            print(f"stopped; fix the problem and resume with: python run_all.py --from {s.name}", flush=True)
            return 1
    print(f"ALL DONE in {fmt_minutes((time.time() - t0) / 60)}: tables in {paths.FINAL / 'tables.json'}, "
          f"figures in {paths.FIGURES}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
