"""Draw every thesis figure from the saved results.

Runs each figure script's main() in turn (the last step of the pipeline, after
final_runs, final_tests and gregory_hansen run). The random portfolios
(random_portfolios.py, seeded) are drawn again first whenever a figure that uses them is
drawn. Each script starts from Matplotlib's defaults and sets its own style, so a figure
comes out the same whether it is drawn here or by running its script on its own.

  script                    thesis figure(s)
  fig_inputs                Assets_house.png, benchmark_select_house.png, monthly_r2_house.png
  fig_peer_vs_random        peer_av_vs_rand_house.png
  fig_peer_average          peer_average_house.png
  fig_regularisation_path   lambdapaths.png (two panels; its global search over supports
                            is the slowest step here)
  fig_headline_models       tri_overlay_house.png, rolling_te_overlay_house.png
  fig_weights               weights1.png, weights2.png
  fig_quartiles             quartile_vs_random_house.png
  fig_transaction_costs     transaction_costs_house.png
  fig_rolling_halflife      rolling_halflife_house.png
  fig_tuned_windows         tuned_windows.png
  fig_fee_sensitivity       fee_sensitivity_house.png
  fig_delta                 delta_headline_house.png (reruns the two penalised headline
                            models at seven values of delta)
  gregory_hansen (plot)     gh_regime_shift_house.png

These are all the figures of the thesis except the reinforcement-learning diagram of the
literature review, which is adapted from a published figure and not drawn from data.

Run:    python src/thesis_figures.py                  # all figures
        python src/thesis_figures.py fig_weights ...  # only the named scripts
Inputs: paths.DATA, paths.FINAL, paths.GRIDS (grid_final.csv, grid_final_paths.npz and
        selection_outside.csv, through final_bandit.selected_setting)
Output: the PNGs in paths.FIGURES; the numbers behind them, and random_portfolios.csv, in
        paths.FIGURE_DATA and paths.RESULTS
"""
import importlib
import sys
import time

import matplotlib
matplotlib.use("Agg")   # write files only, never open windows
import matplotlib.pyplot as plt  # noqa: E402

import paths  # noqa: E402

FIGURE_SCRIPTS = [
    "fig_inputs",
    "fig_peer_vs_random",
    "fig_peer_average",
    "fig_regularisation_path",
    "fig_headline_models",
    "fig_weights",
    "fig_quartiles",
    "fig_transaction_costs",
    "fig_rolling_halflife",
    "fig_tuned_windows",
    "fig_fee_sensitivity",
    "fig_delta",
    "gregory_hansen",
]
USES_RANDOM_PORTFOLIOS = {"fig_peer_vs_random", "fig_quartiles"}


def draw(name):
    """Run one figure script from Matplotlib's defaults, restoring the settings after."""
    t0 = time.time()
    with plt.rc_context():
        plt.rcdefaults()
        module = importlib.import_module(name)
        if name == "gregory_hansen":
            module.plot()
        else:
            module.main()
        plt.close("all")
    print(f"--- {name} done in {time.time() - t0:.0f}s", flush=True)


def main(names=None):
    names = names or FIGURE_SCRIPTS
    unknown = [n for n in names if n not in FIGURE_SCRIPTS]
    if unknown:
        raise SystemExit(f"unknown figure script(s) {unknown}; choose from {FIGURE_SCRIPTS}")
    t0 = time.time()
    if USES_RANDOM_PORTFOLIOS & set(names):
        import random_portfolios
        random_portfolios.main()
    for name in names:
        draw(name)
    print(f"figures in {paths.FIGURES}; done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
