"""
Transaction-cost sensitivity of the four headline models, cost effect only
(thesis Figure fig:fee_sensitivity, fee_sensitivity_house.png).

Each model keeps the weights it chooses at zero cost, and a one-way cost of 0, 5
or 10 basis points is charged on its turnover by metrics.simulate_with_rebased_tri.

- Equal weight (monthly) and the return-space ridge (monthly): the cost never
  enters their decisions, so the saved *__tc5 / *__tc10 runs in <results>/final
  already are this experiment and are scored as they are. Checked: their turnover
  is the same at every cost, and each saved TRI equals the zero-cost TRI times the
  running product of (1 - cost x turnover) over the rebalances so far.
- Fixed 1-year window and tuned bandit (both quarterly, final specification):
  their zero-cost target weights are the final grid's saved paths
  (grid_final_paths.npz: fixed_w252_d1_reb63 and the bandit named by
  final_bandit.selected_setting()), simulated at each cost. The cost never enters
  their decisions either. Checked: the zero-cost rebuild equals the saved final runs
  window_fixedbest__ma__w252__reb63__tc0 and rl_tuned__ma__reb63__tc0.

Every simulated run is clipped to the common start as run_models.run_and_save does
and scored by period_tables.build on the full, training and test periods.

Run:     python src/fig_fee_sensitivity.py [--plot-only]   (--plot-only redraws from the saved CSV)
Inputs:  data/data.xlsx (via run_models.load_data("ma"))
         <results>/grids/selection_outside.csv, grid_final_paths.npz
         <results>/final/{timeseries,turnover}: equal_weight__ma__reb21__tc{0,5,10},
             ridge_returns__ma__w252__reb21__tc{0,5,10}, and the two zero-cost runs above
Outputs: <results>/figures/fee_sensitivity_house.png
         <results>/figure_data/fee_sensitivity_headline.csv   (long: run, window, metrics, model, tc_bps, basis)
         <results>/figure_data/fee_sim/{timeseries,turnover}  the simulated runs, and checks.csv
"""
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402

warnings.filterwarnings("ignore")

FIG_OUT = paths.FIGURES / "fee_sensitivity_house.png"
CSV_OUT = paths.FIGURE_DATA / "fee_sensitivity_headline.csv"
SIM_DIR = paths.FIGURE_DATA / "fee_sim"
TCS = (0, 5, 10)

# (key, label, colour) as in the other headline-model figures
MODELS = [
    ("equal_weight", "Equal weight (monthly)", "#7f7f7f"),
    ("ridge_returns", "Return-space ridge (monthly)", "#2ca02c"),
    ("fixed_w252", "Fixed 1-year window (quarterly)", "#ff7f0e"),
    ("rl_tuned", "Tuned bandit (quarterly)", "#6a0dad"),
]
SAVED = {
    "equal_weight": "equal_weight__ma__reb21__tc{}",
    "ridge_returns": "ridge_returns__ma__w252__reb21__tc{}",
}
REF_TC0 = {
    "rl_tuned": "rl_tuned__ma__reb63__tc0",
    "fixed_w252": "window_fixedbest__ma__w252__reb63__tc0",
}
FIXED_GRID_NAME = "fixed_w252_d1_reb63"

STYLE = {
    "figure.figsize": (10, 5), "figure.dpi": 120, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.35,
    "grid.linestyle": "-", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False, "legend.fontsize": 9,
}


def final_paths(data):
    """Zero-cost target weights (rebalance dates x assets) of the final bandit and the
    fixed 1-year window, from the final grid's saved paths."""
    import final_bandit as fb
    name = fb.selected_setting()
    z = np.load(paths.GRIDS / "grid_final_paths.npz")
    out = {}
    for key, grid_name in (("rl_tuned", name), ("fixed_w252", FIXED_GRID_NAME)):
        idx = z[f"{grid_name}__idx"]
        out[key] = pd.DataFrame(z[f"{grid_name}__W"], index=data.A.index[idx], columns=data.A.columns)
    print(f"final bandit: {name}", flush=True)
    return out


def simulate_and_save(rm, A, B, weights_reb, tc, run, sim_dir):
    from metrics import build_timeseries, save_timeseries, simulate_with_rebased_tri
    p_ret, p_tri, eff_w, turn, _ = simulate_with_rebased_tri(A, weights_reb, tc_bps=tc)
    p_ret, p_tri, eff_w, turn = rm.clip_to_common_start(p_ret, p_tri, eff_w, turn, A.index)
    save_timeseries(run, build_timeseries(p_ret, p_tri, B), sim_dir)
    rm.save_turnover(run, turn, sim_dir)


def _ts(run, rdir):
    return pd.read_csv(os.path.join(rdir, "timeseries", f"{run}.csv"), index_col=0, parse_dates=True)


def _turn(run, rdir):
    return pd.read_csv(os.path.join(rdir, "turnover", f"{run}.csv"), index_col=0, parse_dates=True).iloc[:, 0]


def max_tri_gap(run_a, dir_a, run_b, dir_b):
    a, b = _ts(run_a, dir_a), _ts(run_b, dir_b)
    idx = a.index.intersection(b.index)
    return (len(a), len(b), float(np.abs(a.loc[idx, "portfolio_tri"] - b.loc[idx, "portfolio_tri"]).max()),
            float(np.abs(a.loc[idx, "benchmark_tri"] - b.loc[idx, "benchmark_tri"]).max()))


def saved_cost_only_checks(tmpl, rdir):
    """The saved positive-cost runs of a model whose decisions ignore the cost
    must have the zero-cost turnover and the zero-cost TRI scaled by the running
    product of (1 - c x turnover) over the rebalances so far."""
    out = []
    t0, x0 = _ts(tmpl.format(0), rdir), _turn(tmpl.format(0), rdir)
    for tc in TCS[1:]:
        run = tmpl.format(tc)
        tc_, xc = _ts(run, rdir), _turn(run, rdir)
        same_turn = x0.index.equals(xc.index) and np.allclose(x0.values, xc.values, rtol=0, atol=1e-12)
        keep = (1.0 - tc / 1e4 * x0).reindex(t0.index).fillna(1.0).cumprod()
        implied = t0["portfolio_tri"] * keep
        gap = float(np.abs(implied - tc_["portfolio_tri"].reindex(t0.index)).max())
        out.append(dict(check=f"{run}: turnover equals tc0, max |TRI - cost-only TRI|",
                        value=f"{same_turn}, {gap:.2e}"))
    return out


def build():
    import bandit_tuning as bt
    import period_tables as pt
    import run_models as rm

    final_dir = str(paths.FINAL)
    sim_dir = str(paths.ensure(SIM_DIR))
    data = bt.Data(*rm.load_data("ma"))
    weights = final_paths(data)

    rows, checks = [], []
    for key in ("rl_tuned", "fixed_w252"):
        for tc in TCS:
            run = f"{key}__costonly__tc{tc}"
            simulate_and_save(rm, data.A, data.B, weights[key], tc, run, sim_dir)
            out = pt.build([run], results_dir=sim_dir)
            rows.append(out.assign(model=key, tc_bps=tc, basis="cost only"))
        n_a, n_b, gap_p, gap_b = max_tri_gap(f"{key}__costonly__tc0", sim_dir, REF_TC0[key], final_dir)
        checks.append(dict(check=f"{key} tc0 rebuild vs {REF_TC0[key]}: rows, max |TRI gap|",
                           value=f"{n_a} vs {n_b} rows, portfolio {gap_p:.2e}, benchmark {gap_b:.2e}"))
    for key, tmpl in SAVED.items():
        for tc in TCS:
            out = pt.build([tmpl.format(tc)], results_dir=final_dir)
            rows.append(out.assign(model=key, tc_bps=tc, basis="cost only"))
        checks.extend(saved_cost_only_checks(tmpl, final_dir))

    res = pd.concat(rows, ignore_index=True)
    paths.ensure(paths.FIGURE_DATA)
    res.to_csv(CSV_OUT, index=False)
    pd.DataFrame(checks).to_csv(os.path.join(sim_dir, "checks.csv"), index=False)
    for c in checks:
        print(f"CHECK  {c['check']}: {c['value']}", flush=True)
    return res


def plot(res):
    full = res[(res["window"] == "full") & (res["basis"] == "cost only")]
    with plt.style.context(STYLE, after_reset=True):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for key, label, colour in MODELS:
            s = full[full["model"] == key].sort_values("tc_bps")
            axes[0].plot(s["tc_bps"], 100 * s["drag"], marker="o", lw=1.5, color=colour, label=label)
            axes[1].plot(s["tc_bps"], 100 * s["te"], marker="o", lw=1.5, color=colour, label=label)
        ax = axes[0]
        ax.axhline(0, color="k", lw=0.5, linestyle="--")
        # Label at the left end: the return-space ridge crosses zero near 4 bps and
        # sits just above the line at 10 bps.
        ax.annotate("Peer average (zero fee drag)", xy=(TCS[0], 0), xytext=(4, 4),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8.5, color="0.25")
        ax.set_xlabel("Transaction cost (bps, one-way)")
        ax.set_ylabel("Fee drag p.a. (%): negative = beats peers")
        ax.set_title("Fee Drag vs Transaction Cost")
        ax.set_xticks(TCS)
        ax = axes[1]
        ax.set_xlabel("Transaction cost (bps, one-way)")
        ax.set_ylabel("Annualised tracking error (%)")
        ax.set_title("Tracking Error vs Transaction Cost")
        ax.set_xticks(TCS)
        lo, hi = 100 * full["te"].min(), 100 * full["te"].max()
        ax.set_ylim(lo - 0.15, hi + 0.15)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8.5)
        paths.ensure(FIG_OUT.parent)
        fig.savefig(FIG_OUT)
        plt.close(fig)
    print(f"wrote {FIG_OUT}")


def main(plot_only=False):
    res = pd.read_csv(CSV_OUT) if plot_only else build()
    plot(res)
    show = res[res["window"].isin(["full", "test"])].assign(
        te=lambda t: (100 * t["te"]).round(3), drag=lambda t: (100 * t["drag"]).round(3),
        to=lambda t: (100 * t["to"]).round(1))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(show[["model", "tc_bps", "window", "te", "drag", "to"]]
              .sort_values(["window", "model", "tc_bps"]).to_string(index=False))


if __name__ == "__main__":
    main(plot_only="--plot-only" in sys.argv[1:])
