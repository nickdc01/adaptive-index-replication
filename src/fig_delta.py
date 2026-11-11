"""
Turnover-penalty (delta) sensitivity (thesis Figure fig:delta_sensitivity,
delta_headline_house.png).

Covers the two headline models that carry the penalty, both quarterly: the fixed
1-year window and the tuned bandit. Both use the final tracker (final_bandit.FitB:
the log-TRI ridge with an unpenalised constant and the trend outside the log, and the
penalty delta * n / 252 * ||w - w_drift||^2, where w_drift is the previous target
weights drifted to the rebalance date), and the bandit is rewarded on the forward MAE
alone (final_bandit.SPECS["final"]). Equal weight and the return-space ridge have no
penalty, so they are not in this figure.

For each delta the model is rerun with everything else at its final setting (the
bandit's other settings are those of final_bandit.selected_setting(), the trend-outside
setting with the lowest training MAE among those with training turnover below 150% a
year; the ridge strengths do not depend on delta) and scored as the tuning grid scores
it (run_tuning.segment), so delta = 1 reproduces the grid_final.csv rows. As a check,
the delta = 1 weights must equal the saved paths in grid_final_paths.npz.

Run:     python src/fig_delta.py               (14 model runs; under a minute with warm caches)
         python src/fig_delta.py --plot-only   (redraw from the saved CSV)
Inputs:  data/data.xlsx (via run_tuning.load)
         <results>/grids/selection_outside.csv, grid_final.csv, grid_final_paths.npz
         <results>/cache/ ridge caches (built if missing, which is slow)
Outputs: <results>/figures/delta_headline_house.png
         <results>/figure_data/delta_headline.csv   TE, turnover, fee drag, ADF/ZA p, half-life
                                                    by model, delta and period (full, test)
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bandit_tuning as bt  # noqa: E402
import final_bandit as fb  # noqa: E402
import paths  # noqa: E402
import run_tuning as rt  # noqa: E402

OUT = paths.FIGURES / "delta_headline_house.png"
CSV_OUT = paths.FIGURE_DATA / "delta_headline.csv"
SPEC_NAME = "final"
SPEC = fb.SPECS[SPEC_NAME]
DELTAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0]
FIXED_GRID_NAME = "fixed_w252_d1_reb63"   # the fixed 1-year window in grid_final
MODEL_STYLE = [("Fixed 1-year window (quarterly)", "#ff7f0e"), ("Tuned bandit (quarterly)", "#6a0dad")]

STYLE = {
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False, "legend.fontsize": 9,
}


def run_final(data, ridge_out, cfg):
    """Weight path of a fixed window or a bandit under the final specification."""
    fit = fb.FitB(data, trend_out=SPEC["trend_out"], n_ref=SPEC["n_ref"], ridge_out=ridge_out,
                  to_ref=SPEC["to_ref"])
    if cfg["family"] == "fixed":
        return fb.run_fixed_v(data, int(cfg["window"]), fit, delta=float(cfg["delta"]), reb=int(cfg["reb"]))
    return fb.run_bandit_m(data, cfg, fit, drift=SPEC["drift"], reward=SPEC["reward"],
                           annualise=SPEC["annualise"], standardise=SPEC["standardise"])


def main(deltas=None):
    deltas = DELTAS if deltas is None else [float(d) for d in deltas]
    data = rt.load()
    ridge_out = fb.build_ridge_cache_out(data, bt.schedule(data.n, 21))
    name, cfg = fb.selected_config()   # the tuned bandit (final_bandit.selected_setting)
    base = {k: v for k, v in cfg.items() if k != "delta"}
    models = [   # (label as in MODEL_STYLE, grid name, settings at a given delta)
        (MODEL_STYLE[0][0], FIXED_GRID_NAME, lambda d: dict(family="fixed", window=252, reb=63, delta=d)),
        (MODEL_STYLE[1][0], name, lambda d: dict(base, delta=d)),
    ]
    grid_paths = paths.GRIDS / f"grid_{SPEC_NAME}_paths.npz"
    z = np.load(grid_paths) if grid_paths.exists() else None
    print(f"final bandit: {name}", flush=True)

    rows = []
    for label, grid_name, cfg_of in models:
        for d in deltas:
            idx, chosen, W = run_final(data, ridge_out, cfg_of(d))
            if d == 1.0:
                if z is None:
                    print(f"CHECK skipped: {grid_paths} not found", flush=True)
                else:
                    same = (np.array_equal(np.asarray(W), z[f"{grid_name}__W"])
                            and list(chosen) == list(z[f"{grid_name}__win"]))
                    assert same, f"{label} at delta 1 differs from {grid_name} in {grid_paths.name}"
                    print(f"CHECK {label} at delta 1 equals {grid_name} in {grid_paths.name}", flush=True)
            p_ret, p_tri, _, turn = rt.simulate(data, idx, W)
            for tag, seg in (("full", rt.FULL), ("test", rt.TEST)):
                m = rt.segment(data, p_ret, p_tri, turn, seg)
                rows.append(dict(model=label, delta=d, period=tag, te=m["te"], to=m["to"], drag=m["drag"],
                                 adf=m["adf"], za=m["za"], hl=m["hl"]))
            print(f"{label:34s} delta {d:5}: full TE {rows[-2]['te']:.4%} TO {rows[-2]['to']:.1%} | "
                  f"test TE {rows[-1]['te']:.4%} TO {rows[-1]['to']:.1%}", flush=True)
    df = pd.DataFrame(rows)
    paths.ensure(paths.FIGURE_DATA)
    df.to_csv(CSV_OUT, index=False)
    plot(df, deltas)
    print(f"wrote {CSV_OUT}")


def plot(df, deltas=None):
    """TE and turnover against delta, full and test periods, from the rows main() writes."""
    deltas = DELTAS if deltas is None else deltas
    x = np.arange(len(deltas))
    with plt.style.context(STYLE, after_reset=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for label, colour in MODEL_STYLE:
            for tag, ls, mk in (("full", "-", "o"), ("test", "--", "s")):
                s = df[(df.model == label) & (df.period == tag)].set_index("delta").loc[deltas]
                lab = f"{label}, {'full' if tag == 'full' else 'test'} period"
                ax1.plot(x, 100 * s["te"], ls, marker=mk, color=colour, lw=1.4, ms=4.5, label=lab)
                ax2.plot(x, 100 * s["to"], ls, marker=mk, color=colour, lw=1.4, ms=4.5, label=lab)
        for ax, title, ylab in ((ax1, "Tracking Error against the Turnover Penalty", "Annualised tracking error (%)"),
                                (ax2, "Turnover against the Turnover Penalty", "Turnover (% a year)")):
            if 1.0 in deltas:
                ax.axvspan(deltas.index(1.0) - 0.25, deltas.index(1.0) + 0.25, color="0.85", zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{d:g}" for d in deltas])
            ax.set_xlabel("Turnover penalty $\\delta$ (shaded: $\\delta = 1$, the headline setting)")
            ax.set_title(title)
            ax.set_ylabel(ylab)
        h, lab = ax1.get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout()
        paths.ensure(OUT.parent)
        fig.savefig(OUT)
        plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if "--plot-only" in sys.argv[1:]:
        plot(pd.read_csv(CSV_OUT))
    else:
        main()
