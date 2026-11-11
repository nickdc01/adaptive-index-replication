"""
Regularisation paths of the return-space ridge (thesis Figure fig:lambdapaths, lambdapaths.png).

Two panels at one rebalance date (30 June 2015, 252-day window), both solving the
return-space problem under the model's constraints w >= 0, sum(w) = 1:

  (a) the ridge, as the penalty strength r varies:
          min ||X w - y||^2 + r * sum_j w_j^2
      As r grows the solution is pulled to the equal-weight centre of the simplex.
  (b) the same problem with the L1 term of the elastic net replaced by sum_j w_j^q,
      q = 0.5, with mixing parameter alpha = 0.5, as lambda varies:
          min ||X w - y||^2 + lambda * [alpha * sum_j w_j^q + (1 - alpha) / 2 * sum_j w_j^2]
      Under the constraints an L1 term is constant (sum w = 1) and cannot select assets;
      with q < 1 the penalty favours holding few assets. It is not used in any result.

The penalty in (b) is not convex, so a local solver can stop at a poor local minimum.
Each lambda is therefore solved globally over supports (the sets of assets held): the
minimum is taken over every support of one to three assets, every support met on
multi-start sweeps of lambda upwards and downwards, and every support with one asset
removed from those, each minimised from its centroid and from the best point found so
far. Rounds repeat until no lambda improves. The check stage compares the result at
selected lambdas with the minimum over all supports of up to five assets.

Run:     python src/fig_regularisation_path.py           # compute the paths, then plot
         python src/fig_regularisation_path.py compute   # compute the paths only
         python src/fig_regularisation_path.py check     # path minus the enumeration over
                                                         # supports of up to five assets
         python src/fig_regularisation_path.py plot      # redraw from the saved paths
The global search runs on N_JOBS worker processes (environment variable, default 2) and
takes tens of minutes on two; the check takes about five. The result does not depend on
the number of workers.
Input:   data/data.xlsx (sheets X and Y, via run_models.load_data("ma"))
Outputs: <results>/figures/lambdapaths.png
         <results>/figure_data/regularisation_paths.npz   both paths: penalty grids, weights
                                                          and, for (b), the objective values
"""
import itertools
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import paths  # noqa: E402
from run_models import load_data  # noqa: E402

OUT = paths.FIGURES / "lambdapaths.png"
NPZ = paths.FIGURE_DATA / "regularisation_paths.npz"
DATE = pd.Timestamp("2015-06-30")   # representative mid-sample rebalance date
WINDOW = 252                         # 1-year lookback, as in the baseline models
RIDGE_GRID = np.logspace(-5, 2, 60)  # panel (a): r
Q, ALPHA = 0.5, 0.5                  # panel (b): exponent of the selection term and mixing parameter
LAMBDA_GRID = np.logspace(-8, 0, 81)  # panel (b): lambda
PLOT_RANGE = (-7.0, 0.0)             # panel (b) is drawn for log10(lambda) in this range
OPTS = {"maxiter": 1000, "ftol": 1e-14}
N_JOBS = int(os.environ.get("N_JOBS", 2))

STYLE = {
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False, "legend.fontsize": 8,
}


def window_returns():
    """Asset and benchmark daily returns over the 252-day window ending on DATE, from the
    shared loader exactly as the models receive them; also the asset names and the date."""
    asset_ret, Y = load_data("ma")
    common = asset_ret.index.intersection(Y.index)
    asset_ret, bench = asset_ret.loc[common], Y.loc[common]
    end_loc = asset_ret.index.get_indexer([DATE], method="ffill")[0]
    ra = asset_ret.iloc[end_loc - WINDOW + 1: end_loc + 1].values
    rb = bench.iloc[end_loc - WINDOW + 1: end_loc + 1].values
    return ra, rb, list(asset_ret.columns), asset_ret.index[end_loc].date()


# -----------------------------
# Panel (a): the ridge
# -----------------------------
def solve(ra, rb, r):
    """Constrained ridge weights for penalty r (NaN if the solver fails)."""
    n = ra.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * n

    def loss(w):
        return float(np.sum((ra @ w - rb) ** 2) + r * np.sum(w ** 2))
    res = minimize(loss, np.ones(n) / n, bounds=bnds, constraints=cons,
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    return res.x if res.success else np.full(n, np.nan)


def ridge_path(ra, rb):
    return np.vstack([solve(ra, rb, r) for r in RIDGE_GRID])


# -----------------------------
# Panel (b): the penalty with q < 1, solved globally over supports
# -----------------------------
def penalty(v):
    return ALPHA * np.sum(np.abs(v) ** Q) + (1 - ALPHA) / 2 * np.sum(v ** 2)


def objective(w, lam, ra, rb):
    return float(np.sum((ra @ w - rb) ** 2) + lam * penalty(w))


def solve_support(S, lam, ra, rb, starts):
    """Minimise inside support S (the weights outside S fixed at zero), from the support's
    centroid and from each full-length start restricted to S. Returns (objective, w)."""
    n = ra.shape[1]
    S = list(S)
    k = len(S)
    if k == 1:
        w = np.zeros(n)
        w[S[0]] = 1.0
        return objective(w, lam, ra, rb), w
    Xs = ra[:, S]

    def loss(v):
        return float(np.sum((Xs @ v - rb) ** 2) + lam * penalty(v))
    x0s = [np.full(k, 1.0 / k)]
    for wf in starts:
        v = np.clip(wf[S], 0, None)
        if v.sum() > 1e-6:
            x0s.append(0.9 * v / v.sum() + 0.1 / k)
    best = (np.inf, None)
    for x0 in x0s:
        res = minimize(loss, x0, bounds=[(0, 1)] * k, constraints={"type": "eq", "fun": lambda v: v.sum() - 1},
                       method="SLSQP", options=OPTS)
        v = np.clip(res.x, 0, 1)
        if v.sum() > 0.5:   # back onto the simplex (SLSQP can end a hair off the budget constraint)
            v = v / v.sum()
            o = loss(v)
            if o < best[0]:
                w = np.zeros(n)
                w[S] = v
                best = (o, w)
    return best


def unrestricted(lam, ra, rb, starts):
    """Best of SLSQP over all assets from each start. Returns (objective, w)."""
    n = ra.shape[1]
    best = (np.inf, None)
    for x0 in starts:
        res = minimize(lambda w: objective(w, lam, ra, rb), x0, bounds=[(0, 1)] * n,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP", options=OPTS)
        w = np.clip(res.x, 0, 1)
        if w.sum() > 0.5:
            w = w / w.sum()
            o = objective(w, lam, ra, rb)
            if o < best[0]:
                best = (o, w)
    return best


def support(w, tol=1e-7):
    return tuple(int(j) for j in np.flatnonzero(w > tol))


def search_lambda(lam, candidates, ra, rb, starts):
    """Best over the candidate supports at one lambda. Returns (objective, w)."""
    best = (np.inf, None)
    for S in candidates:
        o, w = solve_support(S, lam, ra, rb, starts)
        if w is not None and o < best[0]:
            best = (o, w)
    return best


def global_path(ra, rb):
    """Weights and objective values of panel (b) on LAMBDA_GRID."""
    n = ra.shape[1]
    ew = np.ones(n) / n
    # 1) multi-start sweeps (lambda upwards, then downwards) seed the pool of supports
    pool, prev, sweep = set(), None, {}
    for lam in LAMBDA_GRID:
        o, w = unrestricted(lam, ra, rb, [ew] + ([prev] if prev is not None else []))
        prev = w
        sweep[lam] = (o, w)
        pool.add(support(w))
    prev = None
    for lam in LAMBDA_GRID[::-1]:
        starts = [ew] + ([prev] if prev is not None else []) + [np.eye(n)[j] * 0.9 + 0.1 / n for j in range(n)]
        o, w = unrestricted(lam, ra, rb, starts)
        prev = w
        if o < sweep[lam][0]:
            sweep[lam] = (o, w)
        pool.add(support(w))
    # 2) rounds over the supports of 1-3 assets, the pool, and the pool with one asset removed
    small = set()
    for k in (1, 2, 3):
        small |= set(itertools.combinations(range(n), k))
    best = {lam: sweep[lam] for lam in LAMBDA_GRID}
    for rnd in range(1, 6):
        drop1 = {tuple(x for x in S if x != j) for S in pool for j in S if len(S) > 1}
        candidates = sorted(small | pool | drop1)
        print(f"round {rnd}: {len(candidates)} candidate supports per lambda", flush=True)
        out = Parallel(n_jobs=N_JOBS)(delayed(search_lambda)(lam, candidates, ra, rb, [best[lam][1]])
                                      for lam in LAMBDA_GRID)
        changed = 0
        for lam, (o, w) in zip(LAMBDA_GRID, out):
            if o < best[lam][0] - 1e-13:
                changed += 1
                best[lam] = (o, w)
            pool.add(support(best[lam][1]))
        print(f"  improved at {changed} lambdas", flush=True)
        if changed == 0:
            break
    W = np.vstack([best[lam][1] for lam in LAMBDA_GRID])
    obj = np.array([best[lam][0] for lam in LAMBDA_GRID])
    return W, obj


def compute():
    ra, rb, names, date_label = window_returns()
    W, obj = global_path(ra, rb)
    Wr = ridge_path(ra, rb)
    paths.ensure(NPZ.parent)
    np.savez(NPZ, lams=LAMBDA_GRID, W=W, O=obj, rs=RIDGE_GRID, Wr=Wr, names=np.array(names), date=str(date_label))
    held = (W > 0.01).sum(axis=1)
    print("assets above 1%: " + " ".join(f"{np.log10(lam):.1f}:{h}" for lam, h in zip(LAMBDA_GRID, held)), flush=True)
    for k in range(0, len(LAMBDA_GRID), 4):
        print(f"  {np.log10(LAMBDA_GRID[k]):5.1f} " + ", ".join(
            f"{names[j]} {W[k, j]:.3f}" for j in np.argsort(-W[k]) if W[k, j] > 1e-7), flush=True)
    print(f"wrote {NPZ}", flush=True)


def _enumerate(lam, ra, rb, w_best, max_assets=5):
    """Best over every support of up to max_assets assets at one lambda."""
    n = ra.shape[1]
    best = (np.inf, None)
    for k in range(1, max_assets + 1):
        for S in itertools.combinations(range(n), k):
            o, w = solve_support(S, lam, ra, rb, [w_best])
            if w is not None and o < best[0]:
                best = (o, w)
    return best


def check():
    """At every eighth lambda and the last, the saved objective minus the minimum over all
    supports of up to five assets (zero or negative when the path is the global minimum
    among those supports)."""
    ra, rb, _, _ = window_returns()
    z = np.load(NPZ)
    lams, W, obj = z["lams"], z["W"], z["O"]
    idx = sorted(set(range(0, len(lams), 8)) | {len(lams) - 1})
    out = Parallel(n_jobs=N_JOBS)(delayed(_enumerate)(lams[m], ra, rb, W[m]) for m in idx)
    for m, (o, w) in zip(idx, out):
        print(f"{np.log10(lams[m]):5.1f} path {obj[m]:.8e} ({len(support(W[m]))} assets)  "
              f"supports of up to 5 {o:.8e} ({len(support(w))} assets)  path - enumeration {obj[m] - o:+.2e}",
              flush=True)


def plot():
    z = np.load(NPZ)
    names, date_label = list(z["names"]), str(z["date"])
    lo, hi = PLOT_RANGE
    keep = (np.log10(z["lams"]) >= lo - 1e-9) & (np.log10(z["lams"]) <= hi + 1e-9)
    n = len(names)
    with plt.style.context(STYLE, after_reset=True):
        colors = plt.cm.tab20.colors
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        panels = ((axes[0], np.log10(z["rs"]), z["Wr"], r"$\log_{10}$ of ridge penalty strength $r$", "(a) Ridge"),
                  (axes[1], np.log10(z["lams"][keep]), z["W"][keep], r"$\log_{10}$ of penalty strength $\lambda$",
                   f"(b) Penalty with $q = {Q:g}$, $\\alpha = {ALPHA:g}$"))
        for ax, x, W, xlabel, title in panels:
            for j, col in enumerate(names):
                ax.plot(x, W[:, j], lw=1.5, color=colors[j % 20], label=col)
            ax.axhline(1.0 / n, color="grey", ls=":", lw=1.2, label=f"Equal weight ($1/{n}$)")
            ax.set_xlabel(xlabel)
            ax.set_title(title, fontsize=11)
            ax.set_xlim(x[0], x[-1])
        axes[0].set_ylabel("Asset weight")
        axes[0].set_ylim(bottom=0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(f"Regularisation paths ({date_label}, 252-day window)", fontweight="bold", fontsize=12)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        paths.ensure(OUT.parent)
        fig.savefig(OUT)
        plt.close(fig)
    print(f"wrote {OUT}  (date {date_label}, {n} assets; panel (b) for log10 lambda {lo:g} to {hi:g})")


def main(stage=None):
    """No stage: compute the paths and plot them. 'compute', 'check' or 'plot': that stage only."""
    if stage not in (None, "compute", "check", "plot"):
        raise SystemExit(f"unknown stage {stage!r}: choose compute, check or plot")
    if stage in (None, "compute"):
        compute()
    if stage == "check":
        check()
    if stage in (None, "plot"):
        plot()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
