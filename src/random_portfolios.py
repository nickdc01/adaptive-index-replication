"""
Draw the 5,000 random Regulation 28 portfolios used to judge how hard the peer
average is to beat or track (the Burns (2004) random-portfolio comparison).

Feeds: peer_av_vs_rand_house.png (fig:peer_vs_random, fig_peer_vs_random.py) and
quartile_vs_random_house.png (fig:quartile_random, fig_quartiles.py).

Constraints (Regulation 28):
  - equity exposure (the five equity sleeves, not property) <= 75%
  - offshore exposure (all hedged and unhedged sleeves)     <= 45%
  - listed property                                         <= 25%
Sampling: uniform over the simplex via Dirichlet(1, ..., 1) (normalised
exponentials, seed 123), then rejection on the three caps; the first 5,000
accepted portfolios are kept. No data are read, so the output holds no
licensed data.

Run:     python src/random_portfolios.py
Output:  <results>/random_portfolios.csv (one row per portfolio, one column per asset)
"""
import numpy as np
import pandas as pd

import paths

OUT = paths.RESULTS / "random_portfolios.csv"

ASSETS = [
    "SA Bonds", "SA Property", "SA Cash & Income", "SA Equities",
    "Unhedged Bonds", "Unhedged Equities", "Unhedged Cash", "Unhedged EM Equities",
    "Hedged Bonds", "Hedged Equities", "Hedged EM Equities",
]
EQUITY = ["SA Equities", "Unhedged Equities", "Unhedged EM Equities",
          "Hedged Equities", "Hedged EM Equities"]
OFFSHORE = ["Unhedged Bonds", "Unhedged Equities", "Unhedged Cash",
            "Unhedged EM Equities", "Hedged Bonds", "Hedged Equities", "Hedged EM Equities"]
PROPERTY = ["SA Property"]

N_TARGET = 5000
OVERSAMPLE = 500_000
SEED = 123


def main():
    idx = {a: i for i, a in enumerate(ASSETS)}
    rng = np.random.default_rng(SEED)

    X = rng.exponential(size=(OVERSAMPLE, len(ASSETS)))
    W = X / X.sum(axis=1, keepdims=True)

    eq = W[:, [idx[a] for a in EQUITY]].sum(1)
    off = W[:, [idx[a] for a in OFFSHORE]].sum(1)
    prop = W[:, [idx[a] for a in PROPERTY]].sum(1)

    keep = (eq <= 0.75) & (off <= 0.45) & (prop <= 0.25)
    W_ok = W[keep]
    print(f"Valid portfolios: {len(W_ok)} of {OVERSAMPLE} ({100*len(W_ok)/OVERSAMPLE:.1f}% accept)")
    if len(W_ok) < N_TARGET:
        raise SystemExit("Not enough valid portfolios; increase OVERSAMPLE.")

    sel = W_ok[:N_TARGET]
    # sanity: constraints hold for every selected portfolio
    e = sel[:, [idx[a] for a in EQUITY]].sum(1)
    o = sel[:, [idx[a] for a in OFFSHORE]].sum(1)
    p = sel[:, [idx[a] for a in PROPERTY]].sum(1)
    assert (e <= 0.75 + 1e-12).all() and (o <= 0.45 + 1e-12).all() and (p <= 0.25 + 1e-12).all()
    assert np.allclose(sel.sum(1), 1.0)
    print(f"equity max={e.max():.3f}  offshore max={o.max():.3f}  property max={p.max():.3f}  (caps 0.75/0.45/0.25)")

    paths.ensure(OUT.parent)
    pd.DataFrame(sel, columns=ASSETS).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({N_TARGET} portfolios)")


if __name__ == "__main__":
    main()
