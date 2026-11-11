"""
The turnover line and the original selection rule of the tuning grids.

TO_LINE: a configuration is eligible only if its training turnover is below
150% a year, strictly (train_to < 1.5, where train_to is the mean one-way
turnover over the complete calendar years 2010-2016, a year with no rebalance
counting as zero; run_tuning.segment). The final selection rule in
training_mae.py (lowest training MAE among the trend-outside settings) uses
this line.

select(): the original rule. Among the eligible configurations, the lowest
training joint p-value (the larger of the ADF and Zivot-Andrews p-values of the
training spread); ties within 0.001 go to the lower training tracking error.
It is used by robustness_grid.py. Run on its own,

    python src/select_tuning.py

it prints the configuration the rule picks from paths.ORIGINAL_GRID /
grid_scores.csv (run_tuning.py grid): the first version of the bandit in
Table tab:original. It skips with a message when that file does not exist.
"""
import os

import pandas as pd

import paths

TO_LINE = 1.5
TIE = 0.001


def select(df):
    """Lowest training joint p among training turnover < 150%; ties within 0.001 -> lower training TE.
    Returns (selected row, eligible rows)."""
    elig = df[(df["train_to"] < TO_LINE) & df["train_joint"].notna()]
    best = elig["train_joint"].min()
    tied = elig[elig["train_joint"] <= best + TIE]
    return tied.sort_values(["train_te", "name"]).iloc[0], elig


def main():
    path = os.path.join(paths.ORIGINAL_GRID, "grid_scores.csv")
    if not os.path.exists(path):
        print(f"SKIP: {path} not found (run: python src/run_tuning.py grid)")
        return None
    df = pd.read_csv(path)
    sel, elig = select(df)
    print(f"{len(df)} configurations, {len(elig)} eligible on training (turnover < {TO_LINE:.0%}).")
    print(f"Selected: {sel['name']}")
    cols = [f"{p}_{k}" for p in ("train", "test", "full") for k in ("te", "to", "adf", "za")]
    print(sel[cols].to_string())
    return sel


if __name__ == "__main__":
    main()
