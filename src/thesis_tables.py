"""Collect the numbers of every numeric thesis table into final/tables.json (stage tables of run_all.py).

Each key of tables.json holds one table (or part of one); values are fractions and
p-values as computed (the thesis prints them rounded, turnover and TE in %). Fee drag is
stored with the thesis sign, so -0.0123 prints as -1.23%. Turnover is the mean over
complete calendar years (column 'to').

  headline         tab:core_results          period_tables.csv
  latency          tab:data_latency          period_tables.csv
  subperiod        tab:subperiod             period_tables.csv
  window           tab:window_comparison     period_tables.csv
  variants         tab:variants              period_tables.csv (random_mean = mean over 20 seeds)
  tpr_sensitivity  text only (turnover-penalised ridge settings)
  fee_r2           tab:core_results_all      period_tables.csv and the timeseries (R^2, directional accuracy)
  neighbours       tab:tuning_key, tab:tuning_neighbours   grids/grid_final_with_mae.csv
  siblings_bandit  tab:siblings              final and final_siblings timeseries, period_tables.csv
  siblings_models  tab:siblings_models       final_siblings/period_tables.csv
  two_breaks       tab:two_breaks            gregory_hansen.csv, two_break.csv, two_break_critical_values.csv
  dm               tab:dm                    inference_dm_pooled38.csv (p_bh = p_bh38)
  dm_variants      tab:dm_variants           inference_dm_pooled38.csv
  dm_h1            tab:dm_h1                 inference_dm_pooled38.csv
  ci               tab:ci                    bootstrap_ci.csv
  original         tab:original              robustness/scores.csv, grids/grid_final.csv,
                                             grids/grid_final_in_train_mae_all.csv; and the
                                             first version's two-break tests of the appendix
                                             text (robustness/two_break_first_version.csv)

The tuned bandit is the setting final_bandit.selected_setting() returns: the lowest
training MAE among the trend-outside settings with training turnover below 150% a year
(grids/selection_outside.csv, from training_mae.py). The neighbours are that setting with
one hyperparameter changed at a time.

tab:models has no numbers beyond the settings (grids/selection_outside.csv, run_models.py).

    python src/thesis_tables.py

Inputs: final/ (final_runs.py, final_tests.py, gregory_hansen.py), final_siblings/
(sibling_runs.py), grids/ (final_bandit.py, training_mae.py), robustness/scores.csv
(original_spec.py). Output: final/tables.json. Under a minute.
"""
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

import paths

warnings.filterwarnings("ignore")
START, TEST, END = "2009-01-28", "2017-01-03", "2024-05-15"
SPLIT = "2020-02-24"

EW, RSR = "equal_weight__ma__reb21__tc0", "ridge_returns__ma__w252__reb21__tc0"
TPR, LTN, LTT = "ridge_trend_to__ma__w252__reb21__d1__tc0", "ridge_no_trend__ma__w252__reb21__tc0", "ridge_trend__ma__w252__reb21__tc0"
BAND = "rl_tuned__ma__reb63__tc0"
FQ = {w: f"window_fixedbest__ma__w{w}__reb63__tc0" for w in (126, 252, 378, 504, 756)}
FM = {w: f"window_fixedbest__ma__w{w}__tc0" for w in (126, 252, 378, 504, 756)}
P60, R28 = "passive_60_40__ma__reb21__tc0", "passive_reg28__ma__reb21__tc0"

# tab:original. The first version and its single changes come from original_spec.py
# (robustness/scores.csv, model 'bandit'); all of them measure the turnover penalty against
# the previous target weights except 'drifted_reference', which changes only that. Then
# three settings of the final grids: the first version's setting with all five changes
# (trend outside, delta by n/252, reward weights drift, MAE reward, penalty against the
# drifted weights), the tuned bandit, and the trend-inside bandit with the lowest training MAE.
ORIGINAL_SCORES = {"first_version": "baseline", "trend_outside": "A_trend_outside_log",
                   "delta_n252": "B_delta_scaled_n252", "delta_n504": "B_delta_scaled_n504",
                   "drifting_weights": "C_reward_drifting_weights", "drifted_reference": "D_drifted_reference",
                   "all_three_n504": "A+B504+C"}
ALL_FIVE = "bandit_k0.6_e0_g1_d1_reb63_ce252"      # grid_final.csv
# The selected setting; the neighbour table lists its neighbours in this order.
CANONICAL_PICK = "bandit_k0.6_e0_g0.95_d1_reb63_ce252"
CANONICAL_NEIGHBOURS = (("k", ("0", "1.2", "2.4")), ("g", ("1", "0.9")), ("ce", ("504", "756")),
                        ("d", ("0", "4")), ("reb", ("21",)))


def selected():
    """The tuned bandit's setting name (final_bandit.selected_setting)."""
    import final_bandit as fb
    return fb.selected_setting()


def neighbours_of(sel, g):
    """(label, setting) of the tuned bandit and its one-at-a-time neighbours: the thesis's
    order for the thesis's setting, otherwise every other grid value of each parameter."""
    p = {"k": sel.split("_k")[1].split("_")[0], "g": sel.split("_g")[1].split("_")[0],
         "d": sel.split("_d")[1].split("_")[0], "reb": sel.split("_reb")[1].split("_")[0],
         "ce": sel.split("_ce")[1]}
    if sel == CANONICAL_PICK:
        others = CANONICAL_NEIGHBOURS
    else:
        b = g[g.family == "bandit"]
        col = dict(k="kappa", g="gamma", ce="ce", d="delta", reb="reb")
        others = tuple((k, tuple(f"{v:g}" for v in sorted(b[col[k]].unique()) if f"{v:g}" != p[k]))
                       for k in ("k", "g", "ce", "d", "reb"))
    out = [("tuned", p)]
    for key, vals in others:
        for v in vals:
            out.append((f"{key}={v}", dict(p, **{key: v})))
    return out


def ts(run, d):
    df = pd.read_csv(d / "timeseries" / f"{run}.csv", parse_dates=["datestamp"]).set_index("datestamp").sort_index()
    return df.loc[START:END]


def r2_diracc(run, d):
    df = ts(run, d)
    rp21 = df.portfolio_tri.pct_change(21).dropna()
    rb21 = df.benchmark_tri.pct_change(21).dropna()
    r2 = 1 - ((rb21 - rp21) ** 2).sum() / ((rb21 - rb21.mean()) ** 2).sum()
    rp, rb = df.portfolio_tri.pct_change().dropna(), df.benchmark_tri.pct_change().dropna()
    return float(r2), float((np.sign(rp) == np.sign(rb)).mean())


def split_adf_eg(run, d):
    df = ts(run, d)
    s = np.log(df.portfolio_tri) - np.log(df.benchmark_tri)
    pre, post = s.loc[:SPLIT], s.loc[SPLIT:]
    a = lambda x: float(adfuller(x.values, regression="ct", autolag="AIC")[1])
    eg = float(coint(np.log(df.benchmark_tri).values, np.log(df.portfolio_tri).values, trend="ct")[1])
    return a(s), a(pre.iloc[:-1]), a(post), eg


def pt_rows(d):
    p = pd.read_csv(d / "period_tables.csv")
    return {(r.run, r.window): r for r in p.itertuples()}


def two_breaks():
    """tab:two_breaks: Gregory-Hansen C/S statistic (test, full), LP and LS with bootstrap
    p-values, the capped statistics, and the critical values in the caption."""
    import gregory_hansen as gh
    import two_break_unit_root as tb
    g = pd.read_csv(paths.FINAL / "gregory_hansen.csv").set_index(["run", "period"])
    b = pd.read_csv(paths.FINAL / "two_break.csv").set_index("run")
    cv = pd.read_csv(paths.FINAL / "two_break_critical_values.csv").iloc[0]
    out = {}
    for run, _ in tb.MODELS:
        r = b.loc[run]
        out[run] = dict(gh_test=float(g.loc[(run, "test"), "CS_stat"]), gh_full=float(g.loc[(run, "full"), "CS_stat"]),
                        lp=float(r.lp_stat), lp_p=float(r.lp_p_boot), ls=float(r.ls_stat), ls_p=float(r.ls_p_boot),
                        lp_capped=float(r.lp_capped), ls_capped=float(r.ls_capped))
    out["critical_values"] = {"gh_cs": [gh.CV["C/S"][q] for q in ("1%", "5%", "10%")],
                              "lp": [float(cv[f"lp_cv_{q}"]) for q in ("0.01", "0.05", "0.1")],
                              "ls": [float(cv[f"ls_cv_{q}"]) for q in ("0.01", "0.05", "0.1")]}
    return out


def dm_tables():
    """tab:dm, tab:dm_variants and tab:dm_h1: DM statistic and the 38-pair BH p-value."""
    p = pd.read_csv(paths.FINAL / "inference_dm_pooled38.csv").set_index(["period", "pair"])

    def cell(pair):
        return {per: {"dm": float(p.loc[(per, pair), "dm_stat"]), "p_bh": float(p.loc[(per, pair), "p_bh38"])}
                for per in ("test", "full")}
    vs_bandit = list(FQ.values()) + [RSR, TPR, EW, R28, P60]
    return ({run: cell(f"{BAND} vs {run}") for run in vs_bandit},
            {run: cell(f"{BAND} vs {run}") for run in (FM[252], FM[378], FM[504], FM[756])},
            {run: cell(f"{run} vs {EW}") for run in [TPR, LTN, LTT] + list(FQ.values())})


def ci():
    """tab:ci: full-period TE and fee drag with their 95% intervals."""
    c = pd.read_csv(paths.FINAL / "bootstrap_ci.csv").set_index("run_name")
    runs = [BAND, RSR, EW] + list(FQ.values()) + [R28, P60, TPR, LTT, FM[252]]
    return {run: dict(te=float(c.loc[run, "te_point"]), te_lo=float(c.loc[run, "te_lo"]), te_hi=float(c.loc[run, "te_hi"]),
                      drag=float(c.loc[run, "fee_drag_point"]), drag_lo=float(c.loc[run, "fee_drag_lo"]),
                      drag_hi=float(c.loc[run, "fee_drag_hi"])) for run in runs}


def original(sel):
    """tab:original: test TE, drag, turnover, ADF, ZA and full TE, turnover, ADF, ZA, plus
    the trend-inside bandit's setting and training MAE in its caption. The trend-inside
    row is the bandit with the lowest training MAE among the trend-inside settings with
    training turnover below 150% a year (ties to the name)."""
    import select_tuning as st
    test_k, full_k = ("te", "drag", "to", "adf", "za"), ("te", "to", "adf", "za")
    s = pd.read_csv(paths.ROBUSTNESS / "scores.csv")
    s = s[s.model == "bandit"].set_index(["run", "period"])
    out = {lab: {"test": {k: float(s.loc[(run, "test"), k]) for k in test_k},
                 "full": {k: float(s.loc[(run, "full"), k]) for k in full_k}} for lab, run in ORIGINAL_SCORES.items()}

    def grid_row(r):
        return {"test": {k: float(r[f"test_{k}"]) for k in test_k}, "full": {k: float(r[f"full_{k}"]) for k in full_k}}
    g = pd.read_csv(paths.GRIDS / "grid_final.csv").set_index("name")
    out["all_five_n252"] = grid_row(g.loc[ALL_FIVE])
    out["tuned"] = grid_row(g.loc[sel])
    gi = pd.read_csv(paths.GRIDS / "grid_final_in_train_mae_all.csv")
    inside = gi[(gi.family == "bandit") & (gi.train_to < st.TO_LINE)].sort_values(["train_mae_all", "name"]).iloc[0]
    out["trend_inside"] = grid_row(inside)
    out["trend_inside_name"] = inside["name"]
    out["trend_inside_train_mae"] = float(inside["train_mae_all"])
    # appendix text: two-break tests on the first version's full-period spread
    fv = paths.ROBUSTNESS / "two_break_first_version.csv"
    if fv.exists():
        r = pd.read_csv(fv).iloc[0]
        out["first_version_two_breaks"] = {k: float(r[k]) for k in ("lp_stat", "lp_p_boot", "ls_stat", "ls_p_boot")}
    else:
        print(f"{fv} missing: run `python src/original_spec.py breaks` for the first version's two-break tests")
    return out


def main():
    P = pt_rows(paths.FINAL)
    out = {}

    def m(run, win, keys=("te", "drag", "to", "adf", "za", "hl"), P=P):
        r = P[(run, win)]
        return {k: float(getattr(r, k)) for k in keys}
    out["headline"] = {run: {"test": m(run, "test"), "full": m(run, "full")} for run in (EW, RSR, FQ[252], BAND)}
    out["latency"] = {lag: {"test": m(r, "test"), "full": m(r, "full")} for lag, r in
                      ((0, BAND), (5, "rl_tuned__ma__reb63__lag5__tc0"), (21, "rl_tuned__ma__reb63__lag21__tc0"))}
    out["subperiod"] = {run: {"train": m(run, "train"), "test": m(run, "test")} for run in [EW, RSR] + list(FQ.values()) + [BAND]}
    out["window"] = {run: {"test": m(run, "test"), "full": m(run, "full")} for run in [BAND] + list(FQ.values())}
    var = [TPR, LTN, LTT, P60, R28] + list(FM.values()) + ["rl_tuned_ctxfree__ma__reb63__tc0", "window_oracle__ma__reb63__tc0",
                                                           "window_oracle_anchor__ma__tc0", "rl_tuned__ma__prior_reb21__tc0"]
    out["variants"] = {run: {"test": m(run, "test"), "full": m(run, "full")} for run in var}
    seeds = [f"window_random__ma__seed{s}__reb63__tc0" for s in range(20)]
    out["variants"]["random_mean"] = {w: {k: float(np.mean([m(s, w)[k] for s in seeds])) for k in ("te", "drag", "to", "adf", "za", "hl")}
                                      for w in ("test", "full")}
    sens = [f"ridge_trend_to__ma__w252__reb21__{t}__tc0" for t in ("d0p01", "d0p1", "d0p5", "d1", "d2p0")] + \
        ["ridge_trend_to__ma__w252__reb5__d1__tc0", "ridge_trend_to__ma__w252__reb63__d1__tc0",
         "ridge_trend_to__ma__w126__reb21__d1__tc0", "ridge_trend_to__ma__w504__reb21__d1__tc0"]
    out["tpr_sensitivity"] = {run: {"full": m(run, "full"), "test": m(run, "test")} for run in sens}
    out["fee_r2"] = {}
    for run in [EW, RSR, FQ[252], BAND] + [FQ[w] for w in (126, 378, 504, 756)] + [TPR, LTN, LTT, P60, R28]:
        r2, da = r2_diracc(run, paths.FINAL)
        out["fee_r2"][run] = {"drag": m(run, "full")["drag"], "r2": r2, "dir": da}
    # tuning neighbours of the selected setting, with training MAE
    g = pd.read_csv(paths.GRIDS / "grid_final_with_mae.csv").set_index("name")
    sel = selected()
    assert sel in g.index, f"selected setting {sel} is not in grid_final_with_mae.csv"
    out["neighbours"] = {}
    for lab, c in neighbours_of(sel, g):
        n = f"bandit_k{c['k']}_e0_g{c['g']}_d{c['d']}_reb{c['reb']}_ce{c['ce']}"
        r = g.loc[n]
        out["neighbours"][lab] = {k: float(r[k]) for k in ("test_te", "test_to", "test_adf", "test_za", "train_mae", "train_adf", "train_za",
                                                           "full_te", "full_to", "full_adf", "full_za", "test_mae", "full_mae")}
    # siblings
    S = pt_rows(paths.FINAL_SIBLINGS)
    out["siblings_bandit"] = {}
    for idx in ("ma_le", "ma_me"):
        run = f"rl_tuned__{idx}__reb63__tc0"
        full, pre, post, eg = split_adf_eg(run, paths.FINAL_SIBLINGS)
        out["siblings_bandit"][idx] = dict(adf=full, adf_pre=pre, adf_post=post, za=float(S[(run, "full")].za), eg=eg,
                                           te=float(S[(run, "full")].te), to=float(S[(run, "full")].to))
    full, pre, post, eg = split_adf_eg(BAND, paths.FINAL)
    out["siblings_bandit"]["ma"] = dict(adf=full, adf_pre=pre, adf_post=post, za=m(BAND, "full")["za"], eg=eg,
                                        te=m(BAND, "full")["te"], to=m(BAND, "full")["to"])
    keys5 = ("te", "drag", "to", "adf", "za")
    out["siblings_models"] = {
        idx: {run: {w: m(run, w, keys5, P=S) for w in ("test", "full")}
              for run in (f"rl_tuned__{idx}__reb63__tc0", f"window_fixedbest__{idx}__w378__reb63__tc0",
                          f"window_fixedbest__{idx}__w756__reb63__tc0", f"window_fixedbest__{idx}__w378__tc0",
                          f"window_fixedbest__{idx}__w756__tc0", f"equal_weight__{idx}__reb21__tc0")}
        for idx in ("ma_le", "ma_me")}
    out["two_breaks"] = two_breaks()
    out["dm"], out["dm_variants"], out["dm_h1"] = dm_tables()
    out["ci"] = ci()
    out["original"] = original(sel)
    path = paths.ensure(paths.FINAL) / "tables.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"wrote {path} [tuned bandit {sel}]")
    return out


if __name__ == "__main__":
    main()
