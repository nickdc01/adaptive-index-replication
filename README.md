# Adaptive replication of a South African multi-asset peer average

Code for the MCom dissertation

> **Adaptive Replication of the South African Multi-Asset High-Equity Peer Average: Non-Negative Ridge Baselines and a Contextual-Bandit for Estimation-Window Selection**
>
> Nicholas de Clercq, Stellenbosch University, 2026 (MCom in Financial Risk Management)

The dissertation asks how well a long-only portfolio of 11 liquid asset-class indices (South African and offshore equities, bonds, cash and listed property) can replicate the ASISA South African Multi-Asset High Equity peer-category average: the daily return, net of fees, of the funds in that category. The funds' holdings are not observed, so each replicating portfolio is fitted to the average's return series alone, walk-forward, using only data available at each date. It compares equal weight, a return-space ridge regression, ridge regressions on log total return indices with an unpenalised constant and a trend for the fee drag (Equation 3.4, with a turnover penalty against the weights held before each rebalance), fixed estimation windows, and a discounted LinUCB contextual bandit that picks the estimation window at each rebalance and is rewarded on the forward mean absolute tracking error of the spread around its fitted drift. The bandit's settings are chosen on a training period (to 30 December 2016) and every model is reported on the later test period (3 January 2017 to 15 May 2024) and the full period (from 28 January 2009). The spreads are judged as a cointegration problem: tracking error and fee drag, plus unit-root, structural-break and cointegration tests of the log spread.

This repository holds the code that produces every table and figure of the dissertation from one input file.

## Data (not included)

The input data are licensed and are not part of this repository:

- the daily returns of the peer-category averages come from Profile Data (Pty) Ltd and are used with permission;
- the asset-class total return indices come from Bloomberg.

To run the code, put your own copy of the data in `data/data.xlsx`. The expected sheets and columns are in [data/README.md](data/README.md). Everything the code writes (tables, figures, caches and logs) is derived from these data, so it goes to `results/`, which is git-ignored.

## Install and run

Python 3.12.8 was used. `requirements.txt` pins the direct dependencies and `constraints.txt` the packages they pull in, at the versions the thesis numbers were produced with.

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
python run_all.py               # every stage, in order, from data/data.xlsx
```

`run_all.py` runs the stages listed below in order, logs each one to `results/logs/`, and stops at the first failure. It lowers its own process priority and uses 2 worker processes unless told otherwise, so the machine stays usable. The full run takes about 9 hours on 2 workers from a clean start, most of it in the ridge caches, the final runs and the bootstrap of the break tests; it is roughly twice as fast on 4 workers. The numbers do not depend on the number of workers.

```
python run_all.py --list              # stages, outputs and estimated runtimes
python run_all.py --from final_runs   # resume at a stage
python run_all.py --only tables figures
python run_all.py --jobs 4            # more worker processes
python run_all.py --fresh             # delete earlier outputs, caches included, then run everything
python run_all.py --optional          # also the self-checks and the original 910-setting grid
```

Set the environment variable `REPLICATION_RESULTS` to write the results somewhere other than `results/`. Each script in `src/` can also be run on its own; its docstring says what it reads, what it writes and how long it takes.

| # | Stage | Script | What it does | Est. time (2 workers) |
|---|---|---|---|---|
| 1 | `random_portfolios` | `random_portfolios.py` | 5,000 seeded random Regulation 28 portfolios | under 1 min |
| 2 | `ridge_cache` | `final_bandit.py cache` | cross-validated ridge strengths of both tracker forms for every monthly decision and window | 1 h |
| 3 | `grids` | `final_bandit.py final` | the two 262-setting tuning grids (trend outside and inside the log) | 45 min |
| 4 | `selection` | `training_mae.py` | training MAE of every setting and the selection rule | 2 min |
| 5 | `final_runs` | `final_runs.py runs` | every high-equity model in the dissertation | 80 min |
| 6 | `siblings` | `sibling_runs.py` | the low- and medium-equity categories | 55 min |
| 7 | `dm` | `final_tests.py dm` | Diebold-Mariano tests and bootstrap confidence intervals | 1 min |
| 8 | `features` | `final_tests.py features` | context features the bandit's choices depend on (a diagnostic; no table or figure reads it) | 1 min |
| 9 | `gh_run` | `gregory_hansen.py run` | Gregory-Hansen cointegration tests with a regime shift | 10 min |
| 10 | `gh_plot` | `gregory_hansen.py plot` | the Gregory-Hansen figure | under 1 min |
| 11 | `breaks` | `final_tests.py breaks` | Lumsdaine-Papell and Lee-Strazicich two-break tests on 11 spreads | 3 h |
| 12 | `ridge_selection` | `ridge_selection.py` | where the turnover-penalised ridge puts its ridge strength (a diagnostic; no table or figure reads it) | 6 min |
| 13 | `original_spec` | `original_spec.py all` | the first version of the bandit and the effect of each change | 20 min |
| 14 | `first_version_breaks` | `original_spec.py breaks` | two-break tests on the first version's spread (appendix text) | 15 min |
| 15 | `longonly` | `longonly_check.py` | hindsight check of the long-only constraint (appendix text) | 5 min |
| 16 | `tables` | `thesis_tables.py` | the numbers of every numeric table, in `final/tables.json` | under 1 min |
| 17 | `prose_numbers` | `prose_numbers.py` | numbers quoted in the text that are in no table | under 1 min |
| 18 | `figures` | `thesis_figures.py` | the figures drawn from saved results | 2 min |
| 19 | `fig_delta` | `thesis_figures.py fig_delta` | turnover-penalty sensitivity (reruns two models at seven values of delta) | 5 min |
| 20 | `fig_regpath` | `thesis_figures.py fig_regularisation_path` | two-panel regularisation path | 30 min |

The ridge caches in `results/cache` are keyed by a hash of the data and are reused by later runs.

## Where each table and figure comes from

Paths are under `results/`. `tables.json` (written by `thesis_tables.py`) holds the numbers of every numeric table, one key per table, as fractions and p-values before rounding.

### Tables

| Table (short caption) | Label | Stage and script | Output |
|---|---|---|---|
| Models compared (three parts) | `tab:models` | `selection` (`training_mae.py`); model settings in `run_models.py` and `final_runs.py` | `grids/selection_outside.csv` (the selected bandit setting); no other numbers |
| Headline comparison | `tab:core_results` | `final_runs`, then `tables` | `final/period_tables.csv`; `tables.json` key `headline` |
| The tuned bandit with reporting lags | `tab:data_latency` | `final_runs`, then `tables` | `tables.json` key `latency` |
| Main models over the training and test periods | `tab:subperiod` | `final_runs`, then `tables` | `tables.json` key `subperiod` |
| Tuned bandit against fixed estimation windows | `tab:window_comparison` | `final_runs`, then `tables` | `tables.json` key `window` |
| Tuned bandit and its discount and horizon neighbours | `tab:tuning_key` | `grids`, `selection`, then `tables` | `grids/grid_final_with_mae.csv`; `tables.json` key `neighbours` |
| Break tests of the spread | `tab:two_breaks` | `gh_run` and `breaks` (`gregory_hansen.py`, `final_tests.py`, `two_break_unit_root.py`), then `tables` | `final/tab_two_breaks.csv`, `final/two_break_critical_values.csv`; `tables.json` key `two_breaks` |
| Diebold-Mariano tests of the tuned bandit against the other models | `tab:dm` | `dm` (`final_tests.py`, `inference.py`, `dm_test.py`, `dm_pooled.py`), then `tables` | `final/inference_dm_pooled38.csv`; `tables.json` key `dm` |
| Other models tested | `tab:variants` | `final_runs`, then `tables` | `tables.json` key `variants` |
| Fee drag, R^2 and directional accuracy over the full period | `tab:core_results_all` | `final_runs`, then `tables` | `tables.json` key `fee_r2` |
| One-at-a-time neighbours of the tuned bandit | `tab:tuning_neighbours` | `grids`, `selection`, then `tables` | `grids/grid_final_with_mae.csv`; `tables.json` key `neighbours` |
| The tuned bandit on the sibling categories | `tab:siblings` | `siblings`, then `tables` | `tables.json` key `siblings_bandit` |
| Models run on the sibling categories | `tab:siblings_models` | `siblings`, then `tables` | `final_siblings/period_tables.csv`; `tables.json` key `siblings_models` |
| Diebold-Mariano tests against the fixed windows at monthly rebalancing | `tab:dm_variants` | `dm`, then `tables` | `tables.json` key `dm_variants` |
| Diebold-Mariano tests of the static ridges against equal weight | `tab:dm_h1` | `dm`, then `tables` | `final/dm_h1.csv`, `final/inference_dm_pooled38.csv`; `tables.json` key `dm_h1` |
| Confidence intervals for tracking error and fee drag | `tab:ci` | `dm`, then `tables` | `final/bootstrap_ci.csv`; `tables.json` key `ci` |
| The first version of the bandit and the effect of each change | `tab:original` | `grids`, `selection`, `original_spec`, then `tables` | `robustness/tab_original.csv`; `tables.json` key `original` |

The first version's two-break statistics quoted beside `tab:original` in the appendix come from stage `first_version_breaks` (`robustness/two_break_first_version.csv`, `tables.json` key `original.first_version_two_breaks`).

### Figures

All figures are written to `results/figures/`; the numbers behind several of them are in `results/figure_data/`.

| Figure (short caption) | Label | Stage and script | Output |
|---|---|---|---|
| Basic reinforcement learning framework | `fig:rlstructure` | none: a diagram adapted from a published figure, not drawn from data | |
| Total return indices of the asset classes | `fig:asset_tri` | `figures` (`fig_inputs.py`) | `Assets_house.png` |
| Total return index of the peer-category average and the number of funds | `fig:peer_tri` | `figures` (`fig_peer_average.py`) | `peer_average_house.png` |
| Weights of the return-space ridge as the penalty strength varies | `fig:lambdapaths` | `fig_regpath` (`fig_regularisation_path.py`) | `lambdapaths.png` |
| Rolling-window estimation | `fig:rolling_window` | none: a diagram drawn in LaTeX | |
| Peer-category average against the trailing top-ten average | `fig:benchmark_select` | `figures` (`fig_inputs.py`) | `benchmark_select_house.png` |
| Peer-category average against random portfolios | `fig:peer_vs_random` | `figures` (`random_portfolios.py`, `fig_peer_vs_random.py`) | `peer_av_vs_rand_house.png` |
| Total return indices of the headline models | `fig:tri_overlay` | `figures` (`fig_headline_models.py`) | `tri_overlay_house.png` |
| Rolling 12-month quartiles against random portfolios | `fig:quartile_random` | `figures` (`random_portfolios.py`, `fig_quartiles.py`) | `quartile_vs_random_house.png` |
| Effective weights of the monthly headline models | `fig:weights_static` | `figures` (`fig_weights.py`) | `weights1.png` |
| Effective weights of the quarterly headline models | `fig:weights_dyn` | `figures` (`fig_weights.py`) | `weights2.png` |
| Trading costs per calendar year | `fig:transaction_costs` | `figures` (`fig_transaction_costs.py`) | `transaction_costs_house.png` |
| Monthly return fit of the tuned bandit | `fig:monthly_r2` | `figures` (`fig_inputs.py`) | `monthly_r2_house.png` |
| Fee drag and tracking error against transaction costs | `fig:fee_sensitivity` | `figures` (`fig_fee_sensitivity.py`) | `fee_sensitivity_house.png` |
| Tracking error and turnover against the turnover penalty | `fig:delta_sensitivity` | `fig_delta` (`fig_delta.py`) | `delta_headline_house.png` |
| Rolling 1-year tracking error of the headline models | `fig:rolling_te` | `figures` (`fig_headline_models.py`) | `rolling_te_overlay_house.png` |
| Gregory-Hansen regime-shift fit of the tuned bandit | `fig:gh_fit` | `gh_plot` (`gregory_hansen.py plot`) | `gh_regime_shift_house.png` |
| Rolling 1-year half-life of the spread | `fig:rolling_halflife` | `figures` (`fig_rolling_halflife.py`) | `rolling_halflife_house.png` |
| Window choices of the tuned bandit | `fig:tuned_windows` | `figures` (`fig_tuned_windows.py`) | `tuned_windows.png` |

### Numbers quoted in the text

Most numbers quoted only in the text come from the outputs above. The rest are written by two small stages:

- `prose_numbers` (`prose_numbers.py`, output `final/prose_numbers.csv`): end values of the total return indices, how often the tuned bandit's rolling tracking error is below the other models', the weights and equity and offshore shares of the headline models, trading costs at 10 basis points, the fit of 21-day returns, the autocorrelation of the daily change in the spread, and the bandit's window choices;
- `longonly` (`longonly_check.py`, outputs `robustness/longonly_hindsight.csv` and `robustness/longonly_rolling.csv`): the hindsight check of the long-only constraint in Appendix A.

A few numbers in the text are worked out by hand from these outputs and have no script of their own, for example the annual returns of the peer average and the top-ten average and the sibling categories' fee drag before and after February 2020.

### Run names

The saved runs (and the keys of `tables.json`) are named `<model>__<category>__<settings>__tc<cost>`.

| Part | Meaning |
|---|---|
| `ma`, `ma_le`, `ma_me` | the high-, low- and medium-equity categories |
| `w126`, `w252`, `w378`, `w504`, `w756` | estimation window of 0.5, 1, 1.5, 2 and 3 years (trading days) |
| `reb5`, `reb21`, `reb63` | rebalanced weekly, monthly and quarterly (no `reb` part: monthly) |
| `d1`, `d0p5`, ... | turnover penalty delta of 1, 0.5, ... (`p` is the decimal point) |
| `lag5`, `lag21` | reporting lag of 5 or 21 trading days |
| `tc0`, `tc5`, `tc10` | transaction costs of 0, 5 and 10 basis points one way |

| Model part | Thesis label |
|---|---|
| `equal_weight` | equal weight |
| `passive_reg28`, `passive_60_40` | the Regulation 28 and 60/40 allocations |
| `ridge_returns` | return-space ridge |
| `ridge_no_trend`, `ridge_trend` | log-TRI ridges without and with trend |
| `ridge_trend_to` | turnover-penalised ridge |
| `window_fixedbest` | fixed window |
| `rl_tuned` | tuned bandit (`rl_tuned__ma__prior_reb21` is the bandit with the earlier settings, monthly) |
| `rl_tuned_ctxfree` | context-free bandit |
| `window_random` | random window (one run per seed) |
| `window_oracle`, `window_oracle_anchor` | look-ahead oracles (tracking error, and drift) |
| `rl_bandit_to`, `rl_bandit` | the bandits with a tracking reward, with and without the turnover penalty |

## Folder layout

```
run_all.py              the entry point: every stage in order
requirements.txt        pinned direct dependencies (Python 3.12.8)
constraints.txt         pinned versions of the packages they pull in
data/
  README.md             layout of data/data.xlsx (the data themselves are not included)
src/
  paths.py              folder layout used by every script
  run_models.py         data loader, the baseline models and the walk-forward runner
  metrics.py            simulation of weights, tracking error, turnover, fee drag, unit-root tests
  rl_window_selection.py, window_baselines.py, bandit_tuning.py, run_tuning.py,
  select_tuning.py      the bandit, the window rules, the first version's tuning grid and its rule
  final_bandit.py       the final specification (Equation 3.4) and its two tuning grids
  training_mae.py       training MAE and the selection rule
  final_runs.py         every high-equity model under the final specification
  sibling_runs.py       the low- and medium-equity categories
  period_tables.py      period metrics of saved runs
  final_tests.py        Diebold-Mariano tests, bootstrap intervals, feature zeroing, break tests
  inference.py, dm_test.py, dm_pooled.py, breakpoint_coint.py,
  two_break_unit_root.py, gregory_hansen.py   the statistical tests
  original_spec.py      the first version of the bandit and tab:original
  longonly_check.py     the hindsight check of the long-only constraint
  prose_numbers.py      numbers quoted in the text that are in no table
  robustness_grid.py    the original grid through the copied loops (optional) and grid summaries
  ridge_selection.py    ridge-strength diagnostic of the turnover-penalised ridge
  random_portfolios.py  the random Regulation 28 portfolios
  thesis_tables.py      collects every numeric table into tables.json
  thesis_figures.py     draws every figure (one fig_*.py script per figure or group)
results/                everything the code writes (git-ignored)
  cache/  grids/  final/  final_siblings/  robustness/  figure_data/  figures/  logs/
```

## Licence

The code is released under the MIT licence (see `LICENSE`). The licence does not cover the data, which remain the property of their providers.
