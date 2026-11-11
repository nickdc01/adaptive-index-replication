# Input data (not included)

The data behind the thesis are licensed and are not part of this repository:

- the daily returns of the South African multi-asset peer-category averages come from Profile Data (Pty) Ltd, used with permission;
- the asset-class total return indices come from Bloomberg.

To run the code, put your own copy in this folder as `data/data.xlsx`. The folder is git-ignored, so the file cannot be committed by accident.

## Layout of `data.xlsx`

Every sheet has a `datestamp` column of daily dates.

| Sheet | Columns | Contents | Used by |
|---|---|---|---|
| `X` | `datestamp` and 11 asset columns, named exactly: `SA Bonds`, `SA Property`, `SA Cash & Income`, `SA Equities`, `Unhedged Bonds`, `Unhedged Equities`, `Unhedged Cash`, `Unhedged EM Equities`, `Hedged Bonds`, `Hedged Equities`, `Hedged EM Equities` | Daily total return index levels in rand, in this column order. Positive, no blanks, sorted unique dates. Its dates set the sample (1 December 2005 to 15 May 2024 in the thesis). | every model, `fig_inputs.py` |
| `Y` | `datestamp`, `fund_name`, `value` | Daily returns (decimals) of the peer-category average that is replicated, the high-equity multi-asset category. `fund_name` is a constant label and is ignored. | every model, `fig_peer_average.py`, `fig_inputs.py` |
| `Y_le`, `Y_me` | `datestamp`, `value` | Daily returns (decimals) of the low- and medium-equity category averages. | `sibling_runs.py` (appendix sibling tables) |
| `top10 av` | `datestamp`, `top10` | Daily returns (decimals) of the average of the trailing one-year top-ten funds in the category. Blank rows are dropped. | `fig_inputs.py` (benchmark selection figure) |
| `all` | `datestamp` and one column per peer fund | Daily fund returns, blank when a fund does not report. Only the number of reporting funds is used. | `fig_peer_average.py` (fund-count panel) |

The loader (`run_models.load_data`) compounds the benchmark returns into an index, moves it onto the dates of sheet `X`, carries it forward over missing days, drops dates with a missing asset level, and turns everything into daily returns.

The scripts assume the thesis sample: sheet `X` from 1 December 2005 to 15 May 2024, with sheet `Y` covering 1 December 2005. The period boundaries (training to 30 December 2016, test from 3 January 2017, full period from 28 January 2009) are fixed in the code, and the 11 asset columns should be in the order above, the order the thesis numbers were produced with.
