"""
Peer-category average and the size of the fund panel (thesis Figure fig:peer_tri,
peer_average_house.png): the peer-category average total return index (top) and
the monthly number of funds with a return (bottom), 1 December 2005 to 15 May 2024.

Count used in the figure: funds with at least one daily return in the month.
Sheet "all" holds only the funds still in the category when the data were extracted,
so counts in earlier years may understate the number of funds in the average.

Run:     python src/fig_peer_average.py
Input:   data/data.xlsx
           sheet "Y"   : daily peer-category average return (decimal), the benchmark
           sheet "all" : daily returns of the funds in the category (only the count is used)
Outputs: <results>/figures/peer_average_house.png
         <results>/figure_data/peer_fund_counts.csv   monthly fund counts, full panel history
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402

OUT = paths.FIGURES / "peer_average_house.png"
COUNTS_CSV = paths.FIGURE_DATA / "peer_fund_counts.csv"
START, END = pd.Timestamp("2005-12-01"), pd.Timestamp("2024-05-15")
BLUE = "#1f4e79"   # the peer-average colour used across the thesis figures

STYLE = {
    "figure.figsize": (10, 5),
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def peer_tri() -> pd.Series:
    """Compound sheet Y on its own calendar (as run_models.load_data does), rebase to 1 on START."""
    y = pd.read_excel(paths.DATA, sheet_name="Y")
    y["datestamp"] = pd.to_datetime(y["datestamp"])
    r = y.set_index("datestamp")["value"].sort_index()
    tri = (1.0 + r).cumprod()
    tri = tri / tri.loc[START]
    return tri.loc[START:END]


def fund_counts() -> pd.DataFrame:
    a = pd.read_excel(paths.DATA, sheet_name="all")
    a["datestamp"] = pd.to_datetime(a["datestamp"])
    a = a.set_index("datestamp").sort_index()
    has = a.notna()
    first = a.apply(lambda s: s.first_valid_index())
    last = a.apply(lambda s: s.last_valid_index())

    month = a.index.to_period("M")
    with_return = has.groupby(month).any().sum(axis=1)
    median_daily = has.sum(axis=1).groupby(month).median()
    last_date = a.index.to_series().groupby(month).max()          # last panel date in each month
    alive_end = pd.Series({p: int(((first <= d) & (last >= d)).sum()) for p, d in last_date.items()})
    entries = first.dt.to_period("M").value_counts().reindex(with_return.index, fill_value=0)
    exits = last.dt.to_period("M").value_counts().reindex(with_return.index, fill_value=0)
    exits.iloc[-1] = 0   # the panel's final month is the data cut-off, not closures

    out = pd.DataFrame({
        "month_end": last_date.dt.date,
        "funds_with_return_in_month": with_return,
        "funds_alive_at_month_end": alive_end,
        "median_funds_reporting_per_day": median_daily,
        "first_returns_in_month": entries,
        "last_returns_in_month": exits,
    })
    out.index = out.index.astype(str)
    out.index.name = "month"
    out["in_thesis_window"] = [(START.to_period("M") <= pd.Period(m) <= END.to_period("M")) for m in out.index]
    return out


def main():
    tri = peer_tri()
    counts = fund_counts()
    paths.ensure(paths.FIGURE_DATA)
    counts.to_csv(COUNTS_CSV)

    c = counts[counts["in_thesis_window"]]
    # Each month's count holds from the first day of the month (step "post"); repeat the
    # last value at END so the final month is drawn.
    c_dates = pd.PeriodIndex(c.index, freq="M").to_timestamp().append(pd.DatetimeIndex([END]))
    c = pd.concat([c, c.iloc[[-1]]])

    with plt.style.context(STYLE, after_reset=True):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1], "hspace": 0.25})
        ax1.plot(tri.index, tri.values, lw=1.2, color=BLUE)
        ax1.set_title("Peer Category Average: Total Return Index")
        ax1.set_ylabel("Total Return Index (rebased to 1.0)")

        ax2.step(c_dates, c["funds_with_return_in_month"], where="post", lw=1.2, color=BLUE)
        ax2.fill_between(c_dates, c["funds_with_return_in_month"], step="post", color=BLUE, alpha=0.12, lw=0)
        ax2.set_title("Number of Funds with a Return in the Month")
        ax2.set_ylabel("Funds")
        ax2.set_ylim(0, None)
        ax2.set_xlim(START, END)
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        paths.ensure(OUT.parent)
        fig.savefig(OUT)
        plt.close(fig)
    print(f"funds with a return: {c.index[0]} {c['funds_with_return_in_month'].iloc[0]}, "
          f"{c.index[-1]} {c['funds_with_return_in_month'].iloc[-1]}")
    print(f"wrote {OUT} and {COUNTS_CSV}")


if __name__ == "__main__":
    main()
