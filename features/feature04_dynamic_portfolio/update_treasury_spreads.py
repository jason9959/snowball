"""Manually refresh the bundled monthly Treasury spread CSV.

This is intentionally not scheduled.  Run it only when the repository's
baseline CSV should be brought up to date; the app itself refreshes missing
months into its runtime cache on demand.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from features.feature04_dynamic_portfolio.market_data import (  # type: ignore
        PACKAGED_SPREAD_CSV,
        SPREAD_COLUMNS,
        _download_series,
    )
else:
    from .market_data import PACKAGED_SPREAD_CSV, SPREAD_COLUMNS, _download_series


def main() -> None:
    start = pd.Timestamp("1982-01-01")
    end = pd.Timestamp(dt.date.today())
    columns = {
        series_id: _download_series(series_id, start, end)
        for series_id in SPREAD_COLUMNS
    }
    frame = pd.concat(columns, axis=1).sort_index()
    PACKAGED_SPREAD_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.rename_axis("Date").reset_index().to_csv(
        PACKAGED_SPREAD_CSV,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.4f",
    )
    print(f"updated {PACKAGED_SPREAD_CSV} ({len(frame)} months)")


if __name__ == "__main__":
    main()
