from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
from dateutil.relativedelta import relativedelta


FREQUENCY_MONTHS = {
    "매월": 1,
    "매분기": 3,
    "매반기": 6,
    "매년": 12,
}


def _next_scheduled_date(anchor: date, frequency: str, occurrence: int) -> date:
    if frequency == "매주":
        return anchor + relativedelta(weeks=occurrence)
    if frequency in FREQUENCY_MONTHS:
        months = FREQUENCY_MONTHS[frequency] * occurrence
        # Always calculate from the original anchor so a shifted execution date
        # never changes the next scheduled date.
        return anchor + relativedelta(months=months)
    raise ValueError(f"지원하지 않는 적립 주기입니다: {frequency}")


def build_contribution_schedule(
    anchor_date: date,
    end_date: date,
    frequency: str,
    trading_dates: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """
    Create scheduled contribution dates and map each one to the first trading
    date on or after the scheduled date.

    The first contribution occurs one full selected period after anchor_date.
    If the scheduled date is not a trading day, execution moves forward to the
    nearest later trading day, while future schedule dates remain anchored to
    the original start date.
    """
    index = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).tz_localize(None).normalize()
    index = index.sort_values().unique()

    rows: list[dict] = []
    occurrence = 1

    while True:
        scheduled = _next_scheduled_date(anchor_date, frequency, occurrence)
        if scheduled > end_date:
            break

        scheduled_ts = pd.Timestamp(scheduled)
        pos = index.searchsorted(scheduled_ts, side="left")
        actual = None
        if pos < len(index):
            candidate = pd.Timestamp(index[pos])
            if candidate.date() <= end_date:
                actual = candidate

        if actual is not None:
            rows.append(
                {
                    "scheduled_date": scheduled_ts,
                    "actual_date": actual,
                }
            )

        occurrence += 1

    return pd.DataFrame(rows, columns=["scheduled_date", "actual_date"])
