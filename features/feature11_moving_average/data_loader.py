from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf


TRADING_DAYS_PER_MONTH = 21


def _normalize_download_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance output to a simple DatetimeIndex + Close frame."""
    if raw.empty:
        raise ValueError("가격 데이터를 찾을 수 없습니다. 티커와 기간을 확인해 주세요.")

    frame = raw.copy()

    # yfinance may return MultiIndex columns even for one ticker.
    if isinstance(frame.columns, pd.MultiIndex):
        if ticker in frame.columns.get_level_values(-1):
            frame = frame.xs(ticker, axis=1, level=-1)
        else:
            frame.columns = frame.columns.get_level_values(0)

    if "Close" not in frame.columns:
        raise ValueError("종가(Close) 데이터를 찾을 수 없습니다.")

    frame = frame[["Close"]].dropna().copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def load_price_data(
    ticker: str,
    start_date: date,
    end_date: date,
    ma_months: int,
) -> pd.DataFrame:
    """
    Download adjusted daily close prices and calculate an N-month SMA.

    N months is approximated as N * 21 trading days, as agreed for v1.
    The download begins earlier than the requested start date so the moving
    average is already available at the beginning of the test whenever the
    ticker has sufficient history.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("종목 티커를 입력해 주세요.")
    if start_date >= end_date:
        raise ValueError("종료일은 시작일보다 뒤여야 합니다.")
    if ma_months < 1:
        raise ValueError("이동평균 개월 수는 1 이상이어야 합니다.")

    window = ma_months * TRADING_DAYS_PER_MONTH

    # Use a generous calendar buffer because weekends/holidays reduce trading days.
    buffer_days = max(int(window * 2.2), 120)
    download_start = start_date - timedelta(days=buffer_days)
    # yfinance's end date is exclusive.
    download_end = end_date + timedelta(days=1)

    raw = yf.download(
        ticker,
        start=download_start.isoformat(),
        end=download_end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    frame = _normalize_download_frame(raw, ticker)
    frame["SMA"] = frame["Close"].rolling(window=window, min_periods=window).mean()

    test = frame.loc[
        (frame.index >= pd.Timestamp(start_date))
        & (frame.index <= pd.Timestamp(end_date))
    ].copy()

    if test.empty:
        raise ValueError("선택한 기간에 거래 데이터가 없습니다.")

    if test["SMA"].isna().any():
        first_valid = frame["SMA"].first_valid_index()
        if first_valid is None or first_valid > test.index[0]:
            raise ValueError(
                "선택한 시작일에 이동평균을 계산할 충분한 과거 데이터가 없습니다. "
                "시작일을 늦추거나 이동평균 개월 수를 줄여 주세요."
            )

    test = test.dropna(subset=["SMA"])
    if test.empty:
        raise ValueError("이동평균을 계산할 수 있는 데이터가 없습니다.")

    return test
