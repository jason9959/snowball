"""Symbol discovery and adjusted daily-close loading."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from common.snowball.symbols import search_symbols


def load_adjusted_close(symbol: str, start: date, end: date) -> pd.Series:
    if start > end:
        raise ValueError("From은 To보다 늦을 수 없습니다.")
    try:
        import yfinance as yf
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("가격 조회에는 yfinance가 필요합니다.") from error
    frame = yf.download(
        symbol, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
        interval="1d", auto_adjust=True, progress=False, threads=False,
    )
    if frame.empty:
        raise ValueError("요청 기간에 사용할 종가가 없습니다.")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.loc[(close.index.date >= start) & (close.index.date <= end)]

