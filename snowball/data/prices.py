"""Price providers kept outside the calculation engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PriceBundle:
    adjusted_close: pd.DataFrame
    adjusted_open: pd.DataFrame
    source: str = "Yahoo Finance"

    def validate(self, symbols: tuple[str, ...]) -> None:
        if self.adjusted_close.empty or self.adjusted_open.empty:
            raise ValueError("가격 데이터가 비어 있습니다.")
        missing = [symbol for symbol in symbols if symbol not in self.adjusted_close or self.adjusted_close[symbol].dropna().empty]
        if missing:
            raise ValueError("가격 데이터가 없는 종목: " + ", ".join(missing))
        if not self.adjusted_close.index.equals(self.adjusted_open.index):
            raise ValueError("수정종가와 수정시가의 날짜가 일치하지 않습니다.")


def download_yahoo_prices(symbols: tuple[str, ...], start: pd.Timestamp, end: pd.Timestamp) -> PriceBundle:
    try:
        import yfinance as yf
    except ImportError as error:
        raise RuntimeError("실시간 가격 다운로드에는 yfinance 설치가 필요합니다.") from error
    symbols = tuple(dict.fromkeys(symbols))
    raw = yf.download(
        list(symbols), start=start, end=end + pd.Timedelta(days=1), auto_adjust=False,
        progress=False, threads=False, group_by="column",
    )
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance에서 가격 데이터를 가져오지 못했습니다.")
    if isinstance(raw.columns, pd.MultiIndex):
        fields = set(raw.columns.get_level_values(0))
        if not {"Adj Close", "Close", "Open"}.issubset(fields):
            raise ValueError("Yahoo 응답에 Open, Close, Adj Close가 모두 필요합니다.")
        adjusted_close, close, open_price = raw["Adj Close"].copy(), raw["Close"].copy(), raw["Open"].copy()
    else:
        if len(symbols) != 1 or not {"Adj Close", "Close", "Open"}.issubset(raw.columns):
            raise ValueError("Yahoo 단일 종목 응답 형식이 올바르지 않습니다.")
        adjusted_close = raw[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
        open_price = raw[["Open"]].rename(columns={"Open": symbols[0]})
    for frame in (adjusted_close, close, open_price):
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    adjusted_close = adjusted_close.apply(pd.to_numeric, errors="coerce").sort_index()
    close = close.apply(pd.to_numeric, errors="coerce").sort_index()
    open_price = open_price.apply(pd.to_numeric, errors="coerce").sort_index()
    adjusted_open = open_price * adjusted_close.divide(close.where(close > 0))
    bundle = PriceBundle(adjusted_close.ffill(), adjusted_open.ffill())
    bundle.validate(symbols)
    return bundle
