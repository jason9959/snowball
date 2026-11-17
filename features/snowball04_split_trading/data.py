"""Symbol discovery and adjusted daily-close loading."""

from __future__ import annotations

from datetime import date, timedelta
import re

import pandas as pd
import requests


def search_symbols(query: str, country: str, limit: int = 12) -> dict[str, str]:
    text = query.strip()
    if len(text) < 1:
        return {}
    options: dict[str, str] = {}
    if country == "미국" and re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,14}", text):
        direct = text.upper()
        options[direct] = f"{direct} · 티커 직접 사용"
    elif country == "한국" and re.fullmatch(r"\d{6}", text):
        options[f"{text}.KS"] = f"{text}.KS · 코스피 코드 직접 사용"
        options[f"{text}.KQ"] = f"{text}.KQ · 코스닥 코드 직접 사용"
    try:
        response = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": text, "quotesCount": 30, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=12,
        )
        response.raise_for_status()
        quotes = response.json().get("quotes", [])
    except (requests.RequestException, ValueError):
        return options
    for quote in quotes:
        symbol = str(quote.get("symbol", "")).strip()
        exchange = str(quote.get("exchange", "")).upper()
        quote_type = str(quote.get("quoteType", "")).upper()
        if not symbol or quote_type not in {"EQUITY", "ETF", "MUTUALFUND"}:
            continue
        is_korea = symbol.endswith((".KS", ".KQ")) or exchange in {"KSC", "KOE"}
        if (country == "한국") != is_korea:
            continue
        name = str(quote.get("longname") or quote.get("shortname") or symbol)
        options[symbol] = f"{symbol} · {name}"
        if len(options) >= limit:
            break
    return options


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

