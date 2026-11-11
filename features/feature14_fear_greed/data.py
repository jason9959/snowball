from __future__ import annotations

import datetime as dt

import pandas as pd
import requests
import streamlit as st
import yfinance as yf


INDICATORS = {
    "fear_and_greed_historical": "종합 공포·탐욕 지수",
    "market_momentum_sp500": "시장 모멘텀 (S&P 500)",
    "market_momentum_sp125": "시장 모멘텀 (125일)",
    "stock_price_strength": "주가 강도",
    "stock_price_breadth": "시장 breadth",
    "put_call_options": "풋·콜 옵션 비율",
    "market_volatility_vix": "VIX 변동성",
    "junk_bond_demand": "정크본드 수요",
    "safe_haven_demand": "안전자산 수요",
}

CNN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://edition.cnn.com/markets/fear-and-greed",
    "Origin": "https://edition.cnn.com",
}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_fear_greed(start_iso: str, end_iso: str) -> pd.DataFrame:
    """CNN 페이지가 사용하는 역사 데이터 엔드포인트를 읽는다.

    CNN은 공식 공개 API를 제공하지 않으므로 응답이 바뀔 수 있다. 날짜 경로를
    사용하면 기본 엔드포인트의 봇 차단을 줄이고 요청 한 번으로 각 지표의
    역사 자료를 받을 수 있다.
    """
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_iso}"
    response = requests.get(url, headers=CNN_HEADERS, timeout=25)
    response.raise_for_status()
    payload = response.json()
    rows: dict[str, pd.Series] = {}
    for key in INDICATORS:
        raw = payload.get(key)
        if key == "fear_and_greed_historical":
            points = raw.get("data", []) if isinstance(raw, dict) else (raw or [])
        else:
            points = raw if isinstance(raw, list) else (raw.get("data", []) if isinstance(raw, dict) else [])
        values = []
        for point in points:
            if not isinstance(point, dict) or point.get("x") is None or point.get("y") is None:
                continue
            values.append((pd.to_datetime(point["x"], unit="ms").normalize(), float(point["y"])))
        if values:
            rows[key] = pd.Series(dict(values), name=key).sort_index()
    if not rows:
        raise ValueError("CNN 공포·탐욕 지수의 역사 데이터를 찾지 못했습니다.")
    result = pd.concat(rows.values(), axis=1).sort_index()
    result.index = pd.to_datetime(result.index).tz_localize(None)
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    result = result.loc[(result.index >= start) & (result.index <= end)]
    if result.empty:
        raise ValueError("입력한 기간에 공포·탐욕 지수 데이터가 없습니다.")
    return result.ffill()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_current_fear_greed() -> dict:
    today = dt.date.today().isoformat()
    frame = fetch_fear_greed((dt.date.today() - dt.timedelta(days=370)).isoformat(), today)
    latest = frame.iloc[-1]
    return {"date": frame.index[-1], "values": latest.to_dict(), "history": frame}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...], start_iso: str, end_iso: str) -> pd.DataFrame:
    end = pd.Timestamp(end_iso) + pd.Timedelta(days=1)
    raw = yf.download(list(tickers), start=start_iso, end=end, auto_adjust=False, progress=False, threads=False, group_by="column")
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance에서 주가 데이터를 가져오지 못했습니다.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise ValueError("주가 응답에 수정 종가가 없습니다.")
        prices = raw["Adj Close"].copy()
    else:
        if "Adj Close" not in raw.columns:
            raise ValueError("주가 응답에 수정 종가가 없습니다.")
        prices = raw[["Adj Close"]].copy()
        prices.columns = [tickers[0]]
    prices = prices.reindex(columns=[ticker for ticker in tickers if ticker in prices.columns])
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.apply(pd.to_numeric, errors="coerce").dropna(how="all").ffill().dropna()
    if prices.empty or len(prices) < 2:
        raise ValueError("공통 주가 데이터가 충분하지 않습니다.")
    return prices

