from datetime import date, timedelta
import pandas as pd
import yfinance as yf


def load_ohlc(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    ticker=ticker.strip().upper()
    if not ticker: raise ValueError('종목 티커를 입력해 주세요.')
    if start_date>=end_date: raise ValueError('종료일은 시작일보다 뒤여야 합니다.')
    raw=yf.download(ticker,start=start_date.isoformat(),end=(end_date+timedelta(days=1)).isoformat(),auto_adjust=True,progress=False,threads=False)
    if raw is None or raw.empty: raise ValueError('가격 데이터를 가져오지 못했습니다.')
    if isinstance(raw.columns,pd.MultiIndex):
        if ticker in raw.columns.get_level_values(-1): raw=raw.xs(ticker,axis=1,level=-1)
        else: raw.columns=raw.columns.get_level_values(0)
    need=['High','Low','Close']
    if any(c not in raw.columns for c in need): raise ValueError('OHLC 데이터가 충분하지 않습니다.')
    out=raw[need].dropna().copy(); out.index=pd.to_datetime(out.index).tz_localize(None).normalize(); return out
