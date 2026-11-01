"""Actual-account holdings, cash and valuation calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AccountValuation:
    history: pd.DataFrame
    holdings: pd.DataFrame
    net_deposits: float
    cash: float
    holdings_value: float
    total_value: float
    profit: float
    return_rate: float | None


def yahoo_symbol(code: Any, market: Any) -> str:
    ticker = str(code or "").strip().upper()
    if str(market) == "한국":
        ticker = ticker.removesuffix(".KS")
        if ticker.isdigit():
            ticker = ticker.zfill(6)
        return f"{ticker}.KS"
    return ticker


def transaction_frame(
    records: Iterable[Mapping[str, Any]],
    account_id: str | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    active = frame["상태"].astype(str) == "정상"
    if account_id is not None:
        if "계좌ID" not in frame:
            frame["계좌ID"] = "default"
        active &= frame["계좌ID"].fillna("default").astype(str) == account_id
    frame = frame[active].copy()
    if frame.empty:
        return frame
    frame["거래일"] = pd.to_datetime(frame["거래일"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["거래일"])
    for column in ("수량", "체결단가", "수수료·세금", "금액", "환율"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["원화금액"] = frame["금액"] * frame["환율"].replace(0, np.nan).fillna(1.0)
    frame["원화비용"] = frame["수수료·세금"] * frame["환율"].replace(0, np.nan).fillna(1.0)
    frame["Yahoo종목"] = [
        yahoo_symbol(code, market) for code, market in zip(frame["종목코드"], frame["시장"])
    ]
    return frame.sort_values(["거래일", "입력시각"], kind="stable")


def required_price_symbols(frame: pd.DataFrame) -> tuple[str, ...]:
    if frame.empty:
        return ()
    trades = frame[frame["유형"].isin(("매수", "매도"))]
    return tuple(dict.fromkeys(ticker for ticker in trades["Yahoo종목"] if ticker))


def download_unadjusted_closes(
    symbols: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download ordinary closes; dividends are recorded separately in the ledger."""
    if not symbols:
        return pd.DataFrame(), pd.Series(dtype=float, name="USDKRW")
    try:
        import yfinance as yf
    except ImportError as error:
        raise RuntimeError("계좌 평가에는 yfinance 설치가 필요합니다.") from error
    needs_fx = any(not symbol.endswith(".KS") for symbol in symbols)
    requested = tuple(dict.fromkeys(symbols + (("KRW=X",) if needs_fx else ())))
    raw = yf.download(
        list(requested),
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise ValueError("현재가 데이터를 가져오지 못했습니다.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in set(raw.columns.get_level_values(0)):
            raise ValueError("현재가 응답에 종가가 없습니다.")
        close = raw["Close"].copy()
    elif len(requested) == 1 and "Close" in raw:
        close = raw[["Close"]].rename(columns={"Close": requested[0]})
    else:
        raise ValueError("현재가 응답 형식이 올바르지 않습니다.")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close.apply(pd.to_numeric, errors="coerce").sort_index()
    fx = close.pop("KRW=X") if "KRW=X" in close else pd.Series(1.0, index=close.index, name="USDKRW")
    missing = [symbol for symbol in symbols if symbol not in close or close[symbol].dropna().empty]
    if missing:
        raise ValueError("현재가가 없는 종목: " + ", ".join(missing))
    return close.loc[:, list(symbols)].ffill(), fx.ffill().rename("USDKRW")


def calculate_account_valuation(
    frame: pd.DataFrame,
    closes: pd.DataFrame | None = None,
    usdkrw: pd.Series | None = None,
    end_date: pd.Timestamp | None = None,
) -> AccountValuation:
    if frame.empty:
        return AccountValuation(
            history=pd.DataFrame(), holdings=pd.DataFrame(), net_deposits=0.0,
            cash=0.0, holdings_value=0.0, total_value=0.0, profit=0.0,
            return_rate=None,
        )
    closes = pd.DataFrame() if closes is None else closes.copy()
    final_date = pd.Timestamp(end_date or pd.Timestamp.now()).normalize()
    first_date = frame["거래일"].min()
    dates = pd.date_range(first_date, final_date, freq="D")

    cash_change = pd.Series(0.0, index=frame.index)
    cash_change.loc[frame["유형"] == "입금"] = frame.loc[frame["유형"] == "입금", "원화금액"]
    cash_change.loc[frame["유형"] == "출금"] = -frame.loc[frame["유형"] == "출금", "원화금액"]
    cash_change.loc[frame["유형"] == "배당"] = frame.loc[frame["유형"] == "배당", "원화금액"]
    cash_change.loc[frame["유형"] == "잔액조정"] = frame.loc[frame["유형"] == "잔액조정", "원화금액"]
    cash_change.loc[frame["유형"] == "매수"] = -(
        frame.loc[frame["유형"] == "매수", "원화금액"]
        + frame.loc[frame["유형"] == "매수", "원화비용"]
    )
    cash_change.loc[frame["유형"] == "매도"] = (
        frame.loc[frame["유형"] == "매도", "원화금액"]
        - frame.loc[frame["유형"] == "매도", "원화비용"]
    )
    daily_cash_flow = cash_change.groupby(frame["거래일"]).sum().reindex(dates, fill_value=0.0)
    cash_balance = daily_cash_flow.cumsum()

    external = pd.Series(0.0, index=frame.index)
    external.loc[frame["유형"] == "입금"] = frame.loc[frame["유형"] == "입금", "원화금액"]
    external.loc[frame["유형"] == "출금"] = -frame.loc[frame["유형"] == "출금", "원화금액"]
    daily_external = external.groupby(frame["거래일"]).sum().reindex(dates, fill_value=0.0)

    trades = frame[frame["유형"].isin(("매수", "매도")) & frame["Yahoo종목"].ne("")].copy()
    holding_values = pd.Series(0.0, index=dates)
    holding_rows: list[dict[str, Any]] = []
    if not trades.empty:
        trades["증감수량"] = np.where(trades["유형"] == "매수", trades["수량"], -trades["수량"])
        quantity_changes = trades.pivot_table(
            index="거래일", columns="Yahoo종목", values="증감수량", aggfunc="sum", fill_value=0.0,
        )
        quantities = quantity_changes.reindex(dates, fill_value=0.0).cumsum()
        price_daily = closes.reindex(dates).ffill() if not closes.empty else pd.DataFrame(index=dates)
        fx_daily = (
            pd.Series(1.0, index=dates)
            if usdkrw is None or usdkrw.empty
            else usdkrw.reindex(dates).ffill().bfill()
        )
        latest_by_symbol = trades.drop_duplicates("Yahoo종목", keep="last").set_index("Yahoo종목")
        for symbol in quantities.columns:
            if symbol not in price_daily or price_daily[symbol].dropna().empty:
                continue
            multiplier = 1.0 if symbol.endswith(".KS") else fx_daily
            position_value = quantities[symbol] * price_daily[symbol] * multiplier
            holding_values = holding_values.add(position_value.fillna(0.0), fill_value=0.0)
            quantity = float(quantities[symbol].iloc[-1])
            if abs(quantity) <= 1e-10:
                continue
            source = latest_by_symbol.loc[symbol]
            latest_price = float(price_daily[symbol].dropna().iloc[-1])
            latest_fx = 1.0 if symbol.endswith(".KS") else float(fx_daily.iloc[-1])
            holding_rows.append({
                "종목코드": str(source["종목코드"]),
                "종목명": str(source["종목명"]),
                "수량": quantity,
                "현재가": latest_price,
                "통화": "KRW" if symbol.endswith(".KS") else "USD",
                "환율": latest_fx,
                "평가금액": quantity * latest_price * latest_fx,
            })

    history = pd.DataFrame({
        "총평가액": cash_balance + holding_values,
        "보유자산": holding_values,
        "현금": cash_balance,
        "입출금": daily_external,
    }, index=dates)
    total_value = float(history["총평가액"].iloc[-1])
    net_deposits = float(daily_external.sum())
    profit = total_value - net_deposits
    return_rate = profit / net_deposits if net_deposits > 0 else None
    holdings = pd.DataFrame(holding_rows)
    if not holdings.empty:
        holdings = holdings.sort_values("평가금액", ascending=False).reset_index(drop=True)
    return AccountValuation(
        history=history,
        holdings=holdings,
        net_deposits=net_deposits,
        cash=float(cash_balance.iloc[-1]),
        holdings_value=float(holding_values.iloc[-1]),
        total_value=total_value,
        profit=profit,
        return_rate=return_rate,
    )
