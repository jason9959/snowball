"""Read-only account dashboard data shared by web frontends."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping

import pandas as pd

from snowball.accounting import (
    AccountPeriodMetrics,
    AccountValuation,
    account_period_metrics,
    calculate_account_valuation,
    required_price_symbols,
    transaction_frame,
)


LEDGER_DISPLAY_COLUMNS = (
    "입력날짜", "입력시각", "거래일", "유형", "시장", "종목코드", "종목명",
    "수량", "체결단가", "거래금액", "수수료·세금", "결제금액", "통화", "환율",
    "메모", "상태",
)


@dataclass(frozen=True)
class AccountDashboardData:
    first_date: date
    start_date: date
    end_date: date
    symbols: tuple[str, ...]
    valuation: AccountValuation
    performance: AccountPeriodMetrics
    cumulative_deposits: float
    cumulative_withdrawals: float
    period_deposits: float
    period_withdrawals: float
    performance_valid: bool
    currency_assets: dict[str, AccountPeriodMetrics]
    ledger: pd.DataFrame


def active_account_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    return transaction_frame(records)


def account_date_bounds(
    records: Iterable[Mapping[str, Any]],
    *,
    today: date,
) -> tuple[date, date]:
    frame = active_account_frame(records)
    if frame.empty:
        raise ValueError("정상 상태의 거래가 없습니다.")
    return frame["거래일"].min().date(), today


def build_account_dashboard(
    records: Iterable[Mapping[str, Any]],
    *,
    start_date: date,
    end_date: date,
    closes: pd.DataFrame | None = None,
    usdkrw: pd.Series | None = None,
) -> AccountDashboardData:
    frame = active_account_frame(records)
    if frame.empty:
        raise ValueError("정상 상태의 거래가 없습니다.")
    first_date = frame["거래일"].min().date()
    if start_date < first_date:
        start_date = first_date
    if start_date > end_date:
        raise ValueError("조회 시작일은 종료일보다 앞서야 합니다.")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    frame_to_end = frame[frame["거래일"] <= end_ts].copy()
    if frame_to_end.empty:
        raise ValueError("선택한 종료일까지의 거래가 없습니다.")

    valuation = calculate_account_valuation(
        frame_to_end,
        pd.DataFrame() if closes is None else closes,
        pd.Series(dtype=float) if usdkrw is None else usdkrw,
        end_date=end_ts,
    )
    performance = account_period_metrics(valuation.history, start_ts, end_ts)
    currency_assets = {
        currency: account_period_metrics(history, start_ts, end_ts)
        for currency, history in valuation.currency_assets.items()
        if not history.loc[(history.index >= start_ts) & (history.index <= end_ts), [
            "매입원가", "시장평가액",
        ]].fillna(0.0).eq(0.0).all().all()
    }
    period_frame = frame_to_end[
        (frame_to_end["거래일"] >= start_ts) & (frame_to_end["거래일"] <= end_ts)
    ]

    ledger = pd.DataFrame(records)
    if not ledger.empty:
        ledger["_거래일"] = pd.to_datetime(ledger["거래일"], errors="coerce")
        input_date = (
            ledger["입력날짜"] if "입력날짜" in ledger
            else pd.Series("", index=ledger.index)
        ).fillna("").astype(str).str.strip()
        input_time = (
            ledger["입력시각"] if "입력시각" in ledger
            else pd.Series("", index=ledger.index)
        ).fillna("").astype(str).str.strip()
        ledger["_입력시각"] = pd.to_datetime(
            (input_date + " " + input_time).str.strip(), format="mixed", errors="coerce",
        ).fillna(pd.to_datetime(input_time, format="mixed", errors="coerce"))
        ledger = ledger[
            (ledger["_거래일"] >= start_ts) & (ledger["_거래일"] <= end_ts)
        ].sort_values(["_거래일", "_입력시각"], ascending=False, kind="stable")
        display_columns = [column for column in LEDGER_DISPLAY_COLUMNS if column in ledger]
        ledger = ledger[display_columns].reset_index(drop=True)

    def amount(kind: str, source: pd.DataFrame) -> float:
        return float(source.loc[source["유형"] == kind, "원화금액"].sum())

    return AccountDashboardData(
        first_date=first_date,
        start_date=start_date,
        end_date=end_date,
        symbols=required_price_symbols(frame),
        valuation=valuation,
        performance=performance,
        cumulative_deposits=amount("입금", frame_to_end),
        cumulative_withdrawals=amount("출금", frame_to_end),
        period_deposits=amount("입금", period_frame),
        period_withdrawals=amount("출금", period_frame),
        performance_valid=valuation.cash >= -1e-6,
        currency_assets=currency_assets,
        ledger=ledger,
    )
