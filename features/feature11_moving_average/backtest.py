from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .schedule import build_contribution_schedule
from common.metrics import sharpe_ratio, sortino_ratio, calmar_ratio, cashflow_xirr


@dataclass
class BacktestOutput:
    daily: pd.DataFrame
    events: pd.DataFrame
    schedule: pd.DataFrame
    metrics: dict


def _build_contribution_map(schedule: pd.DataFrame, amount: float) -> dict[pd.Timestamp, float]:
    if schedule.empty:
        return {}
    counts = schedule.groupby("actual_date").size()
    return {pd.Timestamp(day): float(count * amount) for day, count in counts.items()}


def _max_drawdown(values: pd.Series) -> float:
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def run_backtest(
    price_data: pd.DataFrame,
    requested_start_date: date,
    requested_end_date: date,
    initial_amount: float,
    use_contribution: bool,
    contribution_amount: float,
    contribution_frequency: str,
    confirmation_count: int = 1,
    signal_limit_days: int = 5,
) -> BacktestOutput:
    """Run the moving-average strategy.

    Signal rules
    ------------
    * Close > SMA: BUY
    * Close < SMA: SELL
    * Close == SMA: NEUTRAL

    Position-change confirmation
    ----------------------------
    While CASH, BUY signals are accumulated. While STOCK, SELL signals are
    accumulated. A position change happens immediately when ``confirmation_count``
    target signals have been observed.

    After every target signal, the strategy waits up to ``signal_limit_days``
    *trading days* for the next target signal. Another target signal increments
    the counter and restarts that wait. A NEUTRAL day keeps the pending counter
    alive. If the wait expires before the next target signal arrives, the
    position change is forced at that day's close using the confirmations already
    accumulated. If the opposite signal appears before expiry, all pending
    confirmations are cleared.

    A confirmation_count of 1 reproduces the original immediate-transition
    behavior, so the limit has no practical effect in that case.
    """
    if initial_amount <= 0:
        raise ValueError("초기 투자금은 0보다 커야 합니다.")
    if use_contribution and contribution_amount <= 0:
        raise ValueError("적립 금액은 0보다 커야 합니다.")
    if int(confirmation_count) < 1:
        raise ValueError("포지션 변경 확인 횟수는 1 이상이어야 합니다.")
    if int(signal_limit_days) < 1:
        raise ValueError("시그널 대기 LIMIT은 1거래일 이상이어야 합니다.")

    confirmation_count = int(confirmation_count)
    signal_limit_days = int(signal_limit_days)

    data = price_data.copy().sort_index()
    if data.empty:
        raise ValueError("백테스트할 가격 데이터가 없습니다.")

    schedule = pd.DataFrame(columns=["scheduled_date", "actual_date"])
    if use_contribution:
        schedule = build_contribution_schedule(
            anchor_date=requested_start_date,
            end_date=requested_end_date,
            frequency=contribution_frequency,
            trading_dates=data.index,
        )

    contribution_map = _build_contribution_map(
        schedule,
        contribution_amount if use_contribution else 0.0,
    )

    cash = float(initial_amount)
    shares = 0.0
    position = "CASH"

    # Pending confirmation state. The timer is measured from the most recent
    # target-direction signal and is reset whenever another target signal arrives.
    pending_signal: str | None = None
    pending_count = 0
    last_target_signal_idx: int | None = None

    # Buy & Hold benchmark: invest on the first trading day and invest every
    # contribution immediately, regardless of the moving-average signal.
    bh_cash = float(initial_amount)
    bh_shares = 0.0

    daily_rows: list[dict] = []
    event_rows: list[dict] = []

    first_date = pd.Timestamp(data.index[0])

    for i, (dt, row) in enumerate(data.iterrows()):
        dt = pd.Timestamp(dt)
        price = float(row["Close"])
        sma = float(row["SMA"])
        contribution = float(contribution_map.get(dt, 0.0))

        if i == 0:
            bh_shares = bh_cash / price
            bh_cash = 0.0
        if contribution > 0:
            bh_shares += contribution / price

        if price > sma:
            raw_signal = "BUY"
        elif price < sma:
            raw_signal = "SELL"
        else:
            raw_signal = "NEUTRAL"

        previous_position = position
        position_changed = False
        confirmed_signal = None
        decision_reason = None

        target_signal = "BUY" if position == "CASH" else "SELL"
        opposite_signal = "SELL" if position == "CASH" else "BUY"

        # Opposite signal cancels every pending confirmation immediately.
        if raw_signal == opposite_signal:
            pending_signal = None
            pending_count = 0
            last_target_signal_idx = None

        # Target signal advances the confirmation counter and restarts LIMIT.
        elif raw_signal == target_signal:
            if pending_signal != target_signal:
                pending_signal = target_signal
                pending_count = 1
            else:
                pending_count += 1
            last_target_signal_idx = i

            if pending_count >= confirmation_count:
                confirmed_signal = target_signal
                decision_reason = "COUNT"

        # NEUTRAL does not clear the pending state. If enough trading days have
        # elapsed since the most recent target signal, force the pending change.
        elif (
            raw_signal == "NEUTRAL"
            and pending_signal == target_signal
            and pending_count > 0
            and last_target_signal_idx is not None
            and (i - last_target_signal_idx) >= signal_limit_days
        ):
            confirmed_signal = target_signal
            decision_reason = "LIMIT"

        # Execute the confirmed/forced position change. Contribution ordering
        # follows the user-defined rules for same-day contribution + transition.
        if confirmed_signal == "BUY" and position == "CASH":
            cash += contribution
            invested = cash
            shares = cash / price
            cash = 0.0
            position = "STOCK"
            position_changed = True

            if decision_reason == "LIMIT":
                note = (
                    f"BUY {pending_count}/{confirmation_count}회 확인 후 "
                    f"LIMIT {signal_limit_days}거래일 도달로 현금 전액 매수"
                )
            else:
                note = f"BUY {confirmation_count}회 확인 후 현금 전액 매수"
            if contribution:
                note += " (적립금 포함)"

            event_rows.append(
                {
                    "date": dt,
                    "event": "BUY",
                    "price": price,
                    "amount": invested,
                    "contribution": contribution,
                    "note": note,
                }
            )

            pending_signal = None
            pending_count = 0
            last_target_signal_idx = None

        elif confirmed_signal == "SELL" and position == "STOCK":
            proceeds = shares * price
            shares = 0.0
            cash += proceeds
            position = "CASH"
            position_changed = True

            if decision_reason == "LIMIT":
                note = (
                    f"SELL {pending_count}/{confirmation_count}회 확인 후 "
                    f"LIMIT {signal_limit_days}거래일 도달로 보유 주식 전량 매도"
                )
            else:
                note = f"SELL {confirmation_count}회 확인 후 보유 주식 전량 매도"

            event_rows.append(
                {
                    "date": dt,
                    "event": "SELL",
                    "price": price,
                    "amount": proceeds,
                    "contribution": 0.0,
                    "note": note,
                }
            )

            if contribution > 0:
                cash += contribution
                event_rows.append(
                    {
                        "date": dt,
                        "event": "CONTRIBUTION_CASH",
                        "price": price,
                        "amount": contribution,
                        "contribution": contribution,
                        "note": "매도 후 현금 적립",
                    }
                )

            pending_signal = None
            pending_count = 0
            last_target_signal_idx = None

        elif contribution > 0:
            # No actual position change today: contribution follows the current
            # actual position, not the pending/raw signal.
            if position == "CASH":
                cash += contribution
                event_rows.append(
                    {
                        "date": dt,
                        "event": "CONTRIBUTION_CASH",
                        "price": price,
                        "amount": contribution,
                        "contribution": contribution,
                        "note": "포지션 변경 미확정 상태에서 현금 적립",
                    }
                )
            else:
                bought = contribution / price
                shares += bought
                event_rows.append(
                    {
                        "date": dt,
                        "event": "ADD_BUY",
                        "price": price,
                        "amount": contribution,
                        "contribution": contribution,
                        "note": "포지션 유지 중 적립금으로 추가 매수",
                    }
                )

        # Display how many trading days have elapsed since the most recent target
        # confirmation. Zero means the target signal happened today.
        if pending_signal is not None and last_target_signal_idx is not None:
            limit_elapsed_days = i - last_target_signal_idx
            pending_count_display = pending_count
            pending_signal_display = pending_signal
        else:
            limit_elapsed_days = 0
            pending_count_display = 0
            pending_signal_display = None

        portfolio_value = cash + shares * price
        buy_hold_value = bh_cash + bh_shares * price

        daily_rows.append(
            {
                "date": dt,
                "Close": price,
                "SMA": sma,
                "raw_signal": raw_signal,
                "confirmation_progress": pending_count_display,
                "confirmation_target": confirmation_count,
                "pending_signal": pending_signal_display,
                "limit_elapsed_days": limit_elapsed_days,
                "signal_limit_days": signal_limit_days,
                "confirmed_signal": confirmed_signal,
                "decision_reason": decision_reason,
                "position_changed": position_changed,
                "position": position,
                "previous_position": previous_position,
                "cash": cash,
                "shares": shares,
                "contribution": contribution,
                "portfolio_value": portfolio_value,
                "buy_hold_value": buy_hold_value,
            }
        )

    daily = pd.DataFrame(daily_rows).set_index("date")
    events = pd.DataFrame(
        event_rows,
        columns=["date", "event", "price", "amount", "contribution", "note"],
    )

    total_contributions = initial_amount + float(daily["contribution"].sum())
    final_value = float(daily["portfolio_value"].iloc[-1])
    final_bh_value = float(daily["buy_hold_value"].iloc[-1])
    profit = final_value - total_contributions

    metrics = {
        "effective_start_date": first_date.date(),
        "effective_end_date": pd.Timestamp(daily.index[-1]).date(),
        "initial_amount": float(initial_amount),
        "additional_contributions": float(daily["contribution"].sum()),
        "total_contributions": float(total_contributions),
        "final_value": final_value,
        "profit": profit,
        "return_pct": (profit / total_contributions * 100.0) if total_contributions else np.nan,
        "mdd_pct": _max_drawdown(daily["portfolio_value"]) * 100.0,
        "sharpe": sharpe_ratio(daily["portfolio_value"].pct_change().dropna()),
        "sortino": sortino_ratio(daily["portfolio_value"].pct_change().dropna()),
        "calmar": calmar_ratio(daily["portfolio_value"]),
        "xirr": cashflow_xirr(daily.index, daily["contribution"], final_value, initial_amount),
        "buy_hold_final_value": final_bh_value,
        "buy_hold_return_pct": (
            (final_bh_value - total_contributions) / total_contributions * 100.0
            if total_contributions
            else np.nan
        ),
        "buy_count": int((events["event"] == "BUY").sum()) if not events.empty else 0,
        "sell_count": int((events["event"] == "SELL").sum()) if not events.empty else 0,
        "position_changes": int(events["event"].isin(["BUY", "SELL"]).sum()) if not events.empty else 0,
        "confirmation_count": confirmation_count,
        "signal_limit_days": signal_limit_days,
        "ending_position": str(daily["position"].iloc[-1]),
        "ending_cash": float(daily["cash"].iloc[-1]),
        "ending_shares": float(daily["shares"].iloc[-1]),
    }

    return BacktestOutput(daily=daily, events=events, schedule=schedule, metrics=metrics)
