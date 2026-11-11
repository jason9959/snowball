from __future__ import annotations

import numpy as np
import pandas as pd


def _scheduled_dates(index: pd.DatetimeIndex, frequency: str) -> set[pd.Timestamp]:
    months = {"매일": 1, "매월": 1, "매분기": 3, "매년": 12}.get(frequency, 1)
    if frequency == "매일":
        return set(index[1:])
    dates: set[pd.Timestamp] = set()
    step = 1
    while True:
        target = index[0] + pd.DateOffset(months=months * step)
        if target > index[-1]:
            break
        pos = index.searchsorted(target, side="left")
        if pos < len(index):
            dates.add(index[pos])
        step += 1
    return dates


def run_backtest(
    prices: pd.DataFrame,
    indicator: pd.Series,
    weights: np.ndarray,
    initial: float,
    rebalance_frequency: str,
    threshold: float,
    direction: str,
    contribution: float = 0.0,
    contribution_frequency: str = "매월",
) -> dict:
    """지표 조건이 충족된 예정일에만 목표 비중으로 재조정한다."""
    aligned_indicator = indicator.reindex(prices.index).ffill()
    valid = aligned_indicator.notna()
    prices = prices.loc[valid]
    aligned_indicator = aligned_indicator.loc[valid]
    if len(prices) < 2:
        raise ValueError("지표와 주가가 겹치는 거래일이 충분하지 않습니다.")

    dates = pd.DatetimeIndex(prices.index)
    values = prices.to_numpy(dtype=float)
    shares = float(initial) * weights / values[0]
    portfolio = np.empty(len(dates), dtype=float)
    invested = np.empty(len(dates), dtype=float)
    portfolio[0] = float(initial)
    invested[0] = float(initial)
    rebalance_dates = _scheduled_dates(dates, rebalance_frequency)
    contribution_dates = _scheduled_dates(dates, contribution_frequency) if contribution > 0 else set()
    events = []
    total_invested = float(initial)

    def triggered(score: float) -> bool:
        return score <= threshold if direction == "이하일 때" else score >= threshold

    for pos in range(1, len(dates)):
        date = dates[pos]
        price = values[pos]
        current = float(np.sum(shares * price))
        add = float(contribution) if date in contribution_dates else 0.0
        total_invested += add
        should_rebalance = date in rebalance_dates and triggered(float(aligned_indicator.loc[date]))
        if should_rebalance:
            shares = (current + add) * weights / price
            events.append({"date": date, "type": "조건 충족 리밸런싱", "지표값": float(aligned_indicator.loc[date]), "투입금": add})
        elif add:
            current_weights = shares * price / current if current else weights
            shares += add * current_weights / price
            events.append({"date": date, "type": "적립", "지표값": float(aligned_indicator.loc[date]), "투입금": add})
        portfolio[pos] = float(np.sum(shares * price))
        invested[pos] = total_invested

    series = pd.Series(portfolio, index=dates, name="portfolio_value")
    invested_series = pd.Series(invested, index=dates, name="invested_capital")
    drawdown = series / series.cummax() - 1
    return {
        "values": series,
        "invested": invested_series,
        "drawdown": drawdown,
        "indicator": aligned_indicator,
        "events": pd.DataFrame(events),
    }

