"""Native NiceGUI calculation helpers migrated from the Functions app."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from common.metrics import mdd_recovery_days, sharpe_ratio, sortino_ratio


@dataclass
class ToolResult:
    title: str
    metrics: dict[str, str] = field(default_factory=dict)
    series: dict[str, pd.Series] = field(default_factory=dict)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""


def prices(tickers: list[str], start: dt.date, end: dt.date, *, field: str = "Adj Close") -> pd.DataFrame:
    clean = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not clean:
        raise ValueError("종목을 한 개 이상 입력해 주세요.")
    raw = yf.download(
        clean,
        start=(pd.Timestamp(start) - pd.Timedelta(days=7)).date().isoformat(),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("조회된 가격 데이터가 없습니다.")
    if isinstance(raw.columns, pd.MultiIndex):
        level = field if field in raw.columns.get_level_values(0) else "Adj Close"
        frame = raw[level]
    else:
        column = field if field in raw.columns else "Adj Close"
        frame = raw[[column]].rename(columns={column: clean[0]})
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(clean[0])
    if len(clean) == 1 and len(frame.columns) == 1:
        frame.columns = clean
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    return frame.sort_index().ffill().loc[str(start):str(end)].dropna(how="all")


def comparison(tickers: list[str], start: dt.date, end: dt.date) -> ToolResult:
    frame = prices(tickers, start, end).dropna(axis=1, how="all").dropna()
    if len(frame) < 2:
        raise ValueError("모든 종목의 공통 가격 기간이 충분하지 않습니다.")
    normalized = frame.divide(frame.iloc[0]).mul(100)
    rows = []
    for ticker in frame.columns:
        clean = frame[ticker].dropna()
        daily = clean.pct_change().dropna()
        drawdown = clean / clean.cummax() - 1
        years = max((clean.index[-1] - clean.index[0]).days / 365.2425, 1 / 365.2425)
        rows.append({
            "종목": ticker,
            "기간 수익률(%)": (clean.iloc[-1] / clean.iloc[0] - 1) * 100,
            "연환산 수익률(%)": ((clean.iloc[-1] / clean.iloc[0]) ** (1 / years) - 1) * 100,
            "연 변동성(%)": daily.std() * np.sqrt(252) * 100,
            "MDD(%)": drawdown.min() * 100,
        })
    table = pd.DataFrame(rows).round(2)
    return ToolResult("상대 수익률 비교", {"비교 종목": f"{len(frame.columns)}개"},
                      {str(c): normalized[c].dropna() for c in normalized}, table)


def periodic(ticker: str, start: dt.date, end: dt.date, frequency: str) -> ToolResult:
    series = prices([ticker], start, end).iloc[:, 0].dropna()
    rows = []
    grouped: dict[tuple[int, str], list[tuple[pd.Timestamp, float]]] = {}
    for date, value in series.items():
        timestamp = pd.Timestamp(date)
        if frequency == "월":
            label = f"{timestamp.month}월"
        elif frequency == "분기":
            label = f"{((timestamp.month - 1) // 3) + 1}분기"
        elif frequency == "반기":
            label = "상반기" if timestamp.month <= 6 else "하반기"
        else:
            label = "연간"
        grouped.setdefault((timestamp.year, label), []).append((timestamp, float(value)))
    for (year, label), observations in grouped.items():
        first_date, first_price = observations[0]
        last_date, last_price = observations[-1]
        rows.append({
            "연도": year, "주기": label, "시작일": first_date,
            "종료일": last_date, "수익률": (last_price / first_price - 1) * 100,
        })
    table = pd.DataFrame(rows).sort_values(["연도", "시작일"]).reset_index(drop=True)
    result = table["수익률"] / 100.0
    daily = series.pct_change().dropna()
    drawdown = series / series.cummax() - 1
    years = max((series.index[-1] - series.index[0]).days / 365.2425, 1 / 365.2425)
    metrics = {
        "전체 수익률": f"{(series.iloc[-1] / series.iloc[0] - 1) * 100:.2f}%",
        "연환산 수익률": f"{((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100:.2f}%",
        "연환산 변동성": f"{daily.std(ddof=1) * np.sqrt(252) * 100:.2f}%",
        "최대 낙폭": f"{drawdown.min() * 100:.2f}%",
        "평균": f"{table['수익률'].mean():.2f}%",
        "중앙값": f"{table['수익률'].median():.2f}%",
        "상승 비율": f"{(result > 0).mean() * 100:.1f}%",
    }
    indexed = series / series.iloc[0] * 100
    return ToolResult("주기별 수익률 분석", metrics, {f"{ticker.upper()} · 시작값 100": indexed}, table)


def static_portfolio(
    tickers: list[str], weights: list[float], start: dt.date, end: dt.date,
    initial: float, rebalance: str = "매월", contribution: float = 0.0,
    contribution_frequency: str = "매월",
) -> ToolResult:
    frame = prices(tickers, start, end).dropna()
    if frame.empty:
        raise ValueError("공통 가격 기간이 없습니다.")
    weight = np.asarray(weights, dtype=float)
    if len(weight) != len(frame.columns) or weight.sum() <= 0:
        raise ValueError("종목 수와 동일한 비중을 입력해 주세요.")
    weight = weight / weight.sum()
    dates = pd.DatetimeIndex(frame.index)
    values = frame.to_numpy(dtype=float)
    shares = float(initial) * weight / values[0]
    portfolio_values = np.empty(len(dates), dtype=float)
    invested_values = np.empty(len(dates), dtype=float)
    portfolio_values[0] = float(initial)
    invested_values[0] = float(initial)

    def schedule(frequency: str) -> set[pd.Timestamp]:
        months = {"매월": 1, "매분기": 3, "매년": 12}.get(frequency)
        if months is None:
            return set()
        mapped: set[pd.Timestamp] = set()
        step = 1
        while True:
            planned = dates[0] + pd.DateOffset(months=months * step)
            if planned > dates[-1]:
                break
            position = dates.searchsorted(planned, side="left")
            if position < len(dates):
                mapped.add(dates[position])
            step += 1
        return mapped

    contribution_dates = schedule(contribution_frequency) if contribution > 0 else set()
    rebalance_dates = set(dates[1:]) if rebalance == "매일" else schedule(rebalance)
    total_invested = float(initial)
    events = []
    for index in range(1, len(dates)):
        date = dates[index]
        current_prices = values[index]
        current_value = float(np.sum(shares * current_prices))
        add = float(contribution) if date in contribution_dates else 0.0
        total_invested += add
        if date in rebalance_dates:
            shares = (current_value + add) * weight / current_prices
            if add or rebalance != "매일":
                events.append({"date": date, "type": "리밸런싱", "amount": add})
        elif add:
            current_weights = shares * current_prices / current_value
            shares += add * current_weights / current_prices
            events.append({"date": date, "type": "적립", "amount": add})
        portfolio_values[index] = float(np.sum(shares * current_prices))
        invested_values[index] = total_invested
    nav = pd.Series(portfolio_values, index=dates)
    invested = pd.Series(invested_values, index=dates)
    flows = invested.diff().fillna(0.0)
    performance_nav = (1 + ((nav - flows) / nav.shift(1) - 1).fillna(0.0)).cumprod()
    returns = performance_nav.pct_change().dropna()
    drawdown = performance_nav / performance_nav.cummax() - 1
    years = max((nav.index[-1] - nav.index[0]).days / 365.2425, 1 / 365.2425)
    metrics = {
        "최종 평가액": f"{nav.iloc[-1]:,.0f}",
        "총 투입금": f"{invested.iloc[-1]:,.0f}",
        "누적 수익률": f"{(nav.iloc[-1] / invested.iloc[-1] - 1) * 100:.2f}%",
        "연환산 수익률": f"{((performance_nav.iloc[-1] / performance_nav.iloc[0]) ** (1 / years) - 1) * 100:.2f}%",
        "연 변동성": f"{returns.std() * np.sqrt(252) * 100:.2f}%",
        "MDD": f"{drawdown.min() * 100:.2f}%",
    }
    return ToolResult(
        "정적 포트폴리오", metrics,
        {"포트폴리오": nav, "누적 투입금": invested, "낙폭": drawdown},
        pd.DataFrame(events),
    )


def allocation_sweep(
    tickers: list[str], start: dt.date, end: dt.date, step: int,
    fixed_index: int | None = None, fixed_ratio: int | None = None,
    rebalance: str = "매일", initial: float = 10_000.0,
) -> ToolResult:
    if len(tickers) not in {2, 3}:
        raise ValueError("종목은 2개 또는 3개를 입력해 주세요.")
    rows: list[dict[str, Any]] = []
    units = 100 // step
    if fixed_index is not None and fixed_ratio is not None:
        if not 0 <= fixed_index < len(tickers) or not 0 <= fixed_ratio <= 100:
            raise ValueError("고정 비중 설정이 올바르지 않습니다.")
        remaining = 100 - fixed_ratio
        generated = []
        for value in range(0, remaining + 1, step):
            weights = [0] * len(tickers)
            weights[fixed_index] = fixed_ratio
            others = [index for index in range(len(tickers)) if index != fixed_index]
            if len(others) == 1:
                weights[others[0]] = remaining
                generated.append(tuple(weights))
                break
            weights[others[0]] = value
            weights[others[1]] = remaining - value
            generated.append(tuple(weights))
        combos = iter(generated)
        combo_is_percent = True
    else:
        combos = ((a, units - a) for a in range(units + 1)) if len(tickers) == 2 else (
            (a, b, units - a - b) for a in range(units + 1) for b in range(units - a + 1)
        )
        combo_is_percent = False
    for combo in combos:
        percentages = list(combo) if combo_is_percent else [value * step for value in combo]
        result = static_portfolio(tickers, percentages, start, end, initial, rebalance)
        values = result.series["포트폴리오"].dropna()
        daily = values.pct_change().dropna()
        total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
        years = max((values.index[-1] - values.index[0]).days / 365.2425, 1 / 365.2425)
        cagr = ((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1) * 100
        volatility = daily.std(ddof=1) * np.sqrt(252) * 100 if len(daily) > 1 else 0.0
        drawdown = values / values.cummax() - 1
        mdd = float(drawdown.min() * 100)
        recovery = mdd_recovery_days(values)
        rows.append({
            **{f"{ticker} 비율": value for ticker, value in zip(tickers, percentages)},
            "최종 금액": float(values.iloc[-1]),
            "총수익률": float(total_return),
            "CAGR": float(cagr),
            "변동성": float(volatility),
            "최대낙폭": mdd,
            "샤프지수": float(sharpe_ratio(daily)),
            "Sortino": float(sortino_ratio(daily)),
            "Calmar": float((cagr / 100) / abs(mdd / 100)) if mdd < 0 else np.nan,
            "회복기간(일)": recovery,
        })
    table = pd.DataFrame(rows)
    return ToolResult(
        "비율별 리밸런싱 분석",
        {"조합 수": f"{len(table):,}개", "리밸런싱": rebalance, "기준 투자금": f"{initial:,.0f}"},
        table=table,
    )


def monte_carlo(
    ticker: str, start: dt.date, end: dt.date, years: float, simulations: int,
    initial: float, method: str, monthly_contribution: float = 0.0,
    contribution_frequency: str = "매월",
) -> ToolResult:
    historical = prices([ticker], start, end).iloc[:, 0].dropna()
    log_returns = np.log(historical / historical.shift(1)).dropna()
    if len(log_returns) < 30:
        raise ValueError("시뮬레이션에 필요한 과거 수익률이 부족합니다.")
    days = max(1, int(round(years * 252)))
    rng = np.random.default_rng()
    if method == "normal":
        samples = rng.normal(float(log_returns.mean()), float(log_returns.std(ddof=1)), size=(days, simulations))
    else:
        values = log_returns.to_numpy(dtype=float)
        samples = values[rng.integers(0, len(values), size=(days, simulations))]
    initial_price = float(historical.iloc[-1])
    price_paths = initial_price * np.exp(np.cumsum(samples, axis=0))
    price_paths = np.vstack([np.full((1, simulations), initial_price), price_paths])
    shares = np.full(simulations, float(initial) / initial_price)
    value_paths = np.empty_like(price_paths)
    value_paths[0] = initial
    interval = {"매월": 21, "매분기": 63, "매년": 252}[contribution_frequency]
    contribution_steps = set(range(interval, days + 1, interval)) if monthly_contribution > 0 else set()
    for step in range(1, days + 1):
        if step in contribution_steps:
            shares += monthly_contribution / price_paths[step]
        value_paths[step] = shares * price_paths[step]
    dates = pd.bdate_range(pd.Timestamp(end), periods=days + 1)
    quantiles = np.percentile(value_paths, [5, 50, 95], axis=1)
    final = value_paths[-1]
    metrics = {
        "총 예정 투입금": f"{initial + monthly_contribution * len(contribution_steps):,.0f}",
        "P5": f"{np.quantile(final, .05):,.0f}",
        "P50": f"{np.quantile(final, .5):,.0f}",
        "P95": f"{np.quantile(final, .95):,.0f}",
    }
    return ToolResult(
        "정규분포 몬테카를로" if method == "normal" else "Bootstrap 몬테카를로",
        metrics,
        {"P5": pd.Series(quantiles[0], index=dates), "P50": pd.Series(quantiles[1], index=dates), "P95": pd.Series(quantiles[2], index=dates)},
    )
