"""Native NiceGUI calculation helpers migrated from the Functions app."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class ToolResult:
    title: str
    metrics: dict[str, str] = field(default_factory=dict)
    series: dict[str, pd.Series] = field(default_factory=dict)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""


def prices(tickers: list[str], start: dt.date, end: dt.date, *, field: str = "Close") -> pd.DataFrame:
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
    frame = prices(tickers, start, end).dropna(axis=1, how="all")
    normalized = frame.divide(frame.apply(lambda col: col.dropna().iloc[0])).mul(100)
    returns = (frame.iloc[-1] / frame.apply(lambda col: col.dropna().iloc[0]) - 1) * 100
    table = returns.rename("기간 수익률(%)").round(2).to_frame()
    return ToolResult("상대 수익률 비교", {"비교 종목": f"{len(frame.columns)}개"},
                      {str(c): normalized[c].dropna() for c in normalized}, table)


def periodic(ticker: str, start: dt.date, end: dt.date, frequency: str) -> ToolResult:
    series = prices([ticker], start, end).iloc[:, 0].dropna()
    rule = {"월간": "ME", "분기": "QE", "연간": "YE"}[frequency]
    result = series.resample(rule).last().pct_change().dropna()
    table = pd.DataFrame({"기간": result.index.strftime("%Y-%m"), "수익률(%)": (result.values * 100).round(2)})
    metrics = {
        "평균": f"{result.mean() * 100:.2f}%",
        "중앙값": f"{result.median() * 100:.2f}%",
        "상승 비율": f"{(result > 0).mean() * 100:.1f}%",
    }
    return ToolResult("주기별 수익률 분석", metrics, {ticker.upper(): result}, table)


def static_portfolio(
    tickers: list[str], weights: list[float], start: dt.date, end: dt.date,
    initial: float, rebalance: str = "월간",
) -> ToolResult:
    frame = prices(tickers, start, end).dropna()
    if frame.empty:
        raise ValueError("공통 가격 기간이 없습니다.")
    weight = np.asarray(weights, dtype=float)
    if len(weight) != len(frame.columns) or weight.sum() <= 0:
        raise ValueError("종목 수와 동일한 비중을 입력해 주세요.")
    weight = weight / weight.sum()
    daily = frame.pct_change().fillna(0.0)
    if rebalance == "없음":
        shares = initial * weight / frame.iloc[0].to_numpy()
        nav = pd.Series(frame.to_numpy() @ shares, index=frame.index)
    else:
        nav_values = [float(initial)]
        current_weight = weight.copy()
        last_period = frame.index[0].to_period("M" if rebalance == "월간" else "Y")
        for date, row in daily.iloc[1:].iterrows():
            period = date.to_period("M" if rebalance == "월간" else "Y")
            if period != last_period:
                current_weight = weight.copy()
                last_period = period
            growth = current_weight * (1 + row.to_numpy())
            nav_values.append(nav_values[-1] * growth.sum())
            current_weight = growth / growth.sum()
        nav = pd.Series(nav_values, index=frame.index)
    returns = nav.pct_change().dropna()
    drawdown = nav / nav.cummax() - 1
    years = max((nav.index[-1] - nav.index[0]).days / 365.2425, 1 / 365.2425)
    metrics = {
        "최종 평가액": f"{nav.iloc[-1]:,.0f}",
        "누적 수익률": f"{(nav.iloc[-1] / nav.iloc[0] - 1) * 100:.2f}%",
        "연환산 수익률": f"{((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) * 100:.2f}%",
        "연 변동성": f"{returns.std() * np.sqrt(252) * 100:.2f}%",
        "MDD": f"{drawdown.min() * 100:.2f}%",
    }
    return ToolResult("정적 포트폴리오", metrics, {"포트폴리오": nav, "낙폭": drawdown})


def allocation_sweep(tickers: list[str], start: dt.date, end: dt.date, step: int) -> ToolResult:
    if len(tickers) not in {2, 3}:
        raise ValueError("종목은 2개 또는 3개를 입력해 주세요.")
    rows: list[dict[str, Any]] = []
    units = 100 // step
    combos = ((a, units - a) for a in range(units + 1)) if len(tickers) == 2 else (
        (a, b, units - a - b) for a in range(units + 1) for b in range(units - a + 1)
    )
    for combo in combos:
        result = static_portfolio(tickers, [value * step for value in combo], start, end, 10_000)
        rows.append({**{ticker: value * step for ticker, value in zip(tickers, combo)}, **result.metrics})
    table = pd.DataFrame(rows)
    table["MDD 값"] = table["MDD"].str.rstrip("%").astype(float)
    table["수익률 값"] = table["연환산 수익률"].str.rstrip("%").astype(float)
    table = table.sort_values(["MDD 값", "수익률 값"], ascending=[False, False]).drop(columns=["MDD 값", "수익률 값"])
    return ToolResult("비율별 리밸런싱 분석", {"조합 수": f"{len(table):,}개"}, table=table)


def monte_carlo(
    ticker: str, start: dt.date, end: dt.date, years: int, simulations: int,
    initial: float, method: str,
) -> ToolResult:
    history = prices([ticker], start, end).iloc[:, 0].pct_change().dropna().to_numpy()
    if len(history) < 30:
        raise ValueError("시뮬레이션에 필요한 과거 수익률이 부족합니다.")
    days = int(years * 252)
    rng = np.random.default_rng(42)
    if method == "normal":
        samples = rng.normal(history.mean(), history.std(ddof=1), size=(days, simulations))
    else:
        samples = rng.choice(history, size=(days, simulations), replace=True)
    paths = initial * np.cumprod(1 + samples, axis=0)
    dates = pd.bdate_range(pd.Timestamp(end) + pd.Timedelta(days=1), periods=days)
    quantiles = np.quantile(paths, [0.05, 0.5, 0.95], axis=1)
    final = paths[-1]
    metrics = {
        "P5": f"{np.quantile(final, .05):,.0f}",
        "P50": f"{np.quantile(final, .5):,.0f}",
        "P95": f"{np.quantile(final, .95):,.0f}",
    }
    return ToolResult(
        "정규분포 몬테카를로" if method == "normal" else "Bootstrap 몬테카를로",
        metrics,
        {"P5": pd.Series(quantiles[0], index=dates), "P50": pd.Series(quantiles[1], index=dates), "P95": pd.Series(quantiles[2], index=dates)},
    )
