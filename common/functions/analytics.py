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
    rule = {"월": "ME", "분기": "QE", "반기": "2QE", "년": "YE"}[frequency]
    result = series.resample(rule).last().pct_change().dropna()
    table = pd.DataFrame({"기간": result.index.strftime("%Y-%m"), "수익률(%)": (result.values * 100).round(2)})
    metrics = {
        "평균": f"{result.mean() * 100:.2f}%",
        "중앙값": f"{result.median() * 100:.2f}%",
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
    daily = frame.pct_change().fillna(0.0)
    invested_values: list[float]
    if rebalance == "없음" and contribution <= 0:
        shares = initial * weight / frame.iloc[0].to_numpy()
        nav = pd.Series(frame.to_numpy() @ shares, index=frame.index)
        invested_values = [initial] * len(nav)
    else:
        nav_values = [float(initial)]
        invested_values = [float(initial)]
        current_weight = weight.copy()
        rebalance_months = {"매일": 0, "매월": 1, "매분기": 3, "매년": 12, "없음": -1}[rebalance]
        contribution_months = {"매월": 1, "매분기": 3, "매년": 12}.get(contribution_frequency, 1)
        anchor = frame.index[0]
        last_rebalance = anchor
        last_contribution = anchor
        for date, row in daily.iloc[1:].iterrows():
            elapsed_rebalance = (date.year - last_rebalance.year) * 12 + date.month - last_rebalance.month
            elapsed_contribution = (date.year - last_contribution.year) * 12 + date.month - last_contribution.month
            add = float(contribution) if contribution > 0 and elapsed_contribution >= contribution_months else 0.0
            if add:
                last_contribution = date
            should_rebalance = rebalance_months == 0 or (rebalance_months > 0 and elapsed_rebalance >= rebalance_months)
            if should_rebalance:
                current_weight = weight.copy()
                last_rebalance = date
            growth = current_weight * (1 + row.to_numpy())
            nav_values.append(nav_values[-1] * growth.sum() + add)
            current_weight = growth / growth.sum()
            invested_values.append(invested_values[-1] + add)
        nav = pd.Series(nav_values, index=frame.index)
    invested = pd.Series(invested_values, index=frame.index)
    returns = nav.pct_change().dropna()
    drawdown = nav / nav.cummax() - 1
    years = max((nav.index[-1] - nav.index[0]).days / 365.2425, 1 / 365.2425)
    metrics = {
        "최종 평가액": f"{nav.iloc[-1]:,.0f}",
        "총 투입금": f"{invested.iloc[-1]:,.0f}",
        "누적 수익률": f"{(nav.iloc[-1] / invested.iloc[-1] - 1) * 100:.2f}%",
        "연환산 수익률": f"{((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) * 100:.2f}%",
        "연 변동성": f"{returns.std() * np.sqrt(252) * 100:.2f}%",
        "MDD": f"{drawdown.min() * 100:.2f}%",
    }
    return ToolResult("정적 포트폴리오", metrics, {"포트폴리오": nav, "누적 투입금": invested, "낙폭": drawdown})


def allocation_sweep(
    tickers: list[str], start: dt.date, end: dt.date, step: int,
    fixed_index: int | None = None, fixed_ratio: int | None = None,
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
        result = static_portfolio(tickers, percentages, start, end, 10_000)
        rows.append({**{ticker: value for ticker, value in zip(tickers, percentages)}, **result.metrics})
    table = pd.DataFrame(rows)
    table["MDD 값"] = table["MDD"].str.rstrip("%").astype(float)
    table["수익률 값"] = table["연환산 수익률"].str.rstrip("%").astype(float)
    table = table.sort_values(["MDD 값", "수익률 값"], ascending=[False, False]).drop(columns=["MDD 값", "수익률 값"])
    return ToolResult("비율별 리밸런싱 분석", {"조합 수": f"{len(table):,}개"}, table=table)


def monte_carlo(
    ticker: str, start: dt.date, end: dt.date, years: int, simulations: int,
    initial: float, method: str, monthly_contribution: float = 0.0,
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
    paths = np.empty_like(samples)
    current = np.full(simulations, float(initial))
    for day in range(days):
        current = current * (1 + samples[day])
        if monthly_contribution > 0 and day % 21 == 20:
            current += monthly_contribution
        paths[day] = current
    dates = pd.bdate_range(pd.Timestamp(end) + pd.Timedelta(days=1), periods=days)
    quantiles = np.quantile(paths, [0.05, 0.5, 0.95], axis=1)
    final = paths[-1]
    metrics = {
        "총 예정 투입금": f"{initial + monthly_contribution * years * 12:,.0f}",
        "P5": f"{np.quantile(final, .05):,.0f}",
        "P50": f"{np.quantile(final, .5):,.0f}",
        "P95": f"{np.quantile(final, .95):,.0f}",
    }
    return ToolResult(
        "정규분포 몬테카를로" if method == "normal" else "Bootstrap 몬테카를로",
        metrics,
        {"P5": pd.Series(quantiles[0], index=dates), "P50": pd.Series(quantiles[1], index=dates), "P95": pd.Series(quantiles[2], index=dates)},
    )
