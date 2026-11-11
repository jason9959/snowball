"""Adapters from the copied Functions calculation modules to NiceGUI results."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common.functions.analytics import ToolResult, prices


def inflation_compass(start: dt.date, end: dt.date, initial: float) -> ToolResult:
    from features.feature08_inflation.compass import Config, backtest, metrics, signals

    root = Path(__file__).resolve().parents[2] / "features" / "feature08_inflation" / "data"
    price_frame = pd.read_csv(root / "prices.csv", index_col=0, parse_dates=True)
    fred = pd.read_csv(root / "T5YIE.csv", index_col=0, parse_dates=True).iloc[:, 0]
    price_frame.index = pd.DatetimeIndex(price_frame.index).tz_localize(None)
    fred.index = pd.DatetimeIndex(fred.index).tz_localize(None)
    signal = signals(price_frame, fred, Config())
    nav, spy, events, regimes = backtest(price_frame, signal, start, end, Config())
    stat = metrics(nav)
    return ToolResult(
        "인플레이션 나침반",
        {
            "최종 평가액": f"{nav.iloc[-1] * initial:,.0f}",
            "연환산 수익률": f"{stat['CAGR'] * 100:.2f}%",
            "MDD": f"{stat['MaxDD'] * 100:.2f}%",
            "현재 국면": str(regimes.iloc[-1]) if len(regimes) else "-",
        },
        {"나침반": nav * initial, "SPY": spy * initial},
        events.tail(60),
    )


def dollar_compass(
    start: dt.date, end: dt.date, initial: float, switch_percent: float, fee_percent: float,
) -> ToolResult:
    from features.feature09_dollar.page import add_indicators, backtest_switching, load_market_data, max_drawdown

    data = add_indicators(load_market_data())
    curve, switches = backtest_switching(
        data, start, end, initial, False, 0.0,
        switch_percent / 100.0, fee_percent / 100.0,
    )
    if curve.empty:
        raise ValueError("선택한 기간에 환율 자료가 없습니다.")
    value = curve.set_index("date")["total_value"]
    return ToolResult(
        "달러 환율 나침반",
        {
            "최종 평가액": f"{value.iloc[-1]:,.0f}원",
            "수익률": f"{curve.iloc[-1]['return'] * 100:.2f}%",
            "MDD": f"{max_drawdown(value) * 100:.2f}%",
            "환전 횟수": f"{len(switches):,}회",
        },
        {"포트폴리오": value, "원/달러": curve.set_index("date")["usdkrw"]},
        switches.tail(100),
    )


def financial_dashboard(query: str, start_year: int, end_year: int, source: str) -> ToolResult:
    from features.feature10_financial.data_sources import fetch_dart_companies, fetch_sec_companies, search_companies
    from features.feature10_financial.financials import load_financials

    dart_key = os.getenv("DART_API_KEY", "").strip()
    sec_agent = os.getenv("SEC_USER_AGENT", "snowball-portfolio-lab contact@example.com").strip()
    if source == "DART":
        if not dart_key:
            raise ValueError("DART_API_KEY 환경변수가 필요합니다.")
        companies = fetch_dart_companies(dart_key)
    else:
        companies = fetch_sec_companies(sec_agent)
    matches = search_companies(companies, query, limit=1)
    if not matches:
        raise ValueError("검색어와 일치하는 기업이 없습니다.")
    company = matches[0]
    frame, unit = load_financials(company, start_year, end_year, False, dart_key, sec_agent)
    latest = frame.dropna(how="all").iloc[-1]
    metrics = {str(label): f"{float(value):,.1f} {unit}" for label, value in latest.dropna().head(4).items()}
    return ToolResult(f"{company.name} 재무 대시보드", metrics, table=frame.reset_index())


def moving_average(
    ticker: str, start: dt.date, end: dt.date, months: int, initial: float,
) -> ToolResult:
    from features.feature11_moving_average.backtest import run_backtest
    from features.feature11_moving_average.data_loader import load_price_data

    price = load_price_data(ticker, start, end, months)
    output = run_backtest(price, start, end, initial, False, 0.0, "매월")
    metrics = {str(key): (f"{value * 100:.2f}%" if "return" in str(key).lower() or "mdd" in str(key).lower() else f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)) for key, value in output.metrics.items()}
    return ToolResult(
        "이동평균 투자전략",
        metrics,
        {"포트폴리오": output.daily["portfolio_value"], "종가": output.daily["Close"]},
        output.events.tail(100),
    )


def laoer(ticker: str, start: dt.date, end: dt.date, initial: float, divisions: int) -> ToolResult:
    from features.feature12_laoer.backtest import run_v22
    from features.feature12_laoer.data_loader import load_ohlc

    output = run_v22(load_ohlc(ticker, start, end), initial, divisions)
    metrics = {str(key): (f"{value * 100:.2f}%" if "return" in str(key).lower() or "mdd" in str(key).lower() else f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)) for key, value in output.metrics.items()}
    return ToolResult("라오어 무한매수법", metrics, {"포트폴리오": output.daily["portfolio_value"]}, output.trades.tail(100))


def real_estate_vs_stock(
    ticker: str, start: dt.date, end: dt.date, property_price: float,
    loan: float, initial_stock: float, years: int,
) -> ToolResult:
    from features.feature13_real_estate_vs_stock.monte_carlo import historical_price_path
    from features.feature13_real_estate_vs_stock.simulation import simulate_monthly_comparison

    close = prices([ticker], start, end).iloc[:, 0].dropna()
    path = historical_price_path(close, years).to_numpy()[:, None]
    result = simulate_monthly_comparison(
        path,
        property_price=property_price,
        property_growth_rate=2.0,
        loan=loan,
        annual_rate=4.0,
        loan_years=max(years, 1),
        repayment_method="원리금균등",
        initial_stock_investment=initial_stock,
        monthly_rent=1_000_000,
        rent_growth_cycle_months=24,
        rent_growth_rate=5.0,
        horizon_months=years * 12,
    )
    index = pd.RangeIndex(0, years * 12 + 1, name="개월")
    house = pd.Series(result["house_equity"], index=index)
    stock = pd.Series(result["stock_values"][:, 0], index=index)
    return ToolResult(
        "부동산 vs 주식",
        {"주택 순자산": f"{house.iloc[-1]:,.0f}원", "주식 평가액": f"{stock.iloc[-1]:,.0f}원"},
        {"주택 순자산": house, "주식": stock},
    )


def fear_greed(
    start: dt.date, end: dt.date, ticker: str, threshold: float, initial: float,
) -> ToolResult:
    from features.feature14_fear_greed.backtest import run_backtest
    from features.feature14_fear_greed.data import fetch_fear_greed, fetch_prices

    indicator_frame = fetch_fear_greed(start.isoformat(), end.isoformat())
    indicator = indicator_frame["fear_and_greed_historical"]
    price = fetch_prices((ticker,), start.isoformat(), end.isoformat())
    output = run_backtest(price, indicator, np.array([1.0]), initial, "매월", threshold, "이하일 때")
    value = output["values"]
    return ToolResult(
        "공포·탐욕 리밸런싱",
        {"최종 평가액": f"{value.iloc[-1]:,.0f}", "MDD": f"{output['drawdown'].min() * 100:.2f}%", "리밸런싱": f"{len(output['events'])}회"},
        {"포트폴리오": value, "공포·탐욕": output["indicator"]},
        output["events"].tail(100),
    )


def reb_statistics(
    table_id: str, cycle: str, region_id: str, item_id: str,
    start_year: int, end_year: int,
) -> ToolResult:
    from features.feature15_reb_statistics.analysis import analyze_statistics
    from features.feature15_reb_statistics.api import fetch_statistics, resolve_api_key

    frame, metadata = fetch_statistics(
        stat_table_id=table_id,
        cycle=cycle,
        region_id=region_id,
        item_id=item_id,
        start_year=start_year,
        end_year=end_year,
        api_key=resolve_api_key(),
    )
    analyzed = analyze_statistics(frame, "index")
    series = pd.Series(analyzed["값"].to_numpy(), index=pd.DatetimeIndex(analyzed["날짜"]))
    return ToolResult(
        "한국부동산원 부동산 통계",
        {"조회 건수": f"{len(analyzed):,}건", "전체 건수": f"{metadata['total_count']:,}건"},
        {"지수": series},
        analyzed.tail(200),
        "API 키가 없으면 R-ONE 표본 조회 모드로 실행됩니다." if metadata["sample_mode"] else "",
    )
