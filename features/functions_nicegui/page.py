"""Native NiceGUI screens for all copied Functions calculations."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import numpy as np
import pandas as pd
from nicegui import run, ui

from common.functions.analytics import (
    ToolResult,
    allocation_sweep,
    comparison,
    monte_carlo,
    periodic,
    static_portfolio,
)
from common.functions.specialized import (
    dollar_compass,
    fear_greed,
    financial_dashboard,
    inflation_compass,
    laoer,
    moving_average,
    real_estate_vs_stock,
    reb_statistics,
)
from common.snowball.layout import shell


TOOLS = {
    "return_comparison": "1. 상대 수익률 비교",
    "periodic_returns": "2. 주기별 수익률 분석",
    "portfolio_backtest": "3. 정적 포트폴리오 백테스트",
    "dynamic_portfolio": "4. 동적 포트폴리오 백테스트",
    "allocation_sweep": "5. 비율별 리밸런싱 분석",
    "monte_carlo_normal": "6. 정규분포 몬테카를로",
    "monte_carlo_bootstrap": "7. Bootstrap 몬테카를로",
    "inflation_compass": "8. 인플레이션 나침반",
    "dollar_compass": "9. 달러 환율 나침반",
    "financial_dashboard": "10. 기업 재무 대시보드",
    "moving_average": "11. 이동평균 투자전략",
    "laoer_infinite": "12. 라오어 무한매수법",
    "real_estate_vs_stock": "13. 부동산 vs 주식 투자",
    "fear_greed_rebalance": "14. 공포·탐욕 지수 리밸런싱",
    "reb_statistics": "15. 한국부동산원 부동산 통계",
}


def _date(value: object) -> dt.date:
    return dt.date.fromisoformat(str(value))


def _tickers(value: object) -> list[str]:
    return [item.strip().upper() for item in str(value or "").split(",") if item.strip()]


def _numbers(value: object) -> list[float]:
    return [float(item.strip()) for item in str(value or "").split(",") if item.strip()]


def _chart_options(series: dict[str, pd.Series]) -> dict:
    all_indexes: list[str] = []
    normalized: dict[str, pd.Series] = {}
    for name, raw in series.items():
        clean = pd.Series(raw).replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) > 900:
            positions = np.linspace(0, len(clean) - 1, 900).astype(int)
            clean = clean.iloc[np.unique(positions)]
        normalized[name] = clean
        all_indexes.extend(str(value.date() if hasattr(value, "date") else value) for value in clean.index)
    categories = list(dict.fromkeys(all_indexes))
    output = []
    for name, clean in normalized.items():
        values = {
            str(index.date() if hasattr(index, "date") else index): round(float(value), 5)
            for index, value in clean.items()
        }
        output.append({"name": name, "type": "line", "showSymbol": False, "data": [values.get(item) for item in categories]})
    return {
        "tooltip": {"trigger": "axis"},
        "legend": {"type": "scroll", "bottom": 0},
        "grid": {"left": 55, "right": 20, "top": 25, "bottom": 55},
        "xAxis": {"type": "category", "data": categories, "axisLabel": {"hideOverlap": True}},
        "yAxis": {"type": "value", "scale": True},
        "series": output,
    }


def _render_result(result: ToolResult) -> None:
    ui.label(result.title).classes("section-title")
    if result.metrics:
        with ui.grid(columns=3).classes("result-metric-grid w-full gap-3"):
            for label, value in result.metrics.items():
                with ui.card().classes("metric-card"):
                    with ui.row().classes("metric-row items-center justify-between no-wrap"):
                        ui.label(str(label)).classes("metric-label")
                        ui.label(str(value)).classes("metric-value")
    if result.series:
        ui.echart(_chart_options(result.series)).classes("w-full dashboard-chart")
    if not result.table.empty:
        frame = result.table.copy().tail(500)
        frame.columns = [str(column) for column in frame.columns]
        frame = frame.reset_index(drop=False)
        frame.columns = [str(column) for column in frame.columns]
        rows = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
        columns = [{"name": column, "label": column, "field": column, "sortable": True} for column in frame.columns]
        ui.table(columns=columns, rows=rows, pagination=20).classes("w-full account-table")
    if result.note:
        ui.label(result.note).classes("status-text")


def _common_dates() -> tuple[ui.input, ui.input]:
    today = dt.date.today()
    start = ui.input("시작일", value=(today - dt.timedelta(days=365 * 10)).isoformat()).props("type=date").classes("control-grow")
    end = ui.input("종료일", value=today.isoformat()).props("type=date").classes("control-grow")
    return start, end


@ui.page("/functions/{feature_id}")
def functions_page(feature_id: str) -> None:
    shell()
    title = TOOLS.get(feature_id)
    with ui.column().classes("page-container w-full gap-4"):
        ui.button("전체 기능", icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("flat").classes("page-back-button")
        if title is None:
            ui.label("알 수 없는 기능입니다.").classes("text-negative")
            return
        ui.label(title).classes("page-title")
        ui.label("Functions 원본 계산 소스를 Snowball 저장소에서 실행하는 NiceGUI 화면입니다.").classes("page-subtitle")

        if feature_id == "dynamic_portfolio":
            with ui.card().classes("backtest-controls w-full"):
                ui.label("동적 포트폴리오는 Snowball 개별 백테스트 편집기와 통합되었습니다.").classes("feature-description")
                ui.button("동적 포트폴리오 설정 열기", icon="arrow_forward", on_click=lambda: ui.navigate.to("/backtest")).classes("primary-action")
            return

        with ui.card().classes("backtest-controls w-full"):
            with ui.row().classes("backtest-field-grid w-full"):
                start, end = _common_dates()
            getter: Callable[[], ToolResult]

            if feature_id == "return_comparison":
                tickers = ui.input("종목(쉼표 구분)", value="SPY,QQQ,SCHD").classes("w-full")
                getter = lambda: comparison(_tickers(tickers.value), _date(start.value), _date(end.value))
            elif feature_id == "periodic_returns":
                ticker = ui.input("종목", value="SPY")
                frequency = ui.select(["월간", "분기", "연간"], value="월간", label="수익률 주기")
                getter = lambda: periodic(str(ticker.value), _date(start.value), _date(end.value), str(frequency.value))
            elif feature_id == "portfolio_backtest":
                tickers = ui.input("종목(쉼표 구분)", value="SPY,IEF,GLD")
                weights = ui.input("비중(쉼표 구분)", value="60,30,10")
                initial = ui.number("초기 투자금", value=10_000, min=1)
                rebalance = ui.select(["월간", "연간", "없음"], value="월간", label="리밸런싱")
                getter = lambda: static_portfolio(_tickers(tickers.value), _numbers(weights.value), _date(start.value), _date(end.value), float(initial.value), str(rebalance.value))
            elif feature_id == "allocation_sweep":
                tickers = ui.input("종목 2~3개(쉼표 구분)", value="SPY,IEF")
                step = ui.select([5, 10, 20, 25], value=10, label="비중 간격(%)")
                getter = lambda: allocation_sweep(_tickers(tickers.value), _date(start.value), _date(end.value), int(step.value))
            elif feature_id in {"monte_carlo_normal", "monte_carlo_bootstrap"}:
                ticker = ui.input("종목", value="SPY")
                years = ui.number("예측 기간(년)", value=10, min=1, max=50)
                simulations = ui.number("시뮬레이션 수", value=1000, min=100, max=10000)
                initial = ui.number("초기 투자금", value=10_000, min=1)
                method = "normal" if feature_id.endswith("normal") else "bootstrap"
                getter = lambda: monte_carlo(str(ticker.value), _date(start.value), _date(end.value), int(years.value), int(simulations.value), float(initial.value), method)
            elif feature_id == "inflation_compass":
                initial = ui.number("초기 투자금", value=10_000, min=1)
                getter = lambda: inflation_compass(_date(start.value), _date(end.value), float(initial.value))
            elif feature_id == "dollar_compass":
                initial = ui.number("초기 원화", value=10_000_000, min=1)
                switch = ui.number("전환 비율(%)", value=50, min=1, max=100)
                fee = ui.number("환전 비용(%)", value=0.1, min=0, max=10, step=0.01)
                getter = lambda: dollar_compass(_date(start.value), _date(end.value), float(initial.value), float(switch.value), float(fee.value))
            elif feature_id == "financial_dashboard":
                query = ui.input("기업명 또는 티커", value="Apple")
                source = ui.select(["SEC", "DART"], value="SEC", label="공시 출처")
                getter = lambda: financial_dashboard(str(query.value), _date(start.value).year, _date(end.value).year, str(source.value))
            elif feature_id == "moving_average":
                ticker = ui.input("종목", value="QQQ")
                months = ui.number("이동평균 기간(개월)", value=10, min=1, max=60)
                initial = ui.number("초기 투자금", value=10_000, min=1)
                getter = lambda: moving_average(str(ticker.value), _date(start.value), _date(end.value), int(months.value), float(initial.value))
            elif feature_id == "laoer_infinite":
                ticker = ui.input("종목", value="TQQQ")
                divisions = ui.number("분할 횟수", value=40, min=2, max=200)
                initial = ui.number("초기 투자금", value=10_000, min=1)
                getter = lambda: laoer(str(ticker.value), _date(start.value), _date(end.value), float(initial.value), int(divisions.value))
            elif feature_id == "real_estate_vs_stock":
                ticker = ui.input("주식 종목", value="SPY")
                property_price = ui.number("주택 가격", value=500_000_000, min=1)
                loan = ui.number("대출금", value=300_000_000, min=0)
                initial_stock = ui.number("주식 초기 투자금", value=200_000_000, min=0)
                years = ui.number("비교 기간(년)", value=10, min=1, max=30)
                getter = lambda: real_estate_vs_stock(str(ticker.value), _date(start.value), _date(end.value), float(property_price.value), float(loan.value), float(initial_stock.value), int(years.value))
            elif feature_id == "fear_greed_rebalance":
                ticker = ui.input("투자 종목", value="SPY")
                threshold = ui.number("공포 기준값", value=25, min=0, max=100)
                initial = ui.number("초기 투자금", value=10_000, min=1)
                getter = lambda: fear_greed(_date(start.value), _date(end.value), str(ticker.value), float(threshold.value), float(initial.value))
            else:
                table_id = ui.input("통계표 ID", value="A_2024_00017")
                cycle = ui.select(["MM", "QY", "YY"], value="MM", label="주기")
                region = ui.input("지역 ID", value="500001")
                item = ui.input("항목 ID", value="100001")
                getter = lambda: reb_statistics(str(table_id.value), str(cycle.value), str(region.value), str(item.value), _date(start.value).year, _date(end.value).year)

            run_button = ui.button("계산 실행", icon="play_arrow").classes("primary-action")
            status = ui.label("").classes("status-text")

        result_area = ui.column().classes("w-full gap-4")

        async def execute() -> None:
            run_button.disable()
            result_area.clear()
            status.set_text("데이터를 불러와 계산하고 있습니다…")
            try:
                result = await run.io_bound(getter)
            except Exception as error:
                status.set_text("계산을 완료하지 못했습니다.")
                with result_area:
                    with ui.card().classes("error-card w-full"):
                        ui.label(str(error)).classes("text-negative")
            else:
                with result_area:
                    _render_result(result)
                status.set_text("계산이 완료되었습니다.")
            finally:
                run_button.enable()

        run_button.on("click", execute)
