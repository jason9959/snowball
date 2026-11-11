"""Native NiceGUI screens for all copied Functions calculations."""

from __future__ import annotations

import datetime as dt
import io
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
    with ui.tabs().classes("dashboard-tabs w-full") as tabs:
        chart_tab = ui.tab("차트", icon="show_chart")
        detail_tab = ui.tab("세부내역", icon="table_rows")
        download_tab = ui.tab("다운로드", icon="download")
    with ui.tab_panels(tabs, value=chart_tab).classes("dashboard-panels w-full"):
        with ui.tab_panel(chart_tab):
            if result.series:
                ui.echart(_chart_options(result.series)).classes("w-full dashboard-chart")
            else:
                ui.label("이 결과에는 차트 자료가 없습니다.").classes("empty-state")
        with ui.tab_panel(detail_tab):
            if not result.table.empty:
                frame = result.table.copy().tail(500)
                frame.columns = [str(column) for column in frame.columns]
                frame = frame.reset_index(drop=False)
                frame.columns = [str(column) for column in frame.columns]
                for column in frame.columns:
                    if pd.api.types.is_datetime64_any_dtype(frame[column]):
                        frame[column] = frame[column].dt.strftime("%Y-%m-%d")
                rows = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
                columns = [{"name": column, "label": column, "field": column, "sortable": True} for column in frame.columns]
                ui.table(columns=columns, rows=rows, pagination=20).classes("w-full account-table")
            else:
                ui.label("표시할 세부내역이 없습니다.").classes("empty-state")
        with ui.tab_panel(download_tab):
            with ui.row().classes("gap-3"):
                if not result.table.empty:
                    csv_data = result.table.to_csv(index=True).encode("utf-8-sig")
                    ui.button("CSV 다운로드", icon="download", on_click=lambda: ui.download(csv_data, "result.csv"))
                if result.series or not result.table.empty:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        if result.series:
                            pd.concat(result.series, axis=1).to_excel(writer, sheet_name="차트 데이터")
                        if not result.table.empty:
                            result.table.to_excel(writer, sheet_name="세부내역", index=True)
                    excel_data = output.getvalue()
                    ui.button("Excel 다운로드", icon="download", on_click=lambda: ui.download(excel_data, "result.xlsx"))
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
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("분석 조건").classes("section-title")
                ui.button(
                    "조건 초기화", icon="restart_alt",
                    on_click=lambda: ui.navigate.to(f"/functions/{feature_id}"),
                ).props("outline")
            with ui.row().classes("backtest-field-grid w-full"):
                start, end = _common_dates()
            getter: Callable[[], ToolResult]

            if feature_id == "return_comparison":
                ui.separator()
                ui.label("비교 종목 입력").classes("section-title")
                ui.label("최대 10개까지 입력할 수 있습니다. 티커 또는 종목 코드를 입력하세요.").classes("status-text")
                ticker_inputs = []
                defaults = ["SPY", "QQQ", "SCHD"] + [""] * 7
                with ui.grid(columns=2).classes("backtest-field-grid w-full"):
                    for index, default in enumerate(defaults, start=1):
                        ticker_inputs.append(ui.input(f"종목 {index}", value=default, placeholder="예: AAPL"))
                getter = lambda: comparison(
                    list(dict.fromkeys(str(field.value).strip().upper() for field in ticker_inputs if str(field.value or "").strip())),
                    _date(start.value), _date(end.value),
                )
            elif feature_id == "periodic_returns":
                with ui.row().classes("backtest-field-grid w-full"):
                    ticker = ui.input("분석 종목", value="SPY", placeholder="예: SPY 또는 AAPL")
                    frequency = ui.select(["월", "분기", "반기", "년"], value="월", label="집계 주기")
                ui.label("조회 구간이 주기 중간에 걸치면 해당 부분 기간도 포함합니다.").classes("status-text")
                getter = lambda: periodic(str(ticker.value), _date(start.value), _date(end.value), str(frequency.value))
            elif feature_id == "portfolio_backtest":
                with ui.row().classes("backtest-field-grid w-full"):
                    initial = ui.number("초기 투자금", value=10_000, min=1)
                    investment_type = ui.select(["거치식", "적립식"], value="거치식", label="투자 방식")
                    rebalance = ui.select(["매일", "매월", "매분기", "매년", "없음"], value="매월", label="리밸런싱 주기")
                with ui.row().classes("backtest-field-grid w-full") as contribution_fields:
                    contribution = ui.number("정기 적립금", value=1000, min=1)
                    contribution_frequency = ui.select(["매월", "매분기", "매년"], value="매월", label="적립 주기")
                contribution_fields.set_visibility(False)

                def toggle_contribution() -> None:
                    contribution_fields.set_visibility(str(investment_type.value) == "적립식")

                investment_type.on_value_change(lambda _: toggle_contribution())
                ui.separator()
                ui.label("포트폴리오 구성").classes("section-title")
                ui.label("최대 5개 자산의 목표 비중을 입력하세요. 합계는 100%여야 합니다.").classes("status-text")
                ticker_inputs = []
                weight_inputs = []
                defaults = [("SPY", 60), ("IEF", 30), ("GLD", 10), ("", 0), ("", 0)]
                for index, (default_ticker, default_weight) in enumerate(defaults, start=1):
                    with ui.row().classes("backtest-field-grid w-full"):
                        ticker_inputs.append(ui.input(f"자산 {index}", value=default_ticker, placeholder="예: VTI"))
                        weight_inputs.append(ui.number(f"비중 {index}(%)", value=default_weight, min=0, max=100, step=1))

                def portfolio_result() -> ToolResult:
                    pairs = [
                        (str(ticker.value).strip().upper(), float(weight.value or 0))
                        for ticker, weight in zip(ticker_inputs, weight_inputs)
                        if str(ticker.value or "").strip() and float(weight.value or 0) > 0
                    ]
                    return static_portfolio(
                        [ticker for ticker, _ in pairs], [weight for _, weight in pairs],
                        _date(start.value), _date(end.value), float(initial.value), str(rebalance.value),
                        float(contribution.value) if str(investment_type.value) == "적립식" else 0.0,
                        str(contribution_frequency.value),
                    )

                getter = portfolio_result
            elif feature_id == "allocation_sweep":
                ui.label("자산과 탐색 간격").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    tickers = ui.input("종목 2~3개(쉼표 구분)", value="SPY,IEF")
                    step = ui.select([5, 10, 20, 25], value=10, label="비중 간격(%)")
                fixed = ui.checkbox("한 종목 비중 고정", value=False)
                with ui.row().classes("backtest-field-grid w-full") as fixed_fields:
                    fixed_index = ui.number("고정할 종목 순번", value=1, min=1, max=3)
                    fixed_ratio = ui.number("고정 비중(%)", value=40, min=0, max=100, step=5)
                fixed_fields.set_visibility(False)
                fixed.on_value_change(lambda _: fixed_fields.set_visibility(bool(fixed.value)))
                getter = lambda: allocation_sweep(
                    _tickers(tickers.value), _date(start.value), _date(end.value), int(step.value),
                    int(fixed_index.value) - 1 if fixed.value else None,
                    int(fixed_ratio.value) if fixed.value else None,
                )
            elif feature_id in {"monte_carlo_normal", "monte_carlo_bootstrap"}:
                ui.label("시뮬레이션 설정").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    ticker = ui.input("종목", value="SPY")
                    years = ui.number("예측 기간(년)", value=10, min=1, max=50)
                    simulations = ui.number("시뮬레이션 수", value=1000, min=100, max=10000)
                    initial = ui.number("초기 투자금", value=10_000, min=1)
                recurring = ui.checkbox("적립식 포함", value=False)
                monthly = ui.number("월 적립금", value=500, min=0).classes("control-grow")
                monthly.set_visibility(False)
                recurring.on_value_change(lambda _: monthly.set_visibility(bool(recurring.value)))
                method = "normal" if feature_id.endswith("normal") else "bootstrap"
                getter = lambda: monte_carlo(
                    str(ticker.value), _date(start.value), _date(end.value), int(years.value),
                    int(simulations.value), float(initial.value), method,
                    float(monthly.value) if recurring.value else 0.0,
                )
            elif feature_id == "inflation_compass":
                ui.label("나침반 판정 기준").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    initial = ui.number("초기 투자금", value=10_000, min=1)
                    threshold = ui.number("기대인플레이션 기준(%)", value=2.0, step=0.1)
                    growth_window = ui.number("성장 이동평균(거래일)", value=200, min=20)
                    be_window = ui.number("기대물가 모멘텀 기간", value=60, min=1)
                    asset_window = ui.number("자산 모멘텀 기간", value=60, min=1)
                with ui.expansion("시차 및 거래비용", icon="tune").classes("strategy-expansion w-full"):
                    with ui.row().classes("backtest-field-grid w-full"):
                        fred_lag = ui.number("FRED 지연(거래일)", value=0, min=0)
                        execution_lag = ui.number("체결 지연(거래일)", value=0, min=0, max=1)
                        cost = ui.number("거래비용(bps)", value=0, min=0)
                getter = lambda: inflation_compass(
                    _date(start.value), _date(end.value), float(initial.value), float(threshold.value),
                    int(growth_window.value), int(be_window.value), int(asset_window.value),
                    int(fred_lag.value), int(execution_lag.value), float(cost.value),
                )
            elif feature_id == "dollar_compass":
                ui.label("원화·달러 전환 설정").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    initial = ui.number("초기 원화", value=10_000_000, min=1)
                    switch = ui.number("전환 비율(%)", value=50, min=1, max=100)
                    fee = ui.number("환전 비용(%)", value=0.1, min=0, max=10, step=0.01)
                recurring = ui.checkbox("월 적립식 포함", value=False)
                monthly = ui.number("월 적립금", value=500_000, min=0)
                monthly.set_visibility(False)
                recurring.on_value_change(lambda _: monthly.set_visibility(bool(recurring.value)))
                getter = lambda: dollar_compass(
                    _date(start.value), _date(end.value), float(initial.value),
                    float(switch.value), float(fee.value), float(monthly.value) if recurring.value else 0.0,
                )
            elif feature_id == "financial_dashboard":
                ui.label("공식 공시 조회").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    query = ui.input("기업명·티커·종목코드", value="Apple")
                    source = ui.select(["SEC", "DART"], value="SEC", label="공시 출처")
                    frequency = ui.select(["연간", "분기"], value="연간", label="보고 주기")
                ui.label("DART는 DART_API_KEY, SEC는 SEC_USER_AGENT 환경변수를 사용합니다.").classes("status-text")
                getter = lambda: financial_dashboard(
                    str(query.value), _date(start.value).year, _date(end.value).year,
                    str(source.value), str(frequency.value) == "분기",
                )
            elif feature_id == "moving_average":
                ui.label("이동평균·시그널 설정").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    ticker = ui.input("종목", value="QQQ")
                    months = ui.number("이동평균 기간(개월)", value=10, min=1, max=60)
                    confirmation = ui.number("확인 횟수", value=1, min=1)
                    limit_days = ui.number("시그널 대기 LIMIT(거래일)", value=5, min=1)
                    initial = ui.number("초기 투자금", value=10_000, min=1)
                recurring = ui.checkbox("정기 적립", value=False)
                with ui.row().classes("backtest-field-grid w-full") as recurring_fields:
                    contribution = ui.number("적립 금액", value=500, min=1)
                    contribution_frequency = ui.select(["매월", "매분기", "매년"], value="매월", label="적립 주기")
                recurring_fields.set_visibility(False)
                recurring.on_value_change(lambda _: recurring_fields.set_visibility(bool(recurring.value)))
                getter = lambda: moving_average(
                    str(ticker.value), _date(start.value), _date(end.value), int(months.value),
                    float(initial.value), float(contribution.value) if recurring.value else 0.0,
                    str(contribution_frequency.value), int(confirmation.value), int(limit_days.value),
                )
            elif feature_id == "laoer_infinite":
                ui.label("V2.2 분할매수 설정").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    ticker = ui.input("종목", value="TQQQ")
                    divisions = ui.number("분할 횟수", value=40, min=2, max=200)
                    initial = ui.number("초기 투자금", value=10_000, min=1)
                    fee = ui.number("거래비용(%)", value=0, min=0, max=10, step=0.01)
                quarter_stop = ui.checkbox("쿼터손절 근사 적용", value=False)
                getter = lambda: laoer(
                    str(ticker.value), _date(start.value), _date(end.value),
                    float(initial.value), int(divisions.value), float(fee.value), bool(quarter_stop.value),
                )
            elif feature_id == "real_estate_vs_stock":
                ui.label("주택 구입 조건").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    property_price = ui.number("주택 가격", value=500_000_000, min=1)
                    loan = ui.number("대출금", value=300_000_000, min=0)
                    annual_rate = ui.number("대출금리(%)", value=4.0, min=0, step=0.1)
                    repayment = ui.select(["원리금균등", "원금균등"], value="원리금균등", label="상환 방식")
                    growth = ui.number("주택 연 상승률(%)", value=2.0, step=0.1)
                ui.label("월세·주식 조건").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    ticker = ui.input("주식 종목", value="SPY")
                    initial_stock = ui.number("주식 초기 투자금", value=200_000_000, min=0)
                    monthly_rent = ui.number("월세", value=1_000_000, min=0)
                    rent_cycle = ui.number("월세 인상 주기(개월)", value=24, min=1)
                    rent_growth = ui.number("월세 인상률(%)", value=5.0, min=0, step=0.1)
                    years = ui.number("비교 기간(년)", value=10, min=1, max=30)
                getter = lambda: real_estate_vs_stock(
                    str(ticker.value), _date(start.value), _date(end.value), float(property_price.value),
                    float(loan.value), float(initial_stock.value), int(years.value), float(growth.value),
                    float(annual_rate.value), str(repayment.value), float(monthly_rent.value),
                    int(rent_cycle.value), float(rent_growth.value),
                )
            elif feature_id == "fear_greed_rebalance":
                ui.label("지표와 리밸런싱 조건").classes("section-title")
                indicators = {
                    "fear_and_greed_historical": "종합 공포·탐욕 지수",
                    "market_momentum_sp500": "시장 모멘텀(S&P500)",
                    "market_volatility_vix": "VIX 변동성",
                    "junk_bond_demand": "정크본드 수요",
                    "safe_haven_demand": "안전자산 수요",
                }
                with ui.row().classes("backtest-field-grid w-full"):
                    indicator = ui.select(indicators, value="fear_and_greed_historical", label="판정 지표")
                    direction = ui.select(["이하일 때", "이상일 때"], value="이하일 때", label="조건 방향")
                    threshold = ui.number("기준값", value=25, min=0, max=100)
                    frequency = ui.select(["매일", "매월", "매분기", "매년"], value="매월", label="리밸런싱 주기")
                ui.label("투자 포트폴리오").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    tickers = ui.input("종목(쉼표 구분)", value="SPY")
                    weights = ui.input("비중(쉼표 구분)", value="100")
                    initial = ui.number("초기 투자금", value=10_000, min=1)
                    contribution = ui.number("정기 적립금(0=미사용)", value=0, min=0)
                    contribution_frequency = ui.select(["매월", "매분기", "매년"], value="매월", label="적립 주기")
                getter = lambda: fear_greed(
                    _date(start.value), _date(end.value), _tickers(tickers.value), _numbers(weights.value),
                    float(threshold.value), float(initial.value), str(indicator.value), str(direction.value),
                    str(frequency.value), float(contribution.value), str(contribution_frequency.value),
                )
            else:
                ui.label("R-ONE 조회 조건").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    table_id = ui.input("통계표 ID", value="A_2024_00045")
                    cycle = ui.select({"MM": "월", "QY": "분기", "YY": "년"}, value="MM", label="주기")
                    region = ui.input("지역 ID", value="500001")
                    item = ui.input("항목 ID", value="100001")
                    value_kind = ui.select({"index": "지수", "rate": "변동률"}, value="index", label="값 유형")
                ui.label("RONE_API_KEY가 없으면 공식 표본 조회 모드로 실행됩니다.").classes("status-text")
                getter = lambda: reb_statistics(
                    str(table_id.value), str(cycle.value), str(region.value), str(item.value),
                    _date(start.value).year, _date(end.value).year, str(value_kind.value),
                )

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
