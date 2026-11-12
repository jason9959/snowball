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
    dollar_comparison,
    dollar_overview,
    dynamic_portfolio,
    fear_greed,
    fear_greed_comparison,
    fear_greed_overview,
    financial_company_search,
    financial_dashboard,
    financial_dashboard_company,
    inflation_audit,
    inflation_compass,
    laoer,
    moving_average,
    real_estate_vs_stock,
    reb_statistics,
    reb_statistics_multi,
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

            if feature_id == "dynamic_portfolio":
                from features.feature04_dynamic_portfolio.engine import MomentumConfig
                from features.feature04_dynamic_portfolio.page import (
                    ATTACK_MODES,
                    BENCHMARK_LABELS,
                    CANARY_APPLY_MODES,
                    DEFENSE_MODES,
                    MARKET_SIGNAL_SUPPORT,
                    RESEARCH_PRESETS,
                    SCORE_TYPES,
                )

                ui.label("전략 프리셋과 직접 설정").classes("section-title")
                preset = ui.select(
                    list(RESEARCH_PRESETS) + ["직접 설정"],
                    value="1. 분산투자 방어형", label="설정 불러오기",
                ).classes("w-full")
                with ui.row().classes("backtest-field-grid w-full"):
                    initial = ui.number("초기 투자금", value=10_000, min=1)
                    investment_type = ui.select(["거치식", "적립식"], value="거치식", label="투자 방식")
                    rebalance = ui.select(["월별", "분기", "반기", "연별"], value="월별", label="리밸런싱 주기")
                    benchmark = ui.select(BENCHMARK_LABELS, value="S&P500", label="벤치마크")
                    market = ui.select(["해외 ETF", "국내 ETF"], value="해외 ETF", label="ETF 종류")
                with ui.row().classes("backtest-field-grid w-full") as dynamic_contribution_fields:
                    contribution = ui.number("정기 적립금", value=1_000, min=1)
                    contribution_frequency = ui.select(["매월", "매분기", "매년"], value="매월", label="적립 주기")
                dynamic_contribution_fields.set_visibility(False)
                investment_type.on_value_change(
                    lambda _: dynamic_contribution_fields.set_visibility(investment_type.value == "적립식")
                )
                with ui.row().classes("backtest-field-grid w-full"):
                    use_cost = ui.checkbox("거래비용 반영", value=False)
                    cost_rate = ui.number("편도 거래비용률(%)", value=0.1, min=0, max=10, step=0.01)

                with ui.tabs().classes("dashboard-tabs w-full") as dynamic_tabs:
                    asset_tab = ui.tab("종목선택")
                    attack_tab = ui.tab("투자자산 선택")
                    defense_tab = ui.tab("방어자산 선택")
                    weight_tab = ui.tab("포트폴리오 비중")
                    risk_tab = ui.tab("위험 회피 옵션")
                with ui.tab_panels(dynamic_tabs, value=asset_tab).classes("dashboard-panels w-full"):
                    with ui.tab_panel(asset_tab):
                        attack_tickers = ui.input(
                            "투자자산 후보(쉼표 구분)",
                            value="QQQ, SOXX, XLE, XLK, XLV, EDV, GLD, PDBC",
                        ).classes("w-full")
                        defense_tickers = ui.input("방어자산 후보(쉼표 구분)", value="BIL, IEF, HYG").classes("w-full")
                    with ui.tab_panel(attack_tab):
                        with ui.row().classes("backtest-field-grid w-full"):
                            attack_mode = ui.select(ATTACK_MODES, value="상대 모멘텀", label="투자자산 선택 방식")
                            attack_count = ui.number("선택자산 수", value=4, min=1, max=20)
                            attack_kind = ui.select(["수익률", "이동평균", "모멘텀 스코어"], value="모멘텀 스코어", label="상대모멘텀 산출 기준")
                            attack_months = ui.number("기준 기간(개월)", value=12, min=1, max=60)
                            attack_score = ui.select(SCORE_TYPES, value="1-3-6-12개월 단순", label="모멘텀 스코어")
                            attack_absolute_kind = ui.select(["수익률", "이동평균", "모멘텀 스코어"], value="모멘텀 스코어", label="절대모멘텀 산출 기준")
                            attack_absolute_months = ui.number("절대 기준 기간(개월)", value=12, min=1, max=60)
                            attack_absolute_score = ui.select(SCORE_TYPES, value="1-3-6-12개월 단순", label="절대모멘텀 스코어")
                            attack_use_bil = ui.checkbox("절대모멘텀 기준을 BIL로 변경", value=False)
                    with ui.tab_panel(defense_tab):
                        with ui.row().classes("backtest-field-grid w-full"):
                            defense_enabled = ui.checkbox("방어자산 선택 사용", value=True)
                            defense_mode = ui.select(DEFENSE_MODES, value="상대 모멘텀", label="방어자산 선택 방식")
                            defense_count = ui.number("선택자산 수", value=1, min=1, max=20)
                            defense_kind = ui.select(["수익률", "이동평균", "모멘텀 스코어"], value="모멘텀 스코어", label="방어 모멘텀 기준")
                            defense_months = ui.number("기준 기간(개월)", value=12, min=1, max=60)
                            defense_score = ui.select(SCORE_TYPES, value="1-3-6-12개월 단순", label="방어 스코어")
                            defense_absolute_kind = ui.select(["수익률", "이동평균", "모멘텀 스코어"], value="모멘텀 스코어", label="방어 절대모멘텀 기준")
                            defense_absolute_months = ui.number("방어 절대 기준 기간(개월)", value=12, min=1, max=60)
                            defense_absolute_score = ui.select(SCORE_TYPES, value="1-3-6-12개월 단순", label="방어 절대 스코어")
                            defense_same_momentum = ui.checkbox("투자자산 선택기준과 동일 모멘텀 적용", value=False)
                            include_cash = ui.checkbox("현금을 방어 후보에 포함", value=False)
                            cash_replacement = ui.checkbox("현금 대신 BIL로 보유", value=False)
                            defense_use_bil = ui.checkbox("방어 절대모멘텀 기준을 BIL로 변경", value=False)
                    with ui.tab_panel(weight_tab):
                        with ui.row().classes("backtest-field-grid w-full"):
                            weight_mode = ui.select(
                                ["고정 범위", "마이너스 절대 모멘텀 수", "플러스 절대 모멘텀 자산 개수 비중 한도"],
                                value="마이너스 절대 모멘텀 수", label="비중 결정 방식",
                            )
                            fixed_defense = ui.number("방어자산 고정비중(%)", value=0, min=0, max=100)
                            negative_threshold = ui.number("전체 방어 전환 기준 자산 수", value=5, min=1, max=20)
                    with ui.tab_panel(risk_tab):
                        ui.label("옵션 A · 카나리아 ETF").classes("section-title")
                        with ui.row().classes("backtest-field-grid w-full"):
                            risk_enabled = ui.checkbox("카나리아 ETF 사용", value=True)
                            canary_tickers = ui.input("카나리아 ETF(쉼표 구분)", value="SPY, TIP")
                            risk_apply = ui.select(CANARY_APPLY_MODES, value="방어형", label="위험 신호 적용 방식")
                            risk_direction = ui.select(["마이너스(-)", "플러스(+)"], value="마이너스(-)", label="위험 신호 방향")
                            risk_momentum_type = ui.select(["절대 모멘텀", "상대 모멘텀"], value="절대 모멘텀", label="카나리아 기준")
                            risk_kind = ui.select(["수익률", "이동평균", "모멘텀 스코어"], value="이동평균", label="카나리아 산출 기준")
                            risk_months = ui.number("카나리아 기간(개월)", value=11, min=1, max=60)
                            risk_score = ui.select(SCORE_TYPES, value="1-3-6-12개월 단순", label="카나리아 스코어")
                            risk_same_momentum = ui.checkbox("투자자산 선택기준과 동일 모멘텀 적용", value=False)
                        ui.label("옵션 B · 시장 데이터").classes("section-title")
                        supported_signals = [label for label, supported in MARKET_SIGNAL_SUPPORT.items() if supported]
                        market_risk_enabled = ui.checkbox("시장 데이터 위험 회피 사용", value=True)
                        market_signals = ui.select(
                            supported_signals, value=["S&P500 배당수익률"],
                            label="시장 위험 신호", multiple=True,
                        ).props("use-chips").classes("w-full")
                        with ui.row().classes("backtest-field-grid w-full"):
                            market_combine = ui.select(["OR", "AND"], value="OR", label="신호 결합")
                            dividend_threshold = ui.number("S&P500 배당수익률 기준(%)", value=1.32, min=0, max=20, step=0.01)
                            spread_lag = ui.number("10년-3개월 반영 지연(개월)", value=7, min=0, max=24)
                            spread_active = ui.number("10년-3개월 유지 기간(개월)", value=8, min=1, max=24)
                            spread_threshold = ui.number("10년-3개월 역전 임계값(%p)", value=-0.5, min=-10, max=10, step=0.1)

                def momentum(kind: object, months: object, score: object) -> MomentumConfig:
                    return MomentumConfig(str(kind), int(months), str(score))

                def apply_dynamic_preset() -> None:
                    if preset.value == "직접 설정":
                        return
                    values = RESEARCH_PRESETS[str(preset.value)]
                    attack_tickers.value = ", ".join(values.get("attack_tickers", []))
                    defense_tickers.value = ", ".join(values.get("defense_tickers", []))
                    attack_mode.value = values.get("attack_mode", "상대 모멘텀")
                    attack_count.value = values.get("attack_select_count", 1)
                    attack_momentum = values.get("attack_relative", {})
                    attack_kind.value = attack_momentum.get("kind", "수익률")
                    attack_months.value = attack_momentum.get("months", 12)
                    attack_score.value = attack_momentum.get("score_type", SCORE_TYPES[2])
                    attack_absolute = values.get("attack_absolute", attack_momentum)
                    attack_absolute_kind.value = attack_absolute.get("kind", "수익률")
                    attack_absolute_months.value = attack_absolute.get("months", 12)
                    attack_absolute_score.value = attack_absolute.get("score_type", SCORE_TYPES[2])
                    defense_enabled.value = values.get("defense_enabled", False)
                    defense_mode.value = values.get("defense_mode", "전체 동일 비중")
                    defense_count.value = values.get("defense_select_count", 1)
                    defense_momentum = values.get("defense_relative", {})
                    defense_kind.value = defense_momentum.get("kind", "수익률")
                    defense_months.value = defense_momentum.get("months", 12)
                    defense_score.value = defense_momentum.get("score_type", SCORE_TYPES[2])
                    defense_absolute = values.get("defense_absolute", defense_momentum)
                    defense_absolute_kind.value = defense_absolute.get("kind", "수익률")
                    defense_absolute_months.value = defense_absolute.get("months", 12)
                    defense_absolute_score.value = defense_absolute.get("score_type", SCORE_TYPES[2])
                    defense_same_momentum.value = values.get("defense_same_momentum", False)
                    include_cash.value = values.get("include_cash", False)
                    cash_replacement.value = values.get("cash_replacement", False)
                    attack_use_bil.value = values.get("attack_use_bil", False)
                    defense_use_bil.value = values.get("defense_use_bil", False)
                    weight_mode.value = values.get("weight_mode", "고정 범위")
                    fixed_defense.value = float(values.get("fixed_defense_weight", 0)) * 100
                    negative_threshold.value = values.get("negative_threshold", 1)
                    risk_enabled.value = values.get("risk_enabled", False)
                    canary_tickers.value = ", ".join(values.get("canary_tickers", []))
                    risk_apply.value = values.get("risk_apply_mode", "중립형")
                    risk_direction.value = values.get("risk_direction", "마이너스(-)")
                    risk_momentum_type.value = values.get("risk_momentum_type", "절대 모멘텀")
                    risk_momentum = values.get("risk_custom_momentum", {})
                    risk_kind.value = risk_momentum.get("kind", "수익률")
                    risk_months.value = risk_momentum.get("months", 12)
                    risk_score.value = risk_momentum.get("score_type", SCORE_TYPES[2])
                    risk_same_momentum.value = values.get("risk_same_momentum", False)
                    market_risk_enabled.value = values.get("market_risk_enabled", False)
                    market_signals.value = values.get("market_risk_signals", [])
                    market_combine.value = values.get("market_risk_combine", "OR")
                    dividend_threshold.value = values.get("dividend_threshold", 1.32)
                    spread_lag.value = values.get("spread_lag", 7)
                    spread_active.value = values.get("spread_active", 8)
                    spread_threshold.value = values.get("spread_threshold", -0.5)

                preset.on_value_change(lambda _: apply_dynamic_preset())
                apply_dynamic_preset()

                def dynamic_result() -> ToolResult:
                    attack_relative = momentum(attack_kind.value, attack_months.value, attack_score.value)
                    attack_absolute = momentum(
                        attack_absolute_kind.value, attack_absolute_months.value, attack_absolute_score.value
                    )
                    defense_relative = momentum(defense_kind.value, defense_months.value, defense_score.value)
                    defense_absolute = momentum(
                        defense_absolute_kind.value, defense_absolute_months.value, defense_absolute_score.value
                    )
                    if defense_same_momentum.value:
                        defense_relative, defense_absolute = attack_relative, attack_absolute
                    risk_custom = momentum(risk_kind.value, risk_months.value, risk_score.value)
                    config = {
                        "start_date": _date(start.value), "end_date": _date(end.value),
                        "initial_investment": float(initial.value),
                        "investment_type": str(investment_type.value),
                        "rebalance_frequency": str(rebalance.value), "benchmark": str(benchmark.value),
                        "contribution_amount": float(contribution.value) if investment_type.value == "적립식" else 0.0,
                        "contribution_frequency": str(contribution_frequency.value),
                        "market": str(market.value), "currency": "KRW" if market.value == "국내 ETF" else "USD",
                        "attack_tickers": _tickers(attack_tickers.value), "defense_tickers": _tickers(defense_tickers.value),
                        "attack_mode": str(attack_mode.value), "attack_select_count": int(attack_count.value),
                        "attack_relative": attack_relative, "attack_absolute": attack_absolute,
                        "attack_use_bil": bool(attack_use_bil.value),
                        "defense_enabled": bool(defense_enabled.value), "defense_mode": str(defense_mode.value),
                        "defense_select_count": int(defense_count.value),
                        "defense_relative": defense_relative, "defense_absolute": defense_absolute,
                        "defense_use_bil": bool(defense_use_bil.value), "include_cash": bool(include_cash.value),
                        "cash_replacement_ticker": (("BIL_KRW" if market.value == "국내 ETF" else "BIL") if cash_replacement.value else None),
                        "weight_mode": str(weight_mode.value), "fixed_defense_weight": float(fixed_defense.value) / 100,
                        "negative_threshold": int(negative_threshold.value), "negative_same_momentum": True,
                        "negative_momentum": attack_relative, "positive_same_momentum": True,
                        "positive_momentum": attack_relative,
                        "risk_avoidance": {
                            "enabled": bool(risk_enabled.value), "tickers": _tickers(canary_tickers.value),
                            "apply_mode": str(risk_apply.value), "direction": str(risk_direction.value),
                            "momentum_type": str(risk_momentum_type.value), "same_momentum": bool(risk_same_momentum.value),
                            "custom_momentum": risk_custom,
                        },
                        "market_risk": {
                            "enabled": bool(market_risk_enabled.value), "combine": str(market_combine.value),
                            "signals": list(market_signals.value or []),
                            "spread_settings": {
                                "10년-2년": {"lag": 0, "active": 3, "threshold": 0.0},
                                "10년-3개월": {"lag": int(spread_lag.value), "active": int(spread_active.value), "threshold": float(spread_threshold.value)},
                            },
                            "dividend_threshold": float(dividend_threshold.value), "unemployment_months": 12,
                        },
                        "use_transaction_cost": bool(use_cost.value),
                        "transaction_cost_rate": float(cost_rate.value) / 100,
                    }
                    if not config["attack_tickers"]:
                        raise ValueError("투자자산 후보를 한 개 이상 입력해 주세요.")
                    return dynamic_portfolio(config)

                getter = dynamic_result
            elif feature_id == "return_comparison":
                ui.separator()
                ui.label("비교 종목 입력").classes("section-title")
                ui.label("최대 10개까지 입력할 수 있습니다. 티커 또는 종목 코드를 입력하세요.").classes("status-text")
                ticker_inputs = []
                defaults = [""] * 10
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
                    rebalance = ui.select(["매일", "매월", "매분기", "매년", "없음"], value="매일", label="리밸런싱 주기")
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
                defaults = [("", 0), ("", 0), ("", 0), ("", 0), ("", 0)]
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
                    step = ui.select([5, 10, 20], value=10, label="비중 간격(%)")
                    rebalance = ui.select(["매일", "매월", "매분기", "매년"], value="매일", label="리밸런싱 주기")
                    initial = ui.number("기준 투자금", value=10_000, min=1)
                fixed = ui.checkbox("한 종목 비중 고정", value=False)
                with ui.row().classes("backtest-field-grid w-full") as fixed_fields:
                    fixed_index = ui.number("고정할 종목 순번", value=1, min=1, max=3)
                    fixed_ratio = ui.number("고정 비중(%)", value=40, min=0, max=100, step=5)
                fixed_fields.set_visibility(False)

                def toggle_fixed() -> None:
                    fixed_fields.set_visibility(bool(fixed.value))
                    step.set_enabled(not bool(fixed.value))

                fixed.on_value_change(lambda _: toggle_fixed())

                def allocation_result() -> ToolResult:
                    selected = _tickers(tickers.value)
                    if fixed.value and len(selected) != 3:
                        raise ValueError("고정 비중은 종목 3개일 때 사용할 수 있습니다.")
                    ratio = int(fixed_ratio.value)
                    if fixed.value and ratio % 5:
                        raise ValueError("고정 비중은 5% 단위로 입력해 주세요.")
                    return allocation_sweep(
                        selected, _date(start.value), _date(end.value), 5 if fixed.value else int(step.value),
                        int(fixed_index.value) - 1 if fixed.value else None,
                        ratio if fixed.value else None,
                        str(rebalance.value), float(initial.value),
                    )

                getter = allocation_result
            elif feature_id in {"monte_carlo_normal", "monte_carlo_bootstrap"}:
                ui.label("시뮬레이션 설정").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    ticker = ui.input("종목", value="QQQ")
                    years = ui.number("예측 기간(년)", value=10, min=0.5, max=50, step=0.5)
                    simulations = ui.number("시뮬레이션 수", value=5000, min=100, max=10000)
                    initial = ui.number("초기 투자금", value=10_000, min=1)
                recurring = ui.checkbox("적립식 포함", value=False)
                with ui.row().classes("backtest-field-grid w-full") as recurring_fields:
                    recurring_amount = ui.number("정기 적립금", value=1000, min=0)
                    contribution_frequency = ui.select(["매월", "매분기", "매년"], value="매월", label="적립 주기")
                recurring_fields.set_visibility(False)
                recurring.on_value_change(lambda _: recurring_fields.set_visibility(bool(recurring.value)))
                method = "normal" if feature_id.endswith("normal") else "bootstrap"
                getter = lambda: monte_carlo(
                    str(ticker.value), _date(start.value), _date(end.value), float(years.value),
                    int(simulations.value), float(initial.value), method,
                    float(recurring_amount.value) if recurring.value else 0.0,
                    str(contribution_frequency.value),
                )
            elif feature_id == "inflation_compass":
                inflation_mode = ui.select(
                    ["전략 백테스트", "검증 결과 살펴보기"],
                    value="전략 백테스트", label="분석 화면",
                ).classes("w-full")
                with ui.expansion("전략 이해하기", icon="menu_book").classes("strategy-expansion w-full"):
                    ui.label("월말마다 성장과 기대인플레이션 신호로 다음 달 보유 자산을 결정합니다.")
                    rules = [
                        {"성장": "상승", "인플레이션": "ON", "보유 자산": "XLE · 에너지 100%"},
                        {"성장": "상승", "인플레이션": "OFF", "보유 자산": "XLK · 기술 100%"},
                        {"성장": "하락", "인플레이션": "ON", "보유 자산": "XLU · 유틸리티 100%"},
                        {"성장": "하락", "인플레이션": "OFF", "보유 자산": "XLP 50% + IEF 50%"},
                    ]
                    ui.table(
                        columns=[{"name": key, "label": key, "field": key} for key in rules[0]],
                        rows=rules,
                    ).classes("w-full")
                with ui.column().classes("w-full gap-3") as inflation_backtest:
                    ui.label("나침반 판정 기준").classes("section-title")
                    with ui.row().classes("backtest-field-grid w-full"):
                        initial = ui.number("초기 투자금", value=10_000, min=1)
                        preset = ui.select(
                            ["원문 재현 가정", "보수적 검증", "직접 설정"],
                            value="원문 재현 가정", label="거래 가정",
                        )
                        threshold = ui.number("기대인플레이션 기준(%)", value=2.0, step=0.1)
                        growth_window = ui.number("성장 이동평균(거래일)", value=200, min=20)
                        be_window = ui.number("기대물가 모멘텀 기간", value=60, min=1)
                        asset_window = ui.number("자산 모멘텀 기간", value=60, min=1)
                    with ui.expansion("시차 및 거래비용", icon="tune").classes("strategy-expansion w-full"):
                        with ui.row().classes("backtest-field-grid w-full"):
                            fred_lag = ui.number("FRED 지연(거래일)", value=0, min=0)
                            execution_lag = ui.number("체결 지연(거래일)", value=0, min=0, max=2)
                            cost = ui.number("거래비용(bps)", value=0, min=0)
                with ui.column().classes("w-full gap-3") as inflation_audit_fields:
                    ui.label("거래 지연·비용·신호 구성별 사전 생성 검증 결과를 비교합니다.").classes("status-text")
                    audit_frequency = ui.select(["일별", "월별"], value="일별", label="위험 측정 주기")

                def update_inflation_mode() -> None:
                    inflation_backtest.set_visibility(inflation_mode.value == "전략 백테스트")
                    inflation_audit_fields.set_visibility(inflation_mode.value == "검증 결과 살펴보기")

                inflation_mode.on_value_change(lambda _: update_inflation_mode())
                update_inflation_mode()

                def inflation_result() -> ToolResult:
                    if inflation_mode.value == "검증 결과 살펴보기":
                        return inflation_audit(str(audit_frequency.value))
                    return inflation_compass(
                        _date(start.value), _date(end.value), float(initial.value), float(threshold.value),
                        int(growth_window.value), int(be_window.value), int(asset_window.value),
                        int(fred_lag.value), int(execution_lag.value), float(cost.value), str(preset.value),
                    )

                getter = inflation_result
            elif feature_id == "dollar_compass":
                mode = ui.select(
                    ["오늘의 달러 투자 환경", "거치식·적립식 계산", "지표 비교 차트"],
                    value="거치식·적립식 계산", label="분석 화면",
                ).classes("w-full")
                with ui.column().classes("w-full gap-3") as dollar_backtest:
                    ui.label("원화·달러 전환 설정").classes("section-title")
                    with ui.row().classes("backtest-field-grid w-full"):
                        initial = ui.number("거치 금액(KRW)", value=10_000_000, min=100_000)
                        switch = ui.number("신호 발생 시 환전 비율(%)", value=10, min=1, max=50)
                        fee = ui.number("환전 수수료(%)", value=0.15, min=0, max=1, step=0.01)
                    recurring = ui.checkbox("월 적립식 사용", value=False)
                    monthly = ui.number("월 적립금", value=500_000, min=0)
                    monthly.set_visibility(False)
                    recurring.on_value_change(lambda _: monthly.set_visibility(bool(recurring.value)))
                    ui.label("점수 0~1: 달러→원화 · 2: Stay · 3~4: 원화→달러").classes("status-text")
                with ui.column().classes("w-full gap-3") as dollar_compare:
                    ui.label("두 지표를 시작값 100으로 비교합니다.").classes("status-text")
                    comparison_labels = [
                        "달러인덱스", "원/달러 환율", "엔/달러 환율", "위안/달러 환율",
                        "스위스프랑/달러 환율", "캐나다달러/달러 환율", "유로/달러 환율",
                        "파운드/달러 환율", "호주달러/달러 환율",
                    ]
                    with ui.row().classes("backtest-field-grid w-full"):
                        first_indicator = ui.select(comparison_labels, value=comparison_labels[0], label="첫 번째 지표")
                        second_indicator = ui.select(comparison_labels, value=comparison_labels[1], label="두 번째 지표")
                with ui.column().classes("w-full gap-3") as dollar_environment:
                    ui.label("최신 네 가지 환율 조건과 최근 1년 흐름을 확인합니다.").classes("status-text")

                def update_dollar_mode() -> None:
                    dollar_backtest.set_visibility(mode.value == "거치식·적립식 계산")
                    dollar_compare.set_visibility(mode.value == "지표 비교 차트")
                    dollar_environment.set_visibility(mode.value == "오늘의 달러 투자 환경")

                mode.on_value_change(lambda _: update_dollar_mode())
                update_dollar_mode()

                def dollar_result() -> ToolResult:
                    if mode.value == "오늘의 달러 투자 환경":
                        return dollar_overview()
                    if mode.value == "지표 비교 차트":
                        return dollar_comparison(
                            _date(start.value), _date(end.value), str(first_indicator.value), str(second_indicator.value)
                        )
                    return dollar_compass(
                        _date(start.value), _date(end.value), float(initial.value),
                        float(switch.value), float(fee.value), float(monthly.value) if recurring.value else 0.0,
                    )

                getter = dollar_result
            elif feature_id == "financial_dashboard":
                ui.label("공식 공시 조회").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    query = ui.input("기업명 또는 종목코드", value="", placeholder="예: 삼성전자, 005930, Apple, AAPL")
                    frequency = ui.select(["연도별", "분기별"], value="연도별", label="조회 기준")
                    start_year = ui.number("시작 연도", value=dt.date.today().year - 3, min=2015, max=dt.date.today().year)
                    end_year = ui.number("종료 연도", value=dt.date.today().year - 1, min=2015, max=dt.date.today().year)
                with ui.row().classes("backtest-field-grid w-full") as quarter_fields:
                    start_quarter = ui.select([1, 2, 3, 4], value=1, label="시작 분기")
                    end_quarter = ui.select([1, 2, 3, 4], value=4, label="종료 분기")
                quarter_fields.set_visibility(False)
                frequency.on_value_change(lambda _: quarter_fields.set_visibility(frequency.value == "분기별"))
                company_matches: dict[str, object] = {}
                company = ui.select([], label="검색 결과").classes("w-full")
                company.disable()
                source_label = ui.label("조건을 입력한 뒤 기업 검색을 눌러 주세요.").classes("status-text")

                async def search_financial_company() -> None:
                    normalized = str(query.value or "").strip()
                    if not normalized:
                        ui.notify("기업명 또는 종목코드를 입력해 주세요.", type="warning")
                        return
                    search_button.disable()
                    source_label.set_text("DART·SEC 기업 목록을 검색하고 있습니다...")
                    try:
                        found = await run.io_bound(financial_company_search, normalized, 20, "자동")
                        company_matches.clear()
                        company_matches.update({item.label: item for item in found})
                        company.set_options(list(company_matches))
                        if found:
                            company.value = found[0].label
                            company.enable()
                            source_label.set_text(f"{len(found)}개 기업을 찾았습니다. 기업을 선택한 뒤 조회를 실행하세요.")
                        else:
                            company.value = None
                            company.disable()
                            source_label.set_text("일치하는 기업을 찾지 못했습니다.")
                    except Exception as error:
                        company.disable()
                        source_label.set_text(f"기업 목록을 불러오지 못했습니다: {error}")
                    finally:
                        search_button.enable()

                search_button = ui.button("기업 검색", icon="search", on_click=search_financial_company).props("outline")
                ui.label("DART는 DART_API_KEY, SEC는 SEC_USER_AGENT 환경변수를 사용합니다.").classes("status-text")

                def financial_result() -> ToolResult:
                    if company.value not in company_matches:
                        raise ValueError("기업 검색 후 결과에서 기업을 선택해 주세요.")
                    return financial_dashboard_company(
                        company_matches[str(company.value)], int(start_year.value), int(end_year.value),
                        str(frequency.value) == "분기별", int(start_quarter.value), int(end_quarter.value),
                    )

                getter = financial_result
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
                    divisions = ui.number("분할 횟수", value=40, min=10, max=100)
                    initial = ui.number("초기 투자금", value=40_000, min=100)
                    fee = ui.number("거래비용(%)", value=0, min=0, max=2, step=0.01)
                quarter_stop = ui.checkbox("쿼터손절 모드 포함(근사 구현)", value=False)
                ui.label("V2.2의 T값·별%·전/후반 LOC 매수·부분매도·+10% 지정가를 일봉 OHLC로 근사합니다.").classes("status-text")
                getter = lambda: laoer(
                    str(ticker.value), _date(start.value), _date(end.value),
                    float(initial.value), int(divisions.value), float(fee.value), bool(quarter_stop.value),
                )
            elif feature_id == "real_estate_vs_stock":
                ui.label("주택 구입 조건").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    property_price = ui.number("매수가격", value=500_000_000, min=1)
                    loan = ui.number("대출금", value=300_000_000, min=0)
                    annual_rate = ui.number("대출이자율(연 %)", value=4.0, min=0, step=0.1)
                    loan_years = ui.number("대출기간(년)", value=30, min=1, max=50)
                    repayment = ui.select(["원리금균등상환", "원금균등상환"], value="원리금균등상환", label="대출상환방식")
                    growth = ui.number("부동산 연 상승률(%)", value=2.0, step=0.1)
                ui.label("월세·주식 조건").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    monthly_rent = ui.number("월세", value=1_000_000, min=0)
                    deposit = ui.number("보증금", value=50_000_000, min=0)
                    rent_cycle = ui.select(
                        {0: "사용 안 함", 12: "1년", 24: "2년", 36: "3년", 48: "4년"},
                        value=24, label="월세 인상 주기",
                    )
                    rent_growth = ui.number("월세 상승률(연 %)", value=5.0, min=0, step=0.1)
                    ticker = ui.select(["VOO", "QQQ", "QLD"], value="VOO", label="주식 종목")
                ui.label("주가 경로 설정").classes("section-title")
                with ui.row().classes("backtest-field-grid w-full"):
                    path_method = ui.select(
                        ["기존 주가 이용", "Bootstrap", "정규분포"],
                        value="기존 주가 이용", label="주가 경로 생성 방식",
                    )
                    simulations = ui.select([500, 1000, 1500, 2000, 2500, 3000], value=500, label="몬테카를로 경로 수")
                    years = ui.number("백테스팅 기간(년)", value=10, min=1, max=50)
                simulations.set_enabled(False)
                path_method.on_value_change(lambda _: simulations.set_enabled(path_method.value != "기존 주가 이용"))
                with ui.card().classes("editor-section w-full"):
                    property_initial_label = ui.label("")
                    stock_initial_label = ui.label("")
                    payment_label = ui.label("")

                    def update_real_estate_preview() -> None:
                        from features.feature13_real_estate_vs_stock.calculator import mortgage_payment
                        property_initial = max(float(property_price.value or 0) - float(loan.value or 0), 0.0)
                        stock_initial = max(property_initial - float(deposit.value or 0), 0.0)
                        payment = mortgage_payment(
                            float(loan.value or 0), float(annual_rate.value or 0),
                            int(loan_years.value or 1), str(repayment.value),
                        )
                        property_initial_label.set_text(f"초기자금: {property_initial:,.0f}원")
                        stock_initial_label.set_text(f"주식 초기 투자금: {stock_initial:,.0f}원")
                        payment_label.set_text(f"첫 달 원리금: {payment:,.0f}원")

                    for field in (property_price, loan, annual_rate, loan_years, repayment, deposit):
                        field.on_value_change(lambda _, update=update_real_estate_preview: update())
                    update_real_estate_preview()
                getter = lambda: real_estate_vs_stock(
                    str(ticker.value), _date(start.value), _date(end.value), float(property_price.value),
                    float(loan.value), 0.0, int(years.value), float(growth.value),
                    float(annual_rate.value), str(repayment.value), float(monthly_rent.value),
                    int(rent_cycle.value), float(rent_growth.value), int(loan_years.value),
                    float(deposit.value), str(path_method.value), int(simulations.value),
                )
            elif feature_id == "fear_greed_rebalance":
                indicators = {
                    "fear_and_greed_historical": "종합 공포·탐욕 지수",
                    "market_momentum_sp500": "시장 모멘텀(S&P500)",
                    "market_volatility_vix": "VIX 변동성",
                    "junk_bond_demand": "정크본드 수요",
                    "safe_haven_demand": "안전자산 수요",
                }
                fg_mode = ui.select(
                    ["공포·탐욕 지표 개요", "리밸런싱 백테스트", "지표 비교 차트"],
                    value="리밸런싱 백테스트", label="분석 화면",
                ).classes("w-full")
                with ui.column().classes("w-full gap-3") as fg_backtest:
                    ui.label("지표와 리밸런싱 조건").classes("section-title")
                    with ui.row().classes("backtest-field-grid w-full"):
                        indicator = ui.select(indicators, value="fear_and_greed_historical", label="리밸런싱에 사용할 지표")
                        direction = ui.select(["이하일 때", "이상일 때"], value="이하일 때", label="조건")
                        threshold = ui.number("기준값", value=25, min=0, max=100)
                        frequency = ui.select(["매일", "매월", "매분기", "매년"], value="매월", label="리밸런싱 확인 주기")
                        initial = ui.number("초기 투자금", value=10_000, min=1)
                        investment_type = ui.select(["거치식", "적립식"], value="거치식", label="투자 방식")
                    with ui.row().classes("backtest-field-grid w-full") as fg_contribution_fields:
                        contribution = ui.number("정기 적립금", value=1000, min=1)
                        contribution_frequency = ui.select(["매월", "매분기", "매년"], value="매월", label="적립 주기")
                    fg_contribution_fields.set_visibility(False)
                    investment_type.on_value_change(
                        lambda _: fg_contribution_fields.set_visibility(investment_type.value == "적립식")
                    )
                    ui.label("투자 포트폴리오").classes("section-title")
                    fg_tickers = []
                    fg_weights = []
                    for index, (default_ticker, default_weight) in enumerate(
                        [("SPY", 100), ("", 0), ("", 0), ("", 0), ("", 0)], start=1
                    ):
                        with ui.row().classes("backtest-field-grid w-full"):
                            fg_tickers.append(ui.input(f"자산 {index}", value=default_ticker, placeholder="예: SPY 또는 QQQ"))
                            fg_weights.append(ui.number(f"비중 {index}(%)", value=default_weight, min=0, max=100))
                with ui.column().classes("w-full gap-3") as fg_compare:
                    ui.label("두 공포·탐욕 세부 지표를 비교합니다.").classes("status-text")
                    with ui.row().classes("backtest-field-grid w-full"):
                        fg_first = ui.select(indicators, value="fear_and_greed_historical", label="첫 번째 지표")
                        fg_second = ui.select(indicators, value="market_momentum_sp500", label="두 번째 지표")
                with ui.column().classes("w-full gap-3") as fg_overview:
                    ui.label("선택한 기간의 종합 지표와 세부 지표 흐름을 확인합니다.").classes("status-text")

                def update_fg_mode() -> None:
                    fg_backtest.set_visibility(fg_mode.value == "리밸런싱 백테스트")
                    fg_compare.set_visibility(fg_mode.value == "지표 비교 차트")
                    fg_overview.set_visibility(fg_mode.value == "공포·탐욕 지표 개요")

                fg_mode.on_value_change(lambda _: update_fg_mode())
                update_fg_mode()

                def fear_greed_result() -> ToolResult:
                    if fg_mode.value == "공포·탐욕 지표 개요":
                        return fear_greed_overview(_date(start.value), _date(end.value))
                    if fg_mode.value == "지표 비교 차트":
                        return fear_greed_comparison(
                            _date(start.value), _date(end.value), str(fg_first.value), str(fg_second.value)
                        )
                    pairs = [
                        (str(ticker.value or "").strip().upper(), float(weight.value or 0))
                        for ticker, weight in zip(fg_tickers, fg_weights)
                        if str(ticker.value or "").strip() and float(weight.value or 0) > 0
                    ]
                    return fear_greed(
                        _date(start.value), _date(end.value),
                        [ticker for ticker, _ in pairs], [weight for _, weight in pairs],
                        float(threshold.value), float(initial.value), str(indicator.value), str(direction.value),
                        str(frequency.value),
                        float(contribution.value) if investment_type.value == "적립식" else 0.0,
                        str(contribution_frequency.value),
                    )

                getter = fear_greed_result
            else:
                ui.label("통계 선택").classes("section-title")
                reb_presets = {
                    "전국 아파트 매매가격지수": ("A_2024_00045", "MM", "100001", "index"),
                    "전국 월별 지가변동률": ("A_2024_00903", "MM", "100001", "rate"),
                    "직접 코드 입력": ("A_2024_00045", "MM", "100001", "index"),
                }
                reb_preset = ui.select(list(reb_presets), value="전국 아파트 매매가격지수", label="분석할 통계")
                with ui.row().classes("backtest-field-grid w-full"):
                    table_id = ui.input("통계표 코드", value="A_2024_00045")
                    cycle = ui.select({"MM": "월", "QY": "분기", "YY": "연", "WK": "주"}, value="MM", label="수록 주기")
                    item = ui.input("항목 코드", value="100001")
                    value_kind = ui.select({"index": "지수", "rate": "기간 변동률"}, value="index", label="값의 종류")
                region_map = {"전국": "500001", "수도권": "500002", "지방권": "500003", "5대광역시": "500004", "4대광역시": "500005"}
                regions = ui.select(
                    region_map, value=["500001", "500002", "500003"],
                    label="지역 선택", multiple=True,
                ).props("use-chips").classes("w-full")
                with ui.row().classes("backtest-field-grid w-full"):
                    reb_start_year = ui.number("시작 연도", value=max(2006, dt.date.today().year - 10), min=1900, max=dt.date.today().year)
                    reb_end_year = ui.number("종료 연도", value=dt.date.today().year, min=1900, max=dt.date.today().year)

                def apply_reb_preset() -> None:
                    preset_values = reb_presets[str(reb_preset.value)]
                    table_id.value, cycle.value, item.value, value_kind.value = preset_values
                    custom = reb_preset.value == "직접 코드 입력"
                    for field in (table_id, cycle, item, value_kind):
                        field.set_enabled(custom)

                reb_preset.on_value_change(lambda _: apply_reb_preset())
                apply_reb_preset()
                ui.label("RONE_API_KEY가 없으면 공식 표본 조회 모드로 실행됩니다.").classes("status-text")
                getter = lambda: reb_statistics_multi(
                    str(table_id.value), str(cycle.value), list(regions.value or []), str(item.value),
                    int(reb_start_year.value), int(reb_end_year.value), str(value_kind.value),
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
