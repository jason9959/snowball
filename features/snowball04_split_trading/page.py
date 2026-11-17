"""NiceGUI page for the single-asset split-trading backtest."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nicegui import app, run, ui

from common.snowball.layout import shell
from .data import load_adjusted_close, search_symbols
from .engine import SplitTradingConfig, run_split_backtest
from .optimization import CRITERIA, run_ratio_sweep, run_tranche_sweep
from .sheets import connect_spreadsheet, export_result


FORM_STORAGE_KEY = "split_trading_form_v1"


def _money(value: float, country: str) -> str:
    return f"₩{value:,.0f}" if country == "한국" else f"${value:,.2f}"


def _actual_money(raw_value: float, country: str) -> float:
    """Interpret the input directly in the selected market currency."""
    return round(float(raw_value), 0 if country == "한국" else 2)


def _number(value: float, country: str) -> str:
    """Format market-currency prices and amounts without changing calculations."""
    return f"{float(value):,.0f}" if country == "한국" else f"{float(value):,.2f}"


def _sweep_values(minimum: float, maximum: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("최적화 간격은 0보다 커야 합니다.")
    if minimum > maximum:
        raise ValueError("최적화 최소값은 최대값보다 클 수 없습니다.")
    count = int((maximum - minimum) / step + 1e-9) + 1
    return [round(minimum + index * step, 10) for index in range(count)]


def _table_columns(names: list[str]) -> list[dict[str, object]]:
    return [{"name": name, "label": name, "field": name, "align": "left", "sortable": True}
            for name in names]


def _render_result(result: dict, country: str, symbol_label: str) -> None:
    currency = "KRW" if country == "한국" else "USD"
    chart_digits = 0 if country == "한국" else 2
    ui.label(f"{symbol_label} 결과").classes("section-title")
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in result["split_nav"].index]
    buy_points, sell_points = [], []
    for row in result["trades"].to_dict("records"):
        day = pd.Timestamp(row["거래일"])
        if day not in result["split_nav"].index:
            continue
        point = {
            "value": [day.strftime("%Y-%m-%d"), round(float(result["split_nav"].loc[day]), chart_digits)],
            "name": (
                f"{row['당일 순번']}번 · {row['차수']}차 · "
                f"{_number(row['시뮬레이션 체결가'], country)}"
            ),
        }
        (buy_points if row["매수/매도"] == "매수" else sell_points).append(point)
    ui.echart({
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "grid": {"left": 72, "right": 24, "top": 24, "bottom": 64},
        "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
        "yAxis": {"type": "value", "name": currency, "scale": True},
        "dataZoom": [{"type": "inside"}],
        "series": [
            {"name": "전액 투자", "type": "line", "showSymbol": False,
             "data": result["full_nav"].round(chart_digits).tolist(), "lineStyle": {"width": 2}},
            {"name": "분할매매", "type": "line", "showSymbol": False,
             "data": result["split_nav"].round(chart_digits).tolist(), "lineStyle": {"width": 3}},
            {"name": "매수", "type": "scatter", "data": buy_points, "symbol": "triangle",
             "symbolSize": 11, "itemStyle": {"color": "#d95c5c"}},
            {"name": "매도", "type": "scatter", "data": sell_points, "symbol": "triangle",
             "symbolRotate": 180, "symbolSize": 11, "itemStyle": {"color": "#3f9b65"}},
        ],
    }).classes("dashboard-chart w-full")
    cards = (
        ("전액 투자 최종 평가액", _money(result["full_final"], country)),
        ("전액 투자 총수익률", f"{result['full_return']:.2%}"),
        ("분할매매 최종 평가액", _money(result["split_final"], country)),
        ("분할매매 총수익률", f"{result['split_return']:.2%}"),
        ("수익률 차이", f"{result['return_difference'] * 100:+.2f}%p"),
        ("최종 현금잔액", _money(result["final_cash"], country)),
        ("최종 보유분 평가액", _money(result["final_holdings_value"], country)),
        ("동시 보유 가능 차수", f"{result['max_slots']}차"),
        ("전액 투자 MDD", f"{result['full_max_drawdown']:.2%}"),
        ("분할매매 MDD", f"{result['split_max_drawdown']:.2%}"),
        ("분할매매 현재 낙폭", f"{result['split_current_drawdown']:.2%}"),
        (
            "분할매매 수익률/MDD",
            "-" if result["split_return_mdd"] is None else f"{result['split_return_mdd']:.2f}",
        ),
    )
    with ui.grid().classes("result-metric-grid w-full gap-3"):
        for label, value in cards:
            with ui.card().classes("metric-card w-full"):
                with ui.row().classes("metric-row w-full items-center justify-between no-wrap"):
                    ui.label(label).classes("metric-label")
                    ui.label(value).classes("metric-value")
    recovery = result["split_mdd_recovery_date"] or "회복 전"
    ui.label(
        "분할매매 최대 낙폭 구간 · "
        f"{result['split_mdd_peak_date']} → {result['split_mdd_trough_date']} · "
        f"회복 {recovery} · {result['split_mdd_duration_days']}일"
    ).classes("text-sm text-grey-6")
    ui.label("체결 내역").classes("section-title")
    display = result["trades"].copy()
    for column in ("시뮬레이션 체결가", "체결금액", "실현손익", "체결 후 현금잔액"):
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else _number(float(value), country)
        )
    quantity_digits = 0 if result["quantity_mode"] == "whole" else 6
    display["수량"] = display["수량"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):,.{quantity_digits}f}"
    )
    display["실현수익률"] = display["실현수익률"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):.2%}"
    )
    ui.table(
        columns=_table_columns(list(display.columns)),
        rows=display.fillna("").to_dict("records"), pagination={"rowsPerPage": 15},
    ).props("flat bordered dense").classes("w-full account-table")


def _optimization_table(frame: pd.DataFrame, country: str) -> pd.DataFrame:
    display = frame.copy()
    for column in ("최종 평가액", "최종 현금", "회당 매수금액"):
        if column in display:
            display[column] = display[column].map(lambda value: _number(value, country))
    for column in ("총수익률", "MDD", "현재 낙폭", "매수 하락률", "매도 상승률"):
        if column in display:
            display[column] = display[column].map(lambda value: f"{float(value):.2%}")
    if "수익률/MDD" in display:
        display["수익률/MDD"] = display["수익률/MDD"].map(lambda value: f"{float(value):.2f}")
    return display


def _top_strategy_chart(optimization: dict, country: str) -> None:
    top = optimization["runs"][:5]
    index = top[0]["result"]["split_nav"].index
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in index]
    digits = 0 if country == "한국" else 2
    series = []
    for rank, item in enumerate(top, start=1):
        parameters = item["parameters"]
        if "매수 하락률" in parameters:
            name = (
                f"{rank}위 · 매수 {parameters['매수 하락률']:.0%} / "
                f"매도 {parameters['매도 상승률']:.0%}"
            )
        else:
            name = f"{rank}위 · {parameters['분할 조건']}"
        series.append({
            "name": name, "type": "line", "showSymbol": False,
            "data": item["result"]["split_nav"].round(digits).tolist(),
            "lineStyle": {"width": 3 if rank == 1 else 1.5},
        })
    ui.label("상위 5개 전략 평가액").classes("section-title")
    ui.echart({
        "tooltip": {"trigger": "axis"}, "legend": {"bottom": 0, "type": "scroll"},
        "grid": {"left": 72, "right": 24, "top": 24, "bottom": 74},
        "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
        "yAxis": {"type": "value", "scale": True},
        "dataZoom": [{"type": "inside"}], "series": series,
    }).classes("dashboard-chart w-full")


def _ratio_heatmap(optimization: dict) -> None:
    frame = optimization["table"]
    buys = sorted(frame["매수 하락률"].unique())
    sells = sorted(frame["매도 상승률"].unique())
    buy_pos = {value: index for index, value in enumerate(buys)}
    sell_pos = {value: index for index, value in enumerate(sells)}
    data = [
        [buy_pos[row["매수 하락률"]], sell_pos[row["매도 상승률"]], round(row["총수익률"] * 100, 2)]
        for row in frame.to_dict("records")
    ]
    values = [item[2] for item in data]
    visual_min, visual_max = min(values), max(values)
    if visual_min == visual_max:
        visual_min -= 1
        visual_max += 1
    ui.label("매수·매도 비율별 총수익률").classes("section-title")
    ui.echart({
        "tooltip": {"position": "top"},
        "grid": {"left": 64, "right": 24, "top": 24, "bottom": 70},
        "xAxis": {"type": "category", "name": "매수 하락률", "data": [f"{v:.0%}" for v in buys]},
        "yAxis": {"type": "category", "name": "매도 상승률", "data": [f"{v:.0%}" for v in sells]},
        "visualMap": {
            "min": visual_min, "max": visual_max, "calculable": True,
            "orient": "horizontal", "left": "center", "bottom": 0,
            "inRange": {"color": ["#d95c5c", "#fff7f7", "#f3fbf5", "#3f9b65"]},
        },
        "series": [{
            "type": "heatmap", "data": data,
            "label": {"show": True, "formatter": "{@[2]}%"},
        }],
    }).classes("monthly-heatmap w-full")


def _allocation_chart(optimization: dict) -> None:
    frame = optimization["table"].sort_values("동시 보유 차수")
    labels = frame["분할 조건"].astype(str).tolist()
    ui.label("분할 규모별 수익률과 MDD").classes("section-title")
    ui.echart({
        "tooltip": {"trigger": "axis"}, "legend": {"bottom": 0},
        "grid": {"left": 58, "right": 24, "top": 24, "bottom": 70},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"rotate": 30}},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "{value}%"}},
        "series": [
            {"name": "총수익률", "type": "line", "data": (frame["총수익률"] * 100).round(2).tolist()},
            {"name": "MDD", "type": "line", "data": (frame["MDD"] * 100).round(2).tolist()},
        ],
    }).classes("monthly-distribution w-full")


def _render_optimization(
    optimization: dict, country: str, symbol_label: str, mode: str,
) -> None:
    parameters = optimization["best_parameters"]
    ui.label(f"{symbol_label} 최적화 결과").classes("section-title")
    ui.label(
        f"최적화 기준 · {CRITERIA[optimization['criterion']]} · 최적 조건 · "
        + " / ".join(f"{key} {value:.2%}" if isinstance(value, float) and "률" in key
                     else f"{key} {value}" for key, value in parameters.items())
    ).classes("page-subtitle")
    if mode == "ratio":
        _ratio_heatmap(optimization)
    else:
        _allocation_chart(optimization)
    _top_strategy_chart(optimization, country)
    ui.label("전체 조합 순위").classes("section-title")
    display = _optimization_table(optimization["table"], country)
    ui.table(
        columns=_table_columns(list(display.columns)),
        rows=display.fillna("").to_dict("records"), pagination={"rowsPerPage": 15},
    ).props("flat bordered dense").classes("w-full account-table")
    with ui.expansion("1위 조건 상세 결과", icon="emoji_events", value=True).classes(
        "strategy-expansion w-full"
    ):
        _render_result(optimization["best"], country, symbol_label)


@ui.page("/split-trading")
def split_trading_page() -> None:
    shell()
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    default_start = (pd.Timestamp(today) - pd.DateOffset(years=3)).date()
    stored_value = app.storage.user.get(FORM_STORAGE_KEY, {})
    saved = stored_value if isinstance(stored_value, dict) else {}
    saved_country = str(saved.get("country", "한국"))
    if saved_country not in {"한국", "미국"}:
        saved_country = "한국"
    saved_symbol = str(saved.get("symbol", "")).strip()
    saved_symbol_label = str(saved.get("symbol_label", saved_symbol)).strip()
    saved_period = str(saved.get("period", "3"))
    if saved_period not in {*map(str, range(1, 11)), "custom"}:
        saved_period = "3"
    saved_test_mode = str(saved.get("test_mode", "single"))
    if saved_test_mode not in {"single", "ratio", "allocation"}:
        saved_test_mode = "single"
    saved_criterion = str(saved.get("criterion", "return"))
    if saved_criterion not in CRITERIA:
        saved_criterion = "return"
    state: dict[str, object] = {
        "result": None, "conditions": None, "search_version": 0, "date_sync": False,
        "restoring": True,
    }

    with ui.column().classes("page-container w-full gap-4"):
        with ui.row().classes("items-center gap-2"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("round flat")
            with ui.column().classes("gap-0"):
                ui.label("분할매매 백테스트").classes("page-title")
                ui.label("1차는 보유하고 2차 이후 매수·매도를 반복해 전액 투자와 비교합니다.").classes("page-subtitle")

        with ui.card().classes("backtest-controls w-full"):
            test_mode = ui.select(
                {
                    "single": "단일 백테스트",
                    "ratio": "매수·매도 비율 최적화",
                    "allocation": "분할횟수·회당금액 최적화",
                },
                value=saved_test_mode, label="테스트 유형",
            ).props("outlined").classes("w-full")
            criterion = ui.select(
                CRITERIA, value=saved_criterion, label="최적화 기준",
            ).props("outlined").classes("w-full")
            country = ui.select(["한국", "미국"], value=saved_country, label="국가").props("outlined").classes("control-grow")
            symbol_query = ui.input(
                "종목명 또는 코드", value=str(saved.get("symbol_query", "")),
                placeholder="입력하면 종목 후보를 찾습니다",
            ).props("outlined clearable").classes("w-full")
            symbol_options = {saved_symbol: saved_symbol_label} if saved_symbol else {}
            symbol_select = ui.select(
                symbol_options, value=saved_symbol or None, label="검색 결과",
            ).props("outlined options-dense").classes("w-full")
            search_status = ui.label(
                "저장된 종목을 복원했습니다." if saved_symbol else "종목명이나 코드를 입력하세요."
            ).classes("status-text")
            period = ui.select(
                {**{str(year): f"{year}년" for year in range(1, 11)}, "custom": "직접 설정"},
                value=saved_period, label="기간",
            ).props("outlined").classes("control-grow")
            with ui.row().classes("backtest-field-grid backtest-date-grid w-full"):
                start_input = ui.input(
                    "From", value=str(saved.get("start", default_start.isoformat())),
                ).props("type=date").classes("date-input")
                end_input = ui.input(
                    "To", value=str(saved.get("end", today.isoformat())),
                ).props("type=date").classes("date-input")
            with ui.row().classes("backtest-field-grid w-full"):
                initial_raw = ui.number(
                    "최초 투자총액", value=float(saved.get("initial", 10_000_000)), min=.01, step=1,
                ).props("outlined").classes("control-grow")
                split_method = ui.select(
                    ["금액", "횟수"], value=str(saved.get("split_method", "횟수")), label="분할 방식",
                ).props("outlined").classes("control-grow")
                amount_raw = ui.number(
                    "금액 방식 · 회당 매수금액", value=float(saved.get("amount", 1_000_000)),
                    min=.01, step=1,
                ).props("outlined").classes("control-grow")
                count_preset = ui.select(
                    ["직접입력", *range(1, 11)], value=saved.get("count_preset", 10),
                    label="횟수 방식",
                ).props("outlined").classes("control-grow")
                count_custom = ui.number(
                    "직접 분할횟수", value=int(saved.get("count_custom", 10)), min=1, step=1,
                ).props("outlined").classes("control-grow")
            tranche_display = ui.input("계산된 회당 매수금액").props("outlined readonly").classes("w-full")
            with ui.row().classes("backtest-field-grid w-full"):
                buy_preset = ui.select(
                    ["직접입력", 5, 10, 15, 20, 25, 30], value=saved.get("buy_preset", 10),
                    label="매수 하락률",
                ).props("outlined").classes("control-grow")
                buy_custom = ui.number(
                    "직접 매수 하락률(%)", value=float(saved.get("buy_custom", 10)),
                    min=.01, max=99.99, step=.1,
                ).props("outlined").classes("control-grow")
                sell_preset = ui.select(
                    ["직접입력", 5, 10, 15, 20, 25, 30], value=saved.get("sell_preset", 10),
                    label="매도 상승률",
                ).props("outlined").classes("control-grow")
                sell_custom = ui.number(
                    "직접 매도 상승률(%)", value=float(saved.get("sell_custom", 10)),
                    min=.01, step=.1,
                ).props("outlined").classes("control-grow")
            with ui.card().classes("editor-section w-full") as ratio_sweep_card:
                ui.label("비율 최적화 범위").classes("font-bold")
                with ui.row().classes("backtest-field-grid w-full"):
                    buy_min = ui.number(
                        "매수 하락률 최소(%)", value=float(saved.get("buy_min", 5)),
                        min=.01, max=99.99, step=.5,
                    ).props("outlined").classes("control-grow")
                    buy_max = ui.number(
                        "매수 하락률 최대(%)", value=float(saved.get("buy_max", 30)),
                        min=.01, max=99.99, step=.5,
                    ).props("outlined").classes("control-grow")
                    buy_step = ui.number(
                        "매수 간격(%p)", value=float(saved.get("buy_step", 5)),
                        min=.01, max=99, step=.5,
                    ).props("outlined").classes("control-grow")
                    sell_min = ui.number(
                        "매도 상승률 최소(%)", value=float(saved.get("sell_min", 5)),
                        min=.01, step=.5,
                    ).props("outlined").classes("control-grow")
                    sell_max = ui.number(
                        "매도 상승률 최대(%)", value=float(saved.get("sell_max", 30)),
                        min=.01, step=.5,
                    ).props("outlined").classes("control-grow")
                    sell_step = ui.number(
                        "매도 간격(%p)", value=float(saved.get("sell_step", 5)),
                        min=.01, step=.5,
                    ).props("outlined").classes("control-grow")
            with ui.card().classes("editor-section w-full") as allocation_sweep_card:
                ui.label("분할 규모 최적화 범위").classes("font-bold")
                with ui.row().classes("backtest-field-grid w-full"):
                    count_min = ui.number(
                        "최소 분할횟수", value=int(saved.get("count_min", 2)), min=1, step=1,
                    ).props("outlined").classes("control-grow")
                    count_max = ui.number(
                        "최대 분할횟수", value=int(saved.get("count_max", 20)), min=1, step=1,
                    ).props("outlined").classes("control-grow")
                    count_step = ui.number(
                        "횟수 간격", value=int(saved.get("count_step", 1)), min=1, step=1,
                    ).props("outlined").classes("control-grow")
                    amount_min = ui.number(
                        "최소 회당금액", value=float(saved.get("amount_min", 100_000)),
                        min=.01, step=1,
                    ).props("outlined").classes("control-grow")
                    amount_max = ui.number(
                        "최대 회당금액", value=float(saved.get("amount_max", 2_000_000)),
                        min=.01, step=1,
                    ).props("outlined").classes("control-grow")
                    amount_step = ui.number(
                        "금액 간격", value=float(saved.get("amount_step", 100_000)),
                        min=.01, step=1,
                    ).props("outlined").classes("control-grow")
            quantity_mode = ui.select(
                {
                    "whole": "1주 단위 정수 매매",
                    "fractional_6": "금액 기준 소수점 매매(6자리)",
                },
                value=str(saved.get("quantity_mode", "fractional_6")), label="수량 방식",
            ).props("outlined").classes("w-full")
            currency_help = ui.label("").classes("text-xs text-grey-6")
            run_button = ui.button("백테스트 실행", icon="play_arrow").classes("primary-action w-full")

        status = ui.label("조건을 입력한 뒤 실행하세요.").classes("status-text")
        result_area = ui.column().classes("w-full gap-4")

        def persist_form() -> None:
            if state["restoring"]:
                return
            selected_symbol = str(symbol_select.value or "")
            options = symbol_select.options if isinstance(symbol_select.options, dict) else {}
            app.storage.user[FORM_STORAGE_KEY] = {
                "test_mode": str(test_mode.value),
                "criterion": str(criterion.value),
                "country": str(country.value),
                "symbol_query": str(symbol_query.value or ""),
                "symbol": selected_symbol,
                "symbol_label": str(options.get(selected_symbol, selected_symbol)),
                "period": str(period.value),
                "start": str(start_input.value),
                "end": str(end_input.value),
                "initial": float(initial_raw.value or 0),
                "split_method": str(split_method.value),
                "amount": float(amount_raw.value or 0),
                "count_preset": count_preset.value,
                "count_custom": int(count_custom.value or 1),
                "buy_preset": buy_preset.value,
                "buy_custom": float(buy_custom.value or 0),
                "sell_preset": sell_preset.value,
                "sell_custom": float(sell_custom.value or 0),
                "buy_min": float(buy_min.value or 0),
                "buy_max": float(buy_max.value or 0),
                "buy_step": float(buy_step.value or 0),
                "sell_min": float(sell_min.value or 0),
                "sell_max": float(sell_max.value or 0),
                "sell_step": float(sell_step.value or 0),
                "count_min": int(count_min.value or 1),
                "count_max": int(count_max.value or 1),
                "count_step": int(count_step.value or 1),
                "amount_min": float(amount_min.value or 0),
                "amount_max": float(amount_max.value or 0),
                "amount_step": float(amount_step.value or 0),
                "quantity_mode": str(quantity_mode.value),
            }

        def update_currency_inputs() -> None:
            is_us = country.value == "미국"
            step = .01 if is_us else 1
            digits = 2 if is_us else 0
            for control in (initial_raw, amount_raw, amount_min, amount_max, amount_step):
                if control.value is not None:
                    normalized = round(float(control.value), digits)
                    if float(control.value) != normalized:
                        control.value = normalized
            initial_raw.props(f"step={step}")
            amount_raw.props(f"step={step}")
            currency_help.set_text(
                "달러 금액은 USD를 직접 입력하며 소수점 둘째 자리까지 사용할 수 있습니다."
                if is_us else
                "원화 금액은 원 단위 정수로 입력합니다."
            )

        def update_dynamic_controls() -> None:
            update_currency_inputs()
            mode = str(test_mode.value)
            amount_mode = split_method.value == "금액"
            optimizing_allocation = mode == "allocation"
            fixed_ratio = mode != "ratio"
            criterion.set_visibility(mode != "single")
            ratio_sweep_card.set_visibility(mode == "ratio")
            allocation_sweep_card.set_visibility(optimizing_allocation)
            amount_raw.set_visibility(amount_mode and not optimizing_allocation)
            count_preset.set_visibility(not amount_mode and not optimizing_allocation)
            count_custom.set_visibility(
                not amount_mode and not optimizing_allocation
                and count_preset.value == "직접입력"
            )
            tranche_display.set_visibility(not optimizing_allocation)
            buy_preset.set_visibility(fixed_ratio)
            sell_preset.set_visibility(fixed_ratio)
            buy_custom.set_visibility(fixed_ratio and buy_preset.value == "직접입력")
            sell_custom.set_visibility(fixed_ratio and sell_preset.value == "직접입력")
            count_min.set_visibility(not amount_mode)
            count_max.set_visibility(not amount_mode)
            count_step.set_visibility(not amount_mode)
            amount_min.set_visibility(amount_mode)
            amount_max.set_visibility(amount_mode)
            amount_step.set_visibility(amount_mode)
            raw_total = float(initial_raw.value or 0)
            if amount_mode:
                raw_tranche = float(amount_raw.value or 0)
            else:
                count_value = int(count_custom.value or 1) if count_preset.value == "직접입력" else int(count_preset.value or 1)
                raw_tranche = raw_total / max(1, count_value)
            tranche_display.value = _money(_actual_money(raw_tranche, str(country.value)), str(country.value))
            run_button.set_text(
                "백테스트 실행" if mode == "single" else "최적화 실행"
            )
            persist_form()

        def apply_period() -> None:
            if period.value == "custom":
                return
            end = datetime.now(ZoneInfo("Asia/Seoul")).date()
            state["date_sync"] = True
            end_input.value = end.isoformat()
            start_input.value = (pd.Timestamp(end) - pd.DateOffset(years=int(period.value))).date().isoformat()
            ui.timer(.05, lambda: state.__setitem__("date_sync", False), once=True)

        def mark_custom() -> None:
            if not state["date_sync"]:
                period.value = "custom"

        async def search() -> None:
            state["search_version"] = int(state["search_version"]) + 1
            version = int(state["search_version"])
            query = str(symbol_query.value or "").strip()
            if not query:
                symbol_select.options = {}
                symbol_select.value = None
                symbol_select.update()
                return
            await asyncio.sleep(.45)
            if version != state["search_version"]:
                return
            search_status.set_text("종목을 검색하고 있습니다…")
            try:
                options = await run.io_bound(search_symbols, query, str(country.value))
            except Exception as error:
                search_status.set_text(f"종목 검색 실패: {error}")
                return
            if version != state["search_version"]:
                return
            symbol_select.options = options
            symbol_select.value = next(iter(options), None)
            symbol_select.update()
            search_status.set_text(f"{len(options)}개 후보를 찾았습니다." if options else "일치하는 종목이 없습니다.")
            persist_form()

        def reset_symbol() -> None:
            symbol_query.value = ""
            symbol_select.options = {}
            symbol_select.value = None
            symbol_select.update()
            update_dynamic_controls()
            persist_form()

        async def execute() -> None:
            persist_form()
            run_button.disable()
            result_area.clear()
            status.set_text("수정종가를 불러와 분할매매를 계산하고 있습니다…")
            try:
                if not symbol_select.value:
                    raise ValueError("검색 결과에서 종목을 선택해 주세요.")
                start = date.fromisoformat(str(start_input.value))
                end = date.fromisoformat(str(end_input.value))
                if start > end:
                    raise ValueError("From은 To보다 늦을 수 없습니다.")
                selected_country = str(country.value)
                selected_test_mode = str(test_mode.value)
                capital = _actual_money(float(initial_raw.value or 0), selected_country)
                if split_method.value == "금액" and selected_test_mode != "allocation":
                    tranche = _actual_money(float(amount_raw.value or 0), selected_country)
                    split_input = f"금액 · {_money(tranche, selected_country)}"
                elif split_method.value == "횟수" and selected_test_mode != "allocation":
                    count_value = int(count_custom.value or 1) if count_preset.value == "직접입력" else int(count_preset.value or 1)
                    if count_value < 1:
                        raise ValueError("분할횟수는 1 이상의 정수여야 합니다.")
                    tranche = _actual_money(capital / count_value, selected_country)
                    split_input = f"횟수 · {count_value}회"
                else:
                    tranche = capital
                    split_input = f"{split_method.value} 범위 최적화"
                if selected_test_mode == "ratio":
                    buy = sell = .10
                else:
                    buy = float(buy_custom.value if buy_preset.value == "직접입력" else buy_preset.value) / 100
                    sell = float(sell_custom.value if sell_preset.value == "직접입력" else sell_preset.value) / 100
                selected_quantity_mode = str(quantity_mode.value)
                config = SplitTradingConfig(
                    capital, tranche, buy, sell, selected_quantity_mode,
                )
                symbol = str(symbol_select.value)
                close = await run.io_bound(load_adjusted_close, symbol, start, end)
                label = str(symbol_select.options.get(symbol, symbol))
                optimization = None
                if selected_test_mode == "single":
                    result = await run.io_bound(run_split_backtest, close, config)
                elif selected_test_mode == "ratio":
                    buy_rates = [value / 100 for value in _sweep_values(
                        float(buy_min.value or 0), float(buy_max.value or 0),
                        float(buy_step.value or 0),
                    )]
                    sell_rates = [value / 100 for value in _sweep_values(
                        float(sell_min.value or 0), float(sell_max.value or 0),
                        float(sell_step.value or 0),
                    )]
                    if len(buy_rates) * len(sell_rates) > 400:
                        raise ValueError("비율 조합은 최대 400개까지 계산할 수 있습니다. 간격을 늘려주세요.")
                    if len(buy_rates) * len(sell_rates) * config.max_slots > 20_000:
                        raise ValueError(
                            "비율 조합과 동시 보유 차수가 너무 큽니다. "
                            "비율 간격을 늘리거나 분할횟수를 줄여주세요."
                        )
                    status.set_text(f"{len(buy_rates) * len(sell_rates)}개 비율 조합을 계산하고 있습니다…")
                    optimization = await run.io_bound(
                        run_ratio_sweep, close, config, buy_rates, sell_rates,
                        str(criterion.value),
                    )
                    result = optimization["best"]
                    best = optimization["best_parameters"]
                    buy, sell = best["매수 하락률"], best["매도 상승률"]
                    split_input += (
                        f" · 매수 {float(buy_min.value):g}~{float(buy_max.value):g}%"
                        f" / 매도 {float(sell_min.value):g}~{float(sell_max.value):g}%"
                    )
                else:
                    if split_method.value == "횟수":
                        count_values = _sweep_values(
                            float(count_min.value or 0), float(count_max.value or 0),
                            float(count_step.value or 0),
                        )
                        counts = sorted({int(value) for value in count_values if int(value) >= 1})
                        candidates = [
                            (f"{count_value}회", _actual_money(capital / count_value, selected_country))
                            for count_value in counts
                        ]
                    else:
                        amount_values = _sweep_values(
                            float(amount_min.value or 0), float(amount_max.value or 0),
                            float(amount_step.value or 0),
                        )
                        candidates = [
                            (_money(_actual_money(value, selected_country), selected_country),
                             _actual_money(value, selected_country))
                            for value in amount_values
                        ]
                    if len(candidates) > 100:
                        raise ValueError("분할 규모 조합은 최대 100개까지 계산할 수 있습니다. 간격을 늘려주세요.")
                    estimated_slots = sum(
                        int(capital // tranche_value)
                        for _, tranche_value in candidates if tranche_value > 0
                    )
                    if estimated_slots > 20_000:
                        raise ValueError(
                            "분할 규모 조합의 총 계산량이 너무 큽니다. "
                            "최소 회당금액이나 간격을 늘려주세요."
                        )
                    status.set_text(f"{len(candidates)}개 분할 규모를 계산하고 있습니다…")
                    optimization = await run.io_bound(
                        run_tranche_sweep, close, config, candidates,
                        str(criterion.value),
                    )
                    result = optimization["best"]
                    tranche = float(optimization["best_parameters"]["회당 매수금액"])
                conditions = {
                    "국가": selected_country, "종목": label,
                    "테스트 유형": str(test_mode.options.get(selected_test_mode, selected_test_mode)),
                    "최적화 기준": (
                        CRITERIA[str(criterion.value)] if selected_test_mode != "single" else "해당 없음"
                    ),
                    "요청 기간": f"{start} ~ {end}",
                    "실제 데이터 기간": f"{result['actual_start']} ~ {result['actual_end']}",
                    "최초 투자총액": capital, "분할 방식": split_input,
                    "회당 매수금액": tranche, "매수 하락률": buy,
                    "매도 상승률": sell, "가격 기준": "yfinance auto_adjust 수정종가",
                    "수량 방식": (
                        "1주 단위 정수 매매" if selected_quantity_mode == "whole"
                        else "금액 기준 소수점 매매(6자리)"
                    ),
                    "수수료·세금": 0,
                }
                state.update(
                    result=result, conditions=conditions, symbol_label=label,
                    optimization=optimization, test_mode=selected_test_mode,
                )
            except Exception as error:
                status.set_text("백테스트를 완료하지 못했습니다.")
                with result_area:
                    with ui.card().classes("error-card w-full"):
                        ui.label(str(error)).classes("text-negative")
            else:
                status.set_text(f"{result['actual_start']} ~ {result['actual_end']} 수정종가 기준")
                with result_area:
                    if optimization is None:
                        _render_result(result, selected_country, label)
                    else:
                        _render_optimization(
                            optimization, selected_country, label, selected_test_mode,
                        )
                    export_status = ui.label("현재 결과를 연결된 Google Sheets의 새 탭으로 저장할 수 있습니다.").classes("status-text")
                    export_button = ui.button("Google Sheets로 내보내기", icon="table_view").classes("primary-action")

                    async def export_sheet() -> None:
                        export_button.disable()
                        export_status.set_text("새 Google Sheets 탭을 만드는 중입니다…")
                        try:
                            spreadsheet = await run.io_bound(connect_spreadsheet)
                            title, url = await run.io_bound(
                                export_result, spreadsheet,
                                symbol_name=label.split(" · ", 1)[-1],
                                conditions=conditions, result=result,
                            )
                        except Exception as error:
                            export_status.set_text(f"내보내기 실패: {error}")
                        else:
                            export_status.set_text(f"'{title}' 탭에 저장했습니다.")
                            ui.link("새 탭 열기", url, new_tab=True).classes("text-primary font-bold")
                        finally:
                            export_button.enable()

                    export_button.on("click", export_sheet)
            finally:
                run_button.enable()

        def period_changed() -> None:
            apply_period()
            persist_form()

        def date_changed() -> None:
            mark_custom()
            persist_form()

        async def query_changed() -> None:
            persist_form()
            await search()

        country.on_value_change(lambda _: reset_symbol())
        symbol_query.on_value_change(lambda _: query_changed())
        symbol_select.on_value_change(lambda _: persist_form())
        period.on_value_change(lambda _: period_changed())
        start_input.on_value_change(lambda _: date_changed())
        end_input.on_value_change(lambda _: date_changed())
        for control in (
            test_mode, criterion,
            initial_raw, split_method, amount_raw, count_preset, count_custom,
            buy_preset, buy_custom, sell_preset, sell_custom, quantity_mode,
            buy_min, buy_max, buy_step, sell_min, sell_max, sell_step,
            count_min, count_max, count_step, amount_min, amount_max, amount_step,
        ):
            control.on_value_change(lambda _: update_dynamic_controls())
        run_button.on("click", execute)
        state["restoring"] = False
        update_dynamic_controls()

