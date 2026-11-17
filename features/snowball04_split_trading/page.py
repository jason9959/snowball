"""NiceGUI page for the single-asset split-trading backtest."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nicegui import run, ui

from common.snowball.layout import shell
from .data import load_adjusted_close, search_symbols
from .engine import SplitTradingConfig, run_split_backtest
from .sheets import connect_spreadsheet, export_result


def _money(value: float, country: str) -> str:
    return f"₩{value:,.0f}" if country == "한국" else f"${value:,.2f}"


def _actual_money(raw_value: float, country: str) -> float:
    # The US inputs are cents, as defined in the feature specification.
    return float(raw_value) if country == "한국" else float(raw_value) / 100.0


def _table_columns(names: list[str]) -> list[dict[str, object]]:
    return [{"name": name, "label": name, "field": name, "align": "left", "sortable": True}
            for name in names]


def _render_result(result: dict, country: str, symbol_label: str) -> None:
    currency = "KRW" if country == "한국" else "USD"
    ui.label(f"{symbol_label} 결과").classes("section-title")
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in result["split_nav"].index]
    buy_points, sell_points = [], []
    for row in result["trades"].to_dict("records"):
        day = pd.Timestamp(row["거래일"])
        if day not in result["split_nav"].index:
            continue
        point = {
            "value": [day.strftime("%Y-%m-%d"), round(float(result["split_nav"].loc[day]), 4)],
            "name": f"{row['당일 순번']}번 · {row['차수']}차 · {row['시뮬레이션 체결가']:,.2f}",
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
             "data": result["full_nav"].round(4).tolist(), "lineStyle": {"width": 2}},
            {"name": "분할매매", "type": "line", "showSymbol": False,
             "data": result["split_nav"].round(4).tolist(), "lineStyle": {"width": 3}},
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
    )
    with ui.grid().classes("result-metric-grid w-full gap-3"):
        for label, value in cards:
            with ui.card().classes("metric-card w-full"):
                with ui.row().classes("metric-row w-full items-center justify-between no-wrap"):
                    ui.label(label).classes("metric-label")
                    ui.label(value).classes("metric-value")
    ui.label("체결 내역").classes("section-title")
    display = result["trades"].copy()
    for column in ("시뮬레이션 체결가", "수량", "체결금액", "실현손익", "체결 후 현금잔액"):
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):,.4f}"
        )
    display["실현수익률"] = display["실현수익률"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):.2%}"
    )
    ui.table(
        columns=_table_columns(list(display.columns)),
        rows=display.fillna("").to_dict("records"), pagination={"rowsPerPage": 15},
    ).props("flat bordered dense").classes("w-full account-table")


@ui.page("/split-trading")
def split_trading_page() -> None:
    shell()
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    default_start = (pd.Timestamp(today) - pd.DateOffset(years=3)).date()
    state: dict[str, object] = {
        "result": None, "conditions": None, "search_version": 0, "date_sync": False,
    }

    with ui.column().classes("page-container w-full gap-4"):
        with ui.row().classes("items-center gap-2"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("round flat")
            with ui.column().classes("gap-0"):
                ui.label("분할매매 백테스트").classes("page-title")
                ui.label("1차는 보유하고 2차 이후 매수·매도를 반복해 전액 투자와 비교합니다.").classes("page-subtitle")

        with ui.card().classes("backtest-controls w-full"):
            country = ui.select(["한국", "미국"], value="한국", label="국가").props("outlined").classes("control-grow")
            symbol_query = ui.input("종목명 또는 코드", placeholder="입력하면 종목 후보를 찾습니다").props("outlined clearable").classes("w-full")
            symbol_select = ui.select({}, label="검색 결과").props("outlined options-dense").classes("w-full")
            search_status = ui.label("종목명이나 코드를 입력하세요.").classes("status-text")
            period = ui.select(
                {**{str(year): f"{year}년" for year in range(1, 11)}, "custom": "직접 설정"},
                value="3", label="기간",
            ).props("outlined").classes("control-grow")
            with ui.row().classes("backtest-field-grid backtest-date-grid w-full"):
                start_input = ui.input("From", value=default_start.isoformat()).props("type=date").classes("date-input")
                end_input = ui.input("To", value=today.isoformat()).props("type=date").classes("date-input")
            with ui.row().classes("backtest-field-grid w-full"):
                initial_raw = ui.number("최초 투자총액", value=10_000_000, min=1, step=10_000).props("outlined").classes("control-grow")
                split_method = ui.select(["금액", "횟수"], value="횟수", label="분할 방식").props("outlined").classes("control-grow")
                amount_raw = ui.number("금액 방식 · 회당 매수금액", value=1_000_000, min=1, step=10_000).props("outlined").classes("control-grow")
                count_preset = ui.select(["직접입력", *range(1, 11)], value=10, label="횟수 방식").props("outlined").classes("control-grow")
                count_custom = ui.number("직접 분할횟수", value=10, min=1, step=1).props("outlined").classes("control-grow")
            tranche_display = ui.input("계산된 회당 매수금액").props("outlined readonly").classes("w-full")
            with ui.row().classes("backtest-field-grid w-full"):
                buy_preset = ui.select(["직접입력", 5, 10, 15, 20, 25, 30], value=10, label="매수 하락률").props("outlined").classes("control-grow")
                buy_custom = ui.number("직접 매수 하락률(%)", value=10, min=.01, max=99.99, step=.1).props("outlined").classes("control-grow")
                sell_preset = ui.select(["직접입력", 5, 10, 15, 20, 25, 30], value=10, label="매도 상승률").props("outlined").classes("control-grow")
                sell_custom = ui.number("직접 매도 상승률(%)", value=10, min=.01, step=.1).props("outlined").classes("control-grow")
            ui.checkbox("소수점 매수 허용", value=True).props("disable")
            ui.label("미국 금액 입력값은 센트 단위입니다. 예: 10000 → $100.00").classes("text-xs text-grey-6")
            run_button = ui.button("백테스트 실행", icon="play_arrow").classes("primary-action w-full")

        status = ui.label("조건을 입력한 뒤 실행하세요.").classes("status-text")
        result_area = ui.column().classes("w-full gap-4")

        def update_dynamic_controls() -> None:
            amount_mode = split_method.value == "금액"
            amount_raw.set_visibility(amount_mode)
            count_preset.set_visibility(not amount_mode)
            count_custom.set_visibility(not amount_mode and count_preset.value == "직접입력")
            buy_custom.set_visibility(buy_preset.value == "직접입력")
            sell_custom.set_visibility(sell_preset.value == "직접입력")
            raw_total = float(initial_raw.value or 0)
            if amount_mode:
                raw_tranche = float(amount_raw.value or 0)
            else:
                count_value = int(count_custom.value or 1) if count_preset.value == "직접입력" else int(count_preset.value or 1)
                raw_tranche = raw_total / max(1, count_value)
            tranche_display.value = _money(_actual_money(raw_tranche, str(country.value)), str(country.value))

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

        def reset_symbol() -> None:
            symbol_query.value = ""
            symbol_select.options = {}
            symbol_select.value = None
            symbol_select.update()
            update_dynamic_controls()

        async def execute() -> None:
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
                capital = _actual_money(float(initial_raw.value or 0), selected_country)
                if split_method.value == "금액":
                    tranche = _actual_money(float(amount_raw.value or 0), selected_country)
                    split_input = f"금액 · {_money(tranche, selected_country)}"
                else:
                    count_value = int(count_custom.value or 1) if count_preset.value == "직접입력" else int(count_preset.value or 1)
                    if count_value < 1:
                        raise ValueError("분할횟수는 1 이상의 정수여야 합니다.")
                    tranche = capital / count_value
                    split_input = f"횟수 · {count_value}회"
                buy = float(buy_custom.value if buy_preset.value == "직접입력" else buy_preset.value) / 100
                sell = float(sell_custom.value if sell_preset.value == "직접입력" else sell_preset.value) / 100
                config = SplitTradingConfig(capital, tranche, buy, sell)
                symbol = str(symbol_select.value)
                close = await run.io_bound(load_adjusted_close, symbol, start, end)
                result = await run.io_bound(run_split_backtest, close, config)
                label = str(symbol_select.options.get(symbol, symbol))
                conditions = {
                    "국가": selected_country, "종목": label,
                    "요청 기간": f"{start} ~ {end}",
                    "실제 데이터 기간": f"{result['actual_start']} ~ {result['actual_end']}",
                    "최초 투자총액": capital, "분할 방식": split_input,
                    "회당 매수금액": tranche, "매수 하락률": buy,
                    "매도 상승률": sell, "가격 기준": "yfinance auto_adjust 수정종가",
                    "소수점 매수": "허용", "수수료·세금": 0,
                }
                state.update(result=result, conditions=conditions, symbol_label=label)
            except Exception as error:
                status.set_text("백테스트를 완료하지 못했습니다.")
                with result_area:
                    with ui.card().classes("error-card w-full"):
                        ui.label(str(error)).classes("text-negative")
            else:
                status.set_text(f"{result['actual_start']} ~ {result['actual_end']} 수정종가 기준")
                with result_area:
                    _render_result(result, selected_country, label)
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

        country.on_value_change(lambda _: reset_symbol())
        symbol_query.on_value_change(lambda _: search())
        period.on_value_change(lambda _: apply_period())
        start_input.on_value_change(lambda _: mark_custom())
        end_input.on_value_change(lambda _: mark_custom())
        for control in (country, initial_raw, split_method, amount_raw, count_preset, count_custom, buy_preset, sell_preset):
            control.on_value_change(lambda _: update_dynamic_controls())
        run_button.on("click", execute)
        update_dynamic_controls()

