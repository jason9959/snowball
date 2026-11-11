"""NiceGUI page for mixing two saved Snowball strategies."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from zoneinfo import ZoneInfo

from nicegui import run, ui

from common.snowball.backtest_results import render_backtest_result
from common.snowball.layout import shell
from common.snowball.preset_store import load_browser_presets, restore_local_presets
from snowball.builtin_library import (
    builtin_user_preset,
    individual_builtin_specs,
)
from snowball.user_presets import UserPreset
from .service import prepare_mix_prices, run_portfolio_mix


USER_PRESET_PREFIX = "사용자::"


@ui.page("/mix-backtest")
def page() -> None:
    shell()
    user_presets = load_browser_presets()
    builtin_titles = list(individual_builtin_specs())

    def strategy_options() -> dict[str, str]:
        options = {title: f"기본 · {title}" for title in builtin_titles}
        options.update({
            f"{USER_PRESET_PREFIX}{title}": f"사용자 · {title}"
            for title in sorted(user_presets)
        })
        return options

    def selected_record(value: object) -> UserPreset:
        key = str(value or "")
        if key.startswith(USER_PRESET_PREFIX):
            title = key.removeprefix(USER_PRESET_PREFIX)
            try:
                return user_presets[title]
            except KeyError as error:
                raise ValueError(f"저장된 프리셋을 찾을 수 없습니다: {title}") from error
        return builtin_user_preset(key)

    with ui.column().classes("page-container w-full gap-4"):
        ui.button(
            "메인으로", icon="arrow_back", on_click=lambda: ui.navigate.to("/"),
        ).props("flat no-caps").classes("page-back-button")
        ui.label("포트폴리오 혼합 백테스트").classes("page-title")
        ui.label(
            "매월 직전 평가액을 입력 비율로 다시 나누고, 각 전략이 선택한 종목에 배분합니다."
        ).classes("page-subtitle")

        with ui.card().classes("backtest-controls w-full"):
            options = strategy_options()
            first_select = ui.select(
                options, value=builtin_titles[0], label="첫 번째 프리셋",
            ).props("outlined options-dense").classes("w-full")
            second_select = ui.select(
                options, value=builtin_titles[1], label="두 번째 프리셋",
            ).props("outlined options-dense").classes("w-full")
            with ui.row().classes("backtest-field-grid backtest-date-grid w-full"):
                start_input = ui.input("시작일", value="2010-01-01").props(
                    "type=date"
                ).classes("date-input")
                end_input = ui.input(
                    "종료일", value=datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
                ).props("type=date").classes("date-input")
            with ui.row().classes("backtest-field-grid w-full"):
                initial = ui.number(
                    "초기 투자금", value=10_000, min=100, step=1_000,
                ).props("outlined").classes("control-grow")
                cost = ui.number(
                    "편도 거래비용(bp)", value=0, min=0, max=500, step=1,
                ).props("outlined").classes("control-grow")
                first_weight = ui.number(
                    "첫 번째 프리셋 비중(%)", value=50, min=0, max=100, step=5,
                ).props("outlined").classes("control-grow")
            ratio_summary = ui.label(
                "첫 번째 50% + 두 번째 50% · 매월 목표 비중 복원"
            ).classes("text-sm text-grey-6")
            market_mode = ui.radio(
                {"fixed": "고정 스냅샷", "latest": "최신 데이터"}, value="fixed",
            ).props("inline")
            run_button = ui.button(
                "혼합 백테스트 실행", icon="play_arrow",
            ).classes("primary-action w-full")

        status = ui.label("두 전략과 혼합 비율을 확인하세요.").classes("status-text")
        result_area = ui.column().classes("w-full gap-4")

        def update_summary() -> None:
            value = max(0.0, min(100.0, float(first_weight.value or 0)))
            ratio_summary.set_text(
                f"첫 번째 {value:g}% + 두 번째 {100 - value:g}% · 매월 목표 비중 복원"
            )
            try:
                records = (
                    selected_record(first_select.value), selected_record(second_select.value)
                )
            except ValueError:
                return
            domestic = records[0].domestic_execution and records[1].domestic_execution
            initial.value = 10_000_000 if domestic else 10_000
            initial.min = 100_000 if domestic else 100
            initial.update()

        first_weight.on_value_change(lambda _: update_summary())
        first_select.on_value_change(lambda _: update_summary())
        second_select.on_value_change(lambda _: update_summary())

        async def restore_presets() -> None:
            try:
                await restore_local_presets(user_presets)
            except (ValueError, RuntimeError):
                ui.notify("브라우저 프리셋을 복원하지 못했습니다.", type="warning")
                return
            options = strategy_options()
            first_select.options = options
            second_select.options = options
            first_select.update()
            second_select.update()

        async def execute_mix() -> None:
            run_button.disable()
            result_area.clear()
            status.set_text("공통 가격을 불러와 두 전략의 월별 목표를 계산하고 있습니다…")
            try:
                first = selected_record(first_select.value)
                second = selected_record(second_select.value)
                start_date = date.fromisoformat(str(start_input.value))
                end_date = date.fromisoformat(str(end_input.value))
                status.set_text("1/3 · 두 전략의 공통 Yahoo 가격을 확인하고 있습니다…")
                await run.io_bound(
                    prepare_mix_prices, first, second, start_date, end_date,
                )
                status.set_text("2/3 · 월별 목표 비중을 결합하고 일별 성과를 계산하고 있습니다…")
                result, bundle, cache_hit = await run.io_bound(
                    run_portfolio_mix,
                    first,
                    second,
                    float(first_weight.value or 0) / 100.0,
                    start_date,
                    end_date,
                    initial_investment=float(initial.value or 0),
                    transaction_cost_rate=float(cost.value or 0) / 10_000.0,
                    market_data_mode=str(market_mode.value),
                )
            except Exception as error:
                status.set_text("혼합 백테스트를 완료하지 못했습니다.")
                with result_area:
                    with ui.card().classes("error-card w-full"):
                        ui.label(str(error)).classes("text-negative")
            else:
                status.set_text("3/3 · 결과 차트와 표를 구성하고 있습니다…")
                await asyncio.sleep(0)
                with result_area:
                    render_backtest_result(result, bundle)
                status.set_text(
                    "혼합 백테스트가 완료되었습니다."
                    + (" · 저장된 동일 결과를 불러왔습니다." if cache_hit else "")
                )
            finally:
                run_button.enable()

        run_button.on("click", execute_mix)
        ui.timer(0.2, restore_presets, once=True)
