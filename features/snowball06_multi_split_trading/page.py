"""NiceGUI pages shared by finite and reverse-funded multi-asset strategies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nicegui import app, run, ui

from common.snowball.layout import shell
from common.snowball.period_controls import create_period_controls
from common.snowball.ticker_picker import TickerPicker
from features.snowball04_split_trading.data import load_adjusted_close
from features.snowball04_split_trading.page import _render_result
from .engine import AssetRule, MultiSplitConfig, run_multi_split_backtest
from .persistence import deserialize_multi_result, serialize_multi_result


def _money(value: float, country: str) -> str:
    return f"₩{value:,.0f}" if country == "한국" else f"${value:,.2f}"


def _load_prices(symbols: list[str], start: date, end: date) -> dict[str, pd.Series]:
    output: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=min(5, len(symbols))) as executor:
        pending = {
            executor.submit(load_adjusted_close, symbol, start, end): symbol
            for symbol in symbols
        }
        for future in as_completed(pending):
            symbol = pending[future]
            try:
                output[symbol] = future.result()
            except Exception as error:
                raise ValueError(f"{symbol} 가격 조회 실패: {error}") from error
    return output


def render_multi_split_page(
    *, forced_funding_mode: str, storage_prefix: str, page_title: str, page_subtitle: str,
) -> None:
    if forced_funding_mode not in {"limited", "required"}:
        raise ValueError("지원하지 않는 자금 운용 방식입니다.")
    shell()
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    default_start = (pd.Timestamp(today) - pd.DateOffset(years=3)).date()
    form_key = f"{storage_prefix}_form_v1"
    result_key = f"{storage_prefix}_result_v1"
    saved_raw = app.storage.user.get(form_key, {})
    saved = saved_raw if isinstance(saved_raw, dict) else {}
    saved_country = str(saved.get("country", "미국"))
    default_initial = 10_000_000 if saved_country == "한국" else 10_000
    default_tranche = 1_000_000 if saved_country == "한국" else 1_000
    saved_rules = {
        str(item.get("symbol", "")).upper(): dict(item)
        for item in saved.get("rules", []) if isinstance(item, dict) and item.get("symbol")
    }
    saved_symbols = list(saved_rules) or ["QQQ", "IEF"]
    state: dict[str, object] = {"rule_values": saved_rules, "rendering": False}

    with ui.column().classes("page-container w-full gap-4"):
        with ui.row().classes("items-center gap-2"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("round flat")
            with ui.column().classes("gap-0"):
                ui.label(page_title).classes("page-title")
                ui.label(page_subtitle).classes("page-subtitle")

        with ui.card().classes("backtest-controls w-full"):
            dates = create_period_controls(
                default_start=date.fromisoformat(str(saved.get("start", default_start.isoformat()))),
                default_end=date.fromisoformat(str(saved.get("end", today.isoformat()))),
                default_period=str(saved.get("period", "3")),
            )
            picker = TickerPicker(
                defaults=saved_symbols, max_items=10, label="포트폴리오 종목",
                default_country=saved_country,
            )
            if forced_funding_mode == "limited":
                with ui.row().classes("backtest-field-grid w-full"):
                    initial = ui.number(
                        "전체 투자금", value=float(saved.get("initial", default_initial)), min=.01, step=1,
                    ).props("outlined").classes("control-grow")
                    allocation_mode = ui.select(
                        {"weight": "종목별 비중(%)", "amount": "종목별 배정금액"},
                        value=str(saved.get("allocation_mode", "weight")), label="배정 방식",
                    ).props("outlined").classes("control-grow")
            else:
                initial = None
                allocation_mode = None
                ui.label(
                    "종목별 기준 1회 매수금액으로 모든 신호를 체결하고 전체 최소 필요자금을 역산합니다."
                ).classes("status-text")
            with ui.row().classes("backtest-field-grid w-full"):
                quantity_mode = ui.select(
                    {"whole": "1주 단위 정수 매매", "fractional_6": "소수점 6자리 매매"},
                    value=str(saved.get("quantity_mode", "whole")), label="수량 방식",
                ).props("outlined")
                buy_fee = ui.number("매수수수료(%)", value=float(saved.get("buy_fee", 0)), min=0, max=10, step=.01).props("outlined")
                sell_fee = ui.number("매도수수료(%)", value=float(saved.get("sell_fee", 0)), min=0, max=10, step=.01).props("outlined")
                sell_tax = ui.number("매도세금(%)", value=float(saved.get("sell_tax", 0)), min=0, max=10, step=.01).props("outlined")
                slippage = ui.number("슬리피지(%)", value=float(saved.get("slippage", 0)), min=0, max=10, step=.01).props("outlined")
                fx_cost = ui.number("환전비용(%)", value=float(saved.get("fx_cost", 0)), min=0, max=10, step=.01).props("outlined")

            ui.label("종목별 조건").classes("section-title")
            rule_area = ui.column().classes("w-full gap-3")
            rule_controls: dict[str, dict[str, object]] = {}

            def store_rule(symbol: str, key: str, value: object) -> None:
                values = state["rule_values"]
                assert isinstance(values, dict)
                values.setdefault(symbol, {"symbol": symbol})[key] = float(value or 0)

            def refresh_computed_tranches() -> None:
                if forced_funding_mode != "limited":
                    return
                total = float(initial.value or 0)
                for controls in rule_controls.values():
                    allocation_control = controls["allocation"]
                    slots_control = controls["slots"]
                    hint = controls["computed"]
                    raw = float(allocation_control.value or 0)
                    allocated = (
                        total * raw / 100
                        if str(allocation_mode.value) == "weight" else raw
                    )
                    count = max(1, int(slots_control.value or 1))
                    hint.set_text(
                        f"배정 {_money(allocated, str(picker.country.value))} · "
                        f"계산된 회당 매수금액 {_money(allocated / count, str(picker.country.value))}"
                    )

            def render_rules() -> None:
                state["rendering"] = True
                rule_area.clear()
                rule_controls.clear()
                symbols = picker.values()
                with rule_area:
                    if not symbols:
                        ui.label("검색한 종목을 두 개 이상 추가하세요.").classes("empty-state")
                    for symbol in symbols:
                        values_map = state["rule_values"]
                        assert isinstance(values_map, dict)
                        values = values_map.setdefault(symbol, {"symbol": symbol})
                        with ui.card().classes("editor-section w-full"):
                            ui.label(symbol).classes("font-bold text-primary")
                            with ui.row().classes("backtest-field-grid w-full"):
                                if forced_funding_mode == "limited":
                                    is_weight = str(allocation_mode.value) == "weight"
                                    key = "weight" if is_weight else "amount"
                                    default = (
                                        100.0 / max(len(symbols), 1)
                                        if is_weight
                                        else float(initial.value or default_initial) / max(len(symbols), 1)
                                    )
                                    values.setdefault(key, default)
                                    allocation = ui.number(
                                        "배정 비중(%)" if is_weight else "배정금액",
                                        value=float(values.get(key, default)), min=.01,
                                        max=100 if is_weight else None, step=1,
                                    ).props("outlined")
                                    allocation.on_value_change(
                                        lambda event, item=symbol, field=key: store_rule(item, field, event.value)
                                    )
                                    allocation.on_value_change(lambda _: refresh_computed_tranches())
                                else:
                                    values.setdefault("tranche", default_tranche)
                                    tranche = ui.number(
                                        "기준 1회 매수금액",
                                        value=float(values.get("tranche", default_tranche)), min=.01, step=1,
                                    ).props("outlined")
                                    tranche.on_value_change(
                                        lambda event, item=symbol: store_rule(item, "tranche", event.value)
                                    )
                                values.setdefault("slots", 10)
                                values.setdefault("rate", 10)
                                slots = ui.number(
                                    "분할횟수", value=int(values.get("slots", 10)), min=1, max=1000, step=1,
                                ).props("outlined")
                                rate = ui.number(
                                    "매수·매도율(%)", value=float(values.get("rate", 10)),
                                    min=.01, max=99.99, step=1,
                                ).props("outlined")
                                slots.on_value_change(
                                    lambda event, item=symbol: store_rule(item, "slots", event.value)
                                )
                                slots.on_value_change(lambda _: refresh_computed_tranches())
                                rate.on_value_change(
                                    lambda event, item=symbol: store_rule(item, "rate", event.value)
                                )
                                rule_controls[symbol] = {
                                    "allocation": allocation if forced_funding_mode == "limited" else None,
                                    "tranche": tranche if forced_funding_mode == "required" else None,
                                    "slots": slots, "rate": rate,
                                    "computed": (
                                        ui.label("").classes("status-text")
                                        if forced_funding_mode == "limited" else None
                                    ),
                                }
                    refresh_computed_tranches()
                state["rendering"] = False

            picker.set_on_change(render_rules)
            if allocation_mode is not None:
                allocation_mode.on_value_change(lambda _: render_rules())
            if initial is not None:
                initial.on_value_change(lambda _: refresh_computed_tranches())
            render_rules()
            run_button = ui.button("다종목 백테스트 실행", icon="play_arrow").classes("primary-action w-full")

        status = ui.label("종목별 조건을 확인한 뒤 실행하세요.").classes("status-text")
        result_area = ui.column().classes("w-full gap-4")

        def persist_form() -> None:
            values = state["rule_values"]
            assert isinstance(values, dict)
            app.storage.user[form_key] = {
                "start": str(dates.start.value), "end": str(dates.end.value),
                "period": str(dates.period.value), "country": str(picker.country.value),
                "initial": float(initial.value or 0) if initial is not None else 0,
                "allocation_mode": str(allocation_mode.value) if allocation_mode is not None else "tranche",
                "quantity_mode": str(quantity_mode.value),
                "buy_fee": float(buy_fee.value or 0), "sell_fee": float(sell_fee.value or 0),
                "sell_tax": float(sell_tax.value or 0), "slippage": float(slippage.value or 0),
                "fx_cost": float(fx_cost.value or 0),
                "rules": [values[symbol] for symbol in picker.values() if symbol in values],
            }

        def build_rules() -> list[AssetRule]:
            symbols = picker.values()
            if len(symbols) < 2:
                raise ValueError("포트폴리오 종목을 2개 이상 추가하세요.")
            country = str(picker.country.value)
            korean = [symbol for symbol in symbols if symbol.endswith((".KS", ".KQ"))]
            if country == "한국" and len(korean) != len(symbols):
                raise ValueError("한국 시장 포트폴리오에는 국내 종목만 추가해 주세요.")
            if country == "미국" and korean:
                raise ValueError("미국 시장 포트폴리오에는 미국 종목만 추가해 주세요.")
            output = []
            total = float(initial.value or 0) if initial is not None else 0.0
            allocations = []
            for symbol in symbols:
                controls = rule_controls[symbol]
                slots = int(controls["slots"].value or 1)
                rate = float(controls["rate"].value or 0) / 100
                if forced_funding_mode == "limited":
                    raw = float(controls["allocation"].value or 0)
                    allocation = total * raw / 100 if str(allocation_mode.value) == "weight" else raw
                    tranche = allocation / slots
                    allocations.append(allocation)
                else:
                    tranche = float(controls["tranche"].value or 0)
                    allocation = tranche * slots
                output.append(AssetRule(symbol, tranche, slots, rate, rate, allocation))
            if forced_funding_mode == "limited" and sum(allocations) > total + 1e-6:
                raise ValueError("종목별 배정금액 합계가 전체 투자금을 초과합니다.")
            return output

        async def execute() -> None:
            run_button.disable()
            result_area.clear()
            try:
                start = date.fromisoformat(str(dates.start.value))
                end = date.fromisoformat(str(dates.end.value))
                if start >= end:
                    raise ValueError("시작일은 종료일보다 앞서야 합니다.")
                rules = build_rules()
                persist_form()
                status.set_text("여러 종목의 가격과 거래 신호를 계산하고 있습니다…")
                prices = await run.io_bound(_load_prices, [rule.symbol for rule in rules], start, end)
                config = MultiSplitConfig(
                    initial_capital=float(initial.value or 0) if initial is not None else 0.0,
                    funding_mode=forced_funding_mode, quantity_mode=str(quantity_mode.value),
                    buy_fee_rate=float(buy_fee.value or 0) / 100,
                    sell_fee_rate=float(sell_fee.value or 0) / 100,
                    sell_tax_rate=float(sell_tax.value or 0) / 100,
                    slippage_rate=float(slippage.value or 0) / 100,
                    fx_cost_rate=float(fx_cost.value or 0) / 100,
                )
                result = await run.io_bound(run_multi_split_backtest, prices, rules, config)
                app.storage.user[result_key] = serialize_multi_result(result)
                status.set_text(
                    f"계산 완료 · {result['actual_start']} ~ {result['actual_end']} · "
                    f"{len(rules)}개 종목"
                )
                with result_area:
                    _render_result(result, str(picker.country.value), f"{len(rules)}종목 포트폴리오")
            except Exception as error:
                status.set_text("계산을 완료하지 못했습니다.")
                with result_area:
                    with ui.card().classes("error-card w-full"):
                        ui.label(str(error)).classes("text-negative")
            finally:
                run_button.enable()

        run_button.on("click", execute)
        restored = app.storage.user.get(result_key)
        if isinstance(restored, dict):
            try:
                result = deserialize_multi_result(restored)
                with result_area:
                    _render_result(result, str(picker.country.value), "저장된 다종목 포트폴리오")
                status.set_text("직전 다종목 결과를 복원했습니다.")
            except Exception:
                app.storage.user.pop(result_key, None)


@ui.page("/multi-split-trading")
def page() -> None:
    render_multi_split_page(
        forced_funding_mode="limited", storage_prefix="multi_split_trading",
        page_title="다종목 분할매수",
        page_subtitle="여러 종목이 하나의 고정 투자금과 현금 잔고를 공유합니다.",
    )
