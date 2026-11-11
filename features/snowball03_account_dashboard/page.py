"""Snowball 03 account dashboard and its isolated data caches."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import Lock
from urllib.parse import quote

import numpy as np
import pandas as pd
from nicegui import run, ui

from common.snowball.layout import shell
from snowball.account_store import ACCOUNT_PREFIX, GoogleSheetAccountStore
from snowball.account_dashboard import (
    AccountDashboardData,
    account_date_bounds,
    active_account_frame,
    build_account_dashboard,
)
from snowball.accounting import download_unadjusted_closes, required_price_symbols
from snowball.mailing import (
    RebalanceRecommendation,
    first_xnys_session,
    generate_recommendations,
    load_mailing_portfolios,
)
from snowball.rebalance_compare import build_rebalance_comparison
from snowball.settings import google_sheet_settings

ACCOUNT_CACHE_SECONDS = 300
TRANSACTION_CACHE_SECONDS = 60
PRICE_CACHE_SECONDS = 900
REBALANCE_CACHE_SECONDS = 3600


@dataclass
class _AccountCache:
    tabs: tuple[str, ...] = ()
    loaded_at: float = 0.0


_account_cache = _AccountCache()
_account_cache_lock = Lock()
_transaction_cache: dict[str, tuple[float, list[dict[str, object]]]] = {}
_price_cache: dict[
    tuple[tuple[str, ...], str, str], tuple[float, pd.DataFrame, pd.Series]
] = {}
_rebalance_cache: dict[str, tuple[float, RebalanceRecommendation]] = {}
_data_cache_lock = Lock()


def _connect_store() -> GoogleSheetAccountStore:
    spreadsheet_id, service_account_json = google_sheet_settings()
    if not spreadsheet_id or not service_account_json:
        raise RuntimeError(
            "GOOGLE_SHEET_ID와 GOOGLE_SERVICE_ACCOUNT_JSON 환경변수가 필요합니다."
        )
    return GoogleSheetAccountStore.connect(spreadsheet_id, service_account_json)


def _load_account_tabs(force: bool = False) -> tuple[str, ...]:
    now = time.monotonic()
    with _account_cache_lock:
        if (
            not force
            and _account_cache.tabs
            and now - _account_cache.loaded_at < ACCOUNT_CACHE_SECONDS
        ):
            return _account_cache.tabs
    tabs = _connect_store().list_account_tabs(ensure=False)
    with _account_cache_lock:
        _account_cache.tabs = tabs
        _account_cache.loaded_at = time.monotonic()
    return tabs


def _account_name(tab_name: str) -> str:
    return tab_name.removeprefix(ACCOUNT_PREFIX)


def _load_transactions(account_tab: str, force: bool = False) -> list[dict[str, object]]:
    now = time.monotonic()
    with _data_cache_lock:
        cached = _transaction_cache.get(account_tab)
        if cached and not force and now - cached[0] < TRANSACTION_CACHE_SECONDS:
            return cached[1]
    records = _connect_store().load_transactions_readonly(account_tab)
    with _data_cache_lock:
        _transaction_cache[account_tab] = (now, records)
    return records


def _load_prices(
    symbols: tuple[str, ...], first_date: date, end_date: date, force: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    if not symbols:
        return pd.DataFrame(), pd.Series(dtype=float)
    key = (symbols, first_date.isoformat(), end_date.isoformat())
    now = time.monotonic()
    with _data_cache_lock:
        cached = _price_cache.get(key)
        if cached and not force and now - cached[0] < PRICE_CACHE_SECONDS:
            return cached[1], cached[2]
    closes, usdkrw = download_unadjusted_closes(
        symbols, pd.Timestamp(first_date), pd.Timestamp(end_date),
    )
    with _data_cache_lock:
        _price_cache[key] = (now, closes, usdkrw)
    return closes, usdkrw


def _rebalance_trade_date(as_of: date) -> date:
    trade_date = first_xnys_session(as_of)
    if trade_date <= as_of:
        return trade_date
    previous_month = (pd.Timestamp(as_of).replace(day=1) - pd.Timedelta(days=1)).date()
    return first_xnys_session(previous_month)


def _load_rebalance_recommendation(
    as_of: date, force: bool = False,
) -> RebalanceRecommendation:
    trade_date = _rebalance_trade_date(as_of)
    key = trade_date.isoformat()
    now = time.monotonic()
    with _data_cache_lock:
        cached = _rebalance_cache.get(key)
        if cached and not force and now - cached[0] < REBALANCE_CACHE_SECONDS:
            return cached[1]
    portfolios = load_mailing_portfolios(
        Path(__file__).resolve().parents[2] / "config" / "mailing.json"
    )
    recommendations = generate_recommendations(portfolios, trade_date)
    if not recommendations:
        raise ValueError("리밸런싱 목표를 계산하지 못했습니다.")
    recommendation = recommendations[0]
    with _data_cache_lock:
        _rebalance_cache[key] = (time.monotonic(), recommendation)
    return recommendation


@ui.page("/accounts")
async def accounts_page() -> None:
    shell("accounts")
    with ui.column().classes("page-container w-full gap-4"):
        ui.button(
            "메인으로", icon="arrow_back", on_click=lambda: ui.navigate.to("/"),
        ).props("flat no-caps").classes("page-back-button")
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("투자 계좌 현황").classes("page-title")
                ui.label("‘계좌_’로 시작하는 Google Sheet 탭을 불러옵니다.").classes(
                    "page-subtitle"
                )
            refresh = ui.button(icon="refresh").props("round outline").tooltip(
                "Google Sheet 새로고침"
            )

        status = ui.label("계좌 목록을 불러오는 중입니다…").classes("status-text")
        account_area = ui.column().classes("w-full gap-3")

        async def render_accounts(force: bool = False) -> None:
            refresh.disable()
            status.set_text("Google Sheet에서 계좌 목록을 불러오는 중입니다…")
            account_area.clear()
            try:
                tabs = await run.io_bound(_load_account_tabs, force)
            except Exception as error:
                status.set_text("계좌 목록을 불러오지 못했습니다.")
                with account_area:
                    with ui.card().classes("error-card w-full"):
                        ui.label(str(error)).classes("text-negative")
                        ui.label(
                            "환경변수와 서비스 계정의 Google Sheet 공유 권한을 확인해 주세요."
                        ).classes("text-sm")
            else:
                status.set_text(f"계좌 {len(tabs)}개 · 5분간 캐시")
                with account_area:
                    if not tabs:
                        ui.label("‘계좌_’로 시작하는 탭이 없습니다.").classes("empty-state")
                    for tab_name in tabs:
                        with ui.card().classes("account-card w-full"):
                            with ui.row().classes("w-full items-center justify-between"):
                                with ui.row().classes("items-center gap-3"):
                                    ui.icon("account_balance").classes("text-primary text-2xl")
                                    with ui.column().classes("gap-0"):
                                        ui.label(_account_name(tab_name)).classes("font-bold")
                                        ui.label(tab_name).classes("text-xs text-grey-6")
                                ui.button("대시보드", icon="arrow_forward").props(
                                    "flat no-caps"
                                ).on(
                                    "click",
                                    lambda _, tab=tab_name: ui.navigate.to(
                                        f"/accounts/{quote(tab, safe='')}"
                                    ),
                                )
            finally:
                refresh.enable()

        refresh.on("click", lambda: render_accounts(True))
        await render_accounts()


def _percent(value: float | None) -> str:
    return "-" if value is None or not np.isfinite(value) else f"{value:.2%}"


def _money(value: float) -> str:
    return f"{value:,.0f}원"


def _currency_money(value: float, currency: str) -> str:
    return f"{value:,.0f}원" if currency == "KRW" else f"${value:,.2f}"


def _metric_card(label: str, value: str) -> None:
    with ui.card().classes("metric-card"):
        with ui.row().classes("metric-row w-full items-center justify-between no-wrap"):
            ui.label(label).classes("metric-label")
            ui.label(value).classes("metric-value")


def _render_currency_asset_charts(data: AccountDashboardData) -> None:
    if not data.currency_assets:
        ui.label("선택한 기간에 보유한 원화·달러 자산이 없습니다.").classes("empty-state")
        return
    for currency in ("KRW", "USD"):
        metrics = data.currency_assets.get(currency)
        if metrics is None or metrics.history.empty:
            continue
        history = metrics.history
        latest = history.iloc[-1]
        cost = float(latest.get("매입원가", 0.0))
        market = float(latest.get("시장평가액", 0.0))
        profit = market - cost
        unrealized_return = profit / cost if cost > 1e-10 else None
        label = "원화 자산" if currency == "KRW" else "달러 자산"
        with ui.card().classes("currency-asset-card w-full"):
            ui.label(label).classes("section-title")
            ui.label(
                "보유 종목의 매입원가와 일별 종가 기준 평가액입니다. 현금은 포함하지 않습니다."
            ).classes("text-xs text-grey-6")
            dates = [timestamp.strftime("%Y-%m-%d") for timestamp in history.index]
            ui.echart({
                "tooltip": {"trigger": "axis"},
                "legend": {"bottom": 0, "data": ["매입원가", "시장평가액"]},
                "grid": {"left": 68, "right": 28, "top": 30, "bottom": 58},
                "xAxis": {
                    "type": "category", "data": dates,
                    "axisLabel": {"hideOverlap": True},
                },
                "yAxis": {
                    "type": "value", "name": currency, "scale": True,
                    "axisLabel": {"formatter": "{value}"},
                },
                "dataZoom": [{"type": "inside"}],
                "series": [
                    {
                        "name": "매입원가", "type": "line", "showSymbol": False,
                        "data": history["매입원가"].fillna(0.0).round(2).tolist(),
                        "lineStyle": {"width": 2, "type": "dashed"},
                    },
                    {
                        "name": "시장평가액", "type": "line", "showSymbol": False,
                        "data": history["시장평가액"].fillna(0.0).round(2).tolist(),
                        "lineStyle": {"width": 3},
                        "areaStyle": {"opacity": .08},
                    },
                ],
            }).classes("dashboard-chart w-full")
            cards = (
                ("현재 매입원가", _currency_money(cost, currency)),
                ("현재 평가액", _currency_money(market, currency)),
                ("평가손익", _currency_money(profit, currency)),
                ("평가수익률", _percent(unrealized_return)),
                ("기간 수익률(TWR)", _percent(metrics.period_return)),
                ("최대 낙폭(MDD)", _percent(metrics.max_drawdown)),
            )
            with ui.grid(columns=3).classes("currency-metric-grid w-full gap-3"):
                for metric_label, value in cards:
                    _metric_card(metric_label, value)


def _render_account_metrics(data: AccountDashboardData) -> None:
    valuation = data.valuation
    performance = data.performance
    metrics = (
        ("총 평가액", _money(valuation.total_value)),
        ("현금", _money(valuation.cash)),
        ("보유자산", _money(valuation.holdings_value)),
        ("누적 평가손익", _money(valuation.profit)),
        ("누적 입금", _money(data.cumulative_deposits)),
        ("누적 출금", _money(data.cumulative_withdrawals)),
        ("누적 단순수익률", _percent(valuation.return_rate if data.performance_valid else None)),
        ("기간 수익률(TWR)", _percent(performance.period_return if data.performance_valid else None)),
        ("연환산 수익률", _percent(performance.annualized_return if data.performance_valid else None)),
        ("최대 낙폭(MDD)", _percent(performance.max_drawdown if data.performance_valid else None)),
        ("현재 낙폭", _percent(performance.current_drawdown if data.performance_valid else None)),
        ("기간 순입출금", _money(data.period_deposits - data.period_withdrawals)),
    )
    ui.label("계좌 전체 지표").classes("section-title")
    with ui.grid(columns=4).classes("metric-grid w-full gap-3"):
        for label, value in metrics:
            _metric_card(label, value)
    ui.label(
        "통화별·계좌 전체 기간수익률과 낙폭은 매수·매도 및 입출금 영향을 제거한 "
        "시간가중수익률(TWR) 기준입니다."
    ).classes("text-xs text-grey-6")


def _render_return_chart(data: AccountDashboardData) -> None:
    history = data.performance.history
    if not data.performance_valid:
        ui.label("입금 내역을 보완하면 수익률·낙폭 차트가 표시됩니다.").classes(
            "empty-state"
        )
        return
    if history.empty:
        ui.label("표시할 수익률 데이터가 없습니다.").classes("empty-state")
        return
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in history.index]
    ui.echart({
        "tooltip": {"trigger": "axis", "valueFormatter": "(value) => `${value.toFixed(2)}%`"},
        "legend": {"bottom": 0, "data": ["누적수익률", "낙폭"]},
        "grid": {"left": 56, "right": 28, "top": 28, "bottom": 58},
        "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "{value}%"}, "scale": True},
        "dataZoom": [{"type": "inside"}],
        "series": [
            {"name": "누적수익률", "type": "line", "showSymbol": False,
             "data": (history["누적수익률"] * 100).round(3).tolist(),
             "lineStyle": {"width": 3}},
            {"name": "낙폭", "type": "line", "showSymbol": False,
             "data": (history["낙폭"] * 100).round(3).tolist(),
             "lineStyle": {"width": 2}, "areaStyle": {"opacity": .12}},
        ],
    }).classes("dashboard-chart w-full")


def _table_columns(names: list[str]) -> list[dict[str, object]]:
    return [
        {"name": name, "label": name, "field": name, "align": "left", "sortable": True}
        for name in names
    ]


def _render_dashboard_content(account_tab: str, data: AccountDashboardData) -> None:
    valuation = data.valuation
    performance = data.performance
    if not data.performance_valid:
        with ui.card().classes("warning-card w-full"):
            ui.label("입금 내역을 확인해 주세요").classes("font-bold text-warning")
            ui.label(
                "현금 잔액이 음수입니다. 매수 전에 필요한 입금이 모두 기록될 때까지 "
                "수익률과 낙폭을 숨깁니다."
            ).classes("text-sm")

    ui.label(f"{_account_name(account_tab)} · {data.end_date:%Y-%m-%d} 기준").classes(
        "section-title"
    )
    with ui.tabs().classes("w-full dashboard-tabs") as tabs:
        overview_tab = ui.tab("평가액", icon="analytics")
        return_tab = ui.tab("수익률", icon="show_chart")
        holdings_tab = ui.tab("보유종목", icon="account_balance_wallet")
        ledger_tab = ui.tab("투자이력", icon="receipt_long")
        rebalance_tab = ui.tab("목표 비교", icon="compare_arrows")
    with ui.tab_panels(tabs, value=overview_tab).classes("w-full dashboard-panels"):
        with ui.tab_panel(overview_tab):
            _render_currency_asset_charts(data)
            _render_account_metrics(data)
        with ui.tab_panel(return_tab):
            _render_return_chart(data)
        with ui.tab_panel(holdings_tab):
            if valuation.holdings.empty:
                ui.label("선택한 종료일에 보유 중인 종목이 없습니다.").classes("empty-state")
            else:
                holdings = valuation.holdings.copy()
                holdings["수량"] = holdings["수량"].map(lambda value: f"{value:,.4f}")
                holdings["현재가"] = holdings["현재가"].map(lambda value: f"{value:,.2f}")
                holdings["환율"] = holdings["환율"].map(lambda value: f"{value:,.2f}")
                holdings["평가금액"] = holdings["평가금액"].map(_money)
                rows = holdings.fillna("").astype(str).to_dict("records")
                ui.table(
                    columns=_table_columns(list(holdings.columns)), rows=rows,
                    pagination={"rowsPerPage": 10},
                ).props("flat bordered").classes("w-full account-table")
        with ui.tab_panel(ledger_tab):
            if data.ledger.empty:
                ui.label("조회기간의 거래내역이 없습니다.").classes("empty-state")
            else:
                ledger = data.ledger.fillna("").astype(str)
                ui.table(
                    columns=_table_columns(list(ledger.columns)),
                    rows=ledger.to_dict("records"), pagination={"rowsPerPage": 15},
                ).props("flat bordered dense").classes("w-full account-table")
                ui.button(
                    "조회기간 CSV 다운로드", icon="download",
                    on_click=lambda: ui.download.content(
                        data.ledger.to_csv(index=False).encode("utf-8-sig"),
                        filename=f"{account_tab}_{data.start_date}_{data.end_date}.csv",
                        media_type="text/csv",
                    ),
                ).props("outline no-caps")
            ui.label(
                "거래내역 추가·수정·삭제는 Google Sheet에서 진행한 뒤 새로고침하세요."
            ).classes("text-xs text-grey-6")
        with ui.tab_panel(rebalance_tab):
            with ui.card().classes("migration-note w-full"):
                ui.label("월간 목표 대비 현재 계좌").classes("font-bold")
                ui.label(
                    "메일링과 동일한 프리셋 3 계산을 필요한 경우에만 실행합니다. "
                    "계좌 첫 화면과 다른 탭의 속도에는 영향을 주지 않습니다."
                ).classes("text-sm text-grey-7")
                with ui.row().classes("items-center gap-2"):
                    target_button = ui.button(
                        "목표 불러오기", icon="sync_alt",
                    ).classes("primary-action")
                    target_refresh = ui.button(icon="refresh").props("round outline").tooltip(
                        "목표 데이터 강제 새로고침"
                    )
                target_status = ui.label(
                    "버튼을 누르면 선택한 종료일이 속한 달의 목표와 비교합니다."
                ).classes("status-text")
                target_area = ui.column().classes("w-full gap-3")

                async def render_rebalance(force: bool = False) -> None:
                    target_button.disable()
                    target_refresh.disable()
                    target_status.set_text("메일링과 동일한 목표를 계산하고 있습니다…")
                    target_area.clear()
                    try:
                        recommendation = await run.io_bound(
                            _load_rebalance_recommendation, data.end_date, force,
                        )
                        comparison = await run.io_bound(
                            build_rebalance_comparison,
                            valuation.holdings,
                            cash=valuation.cash,
                            total_value=valuation.total_value,
                            target_selection=recommendation.domestic_selection,
                        )
                    except Exception as error:
                        target_status.set_text("리밸런싱 목표를 불러오지 못했습니다.")
                        with target_area:
                            ui.label(str(error)).classes("text-negative")
                            ui.label(
                                "미국·한국 거래일 데이터와 최신 시장자료를 확인해 주세요."
                            ).classes("text-xs text-grey-6")
                    else:
                        target_status.set_text(
                            f"1시간 캐시 · 판단 기준일 {recommendation.decision_date:%Y-%m-%d}"
                        )
                        with target_area:
                            with ui.grid(columns=3).classes("rebalance-summary w-full gap-3"):
                                _metric_card("미국 목표 매수일", f"{recommendation.trade_date:%Y-%m-%d}")
                                _metric_card("국내 제안일", f"{recommendation.domestic_trade_date:%Y-%m-%d}")
                                _metric_card("비교 계좌 평가액", _money(valuation.total_value))
                            ui.label(recommendation.title).classes("font-bold")
                            ui.label(
                                f"미국 목표 · {recommendation.selection}"
                            ).classes("text-sm text-grey-7")
                            ui.label(
                                f"국내 대체 목표 · {recommendation.domestic_selection}"
                            ).classes("text-sm text-grey-7")
                            display = comparison.copy()
                            display["현재 평가금액"] = display["현재 평가금액"].map(_money)
                            display["현재 비중"] = display["현재 비중"].map(lambda value: f"{value:.2%}")
                            display["목표 비중"] = display["목표 비중"].map(lambda value: f"{value:.2%}")
                            display["목표 평가금액"] = display["목표 평가금액"].map(_money)
                            display["차이 금액"] = display["차이 금액"].map(
                                lambda value: f"{value:+,.0f}원"
                            )
                            _result_table(display, 20)
                            ui.label(
                                "차이 금액은 현재 평가액을 목표 비중에 맞추기 위한 예상 금액입니다. "
                                "수량은 실제 주문 시가가 확정된 뒤 계산해야 합니다."
                            ).classes("text-xs text-grey-6")
                            ui.button(
                                "비교표 CSV", icon="download",
                                on_click=lambda: ui.download.content(
                                    comparison.to_csv(index=False).encode("utf-8-sig"),
                                    filename=(
                                        f"rebalance_{_account_name(account_tab)}_"
                                        f"{recommendation.trade_date}.csv"
                                    ),
                                    media_type="text/csv",
                                ),
                            ).props("outline no-caps")
                    finally:
                        target_button.enable()
                        target_refresh.enable()

                target_button.on("click", lambda: render_rebalance(False))
                target_refresh.on("click", lambda: render_rebalance(True))


@ui.page("/accounts/{account_tab}")
async def account_dashboard_page(account_tab: str) -> None:
    shell("accounts")
    if not account_tab.startswith(ACCOUNT_PREFIX):
        ui.navigate.to("/accounts")
        return
    with ui.column().classes("page-container w-full gap-4"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-2"):
                ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/accounts")).props(
                    "round flat"
                ).tooltip("계좌 목록")
                with ui.column().classes("gap-0"):
                    ui.label(_account_name(account_tab)).classes("page-title")
                    ui.label("Google Sheet 읽기 전용 계좌 대시보드").classes("page-subtitle")
            with ui.row().classes("gap-2"):
                sheet_refresh = ui.button(icon="sync").props("round outline").tooltip(
                    "Google Sheet 새로고침"
                )
                price_refresh = ui.button(icon="currency_exchange").props(
                    "round outline"
                ).tooltip("시세 새로고침")

        status = ui.label("계좌 내역을 불러오는 중입니다…").classes("status-text")
        controls = ui.row().classes("w-full gap-3 items-end")
        content = ui.column().classes("w-full gap-4")
        state: dict[str, object] = {"records": [], "bounds": None}

        async def load_records(force: bool = False) -> bool:
            try:
                records = await run.io_bound(_load_transactions, account_tab, force)
                bounds = account_date_bounds(
                    records, today=datetime.now(ZoneInfo("Asia/Seoul")).date(),
                )
            except Exception as error:
                content.clear()
                status.set_text("계좌 내역을 불러오지 못했습니다.")
                with content:
                    with ui.card().classes("error-card w-full"):
                        ui.label(str(error)).classes("text-negative")
                return False
            state["records"] = records
            state["bounds"] = bounds
            return True

        if not await load_records():
            return
        first_date, today = state["bounds"]  # type: ignore[misc]
        default_start = max(
            first_date,
            (pd.Timestamp(today) - pd.DateOffset(years=5)).date(),
        )
        with controls:
            start_input = ui.input("조회 시작일", value=default_start.isoformat()).props(
                f"type=date min={first_date.isoformat()} max={today.isoformat()}"
            ).classes("date-input")
            end_input = ui.input("조회 종료일", value=today.isoformat()).props(
                f"type=date min={first_date.isoformat()} max={today.isoformat()}"
            ).classes("date-input")
            apply_button = ui.button("조회", icon="search").classes("primary-action")

        async def render(force_sheet: bool = False, force_prices: bool = False) -> None:
            for button in (sheet_refresh, price_refresh, apply_button):
                button.disable()
            content.clear()
            status.set_text("계좌와 시세 데이터를 계산하는 중입니다…")
            try:
                if force_sheet and not await load_records(True):
                    return
                records = state["records"]
                first, latest = account_date_bounds(  # type: ignore[arg-type]
                    records, today=datetime.now(ZoneInfo("Asia/Seoul")).date(),
                )
                start = date.fromisoformat(str(start_input.value))
                end = date.fromisoformat(str(end_input.value))
                if end > latest:
                    end = latest
                    end_input.value = end.isoformat()
                frame = active_account_frame(records)  # type: ignore[arg-type]
                symbols = required_price_symbols(frame)
                closes, usdkrw = await run.io_bound(
                    _load_prices, symbols, first, end, force_prices,
                )
                dashboard = await run.io_bound(
                    build_account_dashboard, records,
                    start_date=start, end_date=end, closes=closes, usdkrw=usdkrw,
                )
            except Exception as error:
                status.set_text("대시보드를 계산하지 못했습니다.")
                with content:
                    with ui.card().classes("error-card w-full"):
                        ui.label(str(error)).classes("text-negative")
            else:
                status.set_text(
                    f"거래내역 1분 · 시세 15분 캐시 · {dashboard.end_date:%Y-%m-%d} 기준"
                )
                with content:
                    _render_dashboard_content(account_tab, dashboard)
            finally:
                for button in (sheet_refresh, price_refresh, apply_button):
                    button.enable()

        apply_button.on("click", lambda: render())
        sheet_refresh.on("click", lambda: render(force_sheet=True))
        price_refresh.on("click", lambda: render(force_prices=True))
        await render()

