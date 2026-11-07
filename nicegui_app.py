"""NiceGUI frontend introduced alongside the existing Streamlit app."""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nicegui import app, run, ui

from snowball.account_store import ACCOUNT_PREFIX, GoogleSheetAccountStore
from snowball.account_dashboard import (
    AccountDashboardData,
    account_date_bounds,
    active_account_frame,
    build_account_dashboard,
)
from snowball.accounting import download_unadjusted_closes, required_price_symbols
from snowball.builtin_library import (
    BUILTIN_PRESET_SPECS,
    get_builtin_spec,
    run_builtin_backtest,
    run_user_backtest,
)
from snowball.core.catalog import US_ETFS
from snowball.exports import diagnostic_frames, diagnostic_zip, market_data_export
from snowball.mailing import (
    RebalanceRecommendation,
    first_xnys_session,
    generate_recommendations,
    load_mailing_portfolios,
)
from snowball.presets import MomentumSpec, SpreadRule
from snowball.rebalance_compare import build_rebalance_comparison
from snowball.settings import google_sheet_settings
from snowball.user_presets import (
    UserPreset,
    export_user_presets,
    import_user_presets,
)


APP_TITLE = "Snowball Portfolio Lab"
LOCAL_PRESET_STORAGE_KEY = "snowball_user_presets_v1"
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


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    """Lightweight readiness endpoint for container platforms."""
    return {"status": "ok"}


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
        Path(__file__).resolve().parent / "config" / "mailing.json"
    )
    recommendations = generate_recommendations(portfolios, trade_date)
    if not recommendations:
        raise ValueError("리밸런싱 목표를 계산하지 못했습니다.")
    recommendation = recommendations[0]
    with _data_cache_lock:
        _rebalance_cache[key] = (time.monotonic(), recommendation)
    return recommendation


def _shell(active: str = "") -> None:
    with ui.header().classes("app-header items-center justify-between px-4"):
        with ui.row().classes("items-center gap-3 cursor-pointer").on(
            "click", lambda: ui.navigate.to("/")
        ):
            ui.label("❄️").classes("text-2xl")
            ui.label(APP_TITLE).classes("text-lg font-bold")
        with ui.row().classes("items-center gap-1"):
            ui.button("홈", icon="home", on_click=lambda: ui.navigate.to("/")).props(
                "flat dense"
            ).classes("nav-button" + (" nav-active" if active == "home" else ""))
            ui.button(
                "계좌", icon="monitoring", on_click=lambda: ui.navigate.to("/accounts")
            ).props("flat dense").classes(
                "nav-button" + (" nav-active" if active == "accounts" else "")
            )


def _feature_card(
    icon: str,
    title: str,
    description: str,
    button_text: str,
    target: str,
    *,
    enabled: bool = True,
) -> None:
    with ui.card().classes("feature-card w-full"):
        with ui.row().classes("items-center gap-3"):
            ui.icon(icon).classes("feature-icon")
            ui.label(title).classes("text-xl font-bold")
        ui.label(description).classes("feature-description")
        button = ui.button(
            button_text,
            icon="arrow_forward",
            on_click=lambda: ui.navigate.to(target),
        ).classes("primary-action w-full")
        if not enabled:
            button.disable()
            ui.label("3단계에서 이전할 예정입니다.").classes("text-xs text-grey-6")


@ui.page("/")
def home_page() -> None:
    _shell("home")
    with ui.column().classes("page-container w-full gap-5"):
        ui.label("원하는 기능을 선택하세요").classes("page-title")
        ui.label(
            "계산 엔진은 그대로 유지하고 화면만 더 빠르고 안정적인 구조로 개편합니다."
        ).classes("page-subtitle")
        with ui.grid(columns=2).classes("feature-grid w-full gap-4"):
            _feature_card(
                "query_stats", "백테스팅",
                "동적자산배분 조건을 구성하고 장기 성과와 리밸런싱 이력을 검증합니다.",
                "백테스팅 열기", "/backtest",
            )
            _feature_card(
                "account_balance_wallet", "투자 계좌 현황",
                "Google Sheet의 계좌별 거래 원장을 읽어 평가액과 투자 성과를 확인합니다.",
                "계좌 현황 보기", "/accounts",
            )
        with ui.card().classes("migration-note w-full"):
            ui.label("개편 진행 상황").classes("font-bold")
            ui.label("4단계 · 계좌 목표 대비 리밸런싱 연결")
            ui.linear_progress(value=0.97).props("rounded color=primary")


@ui.page("/accounts")
async def accounts_page() -> None:
    _shell("accounts")
    with ui.column().classes("page-container w-full gap-4"):
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


def _metric_card(label: str, value: str) -> None:
    with ui.card().classes("metric-card"):
        ui.label(label).classes("metric-label")
        ui.label(value).classes("metric-value")


def _render_value_chart(data: AccountDashboardData) -> None:
    history = data.performance.history
    if history.empty:
        ui.label("표시할 평가액 데이터가 없습니다.").classes("empty-state")
        return
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in history.index]
    ui.echart({
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0, "data": ["총평가액", "보유자산", "입출금"]},
        "grid": {"left": 64, "right": 32, "top": 30, "bottom": 58},
        "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
        "yAxis": [
            {"type": "value", "name": "평가액(원)", "scale": True},
            {"type": "value", "name": "입출금(원)", "scale": True},
        ],
        "dataZoom": [{"type": "inside"}],
        "series": [
            {"name": "총평가액", "type": "line", "showSymbol": False,
             "data": history["총평가액"].round(0).tolist(), "lineStyle": {"width": 3}},
            {"name": "보유자산", "type": "line", "showSymbol": False,
             "data": history["보유자산"].round(0).tolist(), "lineStyle": {"width": 2}},
            {"name": "입출금", "type": "bar", "yAxisIndex": 1,
             "data": history["입출금"].round(0).tolist(), "itemStyle": {"opacity": .55}},
        ],
    }).classes("dashboard-chart w-full")


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
    with ui.grid(columns=4).classes("metric-grid w-full gap-3"):
        for label, value in metrics:
            _metric_card(label, value)
    ui.label(
        "기간수익률과 낙폭은 입출금 영향을 제거한 시간가중수익률(TWR) 기준입니다."
    ).classes("text-xs text-grey-6")

    with ui.tabs().classes("w-full dashboard-tabs") as tabs:
        overview_tab = ui.tab("평가액", icon="monitoring")
        return_tab = ui.tab("수익률", icon="show_chart")
        holdings_tab = ui.tab("보유종목", icon="account_balance_wallet")
        ledger_tab = ui.tab("투자이력", icon="receipt_long")
        rebalance_tab = ui.tab("리밸런싱", icon="notifications")
    with ui.tab_panels(tabs, value=overview_tab).classes("w-full dashboard-panels"):
        with ui.tab_panel(overview_tab):
            _render_value_chart(data)
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
    _shell("accounts")
    if not account_tab.startswith(ACCOUNT_PREFIX):
        ui.navigate.to("/accounts")
        return
    with ui.column().classes("page-container w-full gap-4"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-2"):
                ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/accounts")).props(
                    "round flat"
                )
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
        with controls:
            start_input = ui.input("조회 시작일", value=first_date.isoformat()).props(
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


USER_PRESET_PREFIX = "사용자::"
MOMENTUM_KINDS = ["수익률", "이동평균", "모멘텀 스코어"]
SCORE_TYPES = [
    "1-3-6개월 단순", "3-6-12개월 단순", "1-3-6-12개월 단순",
    "1-3-6-9-12개월 단순", "1-3-6-12개월 가중",
]
ATTACK_MODES = [
    "상대 모멘텀", "절대 모멘텀", "절대 모멘텀 + 방어 스위치",
    "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 방어 스위치",
]
DEFENSE_MODES = [
    "전체 동일 비중", "상대 모멘텀", "절대 모멘텀",
    "상대 & 절대 모멘텀 + 현금 스위치",
]
WEIGHT_MODES = ["고정 범위", "마이너스 절대 모멘텀 수"]
MARKET_SIGNALS = [
    "장단기 금리차 역전 (10년-2년)",
    "장단기 금리차 역전 (10년-3개월)",
    "S&P500 배당수익률",
]


def _load_browser_presets() -> dict[str, UserPreset]:
    raw = str(app.storage.user.get("user_presets_json", "")).strip()
    if not raw:
        return {}
    try:
        records = import_user_presets(raw)
    except ValueError:
        return {}
    return {record.title: record for record in records}


def _save_browser_presets(records: dict[str, UserPreset]) -> None:
    app.storage.user["user_presets_json"] = export_user_presets(
        records.values()
    ).decode("utf-8")


async def _persist_browser_presets(records: dict[str, UserPreset]) -> None:
    """Persist on the server and in durable browser localStorage."""
    payload = export_user_presets(records.values()).decode("utf-8")
    _save_browser_presets(records)
    await ui.run_javascript(
        f"localStorage.setItem({json.dumps(LOCAL_PRESET_STORAGE_KEY)}, "
        f"{json.dumps(payload, ensure_ascii=False)});"
    )


class _StrategyEditor:
    def __init__(self, record: UserPreset):
        self.base = record.strategy
        self.title = ui.input("프리셋 제목", value=record.title).props("outlined").classes("w-full")
        self.domestic = ui.checkbox("국내 ETF로 실행", value=record.domestic_execution)
        assets = sorted(US_ETFS)
        with ui.card().classes("editor-section w-full"):
            ui.label("투자자산").classes("font-bold")
            self.attack_tickers = ui.select(
                assets, value=list(self.base.attack_tickers), label="투자자산 후보군",
                multiple=True,
            ).props("outlined use-chips options-dense").classes("w-full")
            with ui.row().classes("editor-fields w-full"):
                self.attack_mode = ui.select(
                    ATTACK_MODES, value=self.base.attack_mode, label="선택 방식",
                ).props("outlined").classes("control-grow")
                self.attack_count = ui.number(
                    "선택자산 수", value=self.base.attack_select_count, min=1, step=1,
                ).props("outlined").classes("control-grow")
            self.attack_kind, self.attack_months, self.attack_score = self._momentum(
                "투자 모멘텀", self.base.attack_relative,
            )
        with ui.card().classes("editor-section w-full"):
            ui.label("방어자산").classes("font-bold")
            self.defense_tickers = ui.select(
                assets, value=list(self.base.defense_tickers), label="방어자산 후보군",
                multiple=True,
            ).props("outlined use-chips options-dense").classes("w-full")
            with ui.row().classes("editor-fields w-full"):
                self.defense_mode = ui.select(
                    DEFENSE_MODES, value=self.base.defense_mode, label="선택 방식",
                ).props("outlined").classes("control-grow")
                self.defense_count = ui.number(
                    "선택자산 수", value=self.base.defense_select_count, min=1, step=1,
                ).props("outlined").classes("control-grow")
            self.defense_same = ui.checkbox(
                "투자자산과 동일 모멘텀 사용", value=self.base.defense_same_momentum,
            )
            self.defense_kind, self.defense_months, self.defense_score = self._momentum(
                "방어 모멘텀", self.base.defense_relative,
            )
        with ui.card().classes("editor-section w-full"):
            ui.label("비중 및 위험회피").classes("font-bold")
            with ui.row().classes("editor-fields w-full"):
                self.weight_mode = ui.select(
                    WEIGHT_MODES, value=self.base.weight_mode, label="비중 방식",
                ).props("outlined").classes("control-grow")
                self.fixed_defense = ui.number(
                    "고정 방어 비중(%)", value=self.base.fixed_defense_weight * 100,
                    min=0, max=100, step=5,
                ).props("outlined").classes("control-grow")
                self.negative_threshold = ui.number(
                    "전체 방어 전환 기준", value=self.base.negative_threshold,
                    min=1, step=1,
                ).props("outlined").classes("control-grow")
            self.canary_tickers = ui.select(
                assets, value=list(self.base.canary_tickers), label="카나리아 ETF",
                multiple=True,
            ).props("outlined use-chips options-dense").classes("w-full")
            self.canary_kind, self.canary_months, self.canary_score = self._momentum(
                "카나리아 모멘텀", self.base.canary_momentum,
            )
            with ui.row().classes("editor-fields w-full"):
                self.canary_apply = ui.select(
                    ["방어형", "공격형"], value=self.base.canary_apply_mode,
                    label="카나리아 적용 방식",
                ).props("outlined").classes("control-grow")
                self.canary_direction = ui.select(
                    ["마이너스(-)", "플러스(+)"], value=self.base.canary_direction,
                    label="위험 신호 방향",
                ).props("outlined").classes("control-grow")
                rounding = (
                    "원본값" if self.base.canary_decision_percent_digits is None
                    else str(self.base.canary_decision_percent_digits)
                )
                self.canary_rounding = ui.select(
                    ["원본값", "0", "1", "2", "3", "4", "5", "6"],
                    value=rounding, label="판정 반올림 자릿수",
                ).props("outlined").classes("control-grow")
        with ui.card().classes("editor-section w-full"):
            ui.label("시장 위험 신호").classes("font-bold")
            self.market_signals = ui.select(
                MARKET_SIGNALS, value=list(self.base.market_signals), label="활성 신호",
                multiple=True,
            ).props("outlined use-chips options-dense").classes("w-full")
            self.market_combine = ui.radio(
                ["OR", "AND"], value=self.base.market_combine,
            ).props("inline")
            with ui.row().classes("editor-fields w-full"):
                rule_2y = self.base.spread_10y2y or SpreadRule(7, 8, -0.5)
                self.spread_2y_lag = ui.number("10Y-2Y 지연(개월)", value=rule_2y.lag_months, min=0, max=24).props("outlined").classes("control-grow")
                self.spread_2y_active = ui.number("10Y-2Y 반영 기간", value=rule_2y.active_months, min=1, max=24).props("outlined").classes("control-grow")
                self.spread_2y_threshold = ui.number("10Y-2Y 역전값(%p)", value=rule_2y.threshold, step=.1).props("outlined").classes("control-grow")
            with ui.row().classes("editor-fields w-full"):
                rule = self.base.spread_10y3m or SpreadRule(7, 8, -0.5)
                self.spread_lag = ui.number("10Y-3M 지연(개월)", value=rule.lag_months, min=0, max=24).props("outlined").classes("control-grow")
                self.spread_active = ui.number("10Y-3M 반영 기간", value=rule.active_months, min=1, max=24).props("outlined").classes("control-grow")
                self.spread_threshold = ui.number("10Y-3M 역전값(%p)", value=rule.threshold, step=.1).props("outlined").classes("control-grow")
                self.dividend = ui.number("배당수익률 기준(%)", value=self.base.dividend_threshold, min=0, max=10, step=.01).props("outlined").classes("control-grow")

    @staticmethod
    def _momentum(label: str, value: MomentumSpec):
        with ui.row().classes("editor-fields w-full"):
            kind = ui.select(MOMENTUM_KINDS, value=value.kind, label=f"{label} 기준").props("outlined").classes("control-grow")
            months = ui.number("기간(개월)", value=value.months, min=1, max=24, step=1).props("outlined").classes("control-grow")
            score = ui.select(SCORE_TYPES, value=value.score_type, label="스코어 구성").props("outlined").classes("control-grow")
        return kind, months, score

    @staticmethod
    def _spec(kind, months, score) -> MomentumSpec:
        return MomentumSpec(str(kind.value), int(months.value or 1), str(score.value))

    def record(self) -> UserPreset:
        title = str(self.title.value or "").strip()
        attacks = tuple(self.attack_tickers.value or ())
        defenses = tuple(self.defense_tickers.value or ())
        canaries = tuple(self.canary_tickers.value or ())
        signals = tuple(self.market_signals.value or ())
        if not title:
            raise ValueError("프리셋 제목을 입력해 주세요.")
        if title in BUILTIN_PRESET_SPECS:
            raise ValueError("기본 제공 프리셋과 같은 제목은 사용할 수 없습니다.")
        if not attacks or not defenses:
            raise ValueError("투자자산과 방어자산 후보를 하나 이상 선택하세요.")
        attack_momentum = self._spec(self.attack_kind, self.attack_months, self.attack_score)
        defense_momentum = self._spec(self.defense_kind, self.defense_months, self.defense_score)
        canary_momentum = self._spec(self.canary_kind, self.canary_months, self.canary_score)
        rounding_value = str(self.canary_rounding.value)
        strategy = replace(
            self.base, name=title, attack_tickers=attacks, defense_tickers=defenses,
            attack_mode=str(self.attack_mode.value),
            attack_select_count=min(int(self.attack_count.value or 1), len(attacks)),
            attack_relative=attack_momentum, attack_absolute=attack_momentum,
            defense_mode=str(self.defense_mode.value),
            defense_select_count=min(int(self.defense_count.value or 1), len(defenses)),
            defense_relative=defense_momentum, defense_absolute=defense_momentum,
            defense_same_momentum=bool(self.defense_same.value),
            weight_mode=str(self.weight_mode.value),
            fixed_defense_weight=float(self.fixed_defense.value or 0) / 100,
            negative_threshold=min(int(self.negative_threshold.value or 1), len(attacks)),
            canary_tickers=canaries, canary_momentum=canary_momentum,
            canary_apply_mode=str(self.canary_apply.value),
            canary_direction=str(self.canary_direction.value),
            canary_decision_percent_digits=(
                None if rounding_value == "원본값" else int(rounding_value)
            ),
            market_combine=str(self.market_combine.value), market_signals=signals,
            spread_10y2y=(
                SpreadRule(
                    int(self.spread_2y_lag.value or 0), int(self.spread_2y_active.value or 1),
                    float(self.spread_2y_threshold.value or 0),
                ) if "장단기 금리차 역전 (10년-2년)" in signals else self.base.spread_10y2y
            ),
            spread_10y3m=(
                SpreadRule(
                    int(self.spread_lag.value or 0), int(self.spread_active.value or 1),
                    float(self.spread_threshold.value or 0),
                ) if "장단기 금리차 역전 (10년-3개월)" in signals else self.base.spread_10y3m
            ),
            dividend_threshold=float(self.dividend.value or 0),
        )
        record = UserPreset(title, strategy, bool(self.domestic.value))
        record.validate()
        return record


def _result_table(frame: pd.DataFrame, rows_per_page: int = 12) -> None:
    if frame is None or frame.empty:
        ui.label("표시할 자료가 없습니다.").classes("empty-state")
        return
    visible = frame.copy().fillna("")
    for column in visible.columns:
        visible[column] = visible[column].astype(str)
    ui.table(
        columns=_table_columns(list(visible.columns)), rows=visible.to_dict("records"),
        pagination={"rowsPerPage": rows_per_page},
    ).props("flat bordered dense").classes("w-full account-table")


def _monthly_result_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    months = frame.copy()
    periods = pd.PeriodIndex(months["적용 월"], freq="M")
    months["연도"] = periods.year
    months["월"] = periods.month
    pivot = months.pivot(index="연도", columns="월", values="수익률 원시값")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [f"{month}월" for month in pivot.columns]
    pivot = pivot.reset_index()
    for column in pivot.columns[1:]:
        pivot[column] = pivot[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.2%}"
        )
    return pivot


def _render_backtest_result(result: dict, bundle) -> None:
    metrics = result["metrics"]
    metric_values = (
        ("누적 수익률", _percent(metrics.get("total_return"))),
        ("연환산 수익률", _percent(metrics.get("cagr"))),
        ("연 변동성", _percent(metrics.get("annual_volatility"))),
        ("최대 낙폭", _percent(metrics.get("max_drawdown"))),
        ("연 수익률 표준편차", _percent(metrics.get("annual_return_std"))),
        ("최악의 해", _percent(metrics.get("worst_year"))),
    )
    exports = diagnostic_frames(result)
    ui.label(f"{result['preset_name']} 결과").classes("section-title")
    ui.label(
        f"공통 평가기간 · {pd.Timestamp(result['common_start']).date()} ~ "
        f"{pd.Timestamp(result['common_end']).date()} · "
        f"연간 턴오버 {float(result.get('annual_turnover', 0)):.2%}"
    ).classes("text-sm text-grey-6")
    with ui.grid(columns=3).classes("result-metric-grid w-full gap-3"):
        for label, value in metric_values:
            _metric_card(label, value)
    ui.button(
        "비교 진단자료 ZIP 다운로드", icon="folder_zip",
        on_click=lambda: ui.download.content(
            diagnostic_zip(result),
            filename=(
                f"snowball_diagnostics_{pd.Timestamp(result['common_start']).date()}_"
                f"{pd.Timestamp(result['common_end']).date()}.zip"
            ),
            media_type="application/zip",
        ),
    ).props("outline no-caps").classes("diagnostic-download")

    nav = result["nav"].dropna().astype(float)
    chart = pd.DataFrame({"포트폴리오": nav / nav.iloc[0] - 1.0})
    benchmark = result.get("benchmark_ticker", "QQQ")
    if benchmark in bundle.adjusted_close:
        comparison = bundle.adjusted_close[benchmark].reindex(nav.index).ffill().dropna()
        if not comparison.empty:
            comparison = comparison / comparison.iloc[0] - 1.0
            chart["국내 나스닥100" if benchmark.endswith(".KS") else "QQQ"] = comparison
    dates = [value.strftime("%Y-%m-%d") for value in chart.index]
    series = [
        {
            "name": column, "type": "line", "showSymbol": False,
            "data": (chart[column] * 100).round(4).tolist(),
            "lineStyle": {"width": 3 if column == "포트폴리오" else 2},
        }
        for column in chart.columns
    ]
    with ui.tabs().classes("w-full dashboard-tabs") as tabs:
        returns_tab = ui.tab("수익률 차트", icon="show_chart")
        drawdown_tab = ui.tab("손실폭", icon="trending_down")
        monthly_tab = ui.tab("월별 수익률", icon="calendar_month")
        history_tab = ui.tab("리밸런싱", icon="event_repeat")
        decisions_tab = ui.tab("판단 원시값", icon="fact_check")
        snapshot_tab = ui.tab("데이터", icon="database")
        settings_tab = ui.tab("적용 설정", icon="settings")
    with ui.tab_panels(tabs, value=returns_tab).classes("w-full dashboard-panels"):
        with ui.tab_panel(returns_tab):
            ui.echart({
                "tooltip": {"trigger": "axis"},
                "legend": {"bottom": 0, "data": list(chart.columns)},
                "grid": {"left": 58, "right": 24, "top": 24, "bottom": 58},
                "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
                "yAxis": {"type": "value", "scale": True, "axisLabel": {"formatter": "{value}%"}},
                "dataZoom": [{"type": "inside"}], "series": series,
            }).classes("dashboard-chart w-full")
        with ui.tab_panel(drawdown_tab):
            drawdown = result["drawdown"].reindex(nav.index).fillna(0.0) * 100
            ui.echart({
                "tooltip": {"trigger": "axis"},
                "grid": {"left": 54, "right": 24, "top": 24, "bottom": 48},
                "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
                "yAxis": {"type": "value", "axisLabel": {"formatter": "{value}%"}},
                "dataZoom": [{"type": "inside"}],
                "series": [{
                    "name": "손실폭", "type": "line", "showSymbol": False,
                    "data": drawdown.round(3).tolist(), "areaStyle": {"opacity": .18},
                }],
            }).classes("dashboard-chart w-full")
        with ui.tab_panel(monthly_tab):
            _result_table(_monthly_result_table(exports["01_monthly_returns.csv"]), 20)
            ui.button(
                "월별 수익률 CSV", icon="download",
                on_click=lambda: ui.download.content(
                    exports["01_monthly_returns.csv"].to_csv(index=False).encode("utf-8-sig"),
                    filename="monthly_returns.csv", media_type="text/csv",
                ),
            ).props("outline no-caps")
        with ui.tab_panel(history_tab):
            history = result["history"].copy()
            display = [
                column for column in (
                    "적용일", "판단 기준일", "선택된 종목", "턴오버", "거래비용", "평가금액"
                ) if column in history
            ]
            table = history[display].sort_values("적용일", ascending=False).copy()
            _result_table(table)
            ui.button(
                "리밸런싱 CSV", icon="download",
                on_click=lambda: ui.download.content(
                    exports["02_rebalancing.csv"].to_csv(index=False).encode("utf-8-sig"),
                    filename="rebalancing.csv", media_type="text/csv",
                ),
            ).props("outline no-caps")
        with ui.tab_panel(decisions_tab):
            keys = list(result.get("candidate_snapshots", {}))
            if not keys:
                ui.label("판단 원시값이 없습니다.").classes("empty-state")
            else:
                decision_date = ui.select(
                    keys, value=keys[-1], label="리밸런싱 적용일",
                ).props("outlined options-dense").classes("decision-select")
                decision_area = ui.column().classes("w-full gap-4")

                def render_decision() -> None:
                    decision_area.clear()
                    selected = str(decision_date.value)
                    with decision_area:
                        ui.label("자산 선택 계산값").classes("font-bold")
                        _result_table(result["candidate_snapshots"].get(selected, pd.DataFrame()))
                        canary = result.get("canary_audits", {}).get(selected, pd.DataFrame())
                        if not canary.empty:
                            ui.label("카나리아 원시 신호").classes("font-bold")
                            _result_table(canary)
                        market = result.get("market_audits", {}).get(selected, pd.DataFrame())
                        if not market.empty:
                            ui.label("시장 데이터 원시 신호").classes("font-bold")
                            _result_table(market)

                decision_date.on_value_change(lambda _: render_decision())
                render_decision()
                raw = pd.concat([
                    exports["03_asset_signals.csv"], exports["04_canary_signals.csv"],
                    exports["05_market_signals.csv"],
                ], ignore_index=True, sort=False)
                ui.button(
                    "전체 판단 원시값 CSV", icon="download",
                    on_click=lambda: ui.download.content(
                        raw.to_csv(index=False).encode("utf-8-sig"),
                        filename="decision_raw_values.csv", media_type="text/csv",
                    ),
                ).props("outline no-caps")
        with ui.tab_panel(snapshot_tab):
            metadata = result.get("data_snapshot", {})
            ui.label("이번 실행에 사용된 데이터 버전").classes("font-bold")
            ui.code(json.dumps(metadata, ensure_ascii=False, indent=2, default=str)).classes(
                "snapshot-code w-full"
            )
            market_snapshot = market_data_export(result)
            if not market_snapshot.empty:
                _result_table(market_snapshot, 15)
                ui.button(
                    "시장 데이터 CSV", icon="download",
                    on_click=lambda: ui.download.content(
                        market_snapshot.to_csv(index=False).encode("utf-8-sig"),
                        filename="market_data_snapshot.csv", media_type="text/csv",
                    ),
                ).props("outline no-caps")
            with ui.row().classes("w-full gap-3"):
                ui.button(
                    "수정종가 CSV", icon="download",
                    on_click=lambda: ui.download.content(
                        bundle.adjusted_close.rename_axis("Date").to_csv().encode("utf-8-sig"),
                        filename="adjusted_close.csv", media_type="text/csv",
                    ),
                ).props("outline no-caps")
                ui.button(
                    "수정시가 CSV", icon="download",
                    on_click=lambda: ui.download.content(
                        bundle.adjusted_open.rename_axis("Date").to_csv().encode("utf-8-sig"),
                        filename="adjusted_open.csv", media_type="text/csv",
                    ),
                ).props("outline no-caps")
        with ui.tab_panel(settings_tab):
            config = dict(result.get("engine_config", {}))
            config.pop("execution_prices", None)
            config.pop("_market_data", None)
            ui.code(json.dumps(config, ensure_ascii=False, indent=2, default=str)).classes(
                "snapshot-code w-full"
            )


@ui.page("/backtest")
def backtest_page() -> None:
    _shell()
    user_presets = _load_browser_presets()

    def preset_options() -> dict[str, str]:
        options = {title: title for title in BUILTIN_PRESET_SPECS}
        options.update({
            f"{USER_PRESET_PREFIX}{title}": f"사용자 · {title}"
            for title in sorted(user_presets)
        })
        return options

    with ui.column().classes("page-container w-full gap-4"):
        ui.label("백테스팅").classes("page-title")
        ui.label(
            "기본 프리셋을 복사해 조건을 편집하고, 나만의 프리셋으로 저장할 수 있습니다."
        ).classes("page-subtitle")
        with ui.card().classes("backtest-controls w-full"):
            preset = ui.select(
                preset_options(), value=next(iter(BUILTIN_PRESET_SPECS)),
                label="내 프리셋",
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
                weight = ui.number(
                    "프리셋 1 비중(%)", value=50, min=0, max=100, step=5,
                ).props("outlined").classes("control-grow")
            market_mode = ui.radio(
                {"fixed": "고정 스냅샷", "latest": "최신 데이터"},
                value="fixed",
            ).props("inline")
            run_button = ui.button("백테스트 실행", icon="play_arrow").classes(
                "primary-action w-full"
            )

        with ui.expansion(
            "전략 조건 편집", icon="tune", value=False,
        ).classes("strategy-expansion w-full"):
            editor_area = ui.column().classes("w-full gap-3")

        with ui.card().classes("preset-library w-full"):
            ui.label("사용자 프리셋 백업").classes("font-bold")
            ui.label(
                "이 브라우저에 자동 저장됩니다. 다른 기기나 재배포에 대비해 JSON도 보관하세요."
            ).classes("text-sm text-grey-6")
            with ui.row().classes("w-full items-start gap-3"):
                export_button = ui.button(
                    "JSON 다운로드", icon="download", color="secondary",
                ).props("outline")
                upload = ui.upload(
                    label="JSON 불러오기", auto_upload=True, max_files=1,
                    max_file_size=2_000_000,
                ).props("accept=.json flat bordered").classes("preset-upload")

        status = ui.label("조건을 확인한 뒤 실행하세요.").classes("status-text")
        result_area = ui.column().classes("w-full gap-4")
        editor_state: dict[str, _StrategyEditor | None] = {"editor": None}

        def refresh_preset_select(selected: str | None = None) -> None:
            preset.options = preset_options()
            if selected is not None:
                preset.value = selected
            preset.update()

        def selected_user_title() -> str | None:
            value = str(preset.value or "")
            return value.removeprefix(USER_PRESET_PREFIX) if value.startswith(USER_PRESET_PREFIX) else None

        def rebuild_editor() -> None:
            editor_area.clear()
            editor_state["editor"] = None
            value = str(preset.value or "")
            with editor_area:
                custom_title = selected_user_title()
                if custom_title is not None:
                    record = user_presets.get(custom_title)
                    if record is None:
                        ui.label("저장된 사용자 프리셋을 찾을 수 없습니다.").classes("text-negative")
                        return
                else:
                    spec = get_builtin_spec(value)
                    if spec.secondary is not None:
                        ui.label(
                            "결합 프리셋은 위의 혼합 비중만 편집합니다. "
                            "구성 전략은 개별 프리셋을 복사해 저장하세요."
                        ).classes("text-sm text-grey-6")
                        return
                    copied_title = f"{spec.title} 복사본"
                    record = UserPreset(
                        copied_title,
                        replace(spec.primary, name=copied_title),
                        spec.domestic_execution,
                    )
                editor_state["editor"] = _StrategyEditor(record)
                with ui.row().classes("w-full gap-3"):
                    ui.button(
                        "내 프리셋 저장", icon="save", on_click=save_current,
                    ).classes("primary-action")
                    if custom_title is not None:
                        ui.button(
                            "삭제", icon="delete", color="negative",
                            on_click=delete_current,
                        ).props("outline")

        async def save_current() -> None:
            editor = editor_state["editor"]
            if editor is None:
                ui.notify("저장할 개별 전략이 없습니다.", type="warning")
                return
            try:
                record = editor.record()
                old_title = selected_user_title()
                if old_title and old_title != record.title:
                    user_presets.pop(old_title, None)
                user_presets[record.title] = record
                await _persist_browser_presets(user_presets)
            except ValueError as error:
                ui.notify(str(error), type="negative")
                return
            key = f"{USER_PRESET_PREFIX}{record.title}"
            refresh_preset_select(key)
            rebuild_editor()
            ui.notify(f"'{record.title}' 프리셋을 저장했습니다.", type="positive")

        async def delete_current() -> None:
            title = selected_user_title()
            if title is None:
                return
            user_presets.pop(title, None)
            await _persist_browser_presets(user_presets)
            refresh_preset_select(next(iter(BUILTIN_PRESET_SPECS)))
            update_preset_controls()
            rebuild_editor()
            ui.notify(f"'{title}' 프리셋을 삭제했습니다.", type="positive")

        def export_library() -> None:
            if not user_presets:
                ui.notify("다운로드할 사용자 프리셋이 없습니다.", type="warning")
                return
            ui.download.content(
                export_user_presets(user_presets.values()),
                filename="snowball_user_presets.json",
                media_type="application/json",
            )

        async def import_library(event) -> None:
            try:
                imported = import_user_presets(event.content.read())
                collisions = [item.title for item in imported if item.title in BUILTIN_PRESET_SPECS]
                if collisions:
                    raise ValueError("기본 프리셋과 같은 제목은 불러올 수 없습니다.")
                user_presets.update({item.title: item for item in imported})
                await _persist_browser_presets(user_presets)
            except ValueError as error:
                ui.notify(str(error), type="negative")
                return
            finally:
                upload.reset()
            selected = f"{USER_PRESET_PREFIX}{imported[0].title}" if imported else None
            refresh_preset_select(selected)
            update_preset_controls()
            rebuild_editor()
            ui.notify(f"사용자 프리셋 {len(imported)}개를 불러왔습니다.", type="positive")

        async def restore_local_presets() -> None:
            """Restore after free-host restarts which erase server-side storage."""
            try:
                raw = await ui.run_javascript(
                    f"localStorage.getItem({json.dumps(LOCAL_PRESET_STORAGE_KEY)}) || ''"
                )
                if raw:
                    restored = import_user_presets(str(raw))
                    user_presets.update({item.title: item for item in restored})
                    _save_browser_presets(user_presets)
                elif user_presets:
                    await _persist_browser_presets(user_presets)
                else:
                    return
            except (ValueError, RuntimeError):
                ui.notify(
                    "브라우저 프리셋 자동 복원에 실패했습니다. JSON 백업을 이용해 주세요.",
                    type="warning",
                )
                return
            refresh_preset_select(str(preset.value or next(iter(BUILTIN_PRESET_SPECS))))
            rebuild_editor()

        def update_preset_controls() -> None:
            custom_title = selected_user_title()
            if custom_title is None:
                spec = get_builtin_spec(str(preset.value))
                is_domestic = spec.domestic_execution
                weight.set_visibility(spec.secondary is not None)
            else:
                record = user_presets.get(custom_title)
                is_domestic = bool(record and record.domestic_execution)
                weight.set_visibility(False)
            initial.value = 10_000_000 if is_domestic else 10_000
            initial.min = 100_000 if is_domestic else 100
            initial.update()

        def preset_changed() -> None:
            update_preset_controls()
            rebuild_editor()

        preset.on_value_change(lambda _: preset_changed())
        export_button.on("click", export_library)
        upload.on_upload(import_library)
        update_preset_controls()
        rebuild_editor()
        ui.timer(0.2, restore_local_presets, once=True)

        async def execute_backtest() -> None:
            run_button.disable()
            result_area.clear()
            status.set_text("가격과 시장 데이터를 불러와 계산하고 있습니다…")
            try:
                title = str(preset.value)
                common_args = {
                    "initial_investment": float(initial.value or 0),
                    "transaction_cost_rate": float(cost.value or 0) / 10_000.0,
                    "market_data_mode": str(market_mode.value),
                }
                start_date = date.fromisoformat(str(start_input.value))
                end_date = date.fromisoformat(str(end_input.value))
                custom_title = selected_user_title()
                editor = editor_state["editor"]
                if custom_title is not None or editor is not None:
                    if editor is None:
                        raise ValueError("실행할 사용자 전략을 찾을 수 없습니다.")
                    record = editor.record()
                    result, bundle = await run.io_bound(
                        run_user_backtest, record, start_date, end_date, **common_args,
                    )
                else:
                    spec = get_builtin_spec(title)
                    result, bundle = await run.io_bound(
                        run_builtin_backtest, title, start_date, end_date,
                        primary_weight=(
                            float(weight.value or 0) / 100.0 if spec.secondary else None
                        ),
                        **common_args,
                    )
            except Exception as error:
                status.set_text("백테스트를 완료하지 못했습니다.")
                with result_area:
                    with ui.card().classes("error-card w-full"):
                        ui.label(str(error)).classes("text-negative")
            else:
                status.set_text("백테스트가 완료되었습니다.")
                with result_area:
                    _render_backtest_result(result, bundle)
            finally:
                run_button.enable()

        run_button.on("click", execute_backtest)


ui.add_css(
    """
    :root { --snow-bg: #f5f7fb; --snow-text: #20242d; }
    body { background: var(--snow-bg); color: var(--snow-text); }
    .app-header { background: rgba(255,255,255,.96); color: var(--snow-text);
      border-bottom: 1px solid #e7eaf0; backdrop-filter: blur(10px); }
    .page-container { max-width: 1040px; margin: 0 auto; padding: 92px 20px 36px; }
    .page-title { font-size: clamp(1.55rem, 4vw, 2.25rem); font-weight: 800; line-height: 1.2; }
    .page-subtitle { color: #687080; font-size: .95rem; }
    .feature-card, .account-card, .migration-note { border-radius: 20px; border: 1px solid #e8ebf2;
      box-shadow: 0 8px 26px rgba(32,47,79,.06); padding: 22px; }
    .feature-card { min-height: 260px; justify-content: space-between; }
    .feature-icon { font-size: 38px; color: #4059df; background: #eef1ff;
      padding: 10px; border-radius: 14px; }
    .feature-description { color: #687080; line-height: 1.65; min-height: 52px; }
    .primary-action { border-radius: 12px; min-height: 46px; }
    .migration-note { background: linear-gradient(135deg, #f6f8ff, #fff); }
    .status-text { color: #687080; font-size: .85rem; }
    .error-card { border: 1px solid #ffcdd2; background: #fff7f8; border-radius: 16px; }
    .warning-card { border: 1px solid #ffe0a3; background: #fffaf0; border-radius: 16px; }
    .empty-state { width: 100%; text-align: center; padding: 52px 12px; color: #687080; }
    .nav-active { color: #4059df; background: #eef1ff; }
    .date-input { min-width: 180px; }
    .section-title { font-size: 1.2rem; font-weight: 800; margin-top: 8px; }
    .metric-grid { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; }
    .metric-card { border-radius: 16px; border: 1px solid #e8ebf2; box-shadow: none;
      padding: 16px; min-height: 100px; justify-content: center; }
    .metric-label { color: #737b8c; font-size: .82rem; }
    .metric-value { color: #222735; font-size: 1.25rem; font-weight: 800; }
    .dashboard-tabs { overflow-x: auto; flex-wrap: nowrap; }
    .dashboard-panels { border-radius: 18px; border: 1px solid #e8ebf2;
      box-shadow: 0 8px 26px rgba(32,47,79,.05); }
    .dashboard-chart { height: 430px; }
    .account-table { max-width: 100%; overflow-x: auto; }
    .backtest-controls { border-radius: 20px; border: 1px solid #e8ebf2;
      box-shadow: 0 8px 26px rgba(32,47,79,.06); padding: 22px; gap: 16px; }
    .strategy-expansion, .preset-library { border-radius: 20px; border: 1px solid #e8ebf2;
      background: white; box-shadow: 0 8px 26px rgba(32,47,79,.05); }
    .strategy-expansion .q-expansion-item__content { padding: 8px 18px 20px; }
    .editor-section { border-radius: 16px; border: 1px solid #eceef4;
      box-shadow: none; padding: 16px; gap: 12px; }
    .editor-fields, .backtest-field-grid {
      display: grid !important;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      align-items: start;
    }
    .editor-fields > *, .backtest-field-grid > * {
      width: 100% !important;
      min-width: 0 !important;
      max-width: none !important;
      flex: none !important;
    }
    .editor-section .q-field, .backtest-controls .q-field { width: 100%; }
    .preset-library { padding: 18px; gap: 10px; }
    .preset-upload { width: min(420px, 100%); }
    .diagnostic-download { align-self: flex-start; }
    .decision-select { width: min(360px, 100%); }
    .snapshot-code { max-height: 420px; overflow: auto; border-radius: 12px; }
    .control-grow { min-width: 180px; flex: 1; }
    .result-metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
    .rebalance-summary { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
    @media (max-width: 700px) {
      .feature-grid { grid-template-columns: 1fr !important; }
      .page-container { padding: 82px 14px 28px; }
      .feature-card { min-height: 230px; }
      .app-header .nav-button { font-size: 0; }
      .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
      .metric-card { min-height: 88px; padding: 12px; }
      .metric-value { font-size: 1.05rem; }
      .date-input { min-width: calc(50% - 8px); flex: 1; }
      .strategy-expansion .q-expansion-item__content { padding: 6px 10px 14px; }
      .editor-section { padding: 12px; }
      .dashboard-chart { height: 360px; }
      .result-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
      .rebalance-summary { grid-template-columns: 1fr !important; }
      .editor-fields, .backtest-field-grid {
        grid-template-columns: 1fr !important;
        gap: 10px;
      }
      .backtest-date-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
      .editor-fields .control-grow, .backtest-field-grid .control-grow {
        min-width: 0 !important;
        width: 100% !important;
      }
    }
    """
)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title=APP_TITLE,
        favicon="❄️",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("NICEGUI_RELOAD", "0") == "1",
        storage_secret=os.getenv("NICEGUI_STORAGE_SECRET", "snowball-local-development"),
    )
