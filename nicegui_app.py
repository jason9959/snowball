"""NiceGUI frontend introduced alongside the existing Streamlit app."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from threading import Lock
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nicegui import run, ui

from snowball.account_store import ACCOUNT_PREFIX, GoogleSheetAccountStore
from snowball.account_dashboard import (
    AccountDashboardData,
    account_date_bounds,
    active_account_frame,
    build_account_dashboard,
)
from snowball.accounting import download_unadjusted_closes, required_price_symbols
from snowball.builtin_library import BUILTIN_PRESET_SPECS, get_builtin_spec, run_builtin_backtest
from snowball.settings import google_sheet_settings


APP_TITLE = "Snowball Portfolio Lab"
ACCOUNT_CACHE_SECONDS = 300
TRANSACTION_CACHE_SECONDS = 60
PRICE_CACHE_SECONDS = 900


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
            ui.label("3-1단계 · 기본 프리셋 백테스트 연결")
            ui.linear_progress(value=0.75).props("rounded color=primary")


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
                ui.label("프리셋 3 월간 운용").classes("font-bold")
                ui.label("프리셋 1 · 분산투자 방어형 50%")
                ui.label("프리셋 2 · 집중투자 공격형 50%")
                ui.label("운용 시작 · 2026-11-02 시가")
                ui.label(f"선택 계좌 · {_account_name(account_tab)}")
                ui.label("목표 비중과 실제 비중 비교는 다음 세부 단계에서 연결합니다.").classes(
                    "text-xs text-grey-6"
                )


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
    ui.label(f"{result['preset_name']} 결과").classes("section-title")
    ui.label(
        f"공통 평가기간 · {pd.Timestamp(result['common_start']).date()} ~ "
        f"{pd.Timestamp(result['common_end']).date()}"
    ).classes("text-sm text-grey-6")
    with ui.grid(columns=3).classes("result-metric-grid w-full gap-3"):
        for label, value in metric_values:
            _metric_card(label, value)

    nav = result["nav"].dropna().astype(float)
    chart = pd.DataFrame({"포트폴리오": nav})
    benchmark = result.get("benchmark_ticker", "QQQ")
    if benchmark in bundle.adjusted_close:
        comparison = bundle.adjusted_close[benchmark].reindex(nav.index).ffill().dropna()
        if not comparison.empty:
            comparison = comparison / comparison.iloc[0]
            chart["국내 나스닥100" if benchmark.endswith(".KS") else "QQQ"] = comparison
    dates = [value.strftime("%Y-%m-%d") for value in chart.index]
    series = [
        {
            "name": column, "type": "line", "showSymbol": False,
            "data": chart[column].round(5).tolist(),
            "lineStyle": {"width": 3 if column == "포트폴리오" else 2},
        }
        for column in chart.columns
    ]
    with ui.tabs().classes("w-full dashboard-tabs") as tabs:
        returns_tab = ui.tab("수익률 차트", icon="show_chart")
        drawdown_tab = ui.tab("손실폭", icon="trending_down")
        history_tab = ui.tab("리밸런싱", icon="event_repeat")
    with ui.tab_panels(tabs, value=returns_tab).classes("w-full dashboard-panels"):
        with ui.tab_panel(returns_tab):
            ui.echart({
                "tooltip": {"trigger": "axis"},
                "legend": {"bottom": 0, "data": list(chart.columns)},
                "grid": {"left": 54, "right": 24, "top": 24, "bottom": 58},
                "xAxis": {"type": "category", "data": dates, "axisLabel": {"hideOverlap": True}},
                "yAxis": {"type": "value", "scale": True},
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
        with ui.tab_panel(history_tab):
            history = result["history"].copy()
            display = [
                column for column in (
                    "적용일", "판단 기준일", "선택된 종목", "턴오버", "거래비용", "평가금액"
                ) if column in history
            ]
            table = history[display].sort_values("적용일", ascending=False).copy()
            for column in table.columns:
                table[column] = table[column].astype(str)
            ui.table(
                columns=_table_columns(display), rows=table.to_dict("records"),
                pagination={"rowsPerPage": 12},
            ).props("flat bordered dense").classes("w-full account-table")


@ui.page("/backtest")
def backtest_page() -> None:
    _shell()
    with ui.column().classes("page-container w-full gap-4"):
        ui.label("백테스팅").classes("page-title")
        ui.label("기본 제공 프리셋 1~10을 같은 계산 엔진으로 검증합니다.").classes("page-subtitle")
        with ui.card().classes("backtest-controls w-full"):
            preset = ui.select(
                list(BUILTIN_PRESET_SPECS), value=next(iter(BUILTIN_PRESET_SPECS)),
                label="내 프리셋",
            ).props("outlined options-dense").classes("w-full")
            with ui.row().classes("w-full gap-3"):
                start_input = ui.input("시작일", value="2010-01-01").props(
                    "type=date"
                ).classes("date-input")
                end_input = ui.input(
                    "종료일", value=datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
                ).props("type=date").classes("date-input")
            with ui.row().classes("w-full gap-3"):
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

        status = ui.label("조건을 확인한 뒤 실행하세요.").classes("status-text")
        result_area = ui.column().classes("w-full gap-4")

        def update_preset_controls() -> None:
            spec = get_builtin_spec(str(preset.value))
            weight.set_visibility(spec.secondary is not None)
            initial.value = 10_000_000 if spec.domestic_execution else 10_000
            initial.min = 100_000 if spec.domestic_execution else 100
            initial.update()

        preset.on_value_change(lambda _: update_preset_controls())
        update_preset_controls()

        async def execute_backtest() -> None:
            run_button.disable()
            result_area.clear()
            status.set_text("가격과 시장 데이터를 불러와 계산하고 있습니다…")
            try:
                title = str(preset.value)
                spec = get_builtin_spec(title)
                result, bundle = await run.io_bound(
                    run_builtin_backtest,
                    title,
                    date.fromisoformat(str(start_input.value)),
                    date.fromisoformat(str(end_input.value)),
                    initial_investment=float(initial.value or 0),
                    transaction_cost_rate=float(cost.value or 0) / 10_000.0,
                    market_data_mode=str(market_mode.value),
                    primary_weight=(
                        float(weight.value or 0) / 100.0 if spec.secondary else None
                    ),
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
    .control-grow { min-width: 180px; flex: 1; }
    .result-metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
    @media (max-width: 700px) {
      .feature-grid { grid-template-columns: 1fr !important; }
      .page-container { padding: 82px 14px 28px; }
      .feature-card { min-height: 230px; }
      .app-header .nav-button { font-size: 0; }
      .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
      .metric-card { min-height: 88px; padding: 12px; }
      .metric-value { font-size: 1.05rem; }
      .date-input { min-width: calc(50% - 8px); flex: 1; }
      .dashboard-chart { height: 360px; }
      .result-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
      .control-grow { min-width: 100%; }
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
