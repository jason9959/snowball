"""NiceGUI frontend introduced alongside the existing Streamlit app."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from threading import Lock

from nicegui import run, ui

from snowball.account_store import ACCOUNT_PREFIX, GoogleSheetAccountStore
from snowball.settings import google_sheet_settings


APP_TITLE = "Snowball Portfolio Lab"
ACCOUNT_CACHE_SECONDS = 300


@dataclass
class _AccountCache:
    tabs: tuple[str, ...] = ()
    loaded_at: float = 0.0


_account_cache = _AccountCache()
_account_cache_lock = Lock()


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
    tabs = _connect_store().list_account_tabs()
    with _account_cache_lock:
        _account_cache.tabs = tabs
        _account_cache.loaded_at = time.monotonic()
    return tabs


def _account_name(tab_name: str) -> str:
    return tab_name.removeprefix(ACCOUNT_PREFIX)


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
                "백테스팅 열기", "/backtest", enabled=False,
            )
            _feature_card(
                "account_balance_wallet", "투자 계좌 현황",
                "Google Sheet의 계좌별 거래 원장을 읽어 평가액과 투자 성과를 확인합니다.",
                "계좌 현황 보기", "/accounts",
            )
        with ui.card().classes("migration-note w-full"):
            ui.label("개편 진행 상황").classes("font-bold")
            ui.label("1단계 · 실행 기반과 메인 메뉴 구성 완료")
            ui.linear_progress(value=0.25).props("rounded color=primary")


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
                                    lambda _, tab=tab_name: ui.notify(
                                        f"{_account_name(tab)} 대시보드는 2단계에서 연결됩니다."
                                    ),
                                )
            finally:
                refresh.enable()

        refresh.on("click", lambda: render_accounts(True))
        await render_accounts()


@ui.page("/backtest")
def backtest_page() -> None:
    _shell()
    with ui.column().classes("page-container w-full gap-4"):
        ui.label("백테스팅").classes("page-title")
        ui.label("기존 Streamlit 백테스트는 정상 운영 중입니다.").classes("page-subtitle")
        with ui.card().classes("migration-note w-full"):
            ui.icon("construction").classes("text-4xl text-primary")
            ui.label("백테스트 화면은 계좌 대시보드 다음에 이전합니다.").classes("font-bold")
            ui.button("홈으로", on_click=lambda: ui.navigate.to("/")).props("outline")


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
    .empty-state { width: 100%; text-align: center; padding: 52px 12px; color: #687080; }
    .nav-active { color: #4059df; background: #eef1ff; }
    @media (max-width: 700px) {
      .feature-grid { grid-template-columns: 1fr !important; }
      .page-container { padding: 82px 14px 28px; }
      .feature-card { min-height: 230px; }
      .app-header .nav-button { font-size: 0; }
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
