"""Snowball-specific NiceGUI shell and feature menu.

This module intentionally does not reuse the functions app's Streamlit layout.
They serve the same purpose through different UI runtimes and must remain
separate until a future integration explicitly chooses an adapter.
"""

from __future__ import annotations

import os
from urllib.parse import quote

from nicegui import ui


APP_TITLE = "Snowball Portfolio Lab"
FUNCTIONS_APP_URL = os.getenv(
    "FUNCTIONS_APP_URL", "https://cinema99592.streamlit.app"
).rstrip("/")

FUNCTION_TOOLS = (
    ("show_chart", "1. 상대 수익률 비교", "여러 종목을 같은 기준값으로 비교합니다.", "return_comparison"),
    ("calendar_month", "2. 주기별 수익률 분석", "과거 수익률의 주기별 흐름을 확인합니다.", "periodic_returns"),
    ("work", "3. 정적 포트폴리오", "고정 비중 포트폴리오를 백테스트합니다.", "portfolio_backtest"),
    ("sync", "4. 동적 포트폴리오", "모멘텀과 방어 스위치 전략을 검증합니다.", "dynamic_portfolio"),
    ("balance", "5. 비율별 리밸런싱", "자산 비율별 포트폴리오 성과를 비교합니다.", "allocation_sweep"),
    ("casino", "6. 정규분포 몬테카를로", "평균과 변동성으로 미래 경로를 모의합니다.", "monte_carlo_normal"),
    ("extension", "7. Bootstrap 몬테카를로", "실제 과거 수익률을 재표본화합니다.", "monte_carlo_bootstrap"),
    ("explore", "8. 인플레이션 나침반", "성장과 기대인플레이션 국면을 분석합니다.", "inflation_compass"),
    ("attach_money", "9. 달러 환율 나침반", "환율 조건과 원화·달러 전략을 분석합니다.", "dollar_compass"),
    ("apartment", "10. 기업 재무 대시보드", "DART·SEC 공식 공시를 조회합니다.", "financial_dashboard"),
    ("moving", "11. 이동평균 투자전략", "이동평균 매수·매도 전략을 검증합니다.", "moving_average"),
    ("all_inclusive", "12. 라오어 무한매수법", "V2.2 분할매수 규칙을 백테스트합니다.", "laoer_infinite"),
    ("home", "13. 부동산 vs 주식", "주택 매수와 주식 적립을 비교합니다.", "real_estate_vs_stock"),
    ("sentiment_satisfied", "14. 공포·탐욕 리밸런싱", "시장 심리 기준 리밸런싱을 검증합니다.", "fear_greed_rebalance"),
    ("location_city", "15. 한국부동산원 통계", "R-ONE 공식 부동산 통계를 분석합니다.", "reb_statistics"),
)


def shell(active: str = "") -> None:
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


def feature_card(
    icon: str,
    title: str,
    description: str,
    button_text: str,
    target: str,
) -> None:
    with ui.card().classes("feature-card w-full"):
        with ui.row().classes("items-center gap-3"):
            ui.icon(icon).classes("feature-icon")
            ui.label(title).classes("text-xl font-bold")
        ui.label(description).classes("feature-description")
        ui.button(
            button_text,
            icon="arrow_forward",
            on_click=lambda: ui.navigate.to(target),
        ).classes("primary-action w-full")


def functions_card(icon: str, title: str, description: str, feature_id: str) -> None:
    target = f"{FUNCTIONS_APP_URL}/?feature={quote(feature_id)}"
    with ui.card().classes("function-card w-full"):
        with ui.row().classes("items-center gap-3 no-wrap"):
            ui.icon(icon).classes("function-icon")
            with ui.column().classes("gap-1 min-w-0"):
                ui.label(title).classes("text-base font-bold")
                ui.label(description).classes("function-description")
        ui.button(
            "기능 열기",
            icon="open_in_new",
            on_click=lambda: ui.navigate.to(target),
        ).props("flat").classes("function-action w-full")


@ui.page("/")
def home_page() -> None:
    shell("home")
    with ui.column().classes("page-container w-full gap-5"):
        ui.label("Snowball 기능").classes("page-title")
        ui.label(
            "기존 functions 기능과 분리된 Snowball 전용 도구입니다. 통합 시에도 별도 구역으로 표시됩니다."
        ).classes("page-subtitle")
        with ui.grid(columns=3).classes("feature-grid w-full gap-4"):
            feature_card(
                "query_stats", "Snowball 01 · 개별 백테스트",
                "한 개의 동적자산배분 전략을 편집하고 장기 성과를 검증합니다.",
                "개별 백테스트", "/backtest",
            )
            feature_card(
                "call_merge", "Snowball 02 · 포트폴리오 혼합",
                "저장된 두 프리셋을 선택하고 매월 복원할 혼합 비율을 설정합니다.",
                "혼합 백테스트", "/mix-backtest",
            )
            feature_card(
                "account_balance_wallet", "Snowball 03 · 투자 계좌 현황",
                "Google Sheet의 계좌별 거래 원장을 읽어 평가액과 투자 성과를 확인합니다.",
                "계좌 현황 보기", "/accounts",
            )

        ui.separator().classes("my-3")
        ui.label("Functions 투자 도구").classes("page-title section-heading")
        ui.label(
            "기존 계산 소스를 유지한 채 각 기능으로 바로 연결합니다. 화면은 이후 NiceGUI로 순차 이관합니다."
        ).classes("page-subtitle")
        with ui.grid(columns=3).classes("functions-grid w-full gap-3"):
            for icon, title, description, feature_id in FUNCTION_TOOLS:
                functions_card(icon, title, description, feature_id)
