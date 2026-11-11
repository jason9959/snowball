"""Snowball-specific NiceGUI shell and feature menu.

This module intentionally does not reuse the functions app's Streamlit layout.
They serve the same purpose through different UI runtimes and must remain
separate until a future integration explicitly chooses an adapter.
"""

from __future__ import annotations

from nicegui import ui


APP_TITLE = "Snowball Portfolio Lab"


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
