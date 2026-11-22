"""Required-capital discovery page using the shared split-trading engine."""

from nicegui import ui

from features.snowball04_split_trading.page import render_split_trading_page


@ui.page("/infinite-split-trading")
def infinite_split_trading_page() -> None:
    render_split_trading_page(
        forced_funding_mode="required",
        storage_prefix="infinite_split_trading",
        page_title="분할 무한매수",
        page_subtitle=(
            "현금 제한 없이 모든 매수 신호를 체결한 뒤, "
            "횟수·비율별 최소 필요자금을 역산합니다."
        ),
    )
