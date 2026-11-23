"""Route wrapper for multi-asset required-capital split trading."""

from nicegui import ui

from features.snowball06_multi_split_trading.page import render_multi_split_page


@ui.page("/multi-infinite-split-trading")
def page() -> None:
    render_multi_split_page(
        forced_funding_mode="required", storage_prefix="multi_infinite_split",
        page_title="다종목 분할 무한매수",
        page_subtitle="여러 종목의 모든 신호를 체결할 수 있는 전체 최소 필요자금을 찾습니다.",
    )
