"""Minimal NiceGUI entry point for the modular Snowball features."""

from __future__ import annotations

import os

from nicegui import app, ui

from common.snowball.layout import APP_TITLE
from common.snowball.styles import APP_CSS
from features.snowball01_individual_backtest import page as _snowball01_page  # noqa: F401
from features.snowball02_portfolio_mix import page as _snowball02_page  # noqa: F401
from features.snowball03_account_dashboard import page as _snowball03_page  # noqa: F401
from features.functions_nicegui import page as _functions_page  # noqa: F401


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    """Lightweight readiness endpoint for container platforms."""
    return {"status": "ok"}


ui.add_css(APP_CSS)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title=APP_TITLE,
        favicon="❄️",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("NICEGUI_RELOAD", "0") == "1",
        storage_secret=os.getenv("NICEGUI_STORAGE_SECRET", "snowball-local-development"),
    )
