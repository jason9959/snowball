from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd


def create_result_png(
    ticker: str,
    ma_months: int,
    confirmation_count: int,
    signal_limit_days: int,
    requested_start,
    requested_end,
    initial_amount: float,
    use_contribution: bool,
    contribution_amount: float,
    contribution_frequency: str,
    daily: pd.DataFrame,
    metrics: dict,
) -> bytes:
    """Create a portable PNG summary using English labels for font portability."""
    fig = plt.figure(figsize=(12, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.4, 3.0, 3.0, 0.25])

    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")

    contribution_text = (
        f"{contribution_frequency} / {contribution_amount:,.0f}"
        if use_contribution
        else "Off"
    )
    summary = (
        f"{ticker} Moving Average Backtest\n"
        f"Period: {requested_start} ~ {requested_end}    MA: {ma_months} months    "
        f"Confirmation: {confirmation_count} signal(s)    Limit: {signal_limit_days} trading day(s)\n"
        f"Initial capital: {initial_amount:,.0f}    Contribution: {contribution_text}\n\n"
        f"Total contributed: {metrics['total_contributions']:,.0f}\n"
        f"Final value: {metrics['final_value']:,.0f}    Profit: {metrics['profit']:,.0f}\n"
        f"Return: {metrics['return_pct']:.2f}%    MDD: {metrics['mdd_pct']:.2f}%\n"
        f"{ticker} final: {metrics['buy_hold_final_value']:,.0f}    "
        f"{ticker} return: {metrics['buy_hold_return_pct']:.2f}%"
    )
    ax0.text(0.01, 0.98, summary, va="top", ha="left", fontsize=13)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(daily.index, daily["Close"], label=f"{ticker} Adjusted Close")
    ax1.plot(daily.index, daily["SMA"], label=f"SMA ({ma_months} months)")
    ax1.set_title("Price & Moving Average")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(daily.index, daily["portfolio_value"], label="MA Strategy")
    ax2.plot(daily.index, daily["buy_hold_value"], label=ticker.upper())
    cumulative_contribution = metrics["initial_amount"] + daily["contribution"].cumsum()
    ax2.plot(daily.index, cumulative_contribution, label="Total Contributed")
    ax2.set_title("Portfolio Value")
    ax2.grid(alpha=0.25)
    ax2.legend()

    ax3 = fig.add_subplot(gs[3])
    ax3.axis("off")
    ax3.text(
        0.01,
        0.5,
        "Assumptions: same-day close execution, fractional shares, no fees/taxes, adjusted close prices.",
        va="center",
        ha="left",
        fontsize=9,
    )

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()
