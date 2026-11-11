from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def price_chart(daily: pd.DataFrame, events: pd.DataFrame, ticker: str, ma_months: int):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["Close"],
            mode="lines",
            name=f"{ticker} 종가",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["SMA"],
            mode="lines",
            name=f"{ma_months}개월 이동평균",
        )
    )

    if not events.empty:
        # BUY / SELL markers represent only actual position changes after the
        # configured number of consecutive signals has been confirmed.
        buy_events = events[events["event"] == "BUY"]
        sell_events = events[events["event"] == "SELL"]

        if not buy_events.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_events["date"],
                    y=buy_events["price"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=11),
                    name="매수 전환",
                )
            )
        if not sell_events.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_events["date"],
                    y=sell_events["price"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=11),
                    name="매도 전환",
                )
            )

    fig.update_layout(
        title=f"{ticker} 가격과 {ma_months}개월 이동평균",
        xaxis_title="날짜",
        yaxis_title="가격",
        hovermode="x unified",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def portfolio_chart(daily: pd.DataFrame, initial_amount: float, ticker: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["portfolio_value"],
            mode="lines",
            name="이동평균 전략",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["buy_hold_value"],
            mode="lines",
            name=ticker.upper(),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=initial_amount + daily["contribution"].cumsum(),
            mode="lines",
            name="누적 납입금",
        )
    )
    fig.update_layout(
        title="자산 변화",
        xaxis_title="날짜",
        yaxis_title="자산 가치",
        hovermode="x unified",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig
