from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common.metric_ui import metric_dataframe, render_metric_table
from common.metrics import performance_summary
from .backtest import run_backtest
from .data import INDICATORS, fetch_current_fear_greed, fetch_fear_greed, fetch_prices


def _rating(score: float) -> str:
    return "극단적 공포" if score < 25 else "공포" if score < 45 else "중립" if score <= 55 else "탐욕" if score < 75 else "극단적 탐욕"


INDICATOR_DESCRIPTIONS = {
    "fear_and_greed_historical": "시장 전반의 투자 심리를 0~100점으로 나타낸 종합 지표입니다.",
    "market_momentum_sp500": "S&P 500의 장기 추세와 현재 위치를 바탕으로 계산한 모멘텀입니다.",
    "market_momentum_sp125": "S&P 500의 125일 이동평균 대비 추세를 나타냅니다.",
    "stock_price_strength": "52주 신고가와 신저가 종목의 상대적인 강도를 나타냅니다.",
    "stock_price_breadth": "상승 종목과 하락 종목의 시장 breadth를 나타냅니다.",
    "put_call_options": "풋옵션과 콜옵션 거래 비율을 이용한 투자 심리 지표입니다.",
    "market_volatility_vix": "S&P 500 옵션시장의 기대 변동성을 나타내는 VIX 기반 지표입니다.",
    "junk_bond_demand": "위험도가 높은 회사채에 대한 수요를 나타냅니다.",
    "safe_haven_demand": "주식보다 안전자산을 선호하는 정도를 나타냅니다.",
}


def _configure_chart_font() -> None:
    """서버 환경에서도 한글과 유니코드 기호가 네모로 표시되지 않게 한다."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic", "AppleGothic"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def _reset() -> None:
    for key in list(st.session_state):
        if key.startswith("fg_"):
            st.session_state.pop(key, None)
    st.session_state["fear_greed_result"] = None


def _conditions() -> None:
    st.title("😨 공포·탐욕 지수 리밸런싱")
    st.markdown("<p class='step-caption'>시장 심리 지표가 기준값에 도달했을 때 포트폴리오를 목표 비율로 되돌립니다.</p>", unsafe_allow_html=True)
    if st.button("↺ 조건 초기화", key="fg_reset"):
        _reset(); st.rerun()

    try:
        current = fetch_current_fear_greed()
        score = float(current["values"].get("fear_and_greed_historical", np.nan))
        if np.isfinite(score):
            st.info(f"최근 종합 지수: **{score:.1f} · {_rating(score)}** ({current['date']:%Y-%m-%d})")
    except Exception as exc:
        st.warning(f"현재 지수를 불러오지 못했습니다. 백테스트 실행 시 다시 시도합니다. ({exc})")

    today = dt.date.today()
    start_col, end_col, initial_col = st.columns(3)
    with start_col:
        start = st.date_input("시작 날짜", value=st.session_state.setdefault("fg_start", today - dt.timedelta(days=365 * 5)), min_value=dt.date(2010, 1, 1), max_value=today, key="fg_start")
    with end_col:
        end = st.date_input("종료 날짜", value=st.session_state.setdefault("fg_end", today), min_value=dt.date(2010, 1, 1), max_value=today, key="fg_end")
    with initial_col:
        initial = st.number_input("초기 투자금", min_value=1.0, step=1000.0, value=float(st.session_state.setdefault("fg_initial", 10000.0)), key="fg_initial")

    left, right = st.columns(2)
    with left:
        investment_type = st.selectbox("투자 방식", ["거치식", "적립식"], key="fg_investment_type")
        contribution = st.number_input("정기 적립금", min_value=1.0, step=100.0, value=float(st.session_state.setdefault("fg_contribution", 1000.0)), key="fg_contribution", disabled=investment_type != "적립식")
    with right:
        rebalance_frequency = st.selectbox("리밸런싱 확인 주기", ["매일", "매월", "매분기", "매년"], key="fg_rebalance")
        contribution_frequency = st.selectbox("적립 주기", ["매월", "매분기", "매년"], key="fg_contribution_frequency", disabled=investment_type != "적립식")

    st.divider(); st.subheader("지표 조건")
    metric_col, direction_col, threshold_col = st.columns([2, 1, 1])
    with metric_col:
        indicator_key = st.selectbox("리밸런싱에 사용할 지표", list(INDICATORS), format_func=lambda key: INDICATORS[key], key="fg_indicator")
    with direction_col:
        direction = st.selectbox("조건", ["이하일 때", "이상일 때"], key="fg_direction")
    with threshold_col:
        threshold = st.number_input("기준값", min_value=0.0, max_value=100.0, value=float(st.session_state.setdefault("fg_threshold", 25.0)), step=1.0, key="fg_threshold")

    st.divider(); st.subheader("포트폴리오 구성")
    st.caption("티커와 목표 비중의 합계를 100%로 입력하세요.")
    items = []
    for index in range(5):
        ticker_col, weight_col = st.columns([3, 1])
        with ticker_col:
            ticker = st.text_input(f"자산 {index + 1}", placeholder="예: SPY 또는 QQQ", key=f"fg_ticker_{index}").strip().upper()
        with weight_col:
            weight = st.number_input(f"비중 {index + 1} (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"fg_weight_{index}")
        if ticker and weight > 0:
            items.append({"ticker": ticker, "weight": float(weight)})
    st.caption(f"현재 입력 비중: {sum(item['weight'] for item in items):.0f}%")

    back_col, result_col = st.columns(2)
    with back_col:
        back = st.button("뒤로", key="fg_back", use_container_width=True)
    with result_col:
        run = st.button("결과 보기", type="primary", key="fg_run", use_container_width=True)
    if back:
        st.session_state["current_page"] = "feature"; st.rerun()
    if not run:
        return
    if start >= end:
        st.error("시작 날짜는 종료 날짜보다 앞서야 합니다."); return
    if not items or len({item["ticker"] for item in items}) != len(items):
        st.error("서로 다른 티커와 비중을 입력해주세요."); return
    total = sum(item["weight"] for item in items)
    if not np.isclose(total, 100.0):
        st.error("포트폴리오 비중의 합계가 100%여야 합니다."); return
    try:
        tickers = tuple(item["ticker"] for item in items)
        prices = fetch_prices(tickers, start.isoformat(), end.isoformat())
        fg = fetch_fear_greed(start.isoformat(), end.isoformat())
        if indicator_key not in fg.columns:
            raise ValueError("선택한 지표의 데이터가 없습니다.")
        calc = run_backtest(prices, fg[indicator_key], np.array([item["weight"] / 100 for item in items]), float(initial), rebalance_frequency, float(threshold), direction, float(contribution if investment_type == "적립식" else 0), contribution_frequency)
        st.session_state["fear_greed_result"] = {**calc, "items": items, "indicator_key": indicator_key, "start": start, "end": end, "initial": float(initial), "investment_type": investment_type, "rebalance_frequency": rebalance_frequency, "direction": direction, "threshold": float(threshold), "contribution": float(contribution if investment_type == "적립식" else 0), "contribution_frequency": contribution_frequency}
        st.session_state["current_page"] = "fear_greed_results"; st.rerun()
    except Exception as exc:
        st.error(f"백테스트를 계산하지 못했습니다: {exc}")


def _home() -> None:
    st.title("😨 공포·탐욕 지수")
    st.markdown("<p class='step-caption'>시장 심리 지표를 확인하고 이를 활용한 리밸런싱 전략을 테스트합니다.</p>", unsafe_allow_html=True)
    menu = [
        ("fg_home_overview", "📖 **지표 현황 보기**  \n공포·탐욕 사이트에서 제공하는 지표와 흐름을 확인합니다.", "fear_greed_overview"),
        ("fg_home_rebalance", "📈 **리밸런싱 계산**  \n선택한 지표가 기준값에 도달할 때 포트폴리오를 재조정합니다.", "fear_greed_conditions"),
        ("fg_home_compare", "📊 **지표 비교 차트**  \n두 지표를 각각의 축으로 비교합니다.", "fear_greed_compare"),
    ]
    for key, label, target in menu:
        if st.button(label, key=key, use_container_width=True):
            st.session_state["current_page"] = target
            st.rerun()
    st.divider()
    if st.button("뒤로", key="back_fg_home", use_container_width=True):
        st.session_state["current_page"] = "feature"
        st.rerun()


def _overview() -> None:
    st.title("📖 공포·탐욕 지수 현황")
    st.markdown("[CNN Fear & Greed Index 원문 보기](https://edition.cnn.com/markets/fear-and-greed)")
    try:
        current = fetch_current_fear_greed()
        values = current["values"]
        score = float(values.get("fear_and_greed_historical", np.nan))
        if np.isfinite(score):
            st.metric("종합 공포·탐욕 지수", f"{score:.1f}", _rating(score))
        st.caption(f"기준일: {current['date']:%Y-%m-%d} · 0은 극단적 공포, 100은 극단적 탐욕입니다.")
        cards = [(INDICATORS[key], float(values[key])) for key in INDICATORS if key in values and np.isfinite(float(values[key]))]
        for start in range(0, len(cards), 3):
            cols = st.columns(3)
            for col, (label, value) in zip(cols, cards[start:start + 3]):
                col.metric(label, f"{value:.1f}", _rating(value))
        history = current["history"].tail(252)
        st.subheader("지표별 최근 흐름")
        for key, label in INDICATORS.items():
            if key not in history or history[key].dropna().empty:
                continue
            st.markdown(f"**{label}** · {INDICATOR_DESCRIPTIONS[key]}")
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=history.index,
                y=history[key],
                mode="lines",
                name=label,
                line={"color": "#3182F6", "width": 2},
                connectgaps=True,
            ))
            if key == "fear_and_greed_historical":
                for level, name in [(25, "극단적 공포 경계"), (45, "공포 경계"), (55, "중립 경계"), (75, "탐욕 경계")]:
                    figure.add_hline(y=level, line_dash="dot", line_color="#DDE3EA", annotation_text=name, annotation_position="top left")
            figure.update_layout(
                height=300,
                margin={"l": 10, "r": 20, "t": 12, "b": 10},
                showlegend=True,
                legend={"orientation": "h", "y": 1.02, "x": 0},
                xaxis={"title": "날짜", "showgrid": False},
                yaxis={"title": "지표 값", "rangemode": "normal", "showgrid": True, "gridcolor": "#E5E8EB"},
                hovermode="x unified",
            )
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    except Exception as exc:
        st.error(f"지표를 불러오지 못했습니다: {exc}")
    st.divider()
    left, right = st.columns(2)
    with left:
        if st.button("뒤로", key="back_fg_overview", use_container_width=True): st.session_state["current_page"] = "fear_greed_home"; st.rerun()
    with right:
        if st.button("리밸런싱 조건 입력 →", key="run_fg_overview", use_container_width=True): st.session_state["current_page"] = "fear_greed_conditions"; st.rerun()


def _compare() -> None:
    st.title("📊 공포·탐욕 지표 비교")
    st.caption("첫 번째 지표는 왼쪽 Y축, 두 번째 지표는 오른쪽 Y축에 표시합니다.")
    today = dt.date.today()
    start = st.date_input("시작 날짜", value=today - dt.timedelta(days=365), min_value=dt.date(2010, 1, 1), max_value=today, key="fg_compare_start")
    end = st.date_input("종료 날짜", value=today, min_value=dt.date(2010, 1, 1), max_value=today, key="fg_compare_end")
    choices = list(INDICATORS)
    first, second = st.columns(2)
    with first:
        first_key = st.selectbox("첫 번째 지표 · 왼쪽 Y축", choices, format_func=lambda key: INDICATORS[key], key="fg_compare_first")
    with second:
        second_key = st.selectbox("두 번째 지표 · 오른쪽 Y축", choices, index=min(1, len(choices) - 1), format_func=lambda key: INDICATORS[key], key="fg_compare_second")
    if start < end and first_key != second_key:
        try:
            _configure_chart_font()
            history = fetch_fear_greed(start.isoformat(), end.isoformat())[[first_key, second_key]].dropna()
            figure, axis_left = plt.subplots(figsize=(12, 5.2)); axis_right = axis_left.twinx()
            axis_left.plot(history.index, history[first_key], color="#3182F6", linewidth=2, label=INDICATORS[first_key])
            axis_right.plot(history.index, history[second_key], color="#F04452", linewidth=2, label=INDICATORS[second_key])
            axis_left.set_ylabel(INDICATORS[first_key], color="#3182F6"); axis_right.set_ylabel(INDICATORS[second_key], color="#F04452")
            axis_left.set_ylim(0, 100); axis_right.set_ylim(0, 100); axis_left.grid(alpha=.2)
            lines, labels = axis_left.get_legend_handles_labels(); lines2, labels2 = axis_right.get_legend_handles_labels(); axis_left.legend(lines + lines2, labels + labels2, loc="upper left")
            st.pyplot(figure); plt.close(figure)
        except Exception as exc:
            st.error(f"비교 그래프를 만들지 못했습니다: {exc}")
    elif first_key == second_key:
        st.info("서로 다른 지표를 선택해주세요.")
    else:
        st.info("시작 날짜는 종료 날짜보다 앞서야 합니다.")
    st.divider()
    left, right = st.columns(2)
    with left:
        if st.button("뒤로", key="back_fg_compare", use_container_width=True): st.session_state["current_page"] = "fear_greed_home"; st.rerun()
    with right:
        if st.button("리밸런싱 조건 입력 →", key="run_fg_compare", use_container_width=True): st.session_state["current_page"] = "fear_greed_conditions"; st.rerun()


def _results() -> None:
    result = st.session_state.get("fear_greed_result")
    if not result:
        st.session_state["current_page"] = "fear_greed_conditions"; st.rerun(); return
    values = result["values"]; invested = result["invested"]; indicator = result["indicator"]
    _configure_chart_font()
    st.title("😨 공포·탐욕 지수 리밸런싱 결과")
    st.caption(f"{result['start']} ~ {result['end']} · {INDICATORS[result['indicator_key']]} · {result['direction']} {result['threshold']:.0f} · {result['rebalance_frequency']}")
    figure, axis = plt.subplots(figsize=(12, 5.4)); axis2 = axis.twinx()
    axis.plot(values.index, values, color="#3182F6", linewidth=2.2, label="포트폴리오 자산")
    axis.plot(invested.index, invested, color="#8B95A1", linestyle="--", label="총 투입금")
    axis2.plot(indicator.index, indicator, color="#F04452", alpha=.65, label="선택 지표")
    axis2.axhline(result["threshold"], color="#F04452", linestyle=":", alpha=.8)
    axis.set_ylabel("자산 가치"); axis2.set_ylabel("지표 점수"); axis.grid(alpha=.2)
    lines, labels = axis.get_legend_handles_labels(); lines2, labels2 = axis2.get_legend_handles_labels(); axis.legend(lines + lines2, labels + labels2, loc="upper left")
    st.pyplot(figure); plt.close(figure)
    final_value = float(values.iloc[-1]); total_invested = float(invested.iloc[-1])
    metric_cols = st.columns(4)
    metric_cols[0].metric("최종 자산", f"{final_value:,.0f}")
    metric_cols[1].metric("총 투입금", f"{total_invested:,.0f}")
    metric_cols[2].metric("총 수익률", f"{(final_value / total_invested - 1) * 100:+.2f}%")
    metric_cols[3].metric("리밸런싱 횟수", f"{len(result['events'][result['events']['type'].eq('조건 충족 리밸런싱')]) if not result['events'].empty else 0}회")
    contributions = invested.diff().fillna(0)
    perf = performance_summary(values, contributions, float(invested.iloc[0]))
    render_metric_table({"공포·탐욕 조건 전략": perf})
    st.subheader("리밸런싱 이벤트")
    if result["events"].empty: st.info("조건을 충족한 리밸런싱이 없었습니다.")
    else: metric_dataframe(st, result["events"], hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        if st.button("뒤로", key="fg_result_back", use_container_width=True): st.session_state["current_page"] = "fear_greed_conditions"; st.rerun()
    with right:
        st.download_button("결과 저장", result["events"].to_csv(index=False).encode("utf-8-sig"), "fear-greed-rebalance-events.csv", "text/csv", key="fg_save", use_container_width=True)


def render(page: str) -> None:
    if page in {"fear_greed_home", "fear_greed_rebalance", "fear_greed"}: _home()
    elif page == "fear_greed_overview": _overview()
    elif page == "fear_greed_compare": _compare()
    elif page == "fear_greed_conditions": _conditions()
    elif page == "fear_greed_results": _results()
    else: _home()

