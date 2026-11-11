from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from dateutil.relativedelta import relativedelta
import streamlit as st
from common.metric_ui import metric as render_metric, metric_dataframe

from .backtest import run_backtest
from common.metrics import performance_summary
from common.metric_ui import render_metric_table
from .charts import portfolio_chart, price_chart
from .data_loader import load_price_data
from .reporting import create_result_png



def build_defaults() -> dict:
    # Streamlit Cloud servers may run in UTC; use Korea local date for "today".
    default_end = datetime.now(ZoneInfo("Asia/Seoul")).date()
    default_start = default_end - relativedelta(years=10)
    return {
        "ticker": "QQQ",
        "start_date": default_start,
        "end_date": default_end,
        "ma_months": 10,
        "confirmation_count": 1,
        "signal_limit_days": 5,
        "initial_amount": 10_000_000.0,
        "use_contribution": False,
        "contribution_amount": 1_000_000.0,
        "contribution_frequency": "매월",
    }


DEFAULTS = build_defaults()
INPUT_KEYS = list(DEFAULTS.keys())


def initialize_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_inputs() -> None:
    # Rebuild defaults so "today" is always current when reset is pressed.
    for key, value in build_defaults().items():
        st.session_state[key] = value
    st.session_state.pop("backtest_output", None)
    st.session_state.pop("backtest_context", None)
    st.session_state["current_page"] = "f09_conditions"


def go_back() -> None:
    # Input values intentionally remain untouched.
    st.session_state["current_page"] = "f09_conditions"


@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_price_data(ticker: str, start_date: date, end_date: date, ma_months: int):
    return load_price_data(ticker, start_date, end_date, ma_months)


def money(value: float) -> str:
    return f"{value:,.0f}"


def show_input_page() -> None:
    st.title("📈 이동평균 투자전략 백테스트")
    if st.button("↺ 조건 초기화", key="reset_f09_conditions_top"):
        reset_inputs(); st.rerun()
    st.write(
        "종가와 이동평균의 BUY/SELL 시그널을 확인하고, 확인 횟수와 대기 LIMIT 규칙에 따라 포지션을 전환하는 전략을 테스트합니다."
    )
    st.caption("N개월 = N × 21거래일 단순이동평균(SMA), 확인 횟수/LIMIT 충족 시 당일 종가 체결, 수수료·세금·환율 미반영")

    st.divider()

    st.subheader("조건 입력")

    ticker_col, ma_col, confirmation_col, limit_col = st.columns([2, 1, 1, 1])
    with ticker_col:
        st.text_input(
            "종목 티커",
            key="ticker",
            help="예: QQQ, SPY, AAPL, 005930.KS",
        )
    with ma_col:
        st.number_input(
            "이동평균 기간 (개월)",
            min_value=1,
            max_value=60,
            step=1,
            key="ma_months",
        )
    with confirmation_col:
        st.number_input(
            "포지션 변경 확인 횟수",
            min_value=1,
            max_value=30,
            step=1,
            key="confirmation_count",
            help="현재 포지션의 반대 방향 시그널이 몇 번 확인되면 실제 포지션을 변경할지 정합니다.",
        )
    with limit_col:
        st.number_input(
            "시그널 대기 LIMIT (거래일)",
            min_value=1,
            max_value=252,
            step=1,
            key="signal_limit_days",
            help=(
                "같은 방향 시그널이 확인된 뒤 다음 같은 방향 시그널을 기다리는 최대 거래일 수입니다. "
                "반대 시그널이 나오면 대기 상태가 초기화되고, LIMIT까지 반대 시그널이 없으면 현재 누적 횟수로 포지션을 변경합니다."
            ),
        )

    start_col, end_col = st.columns(2)
    with start_col:
        st.date_input("시작일", key="start_date")
    with end_col:
        st.date_input("종료일", key="end_date")

    st.number_input(
        "초기 투자금",
        min_value=1.0,
        step=100_000.0,
        format="%.0f",
        key="initial_amount",
        help="종목 가격과 같은 통화 단위로 계산합니다. 예: 미국 종목이면 USD, 한국 종목이면 KRW 기준으로 해석합니다.",
    )

    st.divider()
    st.subheader("적립식 투자")

    st.checkbox("적립식 투자 사용", key="use_contribution")

    if st.session_state.use_contribution:
        amount_col, frequency_col = st.columns(2)
        with amount_col:
            st.number_input(
                "적립 금액",
                min_value=1.0,
                step=100_000.0,
                format="%.0f",
                key="contribution_amount",
            )
        with frequency_col:
            st.selectbox(
                "적립 주기",
                options=["매주", "매월", "매분기", "매반기", "매년"],
                key="contribution_frequency",
            )

        st.caption(
            "첫 적립일은 시작일에서 선택 주기만큼 지난 날입니다. 해당 날짜가 휴장일이면 이후 가장 가까운 거래일에 적립하며, 다음 예정일은 원래 시작일 기준을 유지합니다."
        )

    st.divider()

    back_col, action_col = st.columns(2)
    with back_col:
        back_clicked = st.button("뒤로", key="back_f09_conditions", use_container_width=True)
    with action_col:
        run_clicked = st.button(
            "백테스트 실행", type="primary", key="run_f09_conditions", use_container_width=True
        )
    if back_clicked:
        st.session_state["current_page"] = "feature"
        st.rerun()

    if run_clicked:
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("종료일은 시작일보다 뒤여야 합니다.")
            return

        with st.spinner("가격 데이터를 불러오고 백테스트를 계산하고 있습니다..."):
            try:
                data = cached_load_price_data(
                    st.session_state.ticker.strip().upper(),
                    st.session_state.start_date,
                    st.session_state.end_date,
                    int(st.session_state.ma_months),
                )
                output = run_backtest(
                    price_data=data,
                    requested_start_date=st.session_state.start_date,
                    requested_end_date=st.session_state.end_date,
                    initial_amount=float(st.session_state.initial_amount),
                    use_contribution=bool(st.session_state.use_contribution),
                    contribution_amount=float(st.session_state.contribution_amount),
                    contribution_frequency=st.session_state.contribution_frequency,
                    confirmation_count=int(st.session_state.confirmation_count),
                    signal_limit_days=int(st.session_state.signal_limit_days),
                )
            except Exception as exc:
                st.error(str(exc))
                return

        st.session_state.backtest_output = output
        st.session_state.backtest_context = {
            key: st.session_state[key] for key in INPUT_KEYS
        }
        st.session_state["current_page"] = "f09_results"
        st.rerun()


def show_result_page() -> None:
    if "backtest_output" not in st.session_state or "backtest_context" not in st.session_state:
        st.session_state["current_page"] = "f09_conditions"
        st.rerun()
        return

    output = st.session_state.backtest_output
    context = st.session_state.backtest_context
    daily = output.daily
    events = output.events
    metrics = output.metrics
    ticker = context["ticker"].strip().upper()

    png_bytes = create_result_png(
        ticker=ticker,
        ma_months=int(context["ma_months"]),
        confirmation_count=int(context["confirmation_count"]),
        signal_limit_days=int(context["signal_limit_days"]),
        requested_start=context["start_date"],
        requested_end=context["end_date"],
        initial_amount=float(context["initial_amount"]),
        use_contribution=bool(context["use_contribution"]),
        contribution_amount=float(context["contribution_amount"]),
        contribution_frequency=context["contribution_frequency"],
        daily=daily,
        metrics=metrics,
    )

    st.title(f"{ticker} 백테스트 결과")
    st.caption(
        f"{context['start_date']} ~ {context['end_date']} · {int(context['ma_months'])}개월 이동평균 · "
        f"포지션 변경 확인 {int(context['confirmation_count'])}회 · "
        f"시그널 대기 LIMIT {int(context['signal_limit_days'])}거래일 · 당일 종가 체결"
    )

    comparison = {}
    for label, column in [('이동평균 전략', 'portfolio_value'), (f'{ticker} 단순 보유', 'buy_hold_value')]:
        values = daily[column]
        details = performance_summary(values, daily['contribution'], float(context['initial_amount']))
        details = {'총 납입금': money(metrics['total_contributions']), '최종 자산': money(values.iloc[-1]), '투자 수익': money(values.iloc[-1] - metrics['total_contributions']), **details}
        if not context['use_contribution']:
            details.pop('적립식 XIRR', None)
        comparison[label] = details
    st.subheader('백테스팅 결과')
    render_metric_table(comparison)
    st.caption('동일한 초기 투자금과 적립 일정으로 비교합니다. 위험 지표에서는 납입금 증가 효과를 제외합니다.')

    st.caption(
        f"실제 첫 거래일: {metrics['effective_start_date']} · 종료 포지션: {metrics['ending_position']} · "
        f"현금 {money(metrics['ending_cash'])} · 보유수량 {metrics['ending_shares']:,.6f}"
    )

    st.divider()
    st.plotly_chart(
        price_chart(daily, events, ticker, int(context["ma_months"])),
        use_container_width=True,
    )

    st.plotly_chart(
        portfolio_chart(daily, float(context["initial_amount"]), ticker),
        use_container_width=True,
    )

    st.divider()
    st.subheader("거래 및 적립 내역")

    if events.empty:
        st.info("표시할 거래/적립 이벤트가 없습니다.")
    else:
        event_labels = {
            "BUY": "매수 전환",
            "SELL": "매도 전환",
            "ADD_BUY": "적립 매수",
            "CONTRIBUTION_CASH": "현금 적립",
        }
        display_events = events.copy()
        display_events["구분"] = display_events["event"].map(event_labels).fillna(display_events["event"])
        display_events["날짜"] = pd.to_datetime(display_events["date"]).dt.strftime("%Y-%m-%d")
        display_events["가격"] = display_events["price"].map(lambda x: f"{x:,.4f}")
        display_events["금액"] = display_events["amount"].map(lambda x: f"{x:,.0f}")
        display_events["적립금"] = display_events["contribution"].map(lambda x: f"{x:,.0f}")
        display_events["설명"] = display_events["note"]
        metric_dataframe(st,
            display_events[["날짜", "구분", "가격", "금액", "적립금", "설명"]],
            use_container_width=True,
            hide_index=True,
        )

    if context["use_contribution"] and not output.schedule.empty:
        with st.expander("적립 예정일 / 실제 적립일 확인"):
            schedule_view = output.schedule.copy()
            schedule_view["예정 적립일"] = schedule_view["scheduled_date"].dt.strftime("%Y-%m-%d")
            schedule_view["실제 적립일"] = schedule_view["actual_date"].dt.strftime("%Y-%m-%d")
            metric_dataframe(st,
                schedule_view[["예정 적립일", "실제 적립일"]],
                use_container_width=True,
                hide_index=True,
            )

    st.caption(
        "가정: 수정 종가(Adjusted Close), 소수점 매수 허용, 수수료·세금·환율·현금이자 미반영. "
        "이 결과는 전략 테스트용이며 실제 체결 결과와 다를 수 있습니다."
    )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.button('뒤로', use_container_width=True, on_click=go_back, key='ma_results_back')
    with right:
        st.download_button('결과 저장', data=png_bytes, file_name=f'{ticker}_backtest.png', mime='image/png', use_container_width=True, key='ma_results_save')


def render(page: str) -> None:
    initialize_state()
    if page == "f09_results":
        show_result_page()
    else:
        show_input_page()
