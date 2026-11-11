from __future__ import annotations

from datetime import date, timedelta
import io
import matplotlib.pyplot as plt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from common.metric_ui import metric as render_metric, metric_dataframe
from common.metrics import sharpe_ratio, sortino_ratio, calmar_ratio, cashflow_xirr
from common.metrics import performance_summary
from common.metric_ui import render_metric_table



USD_KRW_TICKER = "KRW=X"
DOLLAR_INDEX_TICKER = "DX-Y.NYB"
COMPARISON_SERIES = {
    "달러인덱스": DOLLAR_INDEX_TICKER,
    "원/달러 환율": USD_KRW_TICKER,
    "엔/달러 환율": "JPY=X",
    "위안/달러 환율": "CNY=X",
    "스위스프랑/달러 환율": "CHF=X",
    "캐나다달러/달러 환율": "CAD=X",
    "유로/달러 환율": "EURUSD=X",
    "파운드/달러 환율": "GBPUSD=X",
    "호주달러/달러 환율": "AUDUSD=X",
}


def inject_css() -> None:
    st.html(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #667085;
            --line: #d8e0ea;
            --panel: #f7fafc;
            --accent: #0f766e;
            --danger: #b42318;
            --danger-soft: #fee4e2;
            --ok: #067647;
            --ok-soft: #dcfae6;
        }
        .main .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        section[data-testid="stSidebar"] {
            display: none;
        }
        div[data-testid="stAppViewContainer"] > .main {
            margin-left: 0;
        }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
        .hero {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 28px;
            background:
                linear-gradient(135deg, rgba(15, 118, 110, 0.12), rgba(255,255,255,0.72)),
                linear-gradient(0deg, #ffffff, #ffffff);
            margin-bottom: 18px;
        }
        .hero-title {
            font-size: 35px;
            line-height: 1.16;
            font-weight: 760;
            margin: 0 0 8px 0;
        }
        .hero-copy {
            color: var(--muted);
            font-size: 16px;
            line-height: 1.65;
            max-width: 820px;
            margin: 0;
        }
        .note {
            border-left: 4px solid var(--accent);
            background: var(--panel);
            padding: 12px 14px;
            color: var(--ink);
            border-radius: 6px;
            margin: 12px 0 18px 0;
        }
        .metric-card, .status-card, .nav-card, .settings-panel {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 15px;
            background: #ffffff;
            min-height: 112px;
        }
        .metric-label, .status-label {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 6px;
        }
        .metric-value {
            color: var(--ink);
            font-size: 25px;
            font-weight: 760;
            line-height: 1.2;
        }
        .metric-help, .status-help {
            color: var(--muted);
            font-size: 12px;
            margin-top: 6px;
            line-height: 1.45;
        }
        .status-ok, .status-no {
            border-radius: 999px;
            padding: 4px 10px;
            display: inline-block;
            font-size: 13px;
            font-weight: 700;
        }
        .status-ok { color: var(--ok); background: var(--ok-soft); }
        .status-no { color: var(--danger); background: var(--danger-soft); }
        .small-title {
            color: var(--ink);
            font-size: 18px;
            font-weight: 740;
            margin-bottom: 6px;
        }
        div.stButton > button {
            border-radius: 8px;
            min-height: 62px;
            font-weight: 700;
            justify-content: flex-start;
            text-align: left;
            padding: 12px 16px;
        }
        </style>
        """,
    )


def hero(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <section class="hero">
            <p class="hero-title">{title}</p>
            <p class="hero-copy">{copy}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_card(label: str, ok: bool, value: str, standard: str, help_text: str) -> None:
    badge = "충족" if ok else "미충족"
    klass = "status-ok" if ok else "status-no"
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <span class="{klass}">{badge}</span>
            <div class="metric-value" style="font-size:22px;margin-top:8px;">{value}</div>
            <div class="metric-help">기준: {standard}</div>
            <div class="status-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_krw(value: float) -> str:
    return f"{value:,.0f}원"


def format_pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def krw_axis(title: str) -> alt.Axis:
    return alt.Axis(
        title=title,
        labelExpr=(
            "datum.value >= 100000000 ? format(datum.value / 100000000, '.2f') + '억원' : "
            "format(datum.value / 10000000, '.2f') + '천만원'"
        ),
    )


def make_sample_market_data() -> pd.DataFrame:
    dates = pd.date_range(date.today() - timedelta(days=365 * 6), date.today(), freq="B")
    rng = np.random.default_rng(99595)
    dxy = 101 + 5.5 * np.sin(np.linspace(0, 9 * np.pi, len(dates))) + np.cumsum(rng.normal(0, 0.05, len(dates)))
    usdkrw = 1190 + 95 * np.sin(np.linspace(0.6, 8.8 * np.pi, len(dates))) + np.cumsum(rng.normal(0, 1.1, len(dates)))
    usdkrw = np.clip(usdkrw, 1030, 1520)
    return pd.DataFrame({"date": dates, "usdkrw": usdkrw.round(2), "dxy": dxy.round(2), "source": "sample"})


def make_sample_comparison_data() -> pd.DataFrame:
    dates = pd.date_range(date.today() - timedelta(days=365 * 7), date.today(), freq="B")
    rng = np.random.default_rng(995950)
    data = pd.DataFrame({"date": dates})
    for i, label in enumerate(COMPARISON_SERIES):
        base = 95 + i * 9
        cycle = np.sin(np.linspace(i / 3, 8 * np.pi + i / 3, len(dates))) * (3.5 + i * 0.2)
        drift = np.cumsum(rng.normal(0, 0.045 + i * 0.002, len(dates)))
        data[label] = (base + cycle + drift).round(4)
    data["source"] = "sample"
    return data


@st.cache_data(ttl=60 * 60 * 4)
def load_market_data() -> pd.DataFrame:
    try:
        import yfinance as yf

        start = date.today() - timedelta(days=365 * 7)
        end = date.today() + timedelta(days=1)
        raw = yf.download(
            [USD_KRW_TICKER, DOLLAR_INDEX_TICKER],
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=False,
        )
        fx = raw[USD_KRW_TICKER]["Close"].rename("usdkrw")
        dxy = raw[DOLLAR_INDEX_TICKER]["Close"].rename("dxy")
        data = pd.concat([fx, dxy], axis=1).dropna().reset_index()
        data.columns = ["date", "usdkrw", "dxy"]
        data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
        data["source"] = "yfinance"
        if len(data) < 260:
            raise ValueError("not enough market data")
        return data.sort_values("date").reset_index(drop=True)
    except Exception:
        return make_sample_market_data()


@st.cache_data(ttl=60 * 60 * 4)
def load_comparison_data() -> pd.DataFrame:
    try:
        import yfinance as yf

        start = date.today() - timedelta(days=365 * 7)
        end = date.today() + timedelta(days=1)
        raw = yf.download(
            list(COMPARISON_SERIES.values()),
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=False,
        )
        series = []
        for label, ticker in COMPARISON_SERIES.items():
            try:
                close = raw[ticker]["Close"].rename(label)
                series.append(close)
            except Exception:
                continue
        if len(series) < 2:
            raise ValueError("not enough comparison data")
        data = pd.concat(series, axis=1).dropna(how="all").ffill().dropna().reset_index()
        data = data.rename(columns={data.columns[0]: "date"})
        data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
        data["source"] = "yfinance"
        if len(data) < 260:
            raise ValueError("not enough comparison data")
        return data.sort_values("date").reset_index(drop=True)
    except Exception:
        return make_sample_comparison_data()


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy().sort_values("date").reset_index(drop=True)
    out["gap_ratio"] = out["dxy"] / out["usdkrw"] * 100
    out["usdkrw_52w_avg"] = out["usdkrw"].rolling(252, min_periods=60).mean()
    out["dxy_52w_avg"] = out["dxy"].rolling(252, min_periods=60).mean()
    out["gap_52w_avg"] = out["gap_ratio"].rolling(252, min_periods=60).mean()
    out["current_dollar_index"] = out["dxy"]
    out["fair_rate"] = out["dxy"] / out["gap_52w_avg"] * 100
    out["cond_fx_below_avg"] = out["usdkrw"] < out["usdkrw_52w_avg"]
    out["cond_dxy_below_avg"] = out["dxy"] < out["dxy_52w_avg"]
    out["cond_gap_above_avg"] = out["gap_ratio"] > out["gap_52w_avg"]
    out["cond_fx_below_fair"] = out["usdkrw"] < out["fair_rate"]
    condition_cols = ["cond_fx_below_avg", "cond_dxy_below_avg", "cond_gap_above_avg", "cond_fx_below_fair"]
    out["signal_score"] = out[condition_cols].sum(axis=1)
    return out.dropna().reset_index(drop=True)


def backtest_switching(
    data: pd.DataFrame,
    start_date: date,
    end_date: date,
    initial_krw: float,
    recurring_enabled: bool,
    monthly_krw: float,
    switch_pct: float,
    fee_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    view = data[(data["date"].dt.date >= start_date) & (data["date"].dt.date <= end_date)].copy()
    view = view.sort_values("date").reset_index(drop=True)

    if view.empty:
        return pd.DataFrame(), pd.DataFrame()

    first_rate = float(view.iloc[0]["usdkrw"])
    cash = float(initial_krw) / 2
    usd = (float(initial_krw) / 2) * (1 - fee_pct) / first_rate
    total_contribution = float(initial_krw)
    last_contribution_month = None
    last_switch_direction = None
    last_switch_date = None
    curve = []
    switches = []

    for row in view.itertuples(index=False):
        current_date = pd.Timestamp(row.date)
        current_month = (current_date.year, current_date.month)
        rate = float(row.usdkrw)
        score = int(row.signal_score)

        if score <= 1:
            signal_direction = "달러→원화"
        elif score >= 3:
            signal_direction = "원화→달러"
        else:
            signal_direction = None

        can_switch_this_day = False
        next_switch_date = None
        if signal_direction is None:
            last_switch_direction = None
            last_switch_date = None
        elif signal_direction != last_switch_direction:
            can_switch_this_day = True
        elif last_switch_date is not None:
            next_switch_date = last_switch_date + pd.DateOffset(months=1)
            if current_date >= next_switch_date:
                can_switch_this_day = True

        contribution_action = "없음"
        if recurring_enabled and current_month != last_contribution_month and float(monthly_krw) > 0:
            contribution = float(monthly_krw)
            usd_value_before = usd * rate
            total_before = cash + usd_value_before

            if can_switch_this_day and signal_direction == "달러→원화":
                cash += contribution
                contribution_action = "원화 적립"
            elif can_switch_this_day and signal_direction == "원화→달러":
                usd += contribution * (1 - fee_pct) / rate
                contribution_action = "달러 적립"
            elif total_before > 0:
                cash_weight = cash / total_before
                usd_weight = usd_value_before / total_before
                cash += contribution * cash_weight
                usd += contribution * usd_weight * (1 - fee_pct) / rate
                contribution_action = "비율 적립"
            else:
                cash += contribution
                contribution_action = "원화 적립"

            total_contribution += float(monthly_krw)
            last_contribution_month = current_month

        action = "보유"
        action_reason = "환전 없음"
        converted_value = 0.0
        cash_before_switch = cash
        usd_before_switch = usd
        usd_value_before_switch = usd * rate
        total_before_switch = cash + usd_value_before_switch

        if signal_direction is None:
            action_reason = "Stay"
        elif not can_switch_this_day:
            action_reason = f"한달 대기: {next_switch_date.date()}부터 가능" if next_switch_date is not None else "한달 대기"
        elif total_before_switch <= 0:
            action_reason = "평가액 없음"
        elif can_switch_this_day and total_before_switch > 0:
            target_value = total_before_switch * switch_pct
            if signal_direction == "원화→달러" and cash > 0:
                krw_to_convert = min(cash, target_value)
                usd += krw_to_convert * (1 - fee_pct) / rate
                cash -= krw_to_convert
                converted_value = krw_to_convert
                action = "원화→달러"
                action_reason = "환전 실행"
            elif signal_direction == "달러→원화" and usd > 0:
                usd_value_available = usd * rate
                usd_value_to_convert = min(usd_value_available, target_value)
                usd_to_sell = usd_value_to_convert / rate
                usd -= usd_to_sell
                cash += usd_value_to_convert * (1 - fee_pct)
                converted_value = usd_value_to_convert
                action = "달러→원화"
                action_reason = "환전 실행"
            elif signal_direction == "원화→달러":
                action_reason = "원화 부족"
            elif signal_direction == "달러→원화":
                action_reason = "달러 부족"

            if converted_value > 0:
                last_switch_direction = signal_direction
                last_switch_date = current_date
                switches.append(
                    {
                        "date": current_date,
                        "action": action,
                        "rate": rate,
                        "score": score,
                        "total_value": cash + usd * rate,
                        "converted_value": converted_value,
                        "cash_before": cash_before_switch,
                        "usd_value_before": usd_before_switch * rate,
                    }
                )

        usd_value = usd * rate
        total_value = cash + usd_value
        usd_weight = usd_value / total_value if total_value else 0.0
        cash_weight = cash / total_value if total_value else 0.0
        total_value_usd = total_value / rate if rate else 0.0
        curve.append(
            {
                "date": current_date,
                "usdkrw": rate,
                "dxy": float(row.dxy),
                "gap_ratio": float(row.gap_ratio),
                "fair_rate": float(row.fair_rate),
                "signal_score": score,
                "position": f"원화 {cash_weight:.0%} / 달러 {usd_weight:.0%}",
                "action": action,
                "action_reason": action_reason,
                "next_switch_date": next_switch_date,
                "signal_direction": signal_direction or "Stay",
                "contribution_action": contribution_action,
                "cash_krw": cash,
                "usd_amount": usd,
                "usd_value": usd_value,
                "cash_weight": cash_weight,
                "usd_weight": usd_weight,
                "total_value": total_value,
                "total_value_usd": total_value_usd,
                "total_contribution": total_contribution,
                "return": total_value / total_contribution - 1,
                "cond_fx_below_avg": bool(row.cond_fx_below_avg),
                "cond_dxy_below_avg": bool(row.cond_dxy_below_avg),
                "cond_gap_above_avg": bool(row.cond_gap_above_avg),
                "cond_fx_below_fair": bool(row.cond_fx_below_fair),
            }
        )

    return pd.DataFrame(curve), pd.DataFrame(switches)


def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    return float((series / peak - 1).min())



def _goto(page): st.session_state['current_page']=page

def _market(): return add_indicators(load_market_data())

def _png(curve):
    fig,axes=plt.subplots(2,1,figsize=(12,9),constrained_layout=True)
    axes[0].plot(curve['date'],curve['total_value'],label='Portfolio'); axes[0].plot(curve['date'],curve['total_contribution'],label='Contributed',ls='--'); axes[0].grid(alpha=.2); axes[0].legend()
    axes[1].plot(curve['date'],curve['usdkrw'],label='USD/KRW'); axes[1].plot(curve['date'],curve['fair_rate'],label='Fair rate'); axes[1].grid(alpha=.2); axes[1].legend()
    out=io.BytesIO(); fig.savefig(out,format='png',dpi=160,bbox_inches='tight'); plt.close(fig); return out.getvalue()

def render(page:str):
    inject_css()
    if page=='f07_home': return _home()
    if page=='f07_overview': return _overview()
    if page=='f07_comparison': return _comparison()
    if page=='f07_results': return _results()
    return _conditions()

def _home():
    st.title('🧭 달러 환율 나침반'); st.caption('환율 조건과 원화/달러 스위칭 전략을 확인합니다.')
    if st.button('📖 **오늘의 달러 투자 환경**  \n최신 네 가지 조건을 확인합니다.',key='f07_home_overview',use_container_width=True): _goto('f07_overview'); st.rerun()
    if st.button('📈 **거치식 · 적립식 계산**  \n원화/달러 스위칭 전략을 백테스트합니다.',key='f07_home_calc',use_container_width=True): _goto('f07_conditions'); st.rerun()
    if st.button('📊 **지표 비교 차트**  \n두 지표를 시작값 100으로 비교합니다.',key='f07_home_cmp',use_container_width=True): _goto('f07_comparison'); st.rerun()
    st.divider(); l,r=st.columns(2)
    with l:
        if st.button('뒤로',key='back_f07_home',use_container_width=True): _goto('feature'); st.rerun()
    with r:
        if st.button('백테스트 →',key='run_f07_home',use_container_width=True): _goto('f07_conditions'); st.rerun()

def _overview():
    data=_market(); x=data.iloc[-1]
    st.title('💵 오늘의 달러 투자 환경'); st.caption(f"기준일 {pd.Timestamp(x['date']).date()} · 조건 {int(x['signal_score'])}/4")
    cols=st.columns(4)
    vals=[('원/달러',x.cond_fx_below_avg,f"{x.usdkrw:,.2f}원"),('달러지수',x.cond_dxy_below_avg,f"{x.dxy:,.2f}"),('달러갭',x.cond_gap_above_avg,f"{x.gap_ratio:,.3f}"),('적정환율',x.cond_fx_below_fair,f"{x.fair_rate:,.2f}원")]
    for c,(name,ok,val) in zip(cols,vals): render_metric(c, name,val,'충족' if ok else '미충족')
    view=data.tail(252).melt(id_vars=['date'],value_vars=['usdkrw','fair_rate'],var_name='series',value_name='value')
    st.altair_chart(alt.Chart(view).mark_line().encode(x='date:T',y=alt.Y('value:Q',scale=alt.Scale(zero=False)),color='series:N').properties(height=360),use_container_width=True)
    st.divider(); l,r=st.columns(2)
    with l:
        if st.button('이전 단계',key='back_f07_overview',use_container_width=True): _goto('f07_home'); st.rerun()
    with r:
        if st.button('백테스트 조건 입력 →',key='run_f07_overview',use_container_width=True): _goto('f07_conditions'); st.rerun()

def _conditions():
    data=_market(); min_d=pd.Timestamp(data.date.min()).date(); max_d=pd.Timestamp(data.date.max()).date(); saved=st.session_state.get('f07_saved',{})
    st.title('💵 달러 환율 나침반 백테스트'); st.caption('결과 화면에서 돌아와도 마지막 실행 조건을 유지합니다.')
    if st.button('↺ 조건 초기화',key='reset_f07_conditions'): st.session_state.pop('f07_saved',None); st.session_state.pop('f07_result',None); st.rerun()
    c1,c2=st.columns(2); start=c1.date_input('시작일',saved.get('start',max(min_d,max_d-timedelta(days=365*3))),min_value=min_d,max_value=max_d,key='f07_start'); end=c2.date_input('종료일',saved.get('end',max_d),min_value=min_d,max_value=max_d,key='f07_end')
    c1,c2=st.columns(2); initial=c1.number_input('거치 금액 (KRW)',min_value=100000,value=int(saved.get('initial',10000000)),step=100000,key='f07_initial'); fee=c2.number_input('환전 수수료 (%)',min_value=0.0,max_value=1.0,value=float(saved.get('fee',.15)),step=.01,key='f07_fee')
    recurring=st.checkbox('월 적립식 사용',value=bool(saved.get('recurring',False)),key='f07_recurring')
    c1,c2=st.columns(2); monthly=c1.number_input('월 적립금',min_value=0,value=int(saved.get('monthly',500000)),step=10000,disabled=not recurring,key='f07_monthly'); switch=c2.number_input('신호 발생 시 환전 비율 (%)',min_value=1.0,max_value=50.0,value=float(saved.get('switch',10.0)),step=1.0,key='f07_switch')
    st.caption('점수 0~1: 달러→원화 · 2: Stay · 3~4: 원화→달러')
    st.divider(); l,r=st.columns(2)
    with l: back=st.button('뒤로',key='back_f07_conditions',use_container_width=True)
    with r: run=st.button('테스트 실행',key='run_f07_conditions',type='primary',use_container_width=True)
    if back: _goto('f07_home'); st.rerun()
    if run:
        if start>=end: st.error('시작일은 종료일보다 앞서야 합니다.'); return
        curve,switches=backtest_switching(data,start,end,float(initial),bool(recurring),float(monthly),float(switch)/100,float(fee)/100)
        if curve.empty: st.error('계산 가능한 데이터가 없습니다.'); return
        st.session_state['f07_saved']={'start':start,'end':end,'initial':initial,'fee':fee,'recurring':recurring,'monthly':monthly,'switch':switch}; st.session_state['f07_result']=(curve,switches); _goto('f07_results'); st.rerun()

def _results():
    if 'f07_result' not in st.session_state: _goto('f07_conditions'); st.rerun(); return
    curve,switches=st.session_state['f07_result']; ctx=st.session_state['f07_saved']; x=curve.iloc[-1]
    st.title('💵 달러 환율 나침반 결과'); st.caption(f"{ctx['start']} ~ {ctx['end']}")
    cols=st.columns(5); render_metric(cols[0], '총 납입',f"{x.total_contribution:,.0f}원"); render_metric(cols[1], '최종 자산',f"{x.total_value:,.0f}원"); render_metric(cols[2], '수익률',f"{x['return']*100:.2f}%"); render_metric(cols[3], 'MDD',f"{max_drawdown(curve.total_value)*100:.2f}%"); render_metric(cols[4], '전환',f'{len(switches)}회')
    dated = curve.set_index(pd.to_datetime(curve.date))
    increments = dated.total_contribution.diff().fillna(dated.total_contribution.iloc[0] - float(ctx['initial']))
    details = performance_summary(dated.total_value, increments, float(ctx['initial']))
    if not ctx['recurring']:
        details.pop('적립식 XIRR', None)
    render_metric_table({'달러 전략': details})
    st.altair_chart(alt.Chart(curve).mark_line().encode(x='date:T',y=alt.Y('total_value:Q',scale=alt.Scale(zero=False))).properties(height=420),use_container_width=True)
    if not switches.empty: metric_dataframe(st, switches,use_container_width=True,hide_index=True)
    st.divider(); l,r=st.columns(2)
    with l:
        if st.button('뒤로',key='back_f07_results',use_container_width=True): _goto('f07_conditions'); st.rerun()
    with r: st.download_button('결과 저장',_png(curve),'dollar-compass-result.png','image/png',key='download_f07_results',use_container_width=True)

def _comparison():
    data=load_comparison_data(); labels=[x for x in COMPARISON_SERIES if x in data.columns]; min_d=pd.Timestamp(data.date.min()).date(); max_d=pd.Timestamp(data.date.max()).date()
    st.title('📊 달러 지표 비교'); c1,c2=st.columns(2); start=c1.date_input('시작일',max(min_d,max_d-timedelta(days=365)),min_value=min_d,max_value=max_d,key='f07_cmp_s'); end=c2.date_input('종료일',max_d,min_value=min_d,max_value=max_d,key='f07_cmp_e')
    c1,c2=st.columns(2); a=c1.selectbox('첫 번째 지표',labels,key='f07_cmp_a'); b=c2.selectbox('두 번째 지표',labels,index=min(1,len(labels)-1),key='f07_cmp_b')
    if a!=b and start<end:
        v=data[(data.date.dt.date>=start)&(data.date.dt.date<=end)][['date',a,b]].dropna().copy()
        if len(v)>1:
            for m in [a,b]: v[m]=v[m]/v[m].iloc[0]*100
            st.altair_chart(alt.Chart(v.melt('date',var_name='series',value_name='index')).mark_line().encode(x='date:T',y=alt.Y('index:Q',scale=alt.Scale(zero=False)),color='series:N').properties(height=430),use_container_width=True)
    st.divider(); l,r=st.columns(2)
    with l:
        if st.button('이전 단계',key='back_f07_cmp',use_container_width=True): _goto('f07_home'); st.rerun()
    with r:
        if st.button('백테스트 조건 입력 →',key='run_f07_cmp',use_container_width=True): _goto('f07_conditions'); st.rerun()
