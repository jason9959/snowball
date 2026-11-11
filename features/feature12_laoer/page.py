from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
from dateutil.relativedelta import relativedelta
import pandas as pd
import streamlit as st
from common.metric_ui import metric as render_metric, metric_dataframe
from .data_loader import load_ohlc
from .backtest import run_v22
from .charts import price_chart,value_chart,t_chart
from .reporting import create_png


def _goto(page): st.session_state['current_page']=page

def _defaults():
    end=datetime.now(ZoneInfo('Asia/Seoul')).date(); return {'ticker':'TQQQ','start':end-relativedelta(years=10),'end':end,'initial':40_000.0,'divisions':40,'fee':0.0,'quarter_stop':False}

def render(page:str):
    if page=='f10_results': _results()
    else: _conditions()

def _conditions():
    saved=st.session_state.get('f10_saved',{}); d=_defaults()
    st.title('♾️ 라오어 무한매수법 백테스트')
    st.caption('V2.2 규칙을 일봉 OHLC로 근사 재현합니다. LOC는 종가 조건, +10% 지정가는 일중 High로 체결 여부를 판단합니다.')
    if st.button('↺ 조건 초기화',key='reset_f10_conditions'):
        st.session_state.pop('f10_saved',None); st.session_state.pop('f10_result',None); st.rerun()
    c1,c2,c3=st.columns([2,1,1])
    ticker=c1.text_input('종목 티커',value=saved.get('ticker',d['ticker']),key='f10_w_ticker')
    divisions=int(c2.number_input('분할 횟수',min_value=10,max_value=100,value=int(saved.get('divisions',d['divisions'])),step=1,key='f10_w_div'))
    fee=float(c3.number_input('거래비용 (%)',min_value=0.0,max_value=2.0,value=float(saved.get('fee',d['fee'])),step=.01,key='f10_w_fee'))
    c1,c2=st.columns(2)
    start=c1.date_input('시작일',value=saved.get('start',d['start']),key='f10_w_start'); end=c2.date_input('종료일',value=saved.get('end',d['end']),key='f10_w_end')
    initial=float(st.number_input('초기 투자금',min_value=100.0,value=float(saved.get('initial',d['initial'])),step=1000.0,key='f10_w_initial'))
    quarter=st.checkbox('쿼터손절 모드 포함 (근사 구현)',value=bool(saved.get('quarter_stop',False)),key='f10_w_qs')
    st.info('현재 구현은 V2.2의 T값·별%·전/후반 LOC 매수·부분매도·+10% 지정가를 재현합니다. 쿼터손절은 일봉만으로 주문 순서를 완전히 복원하기 어려워 옵션으로 분리했습니다.')
    st.divider(); left,right=st.columns(2)
    with left: back=st.button('뒤로',key='back_f10_conditions',use_container_width=True)
    with right: run=st.button('백테스트 실행',type='primary',key='run_f10_conditions',use_container_width=True)
    if back: _goto('feature'); st.rerun()
    if run:
        try:
            with st.spinner('가격 데이터를 불러와 무한매수법을 계산하고 있습니다...'):
                data=load_ohlc(ticker,start,end); out=run_v22(data,initial,divisions,fee/100,quarter)
            st.session_state['f10_saved']={'ticker':ticker,'start':start,'end':end,'initial':initial,'divisions':divisions,'fee':fee,'quarter_stop':quarter}
            st.session_state['f10_result']=out; _goto('f10_results'); st.rerun()
        except Exception as e: st.error(str(e))

def _results():
    out=st.session_state.get('f10_result'); ctx=st.session_state.get('f10_saved')
    if out is None or ctx is None: _goto('f10_conditions'); st.rerun(); return
    m=out.metrics; daily=out.daily; trades=out.trades; ticker=ctx['ticker'].upper()
    st.title(f'♾️ {ticker} 무한매수법 V2.2 결과')
    st.caption(f"{ctx['start']} ~ {ctx['end']} · {ctx['divisions']}분할 · 거래비용 {ctx['fee']:.2f}%")
    cols=st.columns(6)
    render_metric(cols[0], '초기 투자금',f"{m['initial_capital']:,.0f}"); render_metric(cols[1], '최종 자산',f"{m['final_value']:,.0f}"); render_metric(cols[2], '총수익률',f"{m['return_pct']:.2f}%")
    render_metric(cols[3], 'CAGR',f"{m['cagr_pct']:.2f}%"); render_metric(cols[4], 'MDD',f"{m['mdd_pct']:.2f}%"); render_metric(cols[5], '완료 사이클',f"{m['completed_cycles']}회")
    st.plotly_chart(price_chart(daily,trades,ticker),use_container_width=True)
    st.plotly_chart(value_chart(daily),use_container_width=True)
    st.plotly_chart(t_chart(daily),use_container_width=True)
    st.subheader('거래 내역')
    if trades.empty: st.info('거래 내역이 없습니다.')
    else: metric_dataframe(st, trades,use_container_width=True,hide_index=True)
    png=create_png(ticker,daily,m)
    st.divider(); left,right=st.columns(2)
    with left:
        if st.button('뒤로',key='back_f10_results',use_container_width=True): _goto('f10_conditions'); st.rerun()
    with right: st.download_button('결과 저장',data=png,file_name=f'{ticker}-laoer-v22.png',mime='image/png',key='download_f10_results',use_container_width=True)
