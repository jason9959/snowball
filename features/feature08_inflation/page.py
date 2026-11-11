"""Standalone local Streamlit app; visual conventions adapted from stock_info."""
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import pandas as pd
import streamlit as st
from common.metric_ui import metric as render_metric, metric_dataframe
from .compass import Config, signals, backtest, metrics

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://cssanalytics.wordpress.com/2026/07/27/the-inflation-compass-model/'
REGIMES = {'reflation':'성장 ↑ · 인플레이션 ON', 'goldilocks':'성장 ↑ · 인플레이션 OFF',
           'stagflation':'성장 ↓ · 인플레이션 ON', 'slowdown':'성장 ↓ · 인플레이션 OFF'}
HOLDINGS = {'reflation':'XLE · 에너지', 'goldilocks':'XLK · 기술', 'stagflation':'XLU · 유틸리티',
            'slowdown':'XLP + IEF · 필수소비재 / 미국 국채'}
PRESETS = {'원문 재현 가정':Config(),
           '보수적 검증':Config(fred_lag=1,execution_lag=1,cost_bps=10),
           '직접 설정':Config()}
def go(page):
    st.session_state['current_page']=f'f06_{page}'

def go_main():
    st.session_state['current_page']='feature'

def reset_conditions():
    st.session_state.pop('f06_saved',None)
    st.session_state.pop('f06_result',None)
    st.session_state.pop('f06_config',None)
    for key in [
        'f06_start_widget','f06_end_widget','f06_preset_widget','f06_initial_widget',
        'f06_threshold_widget','f06_growth_widget','f06_be_widget','f06_asset_widget',
        'f06_lag_widget','f06_execution_widget','f06_cost_widget',
    ]:
        st.session_state.pop(key,None)
    go('conditions')

REQUIRED_ASSETS=['SPY','XLE','XLI','XLF','XLB','XLU','XLV','XLP','XLK','IEF']

@st.cache_data(show_spinner=False)
def load_data(stamp):
    prices=pd.read_csv(ROOT/'data/prices.csv',index_col=0,parse_dates=True)
    fred=pd.read_csv(ROOT/'data/T5YIE.csv',index_col=0,parse_dates=True).iloc[:,0]
    prices.index=pd.to_datetime(prices.index).tz_localize(None)
    fred.index=pd.to_datetime(fred.index).tz_localize(None)
    return prices.sort_index(),pd.to_numeric(fred,errors='coerce').sort_index()

@st.cache_data(show_spinner=False)
def calculate(start,end,config,stamp):
    p,f=load_data(stamp)
    s=signals(p,f,config)
    nav,spy,events,regimes=backtest(p,s,start,end,config)
    return nav,spy,events,regimes,s

def formatted_stats(nav,spy):
    rows=[]
    for label,v in [('나침반',nav),('SPY',spy)]:
        m=metrics(v)
        rows.append({'전략':label,'연복리 수익률':f"{m['CAGR']:.2%}",
                     '최대낙폭 · 일별':f"{m['MaxDD']:.2%}",'연 변동성':f"{m['Volatility']:.2%}",
                     '샤프 · 무위험 0%':f"{m['Sharpe_rf0']:.2f}", 'Sortino':f"{m['Sortino']:.2f}", 'Calmar':f"{m['Calmar']:.2f}"})
    return pd.DataFrame(rows)

def yes_no(value):
    return '충족' if bool(value) else '미충족'

def static_line_chart(data, colors, ylabel=None, height=280, annotations=None):
    """Render a clean static PNG so result charts have no interactive toolbar."""
    frame=data.dropna(how='all')
    fig=Figure(figsize=(11.5,height/100),dpi=140,facecolor='white')
    ax=fig.subplots()
    for column,color in zip(frame.columns,colors):
        ax.plot(frame.index,frame[column],label=str(column),color=color,linewidth=1.65)
    ax.set_facecolor('white')
    ax.grid(axis='y',color='#E5E8EB',linewidth=.8)
    ax.grid(axis='x',visible=False)
    ax.tick_params(colors='#6B7684',labelsize=8)
    for side in ['top','right','left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#D1D6DB')
    if ylabel:
        ax.set_ylabel(ylabel,color='#6B7684',fontsize=9)
    locator=mdates.AutoDateLocator(minticks=5,maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(loc='upper left',frameon=False,ncol=min(3,len(frame.columns)),fontsize=8)
    if annotations:
        ymin,ymax=ax.get_ylim()
        for date,label in annotations.items():
            date=pd.Timestamp(date)
            if frame.index.min() <= date <= frame.index.max():
                ax.axvline(date,color='#B8C0CC',linewidth=.8,linestyle='--',alpha=.7)
                ax.annotate(str(label),xy=(date,ymin),xytext=(3,-18),textcoords='offset points',
                            ha='left',va='top',fontsize=8,color='#4E5968',fontweight='bold',rotation=90,
                            clip_on=False)
    fig.tight_layout(pad=.8)
    if annotations:
        fig.subplots_adjust(bottom=.24)
    output=io.BytesIO()
    fig.savefig(output,format='png',bbox_inches='tight',facecolor='white')
    return output.getvalue()

def status_cell(value):
    if value=='충족':
        return 'background-color:#E8F7EE;color:#176B43;font-weight:700'
    if value=='미충족':
        return 'background-color:#FDECEC;color:#B42318;font-weight:700'
    return ''

def regime_cell(value):
    positive=value in ('성장 ↑','인플레이션 ON')
    background='#E8F7EE' if positive else '#FDECEC'
    color='#176B43' if positive else '#B42318'
    return f'background-color:{background};color:{color};font-weight:700'

def result_report_image(nav, spy, initial, events):
    fig=Figure(figsize=(12,8.5),dpi=150,facecolor='white')
    axes=fig.subplots(2,1)
    value=pd.DataFrame({'Compass':nav*initial,'SPY':spy*initial})
    dd=pd.DataFrame({'Compass':(nav/nav.cummax().clip(lower=1)-1)*100,'SPY':(spy/spy.cummax()-1)*100})
    for ax,frame,title,ylabel in [(axes[0],value,'Inflation Compass · Portfolio Value','USD'),(axes[1],dd,'Inflation Compass · Drawdown','%')]:
        for col,color in zip(frame.columns,['#3182F6','#A5ABB3']): ax.plot(frame.index,frame[col],label=col,color=color,lw=1.5)
        ax.set_title(title,loc='left',fontsize=13,fontweight='bold'); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False)
        ax.spines[['top','right','left']].set_visible(False)
    fig.suptitle('인플레이션 나침반 백테스트 결과',fontsize=18,fontweight='bold',x=.05,ha='left')
    fig.tight_layout(rect=[0,.02,1,.96])
    out=io.BytesIO(); fig.savefig(out,format='png',facecolor='white',bbox_inches='tight'); return out.getvalue()

def monthly_decisions(sig, events):
    if events.empty:
        return pd.DataFrame()
    signal_dates=pd.DatetimeIndex(events.signal_date)
    view=sig.loc[signal_dates,[
        'spy','sma','breakeven','breakeven_lookback','breakeven_change',
        'inflation_indicator','basket_slope','growth_up','level_above_target',
        'breakeven_momentum_positive','asset_momentum_positive','inflation_on','regime'
    ]].copy()
    view.insert(0,'체결일',pd.to_datetime(events.date).to_numpy())
    view['성장 조건']=view.pop('growth_up').map(yes_no)
    view['2% 수준 조건']=view.pop('level_above_target').map(yes_no)
    view['기대물가 모멘텀']=view.pop('breakeven_momentum_positive').map(yes_no)
    view['자산 모멘텀']=view.pop('asset_momentum_positive').map(yes_no)
    view['인플레이션 ON']=view.pop('inflation_on').map(yes_no)
    view['국면']=view.pop('regime').map(REGIMES)
    view['보유 자산']=sig.loc[signal_dates,'regime'].map(HOLDINGS)
    return view.rename(columns={
        'spy':'SPY','sma':'SPY 200일 SMA','breakeven':'T5YIE',
        'breakeven_lookback':'60거래일 전 T5YIE','breakeven_change':'T5YIE 60일 변화',
        'inflation_indicator':'인플레이션 지표','basket_slope':'지표 60일 회귀 기울기'
    }).rename_axis('신호일').reset_index()


def render(page: str) -> None:
    price_path=ROOT/'data/prices.csv'
    fred_path=ROOT/'data/T5YIE.csv'
    if not price_path.exists() or not fred_path.exists():
        st.title('🧭 인플레이션 나침반')
        st.error('8번 기능의 로컬 데이터 파일이 없습니다. features/feature08_inflation/data 폴더를 확인해 주세요.')
        if st.button('← 통합 대시보드로 돌아가기',key='f06_missing_back',width='stretch'):
            go_main(); st.rerun()
        return

    stamp=(price_path.stat().st_mtime,fred_path.stat().st_mtime)
    try:
        prices,fred=load_data(stamp)
    except Exception as error:
        st.error(f'저장된 데이터를 읽을 수 없습니다. 상세: {error}')
        return

    calendar_previous_month_end=pd.Timestamp.today().normalize().replace(day=1)-pd.Timedelta(days=1)
    default_end=min(calendar_previous_month_end,prices.index[-1].normalize(),fred.dropna().index[-1].normalize())
    asset_first_dates={asset:prices[asset].dropna().index.min() for asset in REQUIRED_ASSETS}
    common_start=max(asset_first_dates.values()).normalize()
    default_start=max(common_start,default_end-pd.DateOffset(years=10))

    local_page=page.removeprefix('f06_')
    if local_page=='home':
        st.title('🧭 인플레이션 나침반')
        st.markdown('<p class="step-caption">Inflation Compass · 전략을 이해하고, 데이터로 확인하세요.</p>',unsafe_allow_html=True)
        st.markdown('<p class="feature-intro">어떤 분석을 해볼까요?</p>',unsafe_allow_html=True)
        for key,icon,title,desc in [
            ('rules','📖','전략 이해하기','성장과 기대인플레이션이 투자 대상을 바꾸는 원리를 확인합니다.'),
            ('conditions','📈','전략 백테스트','분석 기간과 거래 조건을 정하고 S&P 500과 비교합니다.'),
            ('audit','🔎','검증 결과 살펴보기','거래 지연, 비용, 신호 구성에 따라 성과가 어떻게 달라지는지 확인합니다.')]:
            st.button(f'{icon} **{title}**  \n{desc}',key='f06_menu_'+key,width='stretch',on_click=go,args=(key,))
        st.caption(f'저장된 ETF 데이터: {prices.index[0]:%Y.%m.%d} – {prices.index[-1]:%Y.%m.%d} · USD · 배당·분할 조정 가격')
        st.divider()
        if st.button('기능선택 화면으로 이동', key='global_go_feature', width='stretch'):
            go_main(); st.rerun()

    elif local_page=='rules':
        st.title('📖 전략 이해하기')
        st.write('월말마다 성장과 기대인플레이션 신호를 확인하고 다음 달의 보유 자산을 정하는 전략입니다.')
        metric_dataframe(st, pd.DataFrame([
            {'성장':'상승','인플레이션':'ON','보유 자산':'XLE · 에너지 100%'},
            {'성장':'상승','인플레이션':'OFF','보유 자산':'XLK · 기술 100%'},
            {'성장':'하락','인플레이션':'ON','보유 자산':'XLU · 유틸리티 100%'},
            {'성장':'하락','인플레이션':'OFF','보유 자산':'XLP · 필수소비재 50% + IEF · 국채 50%'}]),hide_index=True,width='stretch')
        st.write('성장 상승은 SPY가 200거래일 이동평균보다 높은 상태입니다. 인플레이션 ON은 5년 기대인플레이션이 2%를 넘으면서, 기대인플레이션의 60거래일 변화 또는 업종 바스켓 비율의 60일 회귀 기울기 중 하나가 양수인 상태입니다.')
        with st.expander('업종 바스켓의 계산 방식'):
            st.code('상승 바스켓 일간 수익률 = 0.5 XLE + (XLI + XLF + XLB) / 6\n하락 바스켓 일간 수익률 = (XLU + XLV + XLP) / 3\n지표 = 상승 바스켓 누적 성장 / 하락 바스켓 누적 성장')
        st.info('인플레이션 ON/OFF는 가격이 오르거나 내렸다는 뜻이 아니라 모델의 조건 충족 여부입니다. 성장 신호 역시 GDP가 아닌 주가 추세입니다.')
        st.write(f'[모델 원문]({SOURCE}) · [FRED 기대인플레이션](https://fred.stlouisfed.org/series/T5YIE)')
        st.caption('연준의 2% 목표는 PCE 물가 기준입니다. 기대인플레이션 지표의 2%와 개념이 완전히 같지는 않습니다.')
        st.divider()
        left,right=st.columns(2)
        with left:
            if st.button('뒤로',key='f06_rules_back',width='stretch'):
                go('home'); st.rerun()
        with right:
            if st.button('백테스트 조건 설정',key='f06_run_rules',type='primary',width='stretch'):
                go('conditions'); st.rerun()

    elif local_page=='conditions':
        st.title('📈 전략 백테스트')
        st.markdown('<p class="step-caption">1. 분석 조건을 정한 뒤 결과를 확인하세요.</p>',unsafe_allow_html=True)
        saved=st.session_state.setdefault('f06_saved', {})
        with st.form('conditions'):
            c1,c2=st.columns(2)
            start=c1.date_input('분석 시작일',value=max(saved.get('start',default_start.date()),common_start.date()),min_value=common_start.date(),max_value=prices.index[-1].date(),key='f06_start_widget')
            end=c2.date_input('분석 종료일',value=min(saved.get('end',default_end.date()),default_end.date()),min_value=common_start.date(),max_value=default_end.date(),key='f06_end_widget')
            st.caption(f'모든 필수 ETF의 공통 데이터 시작일: {common_start:%Y.%m.%d} · 저장된 로컬 데이터로 계산합니다.')
            preset=st.selectbox('거래 가정',list(PRESETS),index=list(PRESETS).index(saved.get('preset','원문 재현 가정')),key='f06_preset_widget')
            st.caption('보수적 검증: FRED 입력 하루 지연 + 다음 거래일 종가 체결 + 완전 교체당 비용 10bp. 원문 재현 가정: 당일 종가 신호·체결, 비용 0.')
            initial=st.number_input('시작 금액 (USD)',min_value=100.0,value=saved.get('initial',10000.0),step=1000.0,key='f06_initial_widget')
            with st.expander('직접 설정 · 파라미터와 거래 조건'):
                st.caption('파라미터는 모든 거래 가정에 적용됩니다. 아래 거래 지연·비용은 ‘직접 설정’에서만 적용됩니다.')
                a,b=st.columns(2)
                threshold=a.number_input('기대인플레이션 기준 (%)',min_value=0.0,max_value=10.0,value=saved.get('threshold',2.0),step=.1,key='f06_threshold_widget')
                growth=b.number_input('성장 이동평균 (거래일)',min_value=20,max_value=504,value=saved.get('growth',200),step=10,key='f06_growth_widget')
                be=a.number_input('기대인플레이션 변화 (거래일)',min_value=2,max_value=252,value=saved.get('be',60),step=10,key='f06_be_widget')
                asset=b.number_input('바스켓 기울기 (거래일)',min_value=2,max_value=252,value=saved.get('asset',60),step=10,key='f06_asset_widget')
                lag=a.selectbox('FRED 입력 지연 (거래일)',[0,1,2],index=saved.get('lag',1),key='f06_lag_widget')
                execution=b.selectbox('신호 후 체결 지연 (거래일)',[0,1,2],index=saved.get('execution',1),key='f06_execution_widget')
                cost=a.number_input('완전 교체당 비용 (bp)',min_value=0.0,max_value=100.0,value=saved.get('cost',10.0),step=5.0,key='f06_cost_widget')
            st.form_submit_button('↺ 조건 초기화',key='f06_reset_submit',on_click=reset_conditions)
            st.divider()
            left_action,right_action=st.columns(2)
            with left_action:
                go_back=st.form_submit_button('뒤로',key='back_f06_conditions',width='stretch')
            with right_action:
                submitted=st.form_submit_button('테스트 실행',type='primary',key='run_f06_conditions',width='stretch')
        if go_back:
            go('home'); st.rerun(); return
        if submitted:
            if start>=end:
                st.error('종료일을 시작일보다 뒤로 설정해 주세요.')
            else:
                config=PRESETS[preset]
                if preset=='직접 설정': config=replace(config,fred_lag=lag,execution_lag=execution,cost_bps=cost)
                config=replace(config,threshold=threshold,growth_window=growth,breakeven_window=be,asset_window=asset)
                try:
                    with st.spinner('월말 신호와 일별 수익률을 계산하고 있습니다…'):
                        result=calculate(str(start),str(end),config,stamp)
                        if len(result[0])<22: raise ValueError('실제 투자 기간이 한 달 이상 되도록 기간을 늘려주세요.')
                    st.session_state['f06_saved']=dict(start=start,end=end,preset=preset,initial=initial,threshold=threshold,growth=growth,be=be,asset=asset,lag=lag,execution=execution,cost=cost)
                    st.session_state['f06_result']=result
                    st.session_state['f06_config']=config
                    go('results'); st.rerun()
                except Exception as error:
                    st.error(f'계산을 완료하지 못했습니다. 기간 또는 데이터 상태를 확인해 주세요. 상세: {error}')

    elif local_page=='results':
        if 'f06_result' not in st.session_state or 'f06_config' not in st.session_state:
            go('conditions'); st.rerun(); return
        st.title('📈 전략 백테스트 결과')
        nav,spy,events,regimes,sig=st.session_state['f06_result']
        cfg=st.session_state['f06_config']
        initial=st.session_state.setdefault('f06_saved', {})['initial']
        st.caption(f'실제 투자 기간 {nav.index[0]:%Y.%m.%d} – {nav.index[-1]:%Y.%m.%d} · FRED {cfg.fred_lag}일 지연 · 체결 {cfg.execution_lag}일 지연 · 비용 {cfg.cost_bps:g}bp')
        st.caption('시작일 이후 첫 유효 월말 신호에 진입합니다. 이동평균과 기대인플레이션 준비 기간 때문에 요청한 시작일보다 늦을 수 있습니다.')
        if cfg.execution_lag==0:
            st.info('당일 종가로 신호를 계산하고 같은 종가에 체결한 가정입니다. 실제 거래 가능성을 별도로 확인해야 합니다.')
        m=metrics(nav); mm=metrics(nav,True)
        first_row=st.columns(2)
        render_metric(first_row[0], '연복리 수익률',f"{m['CAGR']:.2%}")
        render_metric(first_row[1], '최종 평가금액',f'${nav.iloc[-1]*initial:,.0f}')
        second_row=st.columns(2)
        render_metric(second_row[0], '최대낙폭 · 일별',f"{m['MaxDD']:.2%}")
        render_metric(second_row[1], '최대낙폭 · 월별',f"{mm['MaxDD']:.2%}")
        t1,t2,t3=st.tabs(['포트폴리오 성과','룰 지표와 월말 판정','계산 가정'])
        with t1:
            st.subheader('자산 가치 변화')
            value_chart=pd.DataFrame({'Compass':nav*initial,'SPY':spy*initial})
            holding_annotations={}
            previous_holding=None
            for e in events.itertuples():
                holding=HOLDINGS[e.regime].split(' · ')[0]
                if holding != previous_holding:
                    holding_annotations[e.date]=holding
                    previous_holding=holding
            st.image(static_line_chart(value_chart,['#3182F6','#A5ABB3'],'Portfolio value (USD)',340,holding_annotations),width='stretch')
            metric_dataframe(st, formatted_stats(nav,spy),hide_index=True,width='stretch')
            st.subheader('고점 대비 하락률')
            dd=pd.DataFrame({'Compass':(nav/nav.cummax().clip(lower=1)-1)*100,'SPY':(spy/spy.cummax()-1)*100})
            st.image(static_line_chart(dd,['#3182F6','#A5ABB3'],'Drawdown (%)',240,holding_annotations),width='stretch')
        with t2:
            st.subheader('최근 월말 판정')
            decisions=monthly_decisions(sig,events)
            latest=decisions.iloc[-1]
            st.caption(f"{latest['신호일']:%Y.%m.%d} 월말 신호 → {latest['체결일']:%Y.%m.%d} 체결 → {latest['보유 자산']}")
            cards=st.columns(4)
            render_metric(cards[0], '1. 성장',yes_no(latest['SPY']>latest['SPY 200일 SMA']),f"SPY {latest['SPY']:.2f} / SMA {latest['SPY 200일 SMA']:.2f}")
            render_metric(cards[1], '2. 기대물가 수준',yes_no(latest['T5YIE']>cfg.threshold),f"{latest['T5YIE']:.2f}% / 기준 {cfg.threshold:.2f}%")
            render_metric(cards[2], '3. 기대물가 모멘텀',yes_no(latest['T5YIE 60일 변화']>0),f"60일 변화 {latest['T5YIE 60일 변화']:+.2f}%p")
            render_metric(cards[3], '4. 자산 모멘텀',yes_no(latest['지표 60일 회귀 기울기']>0),f"기울기 {latest['지표 60일 회귀 기울기']:+.6f}")
            st.info(f"결합 결과: 기대물가 수준 {latest['2% 수준 조건']} AND (기대물가 모멘텀 {latest['기대물가 모멘텀']} OR 자산 모멘텀 {latest['자산 모멘텀']}) → 인플레이션 ON {latest['인플레이션 ON']} · {latest['국면']}")

            requested_start=pd.Timestamp(st.session_state.setdefault('f06_saved', {})['start'])
            requested_end=pd.Timestamp(st.session_state.setdefault('f06_saved', {})['end'])
            chart=sig.loc[requested_start:requested_end].copy()
            plot_step=max(1,len(chart)//1400)
            plot_chart=chart.iloc[::plot_step].copy()
            if len(chart) and plot_chart.index[-1] != chart.index[-1]:
                plot_chart=pd.concat([plot_chart,chart.iloc[[-1]]])
            st.subheader('1. 성장 조건 · SPY와 이동평균')
            growth_chart=plot_chart[['spy','sma']].rename(columns={'spy':'SPY','sma':f'SPY {cfg.growth_window}D SMA'})
            st.image(static_line_chart(growth_chart,['#3182F6','#FF8A3D'],height=280),width='stretch')
            st.caption('월말 SPY가 이동평균보다 높으면 성장 상승으로 판정합니다.')

            st.subheader('2–3. 기대인플레이션 수준과 모멘텀')
            breakeven_chart=pd.DataFrame({
                'T5YIE':plot_chart.breakeven,
                f'T5YIE {cfg.breakeven_window}D ago':plot_chart.breakeven_lookback,
                f'Threshold {cfg.threshold:.1f}%':cfg.threshold,
            },index=plot_chart.index)
            st.image(static_line_chart(breakeven_chart,['#3182F6','#8B95A1','#FF4B4B'],'Percent',280),width='stretch')
            st.caption(f'수준 조건은 T5YIE > {cfg.threshold:.1f}%, 기대물가 모멘텀은 현재 T5YIE > {cfg.breakeven_window}거래일 전 T5YIE입니다.')

            st.subheader('4. 업종 바스켓 인플레이션 지표')
            indicator_chart=plot_chart[['inflation_indicator']].rename(columns={'inflation_indicator':'Inflation indicator'})
            st.image(static_line_chart(indicator_chart,['#8B5CF6'],height=260),width='stretch')
            slope_chart=pd.DataFrame({f'{cfg.asset_window}D regression slope':plot_chart.basket_slope,'Zero':0.0},index=plot_chart.index)
            st.image(static_line_chart(slope_chart,['#3182F6','#FF4B4B'],height=240),width='stretch')
            st.caption('기울기가 0보다 크면 자산 모멘텀 조건을 충족합니다. 상승 바스켓은 0.5·XLE + 1/6·XLI + 1/6·XLF + 1/6·XLB, 하락 바스켓은 1/3씩 XLU·XLV·XLP입니다.')

            st.subheader('월말 판정과 다음 달 보유 자산')
            display=decisions.copy()
            for date_column in ['신호일','체결일']:
                display[date_column]=pd.to_datetime(display[date_column]).dt.strftime('%Y-%m-%d')
            display[['SPY','SPY 200일 SMA']]=display[['SPY','SPY 200일 SMA']].round(2)
            display[['T5YIE','60거래일 전 T5YIE','T5YIE 60일 변화']]=display[['T5YIE','60거래일 전 T5YIE','T5YIE 60일 변화']].round(2)
            display['인플레이션 지표']=display['인플레이션 지표'].round(4)
            display['지표 60일 회귀 기울기']=display['지표 60일 회귀 기울기'].round(6)
            regime_position=display.columns.get_loc('국면')
            display.insert(regime_position,'국면 · 성장',display['국면'].map(lambda value:'성장 ↑' if '성장 ↑' in value else '성장 ↓'))
            display.insert(regime_position+1,'국면 · 인플레이션',display['국면'].map(lambda value:'인플레이션 ON' if '인플레이션 ON' in value else '인플레이션 OFF'))
            display=display.drop(columns='국면')
            condition_columns=['성장 조건','2% 수준 조건','기대물가 모멘텀','자산 모멘텀','인플레이션 ON']
            styled=(display.style
                    .map(status_cell,subset=condition_columns)
                    .map(regime_cell,subset=['국면 · 성장','국면 · 인플레이션'])
                    .format({'SPY':'{:.2f}','SPY 200일 SMA':'{:.2f}'}))
            metric_dataframe(st, styled,hide_index=True,width='stretch',height=420)
            st.caption('모든 판정은 월말 마지막 거래일 값입니다. 해당 판정으로 정한 자산을 다음 달에 보유합니다.')
        with t3:
            st.write('USD 기준, 배당·분할 조정 종가를 사용합니다. 세금·환율은 포함하지 않습니다. XLP/IEF는 매월 50:50으로 조정하고 월중에는 비중 변화를 허용합니다.')
            st.write('샤프는 무위험수익률 0%의 일별 산술수익률을 연환산합니다. 월말 기준 위험은 월중 손실을 모두 드러내지 못하므로 일별 낙폭을 함께 표시합니다.')
            st.write('FRED 과거 공개본(vintage)은 사용하지 않았습니다. 하루 지연은 보수적 가정이며 당시 이용 가능 정보만 사용했다는 완전한 증명은 아닙니다.')
            st.json(cfg.__dict__)
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('daily.csv',pd.DataFrame({'compass':nav,'SPY':spy,'regime':regimes}).to_csv())
            z.writestr('trades.csv',events.to_csv(index=False))
            z.writestr('signals.csv',sig.loc[:nav.index[-1]].to_csv())
            z.writestr('config.json',json.dumps({'config':cfg.__dict__,'requested':{k:str(v) for k,v in st.session_state.setdefault('f06_saved', {}).items()}},ensure_ascii=False,indent=2))
        report_png=result_report_image(nav,spy,initial,events)
        with st.expander('원본 결과 파일'):
            st.download_button('CSV 묶음 다운로드',buf.getvalue(),'inflation-compass-results.zip','application/zip',width='stretch')
        st.divider()
        back_col,save_col=st.columns(2)
        with back_col:
            if st.button('뒤로',key='back_f06_results',width='stretch'):
                go('conditions'); st.rerun()
        with save_col:
            st.download_button('결과 저장',report_png,'inflation-compass-result.png','image/png',key='download_f06_result',width='stretch')

    elif local_page=='audit':
        st.title('🔎 검증 결과 살펴보기')
        path=ROOT/'results/summary.csv'
        if not path.exists():
            st.info('검증 보고서를 먼저 생성해 주세요.'); st.code('python run_research.py'); st.stop()
        df=pd.read_csv(path)
        labels={'Original_close':'원문 재현 가정','Trade_next_close':'다음 날 종가 체결','FRED_previous_day':'FRED 하루 지연','Conservative_10bp':'보수적 · 10bp','Conservative_20bp':'보수적 · 20bp','Level_only':'물가 수준만','No_basket':'바스켓 제외','No_FRED':'FRED 제외','SPY':'SPY'}
        frequency=st.radio('위험 측정 주기',['일별','월별'],horizontal=True)
        show=df.loc[df.Frequency==('daily' if frequency=='일별' else 'monthly')].copy()
        if 'Calmar' not in show:
            show['Calmar'] = show['CAGR'] / show['MaxDD'].abs().replace(0, float('nan'))
        if 'Sortino' not in show:
            show['Sortino'] = float('nan')
            st.caption('이 검증 파일에는 하락 변동성 자료가 없어 Sortino는 미계산으로 표시합니다. 검증 보고서를 재생성하면 반영됩니다.')
        show['Model']=show.Model.map(labels)
        for c in ['CAGR','Volatility','MaxDD']: show[c]=show[c].map(lambda x:f'{x:.2%}')
        show['Sharpe_rf0']=show.Sharpe_rf0.map(lambda x:f'{x:.2f}')
        for col in ['Sortino','Calmar']:
            if col in show: show[col]=show[col].map(lambda x:'미계산' if pd.isna(x) else f'{x:.2f}')
        metric_dataframe(st, show[['Model','CAGR','MaxDD','Volatility','Sharpe_rf0','Sortino','Calmar','Start','End']].rename(columns={'Model':'검증 조건','CAGR':'연복리','MaxDD':'최대낙폭','Volatility':'연 변동성','Sharpe_rf0':'샤프 · 무위험 0%','Start':'실제 시작','End':'종료'}),hide_index=True,width='stretch')
        st.caption('첫 진입일이 가정에 따라 다릅니다. 정확히 같은 시작일로 맞춘 비교는 전체 보고서에 있습니다.')
        st.subheader('현재 확인된 점')
        original=df[(df.Model=='Original_close')&(df.Frequency=='daily')].iloc[0]
        conservative=df[(df.Model=='Conservative_10bp')&(df.Frequency=='daily')].iloc[0]
        st.write(f"원문 재현 가정의 연복리는 {original.CAGR:.2%}, 보수적 조건은 {conservative.CAGR:.2%}입니다. 원문 수치와 완전히 같지는 않으며, 시작 시점과 데이터·체결 가정을 추가로 대조할 여지가 있습니다.")
        st.write('과거 표본에서 잘 작동했다는 결과와 앞으로도 통한다는 검증은 구분해야 합니다. 원문 발표 이후의 관측 기간은 아직 짧습니다.')
        manifest_path=ROOT/'data/manifest.json'
        with st.expander('원본 데이터와 수집 이력'):
            if manifest_path.exists():
                st.json(json.loads(manifest_path.read_text(encoding='utf-8')))
            else:
                st.info('현재 배포 파일에는 manifest.json이 포함되어 있지 않습니다. 가격·T5YIE CSV와 검증 결과는 원본 프로젝트 파일을 사용합니다.')
        st.divider()
        left,right=st.columns(2)
        with left:
            if st.button('뒤로',key='f06_audit_back',width='stretch'):
                go('home'); st.rerun()
        with right:
            report_path=ROOT/'results/report.html'
            if report_path.exists():
                st.download_button('검증 보고서 저장',report_path.read_bytes(),'inflation-compass-report.html','text/html',key='f06_audit_download',width='stretch')
            else:
                st.button('검증 보고서 저장',disabled=True,key='f06_audit_download_disabled',width='stretch')

    st.divider()
    st.caption('인플레이션 나침반 · 로컬 연구 앱 | 과거 성과를 분석하는 도구이며 미래 수익을 보장하지 않습니다.')
