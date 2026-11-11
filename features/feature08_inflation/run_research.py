"""Run cached-data research and produce a readable Korean HTML report."""
from pathlib import Path
from dataclasses import replace
import html
import numpy as np
import pandas as pd
from compass import Config, signals, backtest, metrics

ROOT = Path(__file__).resolve().parent

def main():
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    prices = pd.read_csv(ROOT/'data/prices.csv', index_col=0, parse_dates=True)
    prices = prices.loc[:'2026-09-09']
    fred = pd.read_csv(ROOT/'data/T5YIE.csv', index_col=0, parse_dates=True).iloc[:,0]
    fred = pd.to_numeric(fred, errors='coerce')
    base = Config()
    configs = {
        'Original_close':base,
        'Trade_next_close':replace(base,execution_lag=1),
        'FRED_previous_day':replace(base,fred_lag=1),
        'Conservative_10bp':replace(base,fred_lag=1,execution_lag=1,cost_bps=10),
        'Conservative_20bp':replace(base,fred_lag=1,execution_lag=1,cost_bps=20),
        'Level_only':replace(base,variant='level_only'),
        'No_basket':replace(base,variant='no_basket'),
        'No_FRED':replace(base,variant='no_fred'),
    }
    rows=[]
    runs={}
    for name, cfg in configs.items():
        sig=signals(prices,fred,cfg)
        nav,spy,events,regimes=backtest(prices,sig,'2003-01-01','2026-06-30',cfg)
        runs[name]=(nav,spy,events,regimes)
        sig.to_csv(out/f'{name}_signals.csv')
        events.to_csv(out/f'{name}_trades.csv',index=False)
        pd.concat([nav,spy.rename('SPY'),regimes],axis=1).to_csv(out/f'{name}_daily.csv')
        for frequency in ['daily','monthly']:
            rows.append({'Model':name,'Frequency':frequency,**metrics(nav,frequency=='monthly')})
    nav,spy,events,regimes=runs['Original_close']
    for frequency in ['daily','monthly']:
        rows.append({'Model':'SPY','Frequency':frequency,**metrics(spy,frequency=='monthly')})
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'summary.csv',index=False)
    # Match all comparison runs to a common close, dropping initial setup returns.
    common_start=max(v[0].index[0] for v in runs.values())
    matched=[]
    for name,(v,b,_,_) in runs.items():
        v=v.loc[common_start:]; v=v/v.iloc[0]
        matched.append({'Model':name,**metrics(v)})
    b=spy.loc[common_start:]; b=b/b.iloc[0]
    matched.append({'Model':'SPY',**metrics(b)})
    matched=pd.DataFrame(matched)
    matched.to_csv(out/'matched_summary.csv',index=False)
    annual={}
    for name,(v,_,_,_) in runs.items():
        r=v.pct_change(fill_method=None).fillna(0)
        r.iloc[0]=v.iloc[0]-1
        annual[name]=(1+r).groupby(r.index.year).prod()-1
    r=spy.pct_change(fill_method=None).fillna(0)
    annual['SPY']=(1+r).groupby(r.index.year).prod()-1
    annual=pd.DataFrame(annual)
    annual.to_csv(out/'annual.csv')
    sensitivity=[]
    for key, values in [('threshold',[1.8,2.0,2.2]),('breakeven_window',[40,60,80]),
                       ('asset_window',[40,60,80]),('growth_window',[100,120,150,200,252])]:
        for value in values:
            cfg=replace(base,**{key:value})
            v,_,_,_=backtest(prices,signals(prices,fred,cfg),'2003-01-01','2026-06-30',cfg)
            sensitivity.append({'Parameter':key,'Value':value,**metrics(v,True)})
    sensitivity=pd.DataFrame(sensitivity)
    sensitivity.to_csv(out/'sensitivity.csv',index=False)
    # Latest completed month-end signal, clearly separate from the article period.
    latest_sig=signals(prices,fred,replace(base,fred_lag=1))
    completed=latest_sig.loc[latest_sig.index.to_period('M') < prices.index[-1].to_period('M')]
    latest=completed.iloc[-1]
    def table(df, index=False):
        df=df.copy()
        for col in ['CAGR','Volatility','MaxDD']:
            if col in df: df[col]=df[col].map(lambda x:f'{x:.2%}')
        for col in ['Sharpe_rf0','Terminal']:
            if col in df: df[col]=df[col].map(lambda x:f'{x:.3f}')
        return df.to_html(index=index,border=0,escape=True)
    report='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Inflation Compass 검증</title>
    <style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f6f9;color:#172331;max-width:1200px;margin:40px auto;padding:0 24px}section{background:white;padding:24px;margin:20px 0;border-radius:12px;overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:14px;width:100%}td,th{padding:9px;text-align:right;border-bottom:1px solid #dde3eb}th{background:#edf2f8}h1,h2{line-height:1.3}code{background:#edf2f8;padding:2px 5px}a{color:#175cac}</style>
    <h1>Inflation Compass · 독립 검증</h1>
    <p>공개 가격과 FRED 데이터를 이용한 연구용 백테스트. 실행 결과와 저자의 주장을 구분합니다.</p>
    <section><h2>무엇을 검증했나</h2><p>기존 stock_info의 레이아웃을 참고한 별도의 로컬 앱입니다. 월말 국면 판정과 동적 ETF 교체를 독립 계산 엔진으로 구현했습니다. app.py에서 조건을 바꾸고 백테스트를 실행할 수 있습니다.</p>
    <p>신호: SPY 200일 평균, T5YIE 2% 기준 및 60일 변화, 업종 바스켓 비율의 60일 회귀 기울기. 성장/인플레이션 조합에 따라 XLE, XLK, XLU, XLP·IEF를 보유합니다.</p>
    <p><a href="https://cssanalytics.wordpress.com/2026/07/27/the-inflation-compass-model/">원문</a>의 성과 주장은 연복리 23.5%, 월말 최대낙폭 −16.2%이며, 저자는 댓글에서 일별 낙폭 −23.6%를 제시했습니다. 아래는 다운로드한 데이터로 별도 계산한 결과입니다.</p></section>
    <section><h2>계산 결과</h2><p>수익률은 USD 기준 배당·분할 조정 종가를 사용합니다. Sharpe_rf0는 무위험수익률을 0으로 둔 산술 평균 수익률/표준편차입니다. 월별과 일별 위험을 구분합니다.</p>'''+table(summary)+'''</section>
    <section><h2>같은 시작일로 맞춘 비교</h2><p>거래 지연에 따라 첫 진입일이 달라지므로 공통 시작일 종가에서 모두 1로 다시 맞췄습니다. 초기 진입 수익·비용은 이 표에서 제외됩니다.</p>'''+table(matched)+'''</section>
    <section><h2>거래·데이터 가정</h2><ul>
    <li>Original_close: 월말 종가로 신호를 계산하고 같은 종가에 체결한 가상 재현. 실거래 가능성을 보장하지 않습니다.</li>
    <li>Trade_next_close: 월말 신호 다음 거래일 종가에 체결합니다. 기존 자산은 체결일까지 유지합니다.</li>
    <li>FRED_previous_day: 주식 거래일 기준 FRED 입력을 하루 늦춥니다.</li>
    <li>Conservative: FRED 하루 지연과 다음 거래일 종가 체결을 함께 적용합니다. 완전 교체당 비용 10/20bp. 첫 진입과 혼합 포트폴리오 재조정에도 회전율에 비례해 비용을 적용합니다.</li>
    <li>XLP/IEF는 월말 50:50으로 조정 후 월중 비중 변화를 허용합니다. 매일 50:50으로 되돌리지 않습니다.</li>
    <li>T5YIE는 실제 관측이 시작된 후 60거래일의 준비 기간을 요구합니다. 원문의 2003년 전체 수익률과 시작 기간이 다를 수 있습니다.</li>
    <li>과거 시점별 공개본(ALFRED vintage)은 확보하지 않았습니다. FRED 하루 지연은 공개 시각 문제를 줄이는 가정이며 완전한 당시 정보 검증은 아닙니다.</li>
    <li>SPY 신호에도 조정 종가를 사용했습니다. 원문은 가격/총수익 지수, 샤프의 무위험수익률 등 일부 세부 가정을 완전히 명시하지 않습니다.</li>
    </ul></section><section><h2>연도별 수익률</h2><p>첫 해는 부분 기간, 2026년은 6월 말까지입니다.</p>'''+annual.map(lambda x:f'{x:.2%}').to_html(border=0)+'''</section>
    <section><h2>파라미터 민감도</h2><p>한 번에 한 변수만 변경했습니다. 시작일이 달라질 수 있으므로 Start 열을 함께 보세요. 최적값 선택이나 독립 표본 검증을 의미하지 않습니다.</p>'''+table(sensitivity)+'''</section>
    <section><h2>해석 시 확인할 점</h2><p>연준의 2% 목표는 <a href="https://www.federalreserve.gov/faqs/economy_14400.htm">PCE 물가 기준</a>입니다. 따라서 TIPS 기반 기대인플레이션의 2%와 동일한 개념으로 해석하면 안 됩니다. 이번 검증은 원문의 임계값을 그대로 사용합니다.</p>
    <p><a href="https://fred.stlouisfed.org/series/T5YIE">FRED T5YIE</a>는 5년 명목금리와 물가연동채 금리 차이에 기반합니다. 현재 공개 데이터는 과거 공개 시각을 직접 입증하지 않습니다.</p>
    <p>이 표본은 전략이 만들어진 뒤 선택한 과거 구간입니다. 파라미터 민감도와 데이터 지연 검사가 좋아도 표본 외 검증을 대신하지 못합니다. CPI 사후 구간 분석, 국면별 통계 검정, 2003년 이전 대체 자료 검증은 이번 범위에 포함하지 않았습니다.</p></section>
    <section><h2>최근 완료 월말 신호</h2><p>역사적 성과표와 분리한 참고 관측치입니다. 실제 거래 지시가 아닙니다.</p>'''+html.escape(str(latest.name.date())+' / '+str(latest['regime']))+'''</section>
    <section><h2>재실행과 감사 자료</h2><p><code>python download_data.py</code>로 입력을 갱신하고 <code>python run_research.py</code>로 다시 계산합니다. data/manifest.json에 원본 URL·해시·수집 시각, results/에 신호·거래·일별 NAV·연도별 수익률이 저장됩니다.</p></section></html>'''
    (out/'report.html').write_text(report,encoding='utf-8')
    print(summary.to_string(index=False))
    print('REPORT:',out/'report.html')

if __name__=='__main__': main()
