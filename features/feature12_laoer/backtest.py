from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class LaoerOutput:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def _mdd(values: pd.Series) -> float:
    dd=values/values.cummax()-1
    return float(dd.min()) if len(dd) else 0.0


def run_v22(data: pd.DataFrame, initial_capital: float, divisions: int=40, fee_pct: float=0.0,
            quarter_stop: bool=False) -> LaoerOutput:
    if initial_capital<=0: raise ValueError('초기 투자금은 0보다 커야 합니다.')
    if divisions<2: raise ValueError('분할 횟수는 2 이상이어야 합니다.')
    if data.empty: raise ValueError('가격 데이터가 없습니다.')
    tranche=float(initial_capital)/int(divisions)
    fee=float(fee_pct)
    cash=float(initial_capital); shares=0.0; cost_basis=0.0
    cycle=1; completed=0; quarter_used=False
    daily=[]; trades=[]; max_t=0.0

    def buy(dt, price, budget, note):
        nonlocal cash, shares, cost_basis
        budget=max(0.0,min(float(budget),cash))
        if budget<=1e-12: return
        net=budget*(1-fee)
        qty=net/price
        cash-=budget; shares+=qty; cost_basis+=budget
        trades.append({'date':dt,'cycle':cycle,'side':'BUY','price':price,'amount':budget,'shares':qty,'note':note})

    def sell(dt, price, qty, note):
        nonlocal cash, shares, cost_basis
        qty=max(0.0,min(float(qty),shares))
        if qty<=1e-12: return
        old_shares=shares
        proceeds=qty*price*(1-fee)
        basis_reduction=cost_basis*(qty/old_shares) if old_shares>0 else 0.0
        shares-=qty; cost_basis=max(0.0,cost_basis-basis_reduction); cash+=proceeds
        trades.append({'date':dt,'cycle':cycle,'side':'SELL','price':price,'amount':proceeds,'shares':qty,'note':note})

    for i,(dt,row) in enumerate(data.sort_index().iterrows()):
        dt=pd.Timestamp(dt); close=float(row['Close']); high=float(row.get('High',close)); low=float(row.get('Low',close))
        start_shares=shares
        # A new cycle starts with one tranche at that day's close.
        if shares<=1e-12:
            buy(dt,close,tranche,'사이클 첫 매수 · 종가')
            quarter_used=False
        else:
            avg=cost_basis/shares if shares>0 else np.nan
            t=cost_basis/tranche if tranche else 0.0
            star=10.0-t/2.0
            # Sell orders are evaluated using prices known before today's buy. Limit sells use High.
            if high>=avg*1.10 and shares>0:
                # If +10% is reached, sell all remaining shares. This subsumes the 1/4 order.
                sell(dt,avg*1.10,shares,'+10% 지정가 · 잔여 전량')
            elif high>=avg*(1+star/100.0) and shares>0:
                sell(dt,avg*(1+star/100.0),shares*0.25,f'별% LOC/부분매도 근사 · {star:+.2f}%')

            if start_shares>0 and shares<=1e-12:
                completed+=1; cycle+=1; quarter_used=False
            else:
                # Optional approximate quarter-stop. One MOC-like 25% sale per high-exposure episode.
                avg=cost_basis/shares if shares>0 else np.nan
                t=cost_basis/tranche if shares>0 else 0.0
                star=10.0-t/2.0
                if quarter_stop and shares>0 and t>=39.1 and not quarter_used:
                    sell(dt,close,shares*0.25,'쿼터손절 근사 · 종가 25% 매도')
                    quarter_used=True
                    avg=cost_basis/shares if shares>0 else np.nan
                    t=cost_basis/tranche if shares>0 else 0.0
                    star=10.0-t/2.0
                if shares>0 and cash>1e-9:
                    if t<20:
                        # Two half-tranche LOC orders; execution price is close when the close satisfies the LOC threshold.
                        if close<=avg:
                            buy(dt,close,tranche*0.5,'전반전 LOC 0.5회 · 평단 이하 종가')
                        if close<=avg*(1+star/100.0):
                            buy(dt,close,tranche*0.5,f'전반전 LOC 0.5회 · 별% {star:+.2f}%')
                    else:
                        if close<=avg*(1+star/100.0):
                            buy(dt,close,tranche,f'후반전 LOC 1회 · 별% {star:+.2f}%')

        value=cash+shares*close
        avg=cost_basis/shares if shares>0 else np.nan
        t=cost_basis/tranche if shares>0 else 0.0
        star=10.0-t/2.0 if shares>0 else np.nan
        max_t=max(max_t,t)
        daily.append({'date':dt,'Close':close,'High':high,'Low':low,'cash':cash,'shares':shares,
                      'cost_basis':cost_basis,'avg_price':avg,'T':t,'star_pct':star,'portfolio_value':value,'cycle':cycle})

    daily_df=pd.DataFrame(daily).set_index('date')
    trades_df=pd.DataFrame(trades)
    final=float(daily_df['portfolio_value'].iloc[-1]); years=max((daily_df.index[-1]-daily_df.index[0]).days/365.25,1/365.25)
    metrics={'initial_capital':float(initial_capital),'final_value':final,'profit':final-initial_capital,
             'return_pct':(final/initial_capital-1)*100,'cagr_pct':((final/initial_capital)**(1/years)-1)*100,
             'mdd_pct':_mdd(daily_df['portfolio_value'])*100,'completed_cycles':completed,
             'max_t':max_t,'ending_cash':cash,'ending_shares':shares,'ending_avg':float(daily_df['avg_price'].iloc[-1]) if shares>0 else np.nan}
    return LaoerOutput(daily_df,trades_df,metrics)
