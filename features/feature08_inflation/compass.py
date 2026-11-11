"""Inflation Compass research engine. No orders, no future-filled observations."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from common.metrics import sortino_ratio, calmar_ratio

ASSETS = ['XLE', 'XLK', 'XLU', 'XLP', 'IEF']
MAP = {'reflation': [1,0,0,0,0], 'goldilocks': [0,1,0,0,0],
       'stagflation': [0,0,1,0,0], 'slowdown': [0,0,0,.5,.5]}

@dataclass(frozen=True)
class Config:
    threshold: float = 2.0
    growth_window: int = 200
    breakeven_window: int = 60
    asset_window: int = 60
    fred_lag: int = 0
    execution_lag: int = 0  # 0 = hypothetical signal-day close, 1 = following close
    cost_bps: float = 0.0  # per unit of one-way turnover, full replacement costs this
    variant: str = 'full'

def signals(prices, fred, config=Config()):
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError('Prices must have a unique ascending date index')
    signal_assets = ['SPY','XLE','XLI','XLF','XLB','XLU','XLV','XLP']
    if prices[signal_assets].isna().any().any() or (prices[signal_assets] <= 0).any().any():
        raise ValueError('Missing/nonpositive signal prices: do not silently fill inputs')
    daily = prices.pct_change(fill_method=None)
    positive = daily.XLE*.5 + (daily.XLI + daily.XLF + daily.XLB)/6
    negative = (daily.XLU + daily.XLV + daily.XLP)/3
    indicator = (1+positive.fillna(0)).cumprod()/(1+negative.fillna(0)).cumprod()
    x = np.arange(config.asset_window, dtype=float)
    x -= x.mean()
    slope = indicator.rolling(config.asset_window).apply(lambda y: np.dot(x, y)/np.dot(x,x), raw=True)
    available = fred.dropna().sort_index()
    # Include non-equity trading dates in the as-of alignment, then apply equity-day lag.
    aligned = available.reindex(available.index.union(prices.index)).ffill().reindex(prices.index)
    source_dates = pd.Series(available.index, index=available.index).reindex(available.index.union(prices.index)).ffill().reindex(prices.index)
    aligned = aligned.shift(config.fred_lag)
    source_dates = source_dates.shift(config.fred_lag)
    sma = prices.SPY.rolling(config.growth_window).mean()
    change = aligned - aligned.shift(config.breakeven_window)
    growth = prices.SPY > sma
    level = aligned > config.threshold
    breakeven_momentum = change > 0
    asset_momentum = slope > 0
    inflation = level & (breakeven_momentum | asset_momentum)
    if config.variant == 'level_only': inflation = aligned > config.threshold
    elif config.variant == 'no_basket': inflation = (aligned > config.threshold) & (change > 0)
    elif config.variant == 'no_fred': inflation = slope > 0
    elif config.variant != 'full': raise ValueError('Unknown variant')
    valid = sma.notna() & slope.notna() & change.notna() & source_dates.notna()
    # Reject stale releases rather than carrying observations through a data outage.
    valid &= (pd.Series(prices.index, index=prices.index)-source_dates).dt.days <= 7
    regime = pd.Series(np.select([growth & inflation, growth & ~inflation, ~growth & inflation],
                                ['reflation','goldilocks','stagflation'], default='slowdown'), index=prices.index)
    result = pd.DataFrame({'spy':prices.SPY,'sma':sma,
        'positive_basket_return':positive,'negative_basket_return':negative,
        'inflation_indicator':indicator,'breakeven':aligned,
        'breakeven_lookback':aligned.shift(config.breakeven_window),
        'fred_source_date':source_dates,'breakeven_change':change,'basket_slope':slope,
        'growth_up':growth,'level_above_target':level,
        'breakeven_momentum_positive':breakeven_momentum,
        'asset_momentum_positive':asset_momentum,
        'inflation_on':inflation,'regime':regime,'valid':valid})
    return result

def backtest(prices, sig, start, end, config=Config()):
    # Find month ends using the full downloaded calendar before slicing the evaluation window.
    idx = prices.index
    monthends = idx.to_period('M') != idx.to_series().shift(-1).dt.to_period('M').to_numpy()
    monthends[-1] = False  # last observed date is not necessarily a complete month
    orders = {}
    for i in np.flatnonzero(monthends):
        if sig.valid.iloc[i] and i+config.execution_lag < len(idx):
            orders[idx[i+config.execution_lag]] = (idx[i], sig.regime.iloc[i])
    begin = pd.Timestamp(start)
    finish = pd.Timestamp(end)
    relevant = [d for d in orders if begin <= d <= finish]
    if not relevant: raise ValueError('No valid month-end signals in requested period')
    first = relevant[0]
    dates = idx[(idx >= first) & (idx <= finish)]
    if prices.loc[dates,ASSETS].isna().any().any() or (prices.loc[dates,ASSETS] <= 0).any().any():
        raise ValueError('Missing/nonpositive holding prices in the investment period')
    units = np.zeros(5)
    cash = 1.0
    values, events, regimes = [], [], []
    current = 'cash'
    price_array = prices.loc[dates, ASSETS].to_numpy(float)
    for date, px in zip(dates, price_array):
        nav = cash + np.dot(units,px)
        if date in orders:
            signal_date, regime = orders[date]
            target = np.array(MAP[regime], dtype=float)
            old = units*px/nav
            turnover = .5*(np.abs(target-old).sum() + cash/nav)
            fee = nav * turnover * config.cost_bps/10000
            nav -= fee
            units = nav*target/px
            cash = 0.0
            events.append({'date':date,'signal_date':signal_date,'regime':regime,
                           'turnover':turnover,'fee':fee,'nav':nav})
            current = regime
        values.append(nav)
        regimes.append(current)
    nav = pd.Series(values,index=dates,name='compass')
    benchmark = prices.loc[dates,'SPY']/prices.loc[first,'SPY']
    return nav, benchmark, pd.DataFrame(events), pd.Series(regimes,index=dates,name='regime')

def metrics(nav, monthly=False):
    # Include initial pre-cost capital in drawdown and first day's return.
    sampled = nav.groupby(nav.index.to_period('M')).last() if monthly else nav
    returns = sampled.pct_change(fill_method=None)
    returns.iloc[0] = sampled.iloc[0]-1
    if monthly and (nav.index.to_period('M') == nav.index[0].to_period('M')).sum() == 1:
        returns = returns.iloc[1:]
    years = (nav.index[-1]-nav.index[0]).days/365.25
    scale = 12 if monthly else 252
    vol = returns.std(ddof=1)*np.sqrt(scale)
    cagr = nav.iloc[-1]**(1/years)-1
    peaks = sampled.cummax().clip(lower=1)
    return {'CAGR':cagr,'Sharpe_rf0':returns.mean()*scale/vol if vol else np.nan,
            'Sortino':sortino_ratio(returns, scale), 'Calmar':cagr / abs((sampled/peaks-1).min()) if (sampled/peaks-1).min() < 0 else np.nan,
            'Volatility':vol,'MaxDD':(sampled/peaks-1).min(),'Terminal':nav.iloc[-1],
            'Start':str(nav.index[0].date()),'End':str(nav.index[-1].date())}
