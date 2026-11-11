"""Common performance and cash-flow metrics used by backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns, periods_per_year: float = 252.0) -> float:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(returns, periods_per_year: float = 252.0, target: float = 0.0) -> float:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    downside = np.minimum(r - target, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(r) else 0.0
    if not len(r) or downside_dev == 0:
        return float("nan")
    return float((r.mean() - target) / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(values) -> float:
    v = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return 0.0
    return float((v / v.cummax() - 1.0).min())


def cagr(values, periods_per_year: float = 252.0) -> float:
    v = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) < 2 or v.iloc[0] <= 0 or v.iloc[-1] <= 0:
        return float("nan")
    years = ((v.index[-1] - v.index[0]).total_seconds() / (365.2425 * 86400)
             if isinstance(v.index, pd.DatetimeIndex) else (len(v) - 1) / periods_per_year)
    return float((v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")


def calmar_ratio(values, periods_per_year: float = 252.0) -> float:
    dd = abs(max_drawdown(values))
    annual = cagr(values, periods_per_year)
    return float(annual / dd) if dd > 0 and np.isfinite(annual) else float("nan")


def mdd_recovery_days(values) -> float:
    """Calendar days from the last pre-MDD high to recovery; NaN if unrecovered."""
    v = pd.Series(values, dtype=float).dropna()
    if v.empty or not isinstance(v.index, pd.DatetimeIndex):
        return float("nan")
    drawdown = v / v.cummax() - 1
    if drawdown.min() >= 0:
        return 0.0
    trough = int(np.argmin(drawdown.to_numpy()))
    peak_value = v.iloc[:trough + 1].max()
    peak = np.flatnonzero(v.iloc[:trough + 1].to_numpy() == peak_value)[-1]
    recovered = np.flatnonzero(v.iloc[trough:].to_numpy() >= peak_value)
    return float((v.index[trough + recovered[0]] - v.index[peak]).days) if len(recovered) else float("nan")


def performance_summary(values, contributions=None, initial_value=None) -> dict:
    """Cash-flow-adjusted daily risk metrics; deposits occur at the day's close."""
    v = pd.Series(values, dtype=float).dropna()
    flows = pd.Series(0.0, index=v.index) if contributions is None else pd.Series(contributions).reindex(v.index).fillna(0.0)
    returns = (v - flows) / v.shift(1) - 1
    base = float(initial_value) if initial_value is not None else float(v.iloc[0] - flows.iloc[0])
    nav = (1 + returns.fillna(0)).cumprod()
    invested = base + float(flows.sum())
    recovery = mdd_recovery_days(nav)
    result = {
        '최종 수익률': f'{(v.iloc[-1] / invested - 1):+.2%}',
        '연환산 수익률': f'{cagr(nav):+.2%}',
        '연환산 변동성': f'{returns.std(ddof=1) * np.sqrt(252):.2%}',
        '최대 낙폭': f'{max_drawdown(nav):.2%}',
        'MDD 회복기간(고점 기준)': '미회복' if pd.isna(recovery) else f'{recovery:.0f}일',
        '샤프지수': sharpe_ratio(returns), 'Sortino': sortino_ratio(returns), 'Calmar': calmar_ratio(nav),
    }
    for key in ('샤프지수', 'Sortino', 'Calmar'):
        result[key] = f'{result[key]:.2f}' if np.isfinite(result[key]) else '-'
    if contributions is not None:
        rate = cashflow_xirr(v.index, flows, v.iloc[-1], base)
        result['적립식 XIRR'] = f'{rate:.2%}' if np.isfinite(rate) else '-'
    return result


def xirr(amounts, dates, guess: float = 0.1) -> float:
    """Annualized return for irregular cash flows (negative deposits, positive ending value)."""
    cash = np.asarray(amounts, dtype=float)
    when = pd.to_datetime(pd.Series(dates)).dt.tz_localize(None).to_numpy()
    valid = np.isfinite(cash)
    cash, when = cash[valid], when[valid]
    if len(cash) < 2 or not (np.any(cash < 0) and np.any(cash > 0)):
        return float("nan")
    days = np.array([(d - when[0]).astype("timedelta64[D]").astype(float) / 365.0 for d in when])
    def f(rate):
        if rate <= -0.999999:
            return np.inf
        return float(np.sum(cash / np.power(1.0 + rate, days)))
    lo, hi = -0.9999, 1.0
    flo, fhi = f(lo), f(hi)
    for _ in range(50):
        if np.sign(flo) != np.sign(fhi):
            break
        hi *= 2
        fhi = f(hi)
    else:
        return float("nan")
    for _ in range(100):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < 1e-8:
            return float(mid)
        if np.sign(flo) == np.sign(fm):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return float((lo + hi) / 2)


def cashflow_xirr(dates, contributions, final_value, initial_value: float = 0.0) -> float:
    dates = list(pd.to_datetime(dates))
    contributions = list(contributions)
    if not dates:
        return float("nan")
    amounts = [-float(initial_value)] + [-float(x) for x in contributions] + [float(final_value)]
    flow_dates = [dates[0]] + dates + [dates[-1]]
    return xirr(amounts, flow_dates)
