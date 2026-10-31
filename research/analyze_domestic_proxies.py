"""Measure how closely each Korean execution proxy follows its US source ETF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from snowball.data.prices import download_yahoo_prices
from snowball.domestic import DOMESTIC_EXECUTION_MAP, DOMESTIC_NAMES


def _mapped_returns(
    us_price: pd.Series, domestic_price: pd.Series, usdkrw: pd.Series,
) -> pd.DataFrame:
    """Move each US close-to-close return to the next available KRX session."""
    krx_dates = domestic_price.dropna().index.sort_values()
    us = us_price.dropna().pct_change().dropna()
    fx = usdkrw.reindex(us.index).ffill().pct_change().fillna(0.0)
    positions = krx_dates.searchsorted(us.index, side="right")
    valid = positions < len(krx_dates)
    source = pd.DataFrame({
        "KRXDate": krx_dates[positions[valid]],
        "USD": us.iloc[np.flatnonzero(valid)].to_numpy(),
        "KRW": ((1.0 + us.iloc[np.flatnonzero(valid)].to_numpy())
                * (1.0 + fx.iloc[np.flatnonzero(valid)].to_numpy()) - 1.0),
    })
    grouped = source.groupby("KRXDate")[["USD", "KRW"]].agg(
        lambda values: float((1.0 + values).prod() - 1.0),
    )
    domestic = domestic_price.pct_change().rename("Domestic")
    return grouped.join(domestic, how="inner").dropna()


def analyze(start: str, end: str) -> pd.DataFrame:
    us_symbols = tuple(dict.fromkeys(DOMESTIC_EXECUTION_MAP))
    domestic_symbols = tuple(dict.fromkeys(DOMESTIC_EXECUTION_MAP.values()))
    symbols = us_symbols + domestic_symbols + ("KRW=X",)
    bundle = download_yahoo_prices(symbols, pd.Timestamp(start), pd.Timestamp(end))
    close = bundle.adjusted_close
    rows: list[dict[str, object]] = []
    for us_ticker, domestic_ticker in DOMESTIC_EXECUTION_MAP.items():
        aligned = _mapped_returns(close[us_ticker], close[domestic_ticker], close["KRW=X"])
        if len(aligned) < 20:
            continue
        weekly = (1.0 + aligned).resample("W-FRI").prod() - 1.0
        monthly = (1.0 + aligned).resample("ME").prod() - 1.0
        daily_correlations = {
            "USD": aligned["Domestic"].corr(aligned["USD"]),
            "KRW": aligned["Domestic"].corr(aligned["KRW"]),
        }
        monthly_correlations = {
            "USD": monthly["Domestic"].corr(monthly["USD"]),
            "KRW": monthly["Domestic"].corr(monthly["KRW"]),
        }
        basis = max(monthly_correlations, key=lambda key: monthly_correlations[key])
        reference = monthly[basis]
        difference = monthly["Domestic"] - reference
        rows.append({
            "미국 ETF": us_ticker,
            "국내 ETF": domestic_ticker.removesuffix(".KS"),
            "국내 ETF명": DOMESTIC_NAMES[domestic_ticker],
            "비교 시작": aligned.index.min().date(),
            "비교 종료": aligned.index.max().date(),
            "관측일": len(aligned),
            "일간 USD 상관": daily_correlations["USD"],
            "일간 환율포함 상관": daily_correlations["KRW"],
            "주간 USD 상관": weekly["Domestic"].corr(weekly["USD"]),
            "주간 환율포함 상관": weekly["Domestic"].corr(weekly["KRW"]),
            "월간 USD 상관": monthly_correlations["USD"],
            "월간 환율포함 상관": monthly_correlations["KRW"],
            "더 가까운 기준": basis,
            "최대 상관": monthly_correlations[basis],
            "연환산 추적오차": difference.std(ddof=1) * np.sqrt(12),
            "수익률 베타": monthly["Domestic"].cov(reference) / reference.var(),
            "누적 수익률 차이": (
                (1.0 + monthly["Domestic"]).prod() - (1.0 + reference).prod()
            ),
        })
    return pd.DataFrame(rows).sort_values(["최대 상관", "미국 ETF"], ascending=[False, True])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--output", default="research/domestic_proxy_analysis.csv")
    args = parser.parse_args()
    result = analyze(args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
