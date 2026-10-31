"""Compare Korean-listed long-duration Treasury candidates with EDV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from snowball.data.prices import download_yahoo_prices
from research.analyze_domestic_proxies import _mapped_returns


CANDIDATES = {
    "304660.KS": "KODEX 미국30년국채울트라선물(H)",
    "458250.KS": "TIGER 미국30년국채스트립액티브(합성 H)",
    "453850.KS": "ACE 미국30년국채액티브(H)",
    "461600.KS": "SOL 미국30년국채액티브(H)",
    "484790.KS": "KODEX 미국30년국채액티브(H)",
    "476760.KS": "ACE 미국30년국채액티브",
}


def _metrics(aligned: pd.DataFrame, period: str) -> dict[str, object]:
    if period == "weekly":
        returns = (1.0 + aligned).resample("W-FRI").prod() - 1.0
        scale = np.sqrt(52)
    else:
        returns = (1.0 + aligned).resample("ME").prod() - 1.0
        scale = np.sqrt(12)
    correlations = {
        "USD": returns["Domestic"].corr(returns["USD"]),
        "KRW": returns["Domestic"].corr(returns["KRW"]),
    }
    basis = max(correlations, key=correlations.get)
    reference = returns[basis]
    difference = returns["Domestic"] - reference
    return {
        f"{period} 기준": basis,
        f"{period} 상관": correlations[basis],
        f"{period} 추적오차": difference.std(ddof=1) * scale,
        f"{period} 베타": returns["Domestic"].cov(reference) / reference.var(),
    }


def analyze(start: str, end: str) -> pd.DataFrame:
    symbols = ("EDV", "KRW=X", *CANDIDATES)
    close = download_yahoo_prices(
        symbols, pd.Timestamp(start), pd.Timestamp(end),
    ).adjusted_close
    rows = []
    for ticker, name in CANDIDATES.items():
        aligned = _mapped_returns(close["EDV"], close[ticker], close["KRW=X"])
        if len(aligned) < 20:
            continue
        rows.append({
            "국내 ETF": ticker.removesuffix(".KS"),
            "국내 ETF명": name,
            "비교 시작": aligned.index.min().date(),
            "비교 종료": aligned.index.max().date(),
            "관측일": len(aligned),
            **_metrics(aligned, "weekly"),
            **_metrics(aligned, "monthly"),
        })
    return pd.DataFrame(rows).sort_values(
        ["monthly 상관", "monthly 추적오차"], ascending=[False, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--output", default="research/edv_candidate_analysis.csv")
    args = parser.parse_args()
    result = analyze(args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
