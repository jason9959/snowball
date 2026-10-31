"""Find a practical Korean ETF basket that approximates DBC and PDBC.

The long-history search uses Korean commodity futures ETFs that were all listed by
2011.  Weights are selected on the earlier 70% of monthly observations and then
reported on the untouched later 30% to reduce in-sample overfitting.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from snowball.data.prices import download_yahoo_prices


COMPONENTS = {
    "130680.KS": "TIGER 원유선물Enhanced(H)",
    "137610.KS": "TIGER 농산물선물Enhanced(H)",
    "138910.KS": "KODEX 구리선물(H)",
    "132030.KS": "KODEX 골드선물(H)",
    "144600.KS": "KODEX 은선물(H)",
}

LIVE_VARIANTS = {
    "TIGER원유_단일": {"oil": "130680.KS"},
    "KODEX원유_단일": {"oil": "261220.KS"},
    "TIGER원유_선물형": {
        "oil": "130680.KS", "agriculture": "137610.KS",
        "copper": "138910.KS", "gold": "132030.KS",
    },
    "KODEX원유_선물형": {
        "oil": "261220.KS", "agriculture": "137610.KS",
        "copper": "138910.KS", "gold": "132030.KS",
    },
    "TIGER원유_ACE금": {
        "oil": "130680.KS", "agriculture": "137610.KS",
        "copper": "138910.KS", "gold": "411060.KS",
    },
    "KODEX원유_ACE금": {
        "oil": "261220.KS", "agriculture": "137610.KS",
        "copper": "138910.KS", "gold": "411060.KS",
    },
    "TIGER원유_현물금구리": {
        "oil": "130680.KS", "agriculture": "137610.KS",
        "copper": "160580.KS", "gold": "411060.KS",
    },
    "KODEX원유_현물금구리": {
        "oil": "261220.KS", "agriculture": "137610.KS",
        "copper": "160580.KS", "gold": "411060.KS",
    },
    "TIGER원유_현물금구리_달러10": {
        "oil": "130680.KS", "agriculture": "137610.KS",
        "copper": "160580.KS", "gold": "411060.KS", "usd": "261240.KS",
    },
    "KODEX원유_현물금구리_달러10": {
        "oil": "261220.KS", "agriculture": "137610.KS",
        "copper": "160580.KS", "gold": "411060.KS", "usd": "261240.KS",
    },
}

TARGET_WEIGHTS = {
    "DBC": {"oil": 0.40, "agriculture": 0.20, "copper": 0.20, "gold": 0.20},
    "PDBC": {"oil": 0.40, "agriculture": 0.20, "copper": 0.15, "gold": 0.25},
}

# Broad-commodity-like bounds.  They prevent a fitted result from collapsing
# back into a disguised crude-oil fund.
BOUNDS = {
    "130680.KS": (0.20, 0.50),
    "137610.KS": (0.15, 0.35),
    "138910.KS": (0.10, 0.30),
    "132030.KS": (0.05, 0.25),
    "144600.KS": (0.00, 0.15),
}


def _weight_grid(step: float = 0.05):
    units = round(1.0 / step)
    symbols = tuple(COMPONENTS)
    ranges = [
        range(round(BOUNDS[s][0] / step), round(BOUNDS[s][1] / step) + 1)
        for s in symbols
    ]
    for candidate in itertools.product(*ranges):
        if sum(candidate) == units:
            yield np.asarray(candidate, dtype=float) / units


def _metrics(portfolio: pd.Series, target: pd.Series) -> dict[str, float]:
    frame = pd.concat([portfolio.rename("portfolio"), target.rename("target")], axis=1).dropna()
    difference = frame["portfolio"] - frame["target"]
    return {
        "months": float(len(frame)),
        "correlation": float(frame["portfolio"].corr(frame["target"])),
        "tracking_error": float(difference.std(ddof=1) * np.sqrt(12)),
        "beta": float(frame["portfolio"].cov(frame["target"]) / frame["target"].var()),
        "portfolio_cagr": float((1.0 + frame["portfolio"]).prod() ** (12 / len(frame)) - 1.0),
        "target_cagr": float((1.0 + frame["target"]).prod() ** (12 / len(frame)) - 1.0),
    }


def _live_weight_grid(step: float = 0.05):
    bounds = ((0.25, 0.50), (0.15, 0.30), (0.10, 0.25), (0.10, 0.30), (0.00, 0.15))
    units = round(1.0 / step)
    ranges = [range(round(low / step), round(high / step) + 1) for low, high in bounds]
    for candidate in itertools.product(*ranges):
        if sum(candidate) == units:
            yield np.asarray(candidate, dtype=float) / units


def analyze(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    live_symbols = tuple(
        dict.fromkeys(symbol for variant in LIVE_VARIANTS.values() for symbol in variant.values())
    )
    symbols = tuple(dict.fromkeys(tuple(COMPONENTS) + live_symbols + ("DBC", "PDBC", "KRW=X")))
    bundle = download_yahoo_prices(symbols, pd.Timestamp(start), pd.Timestamp(end))
    prices = bundle.adjusted_close
    monthly_prices = prices.resample("ME").last()
    returns = monthly_prices.pct_change()
    component_returns = returns[list(COMPONENTS)].dropna()
    weights = list(_weight_grid())
    rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []

    for target_ticker in ("DBC", "PDBC"):
        for basis in ("USD", "KRW"):
            target = returns[target_ticker]
            if basis == "KRW":
                target = (1.0 + target) * (1.0 + returns["KRW=X"]) - 1.0
            frame = component_returns.join(target.rename("target"), how="inner").dropna()
            split = max(24, int(len(frame) * 0.70))
            train = frame.iloc[:split]
            test = frame.iloc[split:]
            matrix = train[list(COMPONENTS)].to_numpy()
            target_values = train["target"].to_numpy()
            best_weight = min(
                weights,
                key=lambda w: np.std(matrix @ w - target_values, ddof=1),
            )
            row: dict[str, object] = {"target": target_ticker, "basis": basis}
            for symbol, weight in zip(COMPONENTS, best_weight):
                row[f"weight_{symbol.removesuffix('.KS')}"] = weight
            for label, sample in (("train", train), ("test", test), ("all", frame)):
                metric = _metrics(
                    sample[list(COMPONENTS)].mul(best_weight, axis=1).sum(axis=1),
                    sample["target"],
                )
                row.update({f"{label}_{key}": value for key, value in metric.items()})
            rows.append(row)
            selected.append({
                "target": target_ticker,
                "basis": basis,
                "train_start": train.index.min().date(),
                "train_end": train.index.max().date(),
                "test_start": test.index.min().date(),
                "test_end": test.index.max().date(),
            })
    live_rows: list[dict[str, object]] = []
    for target_ticker, target_weights in TARGET_WEIGHTS.items():
        for basis in ("USD", "KRW"):
            target = returns[target_ticker]
            if basis == "KRW":
                target = (1.0 + target) * (1.0 + returns["KRW=X"]) - 1.0
            for variant_name, variant in LIVE_VARIANTS.items():
                variant_symbols = list(variant.values())
                frame = returns[variant_symbols].join(target.rename("target"), how="inner").dropna()
                frame = frame.loc[frame.index >= pd.Timestamp("2022-01-31")]
                applied_weights = dict(target_weights)
                if tuple(variant) == ("oil",):
                    applied_weights["oil"] = 1.0
                elif "usd" in variant:
                    applied_weights["oil"] -= 0.05
                    applied_weights["gold"] -= 0.05
                    applied_weights["usd"] = 0.10
                weight_vector = np.asarray([applied_weights[role] for role in variant])
                portfolio = frame[variant_symbols].mul(weight_vector, axis=1).sum(axis=1)
                metric = _metrics(portfolio, frame["target"])
                live_rows.append({
                    "target": target_ticker,
                    "basis": basis,
                    "variant": variant_name,
                    "start": frame.index.min().date(),
                    "end": frame.index.max().date(),
                    **metric,
                })
    optimized_rows: list[dict[str, object]] = []
    for target_ticker in ("DBC", "PDBC"):
        target = (1.0 + returns[target_ticker]) * (1.0 + returns["KRW=X"]) - 1.0
        for oil_name, oil_symbol in (("TIGER", "130680.KS"), ("KODEX", "261220.KS")):
            symbols = [oil_symbol, "137610.KS", "160580.KS", "411060.KS", "261240.KS"]
            frame = returns[symbols].join(target.rename("target"), how="inner").dropna()
            frame = frame.loc[frame.index >= pd.Timestamp("2022-01-31")]
            split = min(36, max(24, int(len(frame) * 0.60)))
            train, test = frame.iloc[:split], frame.iloc[split:]
            matrix = train[symbols].to_numpy()
            target_values = train["target"].to_numpy()
            best_weight = min(
                _live_weight_grid(),
                key=lambda weight: np.std(matrix @ weight - target_values, ddof=1),
            )
            row: dict[str, object] = {
                "target": target_ticker, "oil": oil_name,
                "weight_oil": best_weight[0], "weight_agriculture": best_weight[1],
                "weight_copper": best_weight[2], "weight_gold": best_weight[3],
                "weight_usd": best_weight[4],
            }
            for label, sample in (("train", train), ("test", test), ("all", frame)):
                metric = _metrics(sample[symbols].mul(best_weight, axis=1).sum(axis=1), sample["target"])
                row.update({f"{label}_{key}": value for key, value in metric.items()})
            optimized_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(selected), pd.DataFrame(live_rows), pd.DataFrame(optimized_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--output", default="research/commodity_basket_analysis.csv")
    args = parser.parse_args()
    result, periods, live, optimized = analyze(args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    periods.to_csv(output.with_name("commodity_basket_periods.csv"), index=False, encoding="utf-8-sig")
    live.to_csv(output.with_name("commodity_basket_live_variants.csv"), index=False, encoding="utf-8-sig")
    optimized.to_csv(output.with_name("commodity_basket_live_optimized.csv"), index=False, encoding="utf-8-sig")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nLive variants since 2022")
    print(live.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nOptimized live basket, KRW basis")
    print(optimized.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
