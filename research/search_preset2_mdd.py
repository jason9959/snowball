"""Rebuild the preset-2 MDD search while keeping preset 1 immutable.

The script intentionally uses the production engine, fixed market-data
snapshots, and the same adjusted-open/adjusted-close execution convention as
the Streamlit app.  It writes every tested combination so the research trail
does not depend on a chat transcript.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from pathlib import Path

import pandas as pd

from snowball.data.market_data import (
    correct_estimated_sp500_dividend_yield,
    load_sp500_dividend_yield,
    load_spy_distribution_yield,
    load_treasury_spreads,
)
from snowball.data.prices import download_yahoo_prices
from snowball.metrics import summarize_nav
from snowball.presets import MomentumSpec, PRESETS
from snowball.service import _blend_history, _simulate_blended_portfolio, run_preset


ROOT = Path(__file__).resolve().parent
# The preserved preset-2 diagnostic begins here.  Starting at PDBC's 2014
# launch would let some variants request a defensive signal before PDBC has a
# prior close, which is not a valid apples-to-apples comparison.
START = dt.date(2016, 1, 1)
END = dt.date(2026, 10, 26)
SCORE_TYPES = (
    "1-3-6개월 단순",
    "3-6-12개월 단순",
    "1-3-6-12개월 단순",
    "1-3-6-9-12개월 단순",
    "1-3-6-12개월 가중",
)
MA_MONTHS = (3, 6, 9, 12)
DIVIDEND_THRESHOLDS = tuple(round(value / 10, 1) for value in range(12, 21))


def blend(first: dict, second: dict, prices) -> dict[str, float]:
    common = pd.concat(
        [first["nav"].rename("first"), second["nav"].rename("second")],
        axis=1,
        join="inner",
    ).dropna()
    history = _blend_history(first["history"], second["history"], 0.5)
    nav, _, _, _ = _simulate_blended_portfolio(
        history, prices, common.index, 10_000.0, 0.0,
    )
    metrics = summarize_nav(nav)
    trough = (nav / nav.cummax() - 1.0).idxmin()
    return {**metrics, "mdd_date": str(pd.Timestamp(trough).date())}


def market_data() -> dict[str, pd.Series]:
    start = "2008-01-01"
    end = END.isoformat()
    result = load_treasury_spreads({"T10Y3M"}, start, end, mode="fixed")
    multpl = load_sp500_dividend_yield(start, end, mode="fixed")
    spy = load_spy_distribution_yield(start, end, mode="fixed")
    result["SP500_DIV_YIELD"] = correct_estimated_sp500_dividend_yield(multpl, spy)
    return result


def row_for(preset, first_result, prices, market, recipe: str) -> dict[str, object]:
    second = run_preset(preset, prices, market, START, END)
    combined = blend(first_result, second, prices)
    return {
        "recipe": recipe,
        "attack_score": preset.attack_relative.score_type,
        "defense_ma": preset.defense_relative.months,
        "canary_ma": preset.canary_momentum.months,
        "market_combine": preset.market_combine,
        "market_signals": " + ".join(preset.market_signals) or "none",
        "dividend_threshold": preset.dividend_threshold,
        "canary_rounding_digits": preset.canary_decision_percent_digits,
        "blend_total_return": combined["total_return"],
        "blend_cagr": combined["cagr"],
        "blend_volatility": combined["annual_volatility"],
        "blend_mdd": combined["max_drawdown"],
        "blend_mdd_date": combined["mdd_date"],
        "blend_worst_year": combined["worst_year"],
        "preset2_total_return": second["metrics"]["total_return"],
        "preset2_cagr": second["metrics"]["cagr"],
        "preset2_mdd": second["metrics"]["max_drawdown"],
        "common_start": str(second["common_start"].date()),
        "common_end": str(second["common_end"].date()),
    }


def main() -> None:
    first = PRESETS["1. 분산투자 방어형"]
    base = PRESETS["2. 집중투자 공격형"]
    symbols = tuple(dict.fromkeys(first.symbols + base.symbols))
    print("Downloading one shared price snapshot...", flush=True)
    prices = download_yahoo_prices(
        symbols, pd.Timestamp("2008-01-01"), pd.Timestamp(END),
    )
    market = market_data()
    print("Running immutable preset 1...", flush=True)
    first_result = run_preset(first, prices, market, START, END)

    candidates: list[tuple[str, object]] = [("baseline", base)]
    for score in SCORE_TYPES:
        momentum = MomentumSpec("모멘텀 스코어", score_type=score)
        candidates.append((f"attack-{score}", replace(
            base, name=f"attack-{score}",
            attack_relative=momentum, attack_absolute=momentum,
        )))
    for months in MA_MONTHS:
        momentum = MomentumSpec("이동평균", months)
        candidates.append((f"defense-ma{months}", replace(
            base, name=f"defense-ma{months}",
            defense_relative=momentum, defense_absolute=momentum,
        )))
        candidates.append((f"canary-ma{months}", replace(
            base, name=f"canary-ma{months}", canary_momentum=momentum,
        )))
    for threshold in DIVIDEND_THRESHOLDS:
        candidates.append((f"dividend-{threshold:.1f}", replace(
            base, name=f"dividend-{threshold:.1f}", dividend_threshold=threshold,
        )))
    candidates.extend((
        ("both-and", replace(base, name="both-and", market_combine="AND")),
        ("dividend-only", replace(
            base, name="dividend-only", market_signals=("S&P500 배당수익률",),
        )),
        ("spread-only", replace(
            base, name="spread-only",
            market_signals=("장단기 금리차 역전 (10년-3개월)",),
        )),
        ("canary-only", replace(base, name="canary-only", market_signals=())),
        ("round-1", replace(base, name="round-1", canary_decision_percent_digits=1)),
        ("round-2", replace(base, name="round-2", canary_decision_percent_digits=2)),
    ))
    # Second-stage combinations: the first pass showed that only the dividend
    # threshold materially moved the 2020 drawdown.  Combine that effective
    # axis with each previously discussed momentum/MA alternative.
    for threshold in (1.8, 1.9, 2.0):
        for score in SCORE_TYPES:
            momentum = MomentumSpec("모멘텀 스코어", score_type=score)
            candidates.append((f"joint-div{threshold:.1f}-attack-{score}", replace(
                base, name=f"joint-div{threshold:.1f}-attack-{score}",
                attack_relative=momentum, attack_absolute=momentum,
                dividend_threshold=threshold,
            )))
        for months in MA_MONTHS:
            momentum = MomentumSpec("이동평균", months)
            candidates.append((f"joint-div{threshold:.1f}-defense-ma{months}", replace(
                base, name=f"joint-div{threshold:.1f}-defense-ma{months}",
                defense_relative=momentum, defense_absolute=momentum,
                dividend_threshold=threshold,
            )))
            candidates.append((f"joint-div{threshold:.1f}-canary-ma{months}", replace(
                base, name=f"joint-div{threshold:.1f}-canary-ma{months}",
                canary_momentum=momentum, dividend_threshold=threshold,
            )))

    rows: list[dict[str, object]] = []
    unique: set[tuple[object, ...]] = set()
    for index, (recipe, candidate) in enumerate(candidates, start=1):
        key = (
            candidate.attack_relative, candidate.defense_relative,
            candidate.canary_momentum, candidate.market_combine,
            candidate.market_signals, candidate.dividend_threshold,
            candidate.canary_decision_percent_digits,
        )
        if key in unique:
            continue
        unique.add(key)
        print(f"Core test {index}/{len(candidates)}: {recipe}", flush=True)
        rows.append(row_for(candidate, first_result, prices, market, recipe))
        pd.DataFrame(rows).to_csv(
            ROOT / "preset2_mdd_search_results.partial.csv",
            index=False, encoding="utf-8-sig",
        )

    frame = pd.DataFrame(rows).drop_duplicates(
        [
            "attack_score", "defense_ma", "canary_ma", "market_combine",
            "market_signals", "dividend_threshold", "canary_rounding_digits",
        ]
    )
    frame = frame.sort_values(["blend_mdd", "blend_cagr"], ascending=[False, False])
    frame.to_csv(ROOT / "preset2_mdd_search_results.csv", index=False, encoding="utf-8-sig")

    baseline = frame[
        (frame["recipe"] == "baseline")
        & (frame["attack_score"] == "3-6-12개월 단순")
        & (frame["defense_ma"] == 6)
        & (frame["canary_ma"] == 12)
        & (frame["dividend_threshold"] == 1.6)
    ].iloc[0]
    print("\nBASELINE")
    print(baseline.to_string())
    print("\nTOP 20 BY LOWEST DRAWDOWN")
    print(frame.head(20).to_string(index=False))
    print("\nTOP WITH CAGR >= BASELINE CAGR")
    print(
        frame[frame["blend_cagr"] >= baseline["blend_cagr"]]
        .head(20).to_string(index=False)
    )
    print(f"\nSaved {len(frame)} unique tests to {ROOT / 'preset2_mdd_search_results.csv'}")


if __name__ == "__main__":
    main()
