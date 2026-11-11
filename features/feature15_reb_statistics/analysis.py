from __future__ import annotations

import pandas as pd


def analyze_statistics(frame: pd.DataFrame, value_kind: str) -> pd.DataFrame:
    """Add comparable period and cumulative changes to an R-ONE series."""
    def analyze_group(group: pd.DataFrame) -> pd.DataFrame:
        result = group.sort_values("날짜").copy()
        if value_kind == "index":
            result["기간 변동률 (%)"] = result["값"].pct_change() * 100
            result["누적 변화 (%)"] = (result["값"] / result["값"].iloc[0] - 1) * 100
        else:
            result["기간 변동률 (%)"] = result["값"]
            result["누적 변화 (%)"] = ((1 + result["값"] / 100).cumprod() - 1) * 100
        return result

    if "지역명" not in frame.columns:
        return analyze_group(frame).reset_index(drop=True)
    groups = [analyze_group(group) for _, group in frame.groupby("지역명", sort=False)]
    return pd.concat(groups, ignore_index=True)


def build_region_comparison(frame: pd.DataFrame, value_kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return common-date normalized series and a fair regional ranking."""
    if frame.empty or "지역명" not in frame.columns:
        return pd.DataFrame(), pd.DataFrame()
    pivot = frame.pivot_table(index="날짜", columns="지역명", values="값", aggfunc="last").sort_index()
    common = pivot.dropna(how="any")
    if common.empty:
        return pd.DataFrame(), pd.DataFrame()
    if value_kind == "index":
        normalized = common.divide(common.iloc[0]).multiply(100)
    else:
        normalized = (1 + common / 100).cumprod().multiply(100)
        normalized = normalized.divide(normalized.iloc[0]).multiply(100)

    elapsed_days = max((normalized.index[-1] - normalized.index[0]).days, 1)
    years = elapsed_days / 365.2425
    rows = []
    for region in normalized.columns:
        cumulative = float(normalized[region].iloc[-1] / normalized[region].iloc[0] - 1)
        annualized = (1 + cumulative) ** (1 / years) - 1 if years > 0 and cumulative > -1 else float("nan")
        rows.append(
            {
                "지역": region,
                "누적 상승률 (%)": cumulative * 100,
                "연환산 상승률 (%)": annualized * 100,
                "시작일": normalized.index[0],
                "종료일": normalized.index[-1],
            }
        )
    ranking = pd.DataFrame(rows).sort_values("누적 상승률 (%)", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    ranking.index.name = "순위"
    return normalized, ranking

