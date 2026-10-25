"""Stable, CSV-first loaders for dynamic-portfolio market signals."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable

import pandas as pd
import requests


SPREAD_COLUMNS = {
    "T10Y2Y": "2e2963b9-dc37-455b-8e66-b43a3ed4ff9e",
    "T10Y3M": "6ee33d8e-45c0-4a5a-b1a4-96574058cb46",
}
PACKAGED_SPREAD_CSV = Path(__file__).with_name("data") / "treasury_spreads_monthly.csv"
RUNTIME_SPREAD_CSV = Path(".cache") / "treasury_spreads_monthly.csv"
MULTPL_DIVIDEND_URL = "https://www.multpl.com/s-p-500-dividend-yield/table/by-month"
PACKAGED_DIVIDEND_CSV = Path(__file__).with_name("data") / "sp500_dividend_yield_monthly.csv"
RUNTIME_DIVIDEND_CSV = Path(".cache") / "sp500_dividend_yield_monthly.csv"


def parse_multpl_dividend_yield(html_text: str) -> pd.DataFrame:
    """Parse Multpl's public monthly dividend-yield table.

    ``Estimate`` is retained so callers can disclose which observations were
    projections.  Duplicate observations in a month are resolved to the
    latest dated row (for example, a current estimate and a month-end
    estimate shown together).
    """
    table_match = re.search(
        r"<table[^>]*\bid=[\"']datatable[\"'][^>]*>(.*?)</table>",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if table_match is None:
        raise ValueError("Multpl 응답에서 월별 배당수익률 표를 찾지 못했습니다.")

    records: list[dict[str, object]] = []
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table_match.group(1), flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<td\b[^>]*>(.*?)</td>", row_html, flags=re.IGNORECASE | re.DOTALL)
        if len(cells) < 2:
            continue
        date_text = html.unescape(re.sub(r"<[^>]+>", " ", cells[0])).strip()
        value_text = html.unescape(re.sub(r"<[^>]+>", " ", cells[1])).strip()
        value_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", value_text)
        date = pd.to_datetime(date_text, errors="coerce")
        if pd.isna(date) or value_match is None:
            continue
        records.append({
            "ObservationDate": pd.Timestamp(date),
            "SP500_DIV_YIELD": float(value_match.group(1)),
            "Estimate": bool(re.search(r"title=[\"']Estimate[\"']", cells[1], flags=re.IGNORECASE)),
        })
    if not records:
        raise ValueError("Multpl 월별 배당수익률 표에 유효한 값이 없습니다.")

    frame = pd.DataFrame(records).sort_values("ObservationDate")
    frame["Date"] = frame["ObservationDate"].dt.to_period("M").dt.to_timestamp("M")
    return frame.groupby("Date", as_index=False).last()[
        ["Date", "SP500_DIV_YIELD", "Estimate", "ObservationDate"]
    ]


def _read_dividend_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["SP500_DIV_YIELD", "Estimate", "ObservationDate"])
    frame = pd.read_csv(path)
    required = {"Date", "SP500_DIV_YIELD"}
    if not required.issubset(frame.columns):
        raise ValueError(f"배당수익률 CSV에 필요한 열이 없습니다: {path}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["SP500_DIV_YIELD"] = pd.to_numeric(frame["SP500_DIV_YIELD"], errors="coerce")
    if "Estimate" not in frame.columns:
        frame["Estimate"] = False
    frame["Estimate"] = frame["Estimate"].astype(str).str.casefold().isin({"true", "1", "yes"})
    if "ObservationDate" not in frame.columns:
        frame["ObservationDate"] = frame["Date"]
    frame["ObservationDate"] = pd.to_datetime(frame["ObservationDate"], errors="coerce")
    frame = frame.dropna(subset=["Date", "SP500_DIV_YIELD"]).set_index("Date").sort_index()
    frame.index = frame.index.to_period("M").to_timestamp("M")
    return frame[["SP500_DIV_YIELD", "Estimate", "ObservationDate"]].groupby(level=0).last()


def _download_multpl_dividend_yield(request_get: Callable = requests.get) -> pd.DataFrame:
    response = request_get(
        MULTPL_DIVIDEND_URL,
        timeout=(8, 30),
        headers={"User-Agent": "stock-simulator/1.0"},
    )
    response.raise_for_status()
    return parse_multpl_dividend_yield(response.text).set_index("Date")


def load_sp500_dividend_yield(
    start_iso: str,
    end_iso: str,
    *,
    packaged_path: Path = PACKAGED_DIVIDEND_CSV,
    cache_path: Path = RUNTIME_DIVIDEND_CSV,
    request_get: Callable = requests.get,
) -> pd.Series:
    """Load monthly S&P 500 dividend yield, updating only missing months."""
    start = pd.Timestamp(start_iso).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end_iso).to_period("M").to_timestamp("M")
    required_months = pd.period_range(start, end, freq="M")
    packaged = _read_dividend_csv(packaged_path)
    try:
        cached = _read_dividend_csv(cache_path)
    except (ValueError, OSError, pd.errors.ParserError):
        cached = pd.DataFrame(columns=["SP500_DIV_YIELD", "Estimate", "ObservationDate"])
    combined = pd.concat([packaged, cached]).groupby(level=0).last().sort_index()
    available = combined["SP500_DIV_YIELD"].dropna().index.to_period("M") if not combined.empty else pd.PeriodIndex([], freq="M")
    missing = required_months.difference(available)
    update_error: Exception | None = None
    if len(missing):
        try:
            fetched = _download_multpl_dividend_yield(request_get=request_get)
            combined = pd.concat([combined, fetched]).groupby(level=0).last().sort_index()
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = cache_path.with_suffix(".tmp")
                combined.rename_axis("Date").reset_index().to_csv(temp_path, index=False, date_format="%Y-%m-%d")
                temp_path.replace(cache_path)
            except OSError:
                pass
        except (requests.RequestException, ValueError, OSError) as error:
            update_error = error

    available = combined["SP500_DIV_YIELD"].dropna().index.to_period("M") if not combined.empty else pd.PeriodIndex([], freq="M")
    unresolved = required_months.difference(available)
    if len(unresolved):
        months = ", ".join(str(month) for month in unresolved[:6])
        suffix = f" Multpl 업데이트 실패: {update_error}" if update_error else ""
        raise ValueError(f"배당수익률 CSV에서 필요한 월을 확보하지 못했습니다 ({months}).{suffix}")

    window = combined.loc[start:end]
    result = window["SP500_DIV_YIELD"].astype(float).rename("SP500_DIV_YIELD")
    result.attrs.update({
        "source": "Multpl",
        "source_url": MULTPL_DIVIDEND_URL,
        "estimate_months": [
            str(period)
            for period in window.index[
                window["Estimate"].fillna(False).astype(bool).to_numpy()
            ].to_period("M")
        ],
    })
    return result


def _read_spread_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["T10Y2Y", "T10Y3M"], dtype=float)
    frame = pd.read_csv(path)
    if "Date" not in frame.columns:
        raise ValueError(f"금리 CSV에 Date 열이 없습니다: {path}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
    for column in SPREAD_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.index = frame.index.to_period("M").to_timestamp("M")
    return frame[list(SPREAD_COLUMNS)].groupby(level=0).last()


def _download_series(
    series_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    request_get: Callable = requests.get,
) -> pd.Series:
    dataset_id = SPREAD_COLUMNS[series_id]
    url = f"https://autario.com/api/v1/public/datasets/{dataset_id}/data"
    response = request_get(
        url,
        params={"limit": 20_000, "filter": f"date:gte:{start.date().isoformat()}", "sort": "date:asc"},
        timeout=(8, 25),
        headers={"User-Agent": "stock-simulator/1.0"},
    )
    response.raise_for_status()
    rows = response.json().get("data", [])
    if not rows:
        raise ValueError(f"{series_id} 갱신 응답이 비어 있습니다.")
    dates = pd.to_datetime([row.get("date") for row in rows], errors="coerce")
    values = pd.to_numeric(pd.Series([row.get("value") for row in rows]), errors="coerce")
    daily = pd.Series(values.to_numpy(), index=dates, name=series_id).dropna().sort_index()
    daily = daily.loc[start:end]
    if daily.empty:
        raise ValueError(f"{series_id} 갱신 응답에 요청 기간 데이터가 없습니다.")
    periods = daily.index.to_period("M")
    monthly = daily.groupby(periods).last().dropna()
    monthly.index = monthly.index.to_timestamp("M")
    return monthly


def load_treasury_spreads(
    series_ids: set[str],
    start_iso: str,
    end_iso: str,
    *,
    packaged_path: Path = PACKAGED_SPREAD_CSV,
    cache_path: Path = RUNTIME_SPREAD_CSV,
    request_get: Callable = requests.get,
) -> dict[str, pd.Series]:
    """Use bundled monthly CSV data and update only missing months on demand.

    Runtime cache writes are best-effort because Streamlit Cloud storage is
    ephemeral.  The bundled CSV remains the deterministic baseline.
    """
    requested = set(series_ids) & set(SPREAD_COLUMNS)
    if not requested:
        return {}
    start = pd.Timestamp(start_iso).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end_iso).to_period("M").to_timestamp("M")
    required_months = pd.period_range(start, end, freq="M")

    packaged = _read_spread_csv(packaged_path)
    try:
        cached = _read_spread_csv(cache_path)
    except (ValueError, OSError, pd.errors.ParserError):
        # A Streamlit instance can stop while replacing its ephemeral cache.
        # A damaged runtime cache must never invalidate the bundled baseline.
        cached = pd.DataFrame(columns=list(SPREAD_COLUMNS), dtype=float)
    combined = pd.concat([packaged, cached]).groupby(level=0).last().sort_index()

    def missing_for(series_id: str) -> pd.PeriodIndex:
        available = combined[series_id].dropna().index.to_period("M")
        return required_months.difference(available)

    missing_before = {series_id: missing_for(series_id) for series_id in requested}
    update_errors: dict[str, Exception] = {}
    if any(len(months) for months in missing_before.values()):
        updated = combined.copy()
        for series_id, months in missing_before.items():
            if not len(months):
                continue
            try:
                fetched = _download_series(
                    series_id,
                    months.min().to_timestamp(how="start"),
                    end,
                    request_get=request_get,
                )
            except (requests.RequestException, ValueError, OSError) as error:
                update_errors[series_id] = error
                continue
            updated = updated.combine_first(fetched.to_frame())
            updated.loc[fetched.index, series_id] = fetched
        combined = updated.sort_index()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_suffix(".tmp")
            combined.rename_axis("Date").reset_index().to_csv(temp_path, index=False, date_format="%Y-%m-%d")
            temp_path.replace(cache_path)
        except OSError:
            pass

    missing_after = {series_id: missing_for(series_id) for series_id in requested}
    unresolved = {
        series_id: [str(month) for month in months[:6]]
        for series_id, months in missing_after.items()
        if len(months)
    }
    if unresolved:
        detail = "; ".join(f"{series_id}: {', '.join(months)}" for series_id, months in unresolved.items())
        update_detail = "; ".join(f"{series_id}: {error}" for series_id, error in update_errors.items())
        suffix = f" 자동 업데이트 실패: {update_detail}" if update_detail else ""
        raise ValueError(f"금리 CSV에서 필요한 월을 확보하지 못했습니다 ({detail}).{suffix}")

    return {
        series_id: combined.loc[start:end, series_id].dropna().rename(series_id)
        for series_id in requested
    }


