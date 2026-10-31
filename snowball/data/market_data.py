"""Stable, CSV-first loaders for dynamic-portfolio market signals."""

from __future__ import annotations

import html
import hashlib
import re
import datetime as dt
from pathlib import Path
from typing import Callable

import pandas as pd

from .fred import download_fred_series
from .http import get as http_get


SPREAD_COLUMNS = {
    "T10Y2Y": "10년-2년 금리차",
    "T10Y3M": "10년-3개월 금리차",
}
PACKAGED_SPREAD_CSV = Path(__file__).with_name("data") / "treasury_spreads_monthly.csv"
RUNTIME_SPREAD_CSV = Path(".cache") / "treasury_spreads_monthly.csv"
MULTPL_DIVIDEND_URL = "https://www.multpl.com/s-p-500-dividend-yield/table/by-month"
PACKAGED_DIVIDEND_CSV = Path(__file__).with_name("data") / "sp500_dividend_yield_monthly.csv"
RUNTIME_DIVIDEND_CSV = Path(".cache") / "sp500_dividend_yield_monthly.csv"
PACKAGED_SPY_DIVIDEND_CSV = Path(__file__).with_name("data") / "spy_distribution_yield_monthly.csv"
RUNTIME_SPY_DIVIDEND_CSV = Path(".cache") / "spy_distribution_yield_monthly.csv"
SNAPSHOT_MODES = {"fixed", "latest", "refresh_missing"}


def _file_snapshot_id(path: Path, label: str) -> str:
    """Return a stable identifier for a bundled or cached data file."""
    if not path.exists():
        return f"{label}-missing"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{label}-{digest}"


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


def _download_multpl_dividend_yield(request_get: Callable = http_get) -> pd.DataFrame:
    response = request_get(
        MULTPL_DIVIDEND_URL,
        timeout=(8, 30),
        headers={"User-Agent": "stock-simulator/1.0"},
    )
    response.raise_for_status()
    frame = parse_multpl_dividend_yield(response.text).set_index("Date")
    frame.attrs.update({
        "source": "Multpl",
        "source_url": getattr(response, "url", MULTPL_DIVIDEND_URL),
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sha256": hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
    })
    return frame


def download_latest_sp500_dividend_snapshot(
    request_get: Callable = http_get,
) -> pd.DataFrame:
    """Download a candidate snapshot without replacing the bundled baseline."""
    return _download_multpl_dividend_yield(request_get=request_get)


def compare_dividend_snapshots(
    baseline: pd.DataFrame | pd.Series,
    candidate: pd.DataFrame | pd.Series,
    start_iso: str,
    end_iso: str,
) -> pd.DataFrame:
    """Compare two dividend vintages by month for display and CSV export."""
    def values(data: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(data, pd.DataFrame):
            series = data["SP500_DIV_YIELD"]
        else:
            series = data
        series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
        series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp("M")
        return series.groupby(level=0).last()

    start = pd.Timestamp(start_iso).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end_iso).to_period("M").to_timestamp("M")
    joined = pd.concat(
        [values(baseline).rename("고정 스냅샷"), values(candidate).rename("최신 공급원")],
        axis=1,
    ).loc[start:end]
    joined["차이"] = joined["최신 공급원"] - joined["고정 스냅샷"]
    joined["상태"] = "동일"
    joined.loc[joined["고정 스냅샷"].isna(), "상태"] = "신규"
    joined.loc[joined["최신 공급원"].isna(), "상태"] = "최신값 없음"
    changed = joined[["고정 스냅샷", "최신 공급원"]].notna().all(axis=1) & joined["차이"].abs().gt(1e-12)
    joined.loc[changed, "상태"] = "변경"
    return joined.rename_axis("관측 월").reset_index()


def load_strategy_market_data(
    signals: tuple[str, ...],
    start_iso: str,
    end_iso: str,
    *,
    mode: str = "refresh_missing",
) -> dict[str, pd.Series]:
    """Load strategy signals through one path for UI and automation."""
    result: dict[str, pd.Series] = {}
    spread_ids: set[str] = set()
    if "장단기 금리차 역전 (10년-2년)" in signals:
        spread_ids.add("T10Y2Y")
    if "장단기 금리차 역전 (10년-3개월)" in signals:
        spread_ids.add("T10Y3M")
    result.update(load_treasury_spreads(spread_ids, start_iso, end_iso, mode=mode))
    if "S&P500 배당수익률" in signals:
        requested_start = pd.Timestamp(start_iso).to_period("M").to_timestamp("M")
        calibration_start = requested_start - pd.DateOffset(months=48)
        multpl = load_sp500_dividend_yield(
            calibration_start.date().isoformat(), end_iso, mode=mode,
        )
        try:
            spy = load_spy_distribution_yield(
                calibration_start.date().isoformat(), end_iso, mode=mode,
            )
            dividend = correct_estimated_sp500_dividend_yield(multpl, spy)
        except (ValueError, OSError, RuntimeError) as error:
            dividend = multpl.copy()
            dividend.attrs.update(dict(multpl.attrs))
            dividend.attrs.update({
                "source": "Multpl (SPY correction unavailable)",
                "spy_correction_error": str(error),
                "corrected_months": [],
                "fallback_estimate_months": list(multpl.attrs.get("estimate_months", [])),
            })
        result["SP500_DIV_YIELD"] = dividend.loc[requested_start:]
        result["SP500_DIV_YIELD"].attrs.update(dict(dividend.attrs))
    return result


def load_sp500_dividend_yield(
    start_iso: str,
    end_iso: str,
    *,
    packaged_path: Path = PACKAGED_DIVIDEND_CSV,
    cache_path: Path = RUNTIME_DIVIDEND_CSV,
    request_get: Callable = http_get,
    mode: str = "refresh_missing",
) -> pd.Series:
    """Load monthly S&P 500 dividend yield from a fixed or latest vintage."""
    if mode not in SNAPSHOT_MODES:
        raise ValueError(f"지원하지 않는 스냅샷 모드: {mode}")
    start = pd.Timestamp(start_iso).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end_iso).to_period("M").to_timestamp("M")
    required_months = pd.period_range(start, end, freq="M")
    packaged = _read_dividend_csv(packaged_path)
    cached = pd.DataFrame(columns=["SP500_DIV_YIELD", "Estimate", "ObservationDate"])
    if mode != "fixed":
        try:
            cached = _read_dividend_csv(cache_path)
        except (ValueError, OSError, pd.errors.ParserError):
            pass
    combined = packaged.copy() if mode == "fixed" else pd.concat([packaged, cached]).groupby(level=0).last().sort_index()
    available = combined["SP500_DIV_YIELD"].dropna().index.to_period("M") if not combined.empty else pd.PeriodIndex([], freq="M")
    missing = required_months.difference(available)
    update_error: Exception | None = None
    fetched_provenance: dict[str, object] = {}
    should_fetch = mode == "latest" or (mode == "refresh_missing" and len(missing))
    if should_fetch:
        try:
            fetched = _download_multpl_dividend_yield(request_get=request_get)
            fetched_provenance = dict(fetched.attrs)
            combined = pd.concat([combined, fetched]).groupby(level=0).last().sort_index()
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = cache_path.with_suffix(".tmp")
                combined.rename_axis("Date").reset_index().to_csv(temp_path, index=False, date_format="%Y-%m-%d")
                temp_path.replace(cache_path)
                snapshot_dir = cache_path.parent / "snapshots"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot_hash = str(fetched.attrs.get("sha256", "unknown"))[:12]
                version_path = snapshot_dir / f"sp500_dividend_yield_{snapshot_hash}.csv"
                if not version_path.exists():
                    fetched.rename_axis("Date").reset_index().to_csv(
                        version_path, index=False, date_format="%Y-%m-%d",
                    )
            except OSError:
                pass
        except (ValueError, OSError) as error:
            update_error = error

    available = combined["SP500_DIV_YIELD"].dropna().index.to_period("M") if not combined.empty else pd.PeriodIndex([], freq="M")
    unresolved = required_months.difference(available)
    if len(unresolved):
        months = ", ".join(str(month) for month in unresolved[:6])
        suffix = f" Multpl 업데이트 실패: {update_error}" if update_error else ""
        mode_hint = " 최신 데이터 모드로 다시 실행할 수 있습니다." if mode == "fixed" else ""
        raise ValueError(f"배당수익률 CSV에서 필요한 월을 확보하지 못했습니다 ({months}).{suffix}{mode_hint}")

    window = combined.loc[start:end]
    result = window["SP500_DIV_YIELD"].astype(float).rename("SP500_DIV_YIELD")
    result.attrs.update({
        "source": "Multpl",
        "source_url": MULTPL_DIVIDEND_URL,
        "snapshot_id": (
            f"multpl-{str(fetched_provenance.get('sha256'))[:12]}"
            if fetched_provenance.get("sha256")
            else _file_snapshot_id(packaged_path, "bundled-dividend")
        ),
        "data_policy": {
            "fixed": "bundled_snapshot_only",
            "latest": "latest_candidate_over_bundled_snapshot",
            "refresh_missing": "bundled_snapshot_then_missing_month_refresh",
        }[mode],
        "snapshot_mode": mode,
        "vintage_mode": "current_history_not_point_in_time",
        "observation_dates": {
            str(index.to_period("M")): (
                None if pd.isna(value) else pd.Timestamp(value).date().isoformat()
            )
            for index, value in window["ObservationDate"].items()
        },
        "estimate_months": [
            str(period)
            for period in window.index[
                window["Estimate"].fillna(False).astype(bool).to_numpy()
            ].to_period("M")
        ],
        **fetched_provenance,
    })
    return result


def calculate_spy_distribution_yield(history: pd.DataFrame) -> pd.DataFrame:
    """Calculate State Street-style trailing-365-day SPY distribution yield.

    Raw close is required: adjusted close already contains distributions and
    would count them twice.  Dividend availability is keyed by ex-date.
    """
    required = {"Close", "Dividends"}
    if history.empty or not required.issubset(history.columns):
        raise ValueError("SPY 가격 자료에 Close와 Dividends 열이 필요합니다.")
    frame = history[list(required)].copy()
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = frame.index.normalize()
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame["Dividends"] = pd.to_numeric(frame["Dividends"], errors="coerce").fillna(0.0)
    frame = frame.sort_index()
    closes = frame["Close"].dropna().groupby(frame["Close"].dropna().index.to_period("M")).last()
    rows: list[dict[str, object]] = []
    for period, close in closes.items():
        month_end = period.to_timestamp("M")
        trailing = frame.loc[
            (frame.index > month_end - pd.Timedelta(days=365))
            & (frame.index <= month_end),
            "Dividends",
        ].sum()
        rows.append({
            "Date": month_end,
            "SPY_TTM_DISTRIBUTION": float(trailing),
            "SPY_CLOSE": float(close),
            "SPY_DISTRIBUTION_YIELD": float(trailing / close * 100.0),
        })
    if not rows:
        raise ValueError("SPY 월별 분배수익률을 계산하지 못했습니다.")
    return pd.DataFrame(rows).set_index("Date")


def _download_spy_distribution_yield(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as error:
        raise RuntimeError("SPY 분배수익률 갱신에는 yfinance가 필요합니다.") from error
    history = yf.Ticker("SPY").history(
        start=(start - pd.Timedelta(days=370)).date().isoformat(),
        end=(end + pd.Timedelta(days=2)).date().isoformat(),
        auto_adjust=False,
        actions=True,
    )
    result = calculate_spy_distribution_yield(history).loc[start:end]
    if result.empty:
        raise ValueError("Yahoo Finance에서 SPY 분배수익률을 가져오지 못했습니다.")
    digest = hashlib.sha256(
        result.to_csv(date_format="%Y-%m-%d").encode("utf-8")
    ).hexdigest()
    result.attrs.update({
        "source": "Yahoo Finance SPY prices and distributions",
        "source_url": "https://finance.yahoo.com/quote/SPY/history/",
        "method_url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-500-etf-trust-spy",
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sha256": digest,
    })
    return result


def _read_spy_dividend_csv(path: Path) -> pd.DataFrame:
    columns = ["SPY_TTM_DISTRIBUTION", "SPY_CLOSE", "SPY_DISTRIBUTION_YIELD"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    if "Date" not in frame.columns or not set(columns).issubset(frame.columns):
        raise ValueError(f"SPY 분배수익률 CSV 형식이 올바르지 않습니다: {path}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Date", "SPY_DISTRIBUTION_YIELD"]).set_index("Date").sort_index()
    frame.index = frame.index.to_period("M").to_timestamp("M")
    return frame[columns].groupby(level=0).last()


def load_spy_distribution_yield(
    start_iso: str,
    end_iso: str,
    *,
    packaged_path: Path = PACKAGED_SPY_DIVIDEND_CSV,
    cache_path: Path = RUNTIME_SPY_DIVIDEND_CSV,
    mode: str = "refresh_missing",
) -> pd.Series:
    """Load a reproducible monthly SPY trailing distribution-yield series."""
    if mode not in SNAPSHOT_MODES:
        raise ValueError(f"지원하지 않는 스냅샷 모드: {mode}")
    start = pd.Timestamp(start_iso).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end_iso).to_period("M").to_timestamp("M")
    required = pd.period_range(start, end, freq="M")
    packaged = _read_spy_dividend_csv(packaged_path)
    cached = pd.DataFrame(columns=packaged.columns)
    if mode != "fixed":
        try:
            cached = _read_spy_dividend_csv(cache_path)
        except (ValueError, OSError, pd.errors.ParserError):
            pass
    combined = packaged.copy() if mode == "fixed" else pd.concat([packaged, cached]).groupby(level=0).last().sort_index()
    available = combined["SPY_DISTRIBUTION_YIELD"].dropna().index.to_period("M")
    missing = required.difference(available)
    provenance: dict[str, object] = {}
    should_fetch = mode == "latest" or (mode == "refresh_missing" and len(missing))
    if should_fetch:
        fetched = _download_spy_distribution_yield(start, end)
        provenance = dict(fetched.attrs)
        combined = pd.concat([combined, fetched]).groupby(level=0).last().sort_index()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_suffix(".tmp")
            combined.rename_axis("Date").reset_index().to_csv(temp_path, index=False, date_format="%Y-%m-%d")
            temp_path.replace(cache_path)
        except OSError:
            pass
    available = combined["SPY_DISTRIBUTION_YIELD"].dropna().index.to_period("M")
    unresolved = required.difference(available)
    result = combined.loc[start:end, "SPY_DISTRIBUTION_YIELD"].dropna().astype(float).rename("SPY_DISTRIBUTION_YIELD")
    if result.empty:
        raise ValueError("요청 기간에 사용할 수 있는 SPY 분배수익률 자료가 없습니다.")
    result.attrs.update({
        "source": "Yahoo Finance SPY prices and distributions",
        "snapshot_id": (
            f"yahoo-spy-dividend-{str(provenance.get('sha256'))[:12]}"
            if provenance.get("sha256")
            else _file_snapshot_id(packaged_path, "bundled-spy-dividend")
        ),
        "snapshot_mode": mode,
        "method": "trailing 365-day distributions divided by raw month-end close",
        "missing_months": [str(value) for value in unresolved],
        **provenance,
    })
    return result


def correct_estimated_sp500_dividend_yield(
    multpl: pd.Series,
    spy: pd.Series,
    *,
    calibration_months: int = 36,
    minimum_calibration_months: int = 12,
) -> pd.Series:
    """Replace only Multpl estimates with a level-adjusted SPY proxy.

    The rolling median spread is calculated strictly from earlier confirmed
    Multpl observations, preventing look-ahead in historical decisions.
    """
    result = multpl.astype(float).copy()
    estimate_periods = {str(value) for value in multpl.attrs.get("estimate_months", [])}
    raw_values = {str(index.to_period("M")): float(value) for index, value in multpl.items()}
    spy_values: dict[str, float] = {}
    spreads: dict[str, float] = {}
    corrected: list[str] = []
    fallback: list[str] = []
    aligned_spy = spy.reindex(result.index)
    confirmed_mask = pd.Series(
        [str(index.to_period("M")) not in estimate_periods for index in result.index],
        index=result.index,
    )
    for index in result.index:
        period = str(index.to_period("M"))
        if period not in estimate_periods:
            continue
        spy_value = aligned_spy.get(index)
        prior = pd.concat(
            [result.loc[result.index < index].rename("multpl"), aligned_spy.loc[aligned_spy.index < index].rename("spy")],
            axis=1,
        ).dropna()
        prior = prior.loc[confirmed_mask.reindex(prior.index).fillna(False)].tail(calibration_months)
        if pd.isna(spy_value) or len(prior) < minimum_calibration_months:
            fallback.append(period)
            continue
        spread = float((prior["multpl"] - prior["spy"]).median())
        result.loc[index] = float(spy_value) + spread
        spy_values[period] = float(spy_value)
        spreads[period] = spread
        corrected.append(period)
    digest = hashlib.sha256(
        (str(multpl.attrs.get("snapshot_id", "")) + str(spy.attrs.get("snapshot_id", ""))
         + "|".join(f"{key}:{result.loc[pd.Period(key).to_timestamp('M')]:.12g}" for key in corrected)).encode("utf-8")
    ).hexdigest()[:12]
    result.name = "SP500_DIV_YIELD"
    result.attrs.update(dict(multpl.attrs))
    result.attrs.update({
        "source": "Multpl confirmed + level-adjusted SPY estimate proxy",
        "snapshot_id": f"hybrid-dividend-{digest}",
        "multpl_snapshot_id": multpl.attrs.get("snapshot_id"),
        "spy_snapshot_id": spy.attrs.get("snapshot_id"),
        "raw_multpl_values": raw_values,
        "spy_distribution_yields": spy_values,
        "correction_spreads": spreads,
        "corrected_months": corrected,
        "fallback_estimate_months": fallback,
        "correction_method": f"SPY trailing yield + prior {calibration_months}-month median confirmed spread",
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
    request_get: Callable = http_get,
) -> pd.Series:
    daily, provenance = download_fred_series(
        series_id, start, end, request_get=request_get,
    )
    periods = daily.index.to_period("M")
    monthly = daily.groupby(periods).last().dropna()
    monthly.index = monthly.index.to_timestamp("M")
    monthly.attrs.update(provenance.__dict__)
    return monthly


def load_treasury_spreads(
    series_ids: set[str],
    start_iso: str,
    end_iso: str,
    *,
    packaged_path: Path = PACKAGED_SPREAD_CSV,
    cache_path: Path = RUNTIME_SPREAD_CSV,
    request_get: Callable = http_get,
    mode: str = "refresh_missing",
) -> dict[str, pd.Series]:
    """Use bundled monthly CSV data and update only missing months on demand.

    Runtime cache writes are best-effort because Streamlit Cloud storage is
    ephemeral.  The bundled CSV remains the deterministic baseline.
    """
    if mode not in SNAPSHOT_MODES:
        raise ValueError(f"지원하지 않는 스냅샷 모드: {mode}")
    requested = set(series_ids) & set(SPREAD_COLUMNS)
    if not requested:
        return {}
    start = pd.Timestamp(start_iso).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end_iso).to_period("M").to_timestamp("M")
    required_months = pd.period_range(start, end, freq="M")

    packaged = _read_spread_csv(packaged_path)
    cached = pd.DataFrame(columns=list(SPREAD_COLUMNS), dtype=float)
    if mode != "fixed":
        try:
            cached = _read_spread_csv(cache_path)
        except (ValueError, OSError, pd.errors.ParserError):
            # A Streamlit instance can stop while replacing its ephemeral cache.
            # A damaged runtime cache must never invalidate the bundled baseline.
            pass
    combined = packaged.copy() if mode == "fixed" else pd.concat([packaged, cached]).groupby(level=0).last().sort_index()

    def missing_for(series_id: str) -> pd.PeriodIndex:
        available = combined[series_id].dropna().index.to_period("M")
        return required_months.difference(available)

    missing_before = {series_id: missing_for(series_id) for series_id in requested}
    update_errors: dict[str, Exception] = {}
    fetched_provenance: dict[str, dict[str, object]] = {}
    should_update = mode == "latest" or (
        mode == "refresh_missing" and any(len(months) for months in missing_before.values())
    )
    if should_update:
        updated = combined.copy()
        for series_id in requested:
            months = missing_before[series_id]
            if mode != "latest" and not len(months):
                continue
            try:
                fetched = _download_series(
                    series_id,
                    start if mode == "latest" else months.min().to_timestamp(how="start"),
                    end,
                    request_get=request_get,
                )
            except (ValueError, OSError) as error:
                update_errors[series_id] = error
                continue
            fetched_provenance[series_id] = dict(fetched.attrs)
            updated = updated.combine_first(fetched.to_frame())
            updated.loc[fetched.index, series_id] = fetched
        combined = updated.sort_index()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_suffix(".tmp")
            combined.rename_axis("Date").reset_index().to_csv(temp_path, index=False, date_format="%Y-%m-%d")
            temp_path.replace(cache_path)
            if fetched_provenance:
                version_digest = hashlib.sha256(
                    "|".join(
                        str(fetched_provenance[key].get("sha256", ""))
                        for key in sorted(fetched_provenance)
                    ).encode("utf-8")
                ).hexdigest()[:12]
                snapshot_dir = cache_path.parent / "snapshots"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                version_path = snapshot_dir / f"treasury_spreads_{version_digest}.csv"
                if not version_path.exists():
                    combined.rename_axis("Date").reset_index().to_csv(
                        version_path, index=False, date_format="%Y-%m-%d",
                    )
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

    result = {}
    for series_id in requested:
        series = combined.loc[start:end, series_id].dropna().rename(series_id)
        series.attrs.update({
            "source": "FRED",
            "snapshot_id": (
                f"fred-{series_id}-{str(fetched_provenance[series_id].get('sha256'))[:12]}"
                if series_id in fetched_provenance
                else _file_snapshot_id(packaged_path, "bundled-treasury")
            ),
            "snapshot_mode": mode,
            "data_policy": "bundled_snapshot_only" if mode == "fixed" else "bundled_snapshot_then_missing_month_refresh",
        })
        result[series_id] = series
    return result


