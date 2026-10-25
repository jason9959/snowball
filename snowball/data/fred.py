"""Official FRED CSV client with explicit provenance.

FRED's graph CSV endpoint needs no API key.  The returned observation date is
kept separate from strategy lag rules.  Historical vintages are not implied:
the endpoint returns the currently published history.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import StringIO
from typing import Callable

import pandas as pd

from .http import get as http_get


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@dataclass(frozen=True)
class FredProvenance:
    series_id: str
    url: str
    sha256: str
    vintage_mode: str = "current_history_not_alfred"


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    frame = pd.read_csv(StringIO(text), na_values=["."])
    if frame.shape[1] < 2:
        raise ValueError(f"FRED {series_id} CSV 열이 부족합니다.")
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name=series_id).dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if series.empty:
        raise ValueError(f"FRED {series_id} 데이터가 비어 있습니다.")
    return series


def download_fred_series(
    series_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    request_get: Callable = http_get,
) -> tuple[pd.Series, FredProvenance]:
    response = request_get(
        FRED_CSV_URL,
        params={"id": series_id, "cosd": start.date().isoformat(), "coed": end.date().isoformat()},
        headers={"User-Agent": "snowball-portfolio-lab/0.1"},
        timeout=(8, 30),
    )
    response.raise_for_status()
    text = response.text
    series = parse_fred_csv(text, series_id).loc[start:end]
    if series.empty:
        raise ValueError(f"FRED {series_id} 요청 기간 데이터가 없습니다.")
    provenance = FredProvenance(
        series_id=series_id,
        url=getattr(response, "url", FRED_CSV_URL),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    series.attrs.update(provenance.__dict__)
    return series, provenance
