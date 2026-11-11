from __future__ import annotations

import datetime as dt
import os
from typing import Any

import pandas as pd
import requests


API_URL = "https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do"


def resolve_api_key(secrets: Any | None = None) -> str:
    """Read the R-ONE key without ever exposing it in the UI or source."""
    if secrets is not None:
        try:
            value = secrets.get("RONE_API_KEY", "")
            if value:
                return str(value).strip()
        except Exception:
            pass
    return os.getenv("RONE_API_KEY", "").strip()


def _period_to_date(value: object, cycle: str) -> pd.Timestamp:
    raw = "".join(character for character in str(value) if character.isdigit())
    if cycle == "MM" and len(raw) >= 6:
        return pd.Timestamp(int(raw[:4]), int(raw[4:6]), 1)
    if cycle in {"QY", "Q"} and len(raw) >= 5:
        quarter = max(1, min(4, int(raw[-1])))
        return pd.Timestamp(int(raw[:4]), (quarter - 1) * 3 + 1, 1)
    if cycle in {"WK", "WW"} and len(raw) >= 6:
        return pd.Timestamp(dt.date.fromisocalendar(int(raw[:4]), int(raw[4:6]), 1))
    if len(raw) >= 4:
        return pd.Timestamp(int(raw[:4]), 1, 1)
    return pd.NaT


def parse_response(payload: dict[str, Any], cycle: str) -> tuple[pd.DataFrame, int]:
    blocks = payload.get("SttsApiTblData")
    if not isinstance(blocks, list):
        message = payload.get("RESULT", {}).get("MESSAGE", "응답 형식이 올바르지 않습니다.")
        raise ValueError(message)

    rows: list[dict[str, Any]] = []
    total_count = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for head in block.get("head", []):
            if isinstance(head, dict) and "list_total_count" in head:
                total_count = int(head["list_total_count"])
            result = head.get("RESULT") if isinstance(head, dict) else None
            if isinstance(result, dict) and result.get("CODE") not in {None, "INFO-000"}:
                raise ValueError(result.get("MESSAGE", "R-ONE 조회에 실패했습니다."))
        rows.extend(block.get("row", []))

    if not rows:
        raise ValueError("선택한 조건에 해당하는 R-ONE 자료가 없습니다.")

    frame = pd.DataFrame(rows)
    frame["값"] = pd.to_numeric(frame.get("DTA_VAL"), errors="coerce")
    frame["날짜"] = frame.get("WRTTIME_IDTFR_ID", pd.Series(index=frame.index)).map(
        lambda item: _period_to_date(item, cycle)
    )
    frame = frame.dropna(subset=["날짜", "값"]).sort_values("날짜").reset_index(drop=True)
    if frame.empty:
        raise ValueError("R-ONE 응답에서 날짜와 숫자 값을 해석하지 못했습니다.")
    return frame, total_count


def fetch_statistics(
    *,
    stat_table_id: str,
    cycle: str,
    region_id: str,
    item_id: str,
    start_year: int,
    end_year: int,
    api_key: str = "",
    timeout: float = 30.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params: dict[str, Any] = {
        "STATBL_ID": stat_table_id,
        "DTACYCLE_CD": cycle,
        "CLS_ID": region_id,
        "ITM_ID": item_id,
        "START_WRTTIME": str(start_year),
        "END_WRTTIME": str(end_year),
        "Type": "json",
        "pIndex": 1,
        "pSize": 1000,
    }
    if api_key:
        params["KEY"] = api_key

    response = requests.get(
        API_URL,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PortfolioDashboard/1.0)"},
    )
    response.raise_for_status()
    frame, total_count = parse_response(response.json(), cycle)

    # A keyless request is R-ONE's documented sample mode. It currently returns
    # only the first five rows even when the table contains more observations.
    metadata = {
        "sample_mode": not bool(api_key),
        "returned_count": len(frame),
        "total_count": total_count,
        "request_url": response.url.split("KEY=")[0],
    }
    return frame, metadata

