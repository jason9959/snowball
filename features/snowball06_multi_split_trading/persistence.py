"""Persistence adapters for multi-asset split-trading results."""

from __future__ import annotations

from io import StringIO

import pandas as pd

from features.snowball04_split_trading.persistence import deserialize_result, serialize_result


def serialize_multi_result(result: dict) -> dict:
    base = dict(result)
    summary = base.pop("asset_summary")
    payload = serialize_result(base)
    payload["asset_summary"] = summary.to_json(orient="split")
    return payload


def deserialize_multi_result(payload: dict) -> dict:
    result = deserialize_result(payload)
    result["asset_summary"] = pd.read_json(
        StringIO(payload["asset_summary"]), orient="split",
    )
    return result
