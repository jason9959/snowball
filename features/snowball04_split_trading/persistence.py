"""JSON-friendly persistence helpers for restoring split-trading results."""

from __future__ import annotations

from io import StringIO

import pandas as pd


def _storage_scalar(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def serialize_result(result: dict) -> dict:
    series_keys = ("prices", "full_nav", "split_nav")
    return {
        "scalars": {
            key: _storage_scalar(value)
            for key, value in result.items()
            if key not in {*series_keys, "trades"}
        },
        "series": {
            key: {
                "index": [timestamp.isoformat() for timestamp in result[key].index],
                "values": [float(value) for value in result[key].values],
                "name": str(result[key].name or key),
            }
            for key in series_keys
        },
        "trades": result["trades"].to_json(orient="split", date_format="iso"),
    }


def deserialize_result(payload: dict) -> dict:
    result = dict(payload["scalars"])
    for key, item in payload["series"].items():
        result[key] = pd.Series(
            item["values"], index=pd.to_datetime(item["index"]), name=item["name"],
        )
    result["trades"] = pd.read_json(StringIO(payload["trades"]), orient="split")
    if "거래일" in result["trades"]:
        result["trades"]["거래일"] = pd.to_datetime(result["trades"]["거래일"])
    return result


def serialize_optimization(optimization: dict | None) -> dict | None:
    if optimization is None:
        return None
    return {
        "table": optimization["table"].to_json(orient="split"),
        "metric_winners": {
            metric: serialize_result(result)
            for metric, result in optimization.get("metric_winners", {}).items()
        },
    }


def deserialize_optimization(payload: dict | None) -> dict | None:
    if payload is None:
        return None
    return {
        "table": pd.read_json(StringIO(payload["table"]), orient="split"),
        "metric_winners": {
            metric: deserialize_result(result)
            for metric, result in payload.get("metric_winners", {}).items()
        },
    }
