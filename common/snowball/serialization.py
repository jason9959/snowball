"""JSON-safe presentation helpers shared by every NiceGUI feature.

NiceGUI sends component properties through JSON.  Pandas/Numpy scalar objects
look like ordinary Python values in a DataFrame but are not accepted by the
JSON encoder (notably ``pd.Timestamp``).  Keeping the conversion at the UI
boundary prevents repeat executions and table refreshes from crashing the
whole page with a HTTP 500 response.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """Recursively convert common analytics values to JSON-native values."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return DataFrame records guaranteed to be accepted by NiceGUI."""
    clean = frame.astype(object).where(pd.notna(frame), None)
    return [json_safe(record) for record in clean.to_dict("records")]
