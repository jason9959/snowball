"""Runtime settings shared by the Streamlit and NiceGUI frontends."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Mapping


def _read_streamlit_secrets(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        import tomllib

        with path.open("rb") as stream:
            value = tomllib.load(stream)
        return value if isinstance(value, Mapping) else {}
    except (OSError, ValueError):
        return {}


def runtime_secret(name: str, *, root: Path | None = None) -> str:
    """Read a secret from the environment, then local Streamlit secrets."""
    environment_value = os.getenv(name, "").strip()
    if environment_value:
        return environment_value
    project_root = root or Path(__file__).resolve().parents[1]
    value = _read_streamlit_secrets(project_root / ".streamlit" / "secrets.toml").get(name, "")
    if isinstance(value, Mapping):
        return json.dumps(dict(value))
    return str(value).strip() if value is not None else ""


def google_sheet_settings(*, root: Path | None = None) -> tuple[str, str]:
    return (
        runtime_secret("GOOGLE_SHEET_ID", root=root),
        runtime_secret("GOOGLE_SERVICE_ACCOUNT_JSON", root=root),
    )
