"""Snowball browser preset persistence.

This remains separate from any functions-app storage helper because NiceGUI's
per-user storage and browser localStorage lifecycle differ from Streamlit's.
"""

from __future__ import annotations

import json

from nicegui import app, ui

from snowball.user_presets import UserPreset, export_user_presets, import_user_presets


LOCAL_PRESET_STORAGE_KEY = "snowball_user_presets_v1"


def load_browser_presets() -> dict[str, UserPreset]:
    raw = str(app.storage.user.get("user_presets_json", "")).strip()
    if not raw:
        return {}
    try:
        records = import_user_presets(raw)
    except ValueError:
        return {}
    return {record.title: record for record in records}


def save_browser_presets(records: dict[str, UserPreset]) -> None:
    app.storage.user["user_presets_json"] = export_user_presets(
        records.values()
    ).decode("utf-8")


async def persist_browser_presets(records: dict[str, UserPreset]) -> None:
    payload = export_user_presets(records.values()).decode("utf-8")
    save_browser_presets(records)
    await ui.run_javascript(
        f"localStorage.setItem({json.dumps(LOCAL_PRESET_STORAGE_KEY)}, "
        f"{json.dumps(payload, ensure_ascii=False)});"
    )


async def restore_local_presets(records: dict[str, UserPreset]) -> bool:
    """Restore the durable browser copy after a free-host restart."""
    raw = await ui.run_javascript(
        f"localStorage.getItem({json.dumps(LOCAL_PRESET_STORAGE_KEY)}) || ''"
    )
    if not raw:
        return False
    restored = import_user_presets(str(raw))
    records.update({item.title: item for item in restored})
    save_browser_presets(records)
    return True
