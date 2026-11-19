"""Durable browser storage for split-trading presets."""

from __future__ import annotations

import json

from nicegui import app, ui


USER_STORAGE_KEY = "split_trading_presets_v1"
LOCAL_STORAGE_KEY = "snowball_split_trading_presets_v1"


def load_presets() -> dict[str, dict]:
    raw = app.storage.user.get(USER_STORAGE_KEY, {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(title): dict(values)
        for title, values in raw.items()
        if str(title).strip() and isinstance(values, dict)
    }


def save_presets(records: dict[str, dict]) -> None:
    app.storage.user[USER_STORAGE_KEY] = records


async def persist_presets(records: dict[str, dict]) -> None:
    save_presets(records)
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    await ui.run_javascript(
        f"localStorage.setItem({json.dumps(LOCAL_STORAGE_KEY)}, "
        f"{json.dumps(payload, ensure_ascii=False)});"
    )


async def restore_local_presets(records: dict[str, dict]) -> bool:
    raw = await ui.run_javascript(
        f"localStorage.getItem({json.dumps(LOCAL_STORAGE_KEY)}) || ''"
    )
    if not raw:
        return False
    restored = json.loads(str(raw))
    if not isinstance(restored, dict):
        return False
    records.update({
        str(title): dict(values)
        for title, values in restored.items()
        if str(title).strip() and isinstance(values, dict)
    })
    save_presets(records)
    return True
