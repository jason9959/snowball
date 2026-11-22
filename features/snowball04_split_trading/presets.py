"""Durable browser storage for split-trading presets."""

from __future__ import annotations

import json

from nicegui import app, ui


USER_STORAGE_KEY = "split_trading_presets_v1"
LOCAL_STORAGE_KEY = "snowball_split_trading_presets_v1"


def load_presets(user_storage_key: str = USER_STORAGE_KEY) -> dict[str, dict]:
    raw = app.storage.user.get(user_storage_key, {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(title): dict(values)
        for title, values in raw.items()
        if str(title).strip() and isinstance(values, dict)
    }


def save_presets(
    records: dict[str, dict], user_storage_key: str = USER_STORAGE_KEY,
) -> None:
    app.storage.user[user_storage_key] = records


async def persist_presets(
    records: dict[str, dict],
    user_storage_key: str = USER_STORAGE_KEY,
    local_storage_key: str = LOCAL_STORAGE_KEY,
) -> None:
    save_presets(records, user_storage_key)
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    await ui.run_javascript(
        f"localStorage.setItem({json.dumps(local_storage_key)}, "
        f"{json.dumps(payload, ensure_ascii=False)});"
    )


async def restore_local_presets(
    records: dict[str, dict],
    user_storage_key: str = USER_STORAGE_KEY,
    local_storage_key: str = LOCAL_STORAGE_KEY,
) -> bool:
    raw = await ui.run_javascript(
        f"localStorage.getItem({json.dumps(local_storage_key)}) || ''"
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
    save_presets(records, user_storage_key)
    return True
