"""Portable user-created strategy presets.

The Streamlit session owns the active library. This module keeps its JSON
format independent from the UI so an exported library can be validated and
restored after a refresh, redeploy, or move to another device.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .presets import MomentumSpec, PortfolioPreset, SpreadRule


FORMAT_VERSION = 1


@dataclass(frozen=True)
class UserPreset:
    title: str
    strategy: PortfolioPreset
    domestic_execution: bool = False

    def validate(self) -> None:
        if not self.title.strip():
            raise ValueError("전략 제목을 입력해 주세요.")
        self.strategy.validate()


def _momentum(value: Any, field_name: str) -> MomentumSpec:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 설정이 올바르지 않습니다.")
    return MomentumSpec(
        kind=str(value.get("kind", "수익률")),
        months=int(value.get("months", 12)),
        score_type=str(value.get("score_type", "1-3-6-12개월 단순")),
    )


def _spread(value: Any, field_name: str) -> SpreadRule | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 설정이 올바르지 않습니다.")
    return SpreadRule(
        lag_months=int(value["lag_months"]),
        active_months=int(value["active_months"]),
        threshold=float(value["threshold"]),
    )


def preset_to_dict(preset: PortfolioPreset) -> dict[str, Any]:
    preset.validate()
    return asdict(preset)


def preset_from_dict(value: Any) -> PortfolioPreset:
    if not isinstance(value, dict):
        raise ValueError("전략 설정이 올바르지 않습니다.")
    data = dict(value)
    for key in ("attack_tickers", "defense_tickers", "canary_tickers", "market_signals"):
        raw = data.get(key, ())
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"{key} 목록이 올바르지 않습니다.")
        data[key] = tuple(str(item) for item in raw)
    for key in (
        "attack_relative", "attack_absolute", "defense_relative",
        "defense_absolute", "canary_momentum",
    ):
        data[key] = _momentum(data.get(key), key)
    data["spread_10y2y"] = _spread(data.get("spread_10y2y"), "spread_10y2y")
    data["spread_10y3m"] = _spread(data.get("spread_10y3m"), "spread_10y3m")
    try:
        preset = PortfolioPreset(**data)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"전략 필드를 읽을 수 없습니다: {error}") from error
    preset.validate()
    return preset


def export_user_presets(presets: Iterable[UserPreset]) -> bytes:
    records = list(presets)
    for record in records:
        record.validate()
    payload = {
        "format": "snowball-user-presets",
        "version": FORMAT_VERSION,
        "presets": [
            {
                "title": record.title,
                "domestic_execution": record.domestic_execution,
                "strategy": preset_to_dict(record.strategy),
            }
            for record in records
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def import_user_presets(content: bytes | str) -> list[UserPreset]:
    try:
        text = content.decode("utf-8-sig") if isinstance(content, bytes) else content
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("JSON 프리셋 파일을 읽을 수 없습니다.") from error
    if not isinstance(payload, dict) or payload.get("format") != "snowball-user-presets":
        raise ValueError("Snowball 사용자 프리셋 파일이 아닙니다.")
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError("지원하지 않는 사용자 프리셋 파일 버전입니다.")
    raw_presets = payload.get("presets")
    if not isinstance(raw_presets, list):
        raise ValueError("사용자 프리셋 목록이 없습니다.")
    result: list[UserPreset] = []
    seen: set[str] = set()
    for raw in raw_presets:
        if not isinstance(raw, dict):
            raise ValueError("사용자 프리셋 항목이 올바르지 않습니다.")
        title = str(raw.get("title", "")).strip()
        if not title:
            raise ValueError("제목이 없는 사용자 프리셋이 있습니다.")
        if title in seen:
            raise ValueError(f"중복된 사용자 프리셋 제목이 있습니다: {title}")
        seen.add(title)
        strategy = preset_from_dict(raw.get("strategy"))
        result.append(UserPreset(title, strategy, bool(raw.get("domestic_execution", False))))
    return result
