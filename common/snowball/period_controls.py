"""Shared NiceGUI controls for selecting a year range or exact dates."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from dateutil.relativedelta import relativedelta
from nicegui import ui


YEAR_OPTIONS = {**{str(year): f"{year}년" for year in range(1, 11)}, "custom": "직접 설정"}


def period_start(end: dt.date, years: int, minimum: dt.date | None = None) -> dt.date:
    """Return the calendar date *years* before end, optionally clamped."""
    start = end - relativedelta(years=int(years))
    return max(start, minimum) if minimum is not None else start


@dataclass(frozen=True)
class PeriodControls:
    period: ui.select
    start: ui.input
    end: ui.input


def create_period_controls(
    *,
    default_start: dt.date,
    default_end: dt.date,
    default_period: str = "custom",
    start_label: str = "시작일",
    end_label: str = "종료일",
    minimum: dt.date | None = None,
    maximum: dt.date | None = None,
) -> PeriodControls:
    """Create synchronized ``1년~10년 + 직접 설정`` date controls.

    Changing the end date while a year period is selected keeps that period and
    recalculates the start date. Editing the start date switches to exact-date
    mode so an intentional custom range is never overwritten.
    """
    selected = str(default_period)
    if selected not in YEAR_OPTIONS:
        selected = "custom"
    syncing = {"active": False}

    period = ui.select(YEAR_OPTIONS, value=selected, label="기간").props(
        "outlined options-dense"
    ).classes("control-grow")
    date_props = "type=date"
    if minimum is not None:
        date_props += f" min={minimum.isoformat()}"
    if maximum is not None:
        date_props += f" max={maximum.isoformat()}"
    start = ui.input(start_label, value=default_start.isoformat()).props(
        date_props
    ).classes("date-input")
    end = ui.input(end_label, value=default_end.isoformat()).props(
        date_props
    ).classes("date-input")

    def sync_from_period() -> None:
        if str(period.value) == "custom":
            return
        try:
            end_date = dt.date.fromisoformat(str(end.value))
            start_date = period_start(end_date, int(str(period.value)), minimum)
        except (TypeError, ValueError):
            return
        syncing["active"] = True
        start.value = start_date.isoformat()
        ui.timer(0.05, lambda: syncing.__setitem__("active", False), once=True)

    def start_changed() -> None:
        if not syncing["active"] and str(period.value) != "custom":
            period.value = "custom"

    period.on_value_change(lambda _: sync_from_period())
    start.on_value_change(lambda _: start_changed())
    end.on_value_change(lambda _: sync_from_period())
    return PeriodControls(period=period, start=start, end=end)
