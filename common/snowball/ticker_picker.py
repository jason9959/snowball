"""Reusable searchable ticker list editor for NiceGUI screens."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable

from nicegui import run, ui

from common.snowball.symbols import search_symbols


class TickerPicker:
    """Search, add, list, and remove up to ``max_items`` market symbols."""

    def __init__(
        self,
        *,
        defaults: Iterable[str] = (),
        max_items: int = 10,
        label: str = "종목",
        weighted: bool = False,
        default_weights: Iterable[float] = (),
        default_country: str = "미국",
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.max_items = max(1, int(max_items))
        self.label = label
        self.weighted = weighted
        self._symbols = list(dict.fromkeys(
            str(value).strip().upper() for value in defaults if str(value).strip()
        ))[: self.max_items]
        weights = list(default_weights)
        self._weights = {
            symbol: float(weights[index]) if index < len(weights) else 0.0
            for index, symbol in enumerate(self._symbols)
        }
        self._labels = {symbol: symbol for symbol in self._symbols}
        self._search_version = 0
        self._on_change = on_change

        with ui.column().classes("ticker-picker w-full gap-2"):
            with ui.row().classes("ticker-search-grid w-full items-end"):
                self.country = ui.select(
                    ["미국", "한국"], value=default_country, label="시장",
                ).props("outlined options-dense").classes("ticker-country")
                self.query = ui.input(
                    f"{label} 검색", placeholder="티커·종목명·종목코드",
                ).props("outlined clearable").classes("ticker-query")
                self.candidate = ui.select(
                    {}, label="검색 결과",
                ).props("outlined options-dense").classes("ticker-candidate")
                self.add_button = ui.button(
                    "추가", icon="add", on_click=self.add_selected,
                ).classes("primary-action ticker-add")
            self.status = ui.label(
                f"최대 {self.max_items}개까지 추가할 수 있습니다."
            ).classes("status-text")
            self.list_area = ui.column().classes("ticker-list w-full gap-2")

        self.query.on_value_change(lambda _: self._search_later())
        self.country.on_value_change(lambda _: self._reset_search())
        self._render_list()

    async def _search_later(self) -> None:
        self._search_version += 1
        version = self._search_version
        query = str(self.query.value or "").strip()
        if not query:
            self._set_candidates({})
            self.status.set_text(f"최대 {self.max_items}개까지 추가할 수 있습니다.")
            return
        await asyncio.sleep(0.4)
        if version != self._search_version:
            return
        self.status.set_text("종목을 검색하고 있습니다…")
        try:
            options = await run.io_bound(search_symbols, query, str(self.country.value))
        except Exception as error:
            self.status.set_text(f"종목 검색 실패: {error}")
            return
        if version != self._search_version:
            return
        self._set_candidates(options)
        self.status.set_text(
            f"{len(options)}개 후보를 찾았습니다."
            if options else "일치하는 종목이 없습니다."
        )

    def _set_candidates(self, options: dict[str, str]) -> None:
        self.candidate.options = options
        self.candidate.value = next(iter(options), None)
        self.candidate.update()

    def _reset_search(self) -> None:
        self._search_version += 1
        self.query.value = ""
        self._set_candidates({})

    def add_selected(self) -> None:
        symbol = str(self.candidate.value or "").strip().upper()
        if not symbol:
            self.status.set_text("검색 결과에서 추가할 종목을 선택하세요.")
            return
        if symbol in self._symbols:
            self.status.set_text(f"{symbol}은 이미 목록에 있습니다.")
            return
        if len(self._symbols) >= self.max_items:
            self.status.set_text(f"종목은 최대 {self.max_items}개까지 추가할 수 있습니다.")
            return
        self._symbols.append(symbol)
        self._labels[symbol] = str(self.candidate.options.get(symbol, symbol))
        self._weights[symbol] = 0.0
        self.query.value = ""
        self._set_candidates({})
        self.status.set_text(f"{symbol}을 추가했습니다. ({len(self._symbols)}/{self.max_items})")
        self._render_list()
        if self._on_change is not None:
            self._on_change()

    def _remove(self, symbol: str) -> None:
        if symbol in self._symbols:
            self._symbols.remove(symbol)
        self._weights.pop(symbol, None)
        self._labels.pop(symbol, None)
        self.status.set_text(f"{symbol}을 삭제했습니다. ({len(self._symbols)}/{self.max_items})")
        self._render_list()
        if self._on_change is not None:
            self._on_change()

    def _render_list(self) -> None:
        self.list_area.clear()
        with self.list_area:
            if not self._symbols:
                ui.label("추가된 종목이 없습니다.").classes("ticker-empty")
                return
            for index, symbol in enumerate(self._symbols, start=1):
                with ui.row().classes("ticker-list-row w-full items-center no-wrap"):
                    ui.label(str(index)).classes("ticker-index")
                    ui.label(self._labels.get(symbol, symbol)).classes("ticker-list-label")
                    if self.weighted:
                        weight = ui.number(
                            "비중(%)", value=self._weights.get(symbol, 0.0),
                            min=0, max=100, step=1,
                        ).props("outlined dense").classes("ticker-weight")
                        weight.on_value_change(
                            lambda event, item=symbol: self._weights.__setitem__(
                                item, float(event.value or 0)
                            )
                        )
                    ui.button(
                        icon="delete_outline",
                        on_click=lambda _, item=symbol: self._remove(item),
                    ).props("round flat color=negative").classes("ticker-delete")

    def values(self) -> list[str]:
        return list(self._symbols)

    def value(self) -> str:
        return self._symbols[0] if self._symbols else ""

    def weighted_values(self) -> list[tuple[str, float]]:
        return [(symbol, float(self._weights.get(symbol, 0.0))) for symbol in self._symbols]

    def set_values(
        self, values: Iterable[str], weights: Iterable[float] | None = None,
    ) -> None:
        symbols = list(dict.fromkeys(
            str(value).strip().upper() for value in values if str(value).strip()
        ))[: self.max_items]
        new_weights = list(weights) if weights is not None else []
        self._symbols = symbols
        self._labels = {symbol: symbol for symbol in symbols}
        self._weights = {
            symbol: (
                float(new_weights[index])
                if index < len(new_weights)
                else float(self._weights.get(symbol, 0.0))
            )
            for index, symbol in enumerate(symbols)
        }
        self.status.set_text(f"{len(symbols)}개 종목을 불러왔습니다.")
        self._render_list()
        if self._on_change is not None:
            self._on_change()

    def set_on_change(self, callback: Callable[[], None] | None) -> None:
        self._on_change = callback
