"""Snowball 01 individual portfolio backtest page."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

from nicegui import run, ui

from common.snowball.backtest_results import render_backtest_result
from common.snowball.layout import shell
from common.snowball.preset_store import (
    load_browser_presets,
    persist_browser_presets,
    restore_local_presets,
)
from snowball.builtin_library import (
    BUILTIN_PRESET_SPECS,
    builtin_user_preset,
    get_builtin_spec,
    individual_builtin_specs,
)
from snowball.core.catalog import US_ETFS
from snowball.presets import MomentumSpec, SpreadRule
from snowball.user_presets import UserPreset, export_user_presets, import_user_presets
from .service import prepare_individual_prices, run_individual_backtest

USER_PRESET_PREFIX = "사용자::"
MOMENTUM_KINDS = ["수익률", "이동평균", "모멘텀 스코어"]
SCORE_TYPES = [
    "1-3-6개월 단순", "3-6-12개월 단순", "1-3-6-12개월 단순",
    "1-3-6-9-12개월 단순", "1-3-6-12개월 가중",
]
ATTACK_MODES = [
    "상대 모멘텀", "절대 모멘텀", "절대 모멘텀 + 방어 스위치",
    "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 방어 스위치",
]
DEFENSE_MODES = [
    "전체 동일 비중", "상대 모멘텀", "절대 모멘텀",
    "상대 & 절대 모멘텀 + 현금 스위치",
]
WEIGHT_MODES = ["고정 범위", "마이너스 절대 모멘텀 수"]
MARKET_SIGNALS = [
    "장단기 금리차 역전 (10년-2년)",
    "장단기 금리차 역전 (10년-3개월)",
    "S&P500 배당수익률",
]


class _StrategyEditor:
    def __init__(self, record: UserPreset):
        self.base = record.strategy
        self.title = ui.input("프리셋 제목", value=record.title).props("outlined").classes("w-full")
        self.domestic = ui.checkbox("국내 ETF로 실행", value=record.domestic_execution)
        assets = sorted(US_ETFS)
        with ui.card().classes("editor-section w-full"):
            ui.label("투자자산").classes("font-bold")
            self.attack_tickers = ui.select(
                assets, value=list(self.base.attack_tickers), label="투자자산 후보군",
                multiple=True,
            ).props("outlined use-chips options-dense").classes("w-full")
            with ui.row().classes("editor-fields w-full"):
                self.attack_mode = ui.select(
                    ATTACK_MODES, value=self.base.attack_mode, label="선택 방식",
                ).props("outlined").classes("control-grow")
                self.attack_count = ui.number(
                    "선택자산 수", value=self.base.attack_select_count, min=1, step=1,
                ).props("outlined").classes("control-grow")
            self.attack_kind, self.attack_months, self.attack_score = self._momentum(
                "투자 모멘텀", self.base.attack_relative,
            )
        with ui.card().classes("editor-section w-full"):
            ui.label("방어자산").classes("font-bold")
            self.defense_tickers = ui.select(
                assets, value=list(self.base.defense_tickers), label="방어자산 후보군",
                multiple=True,
            ).props("outlined use-chips options-dense").classes("w-full")
            with ui.row().classes("editor-fields w-full"):
                self.defense_mode = ui.select(
                    DEFENSE_MODES, value=self.base.defense_mode, label="선택 방식",
                ).props("outlined").classes("control-grow")
                self.defense_count = ui.number(
                    "선택자산 수", value=self.base.defense_select_count, min=1, step=1,
                ).props("outlined").classes("control-grow")
            self.defense_same = ui.checkbox(
                "투자자산과 동일 모멘텀 사용", value=self.base.defense_same_momentum,
            )
            self.defense_kind, self.defense_months, self.defense_score = self._momentum(
                "방어 모멘텀", self.base.defense_relative,
            )
        with ui.card().classes("editor-section w-full"):
            ui.label("비중 및 위험회피").classes("font-bold")
            with ui.row().classes("editor-fields w-full"):
                self.weight_mode = ui.select(
                    WEIGHT_MODES, value=self.base.weight_mode, label="비중 방식",
                ).props("outlined").classes("control-grow")
                self.fixed_defense = ui.number(
                    "고정 방어 비중(%)", value=self.base.fixed_defense_weight * 100,
                    min=0, max=100, step=5,
                ).props("outlined").classes("control-grow")
                self.negative_threshold = ui.number(
                    "전체 방어 전환 기준", value=self.base.negative_threshold,
                    min=1, step=1,
                ).props("outlined").classes("control-grow")
            self.canary_tickers = ui.select(
                assets, value=list(self.base.canary_tickers), label="카나리아 ETF",
                multiple=True,
            ).props("outlined use-chips options-dense").classes("w-full")
            self.canary_kind, self.canary_months, self.canary_score = self._momentum(
                "카나리아 모멘텀", self.base.canary_momentum,
            )
            with ui.row().classes("editor-fields w-full"):
                self.canary_apply = ui.select(
                    ["방어형", "공격형"], value=self.base.canary_apply_mode,
                    label="카나리아 적용 방식",
                ).props("outlined").classes("control-grow")
                self.canary_direction = ui.select(
                    ["마이너스(-)", "플러스(+)"], value=self.base.canary_direction,
                    label="위험 신호 방향",
                ).props("outlined").classes("control-grow")
                rounding = (
                    "원본값" if self.base.canary_decision_percent_digits is None
                    else str(self.base.canary_decision_percent_digits)
                )
                self.canary_rounding = ui.select(
                    ["원본값", "0", "1", "2", "3", "4", "5", "6"],
                    value=rounding, label="판정 반올림 자릿수",
                ).props("outlined").classes("control-grow")
        with ui.card().classes("editor-section w-full"):
            ui.label("시장 위험 신호").classes("font-bold")
            self.market_signals = ui.select(
                MARKET_SIGNALS, value=list(self.base.market_signals), label="활성 신호",
                multiple=True,
            ).props("outlined use-chips options-dense").classes("w-full")
            self.market_combine = ui.radio(
                ["OR", "AND"], value=self.base.market_combine,
            ).props("inline")
            with ui.row().classes("editor-fields w-full"):
                rule_2y = self.base.spread_10y2y or SpreadRule(7, 8, -0.5)
                self.spread_2y_lag = ui.number("10Y-2Y 지연(개월)", value=rule_2y.lag_months, min=0, max=24).props("outlined").classes("control-grow")
                self.spread_2y_active = ui.number("10Y-2Y 반영 기간", value=rule_2y.active_months, min=1, max=24).props("outlined").classes("control-grow")
                self.spread_2y_threshold = ui.number("10Y-2Y 역전값(%p)", value=rule_2y.threshold, step=.1).props("outlined").classes("control-grow")
            with ui.row().classes("editor-fields w-full"):
                rule = self.base.spread_10y3m or SpreadRule(7, 8, -0.5)
                self.spread_lag = ui.number("10Y-3M 지연(개월)", value=rule.lag_months, min=0, max=24).props("outlined").classes("control-grow")
                self.spread_active = ui.number("10Y-3M 반영 기간", value=rule.active_months, min=1, max=24).props("outlined").classes("control-grow")
                self.spread_threshold = ui.number("10Y-3M 역전값(%p)", value=rule.threshold, step=.1).props("outlined").classes("control-grow")
                self.dividend = ui.number("배당수익률 기준(%)", value=self.base.dividend_threshold, min=0, max=10, step=.01).props("outlined").classes("control-grow")

    @staticmethod
    def _momentum(label: str, value: MomentumSpec):
        with ui.row().classes("editor-fields w-full"):
            kind = ui.select(MOMENTUM_KINDS, value=value.kind, label=f"{label} 기준").props("outlined").classes("control-grow")
            months = ui.number("기간(개월)", value=value.months, min=1, max=24, step=1).props("outlined").classes("control-grow")
            score = ui.select(SCORE_TYPES, value=value.score_type, label="스코어 구성").props("outlined").classes("control-grow")
        return kind, months, score

    @staticmethod
    def _spec(kind, months, score) -> MomentumSpec:
        return MomentumSpec(str(kind.value), int(months.value or 1), str(score.value))

    def record(self) -> UserPreset:
        title = str(self.title.value or "").strip()
        attacks = tuple(self.attack_tickers.value or ())
        defenses = tuple(self.defense_tickers.value or ())
        canaries = tuple(self.canary_tickers.value or ())
        signals = tuple(self.market_signals.value or ())
        if not title:
            raise ValueError("프리셋 제목을 입력해 주세요.")
        if title in BUILTIN_PRESET_SPECS:
            raise ValueError("기본 제공 프리셋과 같은 제목은 사용할 수 없습니다.")
        if not attacks or not defenses:
            raise ValueError("투자자산과 방어자산 후보를 하나 이상 선택하세요.")
        attack_momentum = self._spec(self.attack_kind, self.attack_months, self.attack_score)
        defense_momentum = self._spec(self.defense_kind, self.defense_months, self.defense_score)
        canary_momentum = self._spec(self.canary_kind, self.canary_months, self.canary_score)
        rounding_value = str(self.canary_rounding.value)
        strategy = replace(
            self.base, name=title, attack_tickers=attacks, defense_tickers=defenses,
            attack_mode=str(self.attack_mode.value),
            attack_select_count=min(int(self.attack_count.value or 1), len(attacks)),
            attack_relative=attack_momentum, attack_absolute=attack_momentum,
            defense_mode=str(self.defense_mode.value),
            defense_select_count=min(int(self.defense_count.value or 1), len(defenses)),
            defense_relative=defense_momentum, defense_absolute=defense_momentum,
            defense_same_momentum=bool(self.defense_same.value),
            weight_mode=str(self.weight_mode.value),
            fixed_defense_weight=float(self.fixed_defense.value or 0) / 100,
            negative_threshold=min(int(self.negative_threshold.value or 1), len(attacks)),
            canary_tickers=canaries, canary_momentum=canary_momentum,
            canary_apply_mode=str(self.canary_apply.value),
            canary_direction=str(self.canary_direction.value),
            canary_decision_percent_digits=(
                None if rounding_value == "원본값" else int(rounding_value)
            ),
            market_combine=str(self.market_combine.value), market_signals=signals,
            spread_10y2y=(
                SpreadRule(
                    int(self.spread_2y_lag.value or 0), int(self.spread_2y_active.value or 1),
                    float(self.spread_2y_threshold.value or 0),
                ) if "장단기 금리차 역전 (10년-2년)" in signals else self.base.spread_10y2y
            ),
            spread_10y3m=(
                SpreadRule(
                    int(self.spread_lag.value or 0), int(self.spread_active.value or 1),
                    float(self.spread_threshold.value or 0),
                ) if "장단기 금리차 역전 (10년-3개월)" in signals else self.base.spread_10y3m
            ),
            dividend_threshold=float(self.dividend.value or 0),
        )
        record = UserPreset(title, strategy, bool(self.domestic.value))
        record.validate()
        return record


@ui.page("/backtest")
def backtest_page() -> None:
    shell()
    user_presets = load_browser_presets()

    def preset_options() -> dict[str, str]:
        options = {title: title for title in individual_builtin_specs()}
        options.update({
            f"{USER_PRESET_PREFIX}{title}": f"사용자 · {title}"
            for title in sorted(user_presets)
        })
        return options

    with ui.column().classes("page-container w-full gap-4"):
        ui.label("백테스팅").classes("page-title")
        ui.label(
            "기본 프리셋을 복사해 조건을 편집하고, 나만의 프리셋으로 저장할 수 있습니다."
        ).classes("page-subtitle")
        with ui.card().classes("backtest-controls w-full"):
            preset = ui.select(
                preset_options(), value=next(iter(BUILTIN_PRESET_SPECS)),
                label="내 프리셋",
            ).props("outlined options-dense").classes("w-full")
            with ui.row().classes("backtest-field-grid backtest-date-grid w-full"):
                start_input = ui.input("시작일", value="2010-01-01").props(
                    "type=date"
                ).classes("date-input")
                end_input = ui.input(
                    "종료일", value=datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
                ).props("type=date").classes("date-input")
            with ui.row().classes("backtest-field-grid w-full"):
                initial = ui.number(
                    "초기 투자금", value=10_000, min=100, step=1_000,
                ).props("outlined").classes("control-grow")
                cost = ui.number(
                    "편도 거래비용(bp)", value=0, min=0, max=500, step=1,
                ).props("outlined").classes("control-grow")
            market_mode = ui.radio(
                {"fixed": "고정 스냅샷", "latest": "최신 데이터"},
                value="fixed",
            ).props("inline")
            run_button = ui.button("백테스트 실행", icon="play_arrow").classes(
                "primary-action w-full"
            )

        with ui.expansion(
            "전략 조건 편집", icon="tune", value=False,
        ).classes("strategy-expansion w-full"):
            editor_area = ui.column().classes("w-full gap-3")

        with ui.card().classes("preset-library w-full"):
            ui.label("사용자 프리셋 백업").classes("font-bold")
            ui.label(
                "이 브라우저에 자동 저장됩니다. 다른 기기나 재배포에 대비해 JSON도 보관하세요."
            ).classes("text-sm text-grey-6")
            with ui.row().classes("w-full items-start gap-3"):
                export_button = ui.button(
                    "JSON 다운로드", icon="download", color="secondary",
                ).props("outline")
                upload = ui.upload(
                    label="JSON 불러오기", auto_upload=True, max_files=1,
                    max_file_size=2_000_000,
                ).props("accept=.json flat bordered").classes("preset-upload")

        status = ui.label("조건을 확인한 뒤 실행하세요.").classes("status-text")
        result_area = ui.column().classes("w-full gap-4")
        editor_state: dict[str, _StrategyEditor | None] = {"editor": None}

        def refresh_preset_select(selected: str | None = None) -> None:
            preset.options = preset_options()
            if selected is not None:
                preset.value = selected
            preset.update()

        def selected_user_title() -> str | None:
            value = str(preset.value or "")
            return value.removeprefix(USER_PRESET_PREFIX) if value.startswith(USER_PRESET_PREFIX) else None

        def rebuild_editor() -> None:
            editor_area.clear()
            editor_state["editor"] = None
            value = str(preset.value or "")
            with editor_area:
                custom_title = selected_user_title()
                if custom_title is not None:
                    record = user_presets.get(custom_title)
                    if record is None:
                        ui.label("저장된 사용자 프리셋을 찾을 수 없습니다.").classes("text-negative")
                        return
                else:
                    spec = get_builtin_spec(value)
                    if spec.secondary is not None:
                        ui.label(
                            "결합 프리셋은 위의 혼합 비중만 편집합니다. "
                            "구성 전략은 개별 프리셋을 복사해 저장하세요."
                        ).classes("text-sm text-grey-6")
                        return
                    copied_title = f"{spec.title} 복사본"
                    record = UserPreset(
                        copied_title,
                        replace(spec.primary, name=copied_title),
                        spec.domestic_execution,
                    )
                editor_state["editor"] = _StrategyEditor(record)
                with ui.row().classes("w-full gap-3"):
                    ui.button(
                        "내 프리셋 저장", icon="save", on_click=save_current,
                    ).classes("primary-action")
                    if custom_title is not None:
                        ui.button(
                            "삭제", icon="delete", color="negative",
                            on_click=delete_current,
                        ).props("outline")

        async def save_current() -> None:
            editor = editor_state["editor"]
            if editor is None:
                ui.notify("저장할 개별 전략이 없습니다.", type="warning")
                return
            try:
                record = editor.record()
                old_title = selected_user_title()
                if old_title and old_title != record.title:
                    user_presets.pop(old_title, None)
                user_presets[record.title] = record
                await persist_browser_presets(user_presets)
            except ValueError as error:
                ui.notify(str(error), type="negative")
                return
            key = f"{USER_PRESET_PREFIX}{record.title}"
            refresh_preset_select(key)
            rebuild_editor()
            ui.notify(f"'{record.title}' 프리셋을 저장했습니다.", type="positive")

        async def delete_current() -> None:
            title = selected_user_title()
            if title is None:
                return
            user_presets.pop(title, None)
            await persist_browser_presets(user_presets)
            refresh_preset_select(next(iter(BUILTIN_PRESET_SPECS)))
            update_preset_controls()
            rebuild_editor()
            ui.notify(f"'{title}' 프리셋을 삭제했습니다.", type="positive")

        def export_library() -> None:
            if not user_presets:
                ui.notify("다운로드할 사용자 프리셋이 없습니다.", type="warning")
                return
            ui.download.content(
                export_user_presets(user_presets.values()),
                filename="snowball_user_presets.json",
                media_type="application/json",
            )

        async def import_library(event) -> None:
            try:
                imported = import_user_presets(event.content.read())
                collisions = [item.title for item in imported if item.title in BUILTIN_PRESET_SPECS]
                if collisions:
                    raise ValueError("기본 프리셋과 같은 제목은 불러올 수 없습니다.")
                user_presets.update({item.title: item for item in imported})
                await persist_browser_presets(user_presets)
            except ValueError as error:
                ui.notify(str(error), type="negative")
                return
            finally:
                upload.reset()
            selected = f"{USER_PRESET_PREFIX}{imported[0].title}" if imported else None
            refresh_preset_select(selected)
            update_preset_controls()
            rebuild_editor()
            ui.notify(f"사용자 프리셋 {len(imported)}개를 불러왔습니다.", type="positive")

        async def restore_presets() -> None:
            """Restore after free-host restarts which erase server-side storage."""
            try:
                restored = await restore_local_presets(user_presets)
                if not restored and user_presets:
                    await persist_browser_presets(user_presets)
                elif not restored:
                    return
            except (ValueError, RuntimeError):
                ui.notify(
                    "브라우저 프리셋 자동 복원에 실패했습니다. JSON 백업을 이용해 주세요.",
                    type="warning",
                )
                return
            refresh_preset_select(str(preset.value or next(iter(BUILTIN_PRESET_SPECS))))
            rebuild_editor()

        def update_preset_controls() -> None:
            custom_title = selected_user_title()
            if custom_title is None:
                spec = get_builtin_spec(str(preset.value))
                is_domestic = spec.domestic_execution
            else:
                record = user_presets.get(custom_title)
                is_domestic = bool(record and record.domestic_execution)
                weight.set_visibility(False)
            initial.value = 10_000_000 if is_domestic else 10_000
            initial.min = 100_000 if is_domestic else 100
            initial.update()

        def preset_changed() -> None:
            update_preset_controls()
            rebuild_editor()

        preset.on_value_change(lambda _: preset_changed())
        export_button.on("click", export_library)
        upload.on_upload(import_library)
        update_preset_controls()
        rebuild_editor()
        ui.timer(0.2, restore_presets, once=True)

        async def execute_backtest() -> None:
            run_button.disable()
            result_area.clear()
            status.set_text("가격과 시장 데이터를 불러와 계산하고 있습니다…")
            try:
                title = str(preset.value)
                common_args = {
                    "initial_investment": float(initial.value or 0),
                    "transaction_cost_rate": float(cost.value or 0) / 10_000.0,
                    "market_data_mode": str(market_mode.value),
                }
                start_date = date.fromisoformat(str(start_input.value))
                end_date = date.fromisoformat(str(end_input.value))
                editor = editor_state["editor"]
                record = editor.record() if editor is not None else builtin_user_preset(title)
                status.set_text("1/3 · Yahoo 가격 데이터를 확인하고 있습니다…")
                await run.io_bound(prepare_individual_prices, record, start_date, end_date)
                status.set_text("2/3 · 시장 지표와 월별 목표를 계산하고 있습니다…")
                result, bundle, cache_hit = await run.io_bound(
                    run_individual_backtest, record, start_date, end_date, **common_args,
                )
            except Exception as error:
                status.set_text("백테스트를 완료하지 못했습니다.")
                with result_area:
                    with ui.card().classes("error-card w-full"):
                        ui.label(str(error)).classes("text-negative")
            else:
                status.set_text("3/3 · 결과 차트와 표를 구성하고 있습니다…")
                await asyncio.sleep(0)
                with result_area:
                    render_backtest_result(result, bundle)
                status.set_text(
                    "백테스트가 완료되었습니다."
                    + (" · 저장된 동일 결과를 불러왔습니다." if cache_hit else "")
                )
            finally:
                run_button.enable()

        run_button.on("click", execute_backtest)

