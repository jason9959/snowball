"""Unattended first-trading-day portfolio recommendations and email delivery."""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .data.market_data import load_strategy_market_data
from .data.prices import PriceBundle, download_yahoo_prices
from .domestic import DOMESTIC_EXECUTION_MAP, domestic_label
from .presets import PRESETS, PortfolioPreset
from .service import run_blended_presets, run_preset


@dataclass(frozen=True)
class MailingPortfolio:
    title: str
    preset: PortfolioPreset
    secondary_preset: PortfolioPreset | None = None
    first_weight: float = 1.0
    tracking_start: dt.date | None = None

    def validate(self) -> None:
        if not self.title.strip():
            raise ValueError("메일링 포트폴리오 제목이 비어 있습니다.")
        self.preset.validate()
        if self.secondary_preset is not None:
            self.secondary_preset.validate()
        if not 0.0 <= self.first_weight <= 1.0:
            raise ValueError("첫 번째 프리셋 비중은 0~1 범위여야 합니다.")


@dataclass(frozen=True)
class PerformanceSnapshot:
    tracking_start: dt.date
    as_of: dt.date
    observations: int
    cumulative_return: float | None = None
    monthly_return: float | None = None
    current_drawdown: float | None = None
    max_drawdown: float | None = None
    max_drawdown_date: dt.date | None = None
    drawdown_peak_date: dt.date | None = None
    drawdown_days: int = 0


@dataclass(frozen=True)
class RebalanceRecommendation:
    title: str
    trade_date: dt.date
    domestic_trade_date: dt.date
    decision_date: dt.date
    selection: str
    domestic_selection: str
    details: tuple[tuple[str, str], ...] = ()
    performance: PerformanceSnapshot | None = None


def load_mailing_portfolios(path: str | Path) -> list[MailingPortfolio]:
    """Load built-in single or blended presets from a small JSON config."""
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"메일링 설정을 읽을 수 없습니다: {error}") from error
    records = payload.get("portfolios") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("메일링 설정에 portfolios 목록이 필요합니다.")

    result: list[MailingPortfolio] = []
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("메일링 포트폴리오 설정이 올바르지 않습니다.")
        primary_name = str(raw.get("preset", ""))
        secondary_name = raw.get("secondary_preset")
        try:
            primary = PRESETS[primary_name]
            secondary = PRESETS[str(secondary_name)] if secondary_name else None
        except KeyError as error:
            raise ValueError(f"알 수 없는 프리셋입니다: {error.args[0]}") from error
        record = MailingPortfolio(
            title=str(raw.get("title", primary_name)).strip(),
            preset=primary,
            secondary_preset=secondary,
            first_weight=float(raw.get("first_weight", 1.0)),
            tracking_start=(
                dt.date.fromisoformat(str(raw["tracking_start"]))
                if raw.get("tracking_start") else None
            ),
        )
        record.validate()
        result.append(record)
    return result


def first_xnys_session(day: dt.date) -> dt.date:
    """Return the first NYSE trading session in ``day``'s calendar month."""
    try:
        import exchange_calendars as xcals
    except ImportError as error:
        raise RuntimeError("exchange-calendars 패키지가 필요합니다.") from error
    calendar = xcals.get_calendar("XNYS")
    start = day.replace(day=1)
    next_month = (pd.Timestamp(start) + pd.offsets.MonthBegin(1)).date()
    end = next_month - dt.timedelta(days=1)
    sessions = calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    if len(sessions) == 0:
        raise ValueError(f"{day:%Y-%m}에 NYSE 거래일이 없습니다.")
    return pd.Timestamp(sessions[0]).date()


def is_first_xnys_session(day: dt.date) -> bool:
    return day == first_xnys_session(day)


def next_xnys_session(day: dt.date) -> dt.date:
    """Return the first NYSE session strictly after ``day``."""
    return _next_exchange_session("XNYS", day)


def next_xkrx_session(day: dt.date) -> dt.date:
    """Return the first Korea Exchange session strictly after ``day``."""
    return _next_exchange_session("XKRX", day)


def _next_exchange_session(calendar_name: str, day: dt.date) -> dt.date:
    try:
        import exchange_calendars as xcals
    except ImportError as error:
        raise RuntimeError("exchange-calendars 패키지가 필요합니다.") from error
    calendar = xcals.get_calendar(calendar_name)
    session = calendar.date_to_session(pd.Timestamp(day), direction="next")
    if pd.Timestamp(session).date() == day:
        session = calendar.next_session(session)
    return pd.Timestamp(session).date()


def is_month_end_xnys_session(day: dt.date) -> bool:
    """True after the final completed NYSE session before a new month."""
    try:
        import exchange_calendars as xcals
    except ImportError as error:
        raise RuntimeError("exchange-calendars 패키지가 필요합니다.") from error
    calendar = xcals.get_calendar("XNYS")
    stamp = pd.Timestamp(day)
    if not calendar.is_session(stamp):
        return False
    return next_xnys_session(day).month != day.month


def _domestic_equivalent(selection: str) -> str:
    """Map US target weights to Korean-listed execution suggestions."""
    mapped: dict[str, float] = {}
    for item in selection.split(", "):
        try:
            ticker, percent = item.rsplit(" ", 1)
            weight = float(percent.rstrip("%")) / 100.0
        except (ValueError, AttributeError):
            continue
        domestic = ticker if ticker == "현금" else DOMESTIC_EXECUTION_MAP.get(ticker)
        label = domestic_label(domestic) if domestic else f"{ticker} 대체 ETF 미정"
        mapped[label] = mapped.get(label, 0.0) + weight
    return ", ".join(f"{label} {weight:.1%}" for label, weight in mapped.items())


def _with_synthetic_trade_session(bundle: PriceBundle, trade_date: dt.date) -> tuple[PriceBundle, dt.date]:
    """Append a signal-only row; no unobserved opening price is predicted."""
    if bundle.adjusted_close.empty:
        raise ValueError("가격 자료가 비어 있습니다.")
    trade_ts = pd.Timestamp(trade_date)
    real = bundle.adjusted_close.loc[bundle.adjusted_close.index < trade_ts]
    if real.empty:
        raise ValueError("리밸런싱 판단 전의 확정 종가가 없습니다.")
    decision_ts = pd.Timestamp(real.index[-1])
    close = bundle.adjusted_close.loc[:decision_ts].copy()
    adjusted_open = bundle.adjusted_open.reindex(close.index).copy()
    # The engine evaluates momentum on the previous row. The synthetic values
    # are used only to create today's schedule row and never as a signal.
    close.loc[trade_ts] = close.loc[decision_ts]
    adjusted_open.loc[trade_ts] = close.loc[decision_ts]
    return PriceBundle(close, adjusted_open, source=f"{bundle.source} + signal-only session"), decision_ts.date()


def _performance_snapshot(
    result: dict, tracking_start: dt.date, decision_date: dt.date,
) -> PerformanceSnapshot:
    nav = result["nav"].dropna().astype(float)
    nav = nav[
        (nav.index >= pd.Timestamp(tracking_start))
        & (nav.index <= pd.Timestamp(decision_date))
    ]
    if nav.empty:
        return PerformanceSnapshot(tracking_start, decision_date, 0)

    cumulative_return = float(nav.iloc[-1] - 1.0)
    month_start = pd.Timestamp(decision_date).replace(day=1)
    before_month = nav[nav.index < month_start]
    month_nav = nav[nav.index >= month_start]
    if month_nav.empty:
        monthly_return = None
    elif before_month.empty:
        monthly_return = float(month_nav.iloc[-1] - 1.0)
    else:
        monthly_return = float(month_nav.iloc[-1] / before_month.iloc[-1] - 1.0)

    running_peak = nav.cummax().clip(lower=1.0)
    drawdown = nav / running_peak - 1.0
    current_drawdown = float(drawdown.iloc[-1])
    max_drawdown = float(drawdown.min())
    max_drawdown_date = pd.Timestamp(drawdown.idxmin()).date()
    peak_value = max(1.0, float(nav.max()))
    if peak_value <= 1.0:
        peak_date = tracking_start
    else:
        peak_date = pd.Timestamp(nav[nav == peak_value].index[-1]).date()
    drawdown_days = max(0, (decision_date - peak_date).days) if current_drawdown < -1e-10 else 0
    return PerformanceSnapshot(
        tracking_start=tracking_start,
        as_of=decision_date,
        observations=len(nav),
        cumulative_return=cumulative_return,
        monthly_return=monthly_return,
        current_drawdown=current_drawdown,
        max_drawdown=max_drawdown,
        max_drawdown_date=max_drawdown_date,
        drawdown_peak_date=peak_date,
        drawdown_days=drawdown_days,
    )


def generate_recommendations(
    portfolios: Iterable[MailingPortfolio],
    trade_date: dt.date,
    *,
    price_loader: Callable[[tuple[str, ...], pd.Timestamp, pd.Timestamp], PriceBundle] = download_yahoo_prices,
    market_loader: Callable[..., dict[str, pd.Series]] = load_strategy_market_data,
) -> list[RebalanceRecommendation]:
    """Calculate target weights from the last completed close before open."""
    records = list(portfolios)
    if not records:
        raise ValueError("계산할 포트폴리오가 없습니다.")
    for record in records:
        record.validate()

    evaluation_start = (pd.Timestamp(trade_date) - pd.DateOffset(months=24)).replace(day=1)
    tracking_starts = [pd.Timestamp(record.tracking_start) for record in records if record.tracking_start]
    calculation_start = min([evaluation_start, *tracking_starts])
    download_start = calculation_start - pd.DateOffset(months=18)
    symbols: list[str] = ["QQQ"]
    signals: list[str] = []
    for record in records:
        symbols.extend(record.preset.symbols)
        signals.extend(record.preset.market_signals)
        if record.secondary_preset is not None:
            symbols.extend(record.secondary_preset.symbols)
            signals.extend(record.secondary_preset.market_signals)
    symbols_tuple = tuple(dict.fromkeys(symbols))
    raw_bundle = price_loader(
        symbols_tuple, download_start, pd.Timestamp(trade_date) - pd.Timedelta(days=1),
    )
    bundle, decision_date = _with_synthetic_trade_session(raw_bundle, trade_date)
    market = market_loader(
        tuple(dict.fromkeys(signals)), download_start.date().isoformat(),
        decision_date.isoformat(), mode="latest",
    )

    output: list[RebalanceRecommendation] = []
    for record in records:
        run_start = (
            record.tracking_start
            if record.tracking_start and record.tracking_start < trade_date
            else evaluation_start.date()
        )
        if record.secondary_preset is None:
            result = run_preset(
                record.preset, bundle, market, run_start, trade_date,
            )
        else:
            result = run_blended_presets(
                record.preset, record.secondary_preset, record.first_weight,
                bundle, market, run_start, trade_date,
                preset_name=record.title,
                first_label=record.preset.name,
                second_label=record.secondary_preset.name,
            )
        history = result["history"]
        selected = history[pd.to_datetime(history["적용일"]).dt.date == trade_date]
        if selected.empty:
            raise ValueError(f"{record.title}: {trade_date}의 리밸런싱 결과가 없습니다.")
        row = selected.iloc[-1]
        details: list[tuple[str, str]] = []
        if record.secondary_preset is not None:
            for preset in (record.preset, record.secondary_preset):
                key = f"{preset.name} 선택"
                if key in row:
                    details.append((preset.name, str(row[key])))
        output.append(RebalanceRecommendation(
            title=record.title,
            trade_date=trade_date,
            domestic_trade_date=next_xkrx_session(decision_date),
            decision_date=pd.Timestamp(row["판단 기준일"]).date(),
            selection=str(row["선택된 종목"]),
            domestic_selection=_domestic_equivalent(str(row["선택된 종목"])),
            details=tuple(details),
            performance=(
                _performance_snapshot(result, record.tracking_start, decision_date)
                if record.tracking_start else None
            ),
        ))
    return output


def render_email(recommendations: Iterable[RebalanceRecommendation]) -> tuple[str, str, str]:
    records = list(recommendations)
    if not records:
        raise ValueError("메일에 넣을 리밸런싱 결과가 없습니다.")
    trade_date = records[0].trade_date
    subject = f"[Snowball] {trade_date:%Y-%m} 월간 리밸런싱 알림"
    text_lines = [
        f"{trade_date:%Y-%m-%d} 목표 리밸런싱",
        "직전 거래일의 확정 종가로 판단한 종목과 비중입니다.", "",
    ]
    cards: list[str] = []
    for record in records:
        text_lines.extend([
            record.title,
            f"- 판단 기준일: {record.decision_date:%Y-%m-%d}",
            f"- 미국 ETF 매수 예정일: {record.trade_date:%Y-%m-%d}",
            f"- 국내 ETF 매수 제안일: {record.domestic_trade_date:%Y-%m-%d}",
            f"- 미국 ETF 목표 비중: {record.selection}",
            f"- 국내 ETF 대체 제안(검증 중): {record.domestic_selection}",
        ])
        text_lines.extend(f"- {label}: {selection}" for label, selection in record.details)
        performance_html = ""
        if record.performance is not None:
            performance = record.performance
            if performance.observations == 0:
                text_lines.append(
                    f"- 성과 추적: {performance.tracking_start:%Y-%m-%d} 시가 매수부터 시작"
                )
                performance_html = (
                    '<div style="margin-top:16px;padding:14px;border-radius:12px;'
                    'background:#f3f6ff;color:#30469b">'
                    f"성과 추적은 <b>{performance.tracking_start:%Y-%m-%d} 시가 매수</b>부터 시작됩니다."
                    "</div>"
                )
            else:
                drawdown_status = (
                    "낙폭 없음 · 신고가"
                    if (performance.current_drawdown or 0.0) >= -1e-10
                    else (
                        f"진행 중 {performance.current_drawdown:.2%} · "
                        f"고점 이후 {performance.drawdown_days}일"
                    )
                )
                text_lines.extend([
                    f"- 모델 운용 시작: {performance.tracking_start:%Y-%m-%d}",
                    f"- 누적 수익률: {performance.cumulative_return:.2%}",
                    f"- 이번 달 수익률: {performance.monthly_return:.2%}",
                    f"- 현재 낙폭: {drawdown_status}",
                    f"- 운용 중 MDD: {performance.max_drawdown:.2%} "
                    f"({performance.max_drawdown_date:%Y-%m-%d})",
                ])
                positive = (performance.cumulative_return or 0.0) >= 0
                cumulative_color = "#137a49" if positive else "#c23b45"
                performance_html = (
                    '<div style="margin-top:18px">'
                    '<div style="font-size:13px;color:#697386;margin-bottom:8px">'
                    f"모델 포트폴리오 성과 · {performance.as_of:%Y-%m-%d} 종가 기준</div>"
                    '<table role="presentation" width="100%" cellspacing="8" '
                    'style="border-collapse:separate;margin:-8px;width:calc(100% + 16px)"><tr>'
                    '<td width="50%" style="padding:13px;background:#f6f8fc;border-radius:12px">'
                    '<div style="font-size:12px;color:#697386">누적 수익률</div>'
                    f'<div style="font-size:21px;font-weight:700;color:{cumulative_color}">'
                    f"{performance.cumulative_return:.2%}</div></td>"
                    '<td width="50%" style="padding:13px;background:#f6f8fc;border-radius:12px">'
                    '<div style="font-size:12px;color:#697386">이번 달 수익률</div>'
                    f'<div style="font-size:21px;font-weight:700">{performance.monthly_return:.2%}</div>'
                    "</td></tr><tr>"
                    '<td style="padding:13px;background:#f6f8fc;border-radius:12px">'
                    '<div style="font-size:12px;color:#697386">현재 낙폭</div>'
                    f'<div style="font-size:15px;font-weight:700">{html.escape(drawdown_status)}</div></td>'
                    '<td style="padding:13px;background:#f6f8fc;border-radius:12px">'
                    '<div style="font-size:12px;color:#697386">운용 중 MDD</div>'
                    f'<div style="font-size:21px;font-weight:700;color:#c23b45">'
                    f"{performance.max_drawdown:.2%}</div>"
                    f'<div style="font-size:11px;color:#697386">{performance.max_drawdown_date:%Y-%m-%d}</div>'
                    "</td></tr></table></div>"
                )
        text_lines.append("")
        detail_html = "".join(
            f"<li><b>{html.escape(label)}</b>: {html.escape(selection)}</li>"
            for label, selection in record.details
        )
        cards.append(
            '<div style="background:#ffffff;border:1px solid #e6e9f0;border-radius:18px;'
            'padding:20px;margin:14px 0;box-shadow:0 4px 18px rgba(30,47,86,.06)">'
            f'<div style="font-size:20px;font-weight:800;margin-bottom:14px">{html.escape(record.title)}</div>'
            '<div style="font-size:13px;color:#697386;line-height:1.7">'
            f"판단 기준일 <b style=\"color:#252a34\">{record.decision_date:%Y-%m-%d}</b> · "
            f"국내 매수일 <b style=\"color:#252a34\">{record.domestic_trade_date:%Y-%m-%d}</b></div>"
            '<div style="margin-top:16px;padding:15px;background:#f6f8fc;border-radius:12px">'
            '<div style="font-size:12px;color:#697386;margin-bottom:5px">미국 ETF 목표 비중</div>'
            f'<div style="font-size:16px;font-weight:700;line-height:1.6">{html.escape(record.selection)}</div>'
            "</div>"
            '<div style="margin-top:10px;padding:15px;background:#fff7e8;border-radius:12px">'
            '<div style="font-size:12px;color:#8a6216;margin-bottom:5px">국내 ETF 실행 제안 · 검증 중</div>'
            f'<div style="font-size:15px;font-weight:700;line-height:1.6">{html.escape(record.domestic_selection)}</div>'
            "</div>"
            + (f'<ul style="padding-left:20px;line-height:1.7;color:#4b5565">{detail_html}</ul>' if detail_html else "")
            + performance_html + "</div>"
        )
    text_lines.append("본 메일은 자동 계산 알림이며 매수·매도 권유가 아닙니다.")
    body_html = (
        '<div style="margin:0;padding:24px 12px;background:#f3f5f9;'
        'font-family:Arial,Apple SD Gothic Neo,sans-serif;color:#252a34">'
        '<div style="max-width:680px;margin:0 auto">'
        '<div style="padding:24px;border-radius:20px;background:linear-gradient(135deg,#3157d5,#6b4ce6);'
        'color:#fff;margin-bottom:14px">'
        '<div style="font-size:13px;opacity:.82">SNOWBALL PORTFOLIO</div>'
        '<div style="font-size:27px;font-weight:800;margin-top:6px">월간 리밸런싱 리포트</div>'
        f'<div style="font-size:15px;margin-top:12px"><b>{trade_date:%Y-%m-%d}</b> 시가 목표</div>'
        '<div style="font-size:12px;opacity:.82;margin-top:4px">직전 미국 거래일의 확정 종가 기준</div>'
        "</div>"
        + "".join(cards)
        + '<div style="font-size:11px;line-height:1.6;color:#7a8495;padding:10px 6px">'
        "성과는 목표 비중대로 시가에 체결된 모델 포트폴리오 기준이며 실제 계좌의 세금, "
        "수수료, 입출금 및 체결가와 다를 수 있습니다. 자동 계산 알림이며 매수·매도 권유가 아닙니다."
        "</div></div></div>"
    )
    return subject, "\n".join(text_lines), body_html


def send_email_from_env(subject: str, text_body: str, html_body: str) -> None:
    """Send with Gmail-compatible SMTP credentials stored in GitHub Secrets."""
    username = os.environ.get("SMTP_USERNAME", "").strip()
    password = os.environ.get("SMTP_APP_PASSWORD", "").replace(" ", "").strip()
    recipients = [item.strip() for item in os.environ.get("MAIL_TO", "").split(",") if item.strip()]
    if not username or not password or not recipients:
        raise ValueError("SMTP_USERNAME, SMTP_APP_PASSWORD, MAIL_TO 환경변수가 필요합니다.")
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = username
    message["To"] = ", ".join(recipients)
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, timeout=30) as smtp:
        smtp.login(username, password)
        smtp.send_message(message)

