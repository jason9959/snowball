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

    def validate(self) -> None:
        if not self.title.strip():
            raise ValueError("메일링 포트폴리오 제목이 비어 있습니다.")
        self.preset.validate()
        if self.secondary_preset is not None:
            self.secondary_preset.validate()
        if not 0.0 <= self.first_weight <= 1.0:
            raise ValueError("첫 번째 프리셋 비중은 0~1 범위여야 합니다.")


@dataclass(frozen=True)
class RebalanceRecommendation:
    title: str
    trade_date: dt.date
    domestic_trade_date: dt.date
    decision_date: dt.date
    selection: str
    domestic_selection: str
    details: tuple[tuple[str, str], ...] = ()


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
    download_start = evaluation_start - pd.DateOffset(months=18)
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
        if record.secondary_preset is None:
            result = run_preset(
                record.preset, bundle, market, evaluation_start.date(), trade_date,
            )
        else:
            result = run_blended_presets(
                record.preset, record.secondary_preset, record.first_weight,
                bundle, market, evaluation_start.date(), trade_date,
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
        text_lines.append("")
        detail_html = "".join(
            f"<li><b>{html.escape(label)}</b>: {html.escape(selection)}</li>"
            for label, selection in record.details
        )
        cards.append(
            '<div style="border:1px solid #e2e5ec;border-radius:12px;padding:16px;margin:12px 0">'
            f"<h3 style=\"margin:0 0 10px\">{html.escape(record.title)}</h3>"
            f"<p style=\"margin:4px 0\"><b>판단 기준일</b>: {record.decision_date:%Y-%m-%d}</p>"
            f"<p style=\"margin:4px 0\"><b>미국 ETF 매수 예정일</b>: {record.trade_date:%Y-%m-%d}</p>"
            f"<p style=\"margin:4px 0\"><b>국내 ETF 매수 제안일</b>: {record.domestic_trade_date:%Y-%m-%d}</p>"
            f"<p style=\"margin:4px 0\"><b>미국 ETF 목표 비중</b>: {html.escape(record.selection)}</p>"
            f"<p style=\"margin:4px 0\"><b>국내 ETF 대체 제안(검증 중)</b>: "
            f"{html.escape(record.domestic_selection)}</p>"
            + (f"<ul>{detail_html}</ul>" if detail_html else "") + "</div>"
        )
    text_lines.append("본 메일은 자동 계산 알림이며 매수·매도 권유가 아닙니다.")
    body_html = (
        '<div style="font-family:Arial,sans-serif;max-width:680px;color:#222">'
        f"<h2>Snowball 월간 리밸런싱</h2>"
        f"<p><b>{trade_date:%Y-%m-%d}</b> 미국장 시가 리밸런싱 목표입니다.<br>"
        "직전 거래일의 확정 종가로 판단했으며 당일 체결가는 예측하지 않습니다.</p>"
        + "".join(cards)
        + '<p style="font-size:12px;color:#666">자동 계산 알림이며 매수·매도 권유가 아닙니다.</p></div>'
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

