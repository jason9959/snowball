"""Command-line entry point for deterministic checks before a UI exists."""

from __future__ import annotations

import argparse
import datetime as dt

from .mailing import (
    generate_recommendations,
    is_month_end_xnys_session,
    load_mailing_portfolios,
    next_xnys_session,
    render_email,
    send_email_from_env,
)
from .presets import PRESETS


def main() -> None:
    parser = argparse.ArgumentParser(prog="snowball")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-presets", help="검증된 연구 프리셋 목록")
    mail = sub.add_parser("mail-rebalance", help="월 말 NYSE 종가 후 다음 달 리밸런싱 메일 발송")
    mail.add_argument("--config", default="config/mailing.json")
    mail.add_argument("--date", help="종가 확정일(YYYY-MM-DD), 기본값은 미국 동부 현지일")
    mail.add_argument("--force", action="store_true", help="월 마지막 거래일이 아니어도 다음 거래일 목표를 계산")
    mail.add_argument("--dry-run", action="store_true", help="메일을 보내지 않고 본문만 출력")
    args = parser.parse_args()
    if args.command == "list-presets":
        for preset in PRESETS.values():
            print(f"{preset.name}: 공격 {', '.join(preset.attack_tickers)} / 방어 {', '.join(preset.defense_tickers)}")
    elif args.command == "mail-rebalance":
        if args.date:
            run_date = dt.date.fromisoformat(args.date)
        else:
            from zoneinfo import ZoneInfo
            run_date = dt.datetime.now(ZoneInfo("America/New_York")).date()
        month_end_session = is_month_end_xnys_session(run_date)
        if not month_end_session and not args.force:
            print(f"SKIP: {run_date} is not the final XNYS session before a new month.")
            return
        trade_date = next_xnys_session(run_date)
        portfolios = load_mailing_portfolios(args.config)
        recommendations = generate_recommendations(portfolios, trade_date)
        subject, text_body, html_body = render_email(recommendations)
        if args.dry_run:
            print(subject)
            print(text_body)
        else:
            send_email_from_env(subject, text_body, html_body)
            print(f"SENT: {subject}")


if __name__ == "__main__":
    main()
