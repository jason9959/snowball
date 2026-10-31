"""Command-line entry point for deterministic checks before a UI exists."""

from __future__ import annotations

import argparse
import datetime as dt

from .mailing import (
    generate_recommendations,
    is_first_xnys_session,
    load_mailing_portfolios,
    render_email,
    send_email_from_env,
)
from .presets import PRESETS


def main() -> None:
    parser = argparse.ArgumentParser(prog="snowball")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-presets", help="검증된 연구 프리셋 목록")
    mail = sub.add_parser("mail-rebalance", help="월 첫 NYSE 거래일 리밸런싱 메일 발송")
    mail.add_argument("--config", default="config/mailing.json")
    mail.add_argument("--date", help="실행 기준일(YYYY-MM-DD), 기본값은 미국 동부 현지일")
    mail.add_argument("--force", action="store_true", help="첫 거래일이 아니어도 계산")
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
        first_session = is_first_xnys_session(run_date)
        if not first_session and not args.force:
            print(f"SKIP: {run_date} is not the first XNYS session of the month.")
            return
        portfolios = load_mailing_portfolios(args.config)
        recommendations = generate_recommendations(portfolios, run_date)
        subject, text_body, html_body = render_email(recommendations)
        if args.dry_run:
            print(subject)
            print(text_body)
        else:
            send_email_from_env(subject, text_body, html_body)
            print(f"SENT: {subject}")


if __name__ == "__main__":
    main()
