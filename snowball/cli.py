"""Command-line entry point for deterministic checks before a UI exists."""

from __future__ import annotations

import argparse

from .presets import PRESETS


def main() -> None:
    parser = argparse.ArgumentParser(prog="snowball")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-presets", help="검증된 연구 프리셋 목록")
    args = parser.parse_args()
    if args.command == "list-presets":
        for preset in PRESETS.values():
            print(f"{preset.name}: 공격 {', '.join(preset.attack_tickers)} / 방어 {', '.join(preset.defense_tickers)}")


if __name__ == "__main__":
    main()
