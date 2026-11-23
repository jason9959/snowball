"""Shared Yahoo Finance symbol discovery for Snowball NiceGUI tools."""

from __future__ import annotations

import re

import requests


def search_symbols(query: str, country: str, limit: int = 12) -> dict[str, str]:
    text = query.strip()
    if not text:
        return {}
    options: dict[str, str] = {}
    direct_options: dict[str, str] = {}
    if country == "미국" and re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,4}", text):
        direct = text.upper()
        direct_options[direct] = f"{direct} · 티커 직접 사용"
    elif country == "한국" and re.fullmatch(r"\d{6}", text):
        direct_options[f"{text}.KS"] = f"{text}.KS · 코스피 코드 직접 사용"
        direct_options[f"{text}.KQ"] = f"{text}.KQ · 코스닥 코드 직접 사용"
    try:
        response = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": text, "quotesCount": 30, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        response.raise_for_status()
        quotes = response.json().get("quotes", [])
    except (requests.RequestException, ValueError):
        return direct_options
    for quote in quotes:
        symbol = str(quote.get("symbol", "")).strip()
        exchange = str(quote.get("exchange", "")).upper()
        quote_type = str(quote.get("quoteType", "")).upper()
        if not symbol or quote_type not in {"EQUITY", "ETF", "MUTUALFUND"}:
            continue
        is_korea = symbol.endswith((".KS", ".KQ")) or exchange in {"KSC", "KOE"}
        if (country == "한국") != is_korea:
            continue
        name = str(quote.get("longname") or quote.get("shortname") or symbol)
        options[symbol] = f"{symbol} · {name}"
        if len(options) >= limit:
            break
    for symbol, label in direct_options.items():
        if len(options) >= limit:
            break
        options.setdefault(symbol, label)
    return options
