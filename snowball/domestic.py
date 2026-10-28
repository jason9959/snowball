"""Domestic-listed execution instruments for the validated US signals."""

from __future__ import annotations


DOMESTIC_EXECUTION_MAP: dict[str, str] = {
    "QQQ": "449190.KS",   # KODEX 미국나스닥100(H)
    "SOXX": "469050.KS",  # RISE 미국반도체NYSE(H)
    "XLE": "218420.KS",   # KODEX 미국S&P500에너지(합성)
    "XLK": "463680.KS",   # KODEX 미국S&P500테크놀로지
    "XLV": "453640.KS",   # KODEX 미국S&P500헬스케어
    "EDV": "304660.KS",   # KODEX 미국채울트라30년선물(H)
    "GLD": "411060.KS",   # ACE KRX금현물
    "DBC": "130680.KS",   # TIGER 원유선물Enhanced(H)
    "BIL": "329750.KS",   # TIGER 미국달러단기채권액티브
    "IEF": "305080.KS",   # TIGER 미국채10년선물
    "HYG": "468380.KS",   # KODEX iShares 미국하이일드액티브
    "SCHD": "452360.KS",  # SOL 미국배당다우존스(H)
    "VWO": "195980.KS",   # PLUS 신흥국MSCI(합성 H)
    "PDBC": "130680.KS",  # TIGER 원유선물Enhanced(H)
}

DOMESTIC_NAMES: dict[str, str] = {
    "449190.KS": "KODEX 미국나스닥100(H)",
    "469050.KS": "RISE 미국반도체NYSE(H)",
    "218420.KS": "KODEX 미국S&P500에너지(합성)",
    "463680.KS": "KODEX 미국S&P500테크놀로지",
    "453640.KS": "KODEX 미국S&P500헬스케어",
    "304660.KS": "KODEX 미국채울트라30년선물(H)",
    "411060.KS": "ACE KRX금현물",
    "130680.KS": "TIGER 원유선물Enhanced(H)",
    "329750.KS": "TIGER 미국달러단기채권액티브",
    "305080.KS": "TIGER 미국채10년선물",
    "468380.KS": "KODEX iShares 미국하이일드액티브",
    "452360.KS": "SOL 미국배당다우존스(H)",
    "195980.KS": "PLUS 신흥국MSCI(합성 H)",
}


def domestic_symbols(us_symbols: tuple[str, ...]) -> tuple[str, ...]:
    missing = sorted(set(us_symbols) - set(DOMESTIC_EXECUTION_MAP))
    if missing:
        raise ValueError("국내 실행 ETF가 정해지지 않은 종목: " + ", ".join(missing))
    return tuple(dict.fromkeys(DOMESTIC_EXECUTION_MAP[symbol] for symbol in us_symbols))


def domestic_label(symbol: str) -> str:
    code = symbol.removesuffix(".KS")
    return f"{code} · {DOMESTIC_NAMES.get(symbol, symbol)}"
