"""Small, dependency-free ETF catalog used by the dynamic portfolio UI.

The catalog is intentionally a suggestion list, not an allow-list.  Users can
still enter a valid Yahoo Finance ticker that is not present here.
"""

from __future__ import annotations


US_ETFS: dict[str, str] = {
    "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "DBC": "Invesco DB Commodity Index Tracking Fund",
    "GLD": "SPDR Gold Shares",
    "IAU": "iShares Gold Trust",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "LQD": "iShares iBoxx Investment Grade Corporate Bond ETF",
    "QQQ": "Invesco QQQ Trust",
    "SCHD": "Schwab U.S. Dividend Equity ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "SPY": "SPDR S&P 500 ETF Trust",
    "TIP": "iShares TIPS Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "VOO": "Vanguard S&P 500 ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
}

KR_ETFS: dict[str, str] = {
    "069500.KS": "KODEX 200",
    "091160.KS": "KODEX 반도체",
    "114800.KS": "KODEX 인버스",
    "133690.KS": "TIGER 미국나스닥100",
    "143850.KS": "TIGER 미국S&P500선물(H)",
    "148070.KS": "KOSEF 국고채10년",
    "152100.KS": "ARIRANG 200",
    "195930.KS": "TIGER 유로스탁스50(합성 H)",
    "261240.KS": "KODEX 미국달러선물",
    "305080.KS": "TIGER 미국채10년선물",
    "360750.KS": "TIGER 미국S&P500",
    "367380.KS": "ACE 미국나스닥100",
    "379800.KS": "KODEX 미국S&P500TR",
    "411060.KS": "ACE KRX금현물",
    "449180.KS": "KODEX 미국S&P500(H)",
}


def catalog_for_market(market: str) -> dict[str, str]:
    return KR_ETFS if market == "국내 ETF" else US_ETFS


def normalize_ticker(value: str, market: str) -> str:
    """Normalize manual input into the Yahoo Finance ticker convention."""
    ticker = value.strip().upper()
    if market == "국내 ETF" and ticker.isdigit() and len(ticker) == 6:
        ticker = f"{ticker}.KS"
    return ticker


def display_name(ticker: str, market: str) -> str:
    name = catalog_for_market(market).get(ticker, ticker)
    return f"{ticker} · {name}" if name != ticker else ticker


def suggestions(query: str, market: str, limit: int = 30) -> list[tuple[str, str]]:
    needle = query.strip().lower()
    if not needle:
        return []
    matches = []
    for ticker, name in catalog_for_market(market).items():
        if needle in ticker.lower() or needle in name.lower():
            matches.append((ticker, name))
    return matches[:limit]


