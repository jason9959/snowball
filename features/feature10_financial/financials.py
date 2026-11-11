"""Normalize DART and SEC filings into the dashboard's common metrics."""
from datetime import date

import pandas as pd

from .data_sources import (
    Company,
    DataSourceError,
    fetch_dart_financial_statement,
    fetch_sec_company_facts,
)


BALANCE = ["자산", "부채", "자본"]
INCOME = ["매출액", "영업이익", "당기순이익"]
CASH = [
    "영업현금흐름", "투자현금흐름", "재무현금흐름",
    "현금 및 현금성자산 증가(감소)", "기초 현금 및 현금성자산",
    "기말 현금 및 현금성자산",
]
ALL_METRICS = BALANCE + INCOME + CASH
DART_REPORT_CODES = {1: "11013", 2: "11012", 3: "11014", 4: "11011"}

DART_TAGS = {
    "자산": ("BS", ["ifrs-full_Assets"]),
    "부채": ("BS", ["ifrs-full_Liabilities"]),
    "자본": ("BS", ["ifrs-full_Equity"]),
    "매출액": ("IS", ["ifrs-full_Revenue", "ifrs-full_SalesRevenue"]),
    "영업이익": ("IS", ["dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities"]),
    "당기순이익": ("IS", ["ifrs-full_ProfitLoss"]),
    "영업현금흐름": ("CF", ["ifrs-full_CashFlowsFromUsedInOperatingActivities"]),
    "투자현금흐름": ("CF", ["ifrs-full_CashFlowsFromUsedInInvestingActivities"]),
    "재무현금흐름": ("CF", ["ifrs-full_CashFlowsFromUsedInFinancingActivities"]),
    "현금 및 현금성자산 증가(감소)": ("CF", ["ifrs-full_IncreaseDecreaseInCashAndCashEquivalents"]),
    "기초 현금 및 현금성자산": ("CF", ["dart_CashAndCashEquivalentsAtBeginningOfPeriodCf"]),
    "기말 현금 및 현금성자산": ("CF", ["dart_CashAndCashEquivalentsAtEndOfPeriodCf"]),
}

SEC_TAGS = {
    "자산": ["Assets"],
    "부채": ["Liabilities"],
    "자본": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
           "StockholdersEquity"],
    "매출액": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "영업이익": ["OperatingIncomeLoss"],
    "당기순이익": ["NetIncomeLoss", "ProfitLoss"],
    "영업현금흐름": ["NetCashProvidedByUsedInOperatingActivities"],
    "투자현금흐름": ["NetCashProvidedByUsedInInvestingActivities"],
    "재무현금흐름": ["NetCashProvidedByUsedInFinancingActivities"],
    "현금 및 현금성자산 증가(감소)": [
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
        "CashAndCashEquivalentsPeriodIncreaseDecrease",
    ],
    "기말 현금 및 현금성자산": [
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalentsAtCarryingValue",
    ],
}


def _number(value):
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _dart_value(rows, metric, field="thstrm_amount"):
    statement, tags = DART_TAGS[metric]
    for tag in tags:
        for row in rows:
            if row.get("sj_div") == statement and row.get("account_id") == tag:
                value = _number(row.get(field))
                if value is not None:
                    return value
    return None


def _subtract(current, previous):
    return None if current is None or previous is None else current - previous


def load_dart_financials(company: Company, start_year: int, end_year: int,
                         quarterly: bool, api_key: str):
    cache = {}

    def report(year, quarter):
        key = (year, quarter)
        if key not in cache:
            try:
                cache[key] = fetch_dart_financial_statement(
                    api_key, company.source_id, year, DART_REPORT_CODES[quarter], "CFS"
                )
            except DataSourceError:
                cache[key] = fetch_dart_financial_statement(
                    api_key, company.source_id, year, DART_REPORT_CODES[quarter], "OFS"
                )
        return cache[key]

    output = []
    periods = [(year, quarter) for year in range(start_year, end_year + 1)
               for quarter in (range(1, 5) if quarterly else [4])]
    for year, quarter in periods:
        try:
            current = report(year, quarter)
        except DataSourceError:
            continue
        row = {"기간": f"{year} Q{quarter}" if quarterly else str(year)}
        for metric in BALANCE:
            row[metric] = _dart_value(current, metric)
        for metric in INCOME:
            if quarterly and quarter == 4:
                row[metric] = _subtract(
                    _dart_value(current, metric),
                    _dart_value(report(year, 3), metric, "thstrm_add_amount"),
                )
            else:
                row[metric] = _dart_value(current, metric)
        for metric in CASH[:4]:
            cumulative = _dart_value(current, metric)
            if quarterly and quarter > 1:
                previous = _dart_value(report(year, quarter - 1), metric)
                row[metric] = _subtract(cumulative, previous)
            else:
                row[metric] = cumulative
        row[CASH[4]] = _dart_value(current, CASH[4])
        row[CASH[5]] = _dart_value(current, CASH[5])
        output.append(row)
    if not output:
        raise DataSourceError("선택한 기간에 DART 재무정보가 없습니다.")
    frame = pd.DataFrame(output).set_index("기간")
    return frame[ALL_METRICS].div(100_000_000), "억 원"


def _sec_units(facts, tags):
    for tag in tags:
        fact = facts.get(tag)
        if fact and fact.get("units", {}).get("USD"):
            return fact["units"]["USD"]
    return []


def _duration_days(item):
    if "start" not in item or "end" not in item:
        return None
    return (date.fromisoformat(item["end"]) - date.fromisoformat(item["start"])).days


def _latest(items):
    return max(items, key=lambda item: (item.get("end", ""), item.get("filed", ""), item.get("accn", "")),
               default=None)


def _sec_instant(facts, metric, year, quarter):
    form, fp = ("10-K", "FY") if quarter == 4 else ("10-Q", f"Q{quarter}")
    items = [item for item in _sec_units(facts, SEC_TAGS[metric])
             if item.get("fy") == year and item.get("form") == form and item.get("fp") == fp
             and "start" not in item]
    selected = _latest(items)
    return selected.get("val") if selected else None


def _sec_duration(facts, metric, year, quarter, kind):
    form, fp = ("10-K", "FY") if quarter == 4 else ("10-Q", f"Q{quarter}")
    items = [item for item in _sec_units(facts, SEC_TAGS[metric])
             if item.get("fy") == year and item.get("form") == form and item.get("fp") == fp
             and _duration_days(item) is not None]
    if kind == "annual":
        items = [item for item in items if 300 <= _duration_days(item) <= 400]
    elif kind == "quarter":
        items = [item for item in items if 60 <= _duration_days(item) <= 120]
    elif kind == "cumulative":
        expected = quarter * 91
        items = [item for item in items if abs(_duration_days(item) - expected) <= 45]
    selected = _latest(items)
    return selected.get("val") if selected else None


def load_sec_financials(company: Company, start_year: int, end_year: int,
                        quarterly: bool, user_agent: str):
    payload = fetch_sec_company_facts(company.source_id, user_agent)
    facts = payload.get("facts", {}).get("us-gaap", {})
    output = []
    periods = [(year, quarter) for year in range(start_year, end_year + 1)
               for quarter in (range(1, 5) if quarterly else [4])]
    for year, quarter in periods:
        row = {"기간": f"{year} Q{quarter}" if quarterly else str(year)}
        for metric in BALANCE:
            row[metric] = _sec_instant(facts, metric, year, quarter)
        for metric in INCOME:
            row[metric] = _sec_duration(
                facts, metric, year, quarter, "quarter" if quarterly else "annual"
            )
        for metric in CASH[:4]:
            current = _sec_duration(
                facts, metric, year, quarter, "cumulative" if quarterly else "annual"
            )
            if quarterly and quarter > 1:
                previous = _sec_duration(facts, metric, year, quarter - 1, "cumulative")
                row[metric] = _subtract(current, previous)
            else:
                row[metric] = current
        ending = _sec_instant(facts, CASH[5], year, quarter)
        row[CASH[5]] = ending
        change = row[CASH[3]]
        row[CASH[4]] = None if ending is None or change is None else ending - change
        if any(value is not None for key, value in row.items() if key != "기간"):
            output.append(row)
    if not output:
        raise DataSourceError("선택한 기간에 SEC 재무정보가 없습니다.")
    frame = pd.DataFrame(output).set_index("기간")
    return frame[ALL_METRICS].div(1_000_000), "백만 USD"


def load_financials(company: Company, start_year: int, end_year: int,
                    quarterly: bool, dart_api_key: str, sec_user_agent: str):
    if company.source == "DART":
        return load_dart_financials(company, start_year, end_year, quarterly, dart_api_key)
    return load_sec_financials(company, start_year, end_year, quarterly, sec_user_agent)
