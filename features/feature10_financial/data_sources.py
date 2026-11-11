"""Official DART and SEC company catalogs and API clients."""
from dataclasses import dataclass
from io import BytesIO
import re
from typing import Iterable
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import requests


DART_BASE_URL = "https://opendart.fss.or.kr/api"
SEC_COMPANY_LIST_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
REQUEST_TIMEOUT = 25


class DataSourceError(RuntimeError):
    """A public filing source could not return usable data."""


@dataclass(frozen=True)
class Company:
    name: str
    symbol: str
    country: str
    source: str
    source_id: str
    exchange: str = ""

    @property
    def label(self):
        parts = [self.name, self.symbol, self.country]
        if self.exchange:
            parts.append(self.exchange)
        return " · ".join(part for part in parts if part)


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def search_companies(companies: Iterable[Company], query: str, limit: int = 20):
    """Rank exact symbol/name matches before prefix and substring matches."""
    needle = _normalize(query)
    if not needle:
        return []
    ranked = []
    for company in companies:
        name = _normalize(company.name)
        symbol = _normalize(company.symbol)
        if needle == symbol:
            rank = 0
        elif needle == name:
            rank = 1
        elif symbol.startswith(needle):
            rank = 2
        elif name.startswith(needle):
            rank = 3
        elif needle in symbol or needle in name:
            rank = 4
        else:
            continue
        # Keep the Company object out of the sort comparison.  SEC can expose
        # duplicate rows with the same rank/name, and dataclass instances are
        # intentionally not orderable.
        ranked.append((rank, len(name), company.name, company.symbol,
                       company.source_id, company))
    return [
        item[-1]
        for item in sorted(
            ranked,
            key=lambda item: (
                int(item[0]), int(item[1]), str(item[2]).casefold(),
                str(item[3]).casefold(), str(item[4]).casefold(),
            ),
        )[:limit]
    ]


def _get_json(url, *, headers=None, params=None):
    try:
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataSourceError("공시 데이터 서버에서 정보를 가져오지 못했습니다.") from exc


def fetch_sec_companies(user_agent: str):
    if not user_agent.strip():
        raise DataSourceError("SEC_USER_AGENT 설정이 필요합니다.")
    payload = _get_json(SEC_COMPANY_LIST_URL, headers={"User-Agent": user_agent})
    fields = payload.get("fields", [])
    try:
        cik_index = fields.index("cik")
        name_index = fields.index("name")
        ticker_index = fields.index("ticker")
        exchange_index = fields.index("exchange")
    except ValueError as exc:
        raise DataSourceError("SEC 기업 목록 형식이 예상과 다릅니다.") from exc
    return [
        Company(str(row[name_index]), str(row[ticker_index]), "미국", "SEC EDGAR",
                str(row[cik_index]), str(row[exchange_index] or ""))
        for row in payload.get("data", []) if row[ticker_index]
    ]


def fetch_sec_company_facts(cik: str, user_agent: str):
    if not user_agent.strip():
        raise DataSourceError("SEC_USER_AGENT 설정이 필요합니다.")
    try:
        number = int(cik)
    except ValueError as exc:
        raise DataSourceError("올바르지 않은 SEC CIK입니다.") from exc
    return _get_json(SEC_COMPANY_FACTS_URL.format(cik=number),
                     headers={"User-Agent": user_agent})


def fetch_dart_companies(api_key: str):
    if not api_key.strip():
        raise DataSourceError("DART_API_KEY 설정이 필요합니다.")
    try:
        response = requests.get(f"{DART_BASE_URL}/corpCode.xml",
                                params={"crtfc_key": api_key}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            xml_bytes = archive.read("CORPCODE.xml")
    except (requests.RequestException, KeyError, OSError) as exc:
        raise DataSourceError("DART 기업 목록을 가져오지 못했습니다.") from exc
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise DataSourceError("DART 기업 목록을 읽지 못했습니다.") from exc
    companies = []
    for item in root.findall("list"):
        symbol = (item.findtext("stock_code") or "").strip()
        companies.append(Company(
            (item.findtext("corp_name") or "").strip(), symbol, "한국", "DART",
            (item.findtext("corp_code") or "").strip(),
        ))
    return companies


def fetch_dart_financial_statement(api_key: str, corp_code: str, year: int,
                                   report_code: str, fs_div: str = "CFS"):
    payload = _get_json(
        f"{DART_BASE_URL}/fnlttSinglAcntAll.json",
        params={"crtfc_key": api_key, "corp_code": corp_code, "bsns_year": year,
                "reprt_code": report_code, "fs_div": fs_div},
    )
    if payload.get("status") != "000":
        raise DataSourceError(payload.get("message") or "DART 재무제표 조회에 실패했습니다.")
    return payload.get("list", [])
