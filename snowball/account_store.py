"""Persistent account records backed by account-prefixed Google Sheet tabs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping
from zoneinfo import ZoneInfo


ACCOUNT_PREFIX = "계좌_"
DEFAULT_ACCOUNT_TAB = "계좌_기본계좌"
LEGACY_LEDGER_TAB = "거래내역"
LEGACY_TRADE_COLUMNS = (
    "거래ID", "입력시각", "거래일", "유형", "시장", "종목코드", "종목명",
    "수량", "체결단가", "수수료·세금", "금액", "통화", "환율", "메모", "상태",
)
TRADE_COLUMNS = LEGACY_TRADE_COLUMNS + ("수정시각",)

SYSTEM_SHEET_SCHEMAS: dict[str, tuple[str, ...]] = {
    "리밸런싱기록": (
        "판단기준일", "미국매수일", "국내매수제안일", "프리셋1", "프리셋2",
        "프리셋3목표", "국내대체목표", "실행상태", "기록시각",
    ),
    "월간성과": (
        "기준월", "현금", "보유자산평가액", "총평가액", "월간수익률", "누적수익률",
        "현재낙폭", "최대낙폭", "누적비용", "모델대비차이", "기록시각",
    ),
    "설정": ("항목", "값", "설명", "수정시각"),
}


class SheetSchemaError(ValueError):
    """Raised when an existing worksheet has unexpected columns."""


def parse_service_account_json(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated service-account mapping without logging secrets."""
    if isinstance(raw, str):
        text = raw.strip()
        assignment = re.match(r"^GOOGLE_SERVICE_ACCOUNT_JSON\s*=\s*(.+)$", text, re.DOTALL)
        if assignment:
            text = assignment.group(1).strip()
        for quote in ("'''", '\"\"\"'):
            if text.startswith(quote) and text.endswith(quote):
                text = text[len(quote):-len(quote)].strip()
                break
        try:
            value = json.loads(text)
            # Render users sometimes paste a JSON-encoded JSON string. Decode
            # the harmless outer layer while keeping the secret out of logs.
            if isinstance(value, str):
                value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("Google 서비스 계정 JSON 형식을 확인하세요.") from error
    elif isinstance(raw, Mapping):
        value = dict(raw)
    else:
        raise ValueError("Google 서비스 계정 인증정보가 필요합니다.")
    required = {"client_email", "private_key", "token_uri"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Google 서비스 계정 필드가 누락되었습니다: {', '.join(missing)}")
    return value


@dataclass
class GoogleSheetAccountStore:
    """Read and write one transaction ledger per ``계좌_`` worksheet."""

    spreadsheet: Any
    _system_schema_ready: bool = field(default=False, init=False, repr=False)
    _normalized_account_tabs: set[str] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def connect(
        cls,
        spreadsheet_id: str,
        service_account_json: str | Mapping[str, Any],
    ) -> "GoogleSheetAccountStore":
        if not spreadsheet_id.strip():
            raise ValueError("GOOGLE_SHEET_ID가 필요합니다.")
        info = parse_service_account_json(service_account_json)
        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except ImportError as error:  # pragma: no cover - deployment dependency guard
            raise RuntimeError("Google Sheet 연동 패키지가 설치되지 않았습니다.") from error
        credentials = Credentials.from_service_account_info(
            info,
            scopes=(
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.file",
            ),
        )
        return cls(gspread.authorize(credentials).open_by_key(spreadsheet_id.strip()))

    @property
    def title(self) -> str:
        return str(self.spreadsheet.title)

    def ensure_schema(self) -> None:
        if not self._system_schema_ready:
            existing = {worksheet.title: worksheet for worksheet in self.spreadsheet.worksheets()}
            for title, columns in SYSTEM_SHEET_SCHEMAS.items():
                worksheet = existing.get(title)
                if worksheet is None:
                    worksheet = self.spreadsheet.add_worksheet(
                        title=title, rows=1_000, cols=max(20, len(columns)),
                    )
                self._ensure_headers(worksheet, columns)
            self._system_schema_ready = True
        account_tabs = self.list_account_tabs(ensure=False)
        if not account_tabs:
            worksheet = self._create_account_tab(DEFAULT_ACCOUNT_TAB)
            self._copy_legacy_transactions(worksheet)
            account_tabs = (DEFAULT_ACCOUNT_TAB,)
        for tab_name in account_tabs:
            self._ensure_account_schema(self.spreadsheet.worksheet(tab_name))

    def list_account_tabs(self, *, ensure: bool = True) -> tuple[str, ...]:
        if ensure:
            self.ensure_schema()
        return tuple(sorted(
            worksheet.title
            for worksheet in self.spreadsheet.worksheets()
            if worksheet.title.startswith(ACCOUNT_PREFIX)
        ))

    def create_account_tab(self, account_name: str) -> str:
        self.ensure_schema()
        cleaned = re.sub(r"[\[\]:*?/\\]", " ", account_name).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            raise ValueError("계좌명을 입력하세요.")
        tab_name = f"{ACCOUNT_PREFIX}{cleaned}"[:100]
        if tab_name in self.list_account_tabs():
            raise ValueError("같은 이름의 계좌 탭이 이미 있습니다.")
        self._create_account_tab(tab_name)
        return tab_name

    def load_transactions(self, account_tab: str) -> list[dict[str, Any]]:
        worksheet = self._account_worksheet(account_tab)
        rows = worksheet.get_all_records(
            expected_headers=list(TRADE_COLUMNS), default_blank="",
        )
        return [{column: row.get(column, "") for column in TRADE_COLUMNS} for row in rows]

    def load_transactions_readonly(self, account_tab: str) -> list[dict[str, Any]]:
        """Read an account ledger without creating or normalizing any cells."""
        if not account_tab.startswith(ACCOUNT_PREFIX):
            raise ValueError("계좌 탭 이름은 '계좌_'로 시작해야 합니다.")
        if account_tab not in self.list_account_tabs(ensure=False):
            raise ValueError("선택한 계좌 탭을 찾을 수 없습니다.")
        worksheet = self.spreadsheet.worksheet(account_tab)
        first_row = tuple(str(value).strip() for value in worksheet.row_values(1))
        if first_row == TRADE_COLUMNS:
            rows = worksheet.get_all_records(
                expected_headers=list(TRADE_COLUMNS), default_blank="",
            )
            return [
                {column: row.get(column, "") for column in TRADE_COLUMNS}
                for row in rows
            ]
        if first_row == LEGACY_TRADE_COLUMNS:
            rows = worksheet.get_all_records(
                expected_headers=list(LEGACY_TRADE_COLUMNS), default_blank="",
            )
            return [
                {
                    **{column: row.get(column, "") for column in LEGACY_TRADE_COLUMNS},
                    "수정시각": "",
                }
                for row in rows
            ]
        raise SheetSchemaError(
            f"'{account_tab}' 탭의 첫 행이 예상 헤더와 다릅니다."
        )

    def append_transaction(self, account_tab: str, record: Mapping[str, Any]) -> None:
        worksheet = self._account_worksheet(account_tab)
        transaction_id = str(record.get("거래ID", ""))
        if not transaction_id:
            raise ValueError("거래ID가 필요합니다.")
        if transaction_id in set(worksheet.col_values(1)[1:]):
            raise ValueError("이미 저장된 거래ID입니다.")
        worksheet.append_row(
            [record.get(column, "") for column in TRADE_COLUMNS], value_input_option="RAW",
        )

    def update_transaction(
        self,
        account_tab: str,
        transaction_id: str,
        record: Mapping[str, Any],
    ) -> bool:
        worksheet = self._account_worksheet(account_tab)
        row_number = self._transaction_row(worksheet, transaction_id)
        if row_number is None:
            return False
        existing_values = worksheet.row_values(row_number)
        existing_values += [""] * (len(TRADE_COLUMNS) - len(existing_values))
        existing = dict(zip(TRADE_COLUMNS, existing_values))
        updated = {**existing, **dict(record)}
        updated["거래ID"] = transaction_id
        updated["입력시각"] = existing.get("입력시각", "") or _kst_now_text()
        updated["수정시각"] = _kst_now_text()
        worksheet.update(
            range_name=f"A{row_number}:{_column_letter(len(TRADE_COLUMNS))}{row_number}",
            values=[[updated.get(column, "") for column in TRADE_COLUMNS]], raw=True,
        )
        return True

    def cancel_transaction(self, account_tab: str, transaction_id: str) -> bool:
        worksheet = self._account_worksheet(account_tab)
        row_number = self._transaction_row(worksheet, transaction_id)
        if row_number is None:
            return False
        worksheet.update(
            range_name=f"O{row_number}:P{row_number}",
            values=[["취소", _kst_now_text()]], raw=True,
        )
        return True

    def _account_worksheet(self, account_tab: str) -> Any:
        if not account_tab.startswith(ACCOUNT_PREFIX):
            raise ValueError("계좌 탭 이름은 '계좌_'로 시작해야 합니다.")
        if account_tab not in self.list_account_tabs():
            raise ValueError("선택한 계좌 탭을 찾을 수 없습니다.")
        worksheet = self.spreadsheet.worksheet(account_tab)
        self._ensure_account_schema(worksheet)
        return worksheet

    def _create_account_tab(self, tab_name: str) -> Any:
        worksheet = self.spreadsheet.add_worksheet(
            title=tab_name, rows=2_000, cols=max(20, len(TRADE_COLUMNS)),
        )
        self._write_headers(worksheet, TRADE_COLUMNS)
        return worksheet

    def _copy_legacy_transactions(self, destination: Any) -> None:
        existing = {worksheet.title: worksheet for worksheet in self.spreadsheet.worksheets()}
        source = existing.get(LEGACY_LEDGER_TAB)
        if source is None or tuple(source.row_values(1)) != LEGACY_TRADE_COLUMNS:
            return
        for row in source.get_all_values()[1:]:
            padded = list(row) + [""] * (len(LEGACY_TRADE_COLUMNS) - len(row))
            padded[1] = _normalize_kst_text(padded[1])
            destination.append_row(padded + [""], value_input_option="RAW")

    def _ensure_account_schema(self, worksheet: Any) -> None:
        first_row = tuple(str(value).strip() for value in worksheet.row_values(1))
        if not first_row:
            self._write_headers(worksheet, TRADE_COLUMNS)
        elif first_row == LEGACY_TRADE_COLUMNS:
            worksheet.update_acell("P1", "수정시각")
        elif first_row != TRADE_COLUMNS:
            raise SheetSchemaError(
                f"'{worksheet.title}' 탭의 첫 행이 예상 헤더와 다릅니다. 기존 데이터 보호를 위해 중단했습니다."
            )
        if worksheet.title not in self._normalized_account_tabs:
            timestamps = worksheet.col_values(2)
            if len(timestamps) > 1:
                normalized = [_normalize_kst_text(value) for value in timestamps[1:]]
                if normalized != timestamps[1:]:
                    worksheet.update(
                        range_name=f"B2:B{len(timestamps)}",
                        values=[[value] for value in normalized],
                        raw=True,
                    )
            self._normalized_account_tabs.add(worksheet.title)

    def _ensure_headers(self, worksheet: Any, columns: tuple[str, ...]) -> None:
        first_row = tuple(str(value).strip() for value in worksheet.row_values(1))
        if not first_row:
            self._write_headers(worksheet, columns)
        elif first_row != columns:
            raise SheetSchemaError(
                f"'{worksheet.title}' 탭의 첫 행이 예상 헤더와 다릅니다. 기존 데이터 보호를 위해 중단했습니다."
            )

    @staticmethod
    def _write_headers(worksheet: Any, columns: tuple[str, ...]) -> None:
        worksheet.update(range_name="A1", values=[list(columns)], raw=True)
        worksheet.freeze(rows=1)
        worksheet.format(
            f"A1:{_column_letter(len(columns))}1",
            {
                "backgroundColor": {"red": 0.91, "green": 0.93, "blue": 1.0},
                "textFormat": {"bold": True},
                "horizontalAlignment": "CENTER",
            },
        )

    @staticmethod
    def _transaction_row(worksheet: Any, transaction_id: str) -> int | None:
        for row_number, value in enumerate(worksheet.col_values(1)[1:], start=2):
            if value == transaction_id:
                return row_number
        return None


def account_display_name(tab_name: str) -> str:
    return tab_name.removeprefix(ACCOUNT_PREFIX)


def _column_letter(number: int) -> str:
    if number < 1:
        raise ValueError("column number must be positive")
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _kst_now_text() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_kst_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Seoul"))
    return parsed.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
