"""Persistent account records backed by a private Google Sheet."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping


TRADE_COLUMNS = (
    "거래ID", "입력시각", "거래일", "유형", "시장", "종목코드", "종목명",
    "수량", "체결단가", "수수료·세금", "금액", "통화", "환율", "메모", "상태",
)

SHEET_SCHEMAS: dict[str, tuple[str, ...]] = {
    "거래내역": TRADE_COLUMNS,
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
        try:
            value = json.loads(raw)
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
    """Read and write account records while preserving an audit trail."""

    spreadsheet: Any
    _schema_ready: bool = field(default=False, init=False, repr=False)

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
        scopes = (
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
        )
        credentials = Credentials.from_service_account_info(info, scopes=scopes)
        spreadsheet = gspread.authorize(credentials).open_by_key(spreadsheet_id.strip())
        return cls(spreadsheet=spreadsheet)

    @property
    def title(self) -> str:
        return str(self.spreadsheet.title)

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        existing = {worksheet.title: worksheet for worksheet in self.spreadsheet.worksheets()}
        for title, columns in SHEET_SCHEMAS.items():
            worksheet = existing.get(title)
            if worksheet is None:
                worksheet = self.spreadsheet.add_worksheet(
                    title=title,
                    rows=1_000,
                    cols=max(20, len(columns)),
                )
            first_row = [str(value).strip() for value in worksheet.row_values(1)]
            if not first_row:
                worksheet.update(range_name="A1", values=[list(columns)])
                worksheet.freeze(rows=1)
                worksheet.format(
                    f"A1:{_column_letter(len(columns))}1",
                    {
                        "backgroundColor": {"red": 0.91, "green": 0.93, "blue": 1.0},
                        "textFormat": {"bold": True},
                        "horizontalAlignment": "CENTER",
                    },
                )
            elif tuple(first_row) != columns:
                raise SheetSchemaError(
                    f"'{title}' 탭의 첫 행이 예상 헤더와 다릅니다. 기존 데이터 보호를 위해 중단했습니다."
                )
        self._schema_ready = True

    def load_transactions(self) -> list[dict[str, Any]]:
        self.ensure_schema()
        rows = self.spreadsheet.worksheet("거래내역").get_all_records(
            expected_headers=list(TRADE_COLUMNS),
            default_blank="",
        )
        return [{column: row.get(column, "") for column in TRADE_COLUMNS} for row in rows]

    def append_transaction(self, record: Mapping[str, Any]) -> None:
        self.ensure_schema()
        worksheet = self.spreadsheet.worksheet("거래내역")
        existing_ids = set(worksheet.col_values(1)[1:])
        transaction_id = str(record.get("거래ID", ""))
        if not transaction_id:
            raise ValueError("거래ID가 필요합니다.")
        if transaction_id in existing_ids:
            raise ValueError("이미 저장된 거래ID입니다.")
        worksheet.append_row(
            [record.get(column, "") for column in TRADE_COLUMNS],
            value_input_option="USER_ENTERED",
        )

    def cancel_transaction(self, transaction_id: str) -> bool:
        self.ensure_schema()
        worksheet = self.spreadsheet.worksheet("거래내역")
        for row_number, value in enumerate(worksheet.col_values(1)[1:], start=2):
            if value == transaction_id:
                status_column = _column_letter(TRADE_COLUMNS.index("상태") + 1)
                worksheet.update_acell(f"{status_column}{row_number}", "취소")
                return True
        return False


def _column_letter(number: int) -> str:
    if number < 1:
        raise ValueError("column number must be positive")
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters
