from __future__ import annotations

import json
import unittest

from snowball.account_store import (
    DEFAULT_ACCOUNT_TAB,
    LEGACY_TRADE_COLUMNS,
    SYSTEM_SHEET_SCHEMAS,
    GoogleSheetAccountStore,
    SheetSchemaError,
    TRADE_COLUMNS,
    account_display_name,
    parse_service_account_json,
)


def _cell_position(address: str) -> tuple[int, int]:
    letters = "".join(character for character in address if character.isalpha())
    digits = "".join(character for character in address if character.isdigit())
    column = 0
    for character in letters:
        column = column * 26 + ord(character.upper()) - 64
    return int(digits or 1) - 1, column - 1


class FakeWorksheet:
    def __init__(self, title: str, rows: list[list[object]] | None = None):
        self.title = title
        self.rows = rows or []
        self.frozen_rows = 0

    def row_values(self, number: int):
        return list(self.rows[number - 1]) if len(self.rows) >= number else []

    def get_all_values(self):
        return [list(row) for row in self.rows]

    def update(self, *, range_name: str, values: list[list[object]], raw: bool = True):
        start = range_name.split(":")[0]
        row_index, column_index = _cell_position(start)
        while len(self.rows) < row_index + len(values):
            self.rows.append([])
        for offset, values_row in enumerate(values):
            target = self.rows[row_index + offset]
            target += [""] * max(0, column_index + len(values_row) - len(target))
            target[column_index:column_index + len(values_row)] = list(values_row)

    def freeze(self, *, rows: int):
        self.frozen_rows = rows

    def format(self, range_name: str, style: dict[str, object]):
        return None

    def get_all_records(self, *, expected_headers: list[str], default_blank: str):
        return [dict(zip(expected_headers, row)) for row in self.rows[1:]]

    def col_values(self, number: int):
        index = number - 1
        return [str(row[index]) for row in self.rows if len(row) > index]

    def append_row(self, row: list[object], *, value_input_option: str):
        self.rows.append(list(row))

    def update_acell(self, address: str, value: object):
        row_index, column_index = _cell_position(address)
        while len(self.rows) <= row_index:
            self.rows.append([])
        self.rows[row_index] += [""] * max(0, column_index + 1 - len(self.rows[row_index]))
        self.rows[row_index][column_index] = value


class FakeSpreadsheet:
    title = "동적자산배분 거래 내역"

    def __init__(self, worksheets: list[FakeWorksheet] | None = None):
        self._worksheets = {worksheet.title: worksheet for worksheet in worksheets or []}

    def worksheets(self):
        return list(self._worksheets.values())

    def add_worksheet(self, *, title: str, rows: int, cols: int):
        worksheet = FakeWorksheet(title)
        self._worksheets[title] = worksheet
        return worksheet

    def worksheet(self, title: str):
        return self._worksheets[title]


class AccountStoreTest(unittest.TestCase):
    def test_parse_service_account_json(self):
        parsed = parse_service_account_json(
            '{"client_email":"bot@example.com","private_key":"secret","token_uri":"https://token"}'
        )
        self.assertEqual(parsed["client_email"], "bot@example.com")

    def test_parse_service_account_json_accepts_streamlit_wrapper(self):
        raw = (
            "GOOGLE_SERVICE_ACCOUNT_JSON = '''"
            '{"client_email":"bot@example.com","private_key":"secret",'
            '"token_uri":"https://token"}'
            "'''"
        )
        self.assertEqual(parse_service_account_json(raw)["private_key"], "secret")

    def test_parse_service_account_json_accepts_encoded_json_string(self):
        raw = json.dumps(
            '{"client_email":"bot@example.com","private_key":"secret",'
            '"token_uri":"https://token"}'
        )
        self.assertEqual(parse_service_account_json(raw)["token_uri"], "https://token")

    def test_ensure_schema_creates_system_and_default_account_tabs(self):
        spreadsheet = FakeSpreadsheet()
        store = GoogleSheetAccountStore(spreadsheet)
        store.ensure_schema()
        expected = set(SYSTEM_SHEET_SCHEMAS) | {DEFAULT_ACCOUNT_TAB}
        self.assertEqual(set(spreadsheet._worksheets), expected)
        self.assertEqual(spreadsheet.worksheet(DEFAULT_ACCOUNT_TAB).rows[0], list(TRADE_COLUMNS))
        self.assertEqual(store.list_account_tabs(), (DEFAULT_ACCOUNT_TAB,))
        self.assertEqual(account_display_name(DEFAULT_ACCOUNT_TAB), "기본계좌")

    def test_legacy_ledger_is_copied_once_to_default_account(self):
        legacy_row = ["id1", "2026-11-01T12:30:01+09:00"] + [""] * 12 + ["정상"]
        spreadsheet = FakeSpreadsheet([
            FakeWorksheet("거래내역", [list(LEGACY_TRADE_COLUMNS), legacy_row]),
        ])
        store = GoogleSheetAccountStore(spreadsheet)
        store.ensure_schema()
        loaded = store.load_transactions(DEFAULT_ACCOUNT_TAB)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["거래ID"], "id1")
        self.assertEqual(loaded[0]["입력시각"], "2026-11-01 12:30:01")
        store.ensure_schema()
        self.assertEqual(len(store.load_transactions(DEFAULT_ACCOUNT_TAB)), 1)

    def test_existing_wrong_account_header_is_not_overwritten(self):
        spreadsheet = FakeSpreadsheet([FakeWorksheet("계좌_연금", [["다른 헤더"]])])
        with self.assertRaises(SheetSchemaError):
            GoogleSheetAccountStore(spreadsheet).ensure_schema()

    def test_existing_account_timestamp_is_normalized(self):
        row = ["id1", "2026-11-01T23:00:01+09:00"] + [""] * 12 + ["정상"]
        spreadsheet = FakeSpreadsheet([
            FakeWorksheet("계좌_일반계좌1", [list(LEGACY_TRADE_COLUMNS), row]),
        ])
        store = GoogleSheetAccountStore(spreadsheet)
        store.ensure_schema()
        loaded = store.load_transactions("계좌_일반계좌1")
        self.assertEqual(loaded[0]["입력시각"], "2026-11-01 23:00:01")
        self.assertEqual(spreadsheet.worksheet("계좌_일반계좌1").row_values(1), list(TRADE_COLUMNS))

    def test_readonly_load_does_not_upgrade_legacy_header_or_timestamp(self):
        original_time = "2026-11-01T23:00:01+09:00"
        row = ["id1", original_time] + [""] * 12 + ["정상"]
        worksheet = FakeWorksheet(
            "계좌_일반계좌1", [list(LEGACY_TRADE_COLUMNS), row],
        )
        store = GoogleSheetAccountStore(FakeSpreadsheet([worksheet]))
        loaded = store.load_transactions_readonly("계좌_일반계좌1")
        self.assertEqual(loaded[0]["입력시각"], original_time)
        self.assertEqual(loaded[0]["수정시각"], "")
        self.assertEqual(worksheet.row_values(1), list(LEGACY_TRADE_COLUMNS))

    def test_readonly_load_accepts_redesigned_unique_headers(self):
        headers = [
            "입력날짜", "입력시각", "거래일", "유형", "시장", "종목코드",
            "종목명", "수량", "체결단가", "거래금액", "수수료·세금",
            "결제금액", "통화", "환율", "메모",
        ]
        row = [
            "2026-11-02", "05:09:15", "2026-11-02", "매수", "미국", "BIL",
            "T-Bill", 2, 91.41, 182.82, 0.12, 182.94, "USD", 1347.3, "",
        ]
        worksheet = FakeWorksheet("계좌_일반계좌1", [headers, row])
        store = GoogleSheetAccountStore(FakeSpreadsheet([worksheet]))
        loaded = store.load_transactions_readonly("계좌_일반계좌1")
        self.assertEqual(loaded[0]["거래금액"], 182.82)
        self.assertEqual(loaded[0]["금액"], 182.82)
        self.assertEqual(loaded[0]["상태"], "")

    def test_readonly_load_rejects_duplicate_headers(self):
        worksheet = FakeWorksheet(
            "계좌_일반계좌1", [["거래일", "유형", "금액", "금액"]],
        )
        store = GoogleSheetAccountStore(FakeSpreadsheet([worksheet]))
        with self.assertRaisesRegex(SheetSchemaError, "중복"):
            store.load_transactions_readonly("계좌_일반계좌1")

    def test_create_append_update_and_cancel_transaction(self):
        spreadsheet = FakeSpreadsheet()
        store = GoogleSheetAccountStore(spreadsheet)
        account_tab = store.create_account_tab("연금저축")
        self.assertEqual(account_tab, "계좌_연금저축")
        record = {column: "" for column in TRADE_COLUMNS}
        record.update({"거래ID": "abc123", "유형": "입금", "금액": 1_000_000, "상태": "정상"})
        store.append_transaction(account_tab, record)
        self.assertEqual(store.load_transactions(account_tab)[0]["금액"], 1_000_000)
        self.assertTrue(store.update_transaction(account_tab, "abc123", {"금액": 2_000_000}))
        updated = store.load_transactions(account_tab)[0]
        self.assertEqual(updated["금액"], 2_000_000)
        self.assertTrue(updated["수정시각"])
        self.assertTrue(store.cancel_transaction(account_tab, "abc123"))
        self.assertEqual(store.load_transactions(account_tab)[0]["상태"], "취소")


if __name__ == "__main__":
    unittest.main()
