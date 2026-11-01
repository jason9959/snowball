from __future__ import annotations

import unittest

from snowball.account_store import (
    GoogleSheetAccountStore,
    SHEET_SCHEMAS,
    SheetSchemaError,
    TRADE_COLUMNS,
    parse_service_account_json,
)


class FakeWorksheet:
    def __init__(self, title: str, rows: list[list[object]] | None = None):
        self.title = title
        self.rows = rows or []
        self.frozen_rows = 0

    def row_values(self, number: int):
        return list(self.rows[number - 1]) if len(self.rows) >= number else []

    def update(self, *, range_name: str, values: list[list[object]]):
        self.rows = [list(values[0])]

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
        letters = "".join(character for character in address if character.isalpha())
        digits = "".join(character for character in address if character.isdigit())
        column = 0
        for character in letters:
            column = column * 26 + ord(character.upper()) - 64
        self.rows[int(digits) - 1][column - 1] = value


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

    def test_ensure_schema_creates_tabs_and_headers(self):
        spreadsheet = FakeSpreadsheet()
        store = GoogleSheetAccountStore(spreadsheet)
        store.ensure_schema()
        self.assertEqual(set(spreadsheet._worksheets), set(SHEET_SCHEMAS))
        for title, columns in SHEET_SCHEMAS.items():
            self.assertEqual(spreadsheet.worksheet(title).rows[0], list(columns))
            self.assertEqual(spreadsheet.worksheet(title).frozen_rows, 1)

    def test_existing_wrong_header_is_not_overwritten(self):
        spreadsheet = FakeSpreadsheet([FakeWorksheet("거래내역", [["다른 헤더"]])])
        with self.assertRaises(SheetSchemaError):
            GoogleSheetAccountStore(spreadsheet).ensure_schema()

    def test_append_load_and_cancel_transaction(self):
        spreadsheet = FakeSpreadsheet()
        store = GoogleSheetAccountStore(spreadsheet)
        record = {column: "" for column in TRADE_COLUMNS}
        record.update({"거래ID": "abc123", "유형": "입금", "금액": 1_000_000, "상태": "정상"})
        store.append_transaction(record)
        loaded = store.load_transactions()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["거래ID"], "abc123")
        self.assertEqual(loaded[0]["금액"], 1_000_000)
        self.assertTrue(store.cancel_transaction("abc123"))
        self.assertEqual(store.load_transactions()[0]["상태"], "취소")


if __name__ == "__main__":
    unittest.main()
