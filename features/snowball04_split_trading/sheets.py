"""Google Sheets export for a completed split-trading run."""

from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd

from snowball.account_store import GoogleSheetAccountStore
from snowball.settings import google_sheet_settings


def connect_spreadsheet():
    spreadsheet_id, service_json = google_sheet_settings()
    if not spreadsheet_id or not service_json:
        raise ValueError("Google Sheets 환경변수가 설정되지 않았습니다.")
    return GoogleSheetAccountStore.connect(spreadsheet_id, service_json).spreadsheet


def _safe_title(value: str) -> str:
    return re.sub(r"[\[\]:*?/\\]", " ", value).strip()[:80] or "분할매매"


def export_result(
    spreadsheet: Any,
    *,
    symbol_name: str,
    conditions: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[str, str]:
    """Create a new numbered worksheet and return its title and URL."""
    clean_name = _safe_title(symbol_name)
    for _ in range(5):
        worksheets = spreadsheet.worksheets()
        names = {sheet.title for sheet in worksheets}
        number = len(worksheets) + 1
        while f"{number}. {clean_name}" in names:
            number += 1
        title = f"{number}. {clean_name}"[:100]
        try:
            sheet = spreadsheet.add_worksheet(title=title, rows=2_000, cols=20)
            break
        except Exception as error:
            message = str(error).lower()
            if "already exist" not in message and "duplicate" not in message:
                raise
            continue
    else:
        raise RuntimeError("새 결과 탭 이름을 확보하지 못했습니다. 다시 시도해 주세요.")

    rows: list[list[Any]] = [["분할매매 백테스트 조건", "값"]]
    rows.extend([[str(key), value] for key, value in conditions.items()])
    rows.extend([
        [], ["결과 요약", "값"],
        ["자금 운용 방식", (
            "필요자금 역산" if result.get("funding_mode") == "required"
            else "투자총액 제한"
        )],
        ["최소 필요자금", result.get("minimum_required_capital") or ""],
        ["최대 자금 필요일", result.get("minimum_required_capital_date") or ""],
        ["안전 여유 10%", result.get("recommended_capital_10") or ""],
        ["안전 여유 20%", result.get("recommended_capital_20") or ""],
        ["안전 여유 30%", result.get("recommended_capital_30") or ""],
        ["전액 투자 최종 평가액", result["full_final"]],
        ["전액 투자 총수익률", result["full_return"]],
        ["분할매매 최종 평가액", result["split_final"]],
        ["분할매매 총수익률", result["split_return"]],
        ["전액 투자 MDD", result["full_max_drawdown"]],
        ["분할매매 MDD", result["split_max_drawdown"]],
        ["분할매매 현재 낙폭", result["split_current_drawdown"]],
        ["분할매매 수익률/MDD", result["split_return_mdd"] or 0],
        ["수익률 차이(%p)", result["return_difference"]],
        ["분할매매 현금잔액", result["final_cash"]],
        ["분할매매 보유분 평가액", result["final_holdings_value"]],
        [],
    ])
    trades: pd.DataFrame = result["trades"]
    trade_header_row = len(rows) + 1
    rows.append(list(trades.columns))
    for row in trades.itertuples(index=False, name=None):
        rows.append(["" if pd.isna(value) else value for value in row])
    sheet.update(range_name="A1", values=rows, raw=True)
    sheet.freeze(rows=1)
    sheet.format("A1:B1", {"textFormat": {"bold": True}, "backgroundColor": {
        "red": .91, "green": .93, "blue": 1.0,
    }})
    money_pattern = "#,##0" if conditions.get("국가") == "한국" else "#,##0.00"
    first_trade_row = trade_header_row + 1
    last_trade_row = max(first_trade_row, first_trade_row + len(trades) - 1)
    sheet.format(f"F{first_trade_row}:F{last_trade_row}", {
        "numberFormat": {"type": "NUMBER", "pattern": money_pattern},
    })
    sheet.format(f"G{first_trade_row}:G{last_trade_row}", {
        "numberFormat": {
            "type": "NUMBER",
            "pattern": "0" if result.get("quantity_mode") == "whole" else "0.000000",
        },
    })
    for column in ("H", "I", "K"):
        sheet.format(f"{column}{first_trade_row}:{column}{last_trade_row}", {
            "numberFormat": {"type": "NUMBER", "pattern": money_pattern},
        })
    sheet.format(f"L{first_trade_row}:L{last_trade_row}", {
        "numberFormat": {"type": "NUMBER", "pattern": money_pattern},
    })
    sheet.format(f"J{first_trade_row}:J{last_trade_row}", {
        "numberFormat": {"type": "PERCENT", "pattern": "0.00%"},
    })
    spreadsheet_id = getattr(spreadsheet, "id", "")
    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit#gid={sheet.id}"
    return title, url

