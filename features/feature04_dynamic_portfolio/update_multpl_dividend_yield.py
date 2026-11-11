"""Refresh the bundled Multpl monthly S&P 500 dividend-yield CSV."""

from __future__ import annotations

from features.feature04_dynamic_portfolio.market_data import (
    PACKAGED_DIVIDEND_CSV,
    _download_multpl_dividend_yield,
)


def main() -> None:
    frame = _download_multpl_dividend_yield()
    PACKAGED_DIVIDEND_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.rename_axis("Date").reset_index().to_csv(
        PACKAGED_DIVIDEND_CSV,
        index=False,
        date_format="%Y-%m-%d",
    )
    print(
        f"saved {len(frame)} rows to {PACKAGED_DIVIDEND_CSV} "
        f"({frame.index.min().date()} ~ {frame.index.max().date()})"
    )


if __name__ == "__main__":
    main()
