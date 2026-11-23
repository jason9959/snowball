import datetime as dt

from common.snowball.period_controls import YEAR_OPTIONS, period_start


def test_year_options_include_one_to_ten_and_custom() -> None:
    assert list(YEAR_OPTIONS) == [*(str(year) for year in range(1, 11)), "custom"]
    assert YEAR_OPTIONS["10"] == "10년"
    assert YEAR_OPTIONS["custom"] == "직접 설정"


def test_period_start_preserves_month_and_day() -> None:
    assert period_start(dt.date(2026, 11, 23), 5) == dt.date(2021, 11, 23)


def test_period_start_handles_leap_day_and_minimum() -> None:
    assert period_start(dt.date(2024, 2, 29), 1) == dt.date(2023, 2, 28)
    assert period_start(
        dt.date(2026, 11, 23), 10, minimum=dt.date(2020, 1, 1)
    ) == dt.date(2020, 1, 1)
