import datetime as dt
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

if "yfinance" not in sys.modules:
    try:
        __import__("yfinance")
    except ModuleNotFoundError:
        sys.modules["yfinance"] = types.SimpleNamespace(download=None)

from common.functions.analytics import allocation_sweep, comparison, monte_carlo, periodic, static_portfolio
from common.metrics import mdd_recovery_days, sharpe_ratio, sortino_ratio


def streamlit_periodic(prices: pd.Series, frequency: str) -> pd.DataFrame:
    rows = []
    grouped = {}
    for date, value in prices.dropna().sort_index().items():
        timestamp = pd.Timestamp(date)
        if frequency == "월":
            label = f"{timestamp.month}월"
        elif frequency == "분기":
            label = f"{((timestamp.month - 1) // 3) + 1}분기"
        elif frequency == "반기":
            label = "상반기" if timestamp.month <= 6 else "하반기"
        else:
            label = "연간"
        grouped.setdefault((timestamp.year, label), []).append((timestamp, float(value)))
    for (year, label), observations in grouped.items():
        first_date, first_price = observations[0]
        last_date, last_price = observations[-1]
        rows.append({
            "연도": year, "주기": label, "시작일": first_date,
            "종료일": last_date, "수익률": (last_price / first_price - 1) * 100,
        })
    return pd.DataFrame(rows).sort_values(["연도", "시작일"]).reset_index(drop=True)


def streamlit_portfolio(prices, weights, initial, rebalance, contribution=0.0, contribution_frequency="매월"):
    dates = pd.DatetimeIndex(prices.index)
    values = prices.to_numpy(dtype=float)

    def schedule(frequency):
        months = {"매월": 1, "매분기": 3, "매년": 12}.get(frequency)
        if months is None:
            return set()
        output, step = set(), 1
        while True:
            target = dates[0] + pd.DateOffset(months=months * step)
            if target > dates[-1]:
                break
            position = dates.searchsorted(target, side="left")
            if position < len(dates):
                output.add(dates[position])
            step += 1
        return output

    shares = initial * weights / values[0]
    portfolio = np.empty(len(dates))
    invested = np.empty(len(dates))
    portfolio[0] = invested[0] = initial
    contribution_dates = schedule(contribution_frequency) if contribution else set()
    rebalance_dates = set(dates[1:]) if rebalance == "매일" else schedule(rebalance)
    total = initial
    for index in range(1, len(dates)):
        date, current_prices = dates[index], values[index]
        current = float(np.sum(shares * current_prices))
        add = contribution if date in contribution_dates else 0.0
        total += add
        if date in rebalance_dates:
            shares = (current + add) * weights / current_prices
        elif add:
            current_weights = shares * current_prices / current
            shares += add * current_weights / current_prices
        portfolio[index] = float(np.sum(shares * current_prices))
        invested[index] = total
    value_series = pd.Series(portfolio, index=dates)
    invested_series = pd.Series(invested, index=dates)
    flows = invested_series.diff().fillna(0)
    nav = (1 + ((value_series - flows) / value_series.shift(1) - 1).fillna(0)).cumprod()
    return value_series, invested_series, nav / nav.cummax() - 1


def streamlit_monte(prices, years, simulations, initial, contribution, seed):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    days = int(round(years * 252))
    rng = np.random.default_rng(seed)
    values = log_returns.to_numpy(dtype=float)
    samples = values[rng.integers(0, len(values), size=(days, simulations))]
    initial_price = float(prices.iloc[-1])
    price_paths = initial_price * np.exp(np.cumsum(samples, axis=0))
    price_paths = np.vstack([np.full((1, simulations), initial_price), price_paths])
    shares = np.full(simulations, initial / initial_price)
    value_paths = np.empty_like(price_paths)
    value_paths[0] = initial
    steps = set(range(21, days + 1, 21)) if contribution else set()
    for step in range(1, days + 1):
        if step in steps:
            shares += contribution / price_paths[step]
        value_paths[step] = shares * price_paths[step]
    return np.percentile(value_paths, [5, 50, 95], axis=1)


class StreamlitNiceGuiParityTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.bdate_range("2018-01-02", "2024-12-31")
        step = np.arange(len(self.index))
        self.frame = pd.DataFrame({
            "SPY": 100 * np.exp(step * 0.00025 + np.sin(step / 17) * 0.01),
            "IEF": 100 * np.exp(step * 0.00008 + np.cos(step / 23) * 0.006),
        }, index=self.index)
        self.start = dt.date(2018, 1, 2)
        self.end = dt.date(2024, 12, 31)

    def _prices(self, tickers, start, end, field="Close"):
        return self.frame[list(tickers)].copy()

    def test_relative_comparison_uses_same_common_period_and_base(self):
        source = self.frame.copy()
        source.loc[source.index[:10], "IEF"] = np.nan
        expected_source = source.dropna()
        expected = expected_source.divide(expected_source.iloc[0]).mul(100)
        with patch("common.functions.analytics.prices", return_value=source):
            result = comparison(["SPY", "IEF"], self.start, self.end)
        for ticker in ("SPY", "IEF"):
            pd.testing.assert_series_equal(result.series[ticker], expected[ticker])

    def test_periodic_returns_equal_streamlit_source(self):
        with patch("common.functions.analytics.prices", side_effect=self._prices):
            actual = periodic("SPY", self.start, self.end, "반기").table
        expected = streamlit_periodic(self.frame["SPY"], "반기")
        pd.testing.assert_frame_equal(actual, expected)

    def test_static_portfolio_equal_streamlit_source(self):
        expected = streamlit_portfolio(self.frame, np.array([0.6, 0.4]), 10_000.0, "매분기", 100.0, "매월")
        with patch("common.functions.analytics.prices", side_effect=self._prices):
            result = static_portfolio(["SPY", "IEF"], [60, 40], self.start, self.end, 10_000, "매분기", 100, "매월")
        pd.testing.assert_series_equal(result.series["포트폴리오"], expected[0], check_names=False)
        pd.testing.assert_series_equal(result.series["누적 투입금"], expected[1], check_names=False)
        pd.testing.assert_series_equal(result.series["낙폭"], expected[2], check_names=False)

    def test_allocation_sweep_equal_streamlit_source(self):
        expected_rows = []
        for first in range(0, 101, 20):
            weights = [first, 100 - first]
            values = streamlit_portfolio(
                self.frame, np.asarray(weights, dtype=float) / 100, 25_000.0, "매분기"
            )[0]
            daily = values.pct_change().dropna()
            total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
            years = max((values.index[-1] - values.index[0]).days / 365.2425, 1 / 365.2425)
            cagr = ((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1) * 100
            volatility = daily.std(ddof=1) * np.sqrt(252) * 100
            mdd = float((values / values.cummax() - 1).min() * 100)
            expected_rows.append({
                "SPY 비율": first,
                "IEF 비율": 100 - first,
                "최종 금액": float(values.iloc[-1]),
                "총수익률": float(total_return),
                "CAGR": float(cagr),
                "변동성": float(volatility),
                "최대낙폭": mdd,
                "샤프지수": float(sharpe_ratio(daily)),
                "Sortino": float(sortino_ratio(daily)),
                "Calmar": float((cagr / 100) / abs(mdd / 100)) if mdd < 0 else np.nan,
                "회복기간(일)": mdd_recovery_days(values),
            })
        with patch("common.functions.analytics.prices", side_effect=self._prices):
            actual = allocation_sweep(
                ["SPY", "IEF"], self.start, self.end, 20,
                rebalance="매분기", initial=25_000,
            ).table
        pd.testing.assert_frame_equal(actual, pd.DataFrame(expected_rows))

    def test_bootstrap_paths_equal_streamlit_source(self):
        expected = streamlit_monte(self.frame["SPY"], 2, 100, 10_000.0, 100.0, 77)
        with patch("common.functions.analytics.prices", side_effect=self._prices), patch(
            "common.functions.analytics.np.random.default_rng",
            return_value=np.random.default_rng(77),
        ):
            result = monte_carlo("SPY", self.start, self.end, 2, 100, 10_000, "bootstrap", 100)
        actual = np.vstack([result.series["P5"], result.series["P50"], result.series["P95"]])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_quarterly_contribution_paths_equal_streamlit_source(self):
        def expected_quarterly():
            log_returns = np.log(self.frame["SPY"] / self.frame["SPY"].shift(1)).dropna()
            days = 252
            rng = np.random.default_rng(91)
            samples = rng.normal(log_returns.mean(), log_returns.std(ddof=1), size=(days, 80))
            price = float(self.frame["SPY"].iloc[-1]) * np.exp(np.cumsum(samples, axis=0))
            price = np.vstack([np.full((1, 80), self.frame["SPY"].iloc[-1]), price])
            shares = np.full(80, 10_000 / self.frame["SPY"].iloc[-1])
            paths = np.empty_like(price)
            paths[0] = 10_000
            for index in range(1, days + 1):
                if index % 63 == 0:
                    shares += 300 / price[index]
                paths[index] = shares * price[index]
            return np.percentile(paths, [5, 50, 95], axis=1)

        expected = expected_quarterly()
        with patch("common.functions.analytics.prices", side_effect=self._prices), patch(
            "common.functions.analytics.np.random.default_rng",
            return_value=np.random.default_rng(91),
        ):
            result = monte_carlo(
                "SPY", self.start, self.end, 1, 80, 10_000,
                "normal", 300, "매분기",
            )
        actual = np.vstack([result.series["P5"], result.series["P50"], result.series["P95"]])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
