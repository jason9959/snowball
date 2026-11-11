from __future__ import annotations

import numpy as np

from .calculator import mortgage_schedule


def simulate_monthly_comparison(
    price_paths: np.ndarray,
    *,
    property_price: float,
    property_growth_rate: float,
    loan: float,
    annual_rate: float,
    loan_years: int,
    repayment_method: str,
    initial_stock_investment: float,
    monthly_rent: float,
    rent_growth_cycle_months: int,
    rent_growth_rate: float,
    horizon_months: int,
) -> dict:
    """Compare property equity with monthly stock contributions or withdrawals."""
    paths = np.asarray(price_paths, dtype=float)
    if paths.ndim == 1:
        paths = paths[:, None]
    if paths.shape[0] < 2 or paths.shape[1] < 1 or np.any(paths <= 0):
        raise ValueError("주가 경로가 올바르지 않습니다.")

    payments, balances = mortgage_schedule(
        loan, annual_rate, loan_years, repayment_method, horizon_months
    )
    monthly_indices = np.rint(
        np.linspace(0, paths.shape[0] - 1, horizon_months + 1)
    ).astype(int)
    monthly_prices = paths[monthly_indices]
    path_count = monthly_prices.shape[1]

    shares = np.full(path_count, max(float(initial_stock_investment), 0.0)) / monthly_prices[0]
    stock_values = np.zeros((horizon_months + 1, path_count), dtype=float)
    investment_flows = np.zeros_like(stock_values)
    unfunded_rent = np.zeros_like(stock_values)
    rents = np.zeros(horizon_months + 1, dtype=float)
    stock_values[0] = shares * monthly_prices[0]

    for month in range(1, horizon_months + 1):
        increases = 0
        if rent_growth_cycle_months:
            increases = (month - 1) // rent_growth_cycle_months
        current_rent = float(monthly_rent) * (
            1 + float(rent_growth_rate) / 100
        ) ** increases
        rents[month] = current_rent

        current_prices = monthly_prices[month]
        values_before_flow = shares * current_prices
        planned_flow = payments[month] - current_rent
        if planned_flow >= 0:
            actual_flows = np.full(path_count, planned_flow, dtype=float)
            shares += actual_flows / current_prices
        else:
            requested = -planned_flow
            withdrawals = np.minimum(values_before_flow, requested)
            actual_flows = -withdrawals
            shares -= withdrawals / current_prices
            unfunded_rent[month] = requested - withdrawals
        investment_flows[month] = actual_flows
        stock_values[month] = shares * current_prices

    elapsed_years = np.arange(horizon_months + 1, dtype=float) / 12
    property_values = float(property_price) * (
        1 + float(property_growth_rate) / 100
    ) ** elapsed_years
    house_equity = property_values - balances

    return {
        "stock_values": stock_values,
        "investment_flows": investment_flows,
        "unfunded_rent": unfunded_rent,
        "rents": rents,
        "payments": payments,
        "loan_balances": balances,
        "property_values": property_values,
        "house_equity": house_equity,
    }
