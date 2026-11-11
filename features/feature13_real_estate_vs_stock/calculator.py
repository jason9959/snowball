from __future__ import annotations

import numpy as np


def mortgage_payment(principal: float, annual_rate: float, years: int, method: str) -> float:
    """첫 달의 대출 상환액을 반환한다."""
    months = max(int(years * 12), 1)
    rate = annual_rate / 100 / 12
    if method == "원금균등상환":
        return principal / months + principal * rate
    if method != "원리금균등상환":
        raise ValueError("지원하지 않는 대출상환방식입니다.")
    if rate == 0:
        return principal / months
    return principal * rate * (1 + rate) ** months / ((1 + rate) ** months - 1)


def mortgage_schedule(
    principal: float,
    annual_rate: float,
    years: int,
    method: str,
    horizon_months: int,
) -> tuple[np.ndarray, np.ndarray]:
    """월말 상환액과 상환 후 잔액을 0개월부터 반환한다."""
    loan_months = max(int(years * 12), 1)
    if horizon_months > loan_months:
        raise ValueError("백테스팅 기간은 대출기간을 넘을 수 없습니다.")

    payments = np.zeros(horizon_months + 1, dtype=float)
    balances = np.zeros(horizon_months + 1, dtype=float)
    balances[0] = max(float(principal), 0.0)
    monthly_rate = float(annual_rate) / 100 / 12
    fixed_payment = mortgage_payment(principal, annual_rate, years, method)
    fixed_principal = float(principal) / loan_months

    for month in range(1, horizon_months + 1):
        opening_balance = balances[month - 1]
        interest = opening_balance * monthly_rate
        if method == "원금균등상환":
            principal_paid = min(fixed_principal, opening_balance)
            payment = principal_paid + interest
        else:
            payment = min(fixed_payment, opening_balance + interest)
            principal_paid = max(payment - interest, 0.0)
        payments[month] = payment
        balances[month] = max(opening_balance - principal_paid, 0.0)

    return payments, balances
