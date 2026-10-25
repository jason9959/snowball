"""Strategy metrics used by both reports and automated searches."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_nav(nav: pd.Series) -> dict[str, float]:
    nav = nav.dropna().astype(float)
    if len(nav) < 2 or nav.iloc[0] <= 0:
        raise ValueError("성과지표를 계산할 NAV가 부족합니다.")
    daily = nav.pct_change().dropna()
    years = max((nav.index[-1] - nav.index[0]).days / 365.2425, 1 / 365.2425)
    total = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    drawdown = nav / nav.cummax() - 1
    annual = nav.resample("YE").last().pct_change().dropna()
    annual_mean = float(annual.mean()) if len(annual) else float("nan")
    downside = np.minimum(annual.to_numpy() - annual_mean, 0.0) if len(annual) else np.array([])
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "annual_volatility": float(daily.std(ddof=1) * np.sqrt(252)) if len(daily) > 1 else float("nan"),
        "annual_return_std": float(annual.std(ddof=1)) if len(annual) > 1 else float("nan"),
        "annual_downside_deviation": float(np.sqrt(np.mean(downside ** 2))) if len(downside) else float("nan"),
        "worst_year": float(annual.min()) if len(annual) else float("nan"),
        "max_drawdown": float(drawdown.min()),
    }
