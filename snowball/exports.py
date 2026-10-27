"""Portable CSV/ZIP exports for comparing backtest implementations."""

from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def monthly_return_export(nav: pd.Series) -> pd.DataFrame:
    monthly = nav.resample("ME").last().pct_change().dropna()
    return pd.DataFrame({
        "적용 월": monthly.index.to_period("M").astype(str),
        "수익률 원시값": monthly.to_numpy(),
        "수익률 (%)": monthly.to_numpy() * 100.0,
    })


def snapshot_export(snapshots: dict[str, pd.DataFrame], category: str) -> pd.DataFrame:
    frames = []
    for applied_date, frame in snapshots.items():
        if frame is None or frame.empty:
            continue
        copy = frame.copy()
        copy.insert(0, "구분", category)
        copy.insert(0, "리밸런싱 적용일", applied_date)
        frames.append(copy)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def diagnostic_frames(result: dict) -> dict[str, pd.DataFrame]:
    history = result.get("history", pd.DataFrame())
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)
    nav = result["nav"].rename("포트폴리오 NAV")
    daily_nav = nav.rename_axis("Date").reset_index()
    daily_nav["일간 수익률"] = nav.pct_change().to_numpy()
    daily_nav["낙폭"] = (nav / nav.cummax() - 1.0).to_numpy()
    return {
        "01_monthly_returns.csv": monthly_return_export(nav),
        "02_rebalancing.csv": history,
        "03_asset_signals.csv": snapshot_export(result.get("candidate_snapshots", {}), "자산 선택"),
        "04_canary_signals.csv": snapshot_export(result.get("canary_audits", {}), "카나리아"),
        "05_market_signals.csv": snapshot_export(result.get("market_audits", {}), "시장 데이터"),
        "06_daily_nav_drawdown.csv": daily_nav,
    }


def diagnostic_zip(result: dict) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, frame in diagnostic_frames(result).items():
            archive.writestr(name, frame.to_csv(index=False).encode("utf-8-sig"))
    return buffer.getvalue()
